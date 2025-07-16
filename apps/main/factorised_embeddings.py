from dataclasses import dataclass
from typing import Optional, Union
import torch
from torch import nn
from torch.nn import functional as F
from lingua.transformer import (
    BaseTransformerArgs,
    apply_rotary_emb,
    flex_attention_comp,
    repeat_kv,
    TiedLinear,
    TransposedTiedLinear,
    RMSNorm

)
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention
)
from .args import LMTransformerArgs, ProjectLayerArgs

import math

def broadcast_add(base: torch.Tensor, delta: torch.Tensor,
                  *, dim: int = -1, g: int = 1) -> torch.Tensor:
    """
    base  : (..., g*D, ...)          – shared projection (smaller)
    delta : (..., H*D, ...)          – per-head LoRA delta (bigger)
    dim   : dimension that holds g*D or H*D
    g     : #templates shared inside each group  (g == 1 ⇒ full sharing)
    returns a tensor shaped like `delta`, equal to broadcast(base) + delta
    """
    if dim < 0:
        dim += base.dim()            # canonicalise

    # ----- shapes & sanity -------------------------------------------------
    big  = delta.shape[dim]          # H*D
    small = base.shape[dim]          # g*D
    assert big % small == 0,  "delta and base feature dims incompatible"
    heads_per_group = big // small   # H / g
    D = small // g                   # single-head feature size

    # ----- reshape base to (..., g, D, ...) -------------------------------
    new_shape = list(base.shape)
    new_shape[dim:dim+1] = [g, D]    # split the feature dim
    base = base.reshape(*new_shape)

    # ----- expand over heads_per_group ------------------------------------
    # insert a broadcast axis right after 'g'
    base = base.unsqueeze(dim + 1)                              # (..., g, 1, D, ...)
    base = base.expand(*base.shape[:dim+1], heads_per_group, D,
                       *base.shape[dim+3:])                     # (..., g, H/g, D, ...)

    # ----- flatten back to (..., H*D, ...) -------------------------------
    flat_shape = list(delta.shape)
    base = base.reshape(*flat_shape)   # now exactly the same shape as delta

    return base + delta

class LoRAModule(nn.Module):
    def __init__(self,
        in_dim,
        out_dim,
        rank=8,
        bias=False,
        w_a_override=None,
        w_b_override=None):

        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = rank*2
        self.bias = bias

        # LoRA params
        if w_a_override is not None:
            self.w_a = w_a_override
        else:
            self.w_a = nn.Linear(
                in_dim,
            rank,
            bias=False,
        )  # A matrix

        if w_b_override is not None:
            self.w_b = w_b_override
        else:
            self.w_b = nn.Linear(
                rank,
                out_dim,
                bias=False,
        )  # B matrix

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_b.weight)


    @torch._dynamo.disable
    def forward(self, x):
        #  low-rank update
        lora = self.w_b(self.w_a(x)) * (self.alpha / self.rank)
        return lora

class FactorisedOut(nn.Module):
    def __init__(self, args: LMTransformerArgs):
        super().__init__()
        self.proj_down = nn.Linear(
            args.dim,
            args.factorised_vocab.d_factorised,
            bias=False)

        nn.init.trunc_normal_(
            self.proj_down.weight,
            mean=0.0,
            std=args.dim ** (-0.5),
            a=-3 * args.dim ** (-0.5),
            b=3 * args.dim ** (-0.5),
        )

        self.unembed = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False,
        )
        nn.init.trunc_normal_(
            self.unembed.weight,
            mean=0.0,
            std=args.dim ** (-0.5),
            a=-3 * args.dim ** (-0.5),
            b=3 * args.dim ** (-0.5),
        )

    def forward(self, x: torch.Tensor):
        intermediate = self.proj_down(x)
        logits = self.unembed(intermediate)
        return logits
class FactorisedTiedOut(nn.Module):
    def __init__(self, args: LMTransformerArgs, tok_embeddings: nn.Embedding, project: Optional[nn.Linear] = None):
        super().__init__()
        if project is not None:
            self.proj_down = TransposedTiedLinear(project)
        else:
            self.proj_down = nn.Linear(
                args.dim,
                args.factorised_vocab.d_emb,
                bias=False)

            nn.init.trunc_normal_(
                self.proj_down.weight,
                mean=0.0,
                std=args.dim ** (-0.5),
                a=-3 * args.dim ** (-0.5),
                b=3 * args.dim ** (-0.5),
            )
        self.unembed = TiedLinear(tok_embeddings)

    def forward(self, x: torch.Tensor):
        intermediate = self.proj_down(x)
        logits = self.unembed(intermediate)
        return logits

def calc_in_dim(args: LMTransformerArgs, i):
    if i == 0:
        if args.factorised_vocab.factorise:
            return args.factorised_vocab.d_factorised
        else:
            return args.factorised_vocab.d_emb
    elif args.project_up_layers is not None and i < len(args.project_up_layers):
        for j in range(i-1, -1, -1):
            for k, d in enumerate((args.project_up_layers[j].d_ffn,
                args.project_up_layers[j].d_attn_out,
                args.project_up_layers[j].d_attn_val,
                args.project_up_layers[j].d_attn_kq)):
                if d is not None:
                    return d if k < 2 else d*args.n_heads
        if args.factorised_vocab.factorise:
            return args.factorised_vocab.d_factorised
        else:
            return args.factorised_vocab.d_emb
    elif args.project_down_layers is not None and i >= (args.n_layers - len(args.project_down_layers)):
        for j in range(i-1, -1, -1):
            if j < (args.n_layers - len(args.project_down_layers)):
                return args.dim
            else:
                o = args.n_layers - len(args.project_down_layers)
                for k, d in enumerate((args.project_down_layers[j-o].d_ffn,
                    args.project_down_layers[j-o].d_attn_out,
                    args.project_down_layers[j-o].d_attn_val,
                    args.project_down_layers[j-o].d_attn_kq)):
                    if d is not None:
                        return d if k < 2 else d*args.n_heads

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        proj_args: ProjectLayerArgs
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.true_head_dim = dim//n_heads
        self.rope_theta = rope_theta

        self.n_heads = self.n_kq_heads = self.n_v_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.kq_head_dim = head_dim
        self.val_head_dim = head_dim
        self.out_dim = dim
        if proj_args.d_attn_kq:
            self.kq_head_dim = proj_args.d_attn_kq
            self.n_kq_heads = (proj_args.d_attn_kq * n_heads)//(self.true_head_dim)
        if proj_args.d_attn_val:
            self.val_head_dim = proj_args.d_attn_val
            self.n_v_heads = (proj_args.d_attn_val * n_heads)//(self.true_head_dim)
            if proj_args.d_attn_kq is None:
                self.n_kq_heads = -(proj_args.in_dim//-self.true_head_dim)
        elif proj_args.d_attn_kq:
            self.val_head_dim = self.kq_head_dim # we are not projecting up in the value head,
            # but we are projecting up in the key and query heads
            self.n_v_heads = self.n_kq_heads
        if proj_args.d_attn_out:
            self.out_dim = proj_args.d_attn_out
            if proj_args.d_attn_val is None:
                if proj_args.d_attn_kq is None:
                    self.n_kq_heads = -(proj_args.in_dim//-self.true_head_dim)
                self.n_v_heads = self.n_kq_heads
        elif proj_args.d_attn_val:
            self.out_dim = self.val_head_dim*n_heads
        elif proj_args.d_attn_kq:
            self.out_dim = self.kq_head_dim*n_heads

        #if not any([proj_args.d_attn_kq, proj_args.d_attn_val, proj_args.d_attn_out]):
        #    self.n_v_heads = self.n_kq_heads = 1
        #    self.out_dim = self.true_head_dim

        self.wqkv_base = nn.Linear(
            proj_args.in_dim,
            384,
            #self.n_kq_heads * self.true_head_dim,
            bias=False,
        )

        self.wq_offset = LoRAModule(
            16,
            384,
            rank=8,
            bias=False
        )


        self.wk_offset = LoRAModule(
            16,
            384,
            rank=8,
            bias=False,
        )


        self.wv_offset = LoRAModule(
            16,
            384,
            rank=8,
            bias=False,
        )

        self.wo = nn.Linear(
            #self.n_v_heads * self.true_head_dim,
            384,
            384,
            #self.out_dim,
            bias=False,
        )

        self.qk_need = self.kq_head_dim // 2 # to make the RoPE shape correct
        self.qk_factor = self.n_v_heads // self.n_kq_heads
    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, dim = x.shape
        x_base = self.wqkv_base(x)
        wq_offset = self.wq_offset(x)
        wk_offset = self.wk_offset(x)
        wv_offset = self.wv_offset(x)

        xq = x_base + wq_offset
        xk = x_base + wk_offset
        xv = x_base + wv_offset

        output_shape = xq.shape[:-1] + (384,)
        # B S D -> B S H D
        #xq = xq.view(bsz, seq_len, self.n_kq_heads, self.true_head_dim)
        xq = xq.view(bsz, seq_len, 16, 24)
        #xk = xk.view(bsz, seq_len, self.n_kq_heads, self.true_head_dim)
        xk = xk.view(bsz, seq_len, 16, 24)
        #xv = xv.view(bsz, seq_len, self.n_v_heads, self.true_head_dim)
        xv = xv.view(bsz, seq_len, 16, 24)

        fc = freq_cis[:seq_len]
        xq, xk = apply_rotary_emb(xq, xk, 1, fc)

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        #xk = repeat_kv(xk, self.heads_per_group, dim=2)
        #xv = repeat_kv(xv, self.heads_per_group, dim=2)
        #xq = repeat_kv(xq, self.qk_factor, dim=2)
        #xk = repeat_kv(xk, self.qk_factor, dim=2)

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        elif attn_impl == "fmha":
            assert mask is None or isinstance(mask, AttentionBias)
            output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)
            # This uses B S H D instead of B H S D of pytorch

        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, torch.Tensor))
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            mask = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=is_causal,
                attn_mask=mask,
            )
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )

        output = self.wo(output.reshape(output_shape))

        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))
        self.wq_offset.reset_parameters()
        self.wk_offset.reset_parameters()
        self.wv_offset.reset_parameters()
        nn.init.trunc_normal_(
            self.wqkv_base.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        attn_final_dim: int,
        proj_args: ProjectLayerArgs,
        mp_size: int = 1,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        out_dim = attn_final_dim
        if proj_args.d_ffn is not None:
            out_dim = proj_args.d_ffn
        self.w1 = nn.Linear(
            attn_final_dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            attn_final_dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            out_dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B S D
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std
        out_init_std = out_init_std / factor
        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )

class ProjectLayer(nn.Module):
    def __init__(self, args: LMTransformerArgs, i: int, type: str):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        #self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        if type == "up":
            assert args.project_up_layers is not None
            self.project_args: ProjectLayerArgs = args.project_up_layers[i]
        elif type == "down":
            assert args.project_down_layers is not None
            k = args.n_layers - len(args.project_down_layers)
            self.project_args: ProjectLayerArgs = args.project_down_layers[i-k]
        self.in_dim = self.project_args.in_dim
        self.head_dim = self.in_dim // self.n_heads

        self.attn_projs = False
        attn_final_dim = self.in_dim#args.dim//self.n_heads
        for i, d in enumerate((self.project_args.d_attn_out,
            self.project_args.d_attn_val,
            self.project_args.d_attn_kq)):
            if d is not None:
                attn_final_dim = d if i < 1 else d*self.n_heads
                self.attn_projs = True
                self.attn_proj = nn.Linear(self.in_dim, attn_final_dim, bias=False)
                if self.project_args.attn_proj_rand:
                    self.attn_proj.weight.requires_grad_(False)
                break

        if self.project_args.d_ffn is not None:
            self.ffn_proj = nn.Linear(attn_final_dim, self.project_args.d_ffn, bias=False)
            if self.project_args.ffn_proj_rand:
                self.ffn_proj.weight.requires_grad_(False)

        self.attention_norm = RMSNorm(self.in_dim, eps=args.norm_eps)
        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
            proj_args=self.project_args
        )

        self.ffn_norm = RMSNorm(attn_final_dim, eps=args.norm_eps)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            attn_final_dim=attn_final_dim,
            proj_args=self.project_args
        )
    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:

        attn_out = self.attention(
            self.attention_norm(x),
            freq_cis,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )
        if self.attn_projs:
            h = self.attn_proj(x) + attn_out
        else:
            h = x + attn_out

        ffn_out = self.feed_forward(self.ffn_norm(h))
        if self.project_args.d_ffn is not None:
            out = self.ffn_proj(h) + ffn_out
        else:
            out = h + ffn_out

        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()

        if hasattr(self, 'attn_proj'):
            nn.init.orthogonal_(
                self.attn_proj.weight
            )

        if hasattr(self, 'ffn_proj'):
            nn.init.orthogonal_(
                self.ffn_proj.weight
            )
