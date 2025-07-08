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
    RMSNorm

)
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention
)
from .args import LMTransformerArgs, ProjectUpLayerArgs

import math
# George: Factorise the output layer for weight tying also
class FactorisedTiedOut(nn.Module):
    def __init__(self, tok_embeddings: nn.Embedding, project: nn.Linear):
        super().__init__()
        self.proj_down = TiedLinear(project)
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
    else:
        assert args.project_layers is not None
        for j in range(i-1, -1, -1):
            for k, d in enumerate((args.project_layers[j].d_ffn,
                args.project_layers[j].d_attn_out,
                args.project_layers[j].d_attn_val,
                args.project_layers[j].d_attn_kq)):
                if d is not None:
                    return d if k < 2 else d*args.n_heads
        if args.factorised_vocab.factorise:
            return args.factorised_vocab.d_factorised
        else:
            return args.factorised_vocab.d_emb


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        proj_args: ProjectUpLayerArgs
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.kq_head_dim = head_dim
        self.val_head_dim = head_dim
        self.out_dim = dim
        if proj_args.d_attn_kq:
            self.kq_head_dim = proj_args.d_attn_kq
        if proj_args.d_attn_val:
            self.val_head_dim = proj_args.d_attn_val
        elif self.kq_head_dim > head_dim:
            self.val_head_dim = self.kq_head_dim # we are not projecting up in the value head,
            # but we are projecting up in the key and query heads
        if proj_args.d_attn_out:
            self.out_dim = proj_args.d_attn_out
        elif self.val_head_dim > head_dim:
            self.out_dim = self.val_head_dim

        self.wq = nn.Linear(
            proj_args.in_dim,
            n_heads * self.kq_head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            proj_args.in_dim,
            n_kv_heads * self.kq_head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            proj_args.in_dim,
            n_kv_heads * self.val_head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * self.val_head_dim,
            self.out_dim,
            bias=False,
        )

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
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape[:-1] + (self.n_heads * self.val_head_dim,)
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.kq_head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.kq_head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.val_head_dim)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

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

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight,
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
        proj_args: ProjectUpLayerArgs,
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
    def __init__(self, args: LMTransformerArgs, i: int):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        assert args.project_layers is not None
        self.project_args: ProjectUpLayerArgs = args.project_layers[i]
        self.in_dim = self.project_args.in_dim

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.attn_projs = False
        attn_final_dim = self.in_dim
        for d in (self.project_args.d_attn_out,
            self.project_args.d_attn_val,
            self.project_args.d_attn_kq):
            if d is not None:
                attn_final_dim = d
                self.attn_projs = True
                self.attn_proj = nn.Linear(self.in_dim, attn_final_dim, bias=False)
                if self.project_args.attn_proj_rand:
                    self.attn_proj.weight.requires_grad_(False)
                break

        if self.project_args.d_ffn is not None:
            self.ffn_proj = nn.Linear(attn_final_dim, self.project_args.d_ffn, bias=False)
            if self.project_args.ffn_proj_rand:
                self.ffn_proj.weight.requires_grad_(False)

        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
            proj_args=self.project_args
        )
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
            nn.init.kaiming_uniform_(
                self.attn_proj.weight,
                a=math.sqrt(5)
            )

        if hasattr(self, 'ffn_proj'):
            nn.init.kaiming_uniform_(
                self.ffn_proj.weight,
                a=math.sqrt(5)
            )
