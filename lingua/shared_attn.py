from numpy._core.numerictypes import bool_
import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional, Union, Tuple
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention
)
import math

from lingua.transformer import apply_rotary_emb, flex_attention_comp, repeat_kv, TiedLinear

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
def str_to_attr(s: str):
    if s == 'q':
        return 'wq_base'
    if s == 'k':
        return 'wk_base'
    if s == 'v':
        return 'wv_base'

    raise ValueError(f"Invalid weight name: {s}")
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        qkv_sharing: Optional[Tuple[Tuple[str, ...], ...]],
        head_sharing: bool,
        grouping: int,
        two_step: bool,
        rank: int
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads
        self.qkv_sharing = qkv_sharing
        self.head_sharing = head_sharing
        self.grouping = grouping if grouping else 1
        self.two_step = two_step
        self.rank = rank

        if qkv_sharing:
            self.n_kv_heads = n_heads
            n_kv_heads = n_heads
            # note this means no GQA if there's any q sharing sharing
            base_dim = head_dim*self.grouping if head_sharing else head_dim*n_heads
            for weight_group in qkv_sharing:
                for i, weight_name in enumerate(weight_group):
                    if i == 0:
                        w_base = nn.Linear(
                            dim,
                            base_dim,
                            bias=False
                        )
                        setattr(self, str_to_attr(weight_name), w_base)
                    else:
                        w = TiedLinear(w_base)
                        setattr(self, str_to_attr(weight_name), w)
        else:
            self.wq_base = nn.Linear(
                dim,
                head_dim * (self.grouping if head_sharing else n_heads),
                bias=False,
            )

            self.wk_base = nn.Linear(
                dim,
                head_dim * (self.grouping if head_sharing else n_kv_heads),
                bias=False,
            )

            self.wv_base = nn.Linear(
                dim,
                head_dim * (self.grouping if head_sharing else n_kv_heads),
                bias=False,
            )
        if two_step:
            self.head_offset = LoRAModule(
                dim,
                n_heads * head_dim,
                rank=rank,
                bias=False
            )

            self.wq_only_offset = LoRAModule(
                dim,
                head_dim,
                rank=rank,
                bias=False
            )

            self.wk_only_offset = LoRAModule(
                dim,
                head_dim,
                rank=rank,
                bias=False
            )

            self.wv_only_offset = LoRAModule(
                dim,
                head_dim,
                rank=rank,
                bias=False
            )
        else:
            #offset_n_heads = n_heads if head_sharing else 1
            #offset_n_kv_heads = n_kv_heads if head_sharing else 1

            self.wq_offset = LoRAModule(
                dim,
                n_heads * head_dim,
                rank=rank,
                bias=False
            )


            self.wk_offset = LoRAModule(
                dim,
                n_kv_heads * head_dim,
                rank=rank,
                bias=False,
            )


            self.wv_offset = LoRAModule(
                dim,
                n_kv_heads * head_dim,
                rank=rank,
                bias=False,
            )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
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


        q_base = self.wq_base(x)
        k_base = self.wk_base(x)
        v_base = self.wv_base(x)

        if self.two_step:
            head_offset = self.head_offset(x)
            wq_only_offset = self.wq_only_offset(x)
            wk_only_offset = self.wk_only_offset(x)
            wv_only_offset = self.wv_only_offset(x)

            wq_offset = broadcast_add(wq_only_offset, head_offset).contiguous()
            wk_offset = broadcast_add(wk_only_offset, head_offset).contiguous()
            wv_offset = broadcast_add(wv_only_offset, head_offset).contiguous()
        else:
            wq_offset = self.wq_offset(x)
            wk_offset = self.wk_offset(x)
            wv_offset = self.wv_offset(x)

        xq = broadcast_add(q_base, wq_offset, g=self.grouping)
        xk = broadcast_add(k_base, wk_offset, g=self.grouping)
        xv = broadcast_add(v_base, wv_offset, g=self.grouping)

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        if not self.qkv_sharing:
            xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
            xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        else:
            xk = xk.view(bsz, seq_len, self.n_heads, self.head_dim)
            xv = xv.view(bsz, seq_len, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        # since qkv sharing means we don't use GQA so no need to repeat kv
        if not self.qkv_sharing:
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

        for attr in ['wq_base', 'wk_base', 'wv_base']:
            if hasattr(self, attr):
                w = getattr(self, attr)
                if hasattr(w, 'weight'):  # Check if it's a real Linear layer, not TiedLinear
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

        # Initialize LoRA modules
        if self.two_step:
            self.head_offset.reset_parameters()
            self.wq_only_offset.reset_parameters()
            self.wk_only_offset.reset_parameters()
            self.wv_only_offset.reset_parameters()
        else:
            self.wq_offset.reset_parameters()
            self.wk_offset.reset_parameters()
            self.wv_offset.reset_parameters()
