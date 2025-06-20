import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional, Union
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention
)
import math

from lingua.transformer import apply_rotary_emb, flex_attention_comp, repeat_kv


def shared_plus_lora(shared_proj, lora_proj, n_heads, head_dim):
    """
    shared_proj : (B, S, head_dim)                – base weight produces ONE head
    lora_proj   : (B, S, n_heads * head_dim)      – LoRA produces per-head delta
    returns     : (B, S, n_heads * head_dim)      – broadcast + add
    """
    B, S, _ = shared_proj.shape
    shared = (shared_proj
              .unsqueeze(2)                       # (B,S,1,D)
              .expand(B, S, n_heads, head_dim)    # (B,S,H,D)
              .reshape(B, S, n_heads * head_dim)) # (B,S,H*D)
    return shared + lora_proj

def project_output(x, wo_base, wo_lora):
    """
    x        : (B, S, H, D)   – attention result
    wo_base  : nn.Linear(D, E) – shared across heads
    wo_lora  : LoRAModule(H*D, E)

    returns  : (B, S, E)
    """
    B, S, H, D = x.shape
    # shared path – one weight reused for every head
    base = wo_base(x)          # (B,S,H,E) because nn.Linear acts on last dim
    base = base.sum(dim=2)     # (B,S,E)   aggregate heads (same as concat+matmul)

    # LoRA delta – full (H*D) view
    delta = wo_lora(x.reshape(B, S, H*D))  # (B,S,E)

    return base + delta


class LoRAModule(nn.Module):
    def __init__(self,
        in_dim,
        out_dim,
        rank=8,
        alpha=16,
        bias=False):

        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.bias = bias

        # LoRA params
        self.w_a = nn.Linear(
            in_dim,
            rank,
            bias=False,
        )  # A matrix

        self.w_b = nn.Linear(
            rank,
            out_dim,
            bias=False,
        )  # B matrix
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_b.weight)



    def forward(self, x):
        #  low-rank update
        lora = self.w_b(self.w_a(x)) * (self.alpha / self.rank)
        return lora

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            dim,
            1 * head_dim,
            bias=False,
        )
        self.wq_share = LoRAModule(
            dim,
            n_heads * head_dim,
            bias=False
        )

        self.wk = nn.Linear(
            dim,
            1 * head_dim,
            bias=False,
        )
        self.wk_share = LoRAModule(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wv = nn.Linear(
            dim,
            1 * head_dim,
            bias=False,
        )
        self.wv_share = LoRAModule(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            1 * head_dim,
            dim,
            bias=False,
        )
        self.wo_share = nn.Linear(
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
        # ---- projections ---------------------------------------------------------
        q_base = self.wq(x)           # (B,S,D)
        q_offset = self.wq_share(x)           # (B,S,H*D)
        xq = shared_plus_lora(q_base, q_offset,
                              n_heads=self.n_heads,
                              head_dim=self.head_dim)

        k_base = self.wk(x)
        k_offset = self.wk_share(x)
        xk = shared_plus_lora(k_base, k_offset,
                              n_heads=self.n_kv_heads,
                              head_dim=self.head_dim)

        v_base = self.wv(x)
        v_offset = self.wv_share(x)
        xv = shared_plus_lora(v_base, v_offset,
                              n_heads=self.n_kv_heads,
                              head_dim=self.head_dim)

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

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

        output = project_output(output, self.wo, self.wo_share)
        #output = self.wo(output.reshape(output_shape)) #+ self.wo_share(x.view_as(x)) but broadcast over the factor of n_heads larger out dim

        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )


        nn.init.trunc_normal_(
            self.wo.weight.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )

        for w in [self.wq_share, self.wk_share, self.wv_share, self.wo_share]:
            w.reset_parameters()
