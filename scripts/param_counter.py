#!/usr/bin/env python3
"""
LLM Parameter Count Calculator
Calculates total parameters for transformer models with optional attention sharing
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

from torch import xpu


def calculate_ffn_hidden_dim(dim: int, ffn_dim_multiplier: Optional[float], multiple_of: int) -> int:
    """Calculate the hidden dimension for the feedforward network"""
    hidden_dim = int(2 * 4 * dim / 3)  # Base: 4 * dim, then 2/3
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    # Round to multiple_of
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def calculate_standard_attention_params(dim: int, n_heads: int, n_kv_heads: int, head_dim: int) -> int:
    """Calculate parameters for standard attention"""
    params = 0
    # Query projection
    params += dim * (n_heads * head_dim)
    # Key projection
    params += dim * (n_kv_heads * head_dim)
    # Value projection
    params += dim * (n_kv_heads * head_dim)
    # Output projection
    params += (n_heads * head_dim) * dim
    return params


def calculate_shared_attention_params(
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    qkv_sharing: Optional[Tuple[Tuple[str, ...], ...]],
    head_sharing: bool,
    grouping: Optional[int],
    rank: Optional[int],
    two_step: bool
) -> int:
    """Calculate parameters for shared attention"""
    params = 0
    grouping = grouping if grouping else 1
    rank = rank if rank else 8  # Default rank

    # Base projections
    if qkv_sharing:
        # With qkv_sharing, we create shared base projections
        base_dim = head_dim * grouping if head_sharing else head_dim * n_heads
        unique_bases = set()

        for weight_group in qkv_sharing:
            # Only the first in each group creates a new parameter matrix
            if len(weight_group) > 0:
                unique_bases.add(weight_group[0])

        # Each unique base creates one projection matrix
        params += len(unique_bases) * (dim * base_dim)
    else:
        # Without qkv_sharing, we have separate base projections
        if head_sharing:
            params += dim * (head_dim * grouping)  # wq_base
            params += dim * (head_dim * grouping)  # wk_base
            params += dim * (head_dim * grouping)  # wv_base
        else:
            params += dim * (n_heads * head_dim)    # wq_base
            params += dim * (n_kv_heads * head_dim)  # wk_base
            params += dim * (n_kv_heads * head_dim)  # wv_base

    # LoRA offset parameters
    if two_step:
        # Shared head offset + individual offsets
        params += dim * rank + rank * (n_heads * head_dim)  # head_offset (w_a + w_b)
        params += dim * rank + rank * head_dim  # wq_only_offset
        params += dim * rank + rank * head_dim  # wk_only_offset
        params += dim * rank + rank * head_dim  # wv_only_offset
    else:
        # Direct per-projection offsets
        params += dim * rank + rank * (n_heads * head_dim)    # wq_offset
        params += dim * rank + rank * (n_kv_heads * head_dim)  # wk_offset
        params += dim * rank + rank * (n_kv_heads * head_dim)  # wv_offset

    # Output projection (same for both)
    params += (n_heads * head_dim) * dim

    return params


def calculate_transformer_params(
    # BaseTransformerArgs
    dim: int = 512,
    n_layers: int = 8,
    head_dim: Optional[int] = None,
    n_heads: Optional[int] = None,
    n_kv_heads: Optional[int] = None,
    ffn_dim_multiplier: Optional[float] = None,
    multiple_of: int = 256,
    norm_eps: float = 1e-5,
    rope_theta: float = 10000.0,
    init_base_std: Optional[float] = None,
    init_std_factor: str = "disabled",
    max_seqlen: int = 1024,
    attn_sharing: bool = False,
    qkv_sharing: Optional[Tuple[Tuple[str, ...], ...]] = None,
    head_sharing: bool = False,
    grouping: Optional[int] = None,
    rank: Optional[int] = None,
    two_step: bool = False,
    # LMTransformerArgs
    seed: int = 42,
    vocab_size: int = -1,
    weight_tying: bool = False,
    sliding_window: Optional[int] = None,
) -> dict:
    """Calculate total parameters for the transformer model"""

    # Validate and set defaults
    assert vocab_size > 0, "vocab_size must be positive"
    assert (head_dim is not None) or (n_heads is not None), "Must specify head_dim or n_heads"

    if head_dim is None:
        head_dim = dim // n_heads
    if n_heads is None:
        n_heads = dim // head_dim
    if n_kv_heads is None:
        n_kv_heads = n_heads

    assert dim % n_heads == 0, "dim must be divisible by n_heads"
    assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    params = {}

    # Token embeddings
    params['embeddings'] = vocab_size * dim

    # Output projection (if not weight tied)
    if weight_tying:
        params['output'] = 0  # Reuses embedding weights
    else:
        params['output'] = dim * vocab_size

    # Final layer norm
    params['final_norm'] = dim

    # Per-layer parameters
    params['layers'] = {}

    # Calculate FFN hidden dimension
    ffn_hidden_dim = calculate_ffn_hidden_dim(dim, ffn_dim_multiplier, multiple_of)

    for i in range(n_layers):
        layer_params = {}

        # Attention
        if attn_sharing:
            layer_params['attention'] = calculate_shared_attention_params(
                dim, n_heads, n_kv_heads, head_dim,
                qkv_sharing, head_sharing, grouping, rank, two_step
            )
        else:
            layer_params['attention'] = calculate_standard_attention_params(
                dim, n_heads, n_kv_heads, head_dim
            )

        # Feedforward
        layer_params['ffn'] = (
            dim * ffn_hidden_dim +      # w1
            dim * ffn_hidden_dim +      # w3
            ffn_hidden_dim * dim        # w2
        )

        # Layer norms
        layer_params['attention_norm'] = dim
        layer_params['ffn_norm'] = dim

        params['layers'][i] = layer_params

    # Calculate totals
    total_params = params['embeddings'] + params['output'] + params['final_norm']
    total_per_layer = 0

    for i in range(n_layers):
        layer_total = sum(params['layers'][i].values())
        total_per_layer += layer_total
        total_params += layer_total

    params['total_per_layer'] = total_per_layer
    params['total'] = total_params

    return params


def format_params(num_params: int) -> str:
    """Format parameter count in human-readable form"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def print_param_breakdown(params: dict, model_name: str = "Model", weight_tying: bool = False):
    """Print detailed parameter breakdown"""
    total = params['total']

    print(f"\n{'='*60}")
    print(f"{model_name} Parameter Breakdown")
    print(f"{'='*60}")

    # Handle embedding/output parameters based on weight tying
    if weight_tying:
        embed_output_params = params['embeddings']
        embed_output_pct = (embed_output_params / total) * 100
        print(f"Embedding Parameters (tied): {embed_output_params:,} ({format_params(embed_output_params)}, {embed_output_pct:.1f}% of total)")
    else:
        embed_params = params['embeddings']
        output_params = params['output']
        embed_output_params = embed_params + output_params
        embed_output_pct = (embed_output_params / total) * 100
        print(f"Embedding Parameters (no tying): {embed_params:,} ({format_params(embed_params)}, {(embed_params/total)*100:.1f}% of total)")
        print(f"Output Parameters: {output_params:,} ({format_params(output_params)}, {(output_params/total)*100:.1f}% of total)")

    norm_pct = (params['final_norm'] / total) * 100
    print(f"Final Norm: {params['final_norm']:,} ({format_params(params['final_norm'])}, {norm_pct:.1f}% of total)")

    print(f"\nPer-Layer Breakdown (first layer):")
    if 0 in params['layers']:
        layer = params['layers'][0]
        layer_total = sum(layer.values())

        # Attention
        attn_layer_pct = (layer['attention'] / layer_total) * 100
        attn_total_pct = (layer['attention']*len(params['layers']) / total) * 100
        print(f"  Attention: {layer['attention']:,} ({format_params(layer['attention'])}, {attn_layer_pct:.1f}% of layer, {attn_total_pct:.1f}% of total)")

        # FFN
        ffn_layer_pct = (layer['ffn'] / layer_total) * 100
        ffn_total_pct = (layer['ffn']*len(params['layers']) / total) * 100
        print(f"  FFN: {layer['ffn']:,} ({format_params(layer['ffn'])}, {ffn_layer_pct:.1f}% of layer, {ffn_total_pct:.1f}% of total)")

        # Attention Norm
        attn_norm_layer_pct = (layer['attention_norm'] / layer_total) * 100
        attn_norm_total_pct = (layer['attention_norm']*len(params['layers']) / total) * 100
        print(f"  Attention Norm: {layer['attention_norm']:,} ({format_params(layer['attention_norm'])}, {attn_norm_layer_pct:.1f}% of layer, {attn_norm_total_pct:.1f}% of total)")

        # FFN Norm
        ffn_norm_layer_pct = (layer['ffn_norm'] / layer_total) * 100
        ffn_norm_total_pct = (layer['ffn_norm']*len(params['layers']) / total) * 100
        print(f"  FFN Norm: {layer['ffn_norm']:,} ({format_params(layer['ffn_norm'])}, {ffn_norm_layer_pct:.1f}% of layer, {ffn_norm_total_pct:.1f}% of total)")

        # Total per layer
        layer_total_pct = (layer_total / total) * 100
        print(f"  Total per layer: {layer_total:,} ({format_params(layer_total)}, {layer_total_pct:.1f}% of total)")

    all_layers_pct = (params['total_per_layer'] / total) * 100
    print(f"\nTotal for all layers: {params['total_per_layer']:,} ({format_params(params['total_per_layer'])}, {all_layers_pct:.1f}% of total)")

    print(f"\n{'='*60}")
    print(f"TOTAL PARAMETERS: {params['total']:,} ({format_params(params['total'])})")
    print(f"{'='*60}")


# Example usage with configurable hyperparameters
if __name__ == "__main__":
    # ========================================
    # CONFIGURE YOUR HYPERPARAMETERS HERE
    # ========================================

    # Model configuration
    config = {
        # BaseTransformerArgs
        "dim": 512,
        "n_layers": 12,
        #"n_layers": 14,
        "head_dim": None,
        "n_heads": 16,
        "n_kv_heads": 8,  # For GQA
        #"ffn_dim_multiplier": 0.8,
        "ffn_dim_multiplier": 1.125,
        "multiple_of": 256,
        "norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "init_base_std": None,
        "init_std_factor": "disabled",
        "max_seqlen": 4096,

        # Attention sharing configuration
        "attn_sharing": False,  # Set to True to enable shared attention
        "qkv_sharing": None,
        #"qkv_sharing": (("q", "k","v"),),
        "head_sharing": True,
        "grouping": 1,  # Number of groups for sharing
        "rank": 8,  # LoRA rank
        "two_step": False,

        # LMTransformerArgs
        "seed": 42,
        "vocab_size": 32000,
        "weight_tying": False,
        "sliding_window": None,
    }

    # Calculate parameters without attention sharing
    print("\n" + "="*60)
    print("CALCULATING PARAMETER COUNTS")
    print("="*60)

    # Standard model
    params_standard = calculate_transformer_params(**config)
    print_param_breakdown(params_standard, "Baseline Shared Transformer (rank=8)")

    rank_16 = calculate_transformer_params(**{**config, "rank": 16})
    rank_32 = calculate_transformer_params(**{**config, "rank": 32})
    rank_64 = calculate_transformer_params(**{**config, "rank": 64})
    rank_128 = calculate_transformer_params(**{**config, "rank": 128})
    print("Rank 16 Increase: ", rank_16['total'] - params_standard['total'])
    print("Rank 32 Increase: ", rank_32['total'] - params_standard['total'])
    print("Rank 64 Increase: ", rank_64['total'] - params_standard['total'])
    print("Rank 128 Increase: ", rank_128['total'] - params_standard['total'])

"""
    # Model with attention sharing
    params_shared = calculate_transformer_params(**{**config, "attn_sharing": True})
    print_param_breakdown(params_shared, "Transformer with Attention Sharing")

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    reduction = params_standard['total'] - params_shared['total']
    reduction_pct = (reduction / params_standard['total']) * 100
    print(f"Parameter Reduction: {format_params(reduction)} ({reduction:,})")
    print(f"Reduction Percentage: {reduction_pct:.2f}%")
    print(f"Compression Ratio: {params_standard['total'] / params_shared['total']:.2f}x")

    # Example with different sharing configurations
    print("\n" + "="*60)
    print("EXAMPLE: QKV SHARING CONFIGURATIONS")
    print("="*60)

    # Share Q and K
    config_qk_share = {**config, "attn_sharing": True, "qkv_sharing": (("q", "k"),)}
    params_qk = calculate_transformer_params(**config_qk_share)
    print(f"Q-K Sharing: {format_params(params_qk['total'])} parameters")

    # Share all QKV
    config_qkv_share = {**config, "attn_sharing": True, "qkv_sharing": (("q", "k", "v"),)}
    params_qkv = calculate_transformer_params(**config_qkv_share)
    print(f"Q-K-V Sharing: {format_params(params_qkv['total'])} parameters")
"""
