# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)

from xformers.ops import fmha, AttentionBias
from lingua.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    RMSNorm,
    TiedLinear,
    cross_entropy,
    # EXPERIMENT 3 -------------------------------------------------------------------------------------
    TransformerBlock,
    FeedForward,
    Attention,
    # EXPERIMENT 3 -------------------------------------------------------------------------------------
)


def create_causal_mask(seqlen, attn_impl, sliding_window):
    if sliding_window is not None and attn_impl == "xformers":
        return fmha.attn_bias.LocalAttentionFromBottomRightMask(
            window_left=sliding_window - 1, window_right=0
        )
    elif attn_impl == "xformers":
        return fmha.attn_bias.LowerTriangularMask()
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


def attention_flops_per_token(n_layers, seq_len, dim, causal):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


@dataclass
class LMTransformerArgs(BaseTransformerArgs):
    seed: int = 42
    rank: int = -1
    vocab_size: int = -1
    weight_tying: bool = False
    sliding_window: Optional[int] = None
    # EXPERIMENT 3 -------------------------------------------------------------------------------------
    # Pass D_emb in the config model arguments
    D_emb: int | None = None
    # EXPERIMENT 3 -------------------------------------------------------------------------------------


# George: Factorise the output layer for weight tying also
class FactorisedTiedLinear(nn.Module):
    def __init__(self, tok_embeddings1: nn.Embedding, tok_embeddings2: nn.Linear):
        super().__init__()
        self.tok_embeddings1 = tok_embeddings1
        self.tok_embeddings2 = tok_embeddings2

    def forward(self, x: torch.Tensor):
        intermediate = torch.matmul(x, self.tok_embeddings2.weight)
        logits = torch.matmul(intermediate, self.tok_embeddings1.weight.t())
        return logits

# EXPERIMENT 3 -------------------------------------------------------------------------------------
# Custom FFN inherited from the class in lingua/transformer.py, this one allows different input/output dimensions
# inherits w1 and w3 (mapping input_dim -> hidden_dim) but redefines w2 to map to output_dim
class CustomFeedForward(FeedForward):
    def __init__(self, input_dim, output_dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        super().__init__(dim=input_dim, hidden_dim=hidden_dim, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier)
        self.w2 = nn.Linear(self.w1.out_features, output_dim, bias=False)


# Custom transformer block class that allows different input/output dimensions
class CustomTransformerBlock(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        n_heads = args.n_heads
        head_dim = args.dim // n_heads
        self.attention = Attention(
            dim=input_dim,
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            rope_theta=args.rope_theta,
        )
        self.attention_norm = RMSNorm(input_dim, eps=args.norm_eps)
        self.ffn = CustomFeedForward(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=4 * input_dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.ffn_norm = RMSNorm(input_dim, eps=args.norm_eps)
        self.residual_projection = nn.Linear(input_dim, output_dim)

    def forward(self, x, freq_cis, tok_idx, mask, attn_impl):
        h = x + self.attention(self.attention_norm(x), freq_cis, tok_idx, mask, attn_impl)
        ffn_out = self.ffn(self.ffn_norm(h))
        h_proj = self.residual_projection(h)
        out = h_proj + ffn_out
        return out
# EXPERIMENT 3 -------------------------------------------------------------------------------------


class LMTransformer(BaseTransformer):
    def __init__(self, args: LMTransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window

        assert args.vocab_size > 0

        # Use factorised embeddings if rank is specified, else use original Lingua embeddings
        if args.rank > 0:
            self.use_factorised = True
            self.tok_embeddings1 = torch.nn.Embedding(args.vocab_size, args.rank)
            # EXPERIMENT 3 -------------------------------------------------------------------------------------
            # Use D_emb instead of dim for the embedding matrix
            self.tok_embeddings2 = torch.nn.Linear(args.rank, args.D_emb)
            # EXPERIMENT 3 -------------------------------------------------------------------------------------
        else:
            self.use_factorised = False
            # EXPERIMENT 3 -------------------------------------------------------------------------------------
            # Use D_emb instead of dim for the embedding matrix
            self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.D_emb)
            # EXPERIMENT 3 -------------------------------------------------------------------------------------

        # EXPERIMENT 3 -------------------------------------------------------------------------------------
        # Set first block to be the CustomTransformerBlock
        self.first_block = CustomTransformerBlock(input_dim=args.D_emb, output_dim=args.dim, args=args)

        # Set remaining blocks to use the standard TransformerBlock
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers - 1)])
        # EXPERIMENT 3 -------------------------------------------------------------------------------------

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)


        # Configure output layer based on weight tying and embedding type
        if args.weight_tying:
            if self.use_factorised:
                self.output = FactorisedTiedLinear(self.tok_embeddings1, self.tok_embeddings2)
            else:
                self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(
                args.dim,
                args.vocab_size,
                bias=False,
            )


    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ):
        bsz, seqlen = token_values.shape

        # Compute embeddings based on whether factorised embeddings are used
        if self.use_factorised:
            tok_emb = self.tok_embeddings1(token_values)
            h = self.tok_embeddings2(tok_emb)
        else:
            h = self.tok_embeddings(token_values)

        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )

        # EXPERIMENT 3 -------------------------------------------------------------------------------------
        # Forward pass thru custom first block
        freq_cis = self.rope_embeddings(seqlen=seqlen, tok_idx=tok_idx)
        h = self.first_block(h, freq_cis, tok_idx, mask, attn_impl)

        # Forward pass thru remaining blocks (standard TransformerBlocks)
        for layer in self.layers:
            h = layer(h, freq_cis, tok_idx, mask, attn_impl)
        # EXPERIMENT 3 -------------------------------------------------------------------------------------


        logits = self.output(self.norm(h))
        if target is not None:
            return cross_entropy(logits, target)
        else:
            return logits

    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        # EXPERIMENT 3 -------------------------------------------------------------------------------------
        # Reset parameters for custom first block attention + FFN
        self.first_block.attention_norm.reset_parameters()
        self.first_block.ffn_norm.reset_parameters()
        # EXPERIMENT 3 -------------------------------------------------------------------------------------


        # Initialise embedding weights based on embedding type
        if self.use_factorised:
            nn.init.trunc_normal_( # Factorised
                self.tok_embeddings1.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
            nn.init.trunc_normal_( # Factorised
                self.tok_embeddings2.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
        else:
            nn.init.trunc_normal_( # Original lingua embeddings
                self.tok_embeddings.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
        # EXPERIMENT 3 -------------------------------------------------------------------------------------
        # Residual connection initialisation for projection of h -> h_proj so it matches custom ffn output dim
        nn.init.trunc_normal_(
            self.first_block.residual_projection.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

        # Initialise weights for the mapping of hidden_dim -> output_dim
        nn.init.trunc_normal_(
            self.first_block.ffn.w2.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        # EXPERIMENT 3 -------------------------------------------------------------------------------------



# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: LMTransformerArgs):
    group_plan: Tuple[int, bool] = []

    # Grouping and output seperately
    if model_args.rank > 0:
        group_plan.append(("tok_embeddings1", False))  # Factorised 
        group_plan.append(("tok_embeddings2", False))  # Factorised 
    else:
        group_plan.append(("tok_embeddings", False)) # Original lingua embeddings

    # EXPERIMENT 3 -------------------------------------------------------------------------------------
    # FSDP grouping for the first (altered) block
    group_plan.append(("first_block", False))

    # FSDP grouping for the remaining (unaltered) blocks
    for i in range(model_args.n_layers - 1):
        group_plan.append((f"layers.{i}", False))
    # EXPERIMENT 3 -------------------------------------------------------------------------------------


    group_plan.append(("output", True))

    return group_plan


# Optional and only used for model/tensor parallelism when tp_size > 1
def tp_parallelize(model, tp_mesh, model_args: LMTransformerArgs, distributed_args):
    assert model_args.dim % distributed_args.tp_size == 0
    assert model_args.vocab_size % distributed_args.tp_size == 0
    assert model_args.n_heads % distributed_args.tp_size == 0
    assert (model_args.n_kv_heads or 0) % distributed_args.tp_size == 0
    assert model_args.n_heads % (model_args.n_kv_heads or 1) == 0

    # Embedding layer tp
    main_plan = {}

    if model_args.rank > 0:
        main_plan["tok_embeddings1"] = ColwiseParallel( # Factorised embeddings
            input_layouts=Replicate(), output_layouts=Shard(1)
        )
        main_plan["tok_embeddings2"] = RowwiseParallel(
            input_layouts=Shard(1), output_layouts=Replicate()
        )
    else:
        main_plan["tok_embeddings"] = ColwiseParallel( # Original Lingua embeddings
            input_layouts=Replicate(), output_layouts=Shard(1)
        )

    main_plan["norm"] = SequenceParallel()
    main_plan["output"] = ColwiseParallel(
        input_layouts=Shard(1), output_layouts=Replicate()
    )

    parallelize_module(
        model,
        tp_mesh,
        main_plan,
    )

    # Attention layers tp
    for layer in model.layers:
        layer_plan = {}

        layer_plan["attention"] = PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        )
        layer_plan["attention_norm"] = SequenceParallel()
        layer_plan["attention.wq"] = ColwiseParallel()
        layer_plan["attention.wk"] = ColwiseParallel()
        layer_plan["attention.wv"] = ColwiseParallel()
        layer_plan["attention.wo"] = RowwiseParallel(output_layouts=Shard(1))

        # Feedforward layers tp
        layer_plan["feed_forward"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )
        layer_plan["ffn_norm"] = SequenceParallel()
        layer_plan["feed_forward.w1"] = ColwiseParallel()
        layer_plan["feed_forward.w3"] = ColwiseParallel()
        layer_plan["feed_forward.w2"] = RowwiseParallel(output_layouts=Shard(1))

        parallelize_module(
            layer,
            tp_mesh,
            layer_plan,
        )

        # Adjusting the number of heads and kv heads according to the tp size
        attn_layer = layer.attention
        attn_layer.n_heads = attn_layer.n_heads // distributed_args.tp_size
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // distributed_args.tp_size
