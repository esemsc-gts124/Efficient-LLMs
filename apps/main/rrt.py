# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import math
import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, BlockMask, flex_attention
from torch.nn import functional as F

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
    repeat_kv,
    apply_rotary_emb,
)

flex_attention_comp = torch.compile(flex_attention)


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


# @dataclass
# class LMTransformerArgs(BaseTransformerArgs):
#     seed: int = 42
#     rank: int = -1
#     vocab_size: int = -1
#     weight_tying: bool = False
#     sliding_window: Optional[int] = None
#     lora_rank: int = 8  # Rank of loras for weight sharing layers
#     ffn_dim: int = None
#     layer_groups: list  # Grouping for which layers share weights

@dataclass
class LMTransformerArgs(BaseTransformerArgs):
    layer_groups: Optional[list] = None  # Grouping for which layers share weights
    seed: int = 42
    rank: int = -1
    vocab_size: int = -1
    weight_tying: bool = False
    sliding_window: Optional[int] = None
    lora_rank: int = 8  # Rank of loras for weight sharing layers
    ffn_dim: int = None

    def __post_init__(self):
        # compute ffn_dim if not provided
        if self.ffn_dim is None:
            hidden_dim = 4 * self.dim
            hidden_dim = int(2 * hidden_dim / 3)
            if self.ffn_dim_multiplier is not None:
                hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
            self.ffn_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)


# George: Factorise the output layer for weight tying also
class FactorisedTiedLinear(nn.Module):
    def __init__(self, tok_embeddings1: nn.Embedding, tok_embeddings2: nn.Linear):
        super().__init__()
        self.tok_embeddings1 = tok_embeddings1
        self.tok_embeddings2 = tok_embeddings2

    def forward(self, x: torch.Tensor):
        intermediate = torch.matmul(x, self.tok_embeddings2.weight)
        # intermediate = F.linear(x, self.tok_embeddings2.weight.t()) # change from matmul to linear

        # logits = self.tok_embeddings1(intermediate)#torch.matmul(intermediate, self.tok_embeddings1.weight.t())
        # logits = F.linear(intermediate, self.tok_embeddings1.weight) # change from matmul to linear
        logits = torch.matmul(intermediate, self.tok_embeddings1.weight.t())
        return logits




# RRT: class to use shared weights with layer-specific LoRAs
class SharedLinearWithLoRA(nn.Module):
    def __init__(self, shared_weight, in_features, out_features, lora_rank):
        super().__init__()
        self.shared_weight = shared_weight # Shared weight tensor across layers
        self.lora_rank = lora_rank
        if lora_rank > 0:
        # Layer-specific LoRA parameters
            self.lora_B = nn.Linear(lora_rank, out_features, bias=False)
            self.lora_A = nn.Linear(in_features, lora_rank, bias=False)

            # Initialise LoRA parameters
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        self.in_features = in_features
        self.out_features = out_features



    def forward(self, x):
        # Base computation w shared weights
        base_output = self.shared_weight(x)#torch.matmul(x, self.shared_weight.weight.t())
        # LoRA adjustment
        if self.lora_rank > 0:
            lora_output = self.lora_A(x)#torch.matmul(x, self.lora_B.weight.t()) # x @ B.T
            lora_output = self.lora_B(lora_output)#torch.matmul(lora_output, self.lora_A.weight.t()) # (x @ B.T) @ A.T
            return base_output + lora_output
        return base_output
    def reset_parameters(self):
        if self.lora_rank > 0:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))



# RRT: class for attention using SharedLinearWithLoRA for linear transformations
class AttentionWithSharedWeights(nn.Module):
    def __init__(
            self,
            args,
            shared_wq,
            shared_wk,
            shared_wv,
            shared_wo,
            lora_rank
    ):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads or args.n_heads
        self.head_dim = args.head_dim
        self.dim = args.dim

        # Shared weights with LoRA for Q, K, V and O projections
        self.wq = SharedLinearWithLoRA(shared_wq, args.dim, args.n_heads * args.head_dim, lora_rank)
        self.wk = SharedLinearWithLoRA(shared_wk, args.dim, args.n_kv_heads * args.head_dim, lora_rank)
        self.wv = SharedLinearWithLoRA(shared_wv, args.dim, args.n_kv_heads * args.head_dim, lora_rank)
        self.wo = SharedLinearWithLoRA(shared_wo, args.n_heads * args.head_dim, args.dim, lora_rank)


    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # Reshape for multi-head attention
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # Handle KV cache if present
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        # Repeat keys and values for grouped-query attention
        if self.n_kv_heads < self.n_heads:
            xk = repeat_kv(xk, self.n_heads // self.n_kv_heads, dim=2)
            xv = repeat_kv(xv, self.n_heads // self.n_kv_heads, dim=2)

        # Attention computation based on attn_impl
        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()
        elif attn_impl == "fmha":
            assert mask is None or isinstance(mask, AttentionBias)
            output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)
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
            output = output.transpose(1, 2).contiguous()
        else:
            raise NotImplementedError(f"Attention implementation {attn_impl} not supported")

        # Final projection
        output = self.wo(output.view(bsz, seq_len, -1))
        return output
    def reset_parameters(self):
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.wq.reset_parameters()
        self.wo.reset_parameters()


# RRT: FFN cusing shared weights with LoRA
class FeedForwardWithSharedWeights(nn.Module):
    def __init__(
            self,
            args,
            shared_w1,
            shared_w2,
            shared_w3,
            lora_rank,
    ):
        super().__init__()
        self.w1 = SharedLinearWithLoRA(shared_w1, args.dim, args.ffn_dim, lora_rank)
        self.w2 = SharedLinearWithLoRA(shared_w2, args.ffn_dim, args.dim, lora_rank)
        self.w3 = SharedLinearWithLoRA(shared_w3, args.dim, args.ffn_dim, lora_rank)

    def forward(self, x: torch.Tensor):
        # B S D
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output

    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.w3.reset_parameters()

# RRT: transformer block class utilising shared weights
class TransformerBlockWithSharedWeights(nn.Module):
    def __init__(
            self,
            args,
            shared_attention_weights,
            shared_ffn_weights,
            lora_rank
    ):
        super().__init__()
        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        self.attention = AttentionWithSharedWeights(
            args,
            shared_attention_weights["wq"],
            shared_attention_weights["wk"],
            shared_attention_weights["wv"],
            shared_attention_weights["wo"],
            lora_rank
        )
        self.feed_forward = FeedForwardWithSharedWeights(
            args,
            shared_ffn_weights["w1"],
            shared_ffn_weights["w2"],
            shared_ffn_weights["w3"],
            lora_rank
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)


    def forward(self, x, freq_cis, tok_idx, mask, attn_impl):
        h = x + self.attention(
            self.attention_norm(x),
            freq_cis,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    # If any issues: get rid of this and redo weight initialisation for individual components
    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters()
        self.feed_forward.reset_parameters()
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()




class LMTransformer(BaseTransformer):
    def __init__(self, args: LMTransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window

        assert args.vocab_size > 0

        if args.head_dim is None:
            args.head_dim = args.dim // args.n_heads
        assert args.dim % args.n_heads == 0, "dim must be divisible by n_heads"

        if args.n_kv_heads is None:
            args.n_kv_heads = args.n_heads
        assert args.n_heads % args.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        # Use factorised embeddings if rank is specified, else use original Lingua embeddings
        if args.rank > 0:
            self.use_factorised = True
            self.tok_embeddings1 = torch.nn.Embedding(args.vocab_size, args.rank)
            self.tok_embeddings2 = torch.nn.Linear(args.rank, args.dim, bias=False) # UNCOMMENT WHEN DONE W EXPERIMENT
        else:
            self.use_factorised = False
            self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim) # UNCOMMENT WHEN DONE W EXPERIMENT

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

        # Validate layer_groups
        all_layers = set(range(args.n_layers))
        grouped_layers = set()
        for group in args.layer_groups:
            for layer in group:
                assert layer not in grouped_layers, "Layer assigned to multiple groups"
                grouped_layers.add(layer)
        assert grouped_layers == all_layers, "layer_groups must cover all layers exactly once"

        # Create weight sets for each group
        self.weight_sets = {}
        init_std = args.dim ** (-0.5)
        for group in args.layer_groups:
            group_key = tuple(group)
            attention_weights = {
                "wq": nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False),
                "wk": nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False),
                "wv": nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False),
                "wo": nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
            }
            ffn_weights = {
                "w1": nn.Linear(args.dim, args.ffn_dim, bias=False),
                "w2": nn.Linear(args.ffn_dim, args.dim, bias=False),
                "w3": nn.Linear(args.dim, args.ffn_dim, bias=False)
            }
            # Initialize weights
            for w in attention_weights.values():
                nn.init.trunc_normal_(w.weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)
            for w in ffn_weights.values():
                nn.init.trunc_normal_(w.weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)
            self.weight_sets[group_key] = {"attention": attention_weights, "ffn": ffn_weights}

        # Create layers with appropriate weight sets and LoRA ranks
        self.layers = nn.ModuleList()
        for i in range(args.n_layers):
            for group in args.layer_groups:
                if i in group:
                    group_key = tuple(group)
                    weight_set = self.weight_sets[group_key]
                    lora_rank = 0 if len(group) == 1 else args.lora_rank
                    self.layers.append(TransformerBlockWithSharedWeights(
                        args,
                        weight_set["attention"],
                        weight_set["ffn"],
                        lora_rank
                    ))
                    break

        # # Define shared weights for attention
        # self.shared_attention_weights = {
        #     "wq": nn.Parameter(torch.zeros(args.n_heads * args.head_dim, args.dim)),
        #     "wk": nn.Parameter(torch.zeros(args.n_kv_heads * args.head_dim, args.dim)),
        #     "wv": nn.Parameter(torch.zeros(args.n_kv_heads * args.head_dim, args.dim)),
        #     "wo": nn.Parameter(torch.zeros(args.dim, args.n_heads * args.head_dim))
        # }

        # # Define shared weights for feed-forward
        # self.shared_ffn_weights = {
        #     "w1": nn.Parameter(torch.zeros(args.ffn_dim, args.dim)),
        #     "w2": nn.Parameter(torch.zeros(args.dim, args.ffn_dim)),
        #     "w3": nn.Parameter(torch.zeros(args.ffn_dim, args.dim))
        # }

        # # Initialize shared weights
        # init_std = args.dim ** (-0.5)
        # for weight in self.shared_attention_weights.values():
        #     nn.init.trunc_normal_(weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)
        # for weight in self.shared_ffn_weights.values():
        #     nn.init.trunc_normal_(weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)

        # # Create layers with shared weights
        # self.layers = nn.ModuleList([
        #     TransformerBlockWithSharedWeights(
        #         args,
        #         self.shared_attention_weights,
        #         self.shared_ffn_weights,
        #         args.lora_rank
        #     )
        #     for _ in range(args.n_layers)
        # ])

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

        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl) # UNCOMMENT WHEN DONE W EXPERIMENT


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
        for x in self.weight_sets.values():
            for y in x.values():
                for z in y.values():
                    nn.init.trunc_normal_(z.weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)

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

    # Grouping by layers
    for i in range(model_args.n_layers):
        group_plan.append((f"layers.{i}", False))

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
