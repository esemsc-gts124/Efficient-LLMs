"""Parameterized parameter counter for layer sharing configs."""
import argparse
import json
from .rrt import LMTransformer, LMTransformerArgs
from .train import get_num_params


def count(dim, ffn_dim_multiplier, n_layers, n_heads, n_kv_heads,
          weight_tying=False, rank=0, lora_rank=0, layer_groups=None, vocab_size=128256):
    if layer_groups is None:
        layer_groups = [[i] for i in range(n_layers)]

    model_args = LMTransformerArgs(
        dim=dim,
        ffn_dim_multiplier=ffn_dim_multiplier,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        weight_tying=weight_tying,
        vocab_size=vocab_size,
        rank=rank,
        lora_rank=lora_rank,
        layer_groups=layer_groups,
    )
    model = LMTransformer(model_args)
    num_params = get_num_params(model)
    return num_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--ffn_dim_multiplier", type=float, default=1.5)
    parser.add_argument("--n_layers", type=int, required=True)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--n_kv_heads", type=int, default=8)
    parser.add_argument("--weight_tying", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--layer_groups", type=str, default=None,
                        help="JSON list of lists, e.g. '[[0,1],[2,3]]'")
    args = parser.parse_args()

    layer_groups = json.loads(args.layer_groups) if args.layer_groups else None

    num_params = count(
        dim=args.dim,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        weight_tying=args.weight_tying,
        rank=args.rank,
        lora_rank=args.lora_rank,
        layer_groups=layer_groups,
    )
    print(f"{num_params}")


if __name__ == "__main__":
    main()
