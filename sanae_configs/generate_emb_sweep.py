#!/usr/bin/env python3
"""Generate embedding rank sweep configs for emb_proportion_FINAL.

Two variants:
  - tied:   weight_tying=True, rank=X% of dim. Emb params = rank × (V + d)
  - untied: factorized_untied=True, weight_tying=False, rank=X% of dim.
            Emb params = 2 × rank × (V + d) (separate factorized input + output)

Parameter equivalence: untied at X% uses same emb params as tied at 2X%.
So untied sweeps 5%–50% (matching tied 10%–100%).

Architecture matches LAYERWISE_v3 baselines exactly.
"""

import os
import yaml

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "emb_proportion_FINAL")

BASELINES = {
    "100M": {
        "dim": 744, "ffn_dim": 2990, "n_layers": 12, "n_heads": 12, "n_kv_heads": 6,
        "steps": 76300, "batch_size": 16,
        "dump_every": 15260, "eval_every": 1536, "log_freq": 100,
        "grad_acc_steps": None,
    },
    "250M": {
        "dim": 1260, "ffn_dim": 3840, "n_layers": 12, "n_heads": 9, "n_kv_heads": None,
        "steps": 76300, "batch_size": 16,
        "dump_every": 15260, "eval_every": 1536, "log_freq": 100,
        "grad_acc_steps": None,
    },
    "500M": {
        "dim": 1536, "ffn_dim": 4728, "n_layers": 16, "n_heads": 12, "n_kv_heads": None,
        "steps": 76300, "batch_size": 16,
        "dump_every": 15260, "eval_every": 1536, "log_freq": 100,
        "grad_acc_steps": None,
    },
    "1B": {
        "dim": 1980, "ffn_dim": 5800, "n_layers": 20, "n_heads": 15, "n_kv_heads": None,
        "steps": 152500, "batch_size": 4,
        "dump_every": 15250, "eval_every": 3072, "log_freq": 100,
        "grad_acc_steps": 2,
    },
}


def make_layer_groups(n_layers):
    return [[i] for i in range(n_layers)]


def make_config(size, pct, baseline, mode):
    """mode: 'tied' or 'untied'"""
    dim = baseline["dim"]
    rank = max(1, round(dim * pct / 100))
    n_layers = baseline["n_layers"]

    name = f"{size}_emb_rank_{pct}pct_{mode}"
    dump_dir = (
        f"/checkpoint/optim/sanaelotfi/weight_shared_llms/checkpoints/"
        f"colm_runs/emb_proportion_FINAL/{size}/{name}"
    )

    model = {
        "dim": dim,
        "ffn_dim": baseline["ffn_dim"],
        "n_layers": n_layers,
        "n_heads": baseline["n_heads"],
        "rank": rank,
        "weight_tying": mode == "tied",
        "layer_groups": make_layer_groups(n_layers),
        "lora_rank": 0,
    }
    if mode == "untied":
        model["factorized_untied"] = True
    if baseline["n_kv_heads"] is not None:
        model["n_kv_heads"] = baseline["n_kv_heads"]

    config = {
        "dump_dir": dump_dir,
        "name": name,
        "steps": baseline["steps"],
        "probe_freq": None,
        "seed": 777,
        "optim": {
            "lr": 3e-4,
            "warmup": 2000,
            "lr_min_ratio": 1.0e-06,
            "clip": 10.0,
        },
        "distributed": {
            "fsdp_type": "no_shard",
            "compile": True,
            "model_dtype": "bf16",
            "matmul_allow_tf32": False,
            "selective_activation_checkpointing": False,
            "tp_size": 1,
            "dp_replicate": 8,
            "dp_shard": 1,
        },
        "model": model,
        "data": {
            "root_dir": "/checkpoint/optim/sanaelotfi/data",
            "sources": {"hq_data_20bt": 1.0},
            "batch_size": baseline["batch_size"],
            "prefetch_size": 32,
            "seq_len": 2048,
            "n_views": 2,
            "load_async": True,
            "tokenizer": {
                "path": "/checkpoint/optim/sanaelotfi/tokenizers/llama3_tokenizer/original/tokenizer.model",
                "name": "tiktoken",
            },
        },
        "profiling": {"run": True},
        "checkpoint": {
            "dump": {"every": baseline["dump_every"], "keep": 1},
            "eval": {"every": baseline["eval_every"], "keep": 1},
        },
        "logging": {
            "freq": baseline["log_freq"],
            "wandb": {
                "project": "colm-experiments",
                "entity": "weight-sharing",
                "name": name,
            },
        },
        "eval": {"validation": {"max_steps": None}},
    }

    if baseline["grad_acc_steps"] is not None:
        config["grad_acc_steps"] = baseline["grad_acc_steps"]

    return config


def main():
    total = 0
    for size, baseline in BASELINES.items():
        size_dir = os.path.join(OUTPUT_DIR, size)
        os.makedirs(size_dir, exist_ok=True)

        # Remove old configs
        for f in os.listdir(size_dir):
            if f.endswith(".yaml"):
                os.remove(os.path.join(size_dir, f))

        dim = baseline["dim"]

        # Tied: 5%, 10%, ..., 100%
        tied_count = 0
        for pct in range(5, 101, 5):
            rank = max(1, round(dim * pct / 100))
            config = make_config(size, pct, baseline, "tied")
            filepath = os.path.join(size_dir, f"rank_{pct}pct_tied.yaml")
            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            tied_count += 1
            total += 1

        # Untied: 5%, 10%, ..., 50% (untied_50% = tied_100% in param budget)
        untied_count = 0
        for pct in range(5, 51, 5):
            rank = max(1, round(dim * pct / 100))
            config = make_config(size, pct, baseline, "untied")
            filepath = os.path.join(size_dir, f"rank_{pct}pct_untied.yaml")
            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            untied_count += 1
            total += 1

        print(f"{size}: dim={dim}, {tied_count} tied + {untied_count} untied = {tied_count + untied_count} configs")
        print(f"  Tied: rank {max(1,round(dim*5/100))} (5%) to {dim} (100%)")
        print(f"  Untied: rank {max(1,round(dim*5/100))} (5%) to {max(1,round(dim*50/100))} (50%)")

    print(f"\nTotal: {total} configs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
