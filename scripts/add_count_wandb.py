import wandb
from typing import Dict, Any, Optional
from param_counter import calculate_transformer_params

def update_wandb_parameter_counts(project_name: str = "40m_baseline"):
    """
    Update wandb runs with missing parameter_count summary metrics.
    """
    # Initialize wandb API
    api = wandb.Api()

    # Get all runs from the project
    runs = api.runs(f"{api.default_entity}/{project_name}")

    print(f"Found {len(runs)} runs in project '{project_name}'")

    updated_count = 0

    for run in runs:
        print(f"\nProcessing run: {run.name} ({run.id})")

        # Check if parameter_count already exists in summary
        if "parameter_count" in run.summary and "parameter_count" in run.config:
            print(f"  ✓ Parameter count already exists: {run.summary['parameter_count']}")
            continue

        # Get the model config from run config
        if "model" not in run.config:
            print("  ⚠ No 'model' key found in config, skipping")
            continue

        model_config = run.config["model"]

        # Create the config dictionary for calculate_transformer_params
        param_config = {
            # BaseTransformerArgs
            "dim": model_config["dim"],
            "n_layers": model_config["n_layers"],
            "head_dim": model_config["head_dim"],
            "n_heads": model_config["n_heads"],
            "n_kv_heads": model_config["n_kv_heads"],
            "ffn_dim_multiplier": model_config["ffn_dim_multiplier"],
            "multiple_of": model_config["multiple_of"],
            "norm_eps": model_config["norm_eps"],
            "rope_theta": model_config["rope_theta"],
            "init_base_std": model_config["init_base_std"],
            "init_std_factor": model_config["init_std_factor"],
            "max_seqlen": model_config["max_seqlen"],

            # Attention sharing configuration
            "attn_sharing": model_config.get("attn_sharing", False),
            "qkv_sharing": model_config.get("qkv_sharing", None),
            "head_sharing": model_config.get("head_sharing", True),
            "grouping": model_config.get("grouping", 1),
            "rank": model_config.get("rank", 8),
            "two_step": model_config.get("two_step", False),

            # LMTransformerArgs
            "seed": model_config["seed"],
            "vocab_size": model_config["vocab_size"],
            "weight_tying": model_config["weight_tying"],
            "sliding_window": model_config["sliding_window"],
        }

        try:
            # Calculate parameter count
            param_counts = calculate_transformer_params(**param_config)
            total_params = param_counts["total"]

            print(f"  📊 Calculated parameter count: {total_params:,}")

            # Update the run summary
            run.summary["parameter_count"] = total_params
            run.summary.update()
            run.config["parameter_count"] = total_params
            run.config.update()

            print(f"  ✅ Updated parameter_count to {total_params:,}")
            updated_count += 1

        except Exception as e:
            print(f"  ❌ Error calculating parameters for run {run.name}: {str(e)}")
            continue

    print(f"\n🎉 Successfully updated {updated_count} runs with parameter counts")

def main():
    """
    Main function to run the parameter count update.
    """
    project_name = "40m_baseline"

    print(f"Starting parameter count update for project: {project_name}")
    print("=" * 60)

    try:
        update_wandb_parameter_counts(project_name)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure you're logged into wandb and have access to the project.")

if __name__ == "__main__":
    main()
