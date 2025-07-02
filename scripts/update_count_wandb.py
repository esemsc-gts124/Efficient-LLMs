from math import floor
import wandb
from typing import Dict, Any, Optional
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
        if run.name != "70M_Frozen_attn":
            print(f"  ✓ Not run of interest")
            continue

        try:
            # Update the run config
            run.config["parameter_count"] = int(floor(run.config["parameter_count"] * .866))
            run.update()

            print(f"  ✅ Updated parameter_count")
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
