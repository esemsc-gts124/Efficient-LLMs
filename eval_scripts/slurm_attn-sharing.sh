#!/bin/bash
#SBATCH --job-name=icml_eval
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=learn
#SBATCH --time=4320
#SBATCH --account=scaling_data_pruning
#SBATCH --qos=alignment_shared
#SBATCH --output=/home/sanaelotfi/Efficient-LLMs/logs/%x_%j.out
#SBATCH --error=/home/sanaelotfi/Efficient-LLMs/logs/%x_%j.err

# Usage: sbatch slurm_attn-sharing.sh eval_scripts/attn-sharing/<run_name>.sh

echo "Starting eval job at $(date)"
start_time=$(date +%s)

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate lingua_251209

# Set these to the appropriate paths
export REPO_PATH="/path/to/Efficient-LLMs"       # att-sharing-final branch checkout
export OUTPUT_PATH="/path/to/eval_results/attn-sharing"   # where results will be saved

EVAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EVAL_SCRIPT_DIR/.." || { echo "Failed to cd to repo root"; exit 1; }

RUN_SCRIPT=$1
if [ -z "$RUN_SCRIPT" ]; then
    echo "Error: pass a run script as argument, e.g.: sbatch slurm_attn-sharing.sh eval_scripts/attn-sharing/Baseline_250m.sh"
    exit 1
fi

if [ ! -f "$RUN_SCRIPT" ]; then
    echo "Error: script not found: $RUN_SCRIPT"
    exit 1
fi

bash "$RUN_SCRIPT"

echo "Job completed at $(date)"
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $((elapsed_time / 3600))h $(((elapsed_time % 3600) / 60))m $((elapsed_time % 60))s"
