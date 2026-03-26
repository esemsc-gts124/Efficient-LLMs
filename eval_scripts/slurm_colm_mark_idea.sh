#!/bin/bash
#SBATCH --job-name=colm_eval_mi
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4320
#SBATCH --account=optim
#SBATCH --qos=h100_alignment_shared
#SBATCH --output=/home/sanaelotfi/eval/colm_logs/%x_%j.out
#SBATCH --error=/home/sanaelotfi/eval/colm_logs/%x_%j.err

# Usage: sbatch eval_scripts/slurm_colm_mark_idea.sh eval_scripts/colm-mark-idea/<run_name>.sh

echo "Starting eval job at $(date)"
start_time=$(date +%s)

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate lingua_251209

export WANDB_API_KEY=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
export TRITON_CACHE_DIR=/checkpoint/optim/sanaelotfi/triton_cache
export EVAL_WANDB_PROJECT="colm_all_evals"
export EVAL_WANDB_ENTITY="weight-sharing"
export HF_DATASETS_TRUST_REMOTE_CODE=1

export REPO_PATH="$HOME/layer_sharing"
export OUTPUT_PATH="/checkpoint/optim/sanaelotfi/weight_shared_llms/eval_results/colm/mark_idea"

EVAL_SCRIPT_DIR="/storage/home/sanaelotfi/eval/eval_scripts"
cd /storage/home/sanaelotfi/eval || { echo "Failed to cd to repo root"; exit 1; }

RUN_SCRIPT=$1
if [ -z "$RUN_SCRIPT" ]; then
    echo "Error: pass a run script, e.g.: sbatch eval_scripts/slurm_colm_mark_idea.sh eval_scripts/colm-mark-idea/250M_mark_baseline.sh"
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
