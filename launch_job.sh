#!/bin/bash
#SBATCH --job-name=icml_weight_sharing
#SBATCH --gpus-per-node=8
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4320
#SBATCH --account=optim
#SBATCH --qos=scaling_data_pruning_high
#SBATCH --qos=h100_alignment_shared
#SBATCH --output=/home/sanaelotfi/Efficient-LLMs/icml_logs/%x_%j.out
#SBATCH --error=/home/sanaelotfi/Efficient-LLMs/icml_logs/%x_%j.err
# Usage: sbatch submit_job.sh path/to/config.yaml

PROJECT_DIR="/home/sanaelotfi/Efficient-LLMs"

export WANDB_API_KEY=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
export TRITON_CACHE_DIR=/checkpoint/optim/sanaelotfi/triton_cache


echo "Starting job at $(date)"
start_time=$(date +%s)

# Activate your environment (adjust as needed)
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate lingua_251209

cd "$PROJECT_DIR" || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

CONFIG_FILE=$1
TRAIN_SCRIPT="apps/main/train.py"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found at $TRAIN_SCRIPT"
    exit 1
fi


torchrun --nproc-per-node=8 -m apps.main.train config=$CONFIG_FILE || { echo "Training script failed"; exit 1; }

echo "Job completed at $(date)"
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $((elapsed_time / 3600))h $(((elapsed_time % 3600) / 60))m $((elapsed_time % 60))s"