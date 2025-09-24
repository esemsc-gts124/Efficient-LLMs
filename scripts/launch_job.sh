#!/bin/bash
#SBATCH --job-name=weight-shared-llms
#SBATCH --gpus-per-node=8
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=learn
#SBATCH --time=4320
#SBATCH --account=scaling_data_pruning
#SBATCH --qos=alignment_shared
#SBATCH --output=/home/sanaelotfi/Efficient-LLMs/logs/%x_%j.out
#SBATCH --error=/home/sanaelotfi/Efficient-LLMs/logs/%x_%j.err
# Usage: sbatch submit_job.sh path/to/config.yaml

PROJECT_DIR="/home/sanaelotfi/Efficient-LLMs"

echo "Starting job at $(date)"
start_time=$(date +%s)

# Activate your environment (adjust as needed)
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate lingua_250820

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

export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

torchrun --nproc-per-node=8 -m apps.main.train config=$CONFIG_FILE || { echo "Training script failed"; exit 1; }

echo "Job completed at $(date)"
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $((elapsed_time / 3600))h $(((elapsed_time % 3600) / 60))m $((elapsed_time % 60))s"