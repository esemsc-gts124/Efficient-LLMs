#!/bin/bash
set -e

# Run: 250M_bulge (wandb ID: 0gc6zcsi)
# Project: weight-sharing/layerwise
# Dump dir: /checkpoint/optim/sanaelotfi/weight_shared_llms/checkpoints/final_runs/LAYERWISE_v3/250M/250M_bulge

PYTHON="/home/sanaelotfi/miniconda3/envs/lingua_251209/bin/python3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}") " && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/../../eval_harness.py"

if [ -z "$REPO_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Error: REPO_PATH and OUTPUT_PATH environment variables must be set"
    echo "  REPO_PATH  = path to Efficient-LLMs repo (layer-sharing-final branch)"
    echo "  OUTPUT_PATH = directory for eval results"
    exit 1
fi

DUMP_DIR="/checkpoint/optim/sanaelotfi/weight_shared_llms/checkpoints/final_runs/LAYERWISE_v3/250M/250M_bulge"

# Find the latest checkpoint in the dump directory
CKPT_DIR=$(ls -d "${DUMP_DIR}/checkpoints/"*/ 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT_DIR" ]; then
    echo "Error: No checkpoint subdirectories found in ${DUMP_DIR}/checkpoints/"
    exit 1
fi

echo "============================================"
echo "Evaluating: 250M_bulge"
echo "Checkpoint: ${CKPT_DIR}"
echo "Output:     ${OUTPUT_PATH}/250M_bulge"
echo "============================================"

$PYTHON "$EVAL_SCRIPT" \
    --repo "$REPO_PATH" \
    --checkpoint "$CKPT_DIR" \
    --tasks hellaswag,piqa,arc_easy,siqa,triviaqa,glue,unscramble \
    --num_fewshot 5 \
    --output "${OUTPUT_PATH}/250M_bulge"

echo "Done: 250M_bulge"
