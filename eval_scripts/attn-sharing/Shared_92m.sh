#!/bin/bash
set -e

# Run: Shared_92m (wandb ID: cn4vvfxv)
# Project: weight-sharing/icml-experiments
# Dump dir: /checkpoint/optim/sanaelotfi/weight_shared_llms/checkpoints/final_runs/attn_sharing/100m_exps/92m_shared

PYTHON="/home/sanaelotfi/miniconda3/envs/lingua_251209/bin/python3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}") " && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/../../eval_harness.py"

if [ -z "$REPO_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Error: REPO_PATH and OUTPUT_PATH environment variables must be set"
    echo "  REPO_PATH  = path to Efficient-LLMs repo (att-sharing-final branch)"
    echo "  OUTPUT_PATH = directory for eval results"
    exit 1
fi

DUMP_DIR="/checkpoint/optim/sanaelotfi/weight_shared_llms/checkpoints/final_runs/attn_sharing/100m_exps/92m_shared"

# Find the latest checkpoint in the dump directory
CKPT_DIR=$(ls -d "${DUMP_DIR}/checkpoints/"*/ 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT_DIR" ]; then
    echo "Error: No checkpoint subdirectories found in ${DUMP_DIR}/checkpoints/"
    exit 1
fi

echo "============================================"
echo "Evaluating: Shared_92m"
echo "Checkpoint: ${CKPT_DIR}"
echo "Output:     ${OUTPUT_PATH}/Shared_92m"
echo "============================================"

$PYTHON "$EVAL_SCRIPT" \
    --repo "$REPO_PATH" \
    --checkpoint "$CKPT_DIR" \
    --tasks hellaswag,piqa,arc_easy,siqa,triviaqa,glue,unscramble \
    --num_fewshot 5 \
    --output "${OUTPUT_PATH}/Shared_92m"

echo "Done: Shared_92m"
