#!/bin/bash
set -e

# Run: cycle_b
# Dump dir: /checkpoint/optim/sanaelotfi/weight_shared_llms/checkpoints/colm_runs/LAYERWISE_v3/cycle_b

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/../../eval_harness.py"

if [ -z "$REPO_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Error: REPO_PATH and OUTPUT_PATH must be set"
    exit 1
fi

DUMP_DIR="/checkpoint/optim/sanaelotfi/weight_shared_llms/checkpoints/colm_runs/LAYERWISE_v3/cycle_b"

CKPT_DIR=$(ls -d "${DUMP_DIR}/checkpoints/"*/ 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT_DIR" ]; then
    echo "Skipping cycle_b: no checkpoint found in ${DUMP_DIR}/checkpoints/"
    exit 0
fi

echo "============================================"
echo "Evaluating: cycle_b"
echo "Checkpoint: ${CKPT_DIR}"
echo "Output:     ${OUTPUT_PATH}/cycle_b"
echo "============================================"

python "$EVAL_SCRIPT" \
    --repo "$REPO_PATH" \
    --checkpoint "$CKPT_DIR" \
    --tasks arc_easy,hellaswag,piqa,winogrande,rte,openbookqa,triviaqa,social_iqa,glue,unscramble \
    --num_fewshot 5 \
    --output "${OUTPUT_PATH}/cycle_b"

echo "Done: cycle_b"
