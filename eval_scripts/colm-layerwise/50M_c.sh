#!/bin/bash
set -e

# Run: 50M_c
# Dump dir: /checkpoint/optim/sanaelotfi/weight_shared_llms/checkpoints/colm_runs/LAYERWISE_v3/50M_c

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/../../eval_harness.py"

if [ -z "$REPO_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Error: REPO_PATH and OUTPUT_PATH must be set"
    exit 1
fi

DUMP_DIR="/checkpoint/optim/sanaelotfi/weight_shared_llms/checkpoints/colm_runs/LAYERWISE_v3/50M_c"

CKPT_DIR=$(ls -d "${DUMP_DIR}/checkpoints/"*/ 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT_DIR" ]; then
    echo "Skipping 50M_c: no checkpoint found in ${DUMP_DIR}/checkpoints/"
    exit 0
fi

echo "============================================"
echo "Evaluating: 50M_c"
echo "Checkpoint: ${CKPT_DIR}"
echo "Output:     ${OUTPUT_PATH}/50M_c"
echo "============================================"

python "$EVAL_SCRIPT" \
    --repo "$REPO_PATH" \
    --checkpoint "$CKPT_DIR" \
    --tasks arc_easy,hellaswag,piqa,winogrande,rte,openbookqa,triviaqa,social_iqa,glue,unscramble \
    --num_fewshot 5 \
    --output "${OUTPUT_PATH}/50M_c"

echo "Done: 50M_c"
