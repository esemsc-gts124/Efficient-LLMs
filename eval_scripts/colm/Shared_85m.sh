#!/bin/bash
set -e

# Run: Shared_85m
# Dump dir: /checkpoint/optim/sanaelotfi/weight_shared_llms/checkpoints/colm_runs/attn_sharing/100m_exps/85m_shared

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/../../eval_harness.py"

if [ -z "$REPO_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Error: REPO_PATH and OUTPUT_PATH must be set"
    exit 1
fi

DUMP_DIR="/checkpoint/optim/sanaelotfi/weight_shared_llms/checkpoints/colm_runs/attn_sharing/100m_exps/85m_shared"

CKPT_DIR=$(ls -d "${DUMP_DIR}/checkpoints/"*/ 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT_DIR" ]; then
    echo "Skipping Shared_85m: no checkpoint found in ${DUMP_DIR}/checkpoints/"
    exit 0
fi

echo "============================================"
echo "Evaluating: Shared_85m"
echo "Checkpoint: ${CKPT_DIR}"
echo "Output:     ${OUTPUT_PATH}/Shared_85m"
echo "============================================"

python "$EVAL_SCRIPT" \
    --repo "$REPO_PATH" \
    --checkpoint "$CKPT_DIR" \
    --tasks arc_easy,hellaswag,piqa,winogrande,rte,openbookqa,triviaqa,siqa,glue,unscramble \
    --num_fewshot 5 \
    --output "${OUTPUT_PATH}/Shared_85m"

echo "Done: Shared_85m"
