#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/../../eval_harness.py"
if [ -z "$REPO_PATH" ] || [ -z "$OUTPUT_PATH" ]; then echo "Error: REPO_PATH and OUTPUT_PATH must be set"; exit 1; fi
DUMP_DIR="/checkpoint/optim/sanaelotfi/weight_shared_llms/checkpoints/colm_runs/REWORKED_EMBEDDINGS/1B_24%_embeddings"
CKPT_DIR=$(ls -d "${DUMP_DIR}/checkpoints/"*/ 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT_DIR" ]; then echo "Skipping 1B_24%_embeddings: no checkpoint found"; exit 0; fi
echo "============================================"
echo "Evaluating: 1B_24%_embeddings"
echo "Checkpoint: ${CKPT_DIR}"
echo "Output:     ${OUTPUT_PATH}/1B_24%_embeddings"
echo "============================================"
python "$EVAL_SCRIPT" --repo "$REPO_PATH" --checkpoint "$CKPT_DIR" --tasks arc_easy,hellaswag,piqa,winogrande,rte,openbookqa,triviaqa,glue --num_fewshot 5 --output "${OUTPUT_PATH}/1B_24%_embeddings"
echo "Done: 1B_24%_embeddings"
