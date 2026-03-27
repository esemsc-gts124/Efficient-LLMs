#!/usr/bin/env python3
"""Generate eval run scripts for the emb_proportion_FINAL embedding rank sweep."""

import os
import yaml

CONFIGS_DIR = os.path.join(
    os.path.dirname(__file__), "../../layer_sharing_final/sanae_configs/emb_proportion_FINAL"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "colm-emb-rank")


def make_script(name, dump_dir):
    return f"""#!/bin/bash
set -e

# Run: {name}
# Dump dir: {dump_dir}

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
EVAL_SCRIPT="${{SCRIPT_DIR}}/../../eval_harness.py"

if [ -z "$REPO_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Error: REPO_PATH and OUTPUT_PATH must be set"
    exit 1
fi

DUMP_DIR="{dump_dir}"

CKPT_DIR=$(ls -d "${{DUMP_DIR}}/checkpoints/"*/ 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT_DIR" ]; then
    echo "Skipping {name}: no checkpoint found in ${{DUMP_DIR}}/checkpoints/"
    exit 0
fi

echo "============================================"
echo "Evaluating: {name}"
echo "Checkpoint: ${{CKPT_DIR}}"
echo "Output:     ${{OUTPUT_PATH}}/{name}"
echo "============================================"

python "$EVAL_SCRIPT" \\
    --repo "$REPO_PATH" \\
    --checkpoint "$CKPT_DIR" \\
    --tasks arc_easy,hellaswag,piqa,winogrande,rte,openbookqa,triviaqa,siqa,glue,unscramble \\
    --num_fewshot 5 \\
    --output "${{OUTPUT_PATH}}/{name}"

echo "Done: {name}"
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    count = 0
    for size in ["100M", "250M", "500M", "1B"]:
        size_dir = os.path.join(CONFIGS_DIR, size)
        if not os.path.isdir(size_dir):
            continue
        for fname in sorted(os.listdir(size_dir)):
            if not fname.endswith(".yaml"):
                continue
            with open(os.path.join(size_dir, fname)) as f:
                config = yaml.safe_load(f)

            name = config["name"]
            dump_dir = config["dump_dir"]

            script = make_script(name, dump_dir)
            out_path = os.path.join(OUTPUT_DIR, f"{name}.sh")
            with open(out_path, "w") as f:
                f.write(script)
            os.chmod(out_path, 0o755)
            count += 1

    print(f"Generated {count} eval scripts in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
