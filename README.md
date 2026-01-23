# LM-Evaluation-Harness for Efficient-LLMs

Standalone evaluation script that runs lm-evaluation-harness against Efficient-LLMs checkpoints.

## Installation

```bash
cd eval-final
uv sync
```

## Usage

### Basic usage

```bash

# Run evaluation
uv run eval_harness.py \
    --repo /path/to/Efficient-LLMs \
    --checkpoint /path/to/checkpoint \
    --tasks hellaswag \
    --output ./results/
```

### Quick test (limit samples)

```bash
uv run eval_harness.py \
    --repo /local/scratch/muchane_667850/Efficient-LLMs \
    --checkpoint /local/scratch/muchane_667850/checkpoints/final_runs/attn_sharing/500m_exps/460m_shared/checkpoints/0000004200 \
    --tasks hellaswag \
    --output ./results/ \
    --limit 10
```

### Multiple tasks with few-shot

```bash
uv run eval_harness.py \
    --repo /path/to/Efficient-LLMs \
    --checkpoint /path/to/checkpoint \
    --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande \
    --output ./results/ \
    --num_fewshot 5
```

### Using with different sharing strategies

The `--repo` flag determines which model module is imported:

- \*att-sharing-final\*\*: Imports from `apps/main/transformer.py` (attention-sharing support)
- **layer-sharing-final**: Imports from `apps/main/rrt.py` (layer-sharing support)

```bash
# For attention-sharing checkpoints
uv run eval_harness.py \
    --repo /path/to/att-sharing-final \
    --checkpoint /path/to/attn_sharing_checkpoint \
    --tasks hellaswag --output ./results/

# For layer-sharing checkpoints
uv run python3 eval_harness.py \
    --repo /path/to/layer-sharing-final \
    --checkpoint /path/to/layer_sharing_checkpoint \
    --tasks hellaswag --output ./results/
```

## CLI Options

| Option          | Short | Default  | Description                           |
| --------------- | ----- | -------- | ------------------------------------- |
| `--repo`        | `-r`  | Required | Path to Efficient-LLMs repository     |
| `--checkpoint`  | `-c`  | Required | Path to checkpoint directory          |
| `--tasks`       | `-t`  | Required | Comma-separated list of lm-eval tasks |
| `--output`      | `-o`  | Required | Output directory for results          |
| `--num_fewshot` | `-n`  | 0        | Number of few-shot examples           |
| `--limit`       | `-l`  | None     | Limit samples per task (for testing)  |
| `--batch_size`  | `-b`  | 8        | Batch size                            |
| `--device`      | `-d`  | cuda     | Device (cuda/cpu)                     |
| `--max_gen_len` |       | 256      | Max generation length                 |

## Programmatic Usage

```python
from eval_harness import run_eval

results = run_eval(
    repo_path="/path/to/Efficient-LLMs",
    ckpt_dir="/path/to/checkpoint",
    tasks=["hellaswag", "piqa"],
    output_dir="./results/",
    num_fewshot=0,
    limit=100,  # Optional: limit samples for testing
)
print(results["results"])
```

## Output

Results are saved to `{output_dir}/results.json`:

```json
{
    "hellaswag": {
        "alias": "hellaswag",
        "acc,none": 0.38,
        "acc_stderr,none": 0.0693,
        "acc_norm,none": 0.4,
        "acc_norm_stderr,none": 0.07
    }
}
```

## Architecture

The script implements the lm-eval `LM` interface with:

1. **`loglikelihood`**: Direct forward pass for scoring continuations
    - Matches training validation approach
    - Proper BOS handling: `add_bos=True` for context, `add_bos=False` for continuation

2. **`generate_until`**: Uses `PackedCausalTransformerGenerator` for autoregressive generation
    - Supports temperature, top_p, top_k sampling
    - Handles stop sequences

3. **`loglikelihood_rolling`**: Chunked forward passes for perplexity
    - Each token scored exactly once
    - Returns float (not tuple like the buggy original)
