#!/usr/bin/env python
"""
Standalone lm-eval harness script for Efficient-LLMs checkpoints.

Works with any Efficient-LLMs branch by dynamically importing model classes.
The --repo flag determines which model module gets imported (e.g., transformer.py
for attention-sharing, rrt.py for layer-sharing).

Usage:
    python eval_harness.py --repo /path/to/Efficient-LLMs --checkpoint /path/to/ckpt --tasks hellaswag --output ./results/
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance
except ImportError:
    # Provide a fallback for when lm_eval is not installed
    LM = object
    Instance = None
    print("Warning: lm_eval not installed. Install with: pip install lm_eval")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_efficient_llms_imports(repo_path: str):
    """Add Efficient-LLMs to sys.path for imports."""
    repo_path = Path(repo_path).resolve()
    if not repo_path.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    logger.info(f"Added {repo_path} to sys.path")


def load_model_for_eval(ckpt_dir: str):
    """
    Load model from checkpoint, consolidating if needed.

    IMPORTANT: This function must be called AFTER setup_efficient_llms_imports()
    so that the imports come from the correct codebase.

    The codebase's generate.py already has the correct model imports
    (e.g., from rrt.py in layer-sharing-final, from transformer.py in Efficient-LLMs).
    """
    from lingua.checkpoint import CONSOLIDATE_FOLDER, CONSOLIDATE_NAME, consolidate_checkpoints
    from apps.main.generate import load_consolidated_model_and_tokenizer

    ckpt_path = Path(ckpt_dir)

    # Check if already consolidated
    if (ckpt_path / CONSOLIDATE_NAME).exists() and (ckpt_path / "params.json").exists():
        consolidate_path = ckpt_path
        logger.info(f"Using existing consolidated checkpoint at {consolidate_path}")
    elif (ckpt_path / CONSOLIDATE_FOLDER / CONSOLIDATE_NAME).exists():
        consolidate_path = ckpt_path / CONSOLIDATE_FOLDER
        logger.info(f"Using existing consolidated checkpoint at {consolidate_path}")
    else:
        # Need to consolidate from distributed checkpoint
        logger.info(f"Consolidating distributed checkpoint at {ckpt_path}")
        consolidate_path = consolidate_checkpoints(str(ckpt_dir))
        logger.info(f"Consolidated to {consolidate_path}")

    # load_consolidated_model_and_tokenizer already has the correct model class imported
    # based on the codebase (see generate.py imports)
    model, tokenizer, config = load_consolidated_model_and_tokenizer(str(consolidate_path))
    return model, tokenizer, config


class EfficientLLMLM(LM):
    """
    Generic LM wrapper for Efficient-LLMs models.

    Implements the lm-eval harness LM interface with:
    - loglikelihood: Direct forward pass (matches training validation)
    - generate_until: Uses PackedCausalTransformerGenerator for autoregressive generation
    - loglikelihood_rolling: Direct forward pass with chunking for perplexity
    """

    def __init__(
        self,
        model,
        tokenizer,
        generator,
        max_length: int = 2048,
        batch_size: int = 8,
        device: str = "cuda",
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.generator = generator  # For generate_until
        self._max_length = max_length
        self._batch_size = batch_size
        self.device = device

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_id

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_length(self):
        return self._max_length

    @max_length.setter
    def max_length(self, value):
        self._max_length = value

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def tok_encode(self, string: str, add_special_tokens: bool = True) -> List[int]:
        """Encode a string to token IDs."""
        return self.tokenizer.encode(string, add_bos=add_special_tokens, add_eos=False)

    def tok_decode(self, tokens: List[int]) -> str:
        """Decode token IDs to a string."""
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """
        Score continuations using direct forward pass.

        For each (context, continuation) pair:
        1. Tokenize context with BOS, continuation without BOS
        2. Concatenate and run through model
        3. Extract log-probs only for continuation tokens
        4. Return (total_log_prob, is_greedy) tuple
        """
        results = []

        for request in tqdm(requests, desc="loglikelihood"):
            context, continuation = request.args

            # Tokenize separately to know the split point
            ctx_tokens = self.tokenizer.encode(context, add_bos=True, add_eos=False)
            cont_tokens = self.tokenizer.encode(continuation, add_bos=False, add_eos=False)

            # Handle empty context (just BOS)
            if len(ctx_tokens) == 0:
                ctx_tokens = [self.tokenizer.bos_id]

            full_tokens = ctx_tokens + cont_tokens

            # Truncate from left if needed (preserve continuation)
            if len(full_tokens) > self.max_length:
                excess = len(full_tokens) - self.max_length
                ctx_tokens = ctx_tokens[excess:]
                full_tokens = ctx_tokens + cont_tokens

            # Need at least 2 tokens for forward pass
            if len(full_tokens) < 2:
                results.append((0.0, True))
                continue

            # Forward pass: input is tokens[:-1], predict tokens[1:]
            input_ids = torch.tensor([full_tokens[:-1]], device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids)  # (1, seq_len, vocab_size)

            # Compute log-probs for continuation tokens only
            # Continuation starts at position len(ctx_tokens)-1 in logits
            # (because logits[i] predicts token[i+1])
            log_probs = F.log_softmax(logits[0].float(), dim=-1)

            cont_start = len(ctx_tokens) - 1
            cont_logprob = 0.0
            is_greedy = True

            for i, target_tok in enumerate(cont_tokens):
                pos = cont_start + i
                if pos >= log_probs.shape[0]:
                    break
                cont_logprob += log_probs[pos, target_tok].item()
                if logits[0, pos].argmax().item() != target_tok:
                    is_greedy = False

            results.append((cont_logprob, is_greedy))

        return results

    def generate_until(self, requests) -> List[str]:
        """
        Generate using PackedCausalTransformerGenerator for proper autoregressive generation.
        """
        results = []

        for request in tqdm(requests, desc="generate_until"):
            prompt, gen_args = request.args

            # Configure generator from request's gen_kwargs
            temperature = gen_args.get("temperature", 0.0)
            top_p = gen_args.get("top_p", None)
            top_k = gen_args.get("top_k", None)
            until = gen_args.get("until", [])
            max_gen_toks = gen_args.get("max_gen_toks", 256)

            # Store original settings
            orig_temperature = self.generator.temperature
            orig_top_p = self.generator.top_p
            orig_top_k = self.generator.top_k
            orig_until = self.generator.until
            orig_max_gen_len = self.generator.max_gen_len

            # Set new settings
            self.generator.temperature = temperature
            self.generator.top_p = top_p
            self.generator.top_k = top_k
            self.generator.until = until
            self.generator.max_gen_len = max_gen_toks

            # Generate
            generations, _, _ = self.generator.generate([prompt])

            # Restore original settings
            self.generator.temperature = orig_temperature
            self.generator.top_p = orig_top_p
            self.generator.top_k = orig_top_k
            self.generator.until = orig_until
            self.generator.max_gen_len = orig_max_gen_len

            # Strip stop sequences from output
            gen = generations[0]
            for stop in until:
                if stop in gen:
                    gen = gen[:gen.index(stop)]

            results.append(gen)

        return results

    def loglikelihood_rolling(self, requests) -> List[float]:
        """
        Compute perplexity using chunked forward passes.

        Each token is scored exactly once. Returns float (not tuple).
        """
        results = []

        for request in tqdm(requests, desc="loglikelihood_rolling"):
            text = request.args[0]
            tokens = self.tokenizer.encode(text, add_bos=True, add_eos=False)

            if len(tokens) < 2:
                results.append(0.0)
                continue

            total_logprob = 0.0

            # Process in chunks, each token scored exactly once
            # We need overlapping context for accurate predictions
            chunk_size = self.max_length

            for start in range(0, len(tokens) - 1, chunk_size):
                end = min(start + chunk_size, len(tokens))
                chunk = tokens[start:end]

                if len(chunk) < 2:
                    continue

                input_ids = torch.tensor([chunk[:-1]], device=self.device)

                with torch.no_grad():
                    logits = self.model(input_ids)

                log_probs = F.log_softmax(logits[0].float(), dim=-1)

                # Score each token in the chunk (except the first input token)
                for i, target_tok in enumerate(chunk[1:]):
                    total_logprob += log_probs[i, target_tok].item()

            results.append(total_logprob)  # Return float, not tuple

        return results


def run_eval(
    repo_path: str,
    ckpt_dir: str,
    tasks: List[str],
    output_dir: str,
    batch_size: int = 8,
    num_fewshot: int = 0,
    limit: Optional[int] = None,
    device: str = "cuda",
    max_gen_len: int = 256,
):
    """
    Run lm-eval harness on Efficient-LLMs checkpoint.

    Args:
        repo_path: Path to Efficient-LLMs repository
        ckpt_dir: Path to checkpoint directory (config in params.json drives architecture)
        tasks: List of lm-eval task names
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        num_fewshot: Number of few-shot examples
        limit: Limit samples per task (for testing)
        device: Device (cuda/cpu)
        max_gen_len: Max generation length
    """
    # Setup imports from Efficient-LLMs
    setup_efficient_llms_imports(repo_path)

    # Now we can import from Efficient-LLMs
    from apps.main.generate import (
        PackedCausalTransformerGenerator,
        PackedCausalTransformerGeneratorArgs,
    )

    # Load model - config in params.json determines architecture
    logger.info(f"Loading model from {ckpt_dir}")
    model, tokenizer, config = load_model_for_eval(ckpt_dir)
    logger.info("Model loaded successfully")

    # Get config values
    max_length = config.model.get("max_seqlen", 2048)
    dtype_str = config.distributed.get("model_dtype", "bf16")

    logger.info(f"Model config: max_seqlen={max_length}, dtype={dtype_str}")

    # Create generator for generate_until tasks
    gen_args = PackedCausalTransformerGeneratorArgs(
        temperature=0.0,
        max_gen_len=max_gen_len,
        max_tokens=max_length,
        dtype=dtype_str,
        device=device,
    )
    generator = PackedCausalTransformerGenerator(gen_args, model, tokenizer)

    # Create LM wrapper
    lm = EfficientLLMLM(
        model=model,
        tokenizer=tokenizer,
        generator=generator,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )

    # Run evaluation
    logger.info(f"Running evaluation on tasks: {tasks}")
    import lm_eval

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        # Only save the 'results' part which is JSON serializable
        json.dump(results["results"], f, indent=2)
    logger.info(f"Results saved to {results_file}")

    # Print results summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for task_name, task_results in results["results"].items():
        print(f"\n{task_name}:")
        for metric, value in task_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    print("=" * 60 + "\n")

    # Log to W&B if configured via environment variables
    wandb_project = os.environ.get("EVAL_WANDB_PROJECT")
    wandb_entity = os.environ.get("EVAL_WANDB_ENTITY")
    if wandb_project:
        try:
            import wandb

            run_name = os.environ.get("EVAL_WANDB_RUN_NAME") or output_path.name
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                config={
                    "checkpoint": ckpt_dir,
                    "tasks": tasks,
                    "num_fewshot": num_fewshot,
                    "batch_size": batch_size,
                    "repo_path": repo_path,
                },
            )

            flat_metrics = {}
            for task_name, task_results in results["results"].items():
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        flat_metrics[f"{task_name}/{metric}"] = value

            wandb.log(flat_metrics)
            wandb.finish()
            logger.info(f"Results logged to W&B project '{wandb_project}' as run '{run_name}'")
        except Exception as exc:
            logger.warning(f"Failed to log to W&B: {exc}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run lm-eval harness on Efficient-LLMs checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (limit to 10 samples)
  python eval_harness.py --repo /path/to/Efficient-LLMs --checkpoint /path/to/ckpt \\
      --tasks hellaswag --output ./results/ --limit 10

  # Full evaluation with multiple tasks
  python eval_harness.py --repo /path/to/Efficient-LLMs --checkpoint /path/to/ckpt \\
      --tasks hellaswag,piqa,arc_easy --output ./results/

  # With few-shot examples
  python eval_harness.py --repo /path/to/Efficient-LLMs --checkpoint /path/to/ckpt \\
      --tasks hellaswag,piqa --output ./results/ --num_fewshot 5

Using with different codebases:
  # For attention-sharing checkpoints (trained with Efficient-LLMs)
  python eval_harness.py --repo /path/to/Efficient-LLMs --checkpoint /path/to/ckpt --tasks hellaswag

  # For layer-sharing checkpoints (trained with layer-sharing-final)
  python eval_harness.py --repo /path/to/layer-sharing-final --checkpoint /path/to/ckpt --tasks hellaswag
        """
    )
    parser.add_argument(
        "--repo", "-r",
        required=True,
        help="Path to Efficient-LLMs repository (determines which model module is imported)"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--tasks", "-t",
        required=True,
        help="Comma-separated list of lm-eval tasks"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_fewshot", "-n",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit samples per task (for testing)"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--device", "-d",
        default="cuda",
        help="Device (default: cuda)"
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=256,
        help="Max generation length (default: 256)"
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]

    run_eval(
        repo_path=args.repo,
        ckpt_dir=args.checkpoint,
        tasks=tasks,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        device=args.device,
        max_gen_len=args.max_gen_len,
    )


if __name__ == "__main__":
    main()
