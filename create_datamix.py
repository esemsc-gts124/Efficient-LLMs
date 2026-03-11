"""
Mix parquet datasets into shuffled JSONL chunks for pretraining.

Usage:
    python mix_data.py --size 100 --input-dir /path/to/parquets --output-dir /path/to/output
    python mix_data.py --size 20  --input-dir /path/to/parquets --output-dir /path/to/output
    python mix_data.py --size 100 --test  # test run with 1/1000 scale
"""
import argparse
import json
import random
import pyarrow.parquet as pq
from pathlib import Path

DEFAULT_PARQUET_DIR = "/net/projects2/interp/pretrain_data/final_parquets"
VAL_TOKENS = 5_000_000  # ~5M tokens for validation set

# Token allocations for the 100B mix
MIX_100B = {
    # Web crawl sources: 55B evenly split (~18.33B each) -- also source val data
    "nt_cc_high":       {"tokens": 18_333_333_333, "web": True},
    "finepdfs_en":      {"tokens": 18_333_333_333, "web": True},
    "nt_cc_diverse_qa": {"tokens": 18_333_333_334, "web": True},
    # Other sources: 45B
    "nt_sft_math":      {"tokens":  5_000_000_000, "web": False},
    "nt_sft_code":      {"tokens":  5_000_000_000, "web": False},
    "nt_sft_general":   {"tokens":  5_000_000_000, "web": False},
    "nt_math":          {"tokens":  4_000_000_000, "web": False},
    "swallow_code":     {"tokens": 10_000_000_000, "web": False},
    "nt_code_synth":    {"tokens": 10_000_000_000, "web": False},
    "dolmino":          {"tokens":  1_000_000_000, "web": False},
    "pes2o":            {"tokens":  5_000_000_000, "web": False},
}


def row_to_jsonl(row: dict) -> str:
    """Convert a parquet row dict to a JSONL-formatted string."""
    metadata = row.get("metadata")
    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, dict):
        metadata = dict(metadata)
    metadata["token_count"] = row["token_count"]
    return json.dumps({"text": row["text"], "id": row["id"], "metadata": metadata})


def collect_rows(parquet_path: str, token_budget: int) -> list[dict]:
    """Read rows from a parquet file until token_budget is reached."""
    pf = pq.ParquetFile(parquet_path)
    rows = []
    tokens_so_far = 0

    for rg_idx in range(pf.metadata.num_row_groups):
        # Read only needed columns to save memory
        table = pf.read_row_group(rg_idx)
        col_names = table.column_names

        has_metadata = "metadata" in col_names
        ids = table.column("id")
        texts = table.column("text")
        token_counts = table.column("token_count")
        metadata_col = table.column("metadata") if has_metadata else None

        for i in range(len(table)):
            tc = token_counts[i].as_py()
            if tokens_so_far + tc > token_budget and tokens_so_far > 0:
                print(f"    Budget reached: {tokens_so_far:,} tokens, {len(rows):,} rows")
                return rows

            row = {
                "id": ids[i].as_py(),
                "text": texts[i].as_py(),
                "token_count": tc,
            }
            if has_metadata and metadata_col[i].as_py() is not None:
                row["metadata"] = metadata_col[i].as_py()

            rows.append(row)
            tokens_so_far += tc

    print(f"    Source exhausted: {tokens_so_far:,} tokens, {len(rows):,} rows")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Mix parquet datasets into JSONL")
    parser.add_argument("--size", type=int, required=True, help="Total tokens in billions (100 or 20)")
    parser.add_argument("--test", action="store_true", help="Test mode: 1/1000 scale")
    parser.add_argument("--prefix", type=str, default=None, help="Output filename prefix")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_PARQUET_DIR, help="Directory containing input parquet files")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    scale = args.size / 100.0
    if args.test:
        scale /= 1000.0
    if args.prefix is None:
        args.prefix = f"pretrain_{args.size}bt"

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scale factor: {scale}")
    print(f"Input dir: {input_dir}")
    print(f"Output prefix: {args.prefix}")
    print(f"Output dir: {output_dir}")
    print()

    # Collect rows from each source
    all_train_rows = []
    web_rows = []

    for name, config in MIX_100B.items():
        token_budget = int(config["tokens"] * scale)
        parquet_path = str(input_dir / f"{name}.parquet")

        # For web sources, add extra tokens for validation
        val_extra = int(VAL_TOKENS * scale / 3) if config["web"] else 0
        effective_budget = token_budget + val_extra

        print(f"Reading {name}: budget={token_budget:,} tokens" +
              (f" (+{val_extra:,} val)" if val_extra else ""))

        rows = collect_rows(parquet_path, effective_budget)

        if config["web"] and val_extra > 0:
            # Split off val rows from the end
            val_tokens_needed = val_extra
            val_split_idx = len(rows)
            val_acc = 0
            for j in range(len(rows) - 1, -1, -1):
                val_acc += rows[j]["token_count"]
                if val_acc >= val_tokens_needed:
                    val_split_idx = j
                    break
            web_rows.extend(rows[val_split_idx:])
            all_train_rows.extend(rows[:val_split_idx])
        else:
            all_train_rows.extend(rows)

    # Summary
    train_tokens = sum(r["token_count"] for r in all_train_rows)
    val_tokens = sum(r["token_count"] for r in web_rows)
    print(f"\nTrain: {len(all_train_rows):,} rows, {train_tokens:,} tokens")
    print(f"Val:   {len(web_rows):,} rows, {val_tokens:,} tokens")

    # Shuffle
    print("\nShuffling...")
    rng = random.Random(args.seed)
    rng.shuffle(all_train_rows)
    rng.shuffle(web_rows)

    # Write validation file
    val_path = output_dir / f"{args.prefix}.val.jsonl"
    print(f"Writing {val_path}")
    with open(val_path, "w") as f:
        for row in web_rows:
            f.write(row_to_jsonl(row) + "\n")

    # Split train into 8 chunks and write
    n = len(all_train_rows)
    chunk_size = n // 8
    for chunk_idx in range(8):
        start = chunk_idx * chunk_size
        end = start + chunk_size if chunk_idx < 7 else n
        chunk = all_train_rows[start:end]
        chunk_tokens = sum(r["token_count"] for r in chunk)

        chunk_path = output_dir / f"{args.prefix}.chunk.{chunk_idx:02d}.jsonl"
        print(f"Writing {chunk_path}: {len(chunk):,} rows, {chunk_tokens:,} tokens")
        with open(chunk_path, "w") as f:
            for row in chunk:
                f.write(row_to_jsonl(row) + "\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
