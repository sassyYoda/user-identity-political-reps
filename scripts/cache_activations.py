"""Cache last-token residual-stream activations for every prompt in the table.

    uv run python scripts/cache_activations.py \
        --table artifacts/prompt_table.csv \
        --out activations/main

One forward pass per prompt through Gemma-2-9B-IT (bf16, no generation, pinned
revision), fp32 on disk in the actcache layout, plus a .meta.json sidecar.
Interrupted runs resume from the exact same command: finished prompts are
skipped and the completed cache is identical to an uninterrupted run's.
--limit N caches only the first N table rows (smoke runs).
"""

import argparse

from polreps.caching import run_caching


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--out", default="activations/main")
    parser.add_argument("--device", default=None, help="default: mps if available")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    computed = run_caching(args.table, args.out, device=args.device, limit=args.limit)

    print(f"done: {computed} prompts computed this run; cache at {args.out}")
    print(f"metadata sidecar at {args.out}.meta.json")


if __name__ == "__main__":
    main()
