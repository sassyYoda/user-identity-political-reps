"""Cache last-token residual-stream activations for every prompt in the table.

    uv run python scripts/cache_activations.py \
        --table artifacts/prompt_table.csv

One forward pass per prompt through the subject model (bf16, no generation,
pinned revision), fp32 on disk in the actcache layout, plus a .meta.json
sidecar. The default --out is the model-scoped activations/<model>/main, so a
new subject model never overwrites a previous model's cache. Interrupted runs
resume from the exact same command: finished prompts are skipped and the
completed cache is identical to an uninterrupted run's. --limit N caches only
the first N table rows (smoke runs).
"""

import argparse

from polreps.caching import run_caching
from polreps.config import MODEL_NAME, cache_dir, model_slug_of_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--out", default=str(cache_dir()))
    parser.add_argument("--device", default=None, help="default: mps if available")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # this script always runs the subject model, so writing into another
    # model's slot would mislabel every downstream artifact
    out_slug = model_slug_of_cache(args.out)
    if out_slug is not None and out_slug != MODEL_NAME.split("/")[-1]:
        parser.error(
            f"--out is scoped to {out_slug!r} but this script caches "
            f"{MODEL_NAME}; refusing to mislabel the cache"
        )

    computed = run_caching(args.table, args.out, device=args.device, limit=args.limit)

    print(f"done: {computed} prompts computed this run; cache at {args.out}")
    print(f"metadata sidecar at {args.out}.meta.json")


if __name__ == "__main__":
    main()
