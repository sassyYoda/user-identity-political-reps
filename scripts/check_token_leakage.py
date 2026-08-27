"""Post-hoc leakage diagnostic for the real probe curve (ticket 05).

    uv run python scripts/check_token_leakage.py \
        --cache activations/main \
        --labels artifacts/prompt_table.csv \
        --out artifacts/leakage_check

The real curve came out far above chance from layer 0, the pre-registered
signature of token-identity leakage rather than a computed representation. The
boring explanation makes a positional prediction we can test from the same
cache: suffix-form conditions ("{} Please ...") put their distinctive tokens
directly before the measured last position, prefix-form conditions ("I am ...
{}") put them a full question away, so surface-token reading should separate
the suffix subset better than the prefix subset in the earliest layers. This
is a post-hoc analysis and its numbers are labeled that way in NUMBERS.md.
"""

import argparse
import json
from pathlib import Path

from polreps import actcache
from polreps.runmeta import save_run_metadata
from polreps.sweep import join_prompt_table, sweep_variant


def split_by_form(conditions):
    prefix = sorted(c for c in conditions if c.endswith("{}"))
    suffix = sorted(c for c in conditions if c.startswith("{}"))
    leftover = set(conditions) - set(prefix) - set(suffix) - {"none"}
    if leftover:
        raise ValueError(f"conditions with neither form: {sorted(leftover)}")
    return prefix, suffix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="artifacts/leakage_check")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 8])
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    acts, prompt_ids = actcache.load_cache(Path(args.cache))
    labels, groups = join_prompt_table(args.labels, prompt_ids)
    prefix, suffix = split_by_form(set(labels))

    picked = acts[args.layers]
    result = {"layers": args.layers}
    for name, subset in (("prefix", prefix), ("suffix", suffix)):
        v = sweep_variant(
            picked, labels, groups, args.splits, args.seed, conditions=subset
        )
        result[name] = v
        accs = ", ".join(
            f"L{layer}={acc:.3f}"
            for layer, acc in zip(args.layers, v["mean_accuracy"])
        )
        print(f"{name} ({len(subset)}-way, chance {v['chance']:.3f}): {accs}")

    out_json = Path(args.out).with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    save_run_metadata(
        out_json, seed=args.seed,
        config={
            "cache_dir": args.cache, "labels_csv": args.labels,
            "layers": args.layers, "n_splits": args.splits,
            "registered": "post-hoc",
        },
    )
    print(f"wrote {out_json} and metadata sidecar")


if __name__ == "__main__":
    main()
