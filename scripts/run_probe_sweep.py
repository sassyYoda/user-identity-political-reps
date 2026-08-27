"""Per-layer probe sweep over a cached activation run.

    uv run python scripts/run_probe_sweep.py \
        --cache activations/main \
        --labels data/prompt_table.csv \
        --out artifacts/probe_curve

Writes <out>.png (accuracy-vs-layer with chance and shuffled-label references),
<out>.json (per-layer numbers), and .meta.json sidecars for both. The labels
CSV needs prompt_id, condition, and base_q_template_hash columns. The binary
Democrat-vs-Republican panel is on by default; if the table spells those
conditions differently, the run fails loudly listing the observed vocabulary —
pass --binary with the actual names.
"""

import argparse

from polreps.sweep import run_sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="artifacts/probe_curve")
    parser.add_argument(
        "--binary", nargs=2, metavar=("COND", "COND"),
        default=("democrat", "republican"),
    )
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    curve = run_sweep(
        args.cache, args.labels, args.out,
        binary=tuple(args.binary) if args.binary else None,
        n_splits=args.splits, seed=args.seed,
    )

    peak = max(
        range(curve["n_layers"]), key=curve["multinomial"]["mean_accuracy"].__getitem__
    )
    print(
        f"multinomial peak: layer {peak} at "
        f"{curve['multinomial']['mean_accuracy'][peak]:.3f} "
        f"(chance {curve['multinomial']['chance']:.3f})"
    )
    print(f"wrote {args.out}.png / .json and metadata sidecars")


if __name__ == "__main__":
    main()
