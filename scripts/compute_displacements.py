"""Difference-in-means displacement vectors per (scaffold condition, layer).

    uv run python scripts/compute_displacements.py \
        --cache activations/main \
        --labels artifacts/prompt_table.csv \
        --out artifacts/displacements

Writes <out>.npz ("conditions", "raw", "unit" — raw is the paired mean of
scaffolded-minus-"none" last-token residuals per condition and layer, unit is
its unit-norm copy) plus a <out>.json manifest and .meta.json sidecars. These
are milestone-2 inputs; nothing here analyzes them.
"""

import argparse

from polreps.displacement import run_displacements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="artifacts/displacements")
    args = parser.parse_args()

    manifest = run_displacements(args.cache, args.labels, args.out)

    supports = sorted(set(manifest["n_sets"].values()))
    print(
        f"{len(manifest['conditions'])} conditions x {manifest['n_layers']} "
        f"layers, matched-set support {supports[0]}"
        + (f"-{supports[-1]}" if len(supports) > 1 else "")
    )
    print(f"wrote {args.out}.npz / .json and metadata sidecars")


if __name__ == "__main__":
    main()
