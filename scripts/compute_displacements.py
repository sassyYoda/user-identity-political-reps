"""Difference-in-means displacement vectors per (scaffold condition, layer).

    uv run python scripts/compute_displacements.py \
        --cache activations/gemma-3-12b-it/main \
        --labels artifacts/prompt_table.csv

Writes <out>.npz ("conditions", "raw", "unit" — raw is the paired mean of
scaffolded-minus-"none" last-token residuals per condition and layer, unit is
its unit-norm copy) plus a <out>.json manifest and .meta.json sidecars. These
are milestone-2 inputs; nothing here analyzes them.
"""

import argparse

from polreps.config import artifacts_stem_for_cache
from polreps.displacement import run_displacements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default=None, help="default: artifacts/<cache's model>/displacements")
    args = parser.parse_args()
    if args.out is None:
        # the default output follows the cache's model (see run_probe_sweep)
        args.out = artifacts_stem_for_cache(args.cache, "displacements")
        if args.out is None:
            parser.error(f"--out is required when --cache is not under activations/<model>/: {args.cache}")

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
