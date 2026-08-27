"""Extract the per-layer reference ideology direction from the content cache.

    uv run python scripts/extract_ideology_direction.py \
        --cache activations/gemma-3-12b-it/content \
        --labels artifacts/content_corpus.csv

Writes <out>.npz (diff_raw/diff_unit from the party difference-in-means,
ridge_raw/ridge_unit from the DW-NOMINATE ridge probe) plus a <out>.json
manifest carrying the per-layer cosine between the two extractors, and
.meta.json sidecars. No layer is chosen here — that is the transfer stage's
job (ticket 02 amendment).
"""

import argparse

from polreps.config import artifacts_stem_for_cache
from polreps.ideology import run_ideology_directions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default=None,
                        help="default: artifacts/<cache's model>/ideology_direction")
    args = parser.parse_args()
    if args.out is None:
        # the default output follows the cache's model (see run_probe_sweep)
        args.out = artifacts_stem_for_cache(args.cache, "ideology_direction")
        if args.out is None:
            parser.error(f"--out is required when --cache is not under activations/<model>/: {args.cache}")

    manifest = run_ideology_directions(args.cache, args.labels, args.out)

    cosines = manifest["diff_ridge_cosine"]
    print(
        f"{manifest['n_layers']} layers x d_model {manifest['d_model']} from "
        f"{manifest['n_rows']} statements"
    )
    print(
        f"diff-in-means vs ridge cosine: min {min(cosines):+.3f}, "
        f"max {max(cosines):+.3f}"
    )
    top = sorted(set(manifest["ridge_alpha"]))
    print(f"ridge alphas chosen per layer (GCV over the fixed grid): {top}")
    print(f"wrote {args.out}.npz / .json and metadata sidecars")


if __name__ == "__main__":
    main()
