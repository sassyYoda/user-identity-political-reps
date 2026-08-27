"""The transfer test, both ways, per layer — the milestone's kill-shot.

    uv run python scripts/run_transfer_test.py

The scaffold-derived Democrat-Republican axis (from the displacement
artifacts) scores content-labeled congressional statements it never saw
(speaker-grouped CV, only a threshold fit per fold), and the content-derived
ideology direction scores the scaffold cache's Democrat/Republican matched
pairs (zero-parameter paired comparison under the fixed sign convention).
Accuracy above chance means the *direction itself* carries ideology across
domains — the surface-token explanation of the saturated probe curve cannot
ride along, and the no-party-token robustness cut removes even the
shared-vocabulary route. Also computes the per-layer
alignment cosine between the scaffold displacement axis and the ideology
direction against a matched-dimension random-direction null (H2a), and picks
the working layer for downstream analyses by two-way transfer accuracy.
Writes <out>.json / <out>.png with .meta.json sidecars.
"""

import argparse

from polreps.config import ARTIFACTS, artifacts_dir, cache_dir, model_slug_of_cache
from polreps.transfer import run_transfer_test

DEM = "I am a Democrat. {}"
REP = "I am a Republican. {}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffold-cache", default=str(cache_dir()))
    parser.add_argument("--scaffold-table", default=str(ARTIFACTS / "prompt_table.csv"))
    parser.add_argument("--content-cache", default=str(cache_dir().with_name("content")))
    parser.add_argument("--content-table", default=str(ARTIFACTS / "content_corpus.csv"))
    parser.add_argument("--displacements", default=str(artifacts_dir() / "displacements.npz"))
    parser.add_argument("--ideology", default=str(artifacts_dir() / "ideology_direction.npz"))
    parser.add_argument("--out", default=None,
                        help="default: artifacts/<caches' model>/transfer_test")
    parser.add_argument("--dem", default=DEM)
    parser.add_argument("--rep", default=REP)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--null-draws", type=int, default=100_000)
    args = parser.parse_args()

    slugs = {model_slug_of_cache(c) for c in (args.scaffold_cache, args.content_cache)}
    if args.out is None:
        # the default output follows the caches' model; mixing models here
        # would compare directions from different residual spaces
        if len(slugs) != 1 or None in slugs:
            parser.error(
                f"caches resolve to model slugs {slugs}; pass --out (and make "
                "sure you mean to mix them)"
            )
        args.out = ARTIFACTS / slugs.pop() / "transfer_test"

    curve = run_transfer_test(
        args.scaffold_cache, args.scaffold_table,
        args.content_cache, args.content_table,
        args.displacements, args.ideology, args.out,
        dem=args.dem, rep=args.rep,
        n_splits=args.splits, seed=args.seed, null_draws=args.null_draws,
    )

    picked = curve["working_layer"]
    for key in ("scaffold_to_content", "scaffold_to_content_no_party_token",
                "content_to_scaffold_diff", "content_to_scaffold_ridge",
                "content_to_scaffold_clean_diff"):
        accs = curve[key]["mean_accuracy"]
        best = max(range(len(accs)), key=accs.__getitem__)
        print(
            f"{key}: chance {curve[key]['chance']:.3f}, layer-0 {accs[0]:.3f}, "
            f"peak {accs[best]:.3f} at layer {best}, "
            f"at working layer {accs[picked['layer']]:.3f}"
        )
    align = curve["alignment"]
    print(
        f"alignment cosine (diff): working layer {picked['alignment_cosine']:+.3f} "
        f"(p {picked['alignment_p']:.2e}), null sd {align['null_sd']:.4f}"
    )
    print(
        f"working layer {picked['layer']} by rule: {picked['rule']}"
    )
    print(f"wrote {args.out}.json / .png and metadata sidecars")


if __name__ == "__main__":
    main()
