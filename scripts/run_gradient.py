"""The projection-gradient headline over all scaffold conditions (ticket 03).

    uv run python scripts/run_gradient.py --alt-layer 39

Projects every condition's displacement — the Cen suite plus the generated
controls — onto the content-derived ideology direction at the transfer-chosen
working layer, ranks the spectrum, and reads it against the two ADR-0003
nulls (within-set label permutations, matched-norm random directions) plus
the validated-voter partisan-lean rank anchor. --layer defaults to the
working layer recorded in the model's transfer_test.json; pass --alt-layer
for robustness panels (39 is the recorded higher-alignment alternative on
the 38-45 plateau). Writes <out>.json / <out>.png with .meta.json sidecars.
"""

import argparse
import json
from pathlib import Path

from polreps.config import ARTIFACTS, DATA_RAW, artifacts_dir, cache_dir, model_slug_of_cache
from polreps.gradient import run_gradient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffold-cache", default=str(cache_dir()))
    parser.add_argument("--scaffold-table", default=str(ARTIFACTS / "prompt_table.csv"))
    parser.add_argument("--control-cache", default=str(cache_dir().with_name("controls")))
    parser.add_argument("--control-table", default=str(ARTIFACTS / "control_table.csv"))
    parser.add_argument("--ideology", default=str(artifacts_dir() / "ideology_direction.npz"))
    parser.add_argument("--displacements", default=str(artifacts_dir() / "displacements.npz"))
    parser.add_argument("--anchor", default=str(DATA_RAW / "pew" / "partisan_lean.json"))
    parser.add_argument("--out", default=None,
                        help="default: artifacts/<caches' model>/gradient")
    parser.add_argument("--layer", type=int, default=None,
                        help="default: the working layer in the model's transfer_test.json")
    parser.add_argument("--alt-layer", type=int, action="append", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permutation-draws", type=int, default=10_000)
    parser.add_argument("--direction-draws", type=int, default=100_000)
    args = parser.parse_args()

    slugs = {model_slug_of_cache(c) for c in (args.scaffold_cache, args.control_cache)}
    if args.out is None:
        if len(slugs) != 1 or None in slugs:
            parser.error(
                f"caches resolve to model slugs {slugs}; pass --out (and make "
                "sure you mean to mix them)"
            )
        args.out = ARTIFACTS / slugs.pop() / "gradient"
    if args.layer is None:
        transfer_json = Path(args.out).parent / "transfer_test.json"
        if not transfer_json.exists():
            parser.error(f"no {transfer_json} to read the working layer from; pass --layer")
        args.layer = json.loads(transfer_json.read_text())["working_layer"]["layer"]

    result = run_gradient(
        args.scaffold_cache, args.scaffold_table,
        args.control_cache, args.control_table,
        args.ideology, args.displacements, args.anchor, args.out,
        layer=args.layer, alt_layers=args.alt_layer or (),
        seed=args.seed, permutation_draws=args.permutation_draws,
        direction_draws=args.direction_draws,
    )

    at = result["layers"][str(args.layer)]
    lo, hi = at["permutation_null"]["band_99"]
    print(f"layer {args.layer}, {at['n_sets']} matched sets per condition")
    print(f"permutation null: 99% band [{lo:+.3f}, {hi:+.3f}], sd {at['permutation_null']['sd']:.3f}")
    print("spectrum (top to bottom):")
    for condition in at["ranking"]:
        print(
            f"  {at['mean_projection'][condition]:+8.3f} ± {at['ci95'][condition]:.3f}"
            f"  perm-p {at['permutation_null']['p_value'][condition]:.2g}"
            f"  rand-p {at['random_direction_null']['p_value'][condition]:.2g}"
            f"  [{result['roles'][condition]}]  {condition}"
        )
    anchor = at["anchor"]
    print(
        f"partisan-lean anchor: rho {anchor['rho']:+.3f} "
        f"(p {anchor['p']:.3g}, {anchor['method']}, n={anchor['n']}; "
        f"unanchored: {anchor['unanchored']})"
    )
    for pair in result["paraphrase_check"]["pairs"]:
        print(
            f"paraphrase: {pair['paraphrase']!r} rank {pair['paraphrase_rank']} "
            f"({pair['paraphrase_projection']:+.3f}) vs {pair['scaffold']!r} "
            f"rank {pair['scaffold_rank']} ({pair['scaffold_projection']:+.3f}), "
            f"displacement cosine {pair['cosine']:+.3f}"
        )
    if result["rank_stability"]:
        stability = result["rank_stability"]
        print(
            f"rank stability, layers {stability['layers']}: "
            f"rho {stability['rho']:+.3f} (p {stability['p']:.3g}); "
            f"identity scaffolds and controls only (n={stability['n_identity']}): "
            f"rho {stability['identity_rho']:+.3f} (p {stability['identity_p']:.3g})"
        )
    print(f"wrote {args.out}.json / .png and metadata sidecars")


if __name__ == "__main__":
    main()
