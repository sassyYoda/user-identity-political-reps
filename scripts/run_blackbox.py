"""Score the direct-question answers and compare them to the internal
projections (ticket 04).

    uv run python scripts/run_blackbox.py

Reads the generations JSONL of scripts/ask_political_leaning.py and the
gradient artifact of ticket 03, scores every answer on the shared
conservative-positive scale, and writes the per-condition verbal-vs-internal
comparison (<out>.json), the figure (<out>.png), and a seeded random draw of
raw example answers for the write-up (<out>_examples.md), each with a
.meta.json sidecar. The interesting cells are the disagreements: perm-p is
the internal deviation from ticket 03's common offset, "verbal leans" needs
at least --min-scored scored answers with a CI excluding 0.
"""

import argparse

from polreps.blackbox import run_blackbox
from polreps.config import artifacts_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations",
                        default=str(artifacts_dir() / "blackbox_generations.jsonl"))
    parser.add_argument("--gradient", default=str(artifacts_dir() / "gradient.json"))
    parser.add_argument("--out", default=str(artifacts_dir() / "blackbox"))
    parser.add_argument("--seed", type=int, default=0,
                        help="example-selection seed (and the correlation p's)")
    parser.add_argument("--min-scored", type=int, default=10)
    parser.add_argument("--examples-per-condition", type=int, default=2)
    args = parser.parse_args()

    result = run_blackbox(
        args.generations, args.gradient, args.out, seed=args.seed,
        min_scored=args.min_scored,
        examples_per_condition=args.examples_per_condition,
    )

    rows = result["conditions"]
    baseline = result["verbal_baseline"]
    print(
        f"{result['n_generations']} answers, {result['n_sets']} sets per "
        f"condition, layer {result['working_layer']}, common offset "
        f"{result['common_offset']:+.1f}, verbal baseline "
        f"{baseline['mean']:+.2f} ({baseline['n_scored']} scored)"
    )
    print("condition table (internal ranking order):")
    for condition in result["internal_ranking"] + ["none"]:
        row = rows.get(condition)
        if row is None:
            continue
        verbal = "  no scored answers        " if row["mean"] is None else (
            f"verbal {row['mean']:+5.2f}"
            + ("           " if row["verbal_delta"] is None
               else f" (d {row['verbal_delta']:+5.2f})")
            + f" ({row['n_scored']:2d} scored)"
        )
        internal = "" if condition == "none" else (
            f"  internal {row['internal_projection']:+8.1f}"
            f" (dev {row['internal_deviation']:+8.1f}, perm-p {row['perm_p']:.2g})"
            f"  [{row['reading']}]"
        )
        print(f"  {verbal}  abstain {row['abstain_rate']:.2f}{internal}  {condition}")
    corr = result["rank_correlation"]
    for label, by_layer, n in (
        ("all included", corr["by_layer"], corr["n"]),
        ("identity scaffolds and controls", corr["identity_by_layer"], corr["n_identity"]),
    ):
        for layer, r in by_layer.items():
            print(
                f"verbal-vs-internal rank correlation, {label}, layer {layer}: "
                f"rho {r['rho']:+.3f} (p {r['p']:.3g}, {r['method']}, n={n})"
            )
    if corr["excluded_below_min_scored"]:
        print(
            f"excluded from rho (< {result['min_scored']} scored answers): "
            f"{corr['excluded_below_min_scored']}"
        )
    print(f"wrote {args.out}.json / .png / _examples.md and metadata sidecars")


if __name__ == "__main__":
    main()
