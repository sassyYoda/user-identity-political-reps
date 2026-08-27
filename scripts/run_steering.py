"""Score the judged steered generations into the dose-response artifact
(ticket 05).

    uv run python scripts/run_steering.py

Pure analysis — the model never loads. Reads the generations and judgments
JSONLs, scores every judge answer with the unit-tested rules, and writes the
dose-response table with the monotonicity statistic and its within-question
permutation p (<out>.json), the slant/no-stance/coherence figure
(<out>.png), and a seeded random draw of judged generations for owner
reading (<out>_examples.md), each with a .meta.json sidecar.
"""

import argparse
import json
from pathlib import Path

from polreps.config import artifacts_dir
from polreps.steering import DIRECTIONS, run_steering


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations",
                        default=str(artifacts_dir() / "steering_generations.jsonl"))
    parser.add_argument("--judgments",
                        default=str(artifacts_dir() / "steering_judgments.jsonl"))
    parser.add_argument("--out", default=str(artifacts_dir() / "steering"))
    parser.add_argument("--seed", type=int, default=0,
                        help="example-selection and permutation seed")
    parser.add_argument("--n-draws", type=int, default=10_000)
    parser.add_argument("--examples-per-cell", type=int, default=2)
    args = parser.parse_args()

    # the generation sidecar (grid, direction norm, seeds) rides along as
    # provenance so the analysis artifact is self-describing
    gen_meta_path = Path(args.generations).with_name(
        Path(args.generations).name + ".meta.json"
    )
    generation_meta = (
        json.loads(gen_meta_path.read_text()).get("config")
        if gen_meta_path.exists() else None
    )

    result = run_steering(
        args.generations, args.judgments, args.out, seed=args.seed,
        n_draws=args.n_draws, examples_per_cell=args.examples_per_cell,
        generation_meta=generation_meta,
    )

    print(
        f"{result['n_generations']} generations over "
        f"{result['n_political_questions']} political questions, "
        f"grid max {result['alpha_max']:g}"
    )
    for direction in DIRECTIONS:
        print(f"{direction}:")
        for alpha, cell in zip(result["alphas"], result["curves"][direction]):
            mean = ("     —" if cell["mean_slant"] is None
                    else f"{cell['mean_slant']:+.2f}")
            coherence = ("—" if cell["coherence_mean"] is None
                         else f"{cell['coherence_mean']:.2f}")
            print(
                f"  alpha {alpha:+10g}: slant {mean} "
                f"({cell['n_scored']:2d}/{cell['n']} scored), "
                f"no-stance {cell['no_stance_rate']:.2f}, "
                f"coherence {coherence}"
            )
        stat = result["monotonicity"][direction]
        if stat["rho"] is None:
            print(f"  monotonicity: {stat['note']} (n={stat['n']})")
        else:
            print(
                f"  monotonicity: rho {stat['rho']:+.3f} "
                f"(within-question permutation p {stat['p']:.2g}, "
                f"n={stat['n']} over {stat['n_questions']} questions)"
            )
        delta = result["extreme_delta"][direction]
        if delta["delta"] is not None:
            ci = "" if delta["ci95"] is None else f" ± {delta['ci95']:.2f}"
            print(
                f"  extremes: slant(+{delta['alpha_max']:g}) − "
                f"slant(−{delta['alpha_max']:g}) = {delta['delta']:+.2f}{ci} "
                f"({delta['n_pairs']} paired questions)"
            )
    print("off-target spot-check (answers with any judged stance):")
    for key, cell in result["offtarget"]["cells"].items():
        coherence = ("—" if cell["coherence_mean"] is None
                     else f"{cell['coherence_mean']:.2f}")
        print(
            f"  {key:>24}: {cell['n_scored']}/{cell['n']} stanced, "
            f"coherence {coherence}"
        )
    print(f"wrote {args.out}.json / .png / _examples.md and metadata sidecars")


if __name__ == "__main__":
    main()
