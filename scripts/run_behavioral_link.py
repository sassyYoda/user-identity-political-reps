"""Embed the base-question answers and correlate the two displacements.

    uv run python scripts/run_behavioral_link.py

The analysis stage of ticket 06: pairs every scaffolded answer with the same
matched set's "none" answer, embeds both with the pinned MiniLM, and
correlates the per-condition mean cosine distance with the internal
displacement read from the registered gradient artifact (primary statistic
and sign prediction in the polreps/behavioral.py docstring, committed before
generation). The embedder runs on CPU; nothing here needs the subject model.
"""

import argparse
from importlib.metadata import version

from polreps import behavioral
from polreps.config import artifacts_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations",
                        default=str(artifacts_dir() / "behavioral_generations.jsonl"))
    parser.add_argument("--gradient", default=str(artifacts_dir() / "gradient.json"))
    parser.add_argument("--out", default=str(artifacts_dir() / "behavioral_link"))
    parser.add_argument("--seed", type=int, default=0,
                        help="example draw and correlation permutations")
    parser.add_argument("--examples-per-condition", type=int, default=2)
    args = parser.parse_args()

    embedder_config = {
        "name": behavioral.EMBEDDER_NAME,
        "revision": behavioral.EMBEDDER_REVISION,
        "sentence_transformers": version("sentence-transformers"),
        "normalized": True,
    }
    result = behavioral.run_behavioral_link(
        args.generations, args.gradient, args.out,
        embed_fn=behavioral.load_embedder(),
        embedder_config=embedder_config,
        seed=args.seed,
        examples_per_condition=args.examples_per_condition,
    )

    correlation = result["rank_correlation"]
    for key in ("primary", "secondary"):
        stat = correlation[key]
        print(
            f"{key} ({stat['internal']}, layer {stat['layer']}): "
            f"rho {stat['rho']:+.3f}, p {stat['p']:.4g}, "
            f"n={result['n_conditions']}"
        )
    for name, stat in correlation["diagnostics"].items():
        if stat["rho"] is None:
            print(f"diagnostic {name}: {stat['note']}")
        else:
            print(f"diagnostic {name}: rho {stat['rho']:+.3f}, p {stat['p']:.4g}")
    reference = result["between_question_reference"]
    print(
        f"between-question reference distance (none answers): "
        f"{reference['mean']:.3f} over {reference['n_pairs']} pairs"
    )


if __name__ == "__main__":
    main()
