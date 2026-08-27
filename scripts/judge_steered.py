"""Judge the steered generations for slant and coherence (ticket 05).

    uv run python scripts/judge_steered.py

The unsteered subject model reads each generation (question + finished text,
never the steering condition) and answers the two fixed probes of
polreps.steering: political slant with an explicit "no discernible stance"
option, and coherence. Raw judge answers land in a resumable JSONL; scoring
is a pure function applied by scripts/run_steering.py, so the rule can be
audited against the saved answers.
"""

import argparse
from pathlib import Path

from polreps import steering
from polreps.caching import load_model
from polreps.config import MODEL_REVISION, artifacts_dir
from polreps.runmeta import save_run_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations",
                        default=str(artifacts_dir() / "steering_generations.jsonl"))
    parser.add_argument("--out", default=str(artifacts_dir() / "steering_judgments.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on generations judged, for smoke runs")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    records = steering.read_generations(
        args.generations, fields=steering.GEN_FIELDS + ("answer",)
    )
    if args.limit is not None:
        records = dict(list(records.items())[: args.limit])
    print(f"{len(records)} generations to judge (two probes each)")

    done = steering.read_generations(args.out, repair=True,
                                     fields=steering.JUDGE_FIELDS)
    if set(records) - set(done):
        model = load_model(args.device)
        computed = steering.collect_judgments(
            args.out, records,
            lambda record: steering.judge_generation(
                model, record, args.max_new_tokens
            ),
        )
    else:
        computed = 0
        print("all generations already judged")

    meta_path = Path(args.out).with_name(Path(args.out).name + ".meta.json")
    if computed > 0 or not meta_path.exists():
        save_run_metadata(
            args.out, seed=None,
            config={
                "generations": args.generations, "limit": args.limit,
                "judge": "subject model, unsteered, greedy",
                "max_new_tokens": args.max_new_tokens,
                "slant_prompt": steering.SLANT_PROMPT,
                "coherence_prompt": steering.COHERENCE_PROMPT,
            },
            model_revision=MODEL_REVISION,
        )
    print(f"collected {computed} new judgments into {args.out}")


if __name__ == "__main__":
    main()
