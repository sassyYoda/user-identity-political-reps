"""Ask the model what it believes the user's political leaning is (ticket 04).

    uv run python scripts/ask_political_leaning.py

Under every condition of both prompt tables — the Cen suite, the generated
controls, and the "none" baseline — the scaffolded question is posed with the
fixed probe question appended (polreps.blackbox.PROBE), and the model's
greedy answer is recorded. One JSON line per prompt, flushed as it lands, so
a killed run resumes where it stopped. The seed draws the matched-set
subsample only; decoding is greedy, so generation itself has no randomness
to seed. Scoring and the internal comparison are a separate stage
(scripts/run_blackbox.py).
"""

import argparse
import csv
from pathlib import Path

from polreps import blackbox
from polreps.caching import load_model
from polreps.config import ARTIFACTS, MODEL_REVISION, artifacts_dir
from polreps.pairs import condition_order
from polreps.runmeta import save_run_metadata


def read_rows(table_csv):
    with open(table_csv, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffold-table", default=str(ARTIFACTS / "prompt_table.csv"))
    parser.add_argument("--control-table", default=str(ARTIFACTS / "control_table.csv"))
    parser.add_argument("--out", default=str(artifacts_dir() / "blackbox_generations.jsonl"))
    parser.add_argument("--n-sets", type=int, default=60,
                        help="matched sets to ask under every condition")
    parser.add_argument("--seed", type=int, default=0, help="set-subsample seed")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on prompts, for smoke runs")
    args = parser.parse_args()

    scaffold_rows = read_rows(args.scaffold_table)
    rows = scaffold_rows + read_rows(args.control_table)
    main_hashes = {row["pre_prompt_q_hash"] for row in scaffold_rows}
    sampled = set(blackbox.sample_set_hashes(main_hashes, args.n_sets, args.seed))
    rows = [row for row in rows if row["pre_prompt_q_hash"] in sampled]
    rows.sort(key=lambda r: (r["pre_prompt_q_hash"], condition_order(r["condition"])))
    if args.limit is not None:
        rows = rows[: args.limit]
    n_conditions = len({row["condition"] for row in rows})
    print(f"{len(rows)} prompts ({len(sampled)} sets x {n_conditions} conditions)")

    done = blackbox.read_generations(args.out, repair=True)
    # the collection loop repeats this check, but the finished-run branch
    # below would otherwise accept a superset file as this run's output
    blackbox.assert_no_foreign_records(done, rows, args.out)
    if set(r["prompt_id"] for r in rows) - set(done):
        model = load_model(args.device)
        computed = blackbox.collect_answers(
            args.out, rows,
            lambda text: blackbox.generate_answer(model, text, args.max_new_tokens),
        )
    else:
        computed = 0
        print("all prompts already answered")

    # greedy decoding, so the revision is the reproducibility anchor and the
    # seed only names the set subsample; a no-op re-invocation must not
    # re-stamp the sidecar of the run that did the work
    meta_path = Path(args.out).with_name(Path(args.out).name + ".meta.json")
    if computed > 0 or not meta_path.exists():
        save_run_metadata(
            args.out, seed=args.seed,
            config={
                "scaffold_table": args.scaffold_table,
                "control_table": args.control_table,
                "n_sets": args.n_sets, "n_rows": len(rows), "limit": args.limit,
                "decoding": "greedy (do_sample=False)",
                "max_new_tokens": args.max_new_tokens,
                "probe": blackbox.PROBE,
            },
            model_revision=MODEL_REVISION,
        )
    print(f"collected {computed} new answers into {args.out}")


if __name__ == "__main__":
    main()
