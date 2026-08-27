"""Locate the coherence cliff and choose the steering alpha grid (ticket 05).

    uv run python scripts/calibrate_steering.py

Kim et al. steered Llama inside |alpha| <= 30; Gemma-3-12B-IT's layer-46
residual norms are ~1.3e5, so the usable window has to be found empirically.
Over a small seeded draw of base questions, generation is steered along the
unit R-D displacement direction at each ladder magnitude (both signs, plus
the unsteered baseline), the unsteered model judges only coherence, and the
grid rule in polreps.steering.calibration_summary picks the symmetric
5-point grid below the first failure. Slant is never judged here — the
dose-response run stays unseen until the pre-registered analysis. Both
stages resume from their JSONLs after a kill.
"""

import argparse
import csv
import json

from polreps import blackbox, steering
from polreps.caching import load_model
from polreps.config import ARTIFACTS, MODEL_REVISION, artifacts_dir
from polreps.runmeta import save_run_metadata

LADDER = (2_500, 5_000, 10_000, 20_000, 40_000, 80_000, 160_000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffold-table", default=str(ARTIFACTS / "prompt_table.csv"))
    parser.add_argument("--displacements",
                        default=str(artifacts_dir() / "displacements.npz"))
    parser.add_argument("--layer", type=int, default=46,
                        help="working layer (ticket 02)")
    parser.add_argument("--ladder", default=",".join(str(m) for m in LADDER))
    parser.add_argument("--n-questions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1,
                        help="question-draw seed (distinct from the main run's 0)")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--coherent-threshold", type=float, default=0.8)
    parser.add_argument("--out", default=str(artifacts_dir() / "steering_calibration"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    magnitudes = [float(m) for m in args.ladder.split(",")]
    grid = sorted({0.0} | {s * m for m in magnitudes for s in (-1, 1)})

    with open(args.scaffold_table, newline="") as f:
        none_rows = [r for r in csv.DictReader(f) if r["condition"] == "none"]
    hashes = blackbox.sample_set_hashes(
        [r["pre_prompt_q_hash"] for r in none_rows], args.n_questions, args.seed
    )
    question_of = {r["pre_prompt_q_hash"]: r["question"] for r in none_rows}
    questions = [(h, question_of[h]) for h in hashes]
    rows = steering.steering_rows(
        questions, grid, "political", directions=("displacement",)
    )
    print(f"{len(rows)} calibration prompts ({len(questions)} questions x {len(grid)} alphas)")

    model = load_model(args.device)
    diff = steering.verify_steering_site(model, args.layer, questions[0][1])
    print(f"hook site verified against resid_post (max |diff| {diff:.3g})")
    direction, norm = steering.steering_direction(args.displacements, args.layer)
    print(f"R-D displacement norm at layer {args.layer}: {norm:.1f}")
    module = steering.resolve_layer_module(model.original_model, args.layer)
    direction_t = steering.direction_tensor(model, direction)

    gen_jsonl = args.out + "_generations.jsonl"
    steering.collect_steered(
        gen_jsonl, rows,
        lambda row: steering.generate_steered(
            model, module, direction_t, row["alpha"], row["question"],
            args.max_new_tokens,
        ),
    )
    records = steering.read_generations(
        gen_jsonl, fields=steering.GEN_FIELDS + ("answer",)
    )

    judge_jsonl = args.out + "_judgments.jsonl"
    steering.collect_judgments(
        judge_jsonl, records,
        lambda record: steering.judge_generation(model, record, coherence_only=True),
    )
    judgments = steering.read_generations(judge_jsonl, fields=steering.JUDGE_FIELDS)

    summary = steering.calibration_summary(
        steering.score_records(records, judgments),
        coherent_threshold=args.coherent_threshold,
    )
    summary["layer"] = args.layer
    summary["direction_norm"] = norm
    out_json = args.out + ".json"
    with open(out_json, "w") as f:
        f.write(json.dumps(summary, indent=2) + "\n")

    config = {
        "scaffold_table": args.scaffold_table, "displacements": args.displacements,
        "layer": args.layer, "ladder": magnitudes,
        "n_questions": args.n_questions, "max_new_tokens": args.max_new_tokens,
        "coherent_threshold": args.coherent_threshold,
        "decoding": "greedy (do_sample=False)",
        "hook_site_max_abs_diff": diff,
    }
    for artifact in (gen_jsonl, judge_jsonl, out_json):
        save_run_metadata(artifact, seed=args.seed, config=config,
                          model_revision=MODEL_REVISION)

    for alpha_key, rates in summary["by_alpha"].items():
        print(
            f"  alpha {alpha_key:>8}: coherent {rates['coherent_rate']:.2f}, "
            f"gibberish {rates['gibberish_rate']:.2f}, "
            f"mean {rates['coherence_mean']:.2f}"
        )
    if summary["grid"] is None:
        print(
            f"cliff at |alpha| {summary['cliff']:g} — even the smallest ladder "
            "magnitude breaks coherence; rerun with a lower --ladder"
        )
    elif summary["cliff_located"]:
        print(f"cliff at |alpha| {summary['cliff']:g}; grid {summary['grid']}")
    else:
        print(
            f"no cliff inside the ladder (largest magnitude "
            f"{summary['alpha_max']:g} still coherent) — extend --ladder "
            "before trusting the grid"
        )


if __name__ == "__main__":
    main()
