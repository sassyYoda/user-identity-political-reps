"""Steered generation for the dose-response experiment (ticket 05).

    uv run python scripts/generate_steered.py

Over the same seeded 60-set subsample as the black-box baseline (seed 0, so
the questions match ticket 04's), the model answers each base question
unscaffolded while alpha times a unit direction is added at the working
layer: the R-D displacement direction and a matched-norm random direction,
each at the four non-zero points of the calibrated symmetric grid, plus one
shared unsteered baseline. The off-target spot-check prompts run the same
grid. Decoding is greedy; the JSONL resumes after a kill. Judging is a
separate stage (scripts/judge_steered.py) so the steering hook can never be
live while the judge runs.
"""

import argparse
import csv
import json
from pathlib import Path

from polreps import blackbox, steering
from polreps.caching import load_model
from polreps.config import ARTIFACTS, MODEL_REVISION, artifacts_dir
from polreps.runmeta import save_run_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffold-table", default=str(ARTIFACTS / "prompt_table.csv"))
    parser.add_argument("--displacements",
                        default=str(artifacts_dir() / "displacements.npz"))
    parser.add_argument("--calibration",
                        default=str(artifacts_dir() / "steering_calibration.json"),
                        help="grid source; --alpha-max overrides it")
    parser.add_argument("--alpha-max", type=float, default=None)
    parser.add_argument("--layer", type=int, default=46,
                        help="working layer (ticket 02)")
    parser.add_argument("--n-sets", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0,
                        help="set-subsample seed (ticket 04's draw)")
    parser.add_argument("--random-direction-seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--out", default=str(artifacts_dir() / "steering_generations.jsonl"))
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on prompts, for smoke runs")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.alpha_max is not None:
        alpha_max = args.alpha_max
    else:
        calibration = json.loads(Path(args.calibration).read_text())
        if calibration.get("alpha_max") is None:
            raise SystemExit(f"{args.calibration} chose no alpha_max — recalibrate")
        if not calibration.get("cliff_located"):
            print("warning: calibration never located a cliff; grid max is the "
                  "largest coherent ladder point")
        if calibration.get("layer") != args.layer:
            raise SystemExit(
                f"calibration ran at layer {calibration.get('layer')}, "
                f"this run wants {args.layer}"
            )
        alpha_max = float(calibration["alpha_max"])
    grid = steering.symmetric_grid(alpha_max)
    print(f"alpha grid: {grid}")

    with open(args.scaffold_table, newline="") as f:
        none_rows = [r for r in csv.DictReader(f) if r["condition"] == "none"]
    hashes = blackbox.sample_set_hashes(
        [r["pre_prompt_q_hash"] for r in none_rows], args.n_sets, args.seed
    )
    question_of = {r["pre_prompt_q_hash"]: r["question"] for r in none_rows}
    questions = [(h, question_of[h]) for h in hashes]

    rows = steering.steering_rows(questions, grid, "political")
    rows += steering.steering_rows(
        sorted(steering.OFFTARGET_QUESTIONS.items()), grid, "offtarget"
    )
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"{len(rows)} prompts ({len(questions)} political questions + "
          f"{len(steering.OFFTARGET_QUESTIONS)} off-target, 9 conditions each)")

    model = load_model(args.device)
    diff = steering.verify_steering_site(model, args.layer, questions[0][1])
    print(f"hook site verified against resid_post (max |diff| {diff:.3g})")
    direction, norm = steering.steering_direction(args.displacements, args.layer)
    print(f"R-D displacement norm at layer {args.layer}: {norm:.1f} "
          f"(grid max is {alpha_max / norm:.1f}x it)")
    random_direction = steering.random_unit_direction(
        direction.shape[0], args.random_direction_seed
    )
    module = steering.resolve_layer_module(model.original_model, args.layer)
    tensor_of = {
        "displacement": steering.direction_tensor(model, direction),
        "random": steering.direction_tensor(model, random_direction),
        # the baseline row never adds anything; any tensor satisfies alpha=0
        "none": steering.direction_tensor(model, direction),
    }

    computed = steering.collect_steered(
        args.out, rows,
        lambda row: steering.generate_steered(
            model, module, tensor_of[row["direction"]], row["alpha"],
            row["question"], args.max_new_tokens,
        ),
    )

    # greedy decoding: the revision anchors reproducibility, the seeds name
    # the set subsample and the random direction; a no-op re-invocation must
    # not re-stamp the sidecar of the run that did the work
    meta_path = Path(args.out).with_name(Path(args.out).name + ".meta.json")
    if computed > 0 or not meta_path.exists():
        save_run_metadata(
            args.out, seed=args.seed,
            config={
                "scaffold_table": args.scaffold_table,
                "displacements": args.displacements,
                "calibration": args.calibration,
                "layer": args.layer, "alpha_grid": grid,
                "direction": "unit-norm (Republican minus Democrat) displacement",
                "direction_norm_raw": norm,
                "random_direction_seed": args.random_direction_seed,
                "n_sets": args.n_sets, "n_rows": len(rows), "limit": args.limit,
                "decoding": "greedy (do_sample=False)",
                "max_new_tokens": args.max_new_tokens,
                "hook_site_max_abs_diff": diff,
                "sign_prediction": steering.SIGN_PREDICTION,
            },
            model_revision=MODEL_REVISION,
        )
    print(f"collected {computed} new generations into {args.out}")


if __name__ == "__main__":
    main()
