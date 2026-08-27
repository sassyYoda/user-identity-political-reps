"""Generate the control-scaffold prompt table (ticket 03, MATS arc).

    uv run python scripts/build_control_table.py

Applies the self-generated control conditions — inert ("I was born in June."),
syntactic ("I am a person."), and the partisan paraphrases — to every base
question of the real prompt table, in the Cen prefix format. The output reuses
the main table's column contract and id recipe, so the caching stage walks it
unchanged and the gradient stage can join both tables at once. Deterministic
given the input table (no sampling), hence no seed.
"""

import argparse
import csv

from polreps.config import ARTIFACTS
from polreps.gradient import CONTROL_ROLES, control_table_rows
from polreps.runmeta import save_run_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-table", default=str(ARTIFACTS / "prompt_table.csv"))
    parser.add_argument("--out", default=str(ARTIFACTS / "control_table.csv"))
    args = parser.parse_args()

    with open(args.prompt_table, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        prompt_rows = list(reader)

    rows = control_table_rows(prompt_rows)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    save_run_metadata(
        args.out, seed=None,
        config={"prompt_table": str(args.prompt_table), "n_rows": len(rows)},
    )
    n_sets = len(rows) // len(CONTROL_ROLES)
    print(
        f"wrote {len(rows)} rows ({n_sets} sets x {len(CONTROL_ROLES)} control "
        f"conditions) to {args.out}"
    )


if __name__ == "__main__":
    main()
