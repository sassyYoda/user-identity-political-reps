"""Build the matched-set prompt table from the Cen et al. 2025 release.

    uv run python scripts/build_prompt_table.py [--max-sets 1000] [--seed 0]

Only the prompt bank is needed, so this pulls two reference JSONs (the
flattened question bank and the prompt-variation list) from the pinned
dataset revision into data/raw/ on first run and reads the local copies
afterwards — the 92k response CSVs in the release are never downloaded.
Writes artifacts/prompt_table.csv (+ .meta.json sidecar) and prints the
summary the ticket asks for. Downstream, the caching stage (ticket 03) walks
this table and keys the activation cache by its prompt_id column.
"""

import argparse
import csv
import json
from collections import Counter

from huggingface_hub import hf_hub_download

from polreps import pairs
from polreps.config import ARTIFACTS, DATA_RAW, DATASET_NAME, DATASET_REVISION
from polreps.runmeta import save_run_metadata

BANK_FILE = "reference_jsons/all_questions_flattened.json"
VARIANTS_FILE = "reference_jsons/prompt_variations.json"
TABLE_COLUMNS = [
    "prompt_id", "condition", "question", "pre_prompt_q_hash",
    "base_q_template_hash", "type", "category", "subcategory",
]


def fetch(filename):
    dataset_dir = DATA_RAW / DATASET_NAME.split("/")[1]
    local = dataset_dir / filename
    if not local.exists():
        hf_hub_download(
            DATASET_NAME,
            filename,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_dir=dataset_dir,
        )
        print(f"downloaded {filename} at revision {DATASET_REVISION[:12]}")
    else:
        print(f"using cached {local.relative_to(DATA_RAW.parent.parent)}")
    return local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sets", type=int, default=1000,
                    help="subsample target; the full bank has 573 sets, so the default keeps all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bank = json.loads(fetch(BANK_FILE).read_text())["questions_collection"]
    variants = json.loads(fetch(VARIANTS_FILE).read_text())["prompt_variants"]

    conditions = pairs.expected_conditions(variants)
    matched, report = pairs.build_matched_sets(bank, conditions)
    sampled = pairs.subsample_sets(matched, args.max_sets, args.seed)
    table = pairs.prompt_table_rows(sampled)

    ARTIFACTS.mkdir(exist_ok=True)
    table_path = ARTIFACTS / "prompt_table.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(table)
    save_run_metadata(
        table_path,
        seed=args.seed,
        config={
            "dataset": DATASET_NAME,
            "dataset_revision": DATASET_REVISION,
            "source_files": [BANK_FILE, VARIANTS_FILE],
            "max_sets": args.max_sets,
        },
    )

    print(f"\nbank rows in: {report['rows_in']}")
    print("dropped as baseline/control:")
    for (qtype, category), n in sorted(report["rows_dropped"].items()):
        print(f"  {n:6d}  {qtype}/{category}")
    print(f"matched sets anchored on \"none\": {report['n_sets']}"
          f" (excluded for missing baseline: {report['sets_missing_none']})")
    print(f"sets after subsample (seed {args.seed}): {len(sampled)}")
    print(f"conditions per set: {len(conditions)} (21 variations + none)")
    print("prompts per condition in the final table:")
    table_counts = Counter(r["condition"] for r in table)
    for condition in sorted(table_counts, key=lambda c: (c != "none", c)):
        print(f"  {table_counts[condition]:6d}  {condition}")
    print(f"\nwrote {len(table)} prompts to {table_path.relative_to(ARTIFACTS.parent)}"
          " (+ .meta.json)")


if __name__ == "__main__":
    main()
