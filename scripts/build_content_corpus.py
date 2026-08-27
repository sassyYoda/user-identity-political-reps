"""Build the content corpus: congressional statements with DW-NOMINATE labels.

    uv run python scripts/build_content_corpus.py [--n-per-party 1000] [--seed 0]

Reads the pinned hein-daily congress files and Voteview members table (see
scripts/fetch_content_data.py), labels each speech with its speaker's party
and nominate_dim1 via a seat + lastname join to Voteview, filters to
substantive-length speeches containing no scaffold phrase from the real
prompt table, and draws a seeded party-balanced sample capped per speaker.
Writes artifacts/content_corpus.csv (+ .meta.json) with the same
prompt_id/question column contract as the prompt table, so
scripts/cache_activations.py walks it unchanged. The final table is
re-verified prefix-free before writing — that check failing is a bug, not a
data property, since filtering already removed offenders.
"""

import argparse
import csv
from collections import Counter

from polreps import content
from polreps.config import ARTIFACTS, DATA_RAW
from polreps.runmeta import save_run_metadata

TABLE_COLUMNS = [
    "prompt_id", "question", "party", "nominate_dim1", "icpsr", "speech_id",
    "word_count", "mentions_party",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--congress", type=int, default=114)
    parser.add_argument("--n-per-party", type=int, default=1000)
    parser.add_argument("--max-per-speaker", type=int, default=10,
                        help="no member's rhetoric should dominate the direction")
    parser.add_argument("--min-words", type=int, default=100,
                        help="floor drops procedural one-liners")
    parser.add_argument("--max-words", type=int, default=400,
                        help="ceiling keeps forward passes short")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-table", default=str(ARTIFACTS / "prompt_table.csv"),
                        help="source of the scaffold phrases to ban")
    args = parser.parse_args()

    hein = DATA_RAW / "hein-daily"
    cong = f"{args.congress:03d}"
    # hein files are OCR-derived latin-1-ish text; a rare undecodable byte is
    # not worth failing the corpus over
    speakermap = content.read_pipe_table(
        (hein / f"{cong}_SpeakerMap.txt").read_text(errors="replace"),
        ("speakerid", "speech_id", "lastname", "chamber", "state", "district",
         "party", "nonvoting"),
    )
    descr = content.read_pipe_table(
        (hein / f"descr_{cong}.txt").read_text(errors="replace"),
        ("speech_id", "word_count"),
    )
    speeches = content.read_speeches(
        (hein / f"speeches_{cong}.txt").read_text(errors="replace")
    )
    with open(DATA_RAW / "voteview" / "HSall_members.csv", newline="") as f:
        seats = content.voteview_members(csv.DictReader(f), args.congress)
    with open(args.prompt_table, newline="") as f:
        conditions = sorted({row["condition"] for row in csv.DictReader(f)})

    labels, join_report = content.join_speaker_labels(speakermap, seats)
    phrases = content.scaffold_phrases(conditions)
    word_counts = {row["speech_id"]: int(row["word_count"]) for row in descr}
    rows, filter_report = content.eligible_rows(
        speeches, word_counts, labels, phrases,
        min_words=args.min_words, max_words=args.max_words,
    )
    picked = content.sample_balanced(
        rows, args.n_per_party, args.max_per_speaker, args.seed
    )
    content.assert_prefix_free([r["question"] for r in picked], phrases)

    table_path = ARTIFACTS / "content_corpus.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(picked)
    save_run_metadata(
        table_path,
        seed=args.seed,
        config={
            "congress": args.congress,
            "n_per_party": args.n_per_party,
            "max_per_speaker": args.max_per_speaker,
            "word_band": [args.min_words, args.max_words],
            "prompt_table": args.prompt_table,
            "sources": ["data/raw/hein-daily", "data/raw/voteview"],
        },
    )

    print(f"speech->speaker join: {join_report}")
    print(f"eligibility filter: {filter_report}")
    for party in ("D", "R"):
        of_party = [r for r in picked if r["party"] == party]
        dims = [r["nominate_dim1"] for r in of_party]
        n_speakers = len({r["icpsr"] for r in of_party})
        print(
            f"{party}: {len(of_party)} speeches, {n_speakers}"
            f" speakers, dim1 {min(dims):+.2f}..{max(dims):+.2f}, "
            f"mean words {sum(r['word_count'] for r in of_party) / len(of_party):.0f}, "
            f"mentions party-family token: {sum(r['mentions_party'] for r in of_party)}"
        )
    print(f"\nwrote {len(picked)} statements to {table_path} (+ .meta.json)")


if __name__ == "__main__":
    main()
