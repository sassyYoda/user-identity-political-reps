"""Pin the real-world partisan-lean anchor data (ticket 03, MATS arc).

    uv run python scripts/fetch_partisan_lean.py

Downloads the Pew Research Center 2024 validated-voter report, verifies it
against the pinned sha256, and writes data/raw/pew/partisan_lean.json mapping
demographic scaffold conditions to that report's 2024 net Republican margin
(Trump share minus Harris share, percentage points — positive is Republican,
matching the project's conservative-positive convention). The margins are
hand-transcribed from the table on report p. 23 ("Trump's support among
Hispanic, Black voters higher in 2024 than in 2020, 2016"); transcription
was fixed before any projection was computed, and re-checking it against the
PDF is an owner task (ticket 07).
"""

import hashlib
import json
from urllib.request import urlretrieve

from polreps.config import DATA_RAW

REPORT_URL = (
    "https://www.pewresearch.org/wp-content/uploads/sites/20/2025/06/"
    "PP-2025.6.26_validated-voters_report.pdf"
)
REPORT_SHA256 = "c18181328eca38d19183a5cb07ad5538265af3df930288fa0611d4dd2d98e1d0"
RETRIEVED = "2026-08-27"

CITATION = (
    "Pew Research Center, June 2025, \"Behind Trump's 2024 Victory: Turnout, "
    "Voting Patterns and Demographics\" (validated-voter study; survey of "
    "U.S. adults conducted Nov. 12-17, 2024, turnout verified against "
    "official state records)"
)

# 2024 Trump minus Harris, percentage points, from the 2024 columns of the
# report's p. 23 table (Men 55-43, Women 46-53, White 55-43, Black 15-83,
# Hispanic 48-51, Asian 40-57)
LEAN_BY_CONDITION = {
    "I am White. {}": 12,
    "I am Black. {}": -68,
    "I am Hispanic. {}": -3,
    "I am Asian. {}": -17,
    "I identify as a man. {}": 12,
    "I identify as a woman. {}": -7,
}


def main():
    pew_dir = DATA_RAW / "pew"
    pew_dir.mkdir(parents=True, exist_ok=True)
    pdf = pew_dir / "PP-2025.6.26_validated-voters_report.pdf"
    if not pdf.exists():
        print(f"downloading {REPORT_URL} ...")
        partial = pdf.with_suffix(".pdf.part")
        urlretrieve(REPORT_URL, partial)
        partial.rename(pdf)
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if digest != REPORT_SHA256:
        raise SystemExit(
            f"{pdf} sha256 {digest} does not match the pinned {REPORT_SHA256} "
            "— the upstream PDF moved; re-verify the transcribed margins"
        )

    anchor = {
        "source": {
            "citation": CITATION,
            "report_pdf": pdf.name,
            "report_url": REPORT_URL,
            "pdf_sha256": REPORT_SHA256,
            "retrieved": RETRIEVED,
            "table": (
                "\"Trump's support among Hispanic, Black voters higher in 2024 "
                "than in 2020, 2016\", report p. 23 (2024 columns)"
            ),
            "entry": (
                "hand-transcribed from the table image; owner verification "
                "against the PDF is a ticket-07 task"
            ),
        },
        "lean_metric": (
            "2024 Trump share minus Harris share among validated voters, "
            "percentage points (positive = Republican, matching the project's "
            "conservative-positive convention)"
        ),
        "notes": [
            "White/Black/Asian are single-race non-Hispanic; Hispanic voters "
            "are of any race; Asian estimates cover English speakers only "
            "(report footnote).",
            "No anchor exists in the table for 'I am American Indian. {}', "
            "'I am a Pacific Islander. {}', or 'I do not identify as a man or "
            "woman. {}' (groups not reported); those scaffolds are excluded "
            "from the rank correlation and the reported n counts only "
            "anchored scaffolds.",
            "Anchors fixed 2026-08-27 before any projection was computed "
            "(pre-registration hygiene).",
        ],
        "lean_by_condition": LEAN_BY_CONDITION,
    }
    out = pew_dir / "partisan_lean.json"
    out.write_text(json.dumps(anchor, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
