"""Content corpus: congressional statements labeled by their speaker's ideology.

The reference ideology direction must be extracted from text whose political
label comes from the *author's measured behavior* (DW-NOMINATE, first
dimension), never from a first-person identity prefix — otherwise the transfer
test could not distinguish an engaged representation from the surface-token
reading the leakage check demonstrated. Source: the Gentzkow/Shapiro/Taddy
parsed Congressional Record (hein-daily edition), one congress, joined to
Voteview member scores by seat — (state, district) for the House, state for
the Senate — plus an accent-folded lastname match (the hein "5-digit
congressperson id" is internal to that corpus, not an ICPSR id). The join
verifies party agreement between the two sources row by row and refuses the
corpus if the agreement rate looks like a broken join rather than
occasional data noise.

Everything here is a pure transform over already-downloaded text; the
download/extract step lives in scripts/fetch_content_data.py and the corpus
CSV assembly in scripts/build_content_corpus.py. The output table reuses the
prompt-table column contract ("prompt_id", "question") so the caching stage
walks it unchanged — a statement is cached at the same seam as a scaffolded
prompt: one user turn, last pre-generation token.
"""

import re
import unicodedata
from collections import Counter

# Voteview party codes for the two-party subset we label with
PARTY_CODE = {"100": "D", "200": "R"}

# below the party-agreement floor the speakerid->ICPSR join is broken, not
# noisy, and every label downstream would be suspect
MIN_PARTY_AGREEMENT = 0.95

# party-family tokens for the mentions_party flag (normalized-text match); the
# robustness cut re-runs transfer on rows where none of these appear
PARTY_TOKENS = ("democrat", "democrats", "democratic", "republican", "republicans", "gop")


def read_pipe_table(text, required_fields):
    """Rows of a hein pipe-delimited file as dicts, keyed by header names.

    The files are pipe-delimited with no quoting, so a stray pipe inside a
    field would silently shear a row; any row whose field count disagrees
    with the header is refused rather than guessed at.
    """
    lines = text.splitlines()
    header = lines[0].split("|")
    missing = [f for f in required_fields if f not in header]
    if missing:
        raise ValueError(f"header {header} lacks required field(s) {missing}")
    rows = []
    for i, line in enumerate(lines[1:], start=2):
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != len(header):
            raise ValueError(
                f"line {i} has {len(parts)} fields, header has {len(header)}"
            )
        rows.append(dict(zip(header, parts)))
    return rows


def read_speeches(text):
    """{speech_id: speech} from a speeches_###.txt. Split on the first pipe
    only — the speech text itself may contain pipes."""
    lines = text.splitlines()
    if lines[0].split("|")[:2] != ["speech_id", "speech"]:
        raise ValueError(f"unexpected speeches header: {lines[0]!r}")
    speeches = {}
    for i, line in enumerate(lines[1:], start=2):
        if not line:
            continue
        speech_id, sep, speech = line.partition("|")
        if not sep:
            raise ValueError(f"line {i} has no pipe delimiter")
        if speech_id in speeches:
            raise ValueError(f"duplicate speech_id {speech_id}")
        speeches[speech_id] = speech
    return speeches


def normalize_lastname(name):
    """Accent-folded uppercase letters and spaces only: Voteview bionames
    carry diacritics ("VELÁZQUEZ"), the hein OCR pipeline is plain ASCII."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z ]", "", folded.upper()).strip()


def voteview_members(rows, congress):
    """One congress's D/R members with NOMINATE scores, indexed by seat.

    Returns {"H": {(state, district): [member, ...]}, "S": {state: [...]}}
    with member = (lastname, party letter, nominate_dim1, icpsr). Rows
    outside the congress, outside House/Senate, outside the two-party codes,
    or with no score are skipped. A seat can legitimately list several
    members (special elections), so disambiguation is the joiner's job.
    """
    seats = {"H": {}, "S": {}}
    n = 0
    for row in rows:
        if int(row["congress"]) != congress or row["chamber"] not in ("House", "Senate"):
            continue
        party = PARTY_CODE.get(row["party_code"])
        if party is None or not row["nominate_dim1"]:
            continue
        member = (
            normalize_lastname(row["bioname"].split(",")[0]),
            party,
            float(row["nominate_dim1"]),
            row["icpsr"],
        )
        if row["chamber"] == "House":
            key = (row["state_abbrev"], int(float(row["district_code"])))
            seats["H"].setdefault(key, []).append(member)
        else:
            seats["S"].setdefault(row["state_abbrev"], []).append(member)
        n += 1
    if n == 0:
        raise ValueError(f"no D/R members with NOMINATE scores for congress {congress}")
    return seats


def match_speaker(row, seats):
    """The unique Voteview member for one SpeakerMap row, or None.

    House rows are looked up by (state, district) — hein codes at-large
    seats as district 0 where Voteview uses 1 — Senate rows by state; in
    both cases the folded lastname must single out exactly one member.
    """
    if row["chamber"] == "H":
        district = int(float(row["district"] or 0))
        candidates = seats["H"].get((row["state"], district), [])
        if not candidates and district == 0:
            candidates = seats["H"].get((row["state"], 1), [])
    else:
        candidates = seats["S"].get(row["state"], [])
    lastname = normalize_lastname(row["lastname"])
    hits = [m for m in candidates if m[0] == lastname]
    return hits[0] if len(hits) == 1 else None


def join_speaker_labels(speakermap_rows, seats):
    """{speech_id: (icpsr, party, nominate_dim1)} plus a drop report.

    Labels come from Voteview; the hein party column is the row-by-row
    cross-check that the seat+lastname match keyed the right person.
    Individual disagreements are dropped and counted, but an agreement rate
    below MIN_PARTY_AGREEMENT means the join itself is wrong, and
    everything stops.
    """
    labels, report, match_of = {}, Counter(), {}
    for row in speakermap_rows:
        if row["nonvoting"] != "voting":
            report["nonvoting"] += 1
            continue
        if row["party"] not in ("D", "R"):
            report["party_not_d_or_r"] += 1
            continue
        if row["speakerid"] not in match_of:
            match_of[row["speakerid"]] = match_speaker(row, seats)
        member = match_of[row["speakerid"]]
        if member is None:
            report["no_seat_match"] += 1
            continue
        _, party, dim1, icpsr = member
        if party != row["party"]:
            report["party_disagrees"] += 1
            continue
        report["matched"] += 1
        labels[row["speech_id"]] = (icpsr, party, dim1)

    checked = report["matched"] + report["party_disagrees"]
    if checked == 0 or report["matched"] / checked < MIN_PARTY_AGREEMENT:
        raise ValueError(
            f"party agreement {report['matched']}/{checked} between hein and "
            "Voteview — the seat+lastname join is broken, refusing to label"
        )
    return labels, dict(report)


def normalize_for_match(text):
    """Lowercase, punctuation to spaces, collapsed whitespace — the space in
    which scaffold phrases are searched for."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def scaffold_phrases(conditions):
    """The 21 scaffold sentences with their "{}" slot removed, normalized.
    Fed by the real prompt table so the guard tracks the actual vocabulary."""
    phrases = []
    for condition in conditions:
        if condition == "none":
            continue
        phrase = normalize_for_match(condition.replace("{}", " "))
        if not phrase:
            raise ValueError(f"condition {condition!r} normalizes to nothing")
        phrases.append(phrase)
    if not phrases:
        raise ValueError("no scaffold conditions to guard against")
    return phrases


def contains_phrase(normalized_text, phrase):
    return f" {phrase} " in f" {normalized_text} "


def assert_prefix_free(texts, phrases):
    """The ticket's guarantee: no scaffold phrase appears anywhere in the
    corpus. Run on the final table, after filtering, as the proof."""
    for i, text in enumerate(texts):
        normalized = normalize_for_match(text)
        for phrase in phrases:
            if contains_phrase(normalized, phrase):
                raise ValueError(
                    f"corpus row {i} contains scaffold phrase {phrase!r}"
                )


def mentions_party(normalized_text):
    return any(contains_phrase(normalized_text, tok) for tok in PARTY_TOKENS)


def eligible_rows(speeches, word_counts, labels, phrases, min_words, max_words):
    """Candidate corpus rows: labeled speeches in the word-count band with no
    scaffold phrase, plus a drop report."""
    rows, report = [], Counter()
    for speech_id, speech in speeches.items():
        if speech_id not in labels:
            report["unlabeled"] += 1
            continue
        words = word_counts.get(speech_id)
        if words is None:
            report["no_descr_row"] += 1
            continue
        if not min_words <= words <= max_words:
            report["outside_word_band"] += 1
            continue
        normalized = normalize_for_match(speech)
        if any(contains_phrase(normalized, p) for p in phrases):
            report["contains_scaffold_phrase"] += 1
            continue
        icpsr, party, dim1 = labels[speech_id]
        report["eligible"] += 1
        rows.append(
            {
                "speech_id": speech_id,
                "question": speech,
                "icpsr": icpsr,
                "party": party,
                "nominate_dim1": dim1,
                "word_count": words,
                "mentions_party": int(mentions_party(normalized)),
            }
        )
    if not rows:
        raise ValueError("no eligible speeches survived filtering")
    return rows, dict(report)


def sample_balanced(rows, n_per_party, max_per_speaker, seed):
    """Seeded draw of n_per_party rows per party, at most max_per_speaker per
    speaker so no single member dominates the direction. Deterministic given
    the seed and stable under input order (rows are keyed by speech_id)."""
    import random

    rng = random.Random(seed)
    picked = []
    for party in ("D", "R"):
        pool = sorted(
            (r for r in rows if r["party"] == party), key=lambda r: r["speech_id"]
        )
        rng.shuffle(pool)
        taken_of = Counter()
        chosen = []
        for row in pool:
            if taken_of[row["icpsr"]] >= max_per_speaker:
                continue
            chosen.append(row)
            taken_of[row["icpsr"]] += 1
            if len(chosen) == n_per_party:
                break
        if len(chosen) < n_per_party:
            raise ValueError(
                f"only {len(chosen)} {party} rows available under the "
                f"per-speaker cap; asked for {n_per_party}"
            )
        picked.extend(chosen)
    picked.sort(key=lambda r: r["speech_id"])
    for row in picked:
        row["prompt_id"] = f"crec-{row['speech_id']}"
    return picked
