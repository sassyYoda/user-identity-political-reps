import pytest

from polreps import content

CONGRESS = 114

SPEAKERMAP = "\n".join(
    [
        "speakerid|speech_id|lastname|firstname|chamber|state|gender|party|district|nonvoting",
        "114120100|s1|SMITH|JANE|S|MA|F|D|0|voting",
        "114120100|s2|SMITH|JANE|S|MA|F|D|0|voting",
        "114203450|s3|JONES|TOM|H|TX|M|R|5|voting",
        "114203450|s4|JONES|TOM|H|TX|M|R|5|voting",
        "114999990|s5|DOE|SAM|H|PR|M|D|1|nonvoting",
        "114111110|s6|ROE|ANN|H|CA|F|I|9|voting",
    ]
)

SPEECHES = "\n".join(
    [
        "speech_id|speech",
        "s1|The minimum wage must rise for working families",
        "s2|We should expand access to healthcare in every state",
        "s3|Taxes are too high and the border | is not secure",
        "s4|I rise to honor the champions of the little league",
        "s5|Delegate speech about the budget and appropriations",
        "s6|Independent speech about infrastructure and roads",
    ]
)

VOTEVIEW = [
    {"congress": "114", "chamber": "Senate", "icpsr": "20100", "state_abbrev": "MA",
     "district_code": "0", "bioname": "SMITH, Jane", "party_code": "100",
     "nominate_dim1": "-0.4"},
    {"congress": "114", "chamber": "House", "icpsr": "3450", "state_abbrev": "TX",
     "district_code": "5", "bioname": "JONES, Tom", "party_code": "200",
     "nominate_dim1": "0.5"},
    {"congress": "113", "chamber": "House", "icpsr": "3450", "state_abbrev": "TX",
     "district_code": "5", "bioname": "JONES, Tom", "party_code": "200",
     "nominate_dim1": "0.9"},
]


def test_read_pipe_table_refuses_sheared_rows():
    rows = content.read_pipe_table(SPEAKERMAP, ("speakerid", "speech_id", "party"))
    assert len(rows) == 6
    assert rows[0]["lastname"] == "SMITH"
    with pytest.raises(ValueError, match="fields"):
        content.read_pipe_table("a|b\n1|2|3", ("a", "b"))
    with pytest.raises(ValueError, match="required"):
        content.read_pipe_table(SPEAKERMAP, ("speakerid", "icpsr"))


def test_read_speeches_splits_on_first_pipe_only():
    speeches = content.read_speeches(SPEECHES)
    assert speeches["s3"] == "Taxes are too high and the border | is not secure"
    with pytest.raises(ValueError, match="duplicate"):
        content.read_speeches("speech_id|speech\ns1|a\ns1|b")


def test_voteview_members_indexes_seats_for_one_congress():
    seats = content.voteview_members(VOTEVIEW, CONGRESS)
    # the 113th-congress row is ignored
    assert seats["S"] == {"MA": [("SMITH", "D", -0.4, "20100")]}
    assert seats["H"] == {("TX", 5): [("JONES", "R", 0.5, "3450")]}


def test_match_speaker_handles_at_large_and_accents():
    seats = {
        "H": {("VT", 1): [("WELCH", "D", -0.3, "20952")],
              ("NY", 7): [("VELAZQUEZ", "D", -0.5, "29572")]},
        "S": {},
    }
    # hein codes at-large districts 0, Voteview codes them 1
    at_large = {"chamber": "H", "state": "VT", "district": "0", "lastname": "WELCH"}
    assert content.match_speaker(at_large, seats) == ("WELCH", "D", -0.3, "20952")
    accented = {"chamber": "H", "state": "NY", "district": "7", "lastname": "VELAZQUEZ"}
    assert content.match_speaker(accented, seats) == ("VELAZQUEZ", "D", -0.5, "29572")
    wrong_name = {"chamber": "H", "state": "NY", "district": "7", "lastname": "NGUYEN"}
    assert content.match_speaker(wrong_name, seats) is None


def test_normalize_lastname_folds_diacritics():
    assert content.normalize_lastname("Velázquez") == "VELAZQUEZ"
    assert content.normalize_lastname("MCMORRIS RODGERS") == "MCMORRIS RODGERS"


def test_match_speaker_refuses_ambiguous_seats():
    # a special election can put two same-name members on one seat; nothing
    # singles one out, so no label
    seats = {"H": {("OH", 8): [("SMITH", "R", 0.4, "1"), ("SMITH", "R", 0.6, "2")]}, "S": {}}
    row = {"chamber": "H", "state": "OH", "district": "8", "lastname": "SMITH"}
    assert content.match_speaker(row, seats) is None


def test_join_speaker_labels_drops_and_reports():
    speakermap = content.read_pipe_table(SPEAKERMAP, ("speakerid",))
    seats = content.voteview_members(VOTEVIEW, CONGRESS)
    labels, report = content.join_speaker_labels(speakermap, seats)

    assert labels["s1"] == ("20100", "D", -0.4)
    assert labels["s3"] == ("3450", "R", 0.5)
    assert report["matched"] == 4
    assert report["nonvoting"] == 1
    assert report["party_not_d_or_r"] == 1


def test_join_refuses_when_party_agreement_collapses():
    # Voteview says everyone is a Republican: the join must be broken
    speakermap = content.read_pipe_table(SPEAKERMAP, ("speakerid",))
    seats = {
        "S": {"MA": [("SMITH", "R", 0.4, "20100")]},
        "H": {("TX", 5): [("JONES", "R", 0.5, "3450")]},
    }
    with pytest.raises(ValueError, match="agreement"):
        content.join_speaker_labels(speakermap, seats)


def test_scaffold_phrases_and_prefix_guard():
    phrases = content.scaffold_phrases(
        ["none", "I am a Democrat. {}", "{} Please cite your sources."]
    )
    assert "i am a democrat" in phrases
    assert "please cite your sources" in phrases

    content.assert_prefix_free(["A speech about democratic values"], phrases)
    with pytest.raises(ValueError, match="scaffold phrase"):
        # survives the hein cleaning that swaps punctuation for spaces
        content.assert_prefix_free(["Colleagues. I am a Democrat. and proud"], phrases)


def test_mentions_party_flag_is_word_bounded():
    assert content.mentions_party(content.normalize_for_match("the GOP plan"))
    assert not content.mentions_party(
        content.normalize_for_match("a democratization effort abroad")
    )


def make_eligible_inputs():
    speeches = content.read_speeches(SPEECHES)
    word_counts = {sid: 200 for sid in speeches}
    labels = {
        "s1": ("12010", "D", -0.4),
        "s2": ("12010", "D", -0.4),
        "s3": ("20345", "R", 0.5),
        "s4": ("20345", "R", 0.5),
    }
    phrases = content.scaffold_phrases(["I am a Democrat. {}"])
    return speeches, word_counts, labels, phrases


def test_eligible_rows_filters_word_band_and_reports():
    speeches, word_counts, labels, phrases = make_eligible_inputs()
    word_counts["s2"] = 5  # procedural-short
    rows, report = content.eligible_rows(
        speeches, word_counts, labels, phrases, min_words=100, max_words=400
    )

    assert {r["speech_id"] for r in rows} == {"s1", "s3", "s4"}
    assert report == {"eligible": 3, "outside_word_band": 1, "unlabeled": 2}
    by_id = {r["speech_id"]: r for r in rows}
    assert by_id["s1"]["party"] == "D"
    assert by_id["s1"]["nominate_dim1"] == -0.4
    assert by_id["s1"]["mentions_party"] == 0


def test_sample_balanced_is_deterministic_and_caps_speakers():
    rows = []
    for party, icpsr in (("D", "11111"), ("D", "22222"), ("R", "33333"), ("R", "44444")):
        for i in range(20):
            rows.append(
                {"speech_id": f"{icpsr}-{i:02d}", "party": party, "icpsr": icpsr}
            )
    picked = content.sample_balanced(rows, n_per_party=10, max_per_speaker=6, seed=0)

    assert len(picked) == 20
    from collections import Counter

    per_party = Counter(r["party"] for r in picked)
    assert per_party == {"D": 10, "R": 10}
    per_speaker = Counter(r["icpsr"] for r in picked)
    assert max(per_speaker.values()) <= 6
    assert all(r["prompt_id"] == f"crec-{r['speech_id']}" for r in picked)

    again = content.sample_balanced(rows, n_per_party=10, max_per_speaker=6, seed=0)
    assert [r["speech_id"] for r in again] == [r["speech_id"] for r in picked]

    with pytest.raises(ValueError, match="per-speaker cap"):
        content.sample_balanced(rows, n_per_party=15, max_per_speaker=6, seed=0)
