"""The Moses endpoints, on the running sentence and the two corpora.

Every number the page quotes is pinned here, so a change in the library or in
this module fails a test rather than quietly changing a sentence on the page.
The last class is the one that matters most: nothing these endpoints return may
carry an identifier, because a widget renders these strings.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from oop_ml_api.concepts.moses_words import (
    MINUTES,
    NOTEBOOK,
    QUOTED_SENTENCE,
    RUNNING_SENTENCE,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

SPLIT = "/concepts/moses-rules/split"
FULL_STOPS = "/concepts/moses-rules/full-stops"
THE_LIST = "/concepts/moses-rules/the-list"
REVERSIBILITY = "/concepts/moses-rules/reversibility"
CORPUS = "/concepts/moses-rules/corpus"


def split_one(text: str, **settings: object) -> dict:
    """One text put to the rules, unwrapped to the view the page quotes."""
    response = client.post(SPLIT, json={"texts": [text], **settings})
    assert response.status_code == 200
    return response.json()["texts"][0]


class TestTheRunningSentence:
    def test_the_sentence_becomes_nine_pieces(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["on_these_rules"]["n_pieces"] == 9
        assert [piece["text"] for piece in view["on_these_rules"]["pieces"]] == [
            "Dr.",
            "Alvarez",
            "didn",
            "'t",
            "expect",
            "the",
            "low-cost",
            "re-analysis",
            ".",
        ]

    def test_every_piece_is_the_slice_its_span_names(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        for piece in view["on_these_rules"]["pieces"]:
            assert piece["text"] == piece["source"]
            assert not piece["is_rewritten"]

    def test_the_spans_are_the_ones_the_page_prints(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert [
            (piece["start"], piece["end"]) for piece in view["on_these_rules"]["pieces"]
        ] == [
            (0, 3),
            (4, 11),
            (12, 16),
            (16, 18),
            (19, 25),
            (26, 29),
            (30, 38),
            (39, 50),
            (50, 51),
        ]

    def test_the_annotation_rules_give_nine_pieces_too(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["on_annotation_rules"]["n_pieces"] == 9
        assert view["on_spaces"]["n_pieces"] == 7
        assert view["on_the_table"]["n_pieces"] == 13

    def test_the_two_nines_differ_in_exactly_one_place(self) -> None:
        view = split_one(RUNNING_SENTENCE)
        here = [piece["text"] for piece in view["on_these_rules"]["pieces"]]
        there = [piece["text"] for piece in view["on_annotation_rules"]["pieces"]]

        assert sorted(set(here) - set(there)) == ["'t", "didn"]
        assert sorted(set(there) - set(here)) == ["did", "n't"]
        assert len(set(here) & set(there)) == 7

    def test_the_sentence_holds_fifty_one_characters_and_forty_five_in_pieces(
        self,
    ) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["n_characters"] == 51
        assert view["n_non_space_characters"] == 45
        assert view["on_these_rules"]["n_characters_in_pieces"] == 45


class TestTheSettings:
    def test_marking_the_hyphens_lengthens_the_sentence_by_four(self) -> None:
        plain = split_one(RUNNING_SENTENCE)
        marked = split_one(RUNNING_SENTENCE, mark_hyphens=True)

        assert plain["on_these_rules"]["n_pieces"] == 9
        assert marked["on_these_rules"]["n_pieces"] == 13
        assert [piece["text"] for piece in marked["on_these_rules"]["pieces"]].count(
            "@-@"
        ) == 2

    def test_a_marked_hyphen_stands_for_one_character_and_undoes_to_it(self) -> None:
        marked = split_one(RUNNING_SENTENCE, mark_hyphens=True)
        placeholders = [
            piece
            for piece in marked["on_these_rules"]["pieces"]
            if piece["text"] == "@-@"
        ]

        assert len(placeholders) == 2
        for piece in placeholders:
            assert piece["end"] - piece["start"] == 1
            assert piece["is_rewritten"]
            assert piece["undone"] == "-"

    def test_rewriting_the_marks_leaves_the_spans_alone(self) -> None:
        view = split_one("don't stop", rewrite_marks=True)
        pieces = view["on_these_rules"]["pieces"]

        assert [piece["text"] for piece in pieces] == ["don", "&apos;t", "stop"]
        assert pieces[1]["start"] == 3
        assert pieces[1]["end"] == 5
        assert pieces[1]["undone"] == "'t"

    def test_the_french_setting_cuts_the_other_way(self) -> None:
        english = split_one("l'analyse", language="en")
        french = split_one("l'analyse", language="fr")

        assert [piece["text"] for piece in english["on_these_rules"]["pieces"]] == [
            "l",
            "'analyse",
        ]
        assert [piece["text"] for piece in french["on_these_rules"]["pieces"]] == [
            "l'",
            "analyse",
        ]

    def test_the_english_contraction_under_the_french_setting(self) -> None:
        french = split_one("don't", language="fr")

        assert [piece["text"] for piece in french["on_these_rules"]["pieces"]] == [
            "don'",
            "t",
        ]

    def test_a_setting_the_rules_have_no_clause_for_is_refused(self) -> None:
        response = client.post(SPLIT, json={"texts": ["a"], "language": "de"})

        assert response.status_code == 422

    def test_an_unknown_keyword_is_refused(self) -> None:
        response = client.post(SPLIT, json={"texts": ["a"], "hyphens": True})

        assert response.status_code == 422

    def test_a_text_longer_than_the_limit_is_refused(self) -> None:
        response = client.post(SPLIT, json={"texts": ["a" * 401]})

        assert response.status_code == 422


class TestTheFullStops:
    def test_ten_decidable_stops_and_three_with_no_answer(self) -> None:
        body = client.get(FULL_STOPS).json()

        assert len(body["probes"]) == 10
        assert len(body["undecidable"]) == 3
        assert body["scoreboard"]["n_cases"] == 10
        assert body["scoreboard"]["n_wanting_kept"] == 8

    def test_the_scores_the_page_quotes(self) -> None:
        board = client.get(FULL_STOPS).json()["scoreboard"]

        assert board["n_right_here"] == 7
        assert board["n_right_by_annotation"] == 8
        assert board["n_right_by_the_table"] == 4

    def test_the_annotation_rules_answer_kept_every_time(self) -> None:
        board = client.get(FULL_STOPS).json()["scoreboard"]

        assert board["n_answers_by_annotation"] == 10
        assert board["n_answers_here"] == 5
        assert board["n_answers_by_the_table"] == 2

    def test_the_three_misses_are_all_abbreviations_that_are_not_listed(self) -> None:
        probes = client.get(FULL_STOPS).json()["probes"]
        missed = [probe["word"] for probe in probes if not probe["right_here"]]

        assert missed == ["approx", "al", "Gov"]

    def test_the_same_word_is_kept_in_one_text_and_split_in_another(self) -> None:
        probes = {
            probe["text"]: probe for probe in client.get(FULL_STOPS).json()["probes"]
        }

        assert probes["No. 5 shows the drift."]["kept_here"]
        assert not probes["No. Nothing shows the drift."]["kept_here"]
        assert not probes["The reading was approx. 40 per cent."]["kept_here"]
        assert probes["The reading was approx. forty per cent."]["kept_here"]
        assert probes["Alvarez et al. ran the analysis."]["kept_here"]
        assert not probes["Alvarez et al. Ran it again."]["kept_here"]

    def test_the_annotation_rules_keep_a_stop_the_page_says_they_keep(self) -> None:
        probes = {
            probe["text"]: probe for probe in client.get(FULL_STOPS).json()["probes"]
        }

        assert probes["Gov. Reyes asked for the figures."]["kept_by_annotation"]
        assert probes["The drift was small. It did not go away."]["kept_by_annotation"]

    def test_the_undecidable_ones_part_the_two_rules(self) -> None:
        undecidable = {
            probe["text"]: probe
            for probe in client.get(FULL_STOPS).json()["undecidable"]
        }
        text = "The re-analysis was signed by Okafor Co."

        assert undecidable[text]["kept_here"]
        assert not undecidable[text]["kept_by_annotation"]


class TestTheList:
    def test_the_list_holds_fifty_five_entries(self) -> None:
        coverage = client.get(THE_LIST).json()["coverage"]

        assert coverage["n_entries"] == 55
        assert coverage["n_named_entries"] == 27
        assert coverage["n_single_letter_entries"] == 26
        assert coverage["n_figure_only_entries"] == 2

    def test_eighteen_sentences_reach_nineteen_of_them(self) -> None:
        coverage = client.get(THE_LIST).json()["coverage"]

        assert coverage["n_texts"] == 18
        assert coverage["n_words_ending_in_one_stop"] == 39
        assert coverage["n_reached"] == 19
        assert coverage["n_never_reached"] == 36
        assert coverage["share_reached"] == 19 / 55

    def test_the_entries_reached_are_the_ones_the_page_names(self) -> None:
        coverage = client.get(THE_LIST).json()["coverage"]

        assert coverage["reached"] == [
            "Art",
            "Co",
            "Corp",
            "Dec",
            "Dr",
            "Inc",
            "J",
            "Jan",
            "Ltd",
            "Mr",
            "Mrs",
            "Ms",
            "Mt",
            "No",
            "Nov",
            "Prof",
            "St",
            "etc",
            "vs",
        ]

    def test_four_of_the_eighteen_sentences_end_in_an_abbreviation(self) -> None:
        coverage = client.get(THE_LIST).json()["coverage"]

        assert coverage["n_texts_ending_in_an_abbreviation"] == 4
        assert coverage["texts_ending_in_an_abbreviation"] == [
            "The contract was signed by Alvarez Instruments Ltd. and by Okafor Co.",
            "Gov. Reyes had asked for the same figures in Jan.",
            "Ms. Haddad read out the reading, the drift and the residual, etc.",
            "The re-analysis cost 1,000 euros and took until Dec.",
        ]

    def test_only_the_two_lists_kept_a_stop_over_these_eighteen_sentences(
        self,
    ) -> None:
        coverage = client.get(THE_LIST).json()["coverage"]
        tally = {entry["clause"]: entry["n_kept"] for entry in coverage["by_clause"]}

        assert coverage["n_kept_in_all"] == 20
        assert tally["the named list"] == 18
        assert tally["the figures list, with a figure after it"] == 2
        assert tally["an interior stop and a letter inside the word"] == 0
        assert tally["a lowercase word after it"] == 0

    def test_the_abbreviations_that_lost_their_stop(self) -> None:
        coverage = client.get(THE_LIST).json()["coverage"]

        for word in ("Gov", "Sen", "Fig", "Rev", "approx"):
            assert word in coverage["lost_the_stop"]

    def test_the_named_entries_are_titles_months_and_company_suffixes(self) -> None:
        body = client.get(THE_LIST).json()

        assert "Dr" in body["named_entries"]
        assert "Dec" in body["named_entries"]
        assert "Gov" not in body["named_entries"]
        assert body["figure_only_entries"] == ["Art", "No"]

    def test_the_written_out_table_holds_ninety_entries(self) -> None:
        body = client.get(THE_LIST).json()

        assert body["n_table_entries"] == 90
        assert body["n_table_entries_kept_whole"] == 46
        assert body["n_table_entries_cut"] == 44
        assert body["n_table_entries_reached"] == 17

    def test_an_entry_may_say_where_a_word_is_cut_and_not_what_it_becomes(self) -> None:
        proposals = client.get(THE_LIST).json()["proposals"]
        verdicts = {
            (entry["word"], tuple(entry["pieces"])): entry for entry in proposals
        }

        assert verdicts[("don't", ("do", "n't"))]["accepted"]
        assert not verdicts[("don't", ("do", "not"))]["accepted"]
        assert verdicts[("Mr.", ("Mr.",))]["accepted"]
        assert not verdicts[("Mr.", ("Mr",))]["accepted"]
        assert verdicts[("cannot", ("can", "not"))]["accepted"]
        assert not verdicts[("New York", ("New", "York"))]["accepted"]

    def test_the_refusals_are_written_in_the_pages_own_words(self) -> None:
        proposals = client.get(THE_LIST).json()["proposals"]
        refused = [entry for entry in proposals if not entry["accepted"]]

        assert len(refused) == 3
        for entry in refused:
            assert entry["verdict"].startswith("refused, because")


class TestReversibility:
    def test_the_pieces_of_the_quoted_sentence_join_back_to_it(self) -> None:
        body = client.get(REVERSIBILITY).json()

        assert body["quoted"]["on_these_rules"]["pieces_join_to_the_source"]
        assert not body["quoted"]["on_annotation_rules"]["pieces_join_to_the_source"]

    def test_the_annotation_rules_rewrite_two_pieces_of_it_and_these_rewrite_none(
        self,
    ) -> None:
        body = client.get(REVERSIBILITY).json()

        assert body["quoted"]["on_these_rules"]["n_rewritten"] == 0
        assert body["quoted"]["on_annotation_rules"]["n_rewritten"] == 2
        assert body["quoted"]["on_these_rules"]["n_pieces"] == 14
        assert body["quoted"]["on_annotation_rules"]["n_pieces"] == 14

    def test_rewriting_the_marks_can_be_undone_where_the_other_rewriting_cannot(
        self,
    ) -> None:
        body = client.get(REVERSIBILITY).json()
        rewritten = body["quoted_with_marks_rewritten"]["on_these_rules"]

        assert rewritten["n_rewritten"] == 3
        assert not rewritten["pieces_join_to_the_source"]
        assert rewritten["undoing_recovers_the_source"]
        assert not body["quoted"]["on_annotation_rules"]["undoing_recovers_the_source"]

    def test_two_different_texts_come_back_as_one_answer(self) -> None:
        pair = client.get(REVERSIBILITY).json()["ambiguity"]

        assert pair["first"] != pair["second"]
        assert pair["first_on_annotation_rules"] == pair["second_on_annotation_rules"]
        assert not pair["told_apart_by_annotation"]
        assert pair["told_apart_here"]

    def test_the_eighteen_sentences_join_back_here_and_not_quite_there(self) -> None:
        body = client.get(REVERSIBILITY).json()

        assert body["n_texts"] == 18
        assert body["n_joining_here"] == 18
        assert body["n_joining_by_annotation"] == 17

    def test_joining_with_one_space_each_is_not_the_source(self) -> None:
        body = client.get(REVERSIBILITY).json()

        assert body["n_glue_exact_here"] == 2
        assert body["n_glue_exact_on_spaces"] == 18

    def test_marking_the_hyphens_lengthens_the_running_sentence(self) -> None:
        body = client.get(REVERSIBILITY).json()

        assert body["running_with_hyphens_marked"]["on_these_rules"]["n_pieces"] == 13
        assert body["running_with_hyphens_marked"]["on_these_rules"][
            "undoing_recovers_the_source"
        ]


class TestTheCorpora:
    def test_the_notebook_counts_the_page_quotes(self) -> None:
        notebook = client.get(CORPUS).json()["notebook"]

        assert notebook["n_texts"] == 6
        assert notebook["n_characters"] == 375
        assert notebook["n_non_space_characters"] == 318
        assert notebook["on_these_rules"]["n_pieces"] == 72
        assert notebook["on_annotation_rules"]["n_pieces"] == 72
        assert notebook["on_spaces"]["n_pieces"] == 63
        assert notebook["on_the_table"]["n_pieces"] == 84

    def test_the_notebook_costs_one_more_distinct_entry_than_the_annotation_rules(
        self,
    ) -> None:
        notebook = client.get(CORPUS).json()["notebook"]

        assert notebook["on_these_rules"]["n_distinct"] == 49
        assert notebook["on_annotation_rules"]["n_distinct"] == 48
        assert notebook["on_spaces"]["n_distinct"] == 47
        assert notebook["only_on_these_rules"] == ["'t", "didn"]
        assert notebook["only_on_annotation_rules"] == ["n't"]

    def test_the_minutes_counts_the_page_quotes(self) -> None:
        minutes = client.get(CORPUS).json()["minutes"]

        assert minutes["n_texts"] == 18
        assert minutes["n_characters"] == 1105
        assert minutes["n_non_space_characters"] == 931
        assert minutes["on_these_rules"]["n_pieces"] == 228
        assert minutes["on_annotation_rules"]["n_pieces"] == 227
        assert minutes["on_spaces"]["n_pieces"] == 192
        assert minutes["on_the_table"]["n_pieces"] == 243

    def test_the_distinct_entries_barely_move_at_this_size(self) -> None:
        minutes = client.get(CORPUS).json()["minutes"]

        assert minutes["on_spaces"]["n_distinct"] == 137
        assert minutes["on_these_rules"]["n_distinct"] == 136

    def test_the_scattered_stems_are_the_failure_the_rules_repair(self) -> None:
        minutes = client.get(CORPUS).json()["minutes"]
        scattered = {
            entry["stem"]: entry["spellings"] for entry in minutes["scattered"]
        }

        assert len(scattered) == 5
        assert scattered["Okafor"] == ["Okafor", "Okafor,"]
        assert scattered["re-analysis"] == ["re-analysis", "re-analysis."]
        assert scattered["1,000"] == ["1,000", "1,000."]

    def test_the_notebook_scatters_one_stem(self) -> None:
        notebook = client.get(CORPUS).json()["notebook"]

        assert [entry["stem"] for entry in notebook["scattered"]] == ["re-analysis"]

    def test_every_rule_keeps_every_character_that_is_not_spacing(self) -> None:
        for name in ("notebook", "minutes"):
            view = client.get(CORPUS).json()[name]
            for rule in ("on_these_rules", "on_annotation_rules", "on_spaces"):
                assert (
                    view[rule]["n_characters_covered"] == view["n_non_space_characters"]
                )

    def test_marking_the_hyphens_costs_twelve_pieces_on_each_corpus(self) -> None:
        body = client.get(CORPUS).json()

        assert body["notebook"]["n_pieces_with_hyphens_marked"] == 84
        assert body["minutes"]["n_pieces_with_hyphens_marked"] == 240

    def test_a_seventh_of_the_minutes_pieces_hold_no_letter_and_no_digit(self) -> None:
        minutes = client.get(CORPUS).json()["minutes"]

        assert minutes["n_pieces_without_letter_or_digit"] == 32
        assert minutes["on_these_rules"]["n_pieces"] == 228

    def test_the_corpora_are_the_ones_the_page_names(self) -> None:
        body = client.get(CORPUS).json()

        assert body["notebook_texts"] == list(NOTEBOOK)
        assert body["minutes_texts"] == list(MINUTES)
        assert body["running"]["source"] == RUNNING_SENTENCE


IDENTIFIER = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b|\b[A-Z][a-z]+[A-Z][A-Za-z]*\b")
"""A snake_case or CamelCase word, which is what an identifier looks like in prose."""

FORBIDDEN = ("oop_ml", "the library", "the API", "the package", "the backend")


def _every_string(payload: object) -> list[str]:
    """Every string anywhere in a response body, keys excluded."""
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, list):
        return [text for item in payload for text in _every_string(item)]
    if isinstance(payload, dict):
        return [text for item in payload.values() for text in _every_string(item)]
    return []


def _readable_strings(payload: object) -> list[str]:
    """Every returned string except the writing the rules read and produced.

    The corpora hold hyphenated compounds and the rules produce a placeholder
    with hyphens in it, which is the writing the page is about rather than
    anything naming the code, so those are excluded and every label, note and
    verdict written here is checked.
    """
    produced = {
        "@-@",
        "low-cost",
        "re-analysis",
        "re-analysis.",
        "re-ran",
        "well-known",
    }
    return [
        text
        for text in _every_string(payload)
        if text not in produced and "-" not in text
    ]


class TestNothingReadableNamesTheCode:
    def test_no_response_carries_an_identifier_or_credits_the_code(self) -> None:
        bodies = [
            client.post(
                SPLIT, json={"texts": [RUNNING_SENTENCE, QUOTED_SENTENCE]}
            ).json(),
            client.get(FULL_STOPS).json(),
            client.get(THE_LIST).json(),
            client.get(REVERSIBILITY).json(),
            client.get(CORPUS).json(),
        ]

        for body in bodies:
            for text in _readable_strings(body):
                assert not IDENTIFIER.search(text), text
                for phrase in FORBIDDEN:
                    assert phrase not in text, text

    def test_no_response_carries_an_em_dash(self) -> None:
        for body in (
            client.get(FULL_STOPS).json(),
            client.get(THE_LIST).json(),
            client.get(REVERSIBILITY).json(),
            client.get(CORPUS).json(),
        ):
            for text in _every_string(body):
                assert "—" not in text
