"""The Penn Treebank endpoints, on the running sentence and the two corpora.

Every number the page quotes is pinned here, so a change in the library or in
this module fails a test rather than quietly changing a sentence on the page.
The last class is the one that matters most: nothing these endpoints return may
carry an identifier, because a widget renders these strings.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from oop_ml_api.concepts.treebank_words import (
    QUOTED_SENTENCE,
    RUNNING_SENTENCE,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

SPLIT = "/concepts/penn-treebank-rules/split"
CONTRACTIONS = "/concepts/penn-treebank-rules/contractions"
FULL_STOPS = "/concepts/penn-treebank-rules/full-stops"
CORPUS = "/concepts/penn-treebank-rules/corpus"


def split_one(text: str) -> dict:
    """One text put to the rules, unwrapped to the view the page quotes."""
    response = client.post(SPLIT, json={"texts": [text]})
    assert response.status_code == 200
    return response.json()["texts"][0]


class TestTheRunningSentence:
    def test_the_sentence_becomes_nine_pieces(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["on_treebank"]["n_pieces"] == 9
        assert [piece["text"] for piece in view["on_treebank"]["pieces"]] == [
            "Dr.",
            "Alvarez",
            "did",
            "n't",
            "expect",
            "the",
            "low-cost",
            "re-analysis",
            ".",
        ]

    def test_every_piece_carries_the_span_it_was_cut_from(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert [
            (piece["start"], piece["end"]) for piece in view["on_treebank"]["pieces"]
        ] == [
            (0, 3),
            (4, 11),
            (12, 15),
            (15, 18),
            (19, 25),
            (26, 29),
            (30, 38),
            (39, 50),
            (50, 51),
        ]

    def test_no_piece_of_the_running_sentence_is_rewritten(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["on_treebank"]["n_rewritten"] == 0
        assert view["on_treebank"]["spans_recover_the_source"] is True

    def test_the_two_sibling_rules_on_the_same_sentence(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["on_spaces"]["n_pieces"] == 7
        assert view["on_boundaries"]["n_pieces"] == 9

    def test_it_keeps_every_character_that_is_not_a_space(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["n_characters"] == 51
        assert view["n_non_space_characters"] == 45
        assert view["on_treebank"]["n_characters_in_pieces"] == 45
        assert view["on_boundaries"]["n_characters_in_pieces"] == 41

    def test_gluing_the_pieces_with_spaces_does_not_give_the_sentence_back(
        self,
    ) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["on_treebank"]["glue_is_exact"] is False
        assert view["on_treebank"]["glued"] == (
            "Dr. Alvarez did n't expect the low-cost re-analysis ."
        )
        assert view["on_spaces"]["glue_is_exact"] is True


class TestTheQuotedSentence:
    def test_the_quotation_marks_come_back_as_something_else(self) -> None:
        view = split_one(QUOTED_SENTENCE)

        assert view["on_treebank"]["n_pieces"] == 14
        assert view["on_treebank"]["n_rewritten"] == 2
        rewritten = [
            piece for piece in view["on_treebank"]["pieces"] if piece["is_rewritten"]
        ]
        assert [piece["text"] for piece in rewritten] == ["``", "''"]
        assert [piece["source"] for piece in rewritten] == ['"', '"']

    def test_a_rewritten_piece_keeps_the_span_of_the_one_character_it_stands_for(
        self,
    ) -> None:
        view = split_one(QUOTED_SENTENCE)
        opening = view["on_treebank"]["pieces"][0]

        assert (opening["start"], opening["end"]) == (0, 1)
        assert len(opening["text"]) == 2

    def test_the_spans_still_recover_everything_that_is_not_a_space(self) -> None:
        view = split_one(QUOTED_SENTENCE)

        assert view["on_treebank"]["spans_recover_the_source"] is True
        assert view["n_non_space_characters"] == 56
        assert view["on_treebank"]["n_characters_in_pieces"] == 56

    def test_naming_the_brackets_changes_nothing_where_there_are_none(self) -> None:
        view = split_one(QUOTED_SENTENCE)

        assert view["on_treebank_naming_brackets"]["n_rewritten"] == 2


class TestBracketsAreRewrittenOnlyWhenAsked:
    def test_a_bracket_is_itself_by_default_and_a_name_on_request(self) -> None:
        view = split_one("She said (quietly) that it was fine.")

        assert [piece["text"] for piece in view["on_treebank"]["pieces"]][2] == "("
        assert [
            piece["text"] for piece in view["on_treebank_naming_brackets"]["pieces"]
        ][2] == "-LRB-"

    def test_the_named_bracket_still_stands_for_one_character(self) -> None:
        view = split_one("She said (quietly) that it was fine.")
        named = [
            piece
            for piece in view["on_treebank_naming_brackets"]["pieces"]
            if piece["is_rewritten"]
        ]

        assert len(named) == 2
        assert all(piece["end"] - piece["start"] == 1 for piece in named)


class TestContractions:
    def test_fifteen_probes_and_six_of_them_are_left_whole(self) -> None:
        body = client.get(CONTRACTIONS).json()

        assert body["n_probes"] == 15
        assert body["n_cut"] == 9
        assert body["n_left_whole"] == 6
        assert body["n_fixed_forms"] == 10
        assert body["n_clitic_endings"] == 8

    def test_the_irregular_hosts_are_what_is_left_over(self) -> None:
        body = client.get(CONTRACTIONS).json()
        found = {
            probe["text"]: [piece["text"] for piece in probe["pieces"]]
            for probe in body["probes"]
        }

        assert found["can't"] == ["ca", "n't"]
        assert found["won't"] == ["wo", "n't"]
        assert found["shan't"] == ["sha", "n't"]
        assert found["didn't"] == ["did", "n't"]

    def test_a_form_with_no_apostrophe_is_still_cut(self) -> None:
        body = client.get(CONTRACTIONS).json()
        found = {
            probe["text"]: [piece["text"] for piece in probe["pieces"]]
            for probe in body["probes"]
        }

        assert found["cannot"] == ["can", "not"]
        assert found["gonna"] == ["gon", "na"]

    def test_the_forms_nobody_listed_come_back_whole_and_say_nothing(self) -> None:
        body = client.get(CONTRACTIONS).json()
        whole = {probe["text"] for probe in body["probes"] if not probe["is_cut"]}

        assert whole == {
            "didn’t",
            "l'analyse",
            "dell'analisi",
            "gotcha",
            "y'all",
            "o'clock",
        }

    def test_only_the_quoted_probe_carries_a_rewritten_piece(self) -> None:
        body = client.get(CONTRACTIONS).json()
        rewritten = {
            probe["text"]: probe["n_rewritten"]
            for probe in body["probes"]
            if probe["n_rewritten"] > 0
        }

        assert rewritten == {'"It\'s fine,"': 2}


class TestTheFinalStop:
    def test_the_abbreviation_keeps_its_stop_and_the_sentence_loses_its_own(
        self,
    ) -> None:
        body = client.get(FULL_STOPS).json()
        found = {probe["text"]: probe for probe in body["probes"]}

        assert found[RUNNING_SENTENCE]["attached_stops"] == ["Dr."]
        assert found[RUNNING_SENTENCE]["has_own_stop_piece"] is True

    def test_a_second_sentence_leaves_the_first_ones_stop_attached(self) -> None:
        body = client.get(FULL_STOPS).json()
        found = {probe["text"]: probe for probe in body["probes"]}
        probe = found["The drift was small. It did not go away."]

        assert probe["attached_stops"] == ["small."]
        assert [piece["text"] for piece in probe["pieces"]][3] == "small."

    def test_an_abbreviation_at_the_end_loses_the_stop_that_belongs_to_it(self) -> None:
        body = client.get(FULL_STOPS).json()
        found = {probe["text"]: probe for probe in body["probes"]}

        assert [piece["text"] for piece in found["She consulted Dr."]["pieces"]] == [
            "She",
            "consulted",
            "Dr",
            ".",
        ]

    def test_a_mark_inside_a_number_stays_put(self) -> None:
        body = client.get(FULL_STOPS).json()
        found = {probe["text"]: probe for probe in body["probes"]}
        pieces = [
            piece["text"] for piece in found["It cost 1,000 and took 3:30."]["pieces"]
        ]

        assert "1,000" in pieces
        assert "3:30" in pieces

    def test_the_six_sentences_lose_five_pieces_when_they_arrive_joined(self) -> None:
        body = client.get(FULL_STOPS).json()

        assert body["sentences"]["separately"] == 72
        assert body["sentences"]["together"] == 67
        assert body["sentences"]["stop_pieces_separately"] == 6
        assert body["sentences"]["stop_pieces_together"] == 1
        assert body["sentences"]["attached_stops_together"] == [
            "Dr.",
            "re-analysis.",
            "room.",
            "arrived.",
            "afternoon.",
            "expected.",
        ]


class TestTheSixSentences:
    def test_the_three_rules_over_the_same_six_sentences(self) -> None:
        view = client.get(CORPUS).json()["notebook"]

        assert view["n_texts"] == 6
        assert view["n_characters"] == 375
        assert view["on_treebank"]["n_pieces"] == 72
        assert view["on_spaces"]["n_pieces"] == 63
        assert view["on_boundaries"]["n_pieces"] == 69

    def test_it_keeps_every_character_that_is_not_a_space(self) -> None:
        view = client.get(CORPUS).json()["notebook"]

        assert view["n_non_space_characters"] == 318
        assert view["on_treebank"]["n_characters_covered"] == 318
        assert view["on_spaces"]["n_characters_covered"] == 318
        assert view["on_boundaries"]["n_characters_covered"] == 303

    def test_the_table_it_would_need_is_one_entry_larger(self) -> None:
        view = client.get(CORPUS).json()["notebook"]

        assert view["on_treebank"]["n_distinct"] == 48
        assert view["on_spaces"]["n_distinct"] == 47
        assert view["on_boundaries"]["n_distinct"] == 47
        assert view["shared_with_spaces"] == 38
        assert view["shared_with_boundaries"] == 41

    def test_the_entries_only_this_rule_has_are_the_marks_and_the_clitic(self) -> None:
        view = client.get(CORPUS).json()["notebook"]

        assert view["only_on_treebank_against_spaces"] == [
            ",",
            ".",
            "2019",
            "afternoon",
            "arrived",
            "away",
            "expected",
            "n't",
            "room",
            "small",
        ]

    def test_the_compounds_are_where_it_parts_from_the_boundary_rule(self) -> None:
        view = client.get(CORPUS).json()["notebook"]

        assert "low-cost" in view["only_on_treebank_against_boundaries"]
        assert "re-analysis" in view["only_on_treebank_against_boundaries"]
        assert "low" in view["only_on_boundaries"]
        assert "cost" in view["only_on_boundaries"]


class TestTheRewritingCensus:
    def test_the_wider_corpus_is_read_by_all_three_rules(self) -> None:
        view = client.get(CORPUS).json()["annotated"]

        assert view["n_texts"] == 22
        assert view["n_characters"] == 888
        assert view["on_treebank"]["n_pieces"] == 266
        assert view["on_spaces"]["n_pieces"] == 155
        assert view["on_boundaries"]["n_pieces"] == 159

    def test_fourteen_of_the_two_hundred_and_sixty_six_pieces_are_rewritten(
        self,
    ) -> None:
        census = client.get(CORPUS).json()["census"]

        assert census["n_pieces"] == 266
        assert census["n_rewritten"] == 14
        assert round(census["share_rewritten"], 4) == 0.0526

    def test_every_rewritten_piece_stands_for_one_quotation_mark(self) -> None:
        census = client.get(CORPUS).json()["census"]

        assert census["by_piece"] == [
            '`` stands for ", 7 times',
            "'' stands for \", 7 times",
        ]

    def test_naming_the_brackets_raises_the_count(self) -> None:
        census = client.get(CORPUS).json()["census"]

        assert census["n_rewritten_naming_brackets"] == 22

    def test_a_third_of_the_pieces_hold_no_letter_and_no_digit(self) -> None:
        census = client.get(CORPUS).json()["census"]

        assert census["n_pieces_without_letter_or_digit"] == 86
        assert census["n_pieces_from_a_cut_host"] == 26


class TestTheRequestIsBounded:
    def test_a_text_longer_than_the_limit_is_refused(self) -> None:
        response = client.post(SPLIT, json={"texts": ["a " * 300]})

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            SPLIT, json={"texts": [RUNNING_SENTENCE], "convert_brackets": True}
        )

        assert response.status_code == 422

    def test_no_texts_at_all_is_refused(self) -> None:
        response = client.post(SPLIT, json={"texts": []})

        assert response.status_code == 422

    def test_more_texts_than_the_limit_is_refused(self) -> None:
        response = client.post(SPLIT, json={"texts": ["a"] * 5})

        assert response.status_code == 422

    def test_an_empty_text_is_read_and_answers_nothing(self) -> None:
        view = split_one("")

        assert view["n_characters"] == 0
        assert view["on_treebank"]["n_pieces"] == 0
        assert view["on_treebank"]["spans_recover_the_source"] is True

    def test_a_text_of_nothing_but_spaces_answers_nothing(self) -> None:
        view = split_one("   ")

        assert view["on_treebank"]["n_pieces"] == 0
        assert view["n_non_space_characters"] == 0


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
    """Every returned string except the corpus and the pieces cut out of it.

    A probe deliberately contains ``a@b.com`` and the treebank's own bracket
    spellings, which are the writing the page is about rather than anything
    naming the code, so the texts the rules read and produced are excluded and
    every label, note and description written here is checked.
    """
    produced = {
        "-LRB-",
        "-RRB-",
        "-LSB-",
        "-RSB-",
        "-LCB-",
        "-RCB-",
        "a@b.com",
        "e-mail",
        "AT&T's",
        "b.com",
        "NYSE",
        "Well-known",
        "well-known",
    }
    return [text for text in _every_string(payload) if text not in produced]


class TestNothingReadableNamesTheCode:
    def test_no_response_carries_an_identifier_or_credits_the_code(self) -> None:
        bodies = [
            client.post(
                SPLIT, json={"texts": [RUNNING_SENTENCE, QUOTED_SENTENCE]}
            ).json(),
            client.get(CONTRACTIONS).json(),
            client.get(FULL_STOPS).json(),
            client.get(CORPUS).json(),
        ]

        for body in bodies:
            for text in _readable_strings(body):
                assert not IDENTIFIER.search(text), text
                for phrase in FORBIDDEN:
                    assert phrase not in text, text

    def test_no_response_carries_an_em_dash(self) -> None:
        for body in (
            client.get(CONTRACTIONS).json(),
            client.get(FULL_STOPS).json(),
            client.get(CORPUS).json(),
        ):
            for text in _every_string(body):
                assert "—" not in text
