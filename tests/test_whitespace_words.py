"""The splitting-on-spaces endpoints, on the sentence and the six-sentence notebook.

Every number the page quotes is pinned here, so a change in the library or in
this module fails a test rather than quietly changing a sentence on the page.
The last class is the one that matters most: nothing these endpoints return may
carry an identifier, because a widget renders these strings.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from oop_ml_api.concepts.whitespace_words import (
    NOTEBOOK,
    ODD_SPACING,
    RUNNING_SENTENCE,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

SPLIT = "/concepts/splitting-on-spaces/split"
SURVEY = "/concepts/splitting-on-spaces/survey"
COST = "/concepts/splitting-on-spaces/cost"

CHINESE = "我来到北京清华大学"


def split_one(text: str) -> dict:
    """One text put to the rule, unwrapped to the view the page quotes."""
    response = client.post(SPLIT, json={"texts": [text]})
    assert response.status_code == 200
    return response.json()["texts"][0]


class TestTheRunningSentence:
    def test_the_sentence_becomes_seven_pieces(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["n_pieces"] == 7
        assert [piece["text"] for piece in view["pieces"]] == [
            "Dr.",
            "Alvarez",
            "didn't",
            "expect",
            "the",
            "low-cost",
            "re-analysis.",
        ]

    def test_every_piece_carries_the_span_it_was_cut_from(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert [(piece["start"], piece["end"]) for piece in view["pieces"]] == [
            (0, 3),
            (4, 11),
            (12, 18),
            (19, 25),
            (26, 29),
            (30, 38),
            (39, 51),
        ]
        for piece in view["pieces"]:
            assert RUNNING_SENTENCE[piece["start"] : piece["end"]] == piece["text"]

    def test_the_pieces_cover_forty_five_of_the_fifty_one_characters(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["n_characters"] == 51
        assert view["n_characters_in_pieces"] == 45
        assert round(view["share_covered"], 4) == 0.8824

    def test_the_six_characters_left_out_are_the_six_spaces(self) -> None:
        view = split_one(RUNNING_SENTENCE)
        between = [gap for gap in view["gaps"] if gap]

        assert between == [" "] * 6
        assert all(gap.isspace() for gap in between)

    def test_single_spacing_is_what_makes_the_glue_exact(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["glued"] == RUNNING_SENTENCE
        assert view["glue_is_exact"] is True
        assert view["rebuild_is_exact"] is True


class TestSpacingTheRuleNeverHeld:
    def test_the_doubled_space_and_the_line_break_give_the_same_seven_pieces(
        self,
    ) -> None:
        odd = split_one(ODD_SPACING)
        plain = split_one(RUNNING_SENTENCE)

        assert odd["n_pieces"] == 7
        assert [piece["text"] for piece in odd["pieces"]] == [
            piece["text"] for piece in plain["pieces"]
        ]

    def test_gluing_with_single_spaces_is_not_the_text_that_went_in(self) -> None:
        odd = split_one(ODD_SPACING)

        assert odd["glue_is_exact"] is False
        assert odd["glued"] == RUNNING_SENTENCE
        assert odd["n_characters"] == 52
        assert odd["n_characters_in_pieces"] == 45

    def test_the_spans_recover_the_text_the_glue_could_not(self) -> None:
        odd = split_one(ODD_SPACING)

        assert odd["rebuilt"] == ODD_SPACING
        assert odd["rebuild_is_exact"] is True

    def test_the_gaps_are_a_doubled_space_and_a_line_break(self) -> None:
        odd = split_one(ODD_SPACING)

        assert [gap for gap in odd["gaps"] if gap] == [
            "  ",
            " ",
            " ",
            "\n",
            " ",
            " ",
        ]


class TestTextWithNothingToCut:
    def test_three_spaces_become_no_pieces_and_nothing_is_refused(self) -> None:
        view = split_one("   ")

        assert view["n_pieces"] == 0
        assert view["pieces"] == []
        assert view["share_covered"] == 0.0

    def test_a_script_with_no_spaces_becomes_one_piece(self) -> None:
        view = split_one(CHINESE)

        assert view["n_pieces"] == 1
        assert view["pieces"][0]["text"] == CHINESE
        assert view["n_characters_in_pieces"] == 9
        assert view["share_covered"] == 1.0
        assert view["glue_is_exact"] is True


class TestWhatCountsAsASpace:
    def test_seven_of_the_ten_characters_end_a_word(self) -> None:
        body = client.get(SURVEY).json()

        assert body["n_separators_that_end_a_word"] == 7
        assert len(body["separators"]) == 10

    def test_the_ordinary_space_and_the_ideographic_space_both_end_a_word(self) -> None:
        probes = {
            probe["code_point"]: probe
            for probe in client.get(SURVEY).json()["separators"]
        }

        assert probes["U+0020"]["ends_a_word"] is True
        assert probes["U+3000"]["ends_a_word"] is True
        assert probes["U+00A0"]["ends_a_word"] is True
        assert probes["U+2029"]["ends_a_word"] is True

    def test_the_zero_width_space_does_not_end_a_word_despite_its_name(self) -> None:
        probes = {
            probe["code_point"]: probe
            for probe in client.get(SURVEY).json()["separators"]
        }

        assert probes["U+200B"]["ends_a_word"] is False
        assert probes["U+200B"]["n_pieces"] == 1
        assert probes["U+00AD"]["ends_a_word"] is False
        assert probes["U+002D"]["ends_a_word"] is False


class TestTheWritingItLeavesAlone:
    def test_there_are_seven_probes_and_the_last_is_the_one_without_spaces(
        self,
    ) -> None:
        breakages = client.get(SURVEY).json()["breakages"]

        assert len(breakages) == 7
        assert breakages[-1]["n_pieces"] == 1
        assert breakages[-1]["pieces"] == [CHINESE]

    def test_punctuation_stays_attached_to_the_word_in_front_of_it(self) -> None:
        breakages = client.get(SURVEY).json()["breakages"]

        assert breakages[0]["pieces"] == [
            "The",
            "drift",
            "did",
            "not",
            "go",
            "away.",
        ]
        assert breakages[1]["pieces"][3] == "small,"
        assert breakages[2]["pieces"][2] == '"low-cost"'
        assert breakages[2]["pieces"][3] == "(twice)"

    def test_the_abbreviation_and_the_sentence_end_come_out_the_same_way(self) -> None:
        breakages = client.get(SURVEY).json()["breakages"]

        assert breakages[3]["pieces"] == [
            "Dr.",
            "Alvarez",
            "ran",
            "the",
            "re-analysis.",
        ]

    def test_the_contraction_and_the_compound_are_left_whole(self) -> None:
        breakages = client.get(SURVEY).json()["breakages"]

        assert "didn't" in breakages[4]["pieces"]
        assert breakages[5]["pieces"] == ["the", "low-cost", "re-analysis"]

    def test_every_probe_names_a_piece_it_actually_produced(self) -> None:
        for probe in client.get(SURVEY).json()["breakages"]:
            assert probe["look_at"] in probe["pieces"]
            assert probe["n_pieces"] == len(probe["pieces"])


class TestWhatTheNotebookCosts:
    def test_the_six_sentences_give_forty_seven_entries_and_sixty_three_pieces(
        self,
    ) -> None:
        body = client.get(COST).json()

        assert body["corpus"] == list(NOTEBOOK)
        assert body["n_characters"] == 375
        assert body["on_spaces"]["n_distinct"] == 47
        assert body["on_spaces"]["n_pieces"] == 63

    def test_nine_entries_carry_a_mark_and_seven_of_them_a_full_stop(self) -> None:
        body = client.get(COST).json()

        assert body["n_entries_with_edge_punctuation"] == 9
        assert body["entries_ending_in_a_full_stop"] == [
            "Dr.",
            "afternoon.",
            "arrived.",
            "away.",
            "expected.",
            "re-analysis.",
            "room.",
        ]

    def test_exactly_one_entry_is_a_word_the_list_already_holds(self) -> None:
        body = client.get(COST).json()

        assert body["duplicated_entries"] == ["re-analysis."]
        assert body["n_distinct_after_stripping"] == 46

    def test_the_other_rule_reaches_the_same_count_by_a_different_route(self) -> None:
        body = client.get(COST).json()

        assert body["on_word_boundaries"]["n_distinct"] == 47
        assert body["n_shared_entries"] == 35
        assert len(body["only_on_spaces"]) == 12
        assert len(body["only_on_word_boundaries"]) == 12
        assert "low-cost" in body["only_on_spaces"]
        assert "low" in body["only_on_word_boundaries"]
        assert "cost" in body["only_on_word_boundaries"]

    def test_splitting_on_spaces_gives_the_shorter_run_of_pieces(self) -> None:
        body = client.get(COST).json()

        assert body["on_spaces"]["n_pieces"] == 63
        assert body["on_word_boundaries"]["n_pieces"] == 69

    def test_the_other_rule_leaves_ten_characters_of_the_sentence_behind(self) -> None:
        body = client.get(COST).json()

        assert body["on_spaces"]["running_covered"] == 45
        assert body["on_word_boundaries"]["running_covered"] == 41
        assert body["on_word_boundaries"]["running_pieces"] == [
            "Dr",
            "Alvarez",
            "didn't",
            "expect",
            "the",
            "low",
            "cost",
            "re",
            "analysis",
        ]
        assert body["on_word_boundaries"]["running_glue_is_exact"] is False
        assert (
            body["on_word_boundaries"]["running_glued"]
            == "Dr Alvarez didn't expect the low cost re analysis"
        )


class TestTheRequestIsBounded:
    def test_a_text_longer_than_the_limit_is_refused(self) -> None:
        response = client.post(SPLIT, json={"texts": ["a " * 300]})

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            SPLIT, json={"texts": [RUNNING_SENTENCE], "separator": " "}
        )

        assert response.status_code == 422

    def test_no_texts_at_all_is_refused(self) -> None:
        response = client.post(SPLIT, json={"texts": []})

        assert response.status_code == 422


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


class TestNothingReadableNamesTheCode:
    def test_no_response_carries_an_identifier_or_credits_the_code(self) -> None:
        bodies = [
            client.post(SPLIT, json={"texts": [RUNNING_SENTENCE, ODD_SPACING]}).json(),
            client.get(SURVEY).json(),
            client.get(COST).json(),
        ]

        for body in bodies:
            for text in _every_string(body):
                assert not IDENTIFIER.search(text), text
                for phrase in FORBIDDEN:
                    assert phrase not in text, text

    def test_no_response_carries_an_em_dash(self) -> None:
        for body in (client.get(SURVEY).json(), client.get(COST).json()):
            for text in _every_string(body):
                assert "—" not in text
