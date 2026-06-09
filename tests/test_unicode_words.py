"""The Unicode word boundary endpoints, on the sentence and the six-sentence notebook.

Every number the page quotes is pinned here, so a change in the library or in
this module fails a test rather than quietly changing a sentence on the page.
The last class is the one that matters most: nothing these endpoints return may
carry an identifier, because a widget renders these strings.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from oop_ml_api.concepts.unicode_words import (
    NOTEBOOK,
    RUNNING_SENTENCE,
    SHORT_TEXT,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

SPLIT = "/concepts/unicode-word-boundaries/split"
CLASSES = "/concepts/unicode-word-boundaries/classes"
SCRIPTS = "/concepts/unicode-word-boundaries/scripts"
COST = "/concepts/unicode-word-boundaries/cost"

JAPANESE = "アルバレス博士は驚いた"
CHINESE = "我来到北京清华大学"


def split_one(text: str) -> dict:
    """One text put to the rules, unwrapped to the view the page quotes."""
    response = client.post(SPLIT, json={"texts": [text]})
    assert response.status_code == 200
    return response.json()["texts"][0]


class TestTheRunningSentence:
    def test_the_sentence_becomes_nine_words(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["on_boundaries"]["n_pieces"] == 9
        assert [piece["text"] for piece in view["on_boundaries"]["pieces"]] == [
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

    def test_the_space_rule_finds_seven_on_the_same_sentence(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["on_spaces"]["n_pieces"] == 7
        assert [piece["text"] for piece in view["on_spaces"]["pieces"]] == [
            "Dr.",
            "Alvarez",
            "didn't",
            "expect",
            "the",
            "low-cost",
            "re-analysis.",
        ]

    def test_every_word_carries_the_span_it_was_cut_from(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert [
            (piece["start"], piece["end"]) for piece in view["on_boundaries"]["pieces"]
        ] == [
            (0, 2),
            (4, 11),
            (12, 18),
            (19, 25),
            (26, 29),
            (30, 33),
            (34, 38),
            (39, 41),
            (42, 50),
        ]

    def test_the_boundary_rule_keeps_fewer_characters_than_the_space_rule(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["n_characters"] == 51
        assert view["on_boundaries"]["n_characters_in_pieces"] == 41
        assert view["on_spaces"]["n_characters_in_pieces"] == 45

    def test_neither_glue_recovers_the_sentence_under_the_boundary_rule(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["on_boundaries"]["glue_is_exact"] is False
        assert (
            view["on_boundaries"]["glued"]
            == "Dr Alvarez didn't expect the low cost re analysis"
        )
        assert view["on_spaces"]["glue_is_exact"] is True

    def test_the_segments_tile_the_sentence(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["n_segments"] == 19
        assert view["segments_tile_the_text"] is True
        assert "".join(segment["text"] for segment in view["segments"]) == (
            RUNNING_SENTENCE
        )

    def test_ten_of_the_nineteen_segments_are_words(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert sum(1 for segment in view["segments"] if segment["is_word"]) == 9

    def test_keeping_punctuation_adds_the_marks_back_as_words(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["n_keeping_punctuation"] == 13
        assert view["keeping_punctuation"] == [
            "Dr",
            ".",
            "Alvarez",
            "didn't",
            "expect",
            "the",
            "low",
            "-",
            "cost",
            "re",
            "-",
            "analysis",
            ".",
        ]


class TestTheCasesThePageQuotes:
    def test_two_spellings_of_one_word_are_both_one_word(self) -> None:
        precomposed = split_one("résumé")
        combining = split_one("résumé")

        assert precomposed["n_characters"] == 6
        assert combining["n_characters"] == 8
        assert precomposed["on_boundaries"]["pieces"][0]["text"] == "résumé"
        assert combining["on_boundaries"]["n_pieces"] == 1
        assert precomposed["on_boundaries"]["n_pieces"] == 1

    def test_an_abbreviation_keeps_its_inner_stop_and_loses_the_other_two(self) -> None:
        view = split_one("The U.S. sensors arrived.")

        assert [piece["text"] for piece in view["on_boundaries"]["pieces"]] == [
            "The",
            "U.S",
            "sensors",
            "arrived",
        ]
        assert [piece["text"] for piece in view["on_spaces"]["pieces"]] == [
            "The",
            "U.S.",
            "sensors",
            "arrived.",
        ]

    def test_a_colon_holds_letters_and_separates_a_time(self) -> None:
        assert [
            piece["text"] for piece in split_one("a:b")["on_boundaries"]["pieces"]
        ] == ["a:b"]
        assert [
            piece["text"] for piece in split_one("12:30")["on_boundaries"]["pieces"]
        ] == ["12", "30"]

    def test_a_low_line_holds_a_name_together(self) -> None:
        view = split_one("snake_case_name")

        assert view["on_boundaries"]["n_pieces"] == 1


class TestOneClassPerCharacter:
    def test_every_character_of_the_short_text_carries_a_class(self) -> None:
        view = split_one(SHORT_TEXT)

        assert view["n_characters"] == 18
        assert len(view["characters"]) == 18

    def test_the_apostrophe_and_the_stop_are_told_apart(self) -> None:
        view = split_one(SHORT_TEXT)
        by_character = {entry["character"]: entry for entry in view["characters"]}

        assert by_character["'"]["description"] == "an apostrophe"
        assert (
            by_character["."]["description"]
            == "may stand between two letters or two digits"
        )
        assert by_character["D"]["description"] == "a letter"
        assert by_character[" "]["description"] == "a space"

    def test_the_breaks_fall_where_the_words_begin(self) -> None:
        view = split_one(SHORT_TEXT)

        assert [
            position
            for position, entry in enumerate(view["characters"])
            if entry["breaks_before"]
        ] == [2, 3, 4, 11, 12]

    def test_the_code_point_is_written_the_way_a_character_table_writes_it(
        self,
    ) -> None:
        view = split_one("a")

        assert view["characters"][0]["code_point"] == "U+0061"


class TestTheClasses:
    def test_the_table_holds_eighteen_classes_and_six_sentences_reach_seven(
        self,
    ) -> None:
        body = client.get(CLASSES).json()

        assert body["n_classes"] == 18
        assert body["n_classes_used"] == 7

    def test_the_census_accounts_for_every_character_of_the_corpus(self) -> None:
        body = client.get(CLASSES).json()

        assert body["n_characters"] == 375
        assert sum(entry["count"] for entry in body["census"]) == 375

    def test_letters_and_spaces_are_the_two_commonest_classes(self) -> None:
        body = client.get(CLASSES).json()

        assert body["census"][0]["description"] == "a letter"
        assert body["census"][0]["count"] == 298
        assert body["census"][1]["description"] == "a space"
        assert body["census"][1]["count"] == 57

    def test_the_colon_and_the_comma_disagree_in_opposite_directions(self) -> None:
        body = client.get(CLASSES).json()
        by_name = {probe["name"]: probe for probe in body["marks"]}

        assert by_name["colon"]["joins_letters"] is True
        assert by_name["colon"]["joins_digits"] is False
        assert by_name["comma"]["joins_letters"] is False
        assert by_name["comma"]["joins_digits"] is True

    def test_the_hyphen_separates_where_the_low_line_joins(self) -> None:
        body = client.get(CLASSES).json()
        by_name = {probe["name"]: probe for probe in body["marks"]}

        assert by_name["hyphen-minus"]["between_letters"] == ["a", "b"]
        assert by_name["hyphen-minus"]["description"] == "none of the above"
        assert by_name["low line"]["between_letters"] == ["a_b"]
        assert (
            by_name["low line"]["description"]
            == "joins whatever it finds on both sides"
        )

    def test_a_narrow_space_holds_two_digits_together(self) -> None:
        body = client.get(CLASSES).json()
        by_name = {probe["name"]: probe for probe in body["marks"]}

        assert by_name["narrow no-break space"]["joins_digits"] is True
        assert by_name["space"]["joins_digits"] is False

    def test_the_two_columns_and_their_disagreements_are_counted(self) -> None:
        body = client.get(CLASSES).json()

        assert body["n_marks_joining_letters"] == 7
        assert body["n_marks_joining_digits"] == 6
        assert body["n_marks_disagreeing"] == 3


class TestScriptsWithoutSpaces:
    def test_the_japanese_sentence_is_seven_words_against_one(self) -> None:
        body = client.get(SCRIPTS).json()
        probe = body["probes"][0]

        assert probe["text"] == JAPANESE
        assert probe["n_characters"] == 11
        assert probe["n_boundary_pieces"] == 7
        assert probe["n_space_pieces"] == 1
        assert probe["boundary_pieces"] == [
            "アルバレス",
            "博",
            "士",
            "は",
            "驚",
            "い",
            "た",
        ]

    def test_the_chinese_line_becomes_one_word_per_character(self) -> None:
        body = client.get(SCRIPTS).json()
        probe = next(item for item in body["probes"] if item["text"] == CHINESE)

        assert probe["n_boundary_pieces"] == 9
        assert probe["n_characters"] == 9
        assert probe["n_space_pieces"] == 1

    def test_thai_is_cut_at_every_character_too(self) -> None:
        body = client.get(SCRIPTS).json()
        probe = next(item for item in body["probes"] if item["text"] == "ภาษาไทย")

        assert probe["n_boundary_pieces"] == 7
        assert probe["n_characters"] == 7

    def test_korean_is_read_the_way_the_space_rule_reads_it(self) -> None:
        body = client.get(SCRIPTS).json()
        probe = next(item for item in body["probes"] if item["text"] == "한국어 단어")

        assert probe["boundary_pieces"] == probe["space_pieces"] == ["한국어", "단어"]

    def test_a_thousands_group_written_with_a_narrow_space_stays_whole(self) -> None:
        body = client.get(SCRIPTS).json()
        probe = next(item for item in body["probes"] if "sensors" in item["text"])

        assert probe["boundary_pieces"] == ["1 000", "sensors"]
        assert probe["space_pieces"] == ["1", "000", "sensors"]

    def test_two_flags_are_two_segments_and_no_words(self) -> None:
        body = client.get(SCRIPTS).json()
        probe = next(item for item in body["probes"] if item["label"] == "two flags")

        assert probe["n_boundary_pieces"] == 0
        assert probe["n_space_pieces"] == 1

    def test_the_lone_hand_is_no_words_at_all(self) -> None:
        body = client.get(SCRIPTS).json()
        probe = next(item for item in body["probes"] if item["text"] == "\U0001f44d")

        assert probe["n_boundary_pieces"] == 0
        assert probe["boundary_covered"] == 0
        assert probe["n_space_pieces"] == 1
        assert probe["space_covered"] == 1

    def test_an_address_loses_the_character_that_makes_it_an_address(self) -> None:
        body = client.get(SCRIPTS).json()
        probe = next(item for item in body["probes"] if item["label"] == "an address")

        assert probe["boundary_pieces"] == ["alvarez", "example.com"]
        assert probe["boundary_covered"] == 18
        assert probe["space_covered"] == 19

    def test_a_file_name_is_cut_at_the_hyphen_and_the_stop_but_not_the_low_line(
        self,
    ) -> None:
        body = client.get(SCRIPTS).json()
        probe = next(item for item in body["probes"] if item["label"] == "a file name")

        assert probe["boundary_pieces"] == ["re", "analysis_2019", "csv"]
        assert probe["n_space_pieces"] == 1


class TestTheCorpus:
    def test_the_six_sentences_are_the_ones_the_page_quotes(self) -> None:
        body = client.get(COST).json()

        assert body["corpus"] == list(NOTEBOOK)
        assert body["n_characters"] == 375

    def test_both_rules_come_to_the_same_number_of_entries(self) -> None:
        body = client.get(COST).json()

        assert body["on_boundaries"]["n_distinct"] == 47
        assert body["on_spaces"]["n_distinct"] == 47
        assert body["n_shared_entries"] == 35

    def test_the_twelve_each_way_are_the_same_words_wearing_marks(self) -> None:
        body = client.get(COST).json()

        assert body["only_on_boundaries"] == [
            "2019",
            "Dr",
            "afternoon",
            "arrived",
            "away",
            "cost",
            "expected",
            "low",
            "ran",
            "re",
            "room",
            "small",
        ]
        assert body["only_on_spaces"] == [
            "2019,",
            "Dr.",
            "afternoon.",
            "arrived.",
            "away.",
            "expected.",
            "low-cost",
            "re-analysis",
            "re-analysis.",
            "re-ran",
            "room.",
            "small,",
        ]

    def test_exactly_one_word_appears_here_in_two_spellings(self) -> None:
        """The claim the page makes about why both tables come to 47.

        Separating a mark shrinks a table only where the same word turns up both
        with the mark and without it, and over six sentences that happens once.
        """
        body = client.get(COST).json()
        marked = set(body["only_on_spaces"])
        pairs = sorted(
            word
            for word in marked
            if any(other != word and other.strip(".,-") == word for other in marked)
        )

        assert pairs == ["re-analysis"]

    def test_the_boundary_rule_reads_more_pieces(self) -> None:
        body = client.get(COST).json()

        assert body["on_spaces"]["n_pieces"] == 63
        assert body["on_boundaries"]["n_pieces"] == 69

    def test_the_boundary_rule_leaves_more_characters_out(self) -> None:
        body = client.get(COST).json()

        assert body["on_boundaries"]["n_characters_covered"] == 303
        assert body["on_spaces"]["n_characters_covered"] == 318
        assert round(body["on_boundaries"]["share_covered"], 4) == 0.808
        assert round(body["on_spaces"]["share_covered"], 4) == 0.848

    def test_the_segments_and_the_kept_marks_are_counted(self) -> None:
        body = client.get(COST).json()

        assert body["n_segments"] == 141
        assert body["n_pieces_keeping_punctuation"] == 84
        assert body["n_distinct_keeping_punctuation"] == 50


class TestTheRequestIsBounded:
    def test_a_text_longer_than_the_limit_is_refused(self) -> None:
        response = client.post(SPLIT, json={"texts": ["a " * 300]})

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            SPLIT, json={"texts": [RUNNING_SENTENCE], "keep_punctuation": True}
        )

        assert response.status_code == 422

    def test_no_texts_at_all_is_refused(self) -> None:
        response = client.post(SPLIT, json={"texts": []})

        assert response.status_code == 422

    def test_more_texts_than_the_limit_is_refused(self) -> None:
        response = client.post(SPLIT, json={"texts": ["a"] * 7})

        assert response.status_code == 422

    def test_an_empty_text_is_read_and_answers_nothing(self) -> None:
        view = split_one("")

        assert view["n_characters"] == 0
        assert view["on_boundaries"]["n_pieces"] == 0
        assert view["n_segments"] == 0
        assert view["segments_tile_the_text"] is True


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
    """Every returned string except the corpus and the probe texts themselves.

    A probe deliberately contains ``a_b``, ``alvarez@example.com`` and
    ``re-analysis_2019.csv``, which are the writing the page is about rather than
    anything naming the code, so the words the rules produced are excluded and
    every label, description and note written here is checked.
    """
    produced = {"a_b", "1_2", "analysis_2019", "re-analysis_2019.csv"}
    return [text for text in _every_string(payload) if text not in produced]


class TestNothingReadableNamesTheCode:
    def test_no_response_carries_an_identifier_or_credits_the_code(self) -> None:
        bodies = [
            client.post(SPLIT, json={"texts": [RUNNING_SENTENCE, SHORT_TEXT]}).json(),
            client.get(CLASSES).json(),
            client.get(SCRIPTS).json(),
            client.get(COST).json(),
        ]

        for body in bodies:
            for text in _readable_strings(body):
                assert not IDENTIFIER.search(text), text
                for phrase in FORBIDDEN:
                    assert phrase not in text, text

    def test_no_response_carries_an_em_dash(self) -> None:
        for body in (
            client.get(CLASSES).json(),
            client.get(SCRIPTS).json(),
            client.get(COST).json(),
        ):
            for text in _every_string(body):
                assert "—" not in text
