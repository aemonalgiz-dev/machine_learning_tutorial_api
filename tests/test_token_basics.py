"""The what-a-token-is endpoints, on the six sentences the page is built from.

Every number the page quotes is pinned here, so a change in the library or in
this module fails a test rather than quietly changing a sentence on the page.
The last class is the one that matters most: nothing these endpoints return may
carry an identifier, because a widget renders these strings.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from oop_ml_api.concepts.token_basics import (
    ACCENTED_TEXT,
    AWKWARD_TEXT,
    NOTEBOOK,
    RUNNING_SENTENCE,
    TWIN_TEXT,
    UNSEEN_TEXT,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

ENCODE = "/concepts/what-a-token-is/encode"
GROWTH = "/concepts/what-a-token-is/vocabulary-growth"
UNKNOWN = "/concepts/what-a-token-is/unknown-answers"


def encode(text: str, scheme: str) -> dict:
    """One text under one scheme, unwrapped to the encoding the page quotes."""
    response = client.post(ENCODE, json={"texts": [text], "scheme": scheme})
    assert response.status_code == 200
    body = response.json()
    return {"table": body, "encoding": body["encodings"][0]}


class TestTheRunningSentence:
    def test_the_character_table_is_thirty_eight_and_the_sentence_fifty_one(
        self,
    ) -> None:
        seen = encode(RUNNING_SENTENCE, "character")

        assert seen["table"]["vocabulary_size"] == 38
        assert seen["table"]["n_pieces_learned"] == 37
        assert seen["encoding"]["n_tokens"] == 51
        assert seen["encoding"]["n_unknown"] == 0

    def test_the_word_table_is_forty_eight_and_the_sentence_seven(self) -> None:
        seen = encode(RUNNING_SENTENCE, "word")

        assert seen["table"]["vocabulary_size"] == 48
        assert seen["table"]["n_pieces_learned"] == 47
        assert seen["encoding"]["n_tokens"] == 7
        assert seen["encoding"]["ids"] == [3, 2, 14, 19, 43, 28, 32]
        assert seen["encoding"]["tokens"][0] == {
            "text": "Dr.",
            "token_id": 3,
            "known": True,
        }

    def test_both_schemes_give_the_sentence_back_exactly(self) -> None:
        for scheme in ("character", "word"):
            seen = encode(RUNNING_SENTENCE, scheme)

            assert seen["encoding"]["decoded"] == RUNNING_SENTENCE
            assert seen["encoding"]["round_trip_exact"] is True

    def test_the_numbers_are_positions_in_the_table(self) -> None:
        seen = encode(RUNNING_SENTENCE, "word")
        table = seen["table"]["vocabulary"]

        assert table[0] == "[UNK]"
        assert [table[token_id] for token_id in seen["encoding"]["ids"]] == [
            "Dr.",
            "Alvarez",
            "didn't",
            "expect",
            "the",
            "low-cost",
            "re-analysis.",
        ]

    def test_the_same_word_in_two_sentences_gets_the_same_number(self) -> None:
        response = client.post(
            ENCODE, json={"texts": [RUNNING_SENTENCE, NOTEBOOK[4]], "scheme": "word"}
        )
        first, second = response.json()["encodings"]

        assert first["ids"][4] == 43
        assert second["ids"][9] == 43
        assert first["tokens"][4]["text"] == second["tokens"][9]["text"] == "the"

    def test_a_capital_letter_makes_a_different_word(self) -> None:
        response = client.post(
            ENCODE, json={"texts": [RUNNING_SENTENCE, NOTEBOOK[4]], "scheme": "word"}
        )
        body = response.json()

        assert body["vocabulary"][6] == "The"
        assert body["vocabulary"][43] == "the"
        assert body["encodings"][1]["ids"][0] == 6

    def test_a_blank_text_becomes_no_words_and_three_characters(self) -> None:
        assert encode("   ", "word")["encoding"]["n_tokens"] == 0
        assert encode("   ", "character")["encoding"]["n_tokens"] == 3


class TestTheEntriesThePageNames:
    def test_the_neighbouring_entries_the_page_quotes(self) -> None:
        table = encode(RUNNING_SENTENCE, "word")["table"]["vocabulary"]

        assert table[29] == "not"
        assert table[30] == "notes"
        assert table[31] == "re-analysis"
        assert table[32] == "re-analysis."

    def test_the_three_pairs_holding_one_word_twice(self) -> None:
        table = encode(RUNNING_SENTENCE, "word")["table"]["vocabulary"]

        assert (table[5], table[39]) == ("She", "she")
        assert (table[6], table[43]) == ("The", "the")
        assert (table[31], table[32]) == ("re-analysis", "re-analysis.")

    def test_the_second_sentence_the_page_prints_in_full(self) -> None:
        response = client.post(ENCODE, json={"texts": [NOTEBOOK[4]], "scheme": "word"})
        encoding = response.json()["encodings"][0]

        assert encoding["ids"] == [6, 15, 47, 40, 44, 26, 47, 27, 42, 43, 17, 39, 20]
        assert encoding["n_unknown"] == 0

    def test_the_first_eighteen_character_numbers(self) -> None:
        seen = encode(RUNNING_SENTENCE, "character")

        assert seen["encoding"]["ids"][:18] == [
            11,
            29,
            5,
            1,
            10,
            24,
            33,
            15,
            29,
            19,
            37,
            1,
            18,
            23,
            18,
            26,
            2,
            31,
        ]
        assert seen["table"]["vocabulary"][1] == " "


class TestGrowth:
    def test_the_word_table_starts_smaller_and_ends_larger(self) -> None:
        steps = client.get(GROWTH).json()["steps"]

        assert [step["character_table"] for step in steps] == [
            25,
            29,
            35,
            37,
            37,
            38,
        ]
        assert [step["word_table"] for step in steps] == [8, 15, 23, 30, 40, 48]

    def test_the_crossover_falls_between_the_fourth_and_fifth_sentence(self) -> None:
        steps = client.get(GROWTH).json()["steps"]

        assert steps[3]["character_table"] > steps[3]["word_table"]
        assert steps[4]["character_table"] < steps[4]["word_table"]

    def test_the_running_totals_are_the_texts_themselves(self) -> None:
        steps = client.get(GROWTH).json()["steps"]

        assert [step["characters_seen"] for step in steps] == [
            51,
            113,
            184,
            244,
            315,
            375,
        ]
        assert [step["words_seen"] for step in steps] == [7, 17, 29, 38, 51, 63]

    def test_a_new_sentence_is_almost_all_new_words_and_almost_no_new_characters(
        self,
    ) -> None:
        steps = client.get(GROWTH).json()["steps"]

        assert steps[4]["next_unknown_characters"] == 1
        assert steps[4]["next_character_tokens"] == 60
        assert steps[4]["next_unknown_words"] == 8
        assert steps[4]["next_word_tokens"] == 12

    def test_the_last_step_has_no_next_sentence(self) -> None:
        last = client.get(GROWTH).json()["steps"][-1]

        assert last["next_text"] is None
        assert last["next_unknown_words"] is None


class TestTheUnseenPiece:
    def test_a_stand_in_makes_two_different_sentences_identical(self) -> None:
        body = client.get(UNKNOWN).json()
        first, second = body["probes"][0], body["probes"][1]

        assert first["text"] == UNSEEN_TEXT
        assert second["text"] == TWIN_TEXT
        assert first["answer"] == second["answer"] == "substituted"
        assert first["ids"] == second["ids"] == [6, 0, 15, 47, 27, 42, 20]
        assert first["decoded"] == second["decoded"]
        assert first["decoded"] == "The [UNK] drift was larger than expected."
        assert first["round_trip_exact"] is False

    def test_a_table_without_a_stand_in_refuses(self) -> None:
        probe = client.get(UNKNOWN).json()["probes"][2]

        assert probe["answer"] == "refused"
        assert probe["ids"] is None
        assert "calibration" in probe["reason"]

    def test_the_closed_table_is_one_smaller_than_the_open_one(self) -> None:
        body = client.get(UNKNOWN).json()

        assert body["table_size"] == 48
        assert body["closed_size"] == 47

    def test_a_character_table_substitutes_the_two_letters_it_never_met(self) -> None:
        probe = client.get(UNKNOWN).json()["probes"][3]

        assert probe["text"] == ACCENTED_TEXT
        assert probe["answer"] == "substituted"
        assert probe["n_tokens"] == 69
        assert probe["decoded"] == (
            "Dr. Alvarez didn't expect the low-cost re-analysis in "
            "[UNK]ngstr[UNK]m units."
        )

    def test_every_byte_value_leaves_nothing_unknown(self) -> None:
        body = client.get(UNKNOWN).json()
        probe = body["probes"][4]

        assert body["byte_size"] == 256
        assert probe["answer"] == "nothing was unknown"
        assert probe["n_tokens"] == 71
        assert probe["round_trip_exact"] is True


class TestTheRoundTrip:
    def test_the_word_scheme_flattens_the_spacing(self) -> None:
        seen = encode(AWKWARD_TEXT, "word")

        assert seen["encoding"]["n_tokens"] == 7
        assert seen["encoding"]["n_unknown"] == 0
        assert seen["encoding"]["round_trip_exact"] is False
        assert seen["encoding"]["decoded"] == RUNNING_SENTENCE

    def test_the_character_scheme_loses_only_the_break_it_never_met(self) -> None:
        seen = encode(AWKWARD_TEXT, "character")

        assert seen["encoding"]["n_tokens"] == 52
        assert seen["encoding"]["n_unknown"] == 1
        assert seen["encoding"]["round_trip_exact"] is False
        assert seen["encoding"]["decoded"] == (
            "Dr.  Alvarez didn't expect[UNK]the low-cost re-analysis."
        )


class TestRefusals:
    def test_no_texts_is_refused(self) -> None:
        response = client.post(ENCODE, json={"texts": [], "scheme": "word"})

        assert response.status_code == 422

    def test_a_scheme_this_page_does_not_have_is_refused(self) -> None:
        response = client.post(ENCODE, json={"texts": ["a"], "scheme": "subword"})

        assert response.status_code == 422

    def test_an_unexpected_field_is_refused(self) -> None:
        response = client.post(
            ENCODE, json={"texts": ["a"], "scheme": "word", "corpus": ["b"]}
        )

        assert response.status_code == 422

    def test_a_text_longer_than_the_limit_is_refused(self) -> None:
        response = client.post(
            ENCODE, json={"texts": ["a" * 401], "scheme": "character"}
        )

        assert response.status_code == 422

    def test_more_texts_than_the_limit_are_refused(self) -> None:
        response = client.post(ENCODE, json={"texts": ["a"] * 9, "scheme": "character"})

        assert response.status_code == 422


CAMEL_CASE = re.compile(r"[a-z][A-Z]")
"""Two words run together, which is what an identifier looks like in prose."""

FORBIDDEN = (
    "oop_ml",
    "the library",
    "the API",
    "unknown_token",
    "Tokenizer",
    "Vocabulary",
    "Error",
)
"""Words and identifiers no string a reader can see may carry."""


def every_string(value: object) -> list[str]:
    """Every string anywhere inside a decoded response body."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in every_string(item)]
    if isinstance(value, list):
        return [text for item in value for text in every_string(item)]
    return []


class TestNothingNamesTheCodeBehindThePage:
    def test_the_refusal_carries_no_identifier(self) -> None:
        reason = client.get(UNKNOWN).json()["probes"][2]["reason"]

        assert CAMEL_CASE.search(reason) is None
        assert reason.startswith("the scheme produced the piece")

    def test_no_endpoint_returns_an_identifier(self) -> None:
        bodies = [
            client.get(UNKNOWN).json(),
            client.get(GROWTH).json(),
            client.post(ENCODE, json={"texts": [UNSEEN_TEXT], "scheme": "word"}).json(),
        ]

        for body in bodies:
            for text in every_string(body):
                if text in NOTEBOOK or text in (UNSEEN_TEXT, TWIN_TEXT):
                    continue
                assert CAMEL_CASE.search(text) is None, text
                for forbidden in FORBIDDEN:
                    assert forbidden not in text, text
