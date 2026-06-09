"""The bytes-and-characters endpoints, on the texts the page is built from.

Every number the page quotes is pinned here, so a change in the library or in
this module fails a test rather than quietly changing a sentence on the page.
The last class is the one that matters most: nothing these endpoints return may
carry an identifier, because a widget renders these strings.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from oop_ml_api.concepts.bytes_and_characters import (
    ACCENTED_TEXT,
    CORPUS,
    GREEK_TEXT,
    LANGUAGES,
    RUNNING_SENTENCE,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

READINGS = "/concepts/bytes-and-characters/readings"
SCRIPTS = "/concepts/bytes-and-characters/scripts"
GROWTH = "/concepts/bytes-and-characters/alphabet-growth"
BYTE_RUNS = "/concepts/bytes-and-characters/byte-runs"
LIMITS = "/concepts/bytes-and-characters/limits"


def read(text: str) -> dict:
    """One text under all three readings, keyed by which reading it was."""
    response = client.post(READINGS, json={"texts": [text]})
    assert response.status_code == 200
    body = response.json()
    seen = body["texts"][0]
    return {
        "tables": body,
        "text": seen,
        **{reading["reading"]: reading for reading in seen["readings"]},
    }


class TestTheThreeTables:
    def test_the_byte_table_holds_every_byte_value(self) -> None:
        assert read(RUNNING_SENTENCE)["tables"]["byte_table"] == 256

    def test_the_character_table_is_the_corpus_plus_one_stand_in(self) -> None:
        tables = read(RUNNING_SENTENCE)["tables"]

        assert tables["n_distinct_characters"] == 36
        assert tables["character_table"] == 37

    def test_the_learned_table_is_the_one_the_merging_page_describes(self) -> None:
        tables = read(RUNNING_SENTENCE)["tables"]

        assert tables["learned_table"] == 137
        assert tables["n_merges"] == 85

    def test_the_corpus_is_eighteen_sentences(self) -> None:
        assert len(CORPUS) == 18
        assert RUNNING_SENTENCE not in CORPUS


class TestTheRunningSentence:
    def test_it_is_the_same_length_read_either_way(self) -> None:
        seen = read(RUNNING_SENTENCE)

        assert seen["text"]["n_characters"] == 51
        assert seen["text"]["n_bytes"] == 51
        assert seen["bytes"]["n_tokens"] == 51
        assert seen["characters"]["n_tokens"] == 51

    def test_both_readings_give_it_back_exactly(self) -> None:
        seen = read(RUNNING_SENTENCE)

        assert seen["bytes"]["n_unseen"] == 0
        assert seen["bytes"]["exact"] is True
        assert seen["characters"]["n_unseen"] == 0
        assert seen["characters"]["exact"] is True

    def test_the_learned_reading_is_shorter_and_loses_a_word_boundary(self) -> None:
        learned = read(RUNNING_SENTENCE)["learned"]

        assert learned["n_tokens"] == 25
        assert learned["n_unseen"] == 1
        assert learned["exact"] is False
        assert learned["decoded"] == (
            "Dr. Alvare[UNK]didn't expect the low-cost re-analysis."
        )

    def test_the_first_byte_numbers_are_the_bytes_themselves(self) -> None:
        byte_reading = read(RUNNING_SENTENCE)["bytes"]

        assert byte_reading["ids"][:5] == [68, 114, 46, 32, 65]
        assert byte_reading["pieces"][3]["text"] == "Ġ"
        assert byte_reading["pieces"][3]["token_id"] == 32


class TestTwoLettersTheCorpusNeverUsed:
    def test_the_byte_reading_gives_it_back_exactly(self) -> None:
        seen = read(ACCENTED_TEXT)

        assert seen["text"]["n_characters"] == 69
        assert seen["text"]["n_bytes"] == 71
        assert seen["bytes"]["n_tokens"] == 71
        assert seen["bytes"]["exact"] is True

    def test_the_character_reading_substitutes_twice(self) -> None:
        characters = read(ACCENTED_TEXT)["characters"]

        assert characters["n_tokens"] == 69
        assert characters["n_unseen"] == 2
        assert characters["exact"] is False
        assert "[UNK]ngstr[UNK]m" in characters["decoded"]

    def test_the_learned_reading_substitutes_three_times(self) -> None:
        learned = read(ACCENTED_TEXT)["learned"]

        assert learned["n_tokens"] == 38
        assert learned["n_unseen"] == 3


class TestAScriptTheCorpusNeverMet:
    def test_only_the_byte_reading_gives_it_back(self) -> None:
        seen = read(GREEK_TEXT)

        assert seen["text"]["n_characters"] == 21
        assert seen["bytes"]["n_tokens"] == 38
        assert seen["bytes"]["exact"] is True

    def test_the_character_reading_loses_seventeen_of_twenty_one(self) -> None:
        characters = read(GREEK_TEXT)["characters"]

        assert characters["n_tokens"] == 21
        assert characters["n_unseen"] == 17
        assert characters["exact"] is False

    def test_the_learned_reading_loses_the_spaces_as_well(self) -> None:
        learned = read(GREEK_TEXT)["learned"]

        assert learned["n_tokens"] == 18
        assert learned["n_unseen"] == 17
        assert " " not in learned["decoded"]


class TestOneSentenceInTwelveLanguages:
    def test_there_are_twelve_of_them(self) -> None:
        rows = client.get(SCRIPTS).json()["rows"]

        assert len(rows) == len(LANGUAGES) == 12
        assert rows[0]["language"] == "English"

    def test_every_one_of_them_comes_back_exactly_under_bytes(self) -> None:
        rows = client.get(SCRIPTS).json()["rows"]

        assert all(row["byte_exact"] for row in rows)

    def test_the_ratio_runs_from_one_to_three(self) -> None:
        rows = {row["language"]: row for row in client.get(SCRIPTS).json()["rows"]}

        assert rows["English"]["bytes_per_character"] == 1.0
        assert rows["Greek"]["n_characters"] == 21
        assert rows["Greek"]["n_bytes"] == 38
        assert rows["Hindi"]["n_characters"] == 18
        assert rows["Hindi"]["n_bytes"] == 48
        assert rows["Chinese"]["bytes_per_character"] == 3.0
        assert rows["Japanese"]["bytes_per_character"] == 3.0

    def test_the_shortest_sentence_in_bytes_is_not_the_english_one(self) -> None:
        rows = {row["language"]: row for row in client.get(SCRIPTS).json()["rows"]}

        assert rows["Chinese"]["n_bytes"] == 18
        assert rows["English"]["n_bytes"] == 24
        assert rows["Hindi"]["n_bytes"] == 48

    def test_a_short_reading_can_be_an_empty_one(self) -> None:
        rows = {row["language"]: row for row in client.get(SCRIPTS).json()["rows"]}

        assert rows["English"]["learned_tokens"] == 8
        assert rows["English"]["learned_unseen"] == 0
        assert rows["Chinese"]["learned_tokens"] == 6
        assert rows["Chinese"]["learned_unseen"] == 6


class TestTheTableThatGrowsAndTheTableThatCannot:
    def test_the_character_table_never_stops_growing(self) -> None:
        steps = client.get(GROWTH).json()["steps"]

        assert len(steps) == 13
        assert steps[0]["character_table"] == 36
        assert steps[-1]["character_table"] == 117
        assert steps[-1]["label"] == "Chinese"

    def test_the_byte_table_is_the_same_at_every_step(self) -> None:
        steps = client.get(GROWTH).json()["steps"]

        assert {step["byte_table"] for step in steps} == {256}

    def test_most_byte_values_are_never_touched(self) -> None:
        steps = client.get(GROWTH).json()["steps"]

        assert steps[0]["byte_values_used"] == 36
        assert steps[-1]["byte_values_used"] == 103


class TestARunOfNumbersThatIsNotText:
    def test_half_the_numbers_stand_for_nothing_on_their_own(self) -> None:
        body = client.get(BYTE_RUNS).json()

        assert body["table_size"] == 256
        assert body["single_ids_that_are_text"] == 128
        assert body["single_ids_that_are_not"] == 128

    def test_one_accented_letter_is_two_numbers_and_neither_alone(self) -> None:
        body = client.get(BYTE_RUNS).json()

        assert body["accented_ids"] == [195, 169]
        assert body["accented_whole"] == "é"
        assert body["accented_first_alone"] == "�"
        assert body["accented_second_alone"] == "�"

    def test_a_drawn_run_is_almost_never_text(self) -> None:
        drawn = {
            run["length"]: run for run in client.get(BYTE_RUNS).json()["random_runs"]
        }

        assert drawn[2]["n_trials"] == 2000
        assert drawn[2]["n_text"] == 546
        assert drawn[4]["n_text"] == 209
        assert drawn[8]["n_text"] == 24
        assert drawn[16]["n_text"] == 0

    def test_half_the_stopping_points_in_the_greek_run_are_not_text(self) -> None:
        runs = {run["language"]: run for run in client.get(BYTE_RUNS).json()["runs"]}

        assert runs["Greek"]["n_ids"] == 38
        assert runs["Greek"]["n_broken"] == 17
        assert runs["English"]["n_ids"] == 24
        assert runs["English"]["n_broken"] == 0

    def test_the_first_greek_number_alone_is_not_a_letter(self) -> None:
        greek = next(
            run for run in client.get(BYTE_RUNS).json()["runs"] if run["n_ids"] == 38
        )

        assert greek["prefixes"][0]["is_text"] is False
        assert greek["prefixes"][1]["is_text"] is True
        assert greek["prefixes"][1]["decoded"] == "Η"


class TestWhatTheClosingPartQuotes:
    def test_a_row_for_every_codepoint_is_not_affordable(self) -> None:
        body = client.get(LIMITS).json()
        tables = {table["label"]: table for table in body["tables"]}

        assert body["codepoints_in_unicode"] == 1114112
        assert body["embedding_width"] == 768
        assert tables["one row per codepoint"]["numbers"] == 855638016
        assert tables["every byte value"]["numbers"] == 196608
        assert tables["the characters the corpus used"]["numbers"] == 28416
        assert tables["pieces merged from the corpus"]["numbers"] == 105216

    def test_most_byte_values_already_have_a_printable_shape(self) -> None:
        body = client.get(LIMITS).json()

        assert body["bytes_that_keep_their_own_shape"] == 188
        assert body["bytes_given_a_borrowed_shape"] == 68

    def test_a_character_takes_one_to_four_bytes(self) -> None:
        widths = {
            sample["label"]: sample["n_bytes"]
            for sample in client.get(LIMITS).json()["widths"]
        }

        assert widths["a Latin letter"] == 1
        assert widths["an accented Latin letter"] == 2
        assert widths["a Greek letter"] == 2
        assert widths["a Han character"] == 3
        assert widths["a raised thumb"] == 4

    def test_one_mark_can_be_several_characters(self) -> None:
        clusters = {
            sample["label"]: sample for sample in client.get(LIMITS).json()["clusters"]
        }

        assert clusters["a raised thumb"]["n_characters"] == 1
        assert clusters["the same thumb, tinted"]["n_characters"] == 2
        assert clusters["a family"]["n_characters"] == 5
        assert clusters["a family"]["n_bytes"] == 18
        assert clusters["the flag of Wales"]["n_characters"] == 7
        assert clusters["the flag of Wales"]["n_bytes"] == 28
        assert all(sample["byte_exact"] for sample in clusters.values())

    def test_two_spellings_that_look_alike_are_different_texts(self) -> None:
        body = client.get(LIMITS).json()
        first, second = body["spellings"]

        assert body["spellings_equal"] is False
        assert first["n_characters"] == 4
        assert first["n_bytes"] == 5
        assert second["n_characters"] == 5
        assert second["n_bytes"] == 6
        assert second["codepoints"][-1] == "U+0301"
        assert first["byte_exact"] is True
        assert second["byte_exact"] is True

    def test_a_text_with_nothing_in_it_becomes_no_numbers(self) -> None:
        cases = {
            case["label"]: case for case in client.get(LIMITS).json()["empty_cases"]
        }

        assert cases["nothing at all"]["byte_tokens"] == 0
        assert cases["nothing at all"]["character_tokens"] == 0
        assert cases["three spaces"]["byte_tokens"] == 3
        assert cases["three spaces"]["character_tokens"] == 3

    def test_a_number_outside_the_table_names_nothing(self) -> None:
        body = client.get(LIMITS).json()

        assert body["out_of_range_refusal"] == (
            "no entry has id 256; ids run from 0 to 255"
        )

    def test_not_every_run_of_codepoints_is_text(self) -> None:
        assert client.get(LIMITS).json()["lone_surrogate_has_a_byte_spelling"] is False

    def test_the_corpus_costs_the_same_read_either_way(self) -> None:
        body = client.get(LIMITS).json()

        assert body["corpus_bytes"] == 763
        assert body["corpus_characters"] == 763
        assert body["corpus_learned"] == 263

    def test_the_english_corpus_touches_thirty_six_byte_values(self) -> None:
        body = client.get(LIMITS).json()

        assert body["corpus_byte_values_used"] == 36
        assert body["all_texts_byte_values_used"] == 103


class TestTheBoundsOnARequest:
    def test_a_text_longer_than_the_limit_is_refused(self) -> None:
        response = client.post(READINGS, json={"texts": ["a" * 301]})

        assert response.status_code == 422

    def test_too_many_texts_are_refused(self) -> None:
        response = client.post(READINGS, json={"texts": ["a"] * 7})

        assert response.status_code == 422

    def test_an_unexpected_field_is_refused(self) -> None:
        response = client.post(READINGS, json={"texts": ["a"], "scheme": "word"})

        assert response.status_code == 422

    def test_a_run_of_codepoints_that_is_not_text_is_refused_at_the_door(self) -> None:
        response = client.post(
            READINGS,
            content=b'{"texts": ["\\ud800"]}',
            headers={"Content-Type": "application/json"},
        )

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

SOURCE_TEXTS = frozenset(
    {RUNNING_SENTENCE, ACCENTED_TEXT, GREEK_TEXT}
    | set(CORPUS)
    | {sentence for _, sentence in LANGUAGES}
)
"""The texts themselves, which are quoted back and are not the page's own words."""


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
        sentence = client.get(LIMITS).json()["out_of_range_refusal"]

        assert CAMEL_CASE.search(sentence) is None
        for forbidden in FORBIDDEN:
            assert forbidden not in sentence

    def test_no_endpoint_returns_an_identifier(self) -> None:
        bodies = [
            client.get(SCRIPTS).json(),
            client.get(GROWTH).json(),
            client.get(BYTE_RUNS).json(),
            client.get(LIMITS).json(),
            client.post(
                READINGS, json={"texts": [RUNNING_SENTENCE, GREEK_TEXT]}
            ).json(),
        ]

        for body in bodies:
            for text in every_string(body):
                if text in SOURCE_TEXTS or any(
                    source.startswith(text) for source in SOURCE_TEXTS
                ):
                    continue
                assert CAMEL_CASE.search(text) is None, text
                for forbidden in FORBIDDEN:
                    assert forbidden not in text, text
