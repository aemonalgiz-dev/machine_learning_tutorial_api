"""The hidden-model segmentation page's endpoints, on the corpora the page carries.

Every number the page quotes in prose is pinned here, so a change underneath that
moved one of them would fail a test rather than quietly rewrite the page.

Four other things are held. The trellis is a second walk over the same tables, so
its winner is checked against the tag sequence the segmenter itself answers with,
on every text the page carries; a trellis that drifted from the recurrence would
fail here rather than reach the page as a plausible-looking grid. The smoothed
probabilities are rebuilt from the counts the module recounted, and checked
against the model's own numbers to the last bit, since the page shows the
fraction rather than the value. The claim that the four places can spell out any
cut of a text is checked against a brute-force enumeration written from the
definition. And the last class asserts that nothing identifier-shaped survives
into any string a widget renders, including into the refusals the page's own
controls can reach.
"""

from __future__ import annotations

import math
import re
from itertools import pairwise, product

import pytest
from fastapi.testclient import TestClient
from oop_ml.core.natural_language_processing.tokenization.segmentation.hidden_markov import (
    ADMISSIBLE_SUCCESSORS,
    FINAL_TAGS,
    INITIAL_TAGS,
    BoundaryTag,
    HiddenMarkovSegmenter,
)

from oop_ml_api.concepts.hidden_markov_words import (
    CORPORA,
    CORPUS_LABELS,
    CorpusKey,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/segmenting-with-a-hidden-model"
SETUP = f"{BASE}/setup"
TABLES = f"{BASE}/tables"
SCENARIOS = f"{BASE}/scenarios"
SMOOTHING = f"{BASE}/smoothing"
SCALE = f"{BASE}/scale"
LIMITS = f"{BASE}/limits"
SEGMENT = f"{BASE}/segment"


def _corpus_for(label: str) -> tuple[tuple[str, ...], ...]:
    """The sentences a scenario was fitted on, found back from the label it carries."""
    for key, written in CORPUS_LABELS.items():
        if written == label:
            return CORPORA[key]
    raise AssertionError(label)


RESEARCH = "研究生命起源"
NAME = "阿尔瓦雷斯研究生命起源"
MINORITY = "研究生很多"
NOVEL = "adcfeb"


def scenario(key: str) -> dict:
    body = client.get(SCENARIOS).json()
    found = [item for item in body["scenarios"] if item["key"] == key]
    assert found, key
    return found[0]


def table(key: str) -> dict:
    body = client.get(TABLES).json()
    found = [item for item in body["tables"] if item["key"] == key]
    assert found, key
    return found[0]


def corpus_of(key: str) -> dict:
    body = client.get(SCALE).json()
    found = [item for item in body["corpora"] if item["key"] == key]
    assert found, key
    return found[0]


class TestEveryEndpointAnswers:
    @pytest.mark.parametrize("path", [SETUP, TABLES, SCENARIOS, SMOOTHING, LIMITS])
    def test_the_quick_endpoints_answer(self, path):
        assert client.get(path).status_code == 200

    def test_the_measured_endpoint_answers(self):
        assert client.get(SCALE).status_code == 200

    def test_the_playground_answers(self):
        response = client.post(
            SEGMENT,
            json={"text": RESEARCH, "corpus": "five sentences", "smoothing": 1.0},
        )

        assert response.status_code == 200
        assert response.json()["tagging"]["words"] == ["研究", "生命", "起源"]


class TestTheTrellisAgreesWithTheSegmenter:
    """The grid the page draws has to be the search the answer came out of."""

    @pytest.mark.parametrize(
        "key", ["research", "student", "novel", "minority", "name", "two-sentence"]
    )
    def test_the_walked_path_is_the_answer(self, key):
        found = scenario(key)
        model = HiddenMarkovSegmenter(smoothing=found["smoothing"]).fit(
            [list(sentence) for sentence in _corpus_for(found["corpus_label"])]
        )

        assert found["tags"] == "".join(
            tag.value for tag in model.tags_of(found["text"])
        )
        assert found["words"] == list(model.split(found["text"]).texts)

    @pytest.mark.parametrize(
        "key", ["research", "student", "novel", "minority", "name", "two-sentence"]
    )
    def test_the_chosen_cell_of_every_column_is_the_tag(self, key):
        found = scenario(key)

        for column, tag in zip(found["columns"], found["tags"], strict=True):
            assert column["tag"] == tag
            chosen = [cell for cell in column["cells"] if cell["on_best_path"]]
            assert len(chosen) == 1
            assert chosen[0]["tag"] == tag

    @pytest.mark.parametrize("key", ["research", "novel", "name"])
    def test_a_place_a_run_cannot_start_in_has_no_score(self, key):
        first = scenario(key)["columns"][0]
        unreachable = {cell["tag"] for cell in first["cells"] if cell["score"] is None}

        assert unreachable == {BoundaryTag.MIDDLE.value, BoundaryTag.END.value}

    def test_the_words_are_a_cut_after_every_ending_place(self):
        """An independent cutter, written from the definition rather than from the code."""
        found = scenario("name")
        pieces: list[str] = []
        current = ""
        for column, tag in zip(found["columns"], found["tags"], strict=True):
            current += column["character"]
            if tag in (BoundaryTag.END.value, BoundaryTag.SINGLE.value):
                pieces.append(current)
                current = ""

        assert current == ""
        assert pieces == found["words"]


class TestTheFourPlaces:
    def test_half_the_steps_cannot_happen(self):
        found = table("two sentences")

        assert found["n_admissible_steps"] == 8
        assert found["n_forbidden_steps"] == 8
        assert sum(1 for step in found["steps"] if not step["admissible"]) == 8

    def test_a_forbidden_step_carries_no_number_at_all(self):
        found = table("five sentences")
        forbidden = [step for step in found["steps"] if not step["admissible"]]

        assert forbidden
        for step in forbidden:
            assert step["probability"] is None
            assert step["log_probability"] is None
            assert step["count"] == 0

    @pytest.mark.parametrize("length", list(range(1, 11)))
    def test_the_places_spell_out_every_cut_and_no_more(self, length):
        """Brute force over all four to the power of the length, from the definition."""
        allowed = 0
        for sequence in product(list(BoundaryTag), repeat=length):
            if sequence[0] not in INITIAL_TAGS or sequence[-1] not in FINAL_TAGS:
                continue
            if all(
                following in ADMISSIBLE_SUCCESSORS[previous]
                for previous, following in pairwise(sequence)
            ):
                allowed += 1

        assert allowed == 2 ** (length - 1)


class TestTheCountsBehindTheProbabilities:
    """Every smoothed number the page shows as a fraction, rebuilt from its counts."""

    @pytest.mark.parametrize("key", ["two sentences", "five sentences"])
    def test_the_starting_fractions_come_from_the_counts(self, key):
        found = table(key)
        smoothing = found["smoothing"]
        admissible = [row for row in found["starts"] if row["admissible"]]

        assert len(admissible) == 2
        for row in admissible:
            expected = (row["count"] + smoothing) / (
                row["total"] + smoothing * len(admissible)
            )
            assert row["probability"] == pytest.approx(expected, abs=0.0, rel=1e-15)

    @pytest.mark.parametrize("key", ["two sentences", "five sentences"])
    def test_the_step_fractions_come_from_the_counts(self, key):
        found = table(key)
        smoothing = found["smoothing"]

        for row in found["steps"]:
            if not row["admissible"]:
                continue
            expected = (row["count"] + smoothing) / (row["total"] + smoothing * 2)
            assert row["probability"] == pytest.approx(expected, abs=0.0, rel=1e-15)

    @pytest.mark.parametrize("key", ["two sentences", "five sentences"])
    def test_the_character_fractions_are_over_the_alphabet_and_one_spare_slot(
        self, key
    ):
        found = table(key)
        smoothing = found["smoothing"]

        assert found["n_slots"] == found["n_alphabet"] + 1
        for row in found["characters"]:
            expected = (row["count"] + smoothing) / (
                row["total"] + smoothing * found["n_slots"]
            )
            assert row["probability"] == pytest.approx(expected, abs=0.0, rel=1e-15)

    @pytest.mark.parametrize("key", ["two sentences", "five sentences"])
    def test_every_place_sums_to_one_over_the_alphabet_and_the_spare(self, key):
        found = table(key)
        floors = {row["tag"]: row["probability"] for row in found["floors"]}

        for tag in BoundaryTag:
            total = (
                sum(
                    row["probability"]
                    for row in found["characters"]
                    if row["tag"] == tag.value
                )
                + floors[tag.value]
            )
            assert total == pytest.approx(1.0)

    def test_the_hand_worked_fractions_of_the_small_corpus(self):
        """The fractions the page prints, worked on paper before any of this existed."""
        found = table("two sentences")
        starts = {row["tag"]: row["probability"] for row in found["starts"]}
        steps = {
            (row["previous"], row["following"]): row["probability"]
            for row in found["steps"]
        }
        characters = {
            (row["tag"], row["character"]): row["probability"]
            for row in found["characters"]
        }

        assert found["alphabet"] == ["a", "b", "c"]
        assert starts["B"] == pytest.approx(3 / 4)
        assert starts["S"] == pytest.approx(1 / 4)
        assert steps[("B", "M")] == pytest.approx(1 / 2)
        assert steps[("B", "E")] == pytest.approx(1 / 2)
        assert steps[("M", "M")] == pytest.approx(1 / 3)
        assert steps[("M", "E")] == pytest.approx(2 / 3)
        assert steps[("E", "B")] == pytest.approx(1 / 3)
        assert steps[("E", "S")] == pytest.approx(2 / 3)
        assert steps[("S", "B")] == pytest.approx(1 / 2)
        assert steps[("S", "S")] == pytest.approx(1 / 2)
        assert characters[("B", "a")] == pytest.approx(1 / 2)
        assert characters[("B", "b")] == pytest.approx(1 / 6)
        assert characters[("S", "c")] == pytest.approx(2 / 5)
        assert characters[("S", "a")] == pytest.approx(1 / 5)

    def test_a_character_never_produced_by_a_place_gets_the_floor(self):
        found = table("two sentences")
        floors = {row["tag"]: row["probability"] for row in found["floors"]}
        characters = {
            (row["tag"], row["character"]): row["probability"]
            for row in found["characters"]
        }

        assert characters[("B", "b")] == pytest.approx(floors["B"])


class TestTheWordNoListHolds:
    """The failure both word-list methods share, on one corpus and one text."""

    def test_the_three_methods_are_taught_the_same_sentences(self):
        found = client.get(SETUP).json()

        assert found["corpus_sentences"] == [["ab", "cd"], ["ab", "ef"], ["cd", "ef"]]
        assert found["corpus_words"] == ["ab", "cd", "ef"]

    def test_neither_word_list_method_finds_a_word_it_was_not_given(self):
        found = client.get(SETUP).json()

        assert found["from_a_greedy_scan"] == list(NOVEL)
        assert found["from_a_word_list"] == list(NOVEL)

    def test_the_tagger_answers_three_words_none_of_them_in_the_corpus(self):
        found = client.get(SETUP).json()

        assert found["from_the_tagger"] == ["ad", "cf", "eb"]
        assert found["from_the_tagger"] == found["reader_reading"]
        assert found["n_new_words"] == 3

    def test_the_surname_the_running_sentence_carries(self):
        found = client.get(SETUP).json()

        assert (
            found["sentence"] == "Dr. Alvarez didn't expect the low-cost re-analysis."
        )
        assert found["name_sentence"] == NAME
        assert found["name_from_a_word_list"] == [
            "阿",
            "尔",
            "瓦",
            "雷",
            "斯",
            "研究",
            "生命",
            "起源",
        ]
        assert found["name_from_the_tagger"] == [
            "阿尔",
            "瓦",
            "雷",
            "斯",
            "研究",
            "生命",
            "起源",
        ]
        assert found["name_from_both"] == found["name_from_the_tagger"]

    def test_no_method_recovers_the_surname_whole(self):
        found = client.get(SETUP).json()

        for answer in (
            found["name_from_a_word_list"],
            found["name_from_the_tagger"],
            found["name_from_both"],
        ):
            assert answer != found["name_reading"]


class TestTheScenariosThePageQuotes:
    def test_the_sentence_with_two_readings(self):
        found = scenario("research")

        assert found["text"] == RESEARCH
        assert found["tags"] == "BEBEBE"
        assert found["words"] == ["研究", "生命", "起源"]
        assert found["total_log_score"] == pytest.approx(-11.750848, abs=1e-6)
        assert found["matches_reader"]

    def test_three_words_the_corpus_never_held(self):
        found = scenario("novel")

        assert found["tags"] == "BEBEBE"
        assert found["words"] == ["ad", "cf", "eb"]
        assert found["n_new_words"] == 3
        assert found["from_a_word_list"] == list(NOVEL)
        assert found["total_log_score"] == pytest.approx(-9.868, abs=1e-3)

    def test_the_sentence_its_own_corpus_contradicts(self):
        found = scenario("minority")

        assert found["text"] == MINORITY
        assert found["tags"] == "BESSS"
        assert found["words"] == ["研究", "生", "很", "多"]
        assert found["from_a_word_list"] == ["研究生", "很", "多"]

    def test_the_surname_in_front_of_the_sentence(self):
        found = scenario("name")

        assert found["tags"] == "BESSSBEBEBE"
        assert found["n_unseen_characters"] == 5
        assert found["n_characters"] == 11

    def test_the_small_corpus_reading_its_own_characters(self):
        found = scenario("two-sentence")

        assert found["text"] == "abcab"
        assert found["tags"] == "BESBE"
        assert found["words"] == ["ab", "c", "ab"]
        assert found["from_a_word_list"] == ["abc", "ab"]


class TestWhatSmoothingDecides:
    def test_the_corpus_disagrees_with_itself_two_to_one(self):
        found = client.get(SMOOTHING).json()

        assert found["n_majority"] == 2
        assert found["n_minority"] == 1
        assert found["majority_reading"] == ["研究", "生", "很", "多"]
        assert found["minority_reading"] == ["研究生", "很", "多"]

    def test_the_usual_smoothing_does_not_reproduce_the_training_sentence(self):
        found = client.get(SMOOTHING).json()
        at_one = next(step for step in found["steps"] if step["smoothing"] == 1.0)

        assert at_one["words"] == found["majority_reading"]
        assert not at_one["reproduces_the_corpus"]

    def test_a_tenth_reproduces_it(self):
        found = client.get(SMOOTHING).json()
        at_a_tenth = next(step for step in found["steps"] if step["smoothing"] == 0.1)

        assert at_a_tenth["words"] == found["minority_reading"]
        assert at_a_tenth["tags"] == "BMESS"

    def test_the_answer_turns_at_a_measured_value(self):
        found = client.get(SMOOTHING).json()
        low, high = found["turns_between"]

        assert low == pytest.approx(0.23806, abs=1e-4)
        assert high == pytest.approx(0.23806, abs=1e-4)
        assert low < high

    def test_every_step_below_the_turn_reproduces_and_every_step_above_does_not(self):
        found = client.get(SMOOTHING).json()
        low, _ = found["turns_between"]

        for step in found["steps"]:
            assert step["reproduces_the_corpus"] == (step["smoothing"] <= low)


class TestHowMuchTaggedTextIsNeeded:
    def test_a_word_list_recovers_almost_nothing_it_was_not_given(self):
        for key in ("characters", "letters"):
            for step in corpus_of(key)["steps"]:
                assert step["word_list"]["recall_on_unseen"] < 0.1

    def test_the_tagger_recovers_a_third_or_more_of_them(self):
        steps = [
            step
            for step in corpus_of("characters")["steps"]
            if step["share_unseen"] > 0.05
        ]

        assert steps
        for step in steps:
            assert step["tagger"]["recall_on_unseen"] > 0.3

    def test_the_tagger_is_the_worse_of_the_two_on_words_the_corpus_held(self):
        for key in ("characters", "letters"):
            for step in corpus_of(key)["steps"]:
                assert (
                    step["tagger"]["recall_on_seen"]
                    < step["word_list"]["recall_on_seen"]
                )

    def test_the_character_language_turns_over_between_thirty_and_forty(self):
        found = corpus_of("characters")

        assert found["turns_between"] == [30, 40]

    def test_the_letters_turn_over_between_eighty_and_a_hundred_and_sixty(self):
        found = corpus_of("letters")

        assert found["turns_between"] == [80, 160]

    def test_the_two_scores_the_page_quotes_on_the_character_language(self):
        steps = {step["n_sentences"]: step for step in corpus_of("characters")["steps"]}

        assert steps[30]["tagger"]["f_measure"] == pytest.approx(0.8659, abs=5e-4)
        assert steps[30]["word_list"]["f_measure"] == pytest.approx(0.8501, abs=5e-4)
        assert steps[320]["tagger"]["f_measure"] == pytest.approx(0.9223, abs=5e-4)
        assert steps[320]["word_list"]["f_measure"] == pytest.approx(1.0, abs=5e-4)

    def test_the_two_scores_the_page_quotes_on_the_letters(self):
        steps = {step["n_sentences"]: step for step in corpus_of("letters")["steps"]}

        assert steps[40]["tagger"]["f_measure"] == pytest.approx(0.5943, abs=5e-4)
        assert steps[40]["word_list"]["f_measure"] == pytest.approx(0.3921, abs=5e-4)
        assert steps[1280]["tagger"]["f_measure"] == pytest.approx(0.5186, abs=5e-4)
        assert steps[1280]["word_list"]["f_measure"] == pytest.approx(0.9808, abs=5e-4)

    def test_the_tagger_stops_improving_on_the_letters_and_falls_back(self):
        steps = {step["n_sentences"]: step for step in corpus_of("letters")["steps"]}

        assert steps[1280]["tagger"]["f_measure"] < steps[320]["tagger"]["f_measure"]

    def test_running_the_two_together_is_never_worse_than_either(self):
        for key in ("characters", "letters"):
            for step in corpus_of(key)["steps"]:
                assert step["together"]["f_measure"] >= step["tagger"]["f_measure"]
                assert step["together"]["f_measure"] >= step["word_list"]["f_measure"]

    def test_a_character_says_twice_as_much_in_one_language_as_in_the_other(self):
        characters = corpus_of("characters")
        letters = corpus_of("letters")

        assert characters["bits_told"] == pytest.approx(0.9207, abs=5e-4)
        assert letters["bits_told"] == pytest.approx(0.4869, abs=5e-4)
        assert characters["tag_bits"] == pytest.approx(1.7518, abs=5e-4)
        assert letters["tag_bits"] == pytest.approx(1.5590, abs=5e-4)

    def test_the_bits_left_are_what_the_character_did_not_settle(self):
        for key in ("characters", "letters"):
            found = corpus_of(key)
            assert found["tag_bits"] - found["bits_left"] == pytest.approx(
                found["bits_told"]
            )

    def test_the_english_example_is_recovered_whole_by_one_method_only(self):
        found = corpus_of("letters")

        assert found["example_from_the_tagger"] == found["example_truth"]
        assert found["example_from_a_word_list"] != found["example_truth"]

    def test_our_own_sentence_defeats_both_of_them(self):
        found = corpus_of("letters")

        assert found["running_text"] == "Dr.Alvarezdidn'texpectthelow-costre-analysis."
        assert found["running_truth"] == [
            "Dr.",
            "Alvarez",
            "didn't",
            "expect",
            "the",
            "low-cost",
            "re-analysis.",
        ]
        assert found["running_from_the_tagger"] != found["running_truth"]
        assert found["running_from_a_word_list"] != found["running_truth"]
        assert len(found["running_from_the_tagger"]) == 13
        assert len(found["running_from_a_word_list"]) == 24

    def test_the_character_language_carries_no_english_sentence(self):
        found = corpus_of("characters")

        assert found["running_text"] is None
        assert found["running_from_the_tagger"] is None


class TestWhereTheCountsSayNothing:
    def test_two_languages_produce_the_identical_model(self):
        found = client.get(LIMITS).json()

        assert found["n_numbers_compared"] == 36
        assert found["n_numbers_equal"] == 36

    def test_the_fit_answers_words_its_own_language_does_not_contain(self):
        found = client.get(LIMITS).json()
        crossed = next(item for item in found["twin_texts"] if item["text"] == "adcb")

        assert crossed["answer"] == ["ad", "cb"]
        assert not crossed["in_the_first_language"]
        assert crossed["in_the_second_language"]

    def test_a_run_of_unseen_characters_is_answered_by_the_steps_alone(self):
        found = client.get(LIMITS).json()
        model = HiddenMarkovSegmenter().fit(
            [list(sentence) for sentence in CORPORA[CorpusKey.FIVE_SENTENCES]]
        )

        assert found["unseen_tags"] == "BESSS"
        for text in found["unseen_texts"]:
            assert "".join(tag.value for tag in model.tags_of(text)) == "BESSS"

    def test_the_run_lengths_the_page_lists(self):
        found = client.get(LIMITS).json()
        tags = {run["n_characters"]: run["tags"] for run in found["unseen_runs"]}

        assert tags[1] == "S"
        assert tags[2] == "BE"
        assert tags[4] == "BEBE"
        assert tags[5] == "BESSS"


class TestTheRefusals:
    def test_a_text_longer_than_the_bound_is_refused(self):
        response = client.post(SEGMENT, json={"text": "研" * 61})

        assert response.status_code == 422

    @pytest.mark.parametrize("smoothing", [0.0, -1.0, 100.0])
    def test_smoothing_outside_the_bounds_is_refused(self, smoothing):
        response = client.post(SEGMENT, json={"text": RESEARCH, "smoothing": smoothing})

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self):
        response = client.post(SEGMENT, json={"text": RESEARCH, "alpha": 1.0})

        assert response.status_code == 422

    def test_an_unknown_corpus_is_refused(self):
        response = client.post(SEGMENT, json={"text": RESEARCH, "corpus": "nothing"})

        assert response.status_code == 422

    @pytest.mark.parametrize("text", ["", "   "])
    def test_a_text_with_nothing_to_tag_is_refused_in_plain_words(self, text):
        response = client.post(SEGMENT, json={"text": text})

        assert response.status_code == 422
        assert response.json()["detail"] == (
            "a text needs at least one character that is not a space"
        )


class TestTheSmallestSmoothingStillAnswers:
    def test_the_playground_reaches_both_readings(self):
        majority = client.post(
            SEGMENT,
            json={"text": MINORITY, "corpus": "contradictory", "smoothing": 1.0},
        ).json()
        minority = client.post(
            SEGMENT,
            json={"text": MINORITY, "corpus": "contradictory", "smoothing": 0.1},
        ).json()

        assert majority["tagging"]["words"] == ["研究", "生", "很", "多"]
        assert minority["tagging"]["words"] == ["研究生", "很", "多"]

    def test_the_tables_come_back_with_the_requested_smoothing(self):
        body = client.post(
            SEGMENT,
            json={"text": RESEARCH, "corpus": "five sentences", "smoothing": 0.25},
        ).json()

        assert body["tables"]["smoothing"] == 0.25

    def test_a_text_with_a_space_in_it_restarts_the_chain(self):
        body = client.post(
            SEGMENT, json={"text": "研究 生命起源", "corpus": "five sentences"}
        ).json()
        tagging = body["tagging"]

        assert tagging["n_characters"] == 6
        assert [column["character"] for column in tagging["columns"]] == list(
            "研究生命起源"
        )
        assert tagging["tags"] == "BEBEBE"
        assert tagging["words"] == ["研究", "生命", "起源"]


IDENTIFIER = re.compile(r"[a-z]+_[a-z]+|[A-Z][a-z]+[A-Z]")
"""Snake case or camel case, either of which would be an identifier reaching a reader."""

FORBIDDEN = (
    "oop_ml",
    "the library",
    "the API",
    "Error",
    "Segmenter",
    "smoothing=",
)


def strings_of(value: object) -> list[str]:
    """Every string anywhere inside a decoded response."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            text
            for key, item in value.items()
            for text in strings_of(item)
            if not key.endswith("_id")
        ]
    if isinstance(value, list):
        return [text for item in value for text in strings_of(item)]
    return []


class TestNothingIdentifierShapedReachesAReader:
    @pytest.mark.parametrize(
        "path", [SETUP, TABLES, SCENARIOS, SMOOTHING, SCALE, LIMITS]
    )
    def test_no_response_carries_an_identifier(self, path):
        for text in strings_of(client.get(path).json()):
            assert not IDENTIFIER.search(text), text
            for word in FORBIDDEN:
                assert word not in text

    def test_no_refusal_carries_an_identifier(self):
        refusals = [
            client.post(SEGMENT, json={"text": ""}),
            client.post(SEGMENT, json={"text": "   "}),
        ]

        for response in refusals:
            detail = response.json()["detail"]
            assert isinstance(detail, str)
            assert not IDENTIFIER.search(detail), detail

    def test_the_playground_answer_carries_none_either(self):
        body = client.post(
            SEGMENT, json={"text": RESEARCH, "corpus": "five sentences"}
        ).json()

        for text in strings_of(body):
            assert not IDENTIFIER.search(text), text


class TestTheCachedWork:
    def test_the_measured_endpoint_is_computed_once(self):
        first = client.get(SCALE).json()
        second = client.get(SCALE).json()

        assert first == second

    def test_the_places_are_the_four_the_page_names(self):
        found = table("five sentences")
        named = {row["tag"]: row["tag_name"] for row in found["starts"]}

        assert named == {
            "B": "begins a word",
            "M": "inside a word",
            "E": "ends a word",
            "S": "a word on its own",
        }

    def test_every_probability_is_the_exponential_of_its_logarithm(self):
        found = table("five sentences")
        rows = (
            [row for row in found["starts"] if row["admissible"]]
            + [row for row in found["steps"] if row["admissible"]]
            + found["characters"]
            + found["floors"]
        )

        for row in rows:
            assert row["probability"] == pytest.approx(math.exp(row["log_probability"]))


class TestTheShapeOfTheTwoLanguages:
    """The counts the page names in prose when it introduces each language."""

    def test_the_character_language(self):
        found = corpus_of("characters")

        assert found["n_words"] == 70
        assert found["n_alphabet"] == 40
        assert found["n_test_sentences"] == 200

    def test_the_letters(self):
        found = corpus_of("letters")

        assert found["n_words"] == 810
        assert found["n_alphabet"] == 26
        assert found["n_test_sentences"] == 200

    def test_the_ceiling_the_page_quotes_for_the_letters(self):
        steps = {step["n_sentences"]: step for step in corpus_of("letters")["steps"]}
        best = max(step["tagger"]["f_measure"] for step in steps.values())

        assert best == pytest.approx(0.6087, abs=5e-4)
        assert steps[320]["tagger"]["f_measure"] == pytest.approx(best)

    def test_the_two_together_scores_the_page_quotes(self):
        characters = {
            step["n_sentences"]: step for step in corpus_of("characters")["steps"]
        }
        letters = {step["n_sentences"]: step for step in corpus_of("letters")["steps"]}

        assert characters[30]["together"]["f_measure"] == pytest.approx(
            0.9262, abs=5e-4
        )
        assert letters[80]["together"]["f_measure"] == pytest.approx(0.7781, abs=5e-4)
        assert letters[80]["tagger"]["f_measure"] == pytest.approx(0.6065, abs=5e-4)

    def test_the_share_of_new_words_at_the_crossing(self):
        found = corpus_of("characters")
        low, _ = found["turns_between"]
        crossing = next(step for step in found["steps"] if step["n_sentences"] == low)

        assert crossing["share_unseen"] == pytest.approx(0.0967, abs=5e-4)
