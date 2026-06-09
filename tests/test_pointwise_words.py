"""The page's endpoints, on the marked-up sentences the page carries.

Every number the page quotes in prose is pinned here, so a change underneath
that moved one of them would fail a test rather than quietly rewrite the page.

Three other things are held. The decomposition of a gap is a second walk over
the same weights, so its total is checked against the score the segmenter itself
reports for that gap, on every gap the page opens; a breakdown that drifted from
the scoring would fail here rather than reach the page as a plausible-looking
list. The count of wrong gaps is checked against a reading rebuilt from the
decisions, since a mark of right or wrong that did not agree with the words
would be worse than no mark at all. And the last class asserts that nothing
identifier-shaped survives into any string a widget renders, including into the
refusals the page's own controls can reach.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient
from oop_ml.core.natural_language_processing.tokenization.segmentation.pointwise import (
    PointwiseSegmenter,
)

from oop_ml_api.concepts.pointwise_words import (
    CORPORA,
    CORPUS_SETTINGS,
    SENTENCE_READING,
    SPACELESS_SENTENCE,
    CorpusKey,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/learning-boundaries-from-examples"
SCENARIOS = f"{BASE}/scenarios"
GAPS = f"{BASE}/gaps"
TRAINING = f"{BASE}/training"
COMPARISON = f"{BASE}/comparison"
LIMITS = f"{BASE}/limits"
SEGMENT = f"{BASE}/segment"


def _scenario(key: str) -> dict:
    body = client.get(SCENARIOS).json()
    for scenario in body["scenarios"]:
        if scenario["key"] == key:
            return scenario
    raise AssertionError(key)


def _opened(key: str) -> dict:
    body = client.get(GAPS).json()
    for gap in body["gaps"]:
        if gap["key"] == key:
            return gap
    raise AssertionError(key)


def _language(key: str) -> dict:
    body = client.get(COMPARISON).json()
    for language in body["languages"]:
        if language["key"] == key:
            return language
    raise AssertionError(key)


def _step(language: dict, n_sentences: int) -> dict:
    for step in language["steps"]:
        if step["n_sentences"] == n_sentences:
            return step
    raise AssertionError(n_sentences)


class TestScenarios:
    def test_every_key_the_page_asks_for_is_there(self):
        keys = {item["key"] for item in client.get(SCENARIOS).json()["scenarios"]}

        assert keys == {
            "sentence",
            "research",
            "surname",
            "unfamiliar",
            "chen",
            "spaced",
            "paired",
            "odd",
        }

    def test_our_own_sentence_has_forty_four_gaps_and_gets_three_wrong(self):
        scenario = _scenario("sentence")

        assert scenario["text"] == SPACELESS_SENTENCE
        assert scenario["n_gaps"] == 44
        assert scenario["n_gaps_wrong"] == 3
        assert scenario["n_features"] == 631
        assert scenario["words"] == [
            "Dr.",
            "Alv",
            "a",
            "rezdidn't",
            "expect",
            "the",
            "low-cost",
            "re-analysis.",
        ]

    def test_the_three_wrong_gaps_are_all_inside_the_surname(self):
        scenario = _scenario("sentence")
        wrong = [gap for gap in scenario["gaps"] if gap["correct"] is False]

        assert [gap["index"] for gap in wrong] == [5, 6, 9]
        assert [gap["before"] for gap in wrong] == ["v", "a", "z"]

    def test_punctuation_sits_left_of_four_gaps_and_only_one_is_a_boundary(self):
        gaps = _scenario("sentence")["gaps"]
        after_punctuation = [gap for gap in gaps if gap["before"] in {".", "'", "-"}]

        assert len(after_punctuation) == 4
        assert sum(gap["expected"] for gap in after_punctuation) == 1

    def test_the_sequence_model_recovers_the_surname_and_loses_the_contraction(self):
        scenario = _scenario("sentence")

        assert "Alvarez" in scenario["from_the_sequence_model"]
        assert "Alvarez" not in scenario["words"]
        assert "didn't" not in scenario["from_the_sequence_model"]
        assert scenario["from_the_sequence_model"] == [
            "Dr.",
            "Alvarez",
            "did",
            "n'te",
            "xpect",
            "the",
            "low-cost",
            "re-anal",
            "ysis.",
        ]

    def test_four_words_of_the_reading_are_absent_from_the_sentences(self):
        scenario = _scenario("sentence")
        held = set(scenario["corpus_words"])

        assert sorted(word for word in SENTENCE_READING if word not in held) == [
            "Alvarez",
            "expect",
            "low-cost",
            "re-analysis.",
        ]
        assert "expect" in scenario["words"]
        assert "low-cost" in scenario["words"]
        assert "re-analysis." in scenario["words"]

    def test_the_five_shared_sentences_answer_the_reading_exactly(self):
        scenario = _scenario("research")

        assert scenario["words"] == ["研究", "生命", "起源"]
        assert scenario["n_gaps_wrong"] == 0
        assert scenario["matches_reader"] is True
        assert scenario["sequence_matches_reader"] is True

    def test_on_the_surname_it_answers_five_pieces_against_the_other_seven(self):
        scenario = _scenario("surname")

        assert scenario["words"] == ["阿尔", "瓦雷斯", "研究", "生命", "起源"]
        assert len(scenario["from_the_sequence_model"]) == 7
        assert scenario["n_gaps_wrong"] == 1

    @pytest.mark.parametrize("key", ["unfamiliar", "chen"])
    def test_a_change_of_script_is_found_where_the_sequence_model_misses_it(self, key):
        scenario = _scenario(key)

        assert scenario["matches_reader"] is True
        assert scenario["sequence_matches_reader"] is False

    def test_what_the_other_method_answers_for_the_unfamiliar_run(self):
        scenario = _scenario("unfamiliar")

        assert scenario["words"] == ["Ortiz", "漢字", "文"]
        assert scenario["from_the_sequence_model"] == ["Ortiz漢", "字文"]

    def test_a_space_is_a_boundary_nobody_is_asked_about(self):
        scenario = _scenario("spaced")

        assert scenario["n_gaps"] == 6
        assert scenario["n_gaps_wrong"] == 0
        assert scenario["n_free_boundaries"] == 1
        assert scenario["matches_reader"] is True
        assert [gap["position"] for gap in scenario["gaps"]] == [0, 1, 2, 5, 6, 7]

    def test_it_is_the_worse_of_the_two_where_every_word_is_two_characters(self):
        scenario = _scenario("paired")

        assert scenario["words"] == ["生命起源"]
        assert scenario["from_the_sequence_model"] == ["生命", "起源"]
        assert scenario["sequence_matches_reader"] is True
        assert scenario["matches_reader"] is False

    def test_the_words_agree_with_the_gaps_that_were_cut(self):
        """The pieces are exactly the run cut where a score came out positive."""
        for scenario in client.get(SCENARIOS).json()["scenarios"]:
            scores = iter(scenario["gaps"])
            rebuilt: list[str] = []
            for run in scenario["text"].split():
                current = run[0]
                for character in run[1:]:
                    if next(scores)["cut"]:
                        rebuilt.append(current)
                        current = ""
                    current += character
                rebuilt.append(current)

            assert rebuilt == scenario["words"], scenario["key"]

    def test_the_marks_agree_with_whether_the_reading_was_reproduced(self):
        for scenario in client.get(SCENARIOS).json()["scenarios"]:
            if scenario["reader_reading"] is None:
                continue

            assert scenario["matches_reader"] == (scenario["n_gaps_wrong"] == 0), (
                scenario["key"]
            )


class TestOpenedGaps:
    def test_the_four_gaps_the_page_opens_are_there(self):
        keys = [gap["key"] for gap in client.get(GAPS).json()["gaps"]]

        assert keys == ["full-stop", "apostrophe", "hyphen", "script"]

    def test_a_breakdown_totals_to_the_score_the_model_reports(self):
        """The decomposition is the score taken apart, not a second opinion."""
        for opened in client.get(GAPS).json()["gaps"]:
            key = CorpusKey(opened["corpus_key"])
            window, epochs, seed = CORPUS_SETTINGS[key]
            model = PointwiseSegmenter(
                window=window, epochs=epochs, random_seed=seed
            ).fit([list(sentence) for sentence in CORPORA[key]])

            assert opened["total"] == model.gap_scores(opened["text"])[opened["gap"]]

    def test_the_three_subtotals_add_back_up_to_the_whole(self):
        for opened in client.get(GAPS).json()["gaps"]:
            total = (
                opened["base_rate_total"]
                + opened["identity_total"]
                + opened["kind_total"]
            )

            assert total == pytest.approx(opened["total"], abs=1e-12)

    def test_the_full_stop_is_cut_by_what_sits_behind_it(self):
        opened = _opened("full-stop")

        assert opened["before"] == "."
        assert opened["after"] == "A"
        assert opened["cut"] is True
        assert opened["expected"] is True
        assert opened["total"] == pytest.approx(7.1283, abs=5e-5)
        assert opened["identity_total"] == pytest.approx(10.4926, abs=5e-5)
        assert opened["n_identity_seen"] == 5
        assert opened["n_identity"] == 11

    @pytest.mark.parametrize("key", ["apostrophe", "hyphen"])
    def test_punctuation_inside_a_word_is_joined(self, key):
        opened = _opened(key)

        assert opened["cut"] is False
        assert opened["expected"] is False
        assert opened["kind_total"] == pytest.approx(-12.0563, abs=5e-5)

    def test_at_a_change_of_script_only_the_kinds_have_been_seen(self):
        opened = _opened("script")

        assert opened["before"] == "z"
        assert opened["after"] == "漢"
        assert opened["n_identity_seen"] == 1
        assert opened["n_kind_seen"] == opened["n_kind"] == 11
        assert opened["total"] == pytest.approx(0.0367, abs=5e-5)
        assert opened["cut"] is True

    def test_the_sentences_contain_no_capital_a_and_no_v_at_all(self):
        """Two of the questions at the full-stop gap could never have carried one."""
        written = "".join(
            word for sentence in CORPORA[CorpusKey.WRITTEN] for word in sentence
        )

        assert "A" not in written
        assert "v" not in written

    def test_the_full_stop_itself_is_the_largest_single_vote(self):
        opened = _opened("full-stop")
        largest = max(opened["votes"], key=lambda vote: vote["weight"])

        assert largest["reads"] == "."
        assert largest["weight"] == pytest.approx(3.6304, abs=5e-5)

    def test_a_kind_feature_is_never_silent_at_any_gap_the_page_opens(self):
        for opened in client.get(GAPS).json()["gaps"]:
            assert opened["n_kind_seen"] == opened["n_kind"]

    def test_the_smallest_fit_there_is(self):
        worked = client.get(GAPS).json()["hand_worked"]

        assert worked["n_features"] == 7
        assert len(worked["features"]) == 7
        assert worked["score_when_cut"] == 7.0
        assert worked["score_when_joined"] == -7.0
        assert worked["words_when_cut"] == ["a", "b"]
        assert worked["words_when_joined"] == ["ab"]
        assert worked["n_updates_by_epoch"] == [1]

    def test_a_gap_in_the_middle_carries_eight_times_the_reach_less_one(self):
        rows = client.get(GAPS).json()["features_by_reach"]

        assert [len(row) for row in rows] == [7, 15, 23]
        for reach, row in enumerate(rows, start=1):
            assert len(row) == 8 * reach - 1

    def test_every_feature_is_one_of_the_three_families(self):
        for opened in client.get(GAPS).json()["gaps"]:
            families = {vote["family"] for vote in opened["votes"]}

            assert families <= {
                "base rate",
                "which character",
                "what kind of character",
            }


class TestTraining:
    def test_the_corpus_holds_a_hundred_and_eighty_nine_marked_gaps(self):
        body = client.get(TRAINING).json()

        assert body["n_marked_gaps"] == 189
        assert body["n_boundaries"] == 34
        assert body["n_features"] == 631

    def test_the_corrections_fall_away_and_reach_zero(self):
        rows = client.get(TRAINING).json()["epoch_rows"]

        assert [row["n_updates"] for row in rows[:6]] == [45, 10, 5, 9, 7, 3]
        assert all(row["n_updates"] == 0 for row in rows[6:])
        assert len(rows) == 20

    def test_eight_orders_give_five_different_readings_and_none_is_the_reading(self):
        body = client.get(TRAINING).json()
        readings = {tuple(row["words"]) for row in body["seed_rows"]}

        assert len(body["seed_rows"]) == 8
        assert len(readings) == 5
        assert body["n_seeds_matching_reader"] == 0

    def test_every_order_settles_within_the_passes_it_was_given(self):
        for row in client.get(TRAINING).json()["seed_rows"]:
            assert 0 < row["n_passes_before_settling"] <= 20
            assert row["n_updates_by_epoch"][-1] == 0


class TestComparison:
    def test_both_languages_are_measured(self):
        keys = [item["key"] for item in client.get(COMPARISON).json()["languages"]]

        assert keys == ["characters", "letters"]

    def test_on_the_character_language_it_is_behind_and_then_ahead(self):
        language = _language("characters")

        assert language["crosses_between"] == [80, 160]
        assert _step(language, 30)["pointwise"]["f_measure"] == pytest.approx(
            0.7974, abs=5e-5
        )
        assert _step(language, 30)["sequence"]["f_measure"] == pytest.approx(
            0.8659, abs=5e-5
        )
        assert _step(language, 320)["pointwise"]["f_measure"] == pytest.approx(
            0.9799, abs=5e-5
        )
        assert _step(language, 320)["sequence"]["f_measure"] == pytest.approx(
            0.9223, abs=5e-5
        )

    def test_the_table_of_marked_gaps_and_weights_the_page_prints(self):
        language = _language("characters")
        printed = [
            (
                step["n_sentences"],
                step["n_marked_gaps"],
                step["n_features"],
                round(step["pointwise"]["f_measure"], 4),
                round(step["sequence"]["f_measure"], 4),
            )
            for step in language["steps"]
            if step["n_sentences"] in {10, 30, 80, 160, 320}
        ]

        assert printed == [
            (10, 89, 487, 0.6244, 0.7811),
            (30, 265, 1048, 0.7974, 0.8659),
            (80, 690, 1686, 0.8795, 0.9060),
            (160, 1361, 2314, 0.9650, 0.9178),
            (320, 2675, 2996, 0.9799, 0.9223),
        ]

    def test_where_it_loses_is_the_words_the_sentences_never_held(self):
        step = _step(_language("characters"), 30)

        assert step["pointwise"]["recall_on_unseen"] == pytest.approx(0.2640, abs=5e-5)
        assert step["sequence"]["recall_on_unseen"] == pytest.approx(0.6560, abs=5e-5)
        assert step["pointwise"]["recall_on_seen"] < step["sequence"]["recall_on_seen"]

    def test_on_an_alphabet_it_climbs_where_the_other_turns_back_down(self):
        language = _language("letters")

        assert _step(language, 80)["pointwise"]["f_measure"] == pytest.approx(
            0.7757, abs=5e-5
        )
        assert _step(language, 320)["sequence"]["f_measure"] == pytest.approx(
            0.6087, abs=5e-5
        )
        assert _step(language, 1280)["pointwise"]["f_measure"] == pytest.approx(
            0.9057, abs=5e-5
        )
        assert _step(language, 1280)["sequence"]["f_measure"] == pytest.approx(
            0.5186, abs=5e-5
        )

    def test_the_other_method_peaks_and_then_falls_on_the_alphabet(self):
        scores = [
            step["sequence"]["f_measure"] for step in _language("letters")["steps"]
        ]

        assert max(scores) == pytest.approx(0.6087, abs=5e-5)
        assert scores[-1] < scores[-3]

    def test_a_partly_marked_corpus_is_worth_far_less_to_the_other_method(self):
        rows = client.get(COMPARISON).json()["annotation"]
        by_share = {row["share_marked"]: row for row in rows}

        assert by_share[0.2]["share_usable_gaps"] == pytest.approx(0.2041, abs=5e-5)
        assert by_share[0.2]["share_known_places"] == pytest.approx(0.0779, abs=5e-5)
        assert by_share[0.4]["share_usable_gaps"] == pytest.approx(0.3949, abs=5e-5)
        assert by_share[0.4]["share_known_places"] == pytest.approx(0.1961, abs=5e-5)
        assert by_share[1.0]["share_usable_gaps"] == 1.0
        assert by_share[1.0]["share_known_places"] == 1.0

    def test_a_share_of_marked_gaps_is_never_worth_more_to_the_other_method(self):
        for row in client.get(COMPARISON).json()["annotation"]:
            assert row["share_known_places"] <= row["share_usable_gaps"] + 1e-12

    def test_a_feature_naming_a_kind_is_never_silent_and_one_naming_a_character_is(
        self,
    ):
        body = client.get(COMPARISON).json()

        assert body["kind_share_seen"] == 1.0
        assert body["identity_share_seen"] == pytest.approx(0.7933, abs=5e-5)
        assert body["n_features_counted"] == 31566


class TestLimits:
    def test_the_reach_needed_is_exactly_the_distance_to_the_deciding_character(self):
        body = client.get(LIMITS).json()

        assert body["windows"] == [1, 2, 3, 4, 5]
        smallest = {
            row["n_filler"]: row["smallest_window_that_works"]
            for row in body["window_rows"]
        }
        assert smallest == {0: 3, 1: 4, 2: 5, 3: None}
        for row in body["window_rows"]:
            assert row["offset"] == -(row["n_filler"] + 3)

    def test_nothing_below_the_needed_reach_answers_both_texts(self):
        for row in client.get(LIMITS).json()["window_rows"]:
            needed = row["smallest_window_that_works"]
            for reach, right in zip([1, 2, 3, 4, 5], row["n_right_by_window"]):
                if needed is None or reach < needed:
                    assert right < row["n_texts"]
                else:
                    assert right == row["n_texts"]

    def test_the_answer_of_independent_parts_scores_below_the_whole_sequence_one(self):
        rows = client.get(LIMITS).json()["coherence_rows"]
        by_name = {row["name"]: row for row in rows}

        assert by_name["one answer per gap"]["sequence_log_score"] == pytest.approx(
            -155.7756, abs=5e-4
        )
        assert by_name["the best whole sequence"][
            "sequence_log_score"
        ] == pytest.approx(-151.0253, abs=5e-4)
        assert by_name["what a reader answers"]["sequence_log_score"] == pytest.approx(
            -153.9847, abs=5e-4
        )

    def test_the_whole_sequence_answer_is_the_best_of_the_three_by_its_own_score(self):
        rows = client.get(LIMITS).json()["coherence_rows"]
        best = max(rows, key=lambda row: row["sequence_log_score"])

        assert best["is_the_sequence_answer"] is True

    def test_a_piece_of_odd_length_is_not_something_the_other_method_refuses(self):
        body = client.get(LIMITS).json()

        assert body["n_four_character_texts"] == 1296
        assert body["n_odd_pointwise"] == 32
        assert body["n_odd_sequence"] == 45
        assert body["n_odd_sequence"] > body["n_odd_pointwise"]

    def test_the_stranded_character_the_page_shows(self):
        body = client.get(LIMITS).json()

        assert body["odd_from_pointwise"] == ["命起源", "生"]
        assert body["odd_from_the_sequence_model"] == ["命起", "源生"]
        assert all(len(word) == 2 for word in body["paired_words"])


class TestSegment:
    def test_a_text_comes_back_decided_gap_by_gap(self):
        response = client.post(
            SEGMENT,
            json={"text": "研究生命起源", "corpus": "five sentences"},
        )

        assert response.status_code == 200
        assert response.json()["scenario"]["words"] == ["研究", "生命", "起源"]
        assert response.json()["opened"] is None

    def test_a_chosen_gap_comes_back_opened_up(self):
        response = client.post(
            SEGMENT,
            json={"text": "研究生命起源", "corpus": "five sentences", "gap": 1},
        )
        body = response.json()

        assert body["opened"]["gap"] == 1
        assert body["opened"]["total"] == body["scenario"]["gaps"][1]["score"]

    def test_a_gap_in_the_second_run_of_a_spaced_text_is_opened_correctly(self):
        response = client.post(
            SEGMENT,
            json={"text": "研究生命 起源研究", "corpus": "five sentences", "gap": 4},
        )
        body = response.json()

        assert body["opened"]["before"] == "源"
        assert body["opened"]["total"] == body["scenario"]["gaps"][4]["score"]

    def test_a_wider_reach_changes_how_many_features_a_gap_carries(self):
        narrow = client.post(
            SEGMENT, json={"text": "研究生命起源", "window": 1, "gap": 2}
        ).json()
        wide = client.post(
            SEGMENT, json={"text": "研究生命起源", "window": 3, "gap": 2}
        ).json()

        assert len(narrow["opened"]["votes"]) == 7
        assert len(wide["opened"]["votes"]) == 23

    def test_a_text_of_nothing_but_spaces_is_refused(self):
        response = client.post(SEGMENT, json={"text": "   "})

        assert response.status_code == 422
        assert "space" in response.json()["detail"]

    def test_an_unknown_keyword_is_refused(self):
        assert client.post(SEGMENT, json={"text": "ab", "reach": 2}).status_code == 422

    @pytest.mark.parametrize(
        "keywords",
        [{"window": 0}, {"window": 99}, {"epochs": 0}, {"random_seed": -1}],
    )
    def test_a_setting_outside_its_bounds_is_refused(self, keywords):
        response = client.post(SEGMENT, json={"text": "ab", **keywords})

        assert response.status_code == 422

    def test_a_text_longer_than_the_limit_is_refused(self):
        response = client.post(SEGMENT, json={"text": "a" * 200})

        assert response.status_code == 422


IDENTIFIER = re.compile(r"[a-z]+_[a-z]+|\b[A-Z][a-z]+[A-Z][A-Za-z]*\b|oop_ml")
"""What an identifier looks like, so a message that grows one fails here."""

FEATURE_KEYS = {"feature", "features", "reach"}
"""The fields that carry a feature's own spelling, which the page prints as code."""


def _free_text(value: object, key: str | None = None) -> list[str]:
    """Every string a widget might render, bar the feature spellings themselves."""
    if isinstance(value, str):
        return [] if key in FEATURE_KEYS else [value]
    if isinstance(value, dict):
        return [
            found
            for inner_key, inner in value.items()
            for found in _free_text(inner, inner_key)
        ]
    if isinstance(value, list):
        return [found for inner in value for found in _free_text(inner, key)]
    return []


class TestNothingIsNamedThatAReaderHasNoUseFor:
    @pytest.mark.parametrize("path", [SCENARIOS, GAPS, TRAINING, COMPARISON, LIMITS])
    def test_no_identifier_survives_into_anything_a_widget_renders(self, path):
        for text in _free_text(client.get(path).json()):
            assert not IDENTIFIER.search(text), text

    def test_no_identifier_survives_into_a_refusal(self):
        for payload in ({"text": "   "}, {"text": "　"}):
            response = client.post(SEGMENT, json=payload)

            assert response.status_code == 422
            assert not IDENTIFIER.search(str(response.json()["detail"]))
