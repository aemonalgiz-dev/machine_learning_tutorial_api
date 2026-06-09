"""The n-grams page's endpoints, on the corpora it carries.

The package these endpoints call has no spec of its own, so every number the
page quotes is pinned here and nowhere else. A change under the model that moved
one of them fails a test rather than quietly rewriting a paragraph.

The refusal tests are the other half: the page may not show a parameter name,
and the last class asserts that nothing identifier-shaped survives the
rewriting or reaches a label a widget prints.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/n-grams"
GRAMS = f"{BASE}/grams"
COUNTS = f"{BASE}/counts"
SMOOTHING = f"{BASE}/smoothing"
SCORES = f"{BASE}/scores"
GENERATION = f"{BASE}/generation"
FIT = f"{BASE}/fit"


def framing_at(body: dict, order: int) -> dict:
    return next(entry for entry in body["framings"] if entry["order"] == order)


def row_at(rows: list[dict], order: int) -> dict:
    return next(row for row in rows if row["order"] == order)


def answer_for(body: dict, label: str, strength: float) -> dict:
    return next(
        entry
        for entry in body["answers"]
        if entry["label"] == label and entry["strength"] == strength
    )


class TestGrams:
    def test_the_running_sentence_is_seven_words_however_it_is_cut(self) -> None:
        body = client.get(GRAMS).json()

        assert body["n_words"] == 7
        assert body["words"][0] == "Dr."
        assert body["words"][-1] == "re-analysis."
        for order in (1, 2, 3, 4):
            assert framing_at(body, order)["n_windows"] == 8

    def test_the_framing_grows_by_one_start_marker_per_order(self) -> None:
        body = client.get(GRAMS).json()

        assert framing_at(body, 1)["framed"][0] == "Dr."
        assert framing_at(body, 3)["framed"][:2] == ["<s>", "<s>"]
        assert framing_at(body, 4)["n_start_markers"] == 3
        for order in (1, 2, 3, 4):
            assert framing_at(body, order)["framed"][-1] == "</s>"

    def test_the_first_window_of_a_trigram_reading_is_two_markers_and_a_word(
        self,
    ) -> None:
        windows = framing_at(client.get(GRAMS).json(), 3)["windows"]

        assert windows[0] == ["<s>", "<s>", "Dr."]
        assert windows[-1] == ["low-cost", "re-analysis.", "</s>"]

    def test_the_training_text_is_forty_four_sentences_and_sixty_four_words(
        self,
    ) -> None:
        body = client.get(GRAMS).json()

        assert body["n_training_sentences"] == 44
        assert body["n_word_occurrences"] == 326
        assert body["n_distinct_words"] == 62
        assert body["vocabulary_size"] == 64

    def test_the_share_of_possible_runs_that_occurred_collapses_with_width(
        self,
    ) -> None:
        rows = client.get(GRAMS).json()["sparsity"]

        assert row_at(rows, 1)["n_seen"] == 63
        assert row_at(rows, 2)["n_seen"] == 141
        assert row_at(rows, 2)["n_possible"] == 4096
        assert row_at(rows, 3)["n_seen"] == 202
        assert row_at(rows, 3)["n_possible"] == 262144
        assert row_at(rows, 5)["n_seen"] == 248
        assert row_at(rows, 5)["n_possible"] == 1073741824

    def test_more_than_half_the_distinct_triples_occurred_exactly_once(self) -> None:
        rows = client.get(GRAMS).json()["sparsity"]

        assert row_at(rows, 1)["n_seen_once"] == 9
        assert row_at(rows, 2)["n_seen_once"] == 41
        assert row_at(rows, 3)["n_seen_once"] == 109
        assert row_at(rows, 3)["share_seen_once"] == 109 / 202
        for order in (1, 2, 3, 4, 5):
            assert row_at(rows, order)["n_occurrences"] == 370

    def test_two_fifths_of_a_new_text_is_triples_that_never_occurred(self) -> None:
        rows = client.get(GRAMS).json()["novelty"]

        assert row_at(rows, 1)["n_unseen"] == 3
        assert row_at(rows, 2)["n_unseen"] == 10
        assert row_at(rows, 3)["n_unseen"] == 21
        assert row_at(rows, 5)["n_unseen"] == 33
        for order in (1, 2, 3, 4, 5):
            assert row_at(rows, order)["n_runs"] == 53
        assert round(row_at(rows, 3)["share_unseen"], 4) == 0.3962

    def test_the_table_grows_with_the_text_and_never_saturates(self) -> None:
        rows = client.get(GRAMS).json()["growth"]

        assert [row["n_sentences"] for row in rows] == [11, 22, 33, 44]
        assert [row["n_word_occurrences"] for row in rows] == [78, 170, 242, 326]
        assert [row["n_bigram_types"] for row in rows] == [54, 90, 118, 141]
        assert [row["n_trigram_types"] for row in rows] == [62, 115, 156, 202]
        assert round(rows[0]["test_perplexity"], 4) == 52.0291
        assert round(rows[-1]["test_perplexity"], 4) == 8.9626

    def test_knowing_nothing_scores_the_size_of_the_vocabulary(self) -> None:
        rows = client.get(GRAMS).json()["growth"]

        assert [row["vocabulary_size"] for row in rows] == [36, 48, 59, 64]
        assert round(rows[0]["knowing_nothing_perplexity"], 4) == 35.9998
        assert round(rows[-1]["knowing_nothing_perplexity"], 4) == 63.9975
        for row in rows:
            assert (
                abs(row["knowing_nothing_perplexity"] - row["vocabulary_size"]) < 0.01
            )

    def test_the_smallest_model_is_beaten_by_knowing_nothing(self) -> None:
        rows = client.get(GRAMS).json()["growth"]

        assert rows[0]["test_perplexity"] > rows[0]["knowing_nothing_perplexity"]
        assert rows[-1]["test_perplexity"] < rows[-1]["knowing_nothing_perplexity"]

    def test_a_smaller_vocabulary_calls_more_of_the_held_out_text_unknown(
        self,
    ) -> None:
        rows = client.get(GRAMS).json()["growth"]

        assert [row["n_unknown"] for row in rows] == [15, 9, 5, 3]
        assert round(rows[0]["unknown_share"], 4) == 0.283
        assert round(rows[-1]["unknown_share"], 4) == 0.0566


class TestCounts:
    def test_three_sentences_give_nine_words_and_fifteen_predicted_positions(
        self,
    ) -> None:
        body = client.get(COUNTS).json()

        assert body["vocabulary_size"] == 9
        assert body["vocabulary"][:2] == ["</s>", "<unk>"]
        assert body["n_predicted_positions"] == 15
        assert len(body["bigrams"]) == 10

    def test_the_end_marker_is_counted_but_never_stands_before_anything(self) -> None:
        unigrams = {row["word"]: row for row in client.get(COUNTS).json()["unigrams"]}

        assert unigrams["</s>"]["count"] == 3
        assert unigrams["</s>"]["n_times_followed"] == 0
        assert unigrams["was"]["count"] == 3
        assert unigrams["was"]["n_times_followed"] == 3
        assert "<s>" not in unigrams

    def test_the_seven_hand_checked_probabilities(self) -> None:
        rows = client.get(COUNTS).json()["probabilities"]
        by_pair = {(row["context"], row["word"]): row for row in rows}

        assert by_pair[("<s>", "the")]["unsmoothed"] == 2 / 3
        assert by_pair[("<s>", "the")]["smoothed"] == 3 / 12
        assert by_pair[("the", "report")]["unsmoothed"] == 1.0
        assert by_pair[("the", "report")]["smoothed"] == 3 / 11
        assert by_pair[("the", "summary")]["count"] == 0
        assert by_pair[("the", "summary")]["unsmoothed"] == 0.0
        assert by_pair[("the", "summary")]["smoothed"] == 1 / 11
        assert by_pair[("was", "short")]["unsmoothed"] == 1 / 3

    def test_a_sentence_the_corpus_holds_and_two_it_does_not(self) -> None:
        rows = client.get(COUNTS).json()["sentences"]
        by_text = {row["text"]: row for row in rows}

        held = by_text["the report was late"]
        assert held["n_predicted"] == 5
        assert held["unsmoothed_probability"] == 4 / 9
        assert round(held["unsmoothed_perplexity"], 4) == 1.1761
        assert round(held["smoothed_perplexity"], 4) == 3.7965

        for text in ("the summary was late", "a report was short"):
            assert by_text[text]["unsmoothed_is_finite"] is False
            assert by_text[text]["unsmoothed_perplexity"] is None
        assert (
            round(by_text["the summary was late"]["smoothed_perplexity"], 4) == 5.0321
        )
        assert round(by_text["a report was short"]["smoothed_perplexity"], 4) == 5.8064

    def test_the_running_sentence_walked_word_by_word(self) -> None:
        body = client.get(COUNTS).json()
        walk = body["walk"]

        assert body["walk_strength"] == 0.005
        assert len(walk) == 8
        assert walk[0]["context"] == "<s>"
        assert walk[0]["count"] == 7
        assert walk[0]["context_total"] == 44
        assert walk[1]["word"] == "Alvarez"
        assert walk[1]["count"] == 7
        assert walk[1]["context_total"] == 7
        assert round(walk[1]["probability"], 4) == 0.957
        assert [step["is_unknown"] for step in walk] == [
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
        ]
        assert walk[3]["context_total"] == 0
        assert walk[3]["probability"] == 1 / 64
        assert body["walk_n_unknown"] == 2
        assert body["walk_n_predicted"] == 8
        assert round(body["walk_perplexity"], 4) == 16.0254
        assert round(body["walk_cross_entropy"], 4) == 4.0023


class TestSmoothing:
    def test_the_best_constant_is_far_below_one_at_every_width_but_the_first(
        self,
    ) -> None:
        sweeps = client.get(SMOOTHING).json()["sweeps"]

        assert row_at(sweeps, 1)["best_strength"] == 0.2
        assert row_at(sweeps, 2)["best_strength"] == 0.005
        assert row_at(sweeps, 3)["best_strength"] == 0.01
        assert row_at(sweeps, 4)["best_strength"] == 0.02
        assert row_at(sweeps, 5)["best_strength"] == 0.02

    def test_the_bigram_winner_beats_adding_one_by_a_factor_of_two_and_a_half(
        self,
    ) -> None:
        bigram = row_at(client.get(SMOOTHING).json()["sweeps"], 2)

        assert round(bigram["best_tuning_perplexity"], 4) == 4.0702
        assert round(bigram["best_test_perplexity"], 4) == 8.9626
        assert round(bigram["laplace_tuning_perplexity"], 4) == 18.064
        assert round(bigram["laplace_test_perplexity"], 4) == 23.6179

    def test_the_choice_does_not_transfer_at_the_narrowest_width(self) -> None:
        unigram = row_at(client.get(SMOOTHING).json()["sweeps"], 1)

        assert round(unigram["best_tuning_perplexity"], 4) == 34.6343
        assert round(unigram["laplace_tuning_perplexity"], 4) == 34.9349
        assert round(unigram["best_test_perplexity"], 4) == 46.73
        assert round(unigram["laplace_test_perplexity"], 4) == 43.06
        assert unigram["best_test_perplexity"] > unigram["laplace_test_perplexity"]

    def test_adding_one_leaves_a_ninth_of_a_certainty(self) -> None:
        body = client.get(SMOOTHING).json()
        rows = {row["strength"]: row for row in body["certain_rows"]}

        assert body["certain_count"] == 7
        assert body["certain_context_total"] == 7
        assert body["certain_unsmoothed_probability"] == 1.0
        assert round(rows[1.0]["probability"], 6) == 0.112676
        assert round(rows[1.0]["entropy"], 4) == 5.8117
        assert round(rows[0.005]["probability"], 6) == 0.956967
        assert round(rows[0.005]["entropy"], 4) == 0.5132

    def test_a_very_large_constant_reaches_the_flat_answer(self) -> None:
        body = client.get(SMOOTHING).json()
        limits = {row["strength"]: row["test_perplexity"] for row in body["limits"]}

        assert body["vocabulary_size"] == 64
        assert body["flat_probability"] == 1 / 64
        assert body["flat_entropy"] == 6.0
        assert round(limits[1.0], 4) == 23.6179
        assert round(limits[100.0], 4) == 61.6957
        assert round(limits[100000.0], 4) == 63.9975

    def test_a_context_nothing_ever_followed_is_answered_flatly(self) -> None:
        body = client.get(SMOOTHING).json()

        for strength in (1.0, 0.005):
            answer = answer_for(body, "after a word never seen", strength)
            assert answer["context_total"] == 0
            assert answer["n_followers"] == 0
            assert answer["entropy"] == 6.0
            assert all(row["probability"] == 1 / 64 for row in answer["top"])

    def test_the_constant_decides_how_much_of_a_certainty_survives(self) -> None:
        body = client.get(SMOOTHING).json()

        loose = answer_for(body, "after “Dr.”", 1.0)
        tight = answer_for(body, "after “Dr.”", 0.005)
        assert loose["n_followers"] == 1
        assert loose["top"][0]["word"] == "Alvarez"
        assert round(loose["entropy"], 4) == 5.8117
        assert round(tight["entropy"], 4) == 0.5132

        spread = answer_for(body, "after “the”", 1.0)
        assert spread["n_followers"] == 18
        assert spread["context_total"] == 46
        assert spread["top"][0]["word"] == "report"
        assert round(spread["entropy"], 4) == 5.6283

    def test_the_start_of_a_sentence_is_the_narrowest_answer_of_the_four(self) -> None:
        body = client.get(SMOOTHING).json()
        start = answer_for(body, "before the first word", 1.0)

        assert start["context"] == ["<s>"]
        assert start["context_total"] == 44
        assert start["n_followers"] == 5
        assert start["top"][0]["word"] == "The"
        assert round(start["entropy"], 4) == 4.9775


class TestScores:
    def test_the_unsmoothed_rule_cannot_score_the_test_text_at_any_width(self) -> None:
        rows = client.get(SCORES).json()["orders"]

        for order in (1, 2, 3, 4, 5):
            assert row_at(rows, order)["unsmoothed_test_is_finite"] is False
            assert row_at(rows, order)["unsmoothed_test_perplexity"] is None

    def test_the_unsmoothed_training_score_falls_with_every_extra_word(self) -> None:
        rows = client.get(SCORES).json()["orders"]
        training = [
            round(row_at(rows, order)["unsmoothed_training_perplexity"], 4)
            for order in (1, 2, 3, 4, 5)
        ]

        assert training == [35.0619, 2.997, 2.0328, 1.6711, 1.6025]
        assert training == sorted(training, reverse=True)

    def test_the_held_out_score_turns_at_two_words(self) -> None:
        rows = client.get(SCORES).json()["orders"]
        tuned = [
            round(row_at(rows, order)["tuned_test_perplexity"], 4)
            for order in (1, 2, 3, 4, 5)
        ]
        laplace = [
            round(row_at(rows, order)["laplace_test_perplexity"], 4)
            for order in (1, 2, 3, 4, 5)
        ]

        assert tuned == [46.73, 8.9626, 13.0023, 20.0899, 23.3974]
        assert laplace == [43.06, 23.6179, 31.048, 36.0869, 37.7205]
        assert min(tuned) == tuned[1]
        assert min(laplace) == laplace[1]

    def test_the_two_readings_of_one_number_agree(self) -> None:
        row = row_at(client.get(SCORES).json()["orders"], 2)

        assert round(row["laplace_test_cross_entropy"], 4) == 4.5618
        assert round(2 ** row["laplace_test_cross_entropy"], 4) == round(
            row["laplace_test_perplexity"], 4
        )

    def test_one_word_in_eighteen_of_the_test_text_was_never_seen(self) -> None:
        body = client.get(SCORES).json()

        assert body["n_test_texts"] == 6
        assert body["n_predicted_positions"] == 53
        assert round(row_at(body["orders"], 2)["unknown_share"], 4) == 0.0566

    def test_a_one_word_window_reads_a_reversed_text_identically(self) -> None:
        rows = client.get(SCORES).json()["reversed_rows"]

        assert (
            row_at(rows, 1)["forwards_perplexity"]
            == row_at(rows, 1)["reversed_perplexity"]
        )
        assert round(row_at(rows, 1)["forwards_perplexity"], 4) == 57.5683
        assert round(row_at(rows, 2)["forwards_perplexity"], 4) == 8.9626
        assert round(row_at(rows, 2)["reversed_perplexity"], 4) == 1715.1308
        assert round(row_at(rows, 3)["reversed_perplexity"], 4) == 111.8429

    def test_a_subject_five_words_back_is_a_coin_toss_until_the_window_reaches_it(
        self,
    ) -> None:
        body = client.get(SCORES).json()
        rows = body["dependency_rows"]

        assert body["dependency_context"][1] == "samples"
        assert body["agreeing_word"] == "were"
        for order in (2, 3, 4, 5):
            assert row_at(rows, order)["agreeing_probability"] == 0.5
            assert row_at(rows, order)["disagreeing_probability"] == 0.5
        assert row_at(rows, 6)["n_context_words"] == 5
        assert row_at(rows, 6)["agreeing_probability"] == 1.0
        assert row_at(rows, 6)["disagreeing_probability"] == 0.0

    def test_the_running_sentence_at_three_widths_and_three_rules(self) -> None:
        rows = client.get(SCORES).json()["running_rows"]
        by_setting = {(row["order"], row["strength"]): row for row in rows}

        assert by_setting[(2, None)]["is_finite"] is False
        assert round(by_setting[(1, 1.0)]["perplexity"], 4) == 68.1435
        assert round(by_setting[(2, 1.0)]["perplexity"], 4) == 30.3903
        assert round(by_setting[(2, 0.005)]["perplexity"], 4) == 16.0254
        assert round(by_setting[(3, 0.005)]["perplexity"], 4) == 15.7859
        assert round(by_setting[(1, 0.005)]["perplexity"], 4) == 242.1801


class TestGeneration:
    def test_a_wider_window_returns_the_training_text_word_for_word(self) -> None:
        rows = client.get(GENERATION).json()["by_order"]
        copies = [row_at(rows, order)["n_exact_copies"] for order in (1, 2, 3, 4, 5, 6)]

        assert copies == [0, 2, 10, 37, 45, 50]
        assert row_at(rows, 1)["n_draws"] == 50
        assert row_at(rows, 6)["share_exact_copies"] == 1.0

    def test_the_drawn_sentences_are_reproducible(self) -> None:
        rows = client.get(GENERATION).json()["by_order"]

        assert row_at(rows, 2)["samples"][2] == "The samples were expected on Monday."
        assert (
            row_at(rows, 6)["samples"][1]
            == "The delay was the reason for the revised estimate."
        )
        assert len(row_at(rows, 3)["samples"]) == 3

    def test_a_peaked_draw_writes_the_corpus_and_a_flat_one_writes_noise(self) -> None:
        body = client.get(GENERATION).json()
        by_temperature = {row["temperature"]: row for row in body["by_temperature"]}

        assert body["temperature_order"] == 3
        assert (
            by_temperature[0.25]["samples"][0]
            == "The analysis of the samples was expected to agree."
        )
        assert by_temperature[1.0]["samples"][0] != by_temperature[0.25]["samples"][0]


class TestFit:
    def test_the_playground_reads_the_running_sentence(self) -> None:
        body = client.post(FIT, json={}).json()

        assert body["order"] == 2
        assert body["strength"] == 0.005
        assert body["vocabulary_size"] == 64
        assert body["n_types"] == 141
        assert body["n_predicted"] == 8
        assert body["n_unknown"] == 2
        assert body["unknown_words"] == ["didn't", "expect"]
        assert body["n_unseen_runs"] == 3
        assert round(body["perplexity"], 4) == 16.0254
        assert body["is_finite"] is True
        assert body["framed"][0] == "<s>"
        assert body["framed"].count("<unk>") == 2

    def test_the_unsmoothed_rule_answers_that_the_sentence_is_impossible(self) -> None:
        body = client.post(FIT, json={"rule": "counts_as_they_stand"}).json()

        assert body["strength"] is None
        assert body["is_finite"] is False
        assert body["perplexity"] is None
        assert body["log_probability"] is None
        assert any(step["probability"] == 0.0 for step in body["steps"])

    def test_the_unsmoothed_rule_can_score_a_sentence_it_was_fitted_on(self) -> None:
        body = client.post(
            FIT,
            json={
                "rule": "counts_as_they_stand",
                "text": "The report was expected on Monday.",
            },
        ).json()

        assert body["is_finite"] is True
        assert round(body["perplexity"], 4) == 2.6103
        assert body["n_unknown"] == 0

    def test_a_wider_window_reads_more_context_at_each_step(self) -> None:
        body = client.post(FIT, json={"order": 3}).json()

        assert body["framed"][:2] == ["<s>", "<s>"]
        assert body["steps"][0]["context"] == "<s> <s>"
        assert len(body["steps"]) == 8
        assert body["n_types"] == 202

    def test_a_one_word_window_conditions_on_nothing(self) -> None:
        body = client.post(FIT, json={"order": 1}).json()

        assert body["steps"][0]["context"] == ""
        assert body["next_context"] == []
        assert round(body["perplexity"], 4) == 242.1801

    def test_the_next_word_answer_comes_back_with_the_walk(self) -> None:
        body = client.post(FIT, json={}).json()

        assert body["next_context"] == ["re-analysis."]
        assert body["next_is_defined"] is True
        assert body["next_top"][0]["word"] == "</s>"
        assert round(body["next_entropy"], 4) == 1.0197

    def test_an_unseen_context_has_no_unsmoothed_answer_at_all(self) -> None:
        body = client.post(
            FIT, json={"rule": "counts_as_they_stand", "text": "the wibble wobble"}
        ).json()

        assert body["next_is_defined"] is False
        assert body["next_entropy"] is None
        assert body["next_top"] == []

    def test_a_seeded_draw_repeats(self) -> None:
        first = client.post(FIT, json={"random_seed": 3}).json()["generated"]
        again = client.post(FIT, json={"random_seed": 3}).json()["generated"]
        other = client.post(FIT, json={"random_seed": 4}).json()["generated"]

        assert first == again
        assert first != other

    def test_a_text_with_no_words_is_refused(self) -> None:
        for text in ("", "   "):
            response = client.post(FIT, json={"text": text})

            assert response.status_code == 422
            assert response.json()["detail"] == "There are no words here to read."

    def test_the_bounds_are_enforced_at_the_door(self) -> None:
        assert client.post(FIT, json={"order": 0}).status_code == 422
        assert client.post(FIT, json={"order": 7}).status_code == 422
        assert client.post(FIT, json={"strength": 0.0}).status_code == 422
        assert client.post(FIT, json={"strength": 101.0}).status_code == 422
        assert client.post(FIT, json={"text": "word " * 60}).status_code == 422

    def test_an_unknown_field_is_refused_rather_than_ignored(self) -> None:
        assert client.post(FIT, json={"smoothing": 1.0}).status_code == 422


class TestNothingOnThePageNamesTheCode:
    """Every string a widget can render, checked for identifiers.

    A refusal is page text as much as a paragraph is, so the message a reader
    meets may not carry a parameter name, a class name or an underscore, and
    neither may a label printed beside a chart.
    """

    IDENTIFIER = re.compile(r"[a-z]+_[a-z]+|[A-Z][a-z]+[A-Z][A-Za-z]*")

    def test_the_refusal_carries_no_identifier(self) -> None:
        detail = client.post(FIT, json={"text": "  "}).json()["detail"]

        assert self.IDENTIFIER.search(detail) is None
        assert "Error" not in detail

    def test_no_label_a_widget_prints_names_the_code(self) -> None:
        labels = [answer["label"] for answer in client.get(SMOOTHING).json()["answers"]]

        assert labels
        for label in labels:
            assert self.IDENTIFIER.search(label) is None

    def test_no_text_a_widget_prints_names_the_code(self) -> None:
        bodies = [
            client.get(GRAMS).json()["sentence"],
            *client.get(COUNTS).json()["corpus"],
            *client.get(SCORES).json()["dependency_corpus"],
            *(
                sample
                for row in client.get(GENERATION).json()["by_order"]
                for sample in row["samples"]
            ),
        ]

        for text in bodies:
            assert self.IDENTIFIER.search(text) is None
