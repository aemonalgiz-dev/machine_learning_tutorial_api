"""The word2vec page's endpoints, on the two-list corpus the page carries.

Every number the page prints is pinned here. The expensive endpoints are cached
in the module under test, so the first test that reaches one pays for it and the
rest are free.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.word2vec import (
    MONEY_WORDS,
    SMALL_CORPUS,
    TWO_LIST_CORPUS,
    VERB_WORDS,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

FIT = "/concepts/word2vec/fit"
COUNTING = "/concepts/word2vec/counting"
WINDOWS = "/concepts/word2vec/windows"
PAIR_STEP = "/concepts/word2vec/pair-step"
SAMPLER = "/concepts/word2vec/sampler"
TREE = "/concepts/word2vec/tree"
COMBINATIONS = "/concepts/word2vec/combinations"
LONGER = "/concepts/word2vec/longer-training"
SEEDS = "/concepts/word2vec/seeds"
RARE = "/concepts/word2vec/rare-word"


class TestTheCorpus:
    def test_it_is_the_corpus_the_specs_fit(self) -> None:
        """Reproduced rather than imported, so a test holds the two together.

        ``test`` is not part of the installed package, so this is skipped
        wherever the library's own suite is not on the path.
        """
        corpora = pytest.importorskip(
            "test.core.natural_language_processing.embeddings.prediction.corpora"
        )

        assert TWO_LIST_CORPUS == corpora.TWO_TOPIC_CORPUS
        assert VERB_WORDS == corpora.VERB_WORDS
        assert MONEY_WORDS == corpora.FINANCE_WORDS

    def test_two_hundred_sentences_of_thirty_one_words(self) -> None:
        body = client.post(FIT, json={"epochs": 1}).json()

        assert body["n_sentences"] == 200
        assert body["n_occurrences"] == 1311
        assert body["n_words"] == 31


class TestFit:
    def test_the_reference_fit_separates_the_two_lists(self) -> None:
        body = client.post(
            FIT, json={"learning_rate": 0.05, "epochs": 5, "word": "walked"}
        ).json()

        assert body["within_topic"] == pytest.approx(0.9962, abs=5e-4)
        assert body["across_topic"] == pytest.approx(0.1696, abs=5e-4)
        assert [entry["word"] for entry in body["neighbours"]][:1] == ["talking"]
        assert all(entry["topic"] == "verb" for entry in body["neighbours"])

    def test_the_objective_falls_and_the_rate_falls_with_it(self) -> None:
        body = client.post(FIT, json={"learning_rate": 0.05, "epochs": 5}).json()

        losses = [entry["mean_loss"] for entry in body["epochs"]]
        rates = [entry["learning_rate"] for entry in body["epochs"]]
        assert losses[0] == pytest.approx(3.1279, abs=5e-4)
        assert losses[-1] == pytest.approx(2.0872, abs=5e-4)
        assert rates == sorted(rates, reverse=True)
        assert rates[0] == pytest.approx(0.040008, abs=1e-5)

    def test_the_two_directions_make_different_numbers_of_pairs(self) -> None:
        forward = client.post(
            FIT, json={"architecture": "skip-gram", "epochs": 1}
        ).json()
        backward = client.post(
            FIT, json={"architecture": "bag-of-words", "epochs": 1}
        ).json()

        assert forward["epochs"][0]["n_pairs"] == 3968
        assert backward["epochs"][0]["n_pairs"] == 1311

    def test_half_of_what_is_learned_is_thrown_away(self) -> None:
        body = client.post(FIT, json={"epochs": 1}).json()

        assert body["kept_numbers"] == 31 * 12
        assert body["discarded_numbers"] == 31 * 12
        assert body["output_rows"] == 31

    def test_the_tree_keeps_one_row_per_internal_node(self) -> None:
        body = client.post(FIT, json={"objective": "tree", "epochs": 1}).json()

        assert body["output_rows"] == 30
        assert body["predicted"] == []

    def test_the_discarded_half_is_what_does_the_predicting(self) -> None:
        body = client.post(
            FIT, json={"learning_rate": 0.05, "epochs": 5, "n_neighbours": 6}
        ).json()

        assert all(entry["topic"] == "verb" for entry in body["predicted"])
        assert body["predicted"][0]["similarity"] == pytest.approx(0.3022, abs=5e-4)
        assert body["predicted_mean_same_list"] == pytest.approx(0.2672, abs=5e-4)
        assert body["predicted_mean_other_list"] == pytest.approx(0.0061, abs=5e-4)

    def test_a_word_outside_the_corpus_is_refused_in_readable_words(self) -> None:
        response = client.post(FIT, json={"epochs": 1, "word": "quibble"})

        assert response.status_code == 422
        assert "quibble" in response.json()["detail"]

    @pytest.mark.parametrize(
        "keywords",
        [
            {"epochs": 0},
            {"epochs": 99},
            {"dimension": 1},
            {"dimension": 999},
            {"learning_rate": 0.0},
            {"window": 0},
            {"architecture": "sideways"},
            {"objective": "softmax"},
            {"size": 3},
        ],
    )
    def test_a_request_outside_the_bounds_is_refused(
        self, keywords: dict[str, object]
    ) -> None:
        assert client.post(FIT, json=keywords).status_code == 422


class TestCounting:
    def test_the_table_is_as_wide_as_the_vocabulary(self) -> None:
        body = client.get(COUNTING).json()

        assert body["row_width"] == 31
        assert body["n_cells"] == 961
        assert body["n_zero_cells"] == 440
        assert body["zero_share"] == pytest.approx(0.4579, abs=5e-4)
        assert body["vector_numbers"] == 372

    def test_counting_alone_separates_the_lists_completely(self) -> None:
        body = client.get(COUNTING).json()

        assert body["within_topic"] == pytest.approx(0.8558, abs=5e-4)
        assert body["across_topic"] == pytest.approx(0.0, abs=1e-12)

    def test_the_largest_counts_beside_walked_are_verb_forms(self) -> None:
        body = client.get(COUNTING).json()

        assert next(iter(body["largest_for_walked"]))["word"] == "walk"
        assert all(entry["topic"] == "verb" for entry in body["largest_for_walked"])


class TestWindows:
    def test_the_small_corpus_is_six_words_and_nine_occurrences(self) -> None:
        body = client.post(WINDOWS, json={"reach": 1}).json()

        assert body["sentences"] == [text.split() for text in SMALL_CORPUS]
        assert body["words"] == ["talks", "walk", "walked", "bond", "market", "price"]
        assert body["counts"] == [2, 2, 2, 1, 1, 1]

    def test_one_direction_makes_a_pair_per_neighbour_and_the_other_per_position(
        self,
    ) -> None:
        body = client.post(
            WINDOWS, json={"reach": 1, "architecture": "skip-gram"}
        ).json()

        assert body["n_pairs_skip_gram"] == 12
        assert body["n_pairs_bag_of_words"] == 9
        assert body["n_pairs"] == 12

    def test_a_wider_reach_makes_more_pairs_one_way_and_not_the_other(self) -> None:
        narrow = client.post(WINDOWS, json={"reach": 1}).json()
        wide = client.post(WINDOWS, json={"reach": 2}).json()

        assert wide["n_pairs_skip_gram"] > narrow["n_pairs_skip_gram"]
        assert wide["n_pairs_bag_of_words"] == narrow["n_pairs_bag_of_words"]

    def test_the_middle_word_of_the_first_sentence_reads_both_sides(self) -> None:
        body = client.post(
            WINDOWS, json={"reach": 1, "architecture": "bag-of-words"}
        ).json()

        middle = next(
            pair
            for pair in body["pairs"]
            if pair["sentence"] == 0 and pair["position"] == 1
        )
        assert middle["centre"] == "walked"
        assert middle["context"] == ["walk", "talks"]
        assert middle["reads"] == ["walk", "talks"]
        assert middle["predicts"] == ["walked"]


class TestPairStep:
    def test_the_pair_worked_by_hand(self) -> None:
        """A half-length vector against one row with it, one against it and one
        across it: two of the three losses are equal and the third is log 2."""
        body = client.post(
            PAIR_STEP,
            json={
                "hidden": {"x": 0.5, "y": 0.0},
                "true_row": {"x": 1.0, "y": 0.0},
                "negative_rows": [{"x": -1.0, "y": 0.0}, {"x": 0.0, "y": 1.0}],
            },
        ).json()

        rows = body["rows"]
        assert [row["score"] for row in rows] == pytest.approx([0.5, -0.5, 0.0])
        assert [row["probability"] for row in rows] == pytest.approx(
            [0.622459, 0.377541, 0.5], abs=1e-6
        )
        assert body["loss"] == pytest.approx(1.641301, abs=1e-6)
        assert rows[2]["loss"] == pytest.approx(0.693147, abs=1e-6)
        assert body["hidden_push"] == pytest.approx(
            {"x": 0.755081, "y": -0.5}, abs=1e-6
        )

    def test_the_true_row_is_pulled_towards_the_word_and_the_others_away(self) -> None:
        body = client.post(
            PAIR_STEP,
            json={
                "hidden": {"x": 1.0, "y": 0.0},
                "true_row": {"x": 0.2, "y": 0.0},
                "negative_rows": [{"x": 0.3, "y": 0.4}],
            },
        ).json()

        assert body["rows"][0]["push_x"] > 0
        assert body["rows"][1]["push_x"] < 0

    def test_a_pair_with_no_negatives_is_refused(self) -> None:
        response = client.post(
            PAIR_STEP,
            json={
                "hidden": {"x": 1.0, "y": 0.0},
                "true_row": {"x": 1.0, "y": 0.0},
                "negative_rows": [],
            },
        )

        assert response.status_code == 422


class TestSampler:
    def test_the_three_quarters_power_lifts_the_rare_word(self) -> None:
        body = client.post(SAMPLER, json={"exponent": 0.75}).json()

        assert body["toy_counts"] == [1000, 10, 1]
        assert body["toy_raw"] == pytest.approx([0.98912, 0.009891, 0.000989], abs=1e-6)
        assert body["toy_flattened"] == pytest.approx(
            [0.964091, 0.030487, 0.005421], abs=1e-6
        )

    def test_on_the_running_corpus_the_odds_narrow(self) -> None:
        body = client.post(SAMPLER, json={"exponent": 0.75}).json()

        assert body["commonest"]["word"] == "debt"
        assert body["commonest"]["count"] == 73
        assert body["rarest"]["word"] == "saying"
        assert body["rarest"]["count"] == 17
        assert body["raw_odds"] == pytest.approx(4.2941, abs=5e-4)
        assert body["flattened_odds"] == pytest.approx(2.9830, abs=5e-4)

    def test_an_exponent_of_one_is_raw_frequency(self) -> None:
        body = client.post(SAMPLER, json={"exponent": 1.0}).json()

        assert body["raw_odds"] == pytest.approx(body["flattened_odds"])
        assert all(row["ratio"] == pytest.approx(1.0) for row in body["rows"])

    def test_the_shares_sum_to_one(self) -> None:
        body = client.post(SAMPLER, json={"exponent": 0.4}).json()

        assert sum(row["flattened_share"] for row in body["rows"]) == pytest.approx(1.0)

    @pytest.mark.parametrize("exponent", [0.0, -1.0, 1.5])
    def test_an_exponent_outside_the_range_is_refused(self, exponent: float) -> None:
        assert client.post(SAMPLER, json={"exponent": exponent}).status_code == 422


class TestTree:
    def test_the_small_tree_gives_the_common_words_the_short_paths(self) -> None:
        body = client.get(TREE).json()

        by_word = {entry["word"]: entry for entry in body["small_paths"]}
        assert by_word["walk"]["depth"] == 2
        assert by_word["walked"]["depth"] == 2
        assert by_word["bond"]["depth"] == 3
        assert body["small_internal_nodes"] == 5
        assert body["small_expected_depth"] == pytest.approx(2.5556, abs=5e-4)
        assert body["small_sweep_rows"] == 6

    def test_on_the_running_corpus_the_tree_barely_beats_a_balanced_one(self) -> None:
        """The counts here run from 73 down to 17, which is nearly flat, and a
        Huffman tree only pays where the counts are skewed."""
        body = client.get(TREE).json()

        assert body["corpus_expected_depth"] == pytest.approx(4.9008, abs=5e-4)
        assert body["corpus_balanced_depth"] == pytest.approx(4.9542, abs=5e-4)
        assert body["corpus_sweep_rows"] == 31
        assert body["corpus_negative_rows"] == 6
        assert body["corpus_shortest_depth"] == 4
        assert body["corpus_longest_depth"] == 6

    def test_the_saving_grows_with_the_vocabulary(self) -> None:
        body = client.get(TREE).json()

        by_size = {row["n_words"]: row for row in body["sizes"]}
        assert by_size[31]["uniform_depth"] == pytest.approx(4.9677, abs=5e-4)
        assert by_size[40000]["uniform_depth"] == pytest.approx(15.3616, abs=5e-4)
        assert by_size[40000]["skewed_depth"] == pytest.approx(10.7469, abs=5e-4)
        assert by_size[40000]["sweep_rows"] == 40000
        assert by_size[40000]["negative_rows"] == 6


class TestCombinations:
    def test_the_reference_rate_leaves_one_of_the_four_undecided(self) -> None:
        body = client.get(COMBINATIONS).json()

        entry = next(
            report
            for report in body["reports"]
            if report["learning_rate"] == 0.025
            and report["architecture"] == "bag-of-words"
            and report["objective"] == "negative-sampling"
        )
        assert entry["within_topic"] == pytest.approx(0.9907, abs=5e-4)
        assert entry["across_topic"] == pytest.approx(0.9873, abs=5e-4)
        assert entry["gap"] == pytest.approx(0.0034, abs=5e-4)

    def test_doubling_the_rate_moves_that_one_apart(self) -> None:
        body = client.get(COMBINATIONS).json()

        entry = next(
            report
            for report in body["reports"]
            if report["learning_rate"] == 0.05
            and report["architecture"] == "bag-of-words"
            and report["objective"] == "negative-sampling"
        )
        assert entry["within_topic"] == pytest.approx(0.9929, abs=5e-4)
        assert entry["across_topic"] == pytest.approx(0.7116, abs=5e-4)

    def test_every_combination_separates_the_lists_at_the_doubled_rate(self) -> None:
        body = client.get(COMBINATIONS).json()

        doubled = [
            report for report in body["reports"] if report["learning_rate"] == 0.05
        ]
        assert len(doubled) == 4
        assert all(report["gap"] > 0.25 for report in doubled)

    def test_the_tree_carries_one_fewer_row_than_there_are_words(self) -> None:
        body = client.get(COMBINATIONS).json()

        for report in body["reports"]:
            expected = 30 if report["objective"] == "tree" else 31
            assert report["output_rows"] == expected

    def test_the_four_agree_on_the_list_and_not_on_the_order(self) -> None:
        body = client.get(COMBINATIONS).json()

        doubled = [
            report for report in body["reports"] if report["learning_rate"] == 0.05
        ]
        lists = [
            {entry["word"] for entry in report["neighbours"]["walked"]}
            for report in doubled
        ]
        assert all(
            entry["topic"] == "verb"
            for report in doubled
            for entry in report["neighbours"]["walked"]
        )
        assert len(set.intersection(*lists)) == 1
        assert len(set.union(*lists)) == 10


class TestLongerTraining:
    def test_the_objective_keeps_falling_after_the_vectors_stop_improving(
        self,
    ) -> None:
        body = client.get(LONGER).json()

        by_epochs = {point["epochs"]: point for point in body["points"]}
        assert by_epochs[5]["final_loss"] > by_epochs[10]["final_loss"]
        assert by_epochs[10]["final_loss"] > by_epochs[20]["final_loss"]
        assert by_epochs[20]["gap"] < by_epochs[10]["gap"]

    def test_the_measured_numbers(self) -> None:
        body = client.get(LONGER).json()

        by_epochs = {point["epochs"]: point for point in body["points"]}
        assert by_epochs[1]["gap"] == pytest.approx(0.0072, abs=5e-4)
        assert by_epochs[10]["gap"] == pytest.approx(0.8404, abs=5e-4)
        assert by_epochs[20]["gap"] == pytest.approx(0.7397, abs=5e-4)
        assert by_epochs[20]["within_topic"] == pytest.approx(0.8510, abs=5e-4)
        assert by_epochs[20]["final_loss"] == pytest.approx(2.0267, abs=5e-4)


class TestSeeds:
    def test_the_coordinates_disagree_and_the_geometry_does_not(self) -> None:
        body = client.get(SEEDS).json()

        assert body["mean_coordinate_gap"] == pytest.approx(0.7816, abs=5e-4)
        assert body["largest_coordinate_gap"] == pytest.approx(2.5336, abs=5e-4)
        assert body["similarity_correlation"] == pytest.approx(0.9996, abs=5e-4)
        assert body["mean_similarity_gap"] == pytest.approx(0.0073, abs=5e-4)

    def test_the_two_lists_of_nearest_words_overlap_in_one_word(self) -> None:
        body = client.get(SEEDS).json()

        assert body["shared_neighbours"] == ["talking"]
        assert len(body["reports"]) == 2
        assert body["reports"][0]["coordinates"] != body["reports"][1]["coordinates"]

    def test_the_same_seed_reproduces_the_fit_to_the_last_bit(self) -> None:
        first = client.post(FIT, json={"epochs": 2, "random_seed": 7}).json()
        second = client.post(FIT, json={"epochs": 2, "random_seed": 7}).json()

        assert first["coordinates"] == second["coordinates"]
        assert first["epochs"] == second["epochs"]


class TestRareWord:
    def test_a_word_seen_twice_moves_far_more_than_one_seen_often(self) -> None:
        body = client.get(RARE).json()

        assert body["rare_count"] == 2
        assert body["common_count"] == 42
        assert body["rare_spread"] > 5 * body["common_spread"]
        assert len(body["rare_similarities"]) == 4

    def test_the_rare_word_lands_beside_a_different_word_each_time(self) -> None:
        body = client.get(RARE).json()

        firsts = {entries[0] for entries in body["rare_neighbour_lists"]}
        assert len(firsts) > 1


IDENTIFIER = re.compile(r"\b(?:[a-z]+_[a-z_]+|[A-Z][a-z]+[A-Z][A-Za-z]*)\b")
"""A snake-cased or camel-cased name, which no string a reader sees may hold."""

FORBIDDEN = ("oop_ml", "the library", "the API", "Error")
"""Words that would credit the code behind the page rather than describe it."""


class TestNothingAReaderSeesNamesTheCode:
    """Every string a widget renders is page text, so it is checked like text.

    The only strings that reach a reader from these endpoints are words of the
    corpus, the architecture and objective names, and the topic labels, so this
    asserts exactly that and fails if a new field starts carrying prose.
    """

    def test_the_words_that_cross_the_wire_hold_no_identifier(self) -> None:
        bodies = [
            client.post(FIT, json={"epochs": 1}).json(),
            client.get(COUNTING).json(),
            client.post(WINDOWS, json={"reach": 1}).json(),
            client.post(SAMPLER, json={"exponent": 0.75}).json(),
            client.get(TREE).json(),
            client.get(SEEDS).json(),
            client.get(RARE).json(),
        ]
        for body in bodies:
            for text in _strings(body):
                assert not IDENTIFIER.search(text), text
                for forbidden in FORBIDDEN:
                    assert forbidden not in text

    def test_a_refusal_a_reader_can_reach_is_written_in_plain_words(self) -> None:
        detail = client.post(FIT, json={"epochs": 1, "word": "elephant"}).json()[
            "detail"
        ]

        assert not IDENTIFIER.search(detail), detail
        assert "elephant" in detail


def _strings(value: object) -> list[str]:
    """Every string anywhere inside a decoded response body."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for entry in value.values() for text in _strings(entry)] + [
            key for key in value if False
        ]
    if isinstance(value, list):
        return [text for entry in value for text in _strings(entry)]
    return []
