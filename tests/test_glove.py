"""The GloVe page's endpoints, on the two-topic corpus the page carries.

Every number the page prints is pinned here. The expensive endpoints are cached
in the module under test, so the first test that reaches one pays for it and the
rest are free.

The load-bearing test of the module is
``TestTheTwoStepRulesShareOneLoop::test_the_accumulating_branch_is_the_librarys_own_fit``.
The page's whole claim about the step rule rests on the two branches differing
only in where the running sum of squared slopes starts, so the branch that
matches the library is checked against the library's own fit, parameter by
parameter, and the other branch is then a difference of rules rather than of
code.
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient
from oop_ml.core.natural_language_processing.embeddings.cooccurrence import (
    ContextWeighting,
    CooccurrenceMatrix,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import TokenisedCorpus
from oop_ml.core.natural_language_processing.embeddings.prediction.glove import GloVe
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)

from oop_ml_api.concepts.glove import (
    ASTRONOMY,
    COOKING,
    FUNCTION_WORDS,
    PAGE_DIMENSION,
    PAGE_EPOCHS,
    PAGE_LEARNING_RATE,
    PAGE_SEED,
    PAGE_WINDOW,
    TWO_TOPICS,
    _matrix,
    _walk,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

FIT = "/concepts/glove/fit"
TABLE = "/concepts/glove/table"
RATIOS = "/concepts/glove/ratios"
WEIGHTING = "/concepts/glove/weighting"
STEP_RULES = "/concepts/glove/step-rules"
WINDOWS = "/concepts/glove/windows"
COST = "/concepts/glove/cost"


class TestTheCorpus:
    def test_it_is_the_corpus_the_specs_fit(self) -> None:
        """Reproduced rather than imported, so a test holds the two together.

        ``test`` is not part of the installed package, so this is skipped
        wherever the library's own suite is not on the path.
        """
        spec = pytest.importorskip(
            "test.core.natural_language_processing.embeddings.prediction.test_glove"
        )

        assert list(COOKING) == spec.COOKING
        assert list(ASTRONOMY) == spec.ASTRONOMY
        assert FUNCTION_WORDS == spec.FUNCTION_WORDS
        assert TWO_TOPICS == spec.TWO_TOPICS

    def test_twenty_four_sentences_of_thirty_seven_words(self) -> None:
        body = client.post(TABLE, json={"text": "corpus"}).json()

        assert body["n_sentences"] == 24
        assert body["n_occurrences"] == 216
        assert len(body["words"]) == 37
        assert body["words"][0] == "the"


class TestTheOneSentenceTable:
    """Five words, worked with a pencil in the page's own prose."""

    def test_every_cell_of_the_harmonic_table(self) -> None:
        body = client.post(TABLE, json={"text": "one-sentence", "window": 2}).json()

        assert body["words"] == ["the", "chef", "soup", "stirs"]
        assert body["table"] == [
            [0.0, 1.5, 1.0, 1.5],
            [1.5, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.5],
            [1.5, 1.0, 0.5, 0.0],
        ]
        assert body["total"] == 11.0
        assert body["n_nonzero"] == 10
        assert body["below_one"] == 2

    def test_giving_every_position_one_full_count_changes_three_cells(self) -> None:
        body = client.post(
            TABLE,
            json={"text": "one-sentence", "window": 2, "weighting": "uniform"},
        ).json()

        assert body["table"] == [
            [0.0, 2.0, 1.0, 2.0],
            [2.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [2.0, 1.0, 1.0, 0.0],
        ]
        assert body["total"] == 14.0
        assert body["below_one"] == 0

    def test_a_repeated_pair_pushes_a_count_above_one(self) -> None:
        body = client.post(TABLE, json={"text": "two-sentences", "window": 2}).json()

        assert body["words"] == ["the", "chef", "stirs", "broth", "soup"]
        assert body["table"][0][1] == 3.0
        assert body["table"][2][3] == 0.5
        assert body["table"][2][4] == 0.5
        assert body["total"] == 22.0


class TestTheCorpusTable:
    def test_the_shape_and_the_zeros(self) -> None:
        body = client.post(TABLE, json={"text": "corpus", "window": 2}).json()

        assert body["n_cells"] == 1369
        assert body["n_nonzero"] == 281
        assert body["zero_share"] == pytest.approx(0.7947, abs=5e-4)
        assert body["total"] == 552.0

    def test_the_logarithms(self) -> None:
        body = client.post(TABLE, json={"text": "corpus", "window": 2}).json()

        assert body["largest"] == 16.5
        assert body["smallest"] == 0.5
        assert body["below_one"] == 86
        assert body["below_one_share"] == pytest.approx(0.3060, abs=5e-4)
        assert body["mean_logarithm"] == pytest.approx(0.2617, abs=5e-4)

    def test_the_two_words_the_page_names_at_the_ends_of_the_counts(self) -> None:
        body = client.post(TABLE, json={"text": "corpus", "window": 2}).json()
        occurrences = dict(zip(body["words"], body["occurrences"], strict=True))

        assert occurrences["the"] == 61
        assert occurrences["oven"] == 2
        assert body["words"][0] == "the"

    def test_a_word_can_be_its_own_neighbour(self) -> None:
        """Two sentences put one word two positions from another copy of itself."""
        body = client.post(TABLE, json={"text": "corpus", "window": 2}).json()
        position = body["words"].index("and")

        assert body["table"][position][position] == 2.0
        assert (
            sum(
                1
                for index in range(len(body["words"]))
                if body["table"][index][index] > 0.0
            )
            == 1
        )

    def test_the_commonest_pair_is_two_words_that_carry_nothing(self) -> None:
        body = client.post(TABLE, json={"text": "corpus", "window": 2}).json()
        first = body["largest_pairs"][0]

        assert {first["word"], first["context"]} == {"the", "and"}
        assert first["count"] == 16.5
        assert first["logarithm"] == pytest.approx(2.8034, abs=5e-4)


class TestTheRatio:
    def test_only_one_probe_reaches_both_words(self) -> None:
        body = client.post(RATIOS, json={}).json()

        assert body["n_shared"] == 1
        assert body["n_only_first"] == 8
        assert body["n_only_second"] == 3
        assert body["n_neither"] == 25

    def test_the_shared_probe_is_a_word_that_separates_nothing(self) -> None:
        body = client.post(RATIOS, json={}).json()
        shared = [row for row in body["rows"] if row["verdict"] == "both"]

        assert [row["probe"] for row in shared] == ["the"]
        assert shared[0]["ratio"] == pytest.approx(0.7319, abs=5e-4)

    def test_a_probe_the_second_word_never_met_has_no_ratio(self) -> None:
        body = client.post(RATIOS, json={}).json()
        stirs = next(row for row in body["rows"] if row["probe"] == "stirs")

        assert stirs["ratio"] is None
        assert stirs["first_probability"] == pytest.approx(0.177778, abs=1e-6)
        assert stirs["second_probability"] == 0.0

    def test_a_word_that_is_not_in_the_corpus_is_refused(self) -> None:
        response = client.post(RATIOS, json={"first_word": "elephant"})

        assert response.status_code == 422


class TestTheWeighting:
    def test_at_the_published_cap_nothing_on_this_corpus_reaches_it(self) -> None:
        body = client.post(WEIGHTING, json={"maximum_count": 100.0}).json()

        assert body["n_at_the_cap"] == 0
        assert body["largest_count"] == 16.5
        assert body["largest_weight"] == pytest.approx(0.258889, abs=1e-6)
        assert body["smallest_weight"] == pytest.approx(0.018803, abs=1e-6)

    def test_a_cap_that_binds_takes_share_from_the_commonest_pairs(self) -> None:
        loose = client.post(WEIGHTING, json={"maximum_count": 100.0}).json()
        tight = client.post(WEIGHTING, json={"maximum_count": 7.5}).json()

        assert tight["n_at_the_cap"] == 10
        assert loose["top_pair_share"] == pytest.approx(0.019038, abs=1e-6)
        assert tight["top_pair_share"] == pytest.approx(0.010766, abs=1e-6)
        assert loose["top_ten_share"] == pytest.approx(0.126477, abs=1e-6)
        assert tight["top_ten_share"] == pytest.approx(0.107661, abs=1e-6)

    def test_a_cap_at_the_largest_count_redistributes_nothing(self) -> None:
        """Only counts *beyond* the cap have their weight held back."""
        loose = client.post(WEIGHTING, json={"maximum_count": 100.0}).json()
        exact = client.post(WEIGHTING, json={"maximum_count": 16.5}).json()

        assert exact["n_at_the_cap"] == 2
        assert exact["top_pair_share"] == pytest.approx(loose["top_pair_share"])
        assert exact["top_ten_share"] == pytest.approx(loose["top_ten_share"])

    def test_the_three_weightings_place_the_ten_largest_pairs_differently(self) -> None:
        body = client.post(WEIGHTING, json={"maximum_count": 100.0}).json()

        assert body["alike_top_ten_share"] == pytest.approx(0.035587, abs=1e-6)
        assert body["top_ten_share"] == pytest.approx(0.126477, abs=1e-6)
        assert body["proportional_top_ten_share"] == pytest.approx(0.175725, abs=1e-6)

    def test_the_pairs_at_a_cap_of_seven_and_a_half_are_named(self) -> None:
        body = client.post(WEIGHTING, json={"maximum_count": 7.5}).json()
        capped = [pair for pair in body["pairs"] if pair["at_the_cap"]]

        assert len(capped) == 10
        assert all("the" in {pair["word"], pair["context"]} for pair in capped)
        assert capped[0]["weight"] == 1.0

    def test_the_curve_is_flat_above_the_cap_and_rises_below_it(self) -> None:
        body = client.post(WEIGHTING, json={"maximum_count": 7.5}).json()
        curve = body["curve"]

        assert curve[0]["weight"] == 0.0
        assert all(
            later["weight"] >= earlier["weight"]
            for earlier, later in itertools.pairwise(curve)
        )
        assert curve[-1]["weight"] == 1.0


class TestTheTwoStepRulesShareOneLoop:
    def test_the_accumulating_branch_is_the_librarys_own_fit(self) -> None:
        """The walk written out here is the library's, to the last bit.

        Without this the page's two curves would be a comparison of two
        implementations rather than of two step rules.
        """
        model = GloVe(
            dimension=PAGE_DIMENSION,
            window=PAGE_WINDOW,
            epochs=PAGE_EPOCHS,
            learning_rate=PAGE_LEARNING_RATE,
            random_seed=PAGE_SEED,
        )
        walk = _walk(model, _matrix("corpus", PAGE_WINDOW, "harmonic"), "accumulating")
        fitted = GloVe(
            dimension=PAGE_DIMENSION,
            window=PAGE_WINDOW,
            epochs=PAGE_EPOCHS,
            learning_rate=PAGE_LEARNING_RATE,
            random_seed=PAGE_SEED,
        ).fit(TWO_TOPICS)

        assert walk.parameters == fitted.parameters
        assert walk.embeddings == fitted.embeddings
        assert walk.objective == fitted.objective_value()
        n_pairs = fitted.history[0].n_pairs
        assert walk.totals == pytest.approx(
            [record.mean_loss * n_pairs for record in fitted.history]
        )

    def test_the_matrix_is_the_one_the_foundation_builds(self) -> None:
        tokenised = TokenisedCorpus.from_texts(TWO_TOPICS, WhitespacePreTokenizer())
        vocabulary = tokenised.vocabulary(1)

        assert _matrix("corpus", PAGE_WINDOW, "harmonic") == (
            CooccurrenceMatrix.from_id_sequences(
                vocabulary,
                tokenised.id_sequences(vocabulary),
                PAGE_WINDOW,
                ContextWeighting.HARMONIC,
            )
        )


class TestTheStepRules:
    def test_the_two_rules_end_ten_times_apart(self) -> None:
        body = client.get(STEP_RULES).json()
        by_rule = {report["step_rule"]: report for report in body["reports"]}

        assert by_rule["accumulating"]["first_total"] == pytest.approx(
            18.9335, abs=5e-4
        )
        assert by_rule["accumulating"]["objective_at_rest"] == pytest.approx(
            0.5148, abs=5e-4
        )
        assert by_rule["released"]["first_total"] == pytest.approx(19.3399, abs=5e-4)
        assert by_rule["released"]["objective_at_rest"] == pytest.approx(
            4.9571, abs=5e-4
        )

    def test_the_released_rule_never_divides_a_slope_by_much(self) -> None:
        """Which is the mechanism: the step stays at the base rate times the slope."""
        body = client.get(STEP_RULES).json()
        by_rule = {report["step_rule"]: report for report in body["reports"]}

        assert by_rule["released"]["smallest_divisor"] == pytest.approx(1.0, abs=1e-5)
        assert by_rule["released"]["largest_divisor"] == pytest.approx(1.0860, abs=5e-4)
        assert by_rule["accumulating"]["smallest_divisor"] == pytest.approx(
            0.004939, abs=1e-6
        )
        assert by_rule["accumulating"]["largest_divisor"] == pytest.approx(
            1.2331, abs=5e-4
        )

    def test_the_same_gap_at_the_published_width_and_reach(self) -> None:
        body = client.get(STEP_RULES).json()
        by_rule = {report["step_rule"]: report for report in body["published_reports"]}

        assert body["published_window"] == 5
        assert body["published_dimension"] == 50
        assert by_rule["accumulating"]["objective_at_rest"] == pytest.approx(
            0.0334, abs=5e-4
        )
        assert by_rule["released"]["objective_at_rest"] == pytest.approx(
            8.8708, abs=5e-4
        )
        assert by_rule["released"]["largest_divisor"] == pytest.approx(1.0237, abs=5e-4)

    def test_the_step_rule_reaches_the_answer_and_not_only_the_objective(self) -> None:
        body = client.get(STEP_RULES).json()
        by_rule = {report["step_rule"]: report for report in body["reports"]}

        assert by_rule["accumulating"]["within_topic"] == pytest.approx(
            0.1188, abs=5e-4
        )
        assert by_rule["accumulating"]["across_topic"] == pytest.approx(
            -0.0778, abs=5e-4
        )
        assert by_rule["released"]["within_topic"] == pytest.approx(0.0447, abs=5e-4)
        assert by_rule["released"]["across_topic"] == pytest.approx(0.0141, abs=5e-4)

    def test_every_curve_falls(self) -> None:
        body = client.get(STEP_RULES).json()

        for report in body["reports"] + body["published_reports"]:
            assert report["final_total"] < report["first_total"]
            assert len(report["totals"]) == PAGE_EPOCHS


class TestTheWindow:
    def test_the_published_reach_puts_half_the_counts_below_one(self) -> None:
        body = client.get(WINDOWS).json()
        by_window = {row["window"]: row for row in body["rows"]}

        assert by_window[5]["below_one"] == 246
        assert by_window[5]["below_one_share"] == pytest.approx(0.5190, abs=5e-4)
        assert by_window[5]["mean_logarithm"] == pytest.approx(-0.1523, abs=5e-4)
        assert by_window[2]["below_one"] == 86
        assert by_window[2]["below_one_share"] == pytest.approx(0.3060, abs=5e-4)
        assert by_window[2]["mean_logarithm"] == pytest.approx(0.2617, abs=5e-4)

    def test_only_the_narrow_reach_separates_the_halves_on_every_start(self) -> None:
        body = client.get(WINDOWS).json()
        by_window = {row["window"]: row for row in body["rows"]}

        assert body["seeds"] == [0, 1, 2]
        assert all(gap > 0.1 for gap in by_window[2]["gaps_by_seed"])
        assert min(by_window[5]["gaps_by_seed"]) < 0.0
        assert min(by_window[3]["gaps_by_seed"]) < 0.0
        assert min(by_window[1]["gaps_by_seed"]) < 0.0

    def test_every_gap_the_page_prints(self) -> None:
        body = client.get(WINDOWS).json()
        by_window = {row["window"]: row for row in body["rows"]}

        assert by_window[1]["gaps_by_seed"] == pytest.approx(
            [-0.0171, 0.0483, 0.0001], abs=5e-4
        )
        assert by_window[2]["gaps_by_seed"] == pytest.approx(
            [0.1966, 0.1314, 0.2284], abs=5e-4
        )
        assert by_window[3]["gaps_by_seed"] == pytest.approx(
            [0.0043, 0.0541, -0.0039], abs=5e-4
        )
        assert by_window[5]["gaps_by_seed"] == pytest.approx(
            [0.0156, 0.0123, -0.0076], abs=5e-4
        )

    def test_the_narrowest_reach_puts_no_count_below_one(self) -> None:
        """At a reach of one every neighbour is next door, so no count is a fraction."""
        body = client.get(WINDOWS).json()
        by_window = {row["window"]: row for row in body["rows"]}

        assert by_window[1]["below_one"] == 0
        assert by_window[1]["mean_logarithm"] == pytest.approx(0.6623, abs=5e-4)
        assert by_window[3]["below_one"] == 178
        assert by_window[3]["below_one_share"] == pytest.approx(0.4564, abs=5e-4)
        assert by_window[3]["mean_logarithm"] == pytest.approx(0.0101, abs=5e-4)

    def test_the_negative_mean_logarithm_lands_in_the_dot_products(self) -> None:
        body = client.get(WINDOWS).json()
        by_window = {row["window"]: row for row in body["rows"]}

        assert by_window[5]["mean_word_bias"] == pytest.approx(-0.0169, abs=5e-4)
        assert by_window[5]["mean_dot_product"] == pytest.approx(-0.1359, abs=5e-4)
        assert by_window[2]["mean_dot_product"] == pytest.approx(0.0510, abs=5e-4)

    def test_a_wider_reach_finds_more_pairs(self) -> None:
        body = client.get(WINDOWS).json()
        pairs = [row["n_pairs"] for row in body["rows"]]

        assert pairs == [158, 281, 390, 474]


class TestTheCost:
    def test_the_table_stops_growing_while_the_text_does_not(self) -> None:
        body = client.get(COST).json()

        assert [row["nonzero_pairs"] for row in body["growth"]] == [281, 281, 281, 281]
        assert [row["ordered_positions"] for row in body["growth"]] == [
            720,
            1440,
            2880,
            5760,
        ]
        assert [row["n_words"] for row in body["growth"]] == [37, 37, 37, 37]

    def test_the_two_sizes(self) -> None:
        body = client.get(COST).json()

        assert body["numbers_per_word_rows"] == 37
        assert body["numbers_per_word_vectors"] == 8
        assert body["numbers_in_all_rows"] == 1369
        assert body["numbers_in_all_vectors"] == 296

    def test_comparing_rows_directly_beats_the_fit_on_the_neighbour_question(
        self,
    ) -> None:
        """Reported rather than hidden: at this size the raw counts win one of two."""
        body = client.get(COST).json()

        assert body["rows_same_topic_share"] == pytest.approx(0.7467, abs=5e-4)
        assert body["fitted_same_topic_share"] == pytest.approx(0.6733, abs=5e-4)
        assert body["rows_same_topic_share"] > body["fitted_same_topic_share"]

    def test_the_fit_beats_the_rows_on_the_two_mean_cosines(self) -> None:
        body = client.get(COST).json()

        assert body["rows_within_topic"] == pytest.approx(0.6161, abs=5e-4)
        assert body["rows_across_topic"] == pytest.approx(0.4756, abs=5e-4)
        assert body["rows_gap"] == pytest.approx(0.1404, abs=5e-4)
        assert body["fitted_gap"] == pytest.approx(0.1966, abs=5e-4)
        assert body["fitted_gap"] > body["rows_gap"]

    def test_the_two_neighbour_lists_for_one_word(self) -> None:
        body = client.get(COST).json()

        assert body["compared_word"] == "onion"
        assert [entry["word"] for entry in body["rows_neighbours"]] == [
            "salt",
            "garlic",
            "butter",
            "pepper",
            "flour",
        ]
        assert all(entry["topic"] == "cooking" for entry in body["rows_neighbours"])
        assert [entry["word"] for entry in body["fitted_neighbours"]][:3] == [
            "gravity",
            "flour",
            "into",
        ]


class TestFit:
    def test_the_page_settings_reproduce_the_measurements(self) -> None:
        body = client.post(FIT, json={}).json()

        assert body["n_pairs"] == 281
        assert body["passes"][0]["total"] == pytest.approx(18.9335, abs=5e-4)
        assert body["objective_at_rest"] == pytest.approx(0.5148, abs=5e-4)
        assert body["within_topic"] == pytest.approx(0.1188, abs=5e-4)
        assert body["across_topic"] == pytest.approx(-0.0778, abs=5e-4)
        assert body["numbers_kept"] == 296

    def test_the_recorded_total_and_the_objective_agree_when_nothing_moves(
        self,
    ) -> None:
        """Two routes to one number, which are the same only where nothing moved.

        At a base step this small the parameters do not leave where they
        started, so the total accumulated during the pass and the objective
        recomputed afterwards are the same quantity.
        """
        body = client.post(FIT, json={"epochs": 1, "learning_rate": 1e-12}).json()
        recorded = body["passes"][0]["total"]

        assert recorded == pytest.approx(22.1272, abs=5e-4)
        assert body["objective_at_rest"] == pytest.approx(recorded, rel=1e-9)
        assert abs(body["objective_at_rest"] - recorded) < 1e-9

    def test_the_objective_at_rest_is_below_the_first_pass(self) -> None:
        body = client.post(FIT, json={}).json()

        assert body["objective_at_rest"] < body["passes"][0]["total"]
        assert body["passes"] == sorted(
            body["passes"], key=lambda entry: -entry["total"]
        )

    def test_the_nearest_words_to_the_moon_are_its_own_half_and_one_shared_word(
        self,
    ) -> None:
        body = client.post(FIT, json={"word": "moon", "n_neighbours": 5}).json()

        assert [entry["word"] for entry in body["neighbours"]] == [
            "orbits",
            "of",
            "planet",
            "astronomer",
            "orbit",
        ]
        assert [entry["topic"] for entry in body["neighbours"]] == [
            "astronomy",
            "shared",
            "astronomy",
            "astronomy",
            "astronomy",
        ]
        assert body["neighbours"][0]["similarity"] == pytest.approx(0.9826, abs=5e-4)

    def test_a_word_from_the_other_half_comes_back_mixed(self) -> None:
        body = client.post(FIT, json={"word": "chef", "n_neighbours": 5}).json()

        assert [entry["word"] for entry in body["neighbours"]] == [
            "stirs",
            "of",
            "orbits",
            "moon",
            "eclipse",
        ]

    def test_the_word_the_page_compares_lands_beside_the_wrong_half_first(self) -> None:
        body = client.post(FIT, json={"word": "onion", "n_neighbours": 5}).json()

        assert body["neighbours"][0]["word"] == "gravity"
        assert body["neighbours"][0]["topic"] == "astronomy"
        assert body["neighbours"][0]["similarity"] == pytest.approx(0.9404, abs=5e-4)

    def test_the_same_seed_reproduces_the_fit_to_the_last_bit(self) -> None:
        first = client.post(FIT, json={"epochs": 3, "random_seed": 7}).json()
        second = client.post(FIT, json={"epochs": 3, "random_seed": 7}).json()

        assert first["coordinates"] == second["coordinates"]
        assert first["passes"] == second["passes"]

    def test_the_released_rule_is_offered_here_too(self) -> None:
        body = client.post(FIT, json={"step_rule": "released"}).json()

        assert body["objective_at_rest"] == pytest.approx(4.9571, abs=5e-4)
        assert body["largest_divisor"] == pytest.approx(1.0860, abs=5e-4)

    def test_a_word_the_corpus_never_held_is_refused(self) -> None:
        response = client.post(FIT, json={"word": "elephant"})

        assert response.status_code == 422

    def test_a_width_beyond_the_bound_is_refused(self) -> None:
        assert client.post(FIT, json={"dimension": 99}).status_code == 422

    def test_an_unknown_keyword_is_refused(self) -> None:
        assert client.post(FIT, json={"nonsense": 1}).status_code == 422

    @pytest.mark.parametrize(
        "keyword",
        [
            {"window": 0},
            {"epochs": 0},
            {"learning_rate": 0.0},
            {"maximum_count": 0.0},
            {"weighting_exponent": 1.5},
        ],
    )
    def test_an_out_of_range_setting_is_refused(
        self, keyword: dict[str, object]
    ) -> None:
        assert client.post(FIT, json=keyword).status_code == 422


class TestTheArithmeticIsTheLibrarys:
    def test_the_weight_the_endpoint_reports_is_the_librarys_own(self) -> None:
        body = client.post(
            WEIGHTING, json={"maximum_count": 100.0, "weighting_exponent": 0.75}
        ).json()
        model = GloVe(maximum_count=100.0, weighting_exponent=0.75)

        for pair in body["pairs"][:20]:
            assert pair["weight"] == pytest.approx(model.weight_of_count(pair["count"]))

    def test_the_objective_at_rest_is_the_librarys_own_sum(self) -> None:
        body = client.post(FIT, json={"epochs": 2}).json()
        model = GloVe(
            dimension=PAGE_DIMENSION,
            window=PAGE_WINDOW,
            epochs=2,
            learning_rate=PAGE_LEARNING_RATE,
            random_seed=PAGE_SEED,
        )
        walk = _walk(model, _matrix("corpus", PAGE_WINDOW, "harmonic"), "accumulating")

        assert body["objective_at_rest"] == pytest.approx(
            model.objective_of(
                _matrix("corpus", PAGE_WINDOW, "harmonic"), walk.parameters
            )
        )

    def test_the_ratio_is_the_two_conditional_shares_divided(self) -> None:
        body = client.post(RATIOS, json={}).json()
        matrix = _matrix("corpus", PAGE_WINDOW, "harmonic")
        counts = np.asarray(matrix)
        totals = np.asarray(matrix.word_totals)
        first = matrix.vocabulary.id_of("chef")

        for row in body["rows"]:
            probe = matrix.vocabulary.id_of(row["probe"])
            assert row["first_probability"] == pytest.approx(
                float(counts[first, probe] / totals[first])
            )


IDENTIFIER = re.compile(r"\b(?:[a-z]+_[a-z_]+|[A-Z][a-z]+[A-Z][A-Za-z]*)\b")
"""A snake-cased or camel-cased name, which no string a reader sees may hold."""

FORBIDDEN = ("oop_ml", "the library", "the API", "Error")
"""Words that would credit the code behind the page rather than describe it."""


class TestNothingAReaderSeesNamesTheCode:
    """Every string a widget renders is page text, so it is checked like text.

    The only strings that reach a reader from these endpoints are words of the
    corpus, the two weighting names, the two step rule names, the topic labels
    and the three verdicts, so this asserts exactly that and fails if a new
    field starts carrying prose.
    """

    def test_the_words_that_cross_the_wire_hold_no_identifier(self) -> None:
        bodies = [
            client.post(FIT, json={"epochs": 1}).json(),
            client.post(TABLE, json={"text": "corpus"}).json(),
            client.post(RATIOS, json={}).json(),
            client.post(WEIGHTING, json={"maximum_count": 7.5}).json(),
            client.get(STEP_RULES).json(),
            client.get(WINDOWS).json(),
            client.get(COST).json(),
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
        return [text for entry in value.values() for text in _strings(entry)]
    if isinstance(value, list):
        return [text for entry in value for text in _strings(entry)]
    return []
