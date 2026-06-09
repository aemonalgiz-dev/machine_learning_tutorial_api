"""The evaluation endpoints, end to end through HTTP.

The page's whole argument is a set of numbers, the training score that only
climbs, the held-out score that turns around and collapses, and the fold
spread that widens exactly where the mean stops deserving trust. These tests
pin that argument on the fixed seeded split so the prose and the widgets
cannot drift from what the API serves.
"""

from itertools import pairwise

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

NOISY_THROW: list[dict] = [
    {"x": 0.0, "y": 0.7},
    {"x": 0.29, "y": 5.2},
    {"x": 0.57, "y": 9.5},
    {"x": 0.86, "y": 13.0},
    {"x": 1.14, "y": 16.9},
    {"x": 1.43, "y": 18.1},
    {"x": 1.71, "y": 19.4},
    {"x": 2.0, "y": 21.0},
    {"x": 2.29, "y": 19.6},
    {"x": 2.57, "y": 19.1},
    {"x": 2.86, "y": 16.8},
    {"x": 3.14, "y": 13.6},
    {"x": 3.43, "y": 11.1},
    {"x": 3.71, "y": 6.2},
    {"x": 4.0, "y": 1.2},
]


class TestSplitFit:
    """One degree fitted on the training share and judged twice."""

    def test_the_seeded_split_is_stable(self) -> None:
        response = client.post(
            "/concepts/evaluation/split-fit",
            json={"points": NOISY_THROW, "degree": 2},
        )

        body = response.json()
        assert body["held_out_indices"] == [0, 1, 10, 12]

    def test_an_honest_degree_scores_well_on_both_shares(self) -> None:
        response = client.post(
            "/concepts/evaluation/split-fit",
            json={"points": NOISY_THROW, "degree": 2},
        )

        body = response.json()
        assert round(body["train_r_squared"], 3) == 0.995
        assert round(body["held_out_r_squared"], 3) == 0.989

    def test_a_wild_degree_flatters_only_the_share_it_saw(self) -> None:
        response = client.post(
            "/concepts/evaluation/split-fit",
            json={"points": NOISY_THROW, "degree": 9},
        )

        body = response.json()
        assert body["train_r_squared"] > 0.99
        assert body["held_out_r_squared"] < -1000.0

    def test_a_degree_the_training_share_cannot_pin_is_refused(self) -> None:
        response = client.post(
            "/concepts/evaluation/split-fit",
            json={"points": NOISY_THROW[:8], "degree": 6},
        )

        assert response.status_code == 422
        assert "training share" in response.json()["detail"]


class TestDegreeCurve:
    """The two score curves the gap chart draws."""

    def curve(self) -> dict:
        response = client.post(
            "/concepts/evaluation/degree-curve", json={"points": NOISY_THROW}
        )
        assert response.status_code == 200
        return response.json()

    def test_the_training_score_never_falls(self) -> None:
        scores = self.curve()["train_scores"]
        assert all(later >= earlier - 1e-9 for earlier, later in pairwise(scores))

    def test_the_held_out_score_turns_around(self) -> None:
        body = self.curve()
        held = dict(zip(body["degrees"], body["held_out_scores"], strict=True))
        assert held[1] < 0.0
        assert held[2] > 0.98
        assert held[9] < -1000.0

    def test_both_curves_share_the_degrees(self) -> None:
        body = self.curve()
        assert body["degrees"] == list(range(1, 10))
        assert len(body["train_scores"]) == len(body["held_out_scores"]) == 9


class TestValidationCurve:
    """Five folds per degree, and the spread that prices the mean."""

    def validations(self) -> list[dict]:
        response = client.post(
            "/concepts/evaluation/validation-curve", json={"points": NOISY_THROW}
        )
        assert response.status_code == 200
        return response.json()["validations"]

    def by_degree(self) -> dict[int, dict]:
        return {entry["degree"]: entry for entry in self.validations()}

    def test_the_honest_degrees_agree_with_the_split(self) -> None:
        entries = self.by_degree()
        assert round(entries[2]["mean_r_squared"], 3) == 0.989
        assert entries[2]["spread"] < 0.05

    def test_a_wild_degree_is_priced_by_its_spread(self) -> None:
        entries = self.by_degree()
        assert entries[8]["mean_r_squared"] < 0.0
        assert entries[8]["spread"] > 100 * entries[2]["spread"]

    def test_every_degree_the_folds_can_pin_is_present(self) -> None:
        degrees = [entry["degree"] for entry in self.validations()]
        assert degrees == list(range(1, 10))


class TestSimilarityCurve:
    """The kernel chart's four curves, labelled by the library itself."""

    def test_the_labels_are_the_kernels_own_descriptions(self) -> None:
        response = client.post("/concepts/kernel-trick/similarity-curve", json={})

        labels = [curve["label"] for curve in response.json()["curves"]]
        assert labels[0] == "a . b"
        assert "** 2" in labels[1]
        assert labels[2].startswith("exp(")
        assert labels[3].startswith("exp(")

    def test_only_the_radial_curves_forget_the_far_away(self) -> None:
        response = client.post("/concepts/kernel-trick/similarity-curve", json={})

        body = response.json()
        far_left = 0
        linear = body["curves"][0]["values"]
        assert abs(linear[far_left]) > 1.0
        for curve in body["curves"][2:]:
            assert abs(curve["values"][far_left]) < 1e-6

    def test_the_radial_curves_peak_at_the_reference(self) -> None:
        response = client.post("/concepts/kernel-trick/similarity-curve", json={})

        body = response.json()
        positions = body["positions"]
        for curve in body["curves"][2:]:
            peak_position = positions[curve["values"].index(max(curve["values"]))]
            assert abs(peak_position - body["reference"]) < 0.06


OVERLAPPING_CROWD: list[dict] = [
    {"x": 118, "y": 24, "label": 0},
    {"x": 120, "y": 25, "label": 0},
    {"x": 122, "y": 28, "label": 0},
    {"x": 125, "y": 31, "label": 0},
    {"x": 140, "y": 45, "label": 0},
    {"x": 168, "y": 66, "label": 0},
    {"x": 150, "y": 50, "label": 1},
    {"x": 159, "y": 57, "label": 1},
    {"x": 162, "y": 61, "label": 1},
    {"x": 178, "y": 78, "label": 1},
    {"x": 180, "y": 80, "label": 1},
    {"x": 183, "y": 83, "label": 1},
]


class TestSplitFamily:
    """The same fit judged under thirty deals, and how far the verdict moves."""

    def family(self, **overrides) -> dict:
        response = client.post(
            "/concepts/evaluation/split-family",
            json={"points": NOISY_THROW, "degree": 2, **overrides},
        )
        assert response.status_code == 200
        return response.json()

    def test_the_default_deal_holds_four_of_fifteen_back(self) -> None:
        body = self.family()
        assert body["n_training"] == 11
        assert body["n_held_out"] == 4
        assert len(body["splits"]) == 30
        assert all(len(split["held_out_indices"]) == 4 for split in body["splits"])

    def test_an_honest_degree_wobbles_a_little_with_the_seed(self) -> None:
        body = self.family()
        assert round(body["mean_train_r_squared"], 4) == 0.9959
        assert round(body["mean_held_out_r_squared"], 4) == 0.9883
        assert round(body["lowest_held_out_r_squared"], 4) == 0.9549
        assert round(body["highest_held_out_r_squared"], 4) == 0.9979
        assert round(body["held_out_spread"], 3) == 0.043

    def test_a_wilder_degree_wobbles_far_more(self) -> None:
        body = self.family(degree=5)
        assert round(body["lowest_held_out_r_squared"], 4) == 0.3552
        assert round(body["held_out_spread"], 3) == 0.643

    def test_two_held_out_rows_make_the_ratio_a_gamble(self) -> None:
        body = self.family(held_out_fraction=0.1)
        assert body["n_held_out"] == 2
        assert body["lowest_held_out_r_squared"] < -8.0

    def test_a_larger_held_out_share_steadies_the_verdict(self) -> None:
        body = self.family(held_out_fraction=0.5)
        assert body["n_training"] == 7
        assert body["n_held_out"] == 8
        assert round(body["held_out_spread"], 4) == 0.0307

    def test_a_single_held_out_row_is_refused_by_name(self) -> None:
        response = client.post(
            "/concepts/evaluation/split-family",
            json={"points": NOISY_THROW, "degree": 2, "held_out_fraction": 0.01},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "UndefinedMetricError"

    def test_a_family_beyond_thirty_seeds_is_refused_at_the_door(self) -> None:
        response = client.post(
            "/concepts/evaluation/split-family",
            json={"points": NOISY_THROW, "degree": 2, "n_seeds": 31},
        )
        assert response.status_code == 422


class TestFoldDeal:
    """The folds dealt one at a time, and the two summaries across them."""

    def deal(self, **overrides) -> dict:
        response = client.post(
            "/concepts/evaluation/fold-deal",
            json={"points": NOISY_THROW, "degree": 2, **overrides},
        )
        assert response.status_code == 200
        return response.json()

    def test_five_folds_hold_every_row_out_exactly_once(self) -> None:
        body = self.deal()
        held = [fold["held_out_indices"] for fold in body["folds"]]
        assert held == [[1, 10, 12], [0, 2, 7], [8, 9, 13], [3, 4, 6], [5, 11, 14]]
        assert sorted(index for fold in held for index in fold) == list(range(15))

    def test_the_mean_across_folds_matches_the_validation_curve(self) -> None:
        body = self.deal()
        assert round(body["mean_r_squared"], 4) == 0.9888
        assert round(body["spread"], 4) == 0.0319
        assert body["n_fits"] == 5

    def test_the_pooled_score_is_a_different_number(self) -> None:
        body = self.deal()
        assert round(body["pooled_r_squared"], 4) == 0.9943
        assert round(body["pooled_mean_squared_error"], 4) == 0.2481

    def test_a_wild_degree_spreads_the_folds_apart(self) -> None:
        body = self.deal(degree=8)
        assert round(body["mean_r_squared"], 3) == -1.44
        assert round(body["spread"], 2) == 11.27
        assert min(fold["r_squared"] for fold in body["folds"]) < -10.0

    def test_leave_one_out_has_no_fold_score_and_still_pools(self) -> None:
        body = self.deal(n_folds=15)
        assert all(fold["r_squared"] is None for fold in body["folds"])
        assert body["mean_r_squared"] is None
        assert body["spread"] is None
        assert round(body["pooled_r_squared"], 4) == 0.9932
        assert round(body["pooled_mean_squared_error"], 4) == 0.2951

    def test_more_folds_than_rows_is_the_library_refusal(self) -> None:
        response = client.post(
            "/concepts/evaluation/fold-deal",
            json={"points": NOISY_THROW, "degree": 2, "n_folds": 16},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "TooFewValuesError"

    def test_a_training_fold_too_small_for_the_degree_is_refused(self) -> None:
        response = client.post(
            "/concepts/evaluation/fold-deal",
            json={"points": NOISY_THROW[:6], "degree": 5, "n_folds": 3},
        )
        assert response.status_code == 422
        assert "training share" in response.json()["detail"]

    def test_a_constant_target_leaves_nothing_to_explain(self) -> None:
        flat = [{"x": point["x"], "y": 5.0} for point in NOISY_THROW]
        response = client.post(
            "/concepts/evaluation/fold-deal", json={"points": flat, "degree": 2}
        )
        assert response.status_code == 422
        assert response.json()["error"] == "UndefinedMetricError"


class TestClassifierFolds:
    """The crowd dealt plain and stratified, and the pooling that survives it."""

    def folds(self, **overrides) -> dict:
        response = client.post(
            "/concepts/evaluation/classifier-folds",
            json={"points": OVERLAPPING_CROWD, "n_folds": 5, **overrides},
        )
        assert response.status_code == 200
        return response.json()

    def test_the_plain_deal_leaves_two_folds_without_a_class(self) -> None:
        body = self.folds(stratified=False)
        assert body["classes_missing_from_a_fold"] == 2
        assert [fold["positives_held_out"] for fold in body["folds"]] == [1, 2, 2, 0, 1]
        assert body["folds_with_defined_recall"] == 4

    def test_averaging_and_pooling_disagree_on_the_plain_deal(self) -> None:
        body = self.folds(stratified=False)
        assert round(body["mean_fold_accuracy"], 4) == 0.4667
        assert body["pooled_accuracy"] == 0.5
        assert body["accuracy_spread"] == 1.0
        assert body["mean_defined_recall"] == 0.5
        assert round(body["pooled_recall"], 4) == 0.3333
        assert body["pooled_true_positives"] == 2
        assert body["pooled_actual_positives"] == 6

    def test_the_stratified_deal_puts_every_class_in_every_fold(self) -> None:
        body = self.folds(stratified=True)
        assert body["classes_missing_from_a_fold"] == 0
        assert [fold["positives_held_out"] for fold in body["folds"]] == [1, 2, 1, 1, 1]
        assert round(body["mean_fold_accuracy"], 4) == 0.8
        assert round(body["pooled_accuracy"], 4) == 0.8333
        assert body["pooled_recall"] == 1.0

    def test_stratifying_cannot_manufacture_a_rare_class(self) -> None:
        one_adult = [dict(point, label=0) for point in OVERLAPPING_CROWD[:-1]] + [
            OVERLAPPING_CROWD[-1]
        ]
        body = self.folds(stratified=True, seed=3, points=one_adult)
        assert body["classes_missing_from_a_fold"] == 4
        assert body["folds_with_defined_recall"] == 1

    def test_a_single_class_is_fitted_and_its_recall_is_undefined(self) -> None:
        children = [dict(point, label=0) for point in OVERLAPPING_CROWD]
        body = self.folds(points=children, n_folds=3)
        assert body["pooled_accuracy"] == 1.0
        assert body["pooled_recall"] is None
        assert body["mean_defined_recall"] is None

    def test_more_folds_than_people_is_refused_by_name(self) -> None:
        response = client.post(
            "/concepts/evaluation/classifier-folds",
            json={"points": OVERLAPPING_CROWD, "n_folds": 13},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "TooFewValuesError"


class TestLeak:
    """The standardiser fitted before and inside the folds, on the throw."""

    def leak(self, **overrides) -> dict:
        response = client.post(
            "/concepts/evaluation/leak",
            json={"points": NOISY_THROW, "degree": 2, "penalty": 0.01, **overrides},
        )
        assert response.status_code == 200
        return response.json()

    def test_the_least_squares_control_cannot_feel_the_leak(self) -> None:
        control = self.leak()["least_squares"]
        assert control["largest_absolute_gap"] < 1e-12
        assert round(control["mean_inside"], 4) == 0.9712

    def test_the_ridge_leak_is_small_and_has_no_direction(self) -> None:
        ridge = self.leak()["ridge"]
        assert len(ridge["outside_scores"]) == 30
        assert round(ridge["flattered_by"], 4) == 0.0007
        assert ridge["seeds_flattered"] == 15
        assert round(ridge["largest_absolute_gap"], 3) == 0.019
        assert round(ridge["mean_inside"], 4) == 0.9669

    def test_a_negative_penalty_is_refused_at_the_door(self) -> None:
        response = client.post(
            "/concepts/evaluation/leak",
            json={"points": NOISY_THROW, "penalty": -1.0},
        )
        assert response.status_code == 422

    def test_too_few_rows_for_five_folds_is_refused_by_name(self) -> None:
        response = client.post(
            "/concepts/evaluation/leak", json={"points": NOISY_THROW[:6]}
        )
        assert response.status_code == 422
        assert response.json()["error"] == "UndefinedMetricError"
