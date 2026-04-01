"""The three ensemble endpoints, pinned to the pages' quoted numbers.

The bagging and forest pages quote a root census at 25 members under the fixed
seed, twenty of twenty-five bagged trees opening on height while the forest
spreads to eight on height and seventeen on weight, and both quote the gap
between the training accuracy and the out-of-bag score on the tangled crowd.
The boosting page works two rounds on three points by hand, predictions 4, 7,
7 after one round and 3.25, 6.25, 8.5 after two. All pinned here.
"""

from itertools import pairwise
from typing import ClassVar

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BAGGING = "/concepts/ensembles/bagging"
FOREST = "/concepts/ensembles/random-forest"
BOOSTING = "/concepts/ensembles/gradient-boosting"

WORKED = [
    {"x": 147, "y": 41, "label": 0},
    {"x": 156, "y": 53, "label": 1},
    {"x": 145, "y": 57, "label": 0},
    {"x": 159, "y": 57, "label": 1},
    {"x": 162, "y": 61, "label": 1},
    {"x": 120, "y": 25, "label": 0},
    {"x": 122, "y": 28, "label": 0},
    {"x": 118, "y": 24, "label": 0},
    {"x": 180, "y": 80, "label": 1},
    {"x": 183, "y": 83, "label": 1},
    {"x": 178, "y": 78, "label": 1},
]

TANGLED = WORKED + [
    {"x": 145, "y": 45, "label": 0},
    {"x": 145, "y": 55, "label": 1},
    {"x": 151, "y": 45, "label": 1},
    {"x": 151, "y": 55, "label": 0},
    {"x": 157, "y": 45, "label": 0},
    {"x": 157, "y": 55, "label": 1},
    {"x": 148, "y": 50, "label": 1},
    {"x": 154, "y": 50, "label": 0},
    {"x": 160, "y": 50, "label": 1},
    {"x": 147, "y": 58, "label": 0},
    {"x": 153, "y": 58, "label": 1},
    {"x": 150, "y": 42, "label": 1},
    {"x": 156, "y": 42, "label": 0},
    {"x": 143, "y": 50, "label": 0},
]

THREE_POINTS = [{"x": 1, "y": 2}, {"x": 2, "y": 6}, {"x": 3, "y": 10}]


class TestBagging:
    def test_the_quoted_root_census(self) -> None:
        body = client.post(BAGGING, json={"points": TANGLED, "n_members": 25}).json()

        assert body["roots_on_height"] == 20
        assert body["roots_on_weight"] == 5

    def test_training_flatters_and_the_out_of_bag_score_does_not(self) -> None:
        body = client.post(BAGGING, json={"points": TANGLED, "n_members": 25}).json()

        assert body["accuracy"] > 0.9
        assert body["out_of_bag_score"] < body["accuracy"] - 0.2

    def test_one_member_still_answers(self) -> None:
        body = client.post(BAGGING, json={"points": TANGLED, "n_members": 1}).json()

        assert 0.0 <= body["out_of_bag_score"] <= 1.0
        assert len(body["regions"]["labels"]) == body["regions"]["cells"]

    def test_a_committee_past_the_cap_is_refused(self) -> None:
        response = client.post(BAGGING, json={"points": TANGLED, "n_members": 51})

        assert response.status_code == 422


class TestRandomForest:
    def test_restriction_spreads_the_roots(self) -> None:
        body = client.post(FOREST, json={"points": TANGLED, "n_members": 25}).json()

        assert body["roots_on_height"] == 8
        assert body["roots_on_weight"] == 17

    def test_the_forest_answers_the_same_shape(self) -> None:
        body = client.post(FOREST, json={"points": TANGLED, "n_members": 25}).json()

        assert body["accuracy"] > 0.9
        assert 0.0 <= body["out_of_bag_score"] <= 1.0


class TestGradientBoosting:
    def test_one_round_matches_the_hand_arithmetic(self) -> None:
        body = client.post(
            BOOSTING,
            json={"points": THREE_POINTS, "rounds": 1, "learning_rate": 0.5},
        ).json()

        assert body["predictions"] == [4.0, 7.0, 7.0]

    def test_two_rounds_match_the_hand_arithmetic(self) -> None:
        body = client.post(
            BOOSTING,
            json={"points": THREE_POINTS, "rounds": 2, "learning_rate": 0.5},
        ).json()

        assert body["predictions"] == [3.25, 6.25, 8.5]

    def test_more_rounds_fit_the_ball_better(self) -> None:
        ball = [
            {"x": 0.0, "y": 0.7},
            {"x": 0.57, "y": 9.5},
            {"x": 1.14, "y": 16.9},
            {"x": 1.71, "y": 19.4},
            {"x": 2.29, "y": 19.6},
            {"x": 2.86, "y": 16.8},
            {"x": 3.43, "y": 11.1},
            {"x": 4.0, "y": 1.2},
        ]
        few = client.post(
            BOOSTING, json={"points": ball, "rounds": 3, "learning_rate": 0.3}
        ).json()
        many = client.post(
            BOOSTING, json={"points": ball, "rounds": 100, "learning_rate": 0.3}
        ).json()

        assert many["r_squared"] > few["r_squared"]
        assert many["r_squared"] > 0.95

    def test_a_rate_past_one_is_refused_in_words(self) -> None:
        response = client.post(
            BOOSTING,
            json={"points": THREE_POINTS, "rounds": 5, "learning_rate": 1.5},
        )

        assert response.status_code == 422
        assert "overshoots" in response.json()["detail"]

    def test_a_round_count_past_the_cap_is_refused(self) -> None:
        response = client.post(
            BOOSTING,
            json={"points": THREE_POINTS, "rounds": 301, "learning_rate": 0.3},
        )

        assert response.status_code == 422


class TestLeaveOutCurve:
    """The bootstrap's missing share, traced against the crowd size."""

    def test_the_worked_numbers_sit_on_the_curve(self) -> None:
        response = client.post(
            "/concepts/ensembles/leave-out-curve", json={"max_crowd": 200}
        )

        body = response.json()
        by_size = dict(zip(body["sizes"], body["probabilities"], strict=True))
        assert round(by_size[25], 4) == 0.3604
        assert round(body["limit"], 4) == 0.3679

    def test_the_curve_settles_toward_the_limit(self) -> None:
        response = client.post(
            "/concepts/ensembles/leave-out-curve", json={"max_crowd": 500}
        )

        body = response.json()
        gap_early = abs(body["probabilities"][10] - body["limit"])
        gap_late = abs(body["probabilities"][-1] - body["limit"])
        assert gap_late < gap_early
        assert gap_late < 0.001


class TestCommitteeVariance:
    """The variance formula's curves, and the floors they cannot pass."""

    def test_each_curve_holds_its_floor(self) -> None:
        response = client.post(
            "/concepts/ensembles/committee-variance", json={"max_members": 200}
        )

        body = response.json()
        for curve in body["curves"]:
            floor = curve["correlation"]
            assert all(value > floor for value in curve["variances"])
            assert abs(curve["variances"][-1] - floor) <= 0.005

    def test_a_lone_member_always_costs_the_whole_variance(self) -> None:
        response = client.post(
            "/concepts/ensembles/committee-variance", json={"max_members": 10}
        )

        for curve in response.json()["curves"]:
            assert curve["variances"][0] == 1.0


class TestBoostingDescent:
    """The leftovers falling round by round, the descent chart's staircase."""

    THROWN_BALL: ClassVar[list[dict]] = [
        {"x": time / 2, "y": 20 * (time / 2) - 4.9 * (time / 2) ** 2}
        for time in range(9)
    ]

    def test_the_leftovers_never_grow(self) -> None:
        response = client.post(
            "/concepts/ensembles/boosting-descent",
            json={"points": self.THROWN_BALL, "rounds": 30, "learning_rate": 0.5},
        )

        body = response.json()
        sums = body["residual_sums"]
        assert body["rounds"] == list(range(31))
        assert all(later <= earlier + 1e-9 for earlier, later in pairwise(sums))

    def test_round_zero_is_the_flat_mean_model(self) -> None:
        response = client.post(
            "/concepts/ensembles/boosting-descent",
            json={"points": self.THROWN_BALL, "rounds": 5, "learning_rate": 0.5},
        )

        heights = [point["y"] for point in self.THROWN_BALL]
        mean = sum(heights) / len(heights)
        spread = sum((value - mean) ** 2 for value in heights)
        assert abs(response.json()["residual_sums"][0] - spread) < 1e-6

    def test_a_rate_past_one_is_refused_here_too(self) -> None:
        response = client.post(
            "/concepts/ensembles/boosting-descent",
            json={"points": self.THROWN_BALL, "rounds": 5, "learning_rate": 1.5},
        )

        assert response.status_code == 422
