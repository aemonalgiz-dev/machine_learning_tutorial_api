"""The curved-regression and penalty endpoints, pinned to the pages' numbers.

The polynomial page's worked example throws a ball whose five exact points
recover the physics, 20 per second and minus 4.9 per second squared, and the
penalty page's worked shrinkage says a raw one-feature ridge slope is the
ordinary slope with the penalty added to its denominator. Both claims are
pinned here so the prose cannot drift from what the library computes.
"""

from itertools import pairwise
from typing import ClassVar

from fastapi.testclient import TestClient
from oop_ml import Feature, RidgeRegression

from oop_ml_api.main import create_app

client = TestClient(create_app())

POLYNOMIAL = "/concepts/multiple-polynomial-regression/fit"
PENALISED = "/concepts/ridge-lasso/fit"

BALL = [
    {"x": 0, "y": 0},
    {"x": 1, "y": 15.1},
    {"x": 2, "y": 20.4},
    {"x": 3, "y": 15.9},
    {"x": 4, "y": 1.6},
]

# Fifteen fixed noisy throws for the degree-9 fits, h = 20t - 4.9t^2 plus a
# small wobble, precomputed so the tests are deterministic.
NOISY_BALL = [
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


class TestPolynomial:
    def test_the_exact_ball_recovers_the_physics_at_degree_two(self) -> None:
        body = client.post(POLYNOMIAL, json={"points": BALL, "degree": 2}).json()

        named = {each["name"]: each["value"] for each in body["coefficients"]}
        assert abs(named["t"] - 20.0) < 1e-9
        assert abs(named["t^2"] + 4.9) < 1e-9
        assert abs(body["intercept"]) < 1e-9
        assert abs(body["r_squared"] - 1.0) < 1e-12

    def test_a_straight_line_explains_almost_none_of_the_arc(self) -> None:
        body = client.post(POLYNOMIAL, json={"points": BALL, "degree": 1}).json()

        assert body["r_squared"] < 0.01

    def test_the_curve_is_sampled_across_the_data(self) -> None:
        body = client.post(POLYNOMIAL, json={"points": BALL, "degree": 2}).json()

        assert len(body["curve"]) == 81
        assert body["curve"][0]["x"] == 0.0
        assert body["curve"][-1]["x"] == 4.0

    def test_too_few_points_for_the_degree_is_refused_in_words(self) -> None:
        response = client.post(POLYNOMIAL, json={"points": BALL, "degree": 5})

        assert response.status_code == 422
        assert "needs at least" in response.json()["detail"]

    def test_a_degree_past_the_cap_is_refused(self) -> None:
        response = client.post(POLYNOMIAL, json={"points": NOISY_BALL, "degree": 10})

        assert response.status_code == 422


class TestPenalised:
    def test_ridge_still_fits_the_arc_well_at_a_small_penalty(self) -> None:
        body = client.post(
            PENALISED,
            json={"points": NOISY_BALL, "model": "ridge", "penalty": 0.01},
        ).json()

        assert body["r_squared"] > 0.95
        assert body["nonzero_count"] == 9
        assert len(body["curve"]) == 81

    def test_ridge_shrinks_but_never_switches_off(self) -> None:
        body = client.post(
            PENALISED,
            json={"points": NOISY_BALL, "model": "ridge", "penalty": 100.0},
        ).json()

        assert body["nonzero_count"] == 9

    def test_lasso_switches_terms_off_as_the_penalty_grows(self) -> None:
        gentle = client.post(
            PENALISED,
            json={"points": NOISY_BALL, "model": "lasso", "penalty": 0.01},
        ).json()
        firm = client.post(
            PENALISED,
            json={"points": NOISY_BALL, "model": "lasso", "penalty": 20.0},
        ).json()

        assert firm["nonzero_count"] < gentle["nonzero_count"]

    def test_a_crushing_lasso_penalty_switches_everything_off(self) -> None:
        body = client.post(
            PENALISED,
            json={"points": NOISY_BALL, "model": "lasso", "penalty": 1000.0},
        ).json()

        assert body["nonzero_count"] == 0

    def test_an_unknown_model_is_refused(self) -> None:
        response = client.post(
            PENALISED,
            json={"points": NOISY_BALL, "model": "elastic", "penalty": 1.0},
        )

        assert response.status_code == 422


class TestWorkedShrinkage:
    """The penalty page's hand arithmetic, pinned against the library itself.

    On the regression page's five people, a raw one-feature ridge slope is
    the covariance sum over the squares sum with the penalty added to the
    denominator, 200 over 250 plus the penalty.
    """

    HEIGHTS: ClassVar[list[float]] = [160.0, 165.0, 170.0, 175.0, 180.0]
    WEIGHTS: ClassVar[list[float]] = [58.0, 66.0, 68.0, 74.0, 74.0]

    def test_the_slope_shrinks_by_the_penalty_in_the_denominator(self) -> None:
        for penalty in (0.0, 50.0, 250.0):
            model = RidgeRegression(penalty=penalty).fit(
                [Feature("height", self.HEIGHTS)], Feature("weight", self.WEIGHTS)
            )
            slope = next(iter(model.coefficients)).value

            assert abs(slope - 200.0 / (250.0 + penalty)) < 1e-9


class TestDegreeSweep:
    """The staircase of fits the degree chart draws."""

    FIVE_MEASURED_PEOPLE: ClassVar[list[dict]] = [
        {"x": 160.0, "y": 58.0},
        {"x": 165.0, "y": 66.0},
        {"x": 170.0, "y": 68.0},
        {"x": 175.0, "y": 74.0},
        {"x": 180.0, "y": 74.0},
    ]

    def test_r_squared_never_falls_as_the_degree_climbs(self) -> None:
        response = client.post(
            "/concepts/multiple-polynomial-regression/degree-sweep",
            json={"points": self.FIVE_MEASURED_PEOPLE, "max_degree": 6},
        )

        scores = [entry["r_squared"] for entry in response.json()["scores"]]
        assert all(later >= earlier - 1e-9 for earlier, later in pairwise(scores))

    def test_the_sweep_stops_where_the_points_run_out(self) -> None:
        response = client.post(
            "/concepts/multiple-polynomial-regression/degree-sweep",
            json={"points": self.FIVE_MEASURED_PEOPLE, "max_degree": 9},
        )

        degrees = [entry["degree"] for entry in response.json()["scores"]]
        assert degrees == [1, 2, 3, 4]

    def test_enough_points_reach_a_perfect_interpolation(self) -> None:
        response = client.post(
            "/concepts/multiple-polynomial-regression/degree-sweep",
            json={"points": self.FIVE_MEASURED_PEOPLE, "max_degree": 4},
        )

        final = response.json()["scores"][-1]
        assert final["degree"] == 4
        assert final["r_squared"] > 0.9999


class TestShrinkagePath:
    """The two paths the ridge and lasso chart traces on the worked five."""

    FIVE_MEASURED_PEOPLE: ClassVar[list[dict]] = [
        {"x": 160.0, "y": 58.0},
        {"x": 165.0, "y": 66.0},
        {"x": 170.0, "y": 68.0},
        {"x": 175.0, "y": 74.0},
        {"x": 180.0, "y": 74.0},
    ]

    def path(self) -> dict:
        response = client.post(
            "/concepts/ridge-lasso/shrinkage-path",
            json={"points": self.FIVE_MEASURED_PEOPLE},
        )
        assert response.status_code == 200
        return response.json()

    def test_the_ridge_path_is_the_worked_formula(self) -> None:
        body = self.path()
        for penalty, slope in zip(body["penalties"], body["ridge_slopes"], strict=True):
            assert abs(slope - 200.0 / (250.0 + penalty)) < 1e-9

    def test_ridge_never_reaches_zero_and_lasso_does(self) -> None:
        body = self.path()
        assert all(slope > 0.0 for slope in body["ridge_slopes"])
        by_penalty = dict(zip(body["penalties"], body["lasso_slopes"], strict=True))
        assert abs(by_penalty[0.0] - 0.8) < 1e-6
        assert by_penalty[400.0] == 0.0
        assert by_penalty[500.0] == 0.0

    def test_the_lasso_path_subtracts_where_ridge_divides(self) -> None:
        body = self.path()
        by_penalty = dict(zip(body["penalties"], body["lasso_slopes"], strict=True))
        assert abs(by_penalty[100.0] - 0.6) < 1e-6
        assert abs(by_penalty[200.0] - 0.4) < 1e-6
