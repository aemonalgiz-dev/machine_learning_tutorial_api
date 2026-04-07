"""The kernel ridge endpoints, pinned to the page's hand arithmetic.

The page works three points, (1, 6), (2, 12) and (3, 18), at a penalty of
one. Centred, the inputs are -1, 0 and 1 and the targets -6, 0 and 6, so
ridge's slope is 12 / (2 + 1) = 4 with an intercept of 4, and the linear
kernel's dual weights come out as -2, 0 and 2, which add up through the
centred inputs to the same slope of 4. Both answer 14 at x = 2.5. The radial
kernel at gamma 1 answers 14.039 there with dual weights of -3.028, 0 and
3.028, and the squared kernel answers 14.4. The centring section quotes the
half-centred variant's slope of 0.8, its line standing at 13.6 over the mean
where the true lines stand at 12, and at a penalty of a thousandth its 0.857
against the right slope's 6. The thrown ball from the polynomial page pins the
ridge line's 0.003 against the radial kernel's 0.873 at a penalty of one and
0.987 at a tenth, and the squared kernel's 0.981. All pinned here.
"""

from typing import ClassVar

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

FIT = "/concepts/kernel-ridge/fit"
CENTRING = "/concepts/kernel-ridge/centring"

WORKED_THREE = [
    {"x": 1.0, "y": 6.0},
    {"x": 2.0, "y": 12.0},
    {"x": 3.0, "y": 18.0},
]

THROWN_BALL = [
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

LINEAR = {"name": "linear"}
RADIAL_AT_ONE = {"name": "rbf", "gamma": 1.0}
SQUARED = {"name": "polynomial", "degree": 2}

CURVE_SAMPLES = 81


def fit_of(points: list[dict], kernel: dict, penalty: float, query_x: float) -> dict:
    response = client.post(
        FIT,
        json={
            "points": points,
            "kernel": kernel,
            "penalty": penalty,
            "query_x": query_x,
        },
    )
    assert response.status_code == 200, response.json()
    return response.json()


def centring_of(points: list[dict], penalty: float) -> dict:
    response = client.post(CENTRING, json={"points": points, "penalty": penalty})
    assert response.status_code == 200, response.json()
    return response.json()


class TestTheLinearKernelControl:
    """Under the linear kernel the fit is ridge regression, and the page says so."""

    def test_ridge_has_the_hand_worked_slope_and_intercept(self) -> None:
        body = fit_of(WORKED_THREE, LINEAR, 1.0, 2.5)

        assert body["ridge_slope"] == 4.0
        assert body["ridge_intercept"] == 4.0

    def test_the_dual_weights_are_minus_two_zero_and_two(self) -> None:
        body = fit_of(WORKED_THREE, LINEAR, 1.0, 2.5)

        assert [round(weight, 9) for weight in body["dual_weights"]] == [-2.0, 0.0, 2.0]

    def test_the_dual_weights_imply_the_slope_ridge_found(self) -> None:
        body = fit_of(WORKED_THREE, LINEAR, 1.0, 2.5)

        assert abs(body["implied_slope"] - 4.0) < 1e-9
        assert abs(body["implied_slope"] - body["ridge_slope"]) < 1e-9

    def test_the_means_the_fit_centred_on(self) -> None:
        body = fit_of(WORKED_THREE, LINEAR, 1.0, 2.5)

        assert body["feature_mean"] == 2.0
        assert body["target_mean"] == 12.0

    def test_the_two_curves_coincide(self) -> None:
        body = fit_of(WORKED_THREE, LINEAR, 1.0, 2.5)

        assert body["largest_gap"] < 1e-9
        assert len(body["curve"]) == len(body["ridge_curve"]) == CURVE_SAMPLES

    def test_both_answer_fourteen_at_two_and_a_half(self) -> None:
        body = fit_of(WORKED_THREE, LINEAR, 1.0, 2.5)

        assert abs(body["prediction_at_query"] - 14.0) < 1e-9
        assert abs(body["ridge_at_query"] - 14.0) < 1e-9

    def test_the_scores_agree_too(self) -> None:
        body = fit_of(WORKED_THREE, LINEAR, 1.0, 2.5)

        assert abs(body["r_squared"] - body["ridge_r_squared"]) < 1e-9

    def test_the_dual_weights_arrive_one_per_point_in_order(self) -> None:
        body = fit_of(THROWN_BALL, LINEAR, 1.0, 2.0)

        assert len(body["dual_weights"]) == len(THROWN_BALL)
        assert body["largest_gap"] < 1e-9


class TestTheRadialKernel:
    """The one prediction the worked example quotes off the straight line."""

    def test_it_answers_fourteen_point_zero_three_nine_at_two_and_a_half(self) -> None:
        body = fit_of(WORKED_THREE, RADIAL_AT_ONE, 1.0, 2.5)

        assert round(body["prediction_at_query"], 3) == 14.039
        assert abs(body["ridge_at_query"] - 14.0) < 1e-9

    def test_the_outer_rows_carry_the_fit_and_the_middle_row_nothing(self) -> None:
        body = fit_of(WORKED_THREE, RADIAL_AT_ONE, 1.0, 2.5)

        first, middle, last = body["dual_weights"]
        assert round(first, 3) == -3.028
        assert round(last, 3) == 3.028
        assert abs(middle) < 1e-9

    def test_no_slope_exists_to_report(self) -> None:
        body = fit_of(WORKED_THREE, RADIAL_AT_ONE, 1.0, 2.5)

        assert body["implied_slope"] is None

    def test_the_curve_is_not_the_line(self) -> None:
        body = fit_of(WORKED_THREE, RADIAL_AT_ONE, 1.0, 2.5)

        assert body["largest_gap"] > 1.0
        assert round(body["r_squared"], 3) == 0.745


class TestTheSquaredKernel:
    """The squared kernel's answer at the same x, for the worked example."""

    def test_it_answers_fourteen_point_four_at_two_and_a_half(self) -> None:
        body = fit_of(WORKED_THREE, SQUARED, 1.0, 2.5)

        assert abs(body["prediction_at_query"] - 14.4) < 1e-9

    def test_its_dual_weights_are_plus_and_minus_one_point_two(self) -> None:
        body = fit_of(WORKED_THREE, SQUARED, 1.0, 2.5)

        first, middle, last = body["dual_weights"]
        assert abs(first + 1.2) < 1e-9
        assert abs(middle) < 1e-9
        assert abs(last - 1.2) < 1e-9


class TestTheThrownBall:
    """The polynomial page's ball, fitted without one manufactured column."""

    def test_the_ridge_line_cannot_follow_the_arc(self) -> None:
        body = fit_of(THROWN_BALL, LINEAR, 1.0, 2.0)

        assert round(body["ridge_r_squared"], 3) == 0.003

    def test_the_radial_kernel_follows_it(self) -> None:
        body = fit_of(THROWN_BALL, RADIAL_AT_ONE, 1.0, 2.0)

        assert round(body["r_squared"], 3) == 0.873

    def test_a_lighter_penalty_lets_it_follow_closer(self) -> None:
        body = fit_of(THROWN_BALL, RADIAL_AT_ONE, 0.1, 2.0)

        assert round(body["r_squared"], 3) == 0.987

    def test_the_squared_kernel_follows_it_closer_still(self) -> None:
        body = fit_of(THROWN_BALL, SQUARED, 1.0, 2.0)

        assert round(body["r_squared"], 3) == 0.981

    def test_the_curve_is_sampled_a_little_past_the_data(self) -> None:
        body = fit_of(THROWN_BALL, RADIAL_AT_ONE, 1.0, 2.0)

        assert body["curve"][0]["x"] < 0.0
        assert body["curve"][-1]["x"] > 4.0
        assert [point["x"] for point in body["curve"]] == [
            point["x"] for point in body["ridge_curve"]
        ]

    def test_the_radial_curve_drifts_back_toward_the_mean_past_the_data(self) -> None:
        body = fit_of(THROWN_BALL, RADIAL_AT_ONE, 1.0, 2.0)

        peak = max(point["y"] for point in body["curve"])
        assert body["curve"][0]["y"] < body["target_mean"] < peak
        assert body["curve"][-1]["y"] < body["target_mean"] < peak


class TestCentring:
    """The control, fitted two right ways and the recorded wrong way."""

    def test_kernel_ridge_lands_on_ridge_exactly(self) -> None:
        body = centring_of(WORKED_THREE, 1.0)

        assert abs(body["kernel_ridge"]["slope"] - body["ridge"]["slope"]) < 1e-9
        assert (
            abs(body["kernel_ridge"]["intercept"] - body["ridge"]["intercept"]) < 1e-9
        )
        assert body["ridge"]["slope"] == 4.0
        assert body["ridge"]["intercept"] == 4.0

    def test_the_two_right_curves_coincide_at_every_sample(self) -> None:
        body = centring_of(WORKED_THREE, 1.0)

        gaps = [
            abs(right["y"] - kernel["y"])
            for right, kernel in zip(
                body["ridge"]["curve"], body["kernel_ridge"]["curve"], strict=True
            )
        ]
        assert len(gaps) == CURVE_SAMPLES
        assert max(gaps) < 1e-9

    def test_both_right_lines_pass_through_the_mean_point(self) -> None:
        body = centring_of(WORKED_THREE, 1.0)

        assert body["mean"] == {"x": 2.0, "y": 12.0}
        assert abs(body["ridge"]["at_mean"] - 12.0) < 1e-9
        assert abs(body["kernel_ridge"]["at_mean"] - 12.0) < 1e-9

    def test_centring_only_the_target_gives_the_wrong_slope(self) -> None:
        body = centring_of(WORKED_THREE, 1.0)

        assert round(body["target_only"]["slope"], 9) == 0.8
        assert body["target_only"]["intercept"] == 12.0

    def test_the_wrong_line_misses_the_mean_point(self) -> None:
        body = centring_of(WORKED_THREE, 1.0)

        assert round(body["target_only"]["at_mean"], 9) == 13.6
        assert len(body["target_only"]["curve"]) == CURVE_SAMPLES

    def test_the_wrong_line_stays_wrong_as_the_penalty_vanishes(self) -> None:
        body = centring_of(WORKED_THREE, 0.001)

        assert round(body["ridge"]["slope"], 2) == 6.0
        assert abs(body["kernel_ridge"]["slope"] - body["ridge"]["slope"]) < 1e-6
        assert round(body["target_only"]["slope"], 3) == 0.857


class TestRefusals:
    """What the endpoints will not compute, and in whose words."""

    SIGMOID_GONE_WRONG: ClassVar[dict] = {
        "name": "sigmoid",
        "gamma": 1.0,
        "constant": -1.0,
    }

    def test_a_sigmoid_that_is_not_a_kernel_is_refused_in_the_library_words(
        self,
    ) -> None:
        response = client.post(
            FIT,
            json={
                "points": THROWN_BALL,
                "kernel": self.SIGMOID_GONE_WRONG,
                "penalty": 1.0,
                "query_x": 2.0,
            },
        )

        assert response.status_code == 422
        assert response.json()["error"] == "InvalidValuesError"
        assert "Mercer" in response.json()["detail"]

    def test_a_kernel_the_library_does_not_have_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={
                "points": WORKED_THREE,
                "kernel": {"name": "cubic"},
                "penalty": 1.0,
                "query_x": 2.5,
            },
        )

        assert response.status_code == 422

    def test_a_parameter_the_kernel_does_not_take_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={
                "points": WORKED_THREE,
                "kernel": {"name": "linear", "gamma": 2.0},
                "penalty": 1.0,
                "query_x": 2.5,
            },
        )

        assert response.status_code == 422

    def test_a_zero_penalty_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={
                "points": WORKED_THREE,
                "kernel": LINEAR,
                "penalty": 0.0,
                "query_x": 2.5,
            },
        )

        assert response.status_code == 422

    def test_a_field_the_request_does_not_take_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={
                "points": WORKED_THREE,
                "kernel": LINEAR,
                "penalty": 1.0,
                "query_x": 2.5,
                "degree": 2,
            },
        )

        assert response.status_code == 422

    def test_a_lone_point_is_refused(self) -> None:
        response = client.post(
            CENTRING, json={"points": WORKED_THREE[:1], "penalty": 1.0}
        )

        assert response.status_code == 422
