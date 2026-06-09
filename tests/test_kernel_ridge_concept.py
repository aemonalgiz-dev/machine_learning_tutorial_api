"""The kernel ridge endpoints, pinned to the page's hand arithmetic.

The page works three people at 169, 170 and 171 centimetres weighing 64, 70
and 76 kilograms at a penalty of one. Centred, the heights are -1, 0 and 1
and the weights -6, 0 and 6, so ridge's slope is 12 / (2 + 1) = 4, the linear
kernel's dual weights come out as -2, 0 and 2, which add up through the
centred heights to the same slope, and both answer 72 at 170.5. The radial
kernel at gamma 1 answers 72.039 there with weights of -3.028, 0 and 3.028,
and the squared kernel 72.4 with 1.2s. The thrown ball pins the identity to
3.7e-14, the ridge line's 0.003 against the radial kernel's 0.873 at a
penalty of one and 0.987 at a tenth, the polynomial ladder, the radial
ladder, the penalty sweep, the extrapolation back to the mean, the sigmoid
knife edge, and the crowd's gamma in centimetres. All pinned here.
"""

from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

FIT = "/concepts/kernel-ridge/fit"
GRAM = "/concepts/kernel-ridge/gram"
SWEEP = "/concepts/kernel-ridge/penalty-sweep"
CENTRING = "/concepts/kernel-ridge/centring"
COST = "/concepts/kernel-ridge/cost"

THREE_PEOPLE = [
    {"x": 169.0, "y": 64.0},
    {"x": 170.0, "y": 70.0},
    {"x": 171.0, "y": 76.0},
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

CROWD = [
    {"x": 152, "y": 51},
    {"x": 155, "y": 61},
    {"x": 158, "y": 56},
    {"x": 161, "y": 66},
    {"x": 164, "y": 62},
    {"x": 167, "y": 71},
    {"x": 170, "y": 66},
    {"x": 173, "y": 76},
    {"x": 176, "y": 71},
    {"x": 179, "y": 81},
    {"x": 182, "y": 75},
    {"x": 185, "y": 86},
    {"x": 188, "y": 81},
    {"x": 191, "y": 91},
    {"x": 194, "y": 86},
]

LINEAR = {"name": "linear"}
RADIAL_AT_ONE = {"name": "rbf", "gamma": 1.0}
SQUARED = {"name": "polynomial", "degree": 2}

CURVE_SAMPLES = 81


def fit_of(
    points: list[dict], kernel: dict, penalty: float, query_x: float, **extra
) -> dict:
    response = client.post(
        FIT,
        json={
            "points": points,
            "kernel": kernel,
            "penalty": penalty,
            "query_x": query_x,
            **extra,
        },
    )
    assert response.status_code == 200, response.json()
    return response.json()


def gram_of(points: list[dict], kernel: dict) -> dict:
    response = client.post(GRAM, json={"points": points, "kernel": kernel})
    assert response.status_code == 200, response.json()
    return response.json()


def centring_of(points: list[dict], penalty: float) -> dict:
    response = client.post(CENTRING, json={"points": points, "penalty": penalty})
    assert response.status_code == 200, response.json()
    return response.json()


class TestTheLinearKernelControl:
    """Under the linear kernel the fit is ridge regression, and the page says so."""

    def test_ridge_has_the_hand_worked_slope(self) -> None:
        body = fit_of(THREE_PEOPLE, LINEAR, 1.0, 170.5)

        assert body["ridge_slope"] == pytest.approx(4.0)
        assert body["feature_mean"] == 170.0
        assert body["target_mean"] == 70.0

    def test_the_dual_weights_are_minus_two_zero_and_two(self) -> None:
        body = fit_of(THREE_PEOPLE, LINEAR, 1.0, 170.5)

        assert [round(weight, 9) for weight in body["dual_weights"]] == [-2.0, 0.0, 2.0]

    def test_the_dual_weights_imply_the_slope_ridge_found(self) -> None:
        body = fit_of(THREE_PEOPLE, LINEAR, 1.0, 170.5)

        assert body["implied_slope"] == pytest.approx(4.0)
        assert body["implied_slope"] == pytest.approx(body["ridge_slope"])

    def test_both_answer_seventy_two_at_the_half_centimetre(self) -> None:
        body = fit_of(THREE_PEOPLE, LINEAR, 1.0, 170.5)

        assert body["prediction_at_query"] == pytest.approx(72.0)
        assert body["ridge_at_query"] == pytest.approx(72.0)
        assert body["fitted"] == pytest.approx([66.0, 70.0, 74.0])

    def test_the_identity_holds_on_the_throw_to_the_number(self) -> None:
        body = fit_of(THROWN_BALL, LINEAR, 1.0, 2.0)

        assert body["largest_gap"] < 1e-12
        assert round(body["ridge_slope"], 6) == 0.261135
        assert body["implied_slope"] == pytest.approx(body["ridge_slope"], abs=1e-9)
        assert round(body["ridge_r_squared"], 6) == 0.002588
        assert body["r_squared"] == pytest.approx(body["ridge_r_squared"], abs=1e-9)
        assert body["prediction_at_query"] == pytest.approx(12.76)
        assert body["ridge_at_query"] == pytest.approx(12.76)
        assert len(body["curve"]) == len(body["ridge_curve"]) == CURVE_SAMPLES

    def test_the_identity_survives_a_tiny_penalty(self) -> None:
        body = fit_of(THROWN_BALL, LINEAR, 0.001, 2.0)

        assert body["largest_gap"] < 1e-9
        assert body["implied_slope"] == pytest.approx(body["ridge_slope"], abs=1e-9)

    def test_the_crowd_line(self) -> None:
        body = fit_of(CROWD, LINEAR, 1.0, 175.0)

        assert round(body["r_squared"], 3) == 0.890
        assert round(body["ridge_slope"], 3) == 0.826
        assert body["largest_gap"] < 1e-9
        assert body["target_mean"] == 72.0


class TestTheDualWeightsAreResidualsOverThePenalty:
    """The optimality condition, read row by row, under every kernel."""

    def test_on_the_three_people(self) -> None:
        body = fit_of(THREE_PEOPLE, LINEAR, 1.0, 170.5)

        assert body["residuals"] == pytest.approx([-2.0, 0.0, 2.0])
        assert body["residuals_over_penalty"] == pytest.approx(body["dual_weights"])
        assert body["dual_residual_gap"] < 1e-12

    @pytest.mark.parametrize("kernel", [LINEAR, RADIAL_AT_ONE, SQUARED])
    @pytest.mark.parametrize("penalty", [0.001, 1.0, 100.0])
    def test_on_the_throw_under_every_kernel(
        self, kernel: dict, penalty: float
    ) -> None:
        body = fit_of(THROWN_BALL, kernel, penalty, 2.0)

        assert body["dual_residual_gap"] < 1e-9 * max(1.0, body["max_abs_dual_weight"])
        assert body["residuals_over_penalty"] == pytest.approx(body["dual_weights"])

    def test_the_weights_scale_as_one_over_the_penalty_while_the_line_stays(
        self,
    ) -> None:
        at_one = fit_of(THROWN_BALL, LINEAR, 1.0, 2.0)
        at_a_thousandth = fit_of(THROWN_BALL, LINEAR, 0.001, 2.0)

        assert round(at_one["max_abs_dual_weight"], 2) == 12.08
        assert round(at_a_thousandth["max_abs_dual_weight"]) == 12105
        assert round(at_one["r_squared"], 6) == 0.002588
        assert round(at_a_thousandth["r_squared"], 6) == 0.002593

    def test_the_model_keeps_every_row(self) -> None:
        body = fit_of(THROWN_BALL, RADIAL_AT_ONE, 1.0, 2.0)

        assert body["n_training_rows"] == 15
        assert len(body["dual_weights"]) == 15


class TestTheGramMatrix:
    """The one table the solve reads, laid out for the three people and the throw."""

    def test_the_linear_table_on_three_people(self) -> None:
        body = gram_of(THREE_PEOPLE, LINEAR)

        assert body["centred_x"] == [-1.0, 0.0, 1.0]
        assert body["matrix"] == [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]]
        assert body["eigenvalues"] == pytest.approx([0.0, 0.0, 2.0], abs=1e-12)
        assert body["zero_eigenvalues"] == 2
        assert body["symmetric"] is True

    def test_the_radial_table_on_three_people(self) -> None:
        body = gram_of(THREE_PEOPLE, RADIAL_AT_ONE)

        assert round(body["matrix"][0][1], 3) == 0.368
        assert round(body["matrix"][0][2], 3) == 0.018
        assert body["matrix"][1][1] == 1.0
        assert [round(value, 3) for value in body["eigenvalues"]] == [
            0.489,
            0.982,
            1.529,
        ]

    def test_the_squared_table_on_three_people(self) -> None:
        body = gram_of(THREE_PEOPLE, SQUARED)

        assert body["matrix"] == [[4.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 4.0]]

    def test_the_throw_has_one_direction_under_the_linear_kernel(self) -> None:
        body = gram_of(THROWN_BALL, LINEAR)

        assert body["zero_eigenvalues"] == 14
        assert round(body["eigenvalues"][-1], 3) == 22.834
        assert abs(body["eigenvalues"][0]) < 1e-13

    def test_the_throw_has_three_under_the_squared_kernel_and_fifteen_under_radial(
        self,
    ) -> None:
        squared = gram_of(THROWN_BALL, SQUARED)
        radial = gram_of(THROWN_BALL, RADIAL_AT_ONE)

        assert squared["zero_eigenvalues"] == 12
        assert radial["zero_eigenvalues"] == 0
        assert radial["eigenvalues"][0] == pytest.approx(8.1e-9, rel=0.05)
        assert [round(value, 3) for value in radial["matrix"][7][7:]] == [
            1.0,
            0.919,
            0.723,
            0.477,
            0.273,
            0.129,
            0.054,
            0.018,
        ]

    def test_too_many_rows_to_read_are_refused(self) -> None:
        response = client.post(GRAM, json={"points": THROWN_BALL * 2, "kernel": LINEAR})

        assert response.status_code == 422


class TestThePolynomialLadder:
    """Degree on the throw, and the degree-one identity centring forces."""

    @pytest.mark.parametrize(
        ("degree", "score", "condition"),
        [
            (1, 0.003, 24),
            (2, 0.981, 72),
            (3, 0.985, 262),
            (4, 0.981, 1061),
            (5, 0.982, 4597),
            (6, 0.980, 20781),
        ],
    )
    def test_the_score_and_the_condition_number(
        self, degree: int, score: float, condition: int
    ) -> None:
        body = fit_of(THROWN_BALL, {"name": "polynomial", "degree": degree}, 1.0, 2.0)

        assert round(body["r_squared"], 3) == score
        assert round(body["condition_number"]) == condition

    def test_degree_one_with_a_constant_is_the_linear_kernel_here(self) -> None:
        body = fit_of(THROWN_BALL, {"name": "polynomial", "degree": 1}, 1.0, 2.0)

        assert body["largest_gap"] < 1e-12
        assert body["implied_slope"] is None

    def test_the_squared_kernel_on_three_people(self) -> None:
        body = fit_of(THREE_PEOPLE, SQUARED, 1.0, 170.5)

        assert body["prediction_at_query"] == pytest.approx(72.4)
        assert [round(weight, 9) for weight in body["dual_weights"]] == [-1.2, 0.0, 1.2]
        assert round(body["r_squared"], 3) == 0.960


class TestTheRadialLadder:
    """Gamma on the throw in seconds, and on the crowd in centimetres."""

    @pytest.mark.parametrize(
        ("gamma", "score"),
        [
            (0.01, 0.012),
            (0.1, 0.450),
            (0.3, 0.797),
            (1.0, 0.873),
            (3.0, 0.880),
            (10.0, 0.844),
            (100.0, 0.750),
        ],
    )
    def test_the_score_by_gamma_on_the_throw(self, gamma: float, score: float) -> None:
        body = fit_of(THROWN_BALL, {"name": "rbf", "gamma": gamma}, 1.0, 2.0)

        assert round(body["r_squared"], 3) == score

    def test_gamma_ten_separates_the_throws(self) -> None:
        body = fit_of(THROWN_BALL, {"name": "rbf", "gamma": 10.0}, 1.0, 2.0)

        assert round(body["gram_smallest_eigenvalue"], 3) == 0.202

    def test_a_lighter_penalty_lets_it_follow_closer(self) -> None:
        body = fit_of(THROWN_BALL, RADIAL_AT_ONE, 0.1, 2.0)

        assert round(body["r_squared"], 3) == 0.987
        assert round(body["max_abs_dual_weight"], 2) == 17.39
        assert round(body["condition_number"], 1) == 57.5

    def test_the_radial_kernel_on_three_people(self) -> None:
        body = fit_of(THREE_PEOPLE, RADIAL_AT_ONE, 1.0, 170.5)

        assert round(body["prediction_at_query"], 3) == 72.039
        first, middle, last = body["dual_weights"]
        assert round(first, 3) == -3.028
        assert round(last, 3) == 3.028
        assert abs(middle) < 1e-9
        assert round(body["r_squared"], 3) == 0.745
        assert body["implied_slope"] is None

    @pytest.mark.parametrize(
        ("gamma", "score"),
        [
            (1.0, 0.750),
            (0.1, 0.804),
            (0.01, 0.835),
            (0.001, 0.824),
            (0.0003, 0.716),
            (0.0001, 0.481),
        ],
    )
    def test_the_score_by_gamma_on_the_crowd(self, gamma: float, score: float) -> None:
        body = fit_of(CROWD, {"name": "rbf", "gamma": gamma}, 1.0, 175.0)

        assert round(body["r_squared"], 3) == score

    def test_gamma_one_on_the_crowd_is_the_identity_table(self) -> None:
        body = fit_of(CROWD, RADIAL_AT_ONE, 1.0, 173.0)

        assert round(body["gram_smallest_eigenvalue"], 5) == 0.99976
        assert round(body["prediction_at_query"], 1) == 74.0
        assert body["dual_weights"] == pytest.approx(
            [(point["y"] - 72.0) / 2.0 for point in CROWD], abs=2e-3
        )

    def test_between_two_people_the_curve_has_returned_to_the_mean(self) -> None:
        body = fit_of(CROWD, RADIAL_AT_ONE, 1.0, 174.5)

        assert round(body["prediction_at_query"], 2) == 72.16


class TestThePenaltySweep:
    """From a billionth to a thousand on the throw, under the radial kernel."""

    PENALTIES: ClassVar[list[float]] = [10.0**exponent for exponent in range(-9, 4)]

    def sweep(self, kernel: dict) -> list[dict]:
        response = client.post(
            SWEEP,
            json={"points": THROWN_BALL, "kernel": kernel, "penalties": self.PENALTIES},
        )
        assert response.status_code == 200, response.json()
        return response.json()["steps"]

    def test_the_small_end_interpolates_and_ill_conditions(self) -> None:
        steps = {step["penalty"]: step for step in self.sweep(RADIAL_AT_ONE)}

        smallest = steps[1e-9]
        assert round(smallest["r_squared"], 6) == 0.999998
        assert smallest["max_abs_dual_weight"] == pytest.approx(1.6e7, rel=0.05)
        assert smallest["condition_number"] == pytest.approx(6.2e8, rel=0.05)
        assert smallest["rows_passed_through"] == 4
        assert round(smallest["farthest_from_mean"], 1) == 122.4

        millionth = steps[1e-6]
        assert round(millionth["r_squared"], 5) == 0.99955
        assert millionth["condition_number"] == pytest.approx(5.6e6, rel=0.05)
        assert round(millionth["farthest_from_mean"], 1) == 17.8

        tenth = steps[0.1]
        assert round(tenth["r_squared"], 3) == 0.987
        assert round(tenth["condition_number"], 1) == 57.5
        assert round(tenth["farthest_from_mean"], 1) == 11.0

    def test_the_large_end_flattens_onto_the_mean(self) -> None:
        steps = {step["penalty"]: step for step in self.sweep(RADIAL_AT_ONE)}

        for penalty, score, weight, farthest in (
            (1.0, 0.873, 5.640, 6.554),
            (10.0, 0.353, 1.047, 2.254),
            (100.0, 0.049, 0.119, 0.302),
            (1000.0, 0.005, 0.012, 0.031),
        ):
            assert round(steps[penalty]["r_squared"], 3) == score
            assert round(steps[penalty]["max_abs_dual_weight"], 3) == weight
            assert round(steps[penalty]["farthest_from_mean"], 3) == farthest

    def test_the_linear_kernel_does_not_change_while_its_condition_explodes(
        self,
    ) -> None:
        steps = {step["penalty"]: step for step in self.sweep(LINEAR)}

        assert round(steps[1e-9]["r_squared"], 6) == 0.002593
        assert steps[1e-9]["condition_number"] == pytest.approx(2.28e10, rel=0.05)
        assert round(steps[1000.0]["r_squared"], 6) == 0.000114

    def test_every_step_carries_a_curve(self) -> None:
        steps = self.sweep(SQUARED)

        assert [step["penalty"] for step in steps] == self.PENALTIES
        assert all(len(step["curve"]) == CURVE_SAMPLES for step in steps)

    def test_a_penalty_below_the_floor_is_refused(self) -> None:
        response = client.post(
            SWEEP, json={"points": THROWN_BALL, "kernel": LINEAR, "penalties": [1e-12]}
        )

        assert response.status_code == 422


class TestExtrapolation:
    """Past the last throw the radial curve returns to the mean; the others do not."""

    @pytest.mark.parametrize(
        ("query", "answer"), [(5.0, 7.468), (6.0, 12.464), (7.0, 12.758), (8.0, 12.76)]
    )
    def test_the_radial_curve_returns_to_the_mean(
        self, query: float, answer: float
    ) -> None:
        body = fit_of(THROWN_BALL, RADIAL_AT_ONE, 0.1, query)

        assert round(body["prediction_at_query"], 3) == answer
        assert body["target_mean"] == pytest.approx(12.76)

    @pytest.mark.parametrize(
        ("query", "answer"), [(5.0, -19.5), (6.0, -49.8), (8.0, -136.5)]
    )
    def test_the_squared_curve_runs_away(self, query: float, answer: float) -> None:
        body = fit_of(THROWN_BALL, SQUARED, 1.0, query)

        assert round(body["prediction_at_query"], 1) == answer

    def test_the_line_keeps_its_slope(self) -> None:
        body = fit_of(THROWN_BALL, LINEAR, 1.0, 8.0)

        assert round(body["prediction_at_query"], 2) == 14.33

    def test_a_wider_curve_range_is_honoured(self) -> None:
        body = fit_of(
            THROWN_BALL, RADIAL_AT_ONE, 0.1, 6.0, curve_from=-4.0, curve_to=12.0
        )

        assert body["curve"][0]["x"] == -4.0
        assert body["curve"][-1]["x"] == 12.0
        assert body["curve"][-1]["y"] == pytest.approx(12.76, abs=1e-6)

    def test_a_backward_curve_range_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={
                "points": THROWN_BALL,
                "kernel": LINEAR,
                "penalty": 1.0,
                "query_x": 1.0,
                "curve_from": 5.0,
                "curve_to": 1.0,
            },
        )

        assert response.status_code == 422


class TestCentring:
    """The control, fitted two right ways and the recorded wrong way."""

    def test_kernel_ridge_lands_on_ridge_exactly_on_the_crowd(self) -> None:
        body = centring_of(CROWD, 1.0)

        assert body["mean"] == {"x": 173.0, "y": 72.0}
        assert round(body["ridge"]["slope"], 4) == 0.8259
        assert body["kernel_ridge"]["slope"] == pytest.approx(
            body["ridge"]["slope"], abs=1e-9
        )
        assert body["ridge"]["at_mean"] == pytest.approx(72.0)
        assert body["kernel_ridge"]["at_mean"] == pytest.approx(72.0)

    def test_centring_only_the_target_gives_a_flat_line_on_the_crowd(self) -> None:
        body = centring_of(CROWD, 1.0)

        assert round(body["target_only"]["slope"], 4) == 0.0046
        assert round(body["target_only"]["at_mean"], 2) == 72.80

    def test_the_wrong_line_stays_wrong_as_the_penalty_vanishes(self) -> None:
        body = centring_of(CROWD, 0.001)

        assert round(body["ridge"]["slope"], 4) == 0.8262
        assert round(body["target_only"]["slope"], 4) == 0.0046

    def test_on_the_throw_the_wrong_slope_looks_closer(self) -> None:
        body = centring_of(THROWN_BALL, 1.0)

        assert round(body["ridge"]["slope"], 4) == 0.2611
        assert round(body["target_only"]["slope"], 4) == 0.0742
        assert round(body["target_only"]["at_mean"], 3) == 12.908
        assert len(body["target_only"]["curve"]) == CURVE_SAMPLES


class TestTheSigmoidKnifeEdge:
    """A function that is a kernel only sometimes, and the solver's guard."""

    HALF_NEGATIVE: ClassVar[dict] = {"name": "sigmoid", "gamma": 1.0, "constant": -0.5}

    def test_the_gram_matrix_has_a_negative_eigenvalue(self) -> None:
        body = gram_of(THROWN_BALL, self.HALF_NEGATIVE)

        assert round(body["eigenvalues"][0], 3) == -4.313
        assert body["symmetric"] is True

    @pytest.mark.parametrize("penalty", [1.0, 4.0, 4.3])
    def test_below_the_eigenvalue_the_fit_is_refused_in_the_library_words(
        self, penalty: float
    ) -> None:
        response = client.post(
            FIT,
            json={
                "points": THROWN_BALL,
                "kernel": self.HALF_NEGATIVE,
                "penalty": penalty,
                "query_x": 2.0,
            },
        )

        assert response.status_code == 422
        assert response.json()["error"] == "InvalidValuesError"
        assert "Mercer" in response.json()["detail"]

    @pytest.mark.parametrize(
        ("penalty", "score"), [(4.4, -228.4), (5.0, -4.5), (10.0, -0.1), (100.0, 0.0)]
    )
    def test_above_it_the_fit_is_accepted_and_worse_than_the_mean(
        self, penalty: float, score: float
    ) -> None:
        body = fit_of(THROWN_BALL, self.HALF_NEGATIVE, penalty, 2.0)

        assert round(body["r_squared"], 1) == score

    def test_a_constant_of_zero_is_indefinite_and_hidden_at_penalty_one(self) -> None:
        kernel = {"name": "sigmoid", "gamma": 1.0, "constant": 0.0}
        gram = gram_of(THROWN_BALL, kernel)
        body = fit_of(THROWN_BALL, kernel, 1.0, 2.0)

        assert round(gram["eigenvalues"][0], 3) == -0.878
        assert round(body["r_squared"], 4) == 0.0023


class TestCost:
    """The timed sweep, pinned by shape and by what each model keeps."""

    def test_the_sweeps_have_the_documented_sizes(self) -> None:
        body = client.get(COST).json()

        assert body["repeats"] == 5
        assert body["queries"] == 100
        assert [point["rows"] for point in body["by_rows"]] == [
            50,
            100,
            200,
            400,
            800,
            1600,
        ]
        assert all(point["features"] == 2 for point in body["by_rows"])
        assert [point["features"] for point in body["by_features"]] == [
            2,
            4,
            8,
            16,
            32,
            64,
        ]
        assert all(point["rows"] == 200 for point in body["by_features"])

    def test_the_kernel_model_keeps_its_rows_and_ridge_keeps_its_coefficients(
        self,
    ) -> None:
        body = client.get(COST).json()

        assert [point["kernel_model_numbers"] for point in body["by_rows"]] == [
            150,
            300,
            600,
            1200,
            2400,
            4800,
        ]
        assert all(point["ridge_model_numbers"] == 3 for point in body["by_rows"])
        assert [point["ridge_model_numbers"] for point in body["by_features"]] == [
            3,
            5,
            9,
            17,
            33,
            65,
        ]

    def test_the_kernel_fit_grows_with_the_rows_and_ridge_does_not(self) -> None:
        body = client.get(COST).json()
        rows = body["by_rows"]

        assert rows[-1]["kernel_fit_ms"] > 20 * rows[0]["kernel_fit_ms"]
        assert rows[-1]["ridge_fit_ms"] < 20 * rows[0]["ridge_fit_ms"]
        assert all(point["kernel_fit_ms"] > 0 for point in rows)


class TestRefusals:
    """What the endpoints will not compute, and in whose words."""

    def test_a_kernel_the_library_does_not_have_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={
                "points": THREE_PEOPLE,
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
                "points": THREE_PEOPLE,
                "kernel": {"name": "linear", "gamma": 2.0},
                "penalty": 1.0,
                "query_x": 2.5,
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("penalty", [0.0, -1.0, 1e-12])
    def test_a_penalty_at_or_below_the_floor_is_refused(self, penalty: float) -> None:
        response = client.post(
            FIT,
            json={
                "points": THREE_PEOPLE,
                "kernel": LINEAR,
                "penalty": penalty,
                "query_x": 2.5,
            },
        )

        assert response.status_code == 422

    def test_a_gamma_of_zero_and_a_degree_of_seven_are_refused(self) -> None:
        for kernel in (
            {"name": "rbf", "gamma": 0.0},
            {"name": "polynomial", "degree": 7},
        ):
            response = client.post(
                FIT,
                json={
                    "points": THROWN_BALL,
                    "kernel": kernel,
                    "penalty": 1.0,
                    "query_x": 1.0,
                },
            )
            assert response.status_code == 422

    def test_a_field_the_request_does_not_take_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={
                "points": THREE_PEOPLE,
                "kernel": LINEAR,
                "penalty": 1.0,
                "query_x": 2.5,
                "degree": 2,
            },
        )

        assert response.status_code == 422

    def test_a_lone_point_and_too_many_points_are_refused(self) -> None:
        lone = client.post(CENTRING, json={"points": THREE_PEOPLE[:1], "penalty": 1.0})
        crowd = client.post(
            FIT,
            json={
                "points": THROWN_BALL * 7,
                "kernel": LINEAR,
                "penalty": 1.0,
                "query_x": 1.0,
            },
        )

        assert lone.status_code == 422
        assert crowd.status_code == 422

    def test_a_non_finite_value_is_refused_without_being_echoed(self) -> None:
        response = client.post(
            FIT,
            content='{"points":[{"x":1.0,"y":NaN},{"x":2.0,"y":1.0}],"kernel":{"name":"linear"},"penalty":1.0,"query_x":1.0}',
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422
        assert "NaN" not in response.text

    def test_a_constant_column_is_refused_by_the_ridge_control(self) -> None:
        response = client.post(
            FIT,
            json={
                "points": [
                    {"x": 5.0, "y": 1.0},
                    {"x": 5.0, "y": 2.0},
                    {"x": 5.0, "y": 6.0},
                ],
                "kernel": RADIAL_AT_ONE,
                "penalty": 1.0,
                "query_x": 100.0,
            },
        )

        assert response.status_code == 422
        assert response.json()["error"] == "AllSameValuesError"

    def test_a_constant_target_fits_and_its_score_is_undefined(self) -> None:
        response = client.post(
            FIT,
            json={
                "points": [
                    {"x": 1.0, "y": 4.0},
                    {"x": 2.0, "y": 4.0},
                    {"x": 3.0, "y": 4.0},
                ],
                "kernel": RADIAL_AT_ONE,
                "penalty": 1.0,
                "query_x": 1.5,
            },
        )

        assert response.status_code == 422
        assert response.json()["error"] == "UndefinedMetricError"

    def test_two_rows_at_the_same_input_are_accepted(self) -> None:
        body = fit_of(
            [{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 3.0}, {"x": 2.0, "y": 5.0}],
            RADIAL_AT_ONE,
            1.0,
            1.0,
        )

        assert abs(body["gram_smallest_eigenvalue"]) < 1e-12
        assert round(body["r_squared"], 3) == 0.520

    def test_two_rows_are_enough(self) -> None:
        body = fit_of(THROWN_BALL[:2], RADIAL_AT_ONE, 1.0, 1.0)

        assert round(body["r_squared"], 3) == 0.144
        assert [round(weight, 3) for weight in body["dual_weights"]] == [-2.082, 2.082]

    def test_a_polynomial_kernel_overflowing_on_raw_inputs_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={
                "points": [
                    {"x": 1e5, "y": 1.0},
                    {"x": 2e5, "y": 2.0},
                    {"x": 3e5, "y": 3.0},
                    {"x": 4e5, "y": 4.0},
                ],
                "kernel": {"name": "polynomial", "degree": 6},
                "penalty": 1.0,
                "query_x": 1.5,
            },
        )

        assert response.status_code == 422
        assert response.json()["error"] == "InvalidValuesError"
