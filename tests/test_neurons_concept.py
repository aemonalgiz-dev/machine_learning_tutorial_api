"""The neuron endpoint, pinned to the page's hand arithmetic.

The page's worked example gives a neuron the weights (2, -1) and the bias 0.5
and shows it the input (1, 1). The score is 2 - 1 + 0.5 = 1.5 whatever the
bend, and then each bend answers its own way. Identity and the rectifier hand
the 1.5 straight back, the sigmoid squashes it to 0.8176 and the hyperbolic
tangent to 0.9051. The slopes at that score are 1, 1, 0.1491 and 0.1807, and
the peaks of the slopes over the whole range are 1, 1, one quarter and 1,
which is the arithmetic behind the vanishing gradient the derivation names.
The score is zero along the line second = 2 * first + 0.5, which on the
widget's window runs from (-1.75, -3) to (1.25, 3). All pinned here, along
with the refusals for a window that does not open upward, for a bend that is
not on the list, and for a request that steps outside the bounds.
"""

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

RESPOND = "/concepts/neurons/respond"

WORKED_WINDOW = {
    "first_input_low": -3.0,
    "first_input_high": 3.0,
    "second_input_low": -3.0,
    "second_input_high": 3.0,
    "cells": 25,
}

WORKED_REQUEST = {
    "weights": [2.0, -1.0],
    "bias": 0.5,
    "activation": "sigmoid",
    "probe": {"first_input": 1.0, "second_input": 1.0},
    "lattice": WORKED_WINDOW,
}

ACTIVATION_NAMES = ["identity", "rectified_linear", "sigmoid", "hyperbolic_tangent"]

# Twenty-five cells over [-3, 3] step by a quarter, so the worked input (1, 1)
# is the sixteenth cell along each side, counting from zero at -3.
WORKED_CELL = 16

# The curve samples every tenth from -6 to 6, so a score of zero is the
# sixty-first sample and a score of three the ninety-first.
ZERO_SAMPLE = 60
THREE_SAMPLE = 90


def respond_with(activation: str, **changes: object) -> dict:
    response = client.post(
        RESPOND, json={**WORKED_REQUEST, "activation": activation, **changes}
    )
    assert response.status_code == 200, response.json()
    return response.json()


class TestTheWorkedScore:
    @pytest.mark.parametrize("activation", ACTIVATION_NAMES)
    def test_every_bend_scores_the_worked_input_at_one_and_a_half(
        self, activation: str
    ) -> None:
        reading = respond_with(activation)["reading"]

        assert reading["score"] == 1.5
        assert reading["probe"] == {"first_input": 1.0, "second_input": 1.0}

    def test_the_formula_is_the_library_description(self) -> None:
        formulas = {
            activation: respond_with(activation)["formula"]
            for activation in ACTIVATION_NAMES
        }

        assert formulas == {
            "identity": "z",
            "rectified_linear": "max(0, z)",
            "sigmoid": "1 / (1 + exp(-z))",
            "hyperbolic_tangent": "tanh(z)",
        }


class TestTheWorkedOutputs:
    def test_identity_hands_the_score_straight_back(self) -> None:
        reading = respond_with("identity")["reading"]

        assert reading["output"] == 1.5
        assert reading["slope"] == 1.0

    def test_the_rectifier_passes_a_positive_score_untouched(self) -> None:
        reading = respond_with("rectified_linear")["reading"]

        assert reading["output"] == 1.5
        assert reading["slope"] == 1.0

    def test_the_sigmoid_squashes_it_to_point_eight_one_seven_six(self) -> None:
        reading = respond_with("sigmoid")["reading"]

        assert reading["output"] == pytest.approx(0.8175744761936437)
        assert round(reading["output"], 4) == 0.8176
        assert reading["slope"] == pytest.approx(0.14914645207033286)
        assert round(reading["slope"], 4) == 0.1491

    def test_the_tangent_squashes_it_to_point_nine_zero_five_one(self) -> None:
        reading = respond_with("hyperbolic_tangent")["reading"]

        assert reading["output"] == pytest.approx(0.9051482536448665)
        assert round(reading["output"], 4) == 0.9051
        assert reading["slope"] == pytest.approx(0.18070663892364836)
        assert round(reading["slope"], 4) == 0.1807

    def test_the_sigmoid_slope_is_output_times_one_minus_output(self) -> None:
        reading = respond_with("sigmoid")["reading"]

        assert reading["slope"] == pytest.approx(
            reading["output"] * (1.0 - reading["output"])
        )

    def test_the_tangent_slope_is_one_minus_output_squared(self) -> None:
        reading = respond_with("hyperbolic_tangent")["reading"]

        assert reading["slope"] == pytest.approx(1.0 - reading["output"] ** 2)

    def test_the_tangent_is_the_sigmoid_recentred(self) -> None:
        # tanh(z) = 2 * sigmoid(2z) - 1, checked at the worked score of 1.5
        # against the sigmoid curve's sample at a score of three.
        sigmoid_at_three = respond_with("sigmoid")["curve"]["outputs"][THREE_SAMPLE]
        tangent = respond_with("hyperbolic_tangent")["reading"]["output"]

        assert tangent == pytest.approx(2.0 * sigmoid_at_three - 1.0)

    def test_a_negative_score_is_where_the_bends_part(self) -> None:
        # The probe (-1, 1) scores 2 * -1 - 1 + 0.5 = -2.5, where the rectifier
        # answers zero with a zero slope and the others still answer something.
        probe = {"first_input": -1.0, "second_input": 1.0}
        readings = {
            activation: respond_with(activation, probe=probe)["reading"]
            for activation in ACTIVATION_NAMES
        }

        assert all(reading["score"] == -2.5 for reading in readings.values())
        assert readings["identity"]["output"] == -2.5
        assert readings["rectified_linear"]["output"] == 0.0
        assert readings["rectified_linear"]["slope"] == 0.0
        assert readings["sigmoid"]["output"] == pytest.approx(0.07585818002124356)
        assert 0.0 < readings["sigmoid"]["output"] < 0.5
        assert readings["hyperbolic_tangent"]["output"] == pytest.approx(
            -0.9866142981514303
        )
        assert -1.0 < readings["hyperbolic_tangent"]["output"] < 0.0


class TestTheActivationCurve:
    def test_the_curve_samples_every_tenth_from_minus_six_to_six(self) -> None:
        curve = respond_with("sigmoid")["curve"]

        assert len(curve["scores"]) == 121
        assert curve["scores"][0] == -6.0
        assert curve["scores"][ZERO_SAMPLE] == 0.0
        assert curve["scores"][-1] == 6.0
        assert len(curve["outputs"]) == 121
        assert len(curve["slopes"]) == 121

    def test_the_peak_slopes_are_one_one_a_quarter_and_one(self) -> None:
        peaks = {
            activation: respond_with(activation)["curve"]["peak_slope"]
            for activation in ACTIVATION_NAMES
        }

        assert peaks == {
            "identity": 1.0,
            "rectified_linear": 1.0,
            "sigmoid": 0.25,
            "hyperbolic_tangent": 1.0,
        }

    def test_the_sigmoid_peaks_at_a_score_of_zero(self) -> None:
        curve = respond_with("sigmoid")["curve"]

        assert curve["slopes"][ZERO_SAMPLE] == 0.25
        assert curve["outputs"][ZERO_SAMPLE] == 0.5

    def test_ten_sigmoid_layers_keep_under_a_millionth_at_best(self) -> None:
        peak = respond_with("sigmoid")["curve"]["peak_slope"]

        assert peak**10 == pytest.approx(9.537e-07, rel=1e-3)
        assert peak**10 < 1e-6

    def test_the_sigmoid_slope_has_vanished_at_the_ends(self) -> None:
        curve = respond_with("sigmoid")["curve"]

        assert curve["slopes"][0] == pytest.approx(0.0024665, abs=1e-7)
        assert curve["slopes"][-1] == pytest.approx(0.0024665, abs=1e-7)
        assert round(curve["slopes"][-1], 4) == 0.0025
        assert round(curve["outputs"][-1], 4) == 0.9975

    def test_the_tangent_slope_vanishes_faster_still(self) -> None:
        curve = respond_with("hyperbolic_tangent")["curve"]

        assert curve["slopes"][ZERO_SAMPLE] == 1.0
        assert curve["slopes"][-1] == pytest.approx(2.4577e-05, rel=1e-3)
        assert round(curve["slopes"][-1], 6) == 0.000025
        assert curve["outputs"][-1] == pytest.approx(0.99998771, abs=1e-8)

    def test_the_rectifier_slope_is_a_step(self) -> None:
        curve = respond_with("rectified_linear")["curve"]

        below = [
            slope
            for score, slope in zip(curve["scores"], curve["slopes"], strict=True)
            if score < 0.0
        ]
        above = [
            slope
            for score, slope in zip(curve["scores"], curve["slopes"], strict=True)
            if score > 0.0
        ]
        assert set(below) == {0.0}
        assert set(above) == {1.0}
        # The library's convention at exactly zero, the negative branch.
        assert curve["slopes"][ZERO_SAMPLE] == 0.0

    def test_the_identity_slope_is_one_everywhere(self) -> None:
        curve = respond_with("identity")["curve"]

        assert set(curve["slopes"]) == {1.0}
        assert curve["outputs"] == curve["scores"]

    def test_the_curve_does_not_depend_on_the_neuron(self) -> None:
        worked = respond_with("sigmoid")["curve"]
        other = respond_with("sigmoid", weights=[-3.0, 0.25], bias=-2.0)["curve"]

        assert other == worked


class TestTheResponseSurface:
    def test_the_worked_input_is_a_cell_of_the_lattice(self) -> None:
        body = respond_with("sigmoid")
        surface = body["surface"]

        assert surface["cells"] == 25
        assert surface["scores"][WORKED_CELL][WORKED_CELL] == 1.5
        assert surface["outputs"][WORKED_CELL][WORKED_CELL] == body["reading"]["output"]

    def test_the_corners_score_by_the_weighted_sum(self) -> None:
        surface = respond_with("identity")["surface"]

        # (-3, -3) scores -6 + 3 + 0.5 and (3, 3) scores 6 - 3 + 0.5.
        assert surface["scores"][0][0] == -2.5
        assert surface["scores"][24][24] == 3.5
        # Row-major from the low corner, so the first row is the low second
        # input and the last column the high first input.
        assert surface["scores"][0][24] == 9.5
        assert surface["scores"][24][0] == -8.5

    def test_the_identity_surface_is_the_score_surface(self) -> None:
        surface = respond_with("identity")["surface"]

        assert surface["outputs"] == surface["scores"]

    def test_the_rectifier_surface_is_flat_on_the_negative_side(self) -> None:
        surface = respond_with("rectified_linear")["surface"]

        for score_row, output_row in zip(
            surface["scores"], surface["outputs"], strict=True
        ):
            for score, output in zip(score_row, output_row, strict=True):
                assert output == max(0.0, score)

    def test_the_surface_echoes_its_window(self) -> None:
        surface = respond_with("sigmoid")["surface"]

        assert surface["first_input_low"] == -3.0
        assert surface["first_input_high"] == 3.0
        assert surface["second_input_low"] == -3.0
        assert surface["second_input_high"] == 3.0
        assert len(surface["scores"]) == 25
        assert all(len(row) == 25 for row in surface["scores"])


class TestTheZeroLine:
    def test_the_worked_neuron_is_zero_along_second_equals_two_first_plus_a_half(
        self,
    ) -> None:
        zero_line = respond_with("sigmoid")["zero_line"]

        assert zero_line == {
            "start": {"first_input": -1.75, "second_input": -3.0},
            "end": {"first_input": 1.25, "second_input": 3.0},
        }

    def test_a_line_through_the_corners_is_still_one_segment(self) -> None:
        zero_line = respond_with("sigmoid", weights=[1.0, 1.0], bias=0.0)["zero_line"]

        assert zero_line == {
            "start": {"first_input": -3.0, "second_input": 3.0},
            "end": {"first_input": 3.0, "second_input": -3.0},
        }

    def test_the_bias_slides_the_line_without_turning_it(self) -> None:
        # Doubling the worked bias keeps the slope of two and shifts the line,
        # so it now enters at (-2, -3) and leaves at (1, 3).
        zero_line = respond_with("sigmoid", bias=1.0)["zero_line"]

        assert zero_line == {
            "start": {"first_input": -2.0, "second_input": -3.0},
            "end": {"first_input": 1.0, "second_input": 3.0},
        }

    def test_two_zero_weights_leave_no_line_and_a_flat_score(self) -> None:
        body = respond_with("sigmoid", weights=[0.0, 0.0])

        assert body["zero_line"] is None
        assert body["reading"]["score"] == 0.5
        assert {score for row in body["surface"]["scores"] for score in row} == {0.5}

    def test_a_line_outside_the_window_is_not_drawn(self) -> None:
        body = respond_with("sigmoid", weights=[1.0, 0.0], bias=-8.0)

        assert body["zero_line"] is None


class TestRefusals:
    def test_a_window_that_does_not_open_upward_is_refused_in_words(self) -> None:
        response = client.post(
            RESPOND,
            json={
                **WORKED_REQUEST,
                "lattice": {
                    **WORKED_WINDOW,
                    "first_input_low": 3.0,
                    "first_input_high": -3.0,
                },
            },
        )

        assert response.status_code == 422
        assert "open upward" in response.json()["detail"]

    def test_a_flat_second_window_is_refused_too(self) -> None:
        response = client.post(
            RESPOND,
            json={
                **WORKED_REQUEST,
                "lattice": {
                    **WORKED_WINDOW,
                    "second_input_low": 1.0,
                    "second_input_high": 1.0,
                },
            },
        )

        assert response.status_code == 422
        assert "second input" in response.json()["detail"]

    def test_softmax_is_not_on_the_list(self) -> None:
        response = client.post(
            RESPOND, json={**WORKED_REQUEST, "activation": "softmax"}
        )

        assert response.status_code == 422

    def test_a_third_weight_is_refused(self) -> None:
        response = client.post(
            RESPOND, json={**WORKED_REQUEST, "weights": [2.0, -1.0, 1.0]}
        )

        assert response.status_code == 422

    def test_a_weight_beyond_the_ceiling_is_refused(self) -> None:
        response = client.post(RESPOND, json={**WORKED_REQUEST, "weights": [11.0, 0.0]})

        assert response.status_code == 422

    def test_a_probe_beyond_the_reach_is_refused(self) -> None:
        response = client.post(
            RESPOND,
            json={
                **WORKED_REQUEST,
                "probe": {"first_input": 11.0, "second_input": 0.0},
            },
        )

        assert response.status_code == 422

    def test_a_lattice_finer_than_the_ceiling_is_refused(self) -> None:
        response = client.post(
            RESPOND,
            json={**WORKED_REQUEST, "lattice": {**WORKED_WINDOW, "cells": 42}},
        )

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(RESPOND, json={**WORKED_REQUEST, "threshold": 0.0})

        assert response.status_code == 422

    def test_an_unknown_field_inside_the_probe_is_refused(self) -> None:
        response = client.post(
            RESPOND,
            json={
                **WORKED_REQUEST,
                "probe": {"first_input": 1.0, "second_input": 1.0, "third_input": 1.0},
            },
        )

        assert response.status_code == 422
