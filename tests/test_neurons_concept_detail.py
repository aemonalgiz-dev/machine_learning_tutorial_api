"""The neuron page's further endpoints, pinned to the numbers the page quotes.

The classification pages' crowd of twenty-five, standardised, fits a logistic
model with weights 2.5537 on height and 1.2944 on weight and a bias of 0.1396,
in 7527 passes, scoring 17 of 25. A neuron built from those three numbers under
a sigmoid answers every person to within 1.2e-16 of the model, which is the
page's claim that a sigmoid neuron is that model. The same crowd under the
rectifier leaves 12 people on the dead side, and scaling the fitted sigmoid
neuron by ten puts 17 of 25 under a hundredth of the peak slope, three of them
at exactly zero because the output has rounded to exactly one. Two bendless
neurons in a chain fit a plane with an R squared of exactly 1.0 and a largest
residual of exactly 0.0, and a sigmoid between them leaves 0.885 and 0.565.
Moving one softmax score of three from 0.1 to 5 drops the untouched first
output from 0.659 to 0.047, a factor of 14.1. And the library's own refusals
for a neuron, which the request models reach before it does, are pinned
against the library directly.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from oop_ml import Identity, Neuron, RectifiedLinear, Sigmoid
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
)

from oop_ml_api.main import create_app

client = TestClient(create_app())

TWIN = "/concepts/neurons/logistic-twin"
CENSUS = "/concepts/neurons/crowd-census"
CHAIN = "/concepts/neurons/chain"
SOFTMAX = "/concepts/neurons/softmax-contrast"
RESPOND = "/concepts/neurons/respond"

ACTIVATION_NAMES = ["identity", "rectified_linear", "sigmoid", "hyperbolic_tangent"]

STANDARD_WINDOW = {
    "first_input_low": -3.0,
    "first_input_high": 3.0,
    "second_input_low": -3.0,
    "second_input_high": 3.0,
    "cells": 25,
}

WORKED_CHAIN = {
    "first_weights": [2.0, -1.0],
    "first_bias": 0.5,
    "between": "identity",
    "second_weight": 1.5,
    "second_bias": -0.25,
    "probe": {"first_input": 1.0, "second_input": 1.0},
    "lattice": STANDARD_WINDOW,
}


def twin() -> dict:
    response = client.get(TWIN)
    assert response.status_code == 200, response.json()
    return response.json()


def census(activation: str, scale: float = 1.0) -> dict:
    fitted = twin()
    response = client.post(
        CENSUS,
        json={
            "weights": [scale * weight for weight in fitted["fitted_weights"]],
            "bias": scale * fitted["fitted_bias"],
            "activation": activation,
        },
    )
    assert response.status_code == 200, response.json()
    return response.json()


def chain_with(**changes: object) -> dict:
    response = client.post(CHAIN, json={**WORKED_CHAIN, **changes})
    assert response.status_code == 200, response.json()
    return response.json()


class TestTheLogisticTwin:
    def test_the_crowd_is_standardised_about_its_own_means(self) -> None:
        body = twin()

        assert body["mean_height"] == pytest.approx(151.48)
        assert body["deviation_height"] == pytest.approx(15.613122685740992)
        assert body["mean_weight"] == pytest.approx(51.48)
        assert body["deviation_weight"] == pytest.approx(14.447477288440359)
        assert body["n_people"] == 25

    def test_the_fit_lands_on_the_quoted_weights(self) -> None:
        body = twin()

        assert body["fitted_weights"] == pytest.approx(
            [2.553650898243748, 1.2944236219886285]
        )
        assert round(body["fitted_weights"][0], 4) == 2.5537
        assert round(body["fitted_weights"][1], 4) == 1.2944
        assert body["fitted_bias"] == pytest.approx(0.1396450137847573)
        assert round(body["fitted_bias"], 4) == 0.1396
        assert body["epochs_run"] == 7527
        assert body["converged"] is True

    def test_the_fitted_weights_fit_inside_the_playground(self) -> None:
        body = twin()

        assert all(abs(weight) <= 10.0 for weight in body["fitted_weights"])
        assert abs(body["fitted_bias"]) <= 10.0

    def test_one_straight_line_gets_seventeen_of_twenty_five(self) -> None:
        body = twin()

        assert body["accuracy"] == pytest.approx(0.68)
        assert body["n_correct"] == 17

    def test_the_neuron_and_the_model_agree_to_the_last_bits(self) -> None:
        body = twin()

        assert body["largest_gap"] < 1.2e-16
        assert all(person["gap"] < 1.2e-16 for person in body["people"])
        assert all(
            person["neuron_output"] == pytest.approx(person["model_probability"])
            for person in body["people"]
        )

    def test_the_first_person_is_a_child_scored_below_zero(self) -> None:
        first = twin()["people"][0]

        assert first["height"] == 147.0
        assert first["weight"] == 41.0
        assert first["is_adult"] == 0
        assert first["standardised_height"] == pytest.approx(-0.2869381154668978)
        assert first["standardised_weight"] == pytest.approx(-0.7253861550199633)
        assert first["score"] == pytest.approx(-1.532051736639)
        assert first["model_probability"] == pytest.approx(0.1776936905350905)
        assert first["predicted_adult"] == 0

    def test_the_second_person_is_an_adult_scored_above_zero(self) -> None:
        second = twin()["people"][1]

        assert second["height"] == 156.0
        assert second["weight"] == 53.0
        assert second["standardised_height"] == pytest.approx(0.28950006292642494)
        assert second["standardised_weight"] == pytest.approx(0.10520867897236133)
        assert second["score"] == pytest.approx(1.0151117088184867)
        assert second["model_probability"] == pytest.approx(0.7340193265578332)
        assert second["predicted_adult"] == 1

    def test_the_call_is_the_output_against_one_half(self) -> None:
        for person in twin()["people"]:
            assert person["predicted_adult"] == int(person["neuron_output"] >= 0.5)

    def test_the_boundary_is_where_the_fitted_score_is_zero(self) -> None:
        line = twin()["zero_line"]

        assert line["start"]["first_input"] == pytest.approx(1.4659896757053892)
        assert line["start"]["second_input"] == -3.0
        assert line["end"]["first_input"] == pytest.approx(-1.5753585905251846)
        assert line["end"]["second_input"] == 3.0

    def test_the_twin_is_the_playground_neuron_at_a_person(self) -> None:
        body = twin()
        person = body["people"][1]
        response = client.post(
            RESPOND,
            json={
                "weights": body["fitted_weights"],
                "bias": body["fitted_bias"],
                "activation": "sigmoid",
                "probe": {
                    "first_input": person["standardised_height"],
                    "second_input": person["standardised_weight"],
                },
                "lattice": STANDARD_WINDOW,
            },
        )

        assert response.status_code == 200
        assert response.json()["reading"]["output"] == person["neuron_output"]


class TestTheCrowdCensus:
    def test_the_fitted_sigmoid_neuron_saturates_on_six_people(self) -> None:
        body = census("sigmoid")

        assert body["peak_slope"] == 0.25
        assert body["saturation_threshold"] == 0.0025
        assert body["n_dead"] == 0
        assert body["n_saturated"] == 6
        assert body["smallest_slope"] == pytest.approx(0.00029763971771089163)
        assert round(body["smallest_slope"], 6) == 0.000298
        assert body["largest_slope"] == pytest.approx(0.24999501097721857)

    def test_the_rectifier_leaves_twelve_people_dead(self) -> None:
        body = census("rectified_linear")

        assert body["n_dead"] == 12
        assert body["n_saturated"] == 12
        assert body["smallest_slope"] == 0.0
        assert body["largest_slope"] == 1.0
        dead = [person for person in body["people"] if person["slope"] == 0.0]
        assert all(person["score"] <= 0.0 for person in dead)
        assert all(person["output"] == 0.0 for person in dead)

    def test_the_tangent_saturates_on_the_same_six(self) -> None:
        body = census("hyperbolic_tangent")

        assert body["n_dead"] == 0
        assert body["n_saturated"] == 6
        assert body["smallest_slope"] == pytest.approx(3.547798669290003e-07)

    def test_the_identity_saturates_nowhere(self) -> None:
        body = census("identity", scale=20.0)

        assert body["n_dead"] == 0
        assert body["n_saturated"] == 0
        assert body["smallest_slope"] == 1.0

    def test_scaling_the_fitted_sigmoid_by_four_saturates_ten(self) -> None:
        body = census("sigmoid", scale=4.0)

        assert body["n_dead"] == 0
        assert body["n_saturated"] == 10

    def test_scaling_by_ten_makes_three_slopes_exactly_zero(self) -> None:
        # The output rounds to exactly one in float64, so output times one
        # minus output is exactly zero: saturation has become a dead unit.
        body = census("sigmoid", scale=10.0)

        assert body["n_dead"] == 3
        assert body["n_saturated"] == 17
        assert body["smallest_slope"] == 0.0
        dead = [person for person in body["people"] if person["slope"] == 0.0]
        assert all(person["output"] == 1.0 for person in dead)

    def test_scaling_by_a_quarter_saturates_nobody(self) -> None:
        body = census("sigmoid", scale=0.25)

        assert body["n_saturated"] == 0
        assert body["smallest_slope"] == pytest.approx(0.10263136669651832)

    def test_the_census_reads_the_same_crowd_as_the_twin(self) -> None:
        fitted = twin()
        body = census("sigmoid")

        for twin_person, census_person in zip(
            fitted["people"], body["people"], strict=True
        ):
            assert census_person["height"] == twin_person["height"]
            assert census_person["score"] == twin_person["score"]
            assert census_person["output"] == twin_person["neuron_output"]

    def test_a_weight_beyond_the_census_ceiling_is_refused(self) -> None:
        response = client.post(
            CENSUS, json={"weights": [101.0, 0.0], "bias": 0.0, "activation": "sigmoid"}
        )

        assert response.status_code == 422

    def test_softmax_is_not_a_census_bend_either(self) -> None:
        response = client.post(
            CENSUS, json={"weights": [1.0, 1.0], "bias": 0.0, "activation": "softmax"}
        )

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            CENSUS,
            json={"weights": [1.0, 1.0], "bias": 0.0, "activation": "sigmoid", "k": 1},
        )

        assert response.status_code == 422


class TestTheChain:
    def test_two_bendless_neurons_are_one_neuron(self) -> None:
        body = chain_with(between="identity")

        assert body["collapsed_weights"] == [3.0, -1.5]
        assert body["collapsed_bias"] == 0.5
        assert body["plane"]["height_weight"] == pytest.approx(3.0)
        assert body["plane"]["weight_weight"] == pytest.approx(-1.5)
        assert body["plane"]["intercept"] == pytest.approx(0.5)
        assert body["plane"]["r_squared"] == 1.0
        assert body["plane"]["largest_residual"] == 0.0

    def test_the_tall_heavy_person_through_the_bendless_chain(self) -> None:
        reading = chain_with(between="identity")["reading"]

        assert reading["first_score"] == 1.5
        assert reading["first_output"] == 1.5
        assert reading["second_score"] == 2.0
        assert reading["output"] == 2.0
        assert reading["collapsed_output"] == 2.0

    def test_a_sigmoid_between_leaves_a_surface_no_plane_fits(self) -> None:
        body = chain_with(between="sigmoid")

        assert body["plane"]["r_squared"] == pytest.approx(0.8852478338221897)
        assert round(body["plane"]["r_squared"], 3) == 0.885
        assert body["plane"]["largest_residual"] == pytest.approx(0.5649749005100979)
        assert round(body["plane"]["largest_residual"], 3) == 0.565
        assert body["reading"]["first_output"] == pytest.approx(0.8175744761936437)
        assert body["reading"]["output"] == pytest.approx(0.9763617142904655)
        assert round(body["reading"]["output"], 4) == 0.9764
        assert body["reading"]["collapsed_output"] == 2.0

    def test_the_rectifier_and_the_tangent_between_do_the_same(self) -> None:
        rectifier = chain_with(between="rectified_linear")["plane"]
        tangent = chain_with(between="hyperbolic_tangent")["plane"]

        assert rectifier["r_squared"] == pytest.approx(0.8066968710936133)
        assert rectifier["largest_residual"] == pytest.approx(4.4977846153846155)
        assert tangent["r_squared"] == pytest.approx(0.8117219340935955)
        assert tangent["largest_residual"] == pytest.approx(1.262907349019843)

    def test_the_chain_surface_is_the_lattice(self) -> None:
        body = chain_with()

        assert body["cells"] == 25
        assert len(body["outputs"]) == 25
        assert all(len(row) == 25 for row in body["outputs"])
        # (1, 1) is the sixteenth cell along each side, counting from -3.
        assert body["outputs"][16][16] == body["reading"]["output"]

    def test_a_flat_chain_has_no_r_squared_to_report(self) -> None:
        body = chain_with(between="sigmoid", second_weight=0.0, second_bias=1.0)

        assert body["plane"]["r_squared"] is None
        assert body["plane"]["largest_residual"] == 0.0
        assert body["plane"]["intercept"] == pytest.approx(1.0)

    def test_a_window_that_does_not_open_upward_is_refused_in_words(self) -> None:
        response = client.post(
            CHAIN,
            json={
                **WORKED_CHAIN,
                "lattice": {**STANDARD_WINDOW, "first_input_high": -3.0},
            },
        )

        assert response.status_code == 422
        assert "open upward" in response.json()["detail"]

    def test_a_second_weight_beyond_the_ceiling_is_refused(self) -> None:
        response = client.post(CHAIN, json={**WORKED_CHAIN, "second_weight": 11.0})

        assert response.status_code == 422


class TestTheSoftmaxContrast:
    def test_moving_one_score_moves_every_output(self) -> None:
        body = client.get(SOFTMAX).json()

        assert body["before"]["scores"] == [2.0, 1.0, 0.1]
        assert body["before"]["outputs"] == pytest.approx(
            [0.6590011388859679, 0.24243297070471392, 0.09856589040931818]
        )
        assert body["after"]["scores"] == [2.0, 1.0, 5.0]
        assert body["after"]["outputs"] == pytest.approx(
            [0.0466126225779739, 0.01714782554552039, 0.9362395518765058]
        )
        assert body["first_output_ratio"] == pytest.approx(14.137825817107512)
        assert round(body["first_output_ratio"], 1) == 14.1

    def test_both_rows_sum_to_one(self) -> None:
        body = client.get(SOFTMAX).json()

        assert sum(body["before"]["outputs"]) == pytest.approx(1.0)
        assert sum(body["after"]["outputs"]) == pytest.approx(1.0)


class TestTheLibraryContracts:
    """What the library refuses and accepts, probed directly.

    The request models refuse an empty or third weight before the library is
    reached, and a non-finite number cannot be written in JSON at all, so
    these are the library's own answers rather than the API's.
    """

    def test_a_neuron_with_no_weights_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            Neuron([], 0.0, Identity())

    def test_a_non_finite_weight_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Neuron([float("nan"), 1.0], 0.0, Identity())

    def test_a_non_finite_bias_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Neuron([1.0, 1.0], float("inf"), Identity())

    def test_a_row_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(NonEqualArrayLengthError):
            Neuron([1.0, 1.0], 0.0, Identity()).respond_to([1.0, 2.0, 3.0])

    def test_an_empty_row_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            Neuron([1.0, 1.0], 0.0, Identity()).respond_to([])

    def test_a_non_finite_input_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Neuron([1.0, 1.0], 0.0, Identity()).respond_to([float("nan"), 1.0])

    def test_a_non_numeric_input_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Neuron([1.0, 1.0], 0.0, Identity()).respond_to(["a", "b"])

    def test_integers_and_a_single_weight_are_accepted(self) -> None:
        assert Neuron([1.0, 1.0], 0.0, Identity()).respond_to([1, 2]).score == 3.0
        assert Neuron([1.0], 0.0, Identity()).respond_to([3.0]).output == 3.0

    def test_the_rectifier_answers_zero_slope_at_exactly_zero(self) -> None:
        assert float(RectifiedLinear().derivative_at(np.asarray(0.0))) == 0.0

    def test_a_score_of_minus_two_hundred_and_ten_neither_overflows_nor_warns(
        self,
    ) -> None:
        response = client.post(
            RESPOND,
            json={
                "weights": [10.0, 10.0],
                "bias": -10.0,
                "activation": "sigmoid",
                "probe": {"first_input": -10.0, "second_input": -10.0},
                "lattice": STANDARD_WINDOW,
            },
        )

        reading = response.json()["reading"]
        assert reading["score"] == -210.0
        assert reading["output"] == pytest.approx(6.282880511239462e-92)
        assert reading["slope"] == pytest.approx(6.282880511239462e-92)

    def test_a_score_of_one_hundred_and_ninety_rounds_to_exactly_one(self) -> None:
        response = client.post(
            RESPOND,
            json={
                "weights": [10.0, 10.0],
                "bias": -10.0,
                "activation": "sigmoid",
                "probe": {"first_input": 10.0, "second_input": 10.0},
                "lattice": STANDARD_WINDOW,
            },
        )

        reading = response.json()["reading"]
        assert reading["score"] == 190.0
        assert reading["output"] == 1.0
        assert reading["slope"] == 0.0

    def test_the_sigmoid_slope_is_exactly_zero_once_the_output_is_one(self) -> None:
        assert float(Sigmoid().derivative_at(np.asarray(40.0))) == 0.0
        assert float(Sigmoid().derivative_at(np.asarray(36.0))) > 0.0


def whole_score_samples(scores: list[float]) -> dict[int, int]:
    """Which sample sits at each whole-number score, ignoring the tenths."""
    return {
        round(score): index
        for index, score in enumerate(scores)
        if abs(score - round(score)) < 1e-9
    }


class TestTheWholeScoreSamples:
    """The table in section 16 and the slopes at the short heavy person."""

    def curve(self, activation: str) -> dict:
        response = client.post(
            RESPOND,
            json={
                "weights": [2.0, -1.0],
                "bias": 0.5,
                "activation": activation,
                "probe": {"first_input": -1.0, "second_input": 1.0},
                "lattice": STANDARD_WINDOW,
            },
        )
        assert response.status_code == 200
        return response.json()

    def test_the_sigmoid_at_whole_scores(self) -> None:
        curve = self.curve("sigmoid")["curve"]
        at = whole_score_samples(curve["scores"])

        outputs = [
            round(curve["outputs"][at[score]], 4) for score in (0, 1, 2, 3, 4, 6)
        ]
        slopes = [round(curve["slopes"][at[score]], 4) for score in (0, 1, 2, 3, 4, 6)]
        assert outputs == [0.5, 0.7311, 0.8808, 0.9526, 0.982, 0.9975]
        assert slopes == [0.25, 0.1966, 0.105, 0.0452, 0.0177, 0.0025]

    def test_the_tangent_at_whole_scores(self) -> None:
        curve = self.curve("hyperbolic_tangent")["curve"]
        at = whole_score_samples(curve["scores"])

        outputs = [
            round(curve["outputs"][at[score]], 4) for score in (0, 1, 2, 3, 4, 6)
        ]
        slopes = [round(curve["slopes"][at[score]], 4) for score in (0, 1, 2, 3, 4)]
        assert outputs == [0.0, 0.7616, 0.964, 0.9951, 0.9993, 1.0]
        assert slopes == [1.0, 0.42, 0.0707, 0.0099, 0.0013]
        assert round(curve["slopes"][at[6]], 6) == 0.000025

    def test_the_short_heavy_person_slopes(self) -> None:
        sigmoid = self.curve("sigmoid")["reading"]
        tangent = self.curve("hyperbolic_tangent")["reading"]

        assert sigmoid["score"] == -2.5
        assert round(sigmoid["output"], 4) == 0.0759
        assert round(sigmoid["slope"], 4) == 0.0701
        assert round(tangent["output"], 4) == -0.9866
        assert round(tangent["slope"], 4) == 0.0266

    def test_the_tangent_is_twice_the_sigmoid_at_twice_the_score_less_one(
        self,
    ) -> None:
        sigmoid = self.curve("sigmoid")["curve"]
        at = whole_score_samples(sigmoid["scores"])
        tangent_at_three_halves = 0.9051482536448665

        assert 2.0 * sigmoid["outputs"][at[3]] - 1.0 == pytest.approx(
            tangent_at_three_halves
        )
