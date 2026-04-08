"""The backpropagation endpoint, pinned to the page's hand arithmetic.

The page works one row, (1, 2) with target 1, through three rectified hidden
neurons with weights (1, 1), (1, -1) and (-1, 1) and no biases, into one
linear output neuron with weights (1, -1, 2) and no bias. Forward, the hidden
scores are (3, -1, 1), the rectifier keeps (3, 0, 1), the output is 5, and
squared error against 1 is 16 halved, 8, with a slope of 4 at the output.

Backward, the output layer's weight gradient is the slope times what it read,
4 times (3, 0, 1) = (12, 0, 4), and it passes 4 times (1, -1, 2) = (4, -4, 8)
down. The hidden layer's slopes are (1, 0, 1), so its deltas are (4, 0, 8), its
weight gradient is each delta times (1, 2), and the largest slope anywhere is
16. One step at 0.05 takes the output to 0.52 and the loss to 0.1152. Every one
of those is pinned here, as is the finite-difference check on the output
layer's first weight, which agrees with 12 to better than a millionth.
"""

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

STEP = "/concepts/backpropagation/step"

WORKED_EXAMPLE = {
    "inputs": [1.0, 2.0],
    "target": 1.0,
    "hidden_activation": "rectified_linear",
    "hidden_neurons": [
        {"weights": [1.0, 1.0], "bias": 0.0},
        {"weights": [1.0, -1.0], "bias": 0.0},
        {"weights": [-1.0, 1.0], "bias": 0.0},
    ],
    "output_neuron": {"weights": [1.0, -1.0, 2.0], "bias": 0.0},
    "learning_rate": 0.05,
    "checked_weight": {"layer_index": 1, "neuron_index": 0, "input_index": 0},
}


def worked_example(**changes: object) -> dict:
    body = client.post(STEP, json={**WORKED_EXAMPLE, **changes}).json()
    assert "layers" in body, body
    return body


class TestForwardPass:
    def test_the_hidden_scores_are_three_minus_one_and_one(self) -> None:
        hidden = worked_example()["layers"][0]

        assert hidden["inputs"] == [1.0, 2.0]
        assert hidden["scores"] == [3.0, -1.0, 1.0]

    def test_the_rectifier_keeps_three_zero_and_one(self) -> None:
        hidden = worked_example()["layers"][0]

        assert hidden["outputs"] == [3.0, 0.0, 1.0]

    def test_the_output_neuron_reads_the_hidden_outputs_and_answers_five(
        self,
    ) -> None:
        body = worked_example()
        output = body["layers"][1]

        assert output["inputs"] == [3.0, 0.0, 1.0]
        assert output["scores"] == [5.0]
        assert output["outputs"] == [5.0]
        assert body["prediction"] == 5.0

    def test_the_loss_is_eight_with_a_slope_of_four_at_the_output(self) -> None:
        body = worked_example()

        assert body["loss_before"] == 8.0
        assert body["loss_gradient"] == 4.0


class TestBackwardPass:
    def test_the_output_layer_receives_the_loss_slope_through_a_unit_bend(
        self,
    ) -> None:
        output = worked_example()["layers"][1]

        assert output["arriving"] == [4.0]
        assert output["slopes"] == [1.0]
        assert output["bias_gradient"] == [4.0]

    def test_the_output_layer_gradient_is_the_delta_times_what_it_read(
        self,
    ) -> None:
        output = worked_example()["layers"][1]

        assert output["weight_gradient"] == [[12.0, 0.0, 4.0]]

    def test_the_output_layer_passes_down_its_delta_times_its_weights(
        self,
    ) -> None:
        output = worked_example()["layers"][1]

        assert output["passed_down"] == [4.0, -4.0, 8.0]

    def test_the_hidden_layer_receives_exactly_what_was_passed_down(self) -> None:
        body = worked_example()
        hidden, output = body["layers"]

        assert hidden["arriving"] == output["passed_down"]
        assert hidden["arriving"] == [4.0, -4.0, 8.0]

    def test_the_rectifier_slopes_are_one_where_the_score_was_positive(
        self,
    ) -> None:
        hidden = worked_example()["layers"][0]

        assert hidden["slopes"] == [1.0, 0.0, 1.0]
        assert hidden["bias_gradient"] == [4.0, 0.0, 8.0]

    def test_the_hidden_gradients_are_the_deltas_times_the_row(self) -> None:
        hidden = worked_example()["layers"][0]

        assert hidden["weight_gradient"] == [[4.0, 8.0], [0.0, 0.0], [8.0, 16.0]]

    def test_the_dead_rectifier_gets_no_gradient_at_all(self) -> None:
        hidden = worked_example()["layers"][0]

        assert hidden["weight_gradient"][1] == [0.0, 0.0]
        assert hidden["bias_gradient"][1] == 0.0

    def test_the_hidden_layer_passes_a_slope_down_to_the_row_itself(self) -> None:
        hidden = worked_example()["layers"][0]

        assert hidden["passed_down"] == [-4.0, 12.0]

    def test_the_largest_slope_anywhere_is_sixteen(self) -> None:
        assert worked_example()["largest_movement"] == 16.0


class TestOneStep:
    def test_every_parameter_moves_a_twentieth_of_its_slope(self) -> None:
        body = worked_example()
        hidden, output = body["layers"]

        assert output["weights_after"] == [pytest.approx([0.4, -1.0, 1.8])]
        assert output["biases_after"] == [pytest.approx(-0.2)]
        assert hidden["weights_after"] == [
            pytest.approx([0.8, 0.6]),
            pytest.approx([1.0, -1.0]),
            pytest.approx([-1.4, 0.2]),
        ]
        assert hidden["biases_after"] == pytest.approx([-0.2, 0.0, -0.4])

    def test_the_stepped_network_answers_naught_point_five_two(self) -> None:
        body = worked_example()

        assert body["prediction_after"] == pytest.approx(0.52)

    def test_the_loss_falls_from_eight_to_the_hand_worked_figure(self) -> None:
        body = worked_example()

        assert body["loss_after"] == pytest.approx(0.1152)

    def test_a_smaller_step_lands_where_the_hand_arithmetic_says(self) -> None:
        body = worked_example(learning_rate=0.01)

        assert body["prediction_after"] == pytest.approx(3.408)
        assert body["loss_after"] == pytest.approx(2.899232)

    def test_stepping_again_from_the_stepped_weights_continues_the_walk(
        self,
    ) -> None:
        first = worked_example()
        hidden, output = first["layers"]
        second = worked_example(
            hidden_neurons=[
                {"weights": weights, "bias": bias}
                for weights, bias in zip(
                    hidden["weights_after"], hidden["biases_after"], strict=True
                )
            ],
            output_neuron={
                "weights": output["weights_after"][0],
                "bias": output["biases_after"][0],
            },
        )

        assert second["loss_before"] == pytest.approx(first["loss_after"])
        assert second["loss_after"] < second["loss_before"]

    def test_after_the_step_the_third_neuron_goes_dark_as_well(self) -> None:
        first = worked_example()
        hidden, output = first["layers"]
        second = worked_example(
            hidden_neurons=[
                {"weights": weights, "bias": bias}
                for weights, bias in zip(
                    hidden["weights_after"], hidden["biases_after"], strict=True
                )
            ],
            output_neuron={
                "weights": output["weights_after"][0],
                "bias": output["biases_after"][0],
            },
        )

        assert second["layers"][0]["scores"] == pytest.approx([1.8, -1.0, -1.4])
        assert second["layers"][0]["outputs"] == pytest.approx([1.8, 0.0, 0.0])
        assert second["prediction"] == pytest.approx(0.52)


class TestFiniteDifference:
    def test_the_chain_rule_and_the_nudge_agree_on_the_first_output_weight(
        self,
    ) -> None:
        check = worked_example()["finite_difference"]

        assert check["layer_index"] == 1
        assert check["neuron_index"] == 0
        assert check["input_index"] == 0
        assert check["analytic"] == 12.0
        assert check["epsilon"] == 1e-6
        assert check["disagreement"] < 1e-6

    def test_the_analytic_slope_is_the_one_written_on_the_edge(self) -> None:
        body = worked_example()

        assert (
            body["finite_difference"]["analytic"]
            == body["layers"][1]["weight_gradient"][0][0]
        )

    def test_the_check_can_be_pointed_at_a_hidden_weight(self) -> None:
        check = worked_example(
            checked_weight={"layer_index": 0, "neuron_index": 2, "input_index": 1}
        )["finite_difference"]

        assert check["analytic"] == 16.0
        assert check["disagreement"] < 1e-6

    def test_the_nudge_agrees_that_nothing_moves_a_dead_neuron(self) -> None:
        check = worked_example(
            checked_weight={"layer_index": 0, "neuron_index": 1, "input_index": 0}
        )["finite_difference"]

        assert check["analytic"] == 0.0
        assert check["numerical"] == 0.0


class TestOtherBends:
    def test_a_sigmoid_hidden_layer_still_agrees_with_the_nudge(self) -> None:
        check = worked_example(
            hidden_activation="sigmoid",
            checked_weight={"layer_index": 0, "neuron_index": 0, "input_index": 1},
        )["finite_difference"]

        assert check["analytic"] != 0.0
        assert check["disagreement"] < 1e-6

    def test_a_sigmoid_slope_never_exceeds_a_quarter(self) -> None:
        hidden = worked_example(hidden_activation="sigmoid")["layers"][0]

        assert all(0.0 < slope <= 0.25 for slope in hidden["slopes"])

    def test_an_identity_hidden_layer_has_unit_slopes(self) -> None:
        hidden = worked_example(hidden_activation="identity")["layers"][0]

        assert hidden["slopes"] == [1.0, 1.0, 1.0]

    def test_a_hyperbolic_tangent_hidden_layer_still_agrees_with_the_nudge(
        self,
    ) -> None:
        check = worked_example(
            hidden_activation="hyperbolic_tangent",
            checked_weight={"layer_index": 0, "neuron_index": 2, "input_index": 0},
        )["finite_difference"]

        assert check["disagreement"] < 1e-6


class TestRefusals:
    def test_a_hidden_neuron_with_the_wrong_weight_count_is_refused(self) -> None:
        response = client.post(
            STEP,
            json={
                **WORKED_EXAMPLE,
                "hidden_neurons": [
                    {"weights": [1.0, 1.0, 1.0], "bias": 0.0},
                    {"weights": [1.0, -1.0], "bias": 0.0},
                    {"weights": [-1.0, 1.0], "bias": 0.0},
                ],
            },
        )

        assert response.status_code == 422

    def test_an_output_neuron_with_the_wrong_weight_count_is_refused(self) -> None:
        response = client.post(
            STEP,
            json={**WORKED_EXAMPLE, "output_neuron": {"weights": [1.0], "bias": 0.0}},
        )

        assert response.status_code == 422

    def test_a_checked_weight_the_layer_does_not_have_is_refused_in_words(
        self,
    ) -> None:
        response = client.post(
            STEP,
            json={
                **WORKED_EXAMPLE,
                "checked_weight": {
                    "layer_index": 1,
                    "neuron_index": 2,
                    "input_index": 0,
                },
            },
        )

        assert response.status_code == 422
        assert "the output layer" in response.json()["detail"]

    def test_a_hidden_input_past_the_row_width_is_refused_in_words(self) -> None:
        response = client.post(
            STEP,
            json={
                **WORKED_EXAMPLE,
                "checked_weight": {
                    "layer_index": 0,
                    "neuron_index": 0,
                    "input_index": 2,
                },
            },
        )

        assert response.status_code == 422
        assert "the hidden layer" in response.json()["detail"]

    def test_a_learning_rate_of_zero_is_refused(self) -> None:
        response = client.post(STEP, json={**WORKED_EXAMPLE, "learning_rate": 0.0})

        assert response.status_code == 422

    def test_a_learning_rate_past_one_is_refused(self) -> None:
        response = client.post(STEP, json={**WORKED_EXAMPLE, "learning_rate": 1.5})

        assert response.status_code == 422

    def test_an_unknown_bend_is_refused(self) -> None:
        response = client.post(
            STEP, json={**WORKED_EXAMPLE, "hidden_activation": "softmax"}
        )

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(STEP, json={**WORKED_EXAMPLE, "epochs": 3})

        assert response.status_code == 422
