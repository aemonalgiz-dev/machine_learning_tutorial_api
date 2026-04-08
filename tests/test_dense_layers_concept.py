"""The dense-layers endpoints, pinned to the page's hand arithmetic.

The page works the row (2, 1) through a two-to-three-to-one stack with
whole-number weights. Hidden rows (1, 2), (-1, 3) and (1, -3) with biases 0,
1 and 0 score 4, 2 and -1. The rectifier keeps the first two and turns the
third to 0, and the output neuron, weights (1, -1, 2) and bias 1, scores
4 - 2 + 0 + 1 = 3, which the identity hands back unchanged. Under a sigmoid
the third hidden neuron answers 0.2689 instead and the output moves to
1.6391. With both bends removed the stack collapses to one affine map with
weights (4, -7) and bias 0, which answers 1 at the row, and the library says
the same when the two layers are run with identity bends. A hidden layer
answering three numbers refuses to meet an output layer reading four, in the
library's own words, with no row sent. All pinned.
"""

import math

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

FORWARD = "/concepts/dense-layers/forward"
JOIN = "/concepts/dense-layers/join"

WORKED_ROW = [2, 1]

WORKED_HIDDEN = {
    "weights": [[1, 2], [-1, 3], [1, -3]],
    "biases": [0, 1, 0],
    "activation": "rectified_linear",
}

WORKED_OUTPUT = {
    "weights": [[1, -1, 2]],
    "biases": [1],
    "activation": "identity",
}


def forward_with(
    hidden_activation: str = "rectified_linear",
    output_activation: str = "identity",
) -> dict:
    response = client.post(
        FORWARD,
        json={
            "inputs": WORKED_ROW,
            "hidden": {**WORKED_HIDDEN, "activation": hidden_activation},
            "output": {**WORKED_OUTPUT, "activation": output_activation},
        },
    )
    assert response.status_code == 200
    return response.json()


def sigmoid(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-score))


class TestTheWorkedExample:
    def test_the_hidden_scores_are_four_two_and_minus_one(self) -> None:
        body = forward_with()

        assert body["layers"][0]["scores"] == [4.0, 2.0, -1.0]

    def test_the_rectifier_switches_the_third_neuron_off(self) -> None:
        body = forward_with()

        assert body["layers"][0]["outputs"] == [4.0, 2.0, 0.0]

    def test_the_output_scores_three_and_the_identity_keeps_it(self) -> None:
        body = forward_with()

        assert body["layers"][1]["scores"] == [3.0]
        assert body["layers"][1]["outputs"] == [3.0]
        assert body["output"] == 3.0

    def test_each_layer_reads_what_the_one_beneath_answered(self) -> None:
        body = forward_with()

        hidden, output = body["layers"]
        assert hidden["inputs"] == [2.0, 1.0]
        assert output["inputs"] == hidden["outputs"]

    def test_the_shapes_are_tuples_of_extents_end_to_end(self) -> None:
        body = forward_with()

        hidden, output = body["layers"]
        assert (hidden["reads"], hidden["answers"]) == ([2], [3])
        assert (output["reads"], output["answers"]) == ([3], [1])
        assert body["shape"] == {"reads": [2], "answers": [1]}

    def test_the_formulas_are_the_library_descriptions(self) -> None:
        body = forward_with()

        hidden, output = body["layers"]
        assert hidden["activation"] == "rectified_linear"
        assert hidden["formula"] == "max(0, z)"
        assert output["activation"] == "identity"
        assert output["formula"] == "z"

    def test_the_pass_states_that_it_is_predicting(self) -> None:
        body = forward_with()

        assert body["purpose"] == "predicting"


class TestTheOtherBends:
    def test_a_sigmoid_wakes_the_dead_neuron_to_point_two_six_eight_nine(self) -> None:
        body = forward_with(hidden_activation="sigmoid")

        outputs = body["layers"][0]["outputs"]
        assert [round(value, 4) for value in outputs] == [0.982, 0.8808, 0.2689]
        assert outputs == [sigmoid(4.0), sigmoid(2.0), sigmoid(-1.0)]

    def test_under_a_sigmoid_the_output_moves_to_one_point_six_three_nine_one(
        self,
    ) -> None:
        body = forward_with(hidden_activation="sigmoid")

        hidden_outputs = body["layers"][0]["outputs"]
        by_hand = (
            1 * hidden_outputs[0] - 1 * hidden_outputs[1] + 2 * hidden_outputs[2] + 1
        )
        assert round(body["output"], 4) == 1.6391
        assert math.isclose(body["output"], by_hand, rel_tol=0, abs_tol=1e-12)

    def test_the_hyperbolic_tangent_bends_the_same_scores(self) -> None:
        body = forward_with(hidden_activation="hyperbolic_tangent")

        assert body["layers"][0]["scores"] == [4.0, 2.0, -1.0]
        assert body["layers"][0]["outputs"] == [
            math.tanh(4.0),
            math.tanh(2.0),
            math.tanh(-1.0),
        ]
        assert body["layers"][0]["formula"] == "tanh(z)"

    def test_a_sigmoid_output_layer_squashes_the_three(self) -> None:
        body = forward_with(output_activation="sigmoid")

        assert body["layers"][1]["scores"] == [3.0]
        assert body["output"] == sigmoid(3.0)
        assert body["layers"][1]["formula"] == "1 / (1 + exp(-z))"


class TestWithoutTheBends:
    def test_the_two_layers_collapse_to_four_minus_seven_and_zero(self) -> None:
        body = forward_with()

        assert body["without_bends"]["weights"] == [4.0, -7.0]
        assert body["without_bends"]["bias"] == 0.0

    def test_the_collapsed_map_answers_one_at_the_row(self) -> None:
        body = forward_with()

        assert body["without_bends"]["output"] == 1.0
        assert 4.0 * 2 + (-7.0) * 1 + 0.0 == body["without_bends"]["output"]

    def test_two_identity_bends_give_the_collapsed_answer(self) -> None:
        body = forward_with(hidden_activation="identity", output_activation="identity")

        assert body["output"] == 1.0
        assert body["output"] == body["without_bends"]["output"]
        assert body["layers"][0]["outputs"] == [4.0, 2.0, -1.0]

    def test_the_collapse_ignores_the_bends_asked_for(self) -> None:
        rectified = forward_with()["without_bends"]
        squashed = forward_with(hidden_activation="sigmoid")["without_bends"]

        assert rectified == squashed


class TestJoin:
    def test_three_answered_and_three_read_is_a_stack(self) -> None:
        response = client.post(
            JOIN,
            json={
                "beneath": {"reads": 2, "answers": 3},
                "above": {"reads": 3, "answers": 1},
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["layers"] == [
            {"reads": [2], "answers": [3]},
            {"reads": [3], "answers": [1]},
        ]
        assert body["stack"] == {"reads": [2], "answers": [1]}

    def test_three_answered_and_four_read_is_refused_in_the_library_words(
        self,
    ) -> None:
        response = client.post(
            JOIN,
            json={
                "beneath": {"reads": 2, "answers": 3},
                "above": {"reads": 4, "answers": 1},
            },
        )

        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "ShapeMismatchError"
        assert body["detail"] == "layer 0 answers with (3,) and layer 1 reads (4,)"

    def test_only_the_seam_is_checked_and_the_ends_are_free(self) -> None:
        response = client.post(
            JOIN,
            json={
                "beneath": {"reads": 8, "answers": 5},
                "above": {"reads": 5, "answers": 8},
            },
        )

        assert response.status_code == 200
        assert response.json()["stack"] == {"reads": [8], "answers": [8]}

    def test_a_width_beyond_the_bound_is_refused(self) -> None:
        response = client.post(
            JOIN,
            json={
                "beneath": {"reads": 2, "answers": 9},
                "above": {"reads": 9, "answers": 1},
            },
        )

        assert response.status_code == 422

    def test_a_width_of_zero_is_refused(self) -> None:
        response = client.post(
            JOIN,
            json={
                "beneath": {"reads": 2, "answers": 0},
                "above": {"reads": 0, "answers": 1},
            },
        )

        assert response.status_code == 422


class TestRefusals:
    def test_a_row_of_the_wrong_width_is_refused(self) -> None:
        response = client.post(
            FORWARD,
            json={
                "inputs": [1, 2, 3],
                "hidden": WORKED_HIDDEN,
                "output": WORKED_OUTPUT,
            },
        )

        assert response.status_code == 422

    def test_a_hidden_neuron_with_the_wrong_number_of_weights_is_refused(
        self,
    ) -> None:
        response = client.post(
            FORWARD,
            json={
                "inputs": WORKED_ROW,
                "hidden": {**WORKED_HIDDEN, "weights": [[1, 2, 3], [-1, 3], [1, -3]]},
                "output": WORKED_OUTPUT,
            },
        )

        assert response.status_code == 422

    def test_a_fourth_hidden_neuron_is_refused(self) -> None:
        response = client.post(
            FORWARD,
            json={
                "inputs": WORKED_ROW,
                "hidden": {
                    "weights": [[1, 2], [-1, 3], [1, -3], [0, 0]],
                    "biases": [0, 1, 0, 0],
                    "activation": "rectified_linear",
                },
                "output": WORKED_OUTPUT,
            },
        )

        assert response.status_code == 422

    def test_softmax_is_not_on_the_list(self) -> None:
        response = client.post(
            FORWARD,
            json={
                "inputs": WORKED_ROW,
                "hidden": {**WORKED_HIDDEN, "activation": "softmax"},
                "output": WORKED_OUTPUT,
            },
        )

        assert response.status_code == 422

    def test_a_weight_beyond_the_magnitude_bound_is_refused(self) -> None:
        response = client.post(
            FORWARD,
            json={
                "inputs": WORKED_ROW,
                "hidden": {**WORKED_HIDDEN, "weights": [[1e7, 2], [-1, 3], [1, -3]]},
                "output": WORKED_OUTPUT,
            },
        )

        assert response.status_code == 422

    def test_an_unexpected_field_is_refused(self) -> None:
        response = client.post(
            FORWARD,
            json={
                "inputs": WORKED_ROW,
                "hidden": WORKED_HIDDEN,
                "output": WORKED_OUTPUT,
                "purpose": "training",
            },
        )

        assert response.status_code == 422
