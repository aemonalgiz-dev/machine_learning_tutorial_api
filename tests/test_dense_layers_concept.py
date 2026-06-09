"""The dense-layers endpoints, pinned to the page's hand arithmetic.

The page works the row (1, 2) through the backpropagation page's 2-3-1
network: hidden neurons (1, 1), (1, -1) and (-1, 1) with no biases under the
rectifier, and the output neuron (1, -1, 2) with no bias under the identity.
The hidden scores are 3, -1 and 1, the rectifier turns the middle one to 0,
and the output is 3 - 0 + 2 = 5. Under a sigmoid the hidden layer answers
0.9526, 0.2689 and 0.7311 and the output moves to 2.1457. With both bends
removed the two layers collapse to one map with weights (-2, 4) and bias 0,
which answers 6 at the row, and the library says the same when the two
layers run with identity bends. A block of three rows goes up as (3, 2) then
(3, 3) then (3, 1). A chain whose second seam is wrong is refused naming
layer 1 and layer 2. On the crowd, one hidden layer of six rectifiers calls
every person correctly from four of six starts and a layer of one never does.
All pinned.
"""

import math

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

FORWARD = "/concepts/dense-layers/forward"
BLOCK = "/concepts/dense-layers/block"
ROUTES = "/concepts/dense-layers/routes"
JOIN = "/concepts/dense-layers/join"
CHAIN = "/concepts/dense-layers/chain"
CARVE = "/concepts/dense-layers/carve"

WORKED_ROW = [1, 2]

WORKED_HIDDEN = {
    "weights": [[1, 1], [1, -1], [-1, 1]],
    "biases": [0, 0, 0],
    "activation": "rectified_linear",
}

WORKED_OUTPUT = {
    "weights": [[1, -1, 2]],
    "biases": [0],
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
    assert response.status_code == 200, response.json()
    return response.json()


def block_with(rows: list[list[float]]) -> dict:
    response = client.post(
        BLOCK, json={"rows": rows, "hidden": WORKED_HIDDEN, "output": WORKED_OUTPUT}
    )
    assert response.status_code == 200, response.json()
    return response.json()


def carve(widths: list[int]) -> dict:
    response = client.post(CARVE, json={"hidden_widths": widths})
    assert response.status_code == 200, response.json()
    return response.json()


def sigmoid(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-score))


class TestTheWorkedExample:
    def test_the_hidden_scores_are_three_minus_one_and_one(self) -> None:
        body = forward_with()

        assert body["layers"][0]["scores"] == [3.0, -1.0, 1.0]

    def test_the_rectifier_switches_the_middle_neuron_off(self) -> None:
        body = forward_with()

        assert body["layers"][0]["outputs"] == [3.0, 0.0, 1.0]

    def test_the_output_scores_five_and_the_identity_keeps_it(self) -> None:
        body = forward_with()

        assert body["layers"][1]["scores"] == [5.0]
        assert body["layers"][1]["outputs"] == [5.0]
        assert body["output"] == 5.0

    def test_each_layer_reads_what_the_one_beneath_answered(self) -> None:
        body = forward_with()

        hidden, output = body["layers"]
        assert hidden["inputs"] == [1.0, 2.0]
        assert output["inputs"] == hidden["outputs"]

    def test_the_shapes_are_tuples_of_extents_end_to_end(self) -> None:
        body = forward_with()

        hidden, output = body["layers"]
        assert (hidden["reads"], hidden["answers"]) == ([2], [3])
        assert (output["reads"], output["answers"]) == ([3], [1])
        assert body["shape"] == {"reads": [2], "answers": [1]}
        assert body["n_layers"] == 2

    def test_the_weight_matrices_have_one_row_per_neuron(self) -> None:
        body = forward_with()

        hidden, output = body["layers"]
        assert hidden["weight_matrix"] == [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]]
        assert hidden["bias_vector"] == [0.0, 0.0, 0.0]
        assert output["weight_matrix"] == [[1.0, -1.0, 2.0]]
        assert output["bias_vector"] == [0.0]

    def test_the_network_holds_thirteen_parameters(self) -> None:
        body = forward_with()

        assert [layer["n_parameters"] for layer in body["layers"]] == [9, 4]
        assert body["n_parameters"] == 13

    def test_the_neuron_route_prints_each_dot_product_term_by_term(self) -> None:
        body = forward_with()

        hidden, output = body["layers"]
        assert hidden["neuron_route"]["products"] == [
            [1.0, 2.0],
            [1.0, -2.0],
            [-1.0, 2.0],
        ]
        assert hidden["neuron_route"]["scores"] == [3.0, -1.0, 1.0]
        assert hidden["neuron_route"]["outputs"] == [3.0, 0.0, 1.0]
        assert output["neuron_route"]["products"] == [[3.0, 0.0, 2.0]]

    def test_the_two_routes_agree_exactly_on_whole_numbers(self) -> None:
        body = forward_with()

        assert [layer["route_gap"] for layer in body["layers"]] == [0.0, 0.0]

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
        assert [round(value, 4) for value in outputs] == [0.9526, 0.2689, 0.7311]
        assert outputs == [sigmoid(3.0), sigmoid(-1.0), sigmoid(1.0)]

    def test_under_a_sigmoid_the_output_moves_to_two_point_one_four_five_seven(
        self,
    ) -> None:
        body = forward_with(hidden_activation="sigmoid")

        hidden_outputs = body["layers"][0]["outputs"]
        by_hand = 1 * hidden_outputs[0] - 1 * hidden_outputs[1] + 2 * hidden_outputs[2]
        assert round(body["output"], 4) == 2.1457
        assert math.isclose(body["output"], by_hand, rel_tol=0, abs_tol=1e-12)

    def test_the_hyperbolic_tangent_bends_the_same_scores(self) -> None:
        body = forward_with(hidden_activation="hyperbolic_tangent")

        assert body["layers"][0]["scores"] == [3.0, -1.0, 1.0]
        assert body["layers"][0]["outputs"] == [
            math.tanh(3.0),
            math.tanh(-1.0),
            math.tanh(1.0),
        ]
        assert round(body["output"], 4) == 3.2798
        assert body["layers"][0]["formula"] == "tanh(z)"

    def test_a_sigmoid_output_layer_squashes_the_five(self) -> None:
        body = forward_with(output_activation="sigmoid")

        assert body["layers"][1]["scores"] == [5.0]
        assert body["output"] == sigmoid(5.0)
        assert round(body["output"], 4) == 0.9933
        assert body["layers"][1]["formula"] == "1 / (1 + exp(-z))"


class TestWithoutTheBends:
    def test_the_two_layers_collapse_to_minus_two_four_and_zero(self) -> None:
        body = forward_with()

        assert body["without_bends"]["weights"] == [-2.0, 4.0]
        assert body["without_bends"]["bias"] == 0.0

    def test_the_collapsed_map_answers_six_at_the_row(self) -> None:
        body = forward_with()

        assert body["without_bends"]["output"] == 6.0
        assert -2.0 * 1 + 4.0 * 2 + 0.0 == body["without_bends"]["output"]

    def test_two_identity_bends_give_the_collapsed_answer(self) -> None:
        body = forward_with(hidden_activation="identity", output_activation="identity")

        assert body["output"] == 6.0
        assert body["output"] == body["without_bends"]["output"]
        assert body["layers"][0]["outputs"] == [3.0, -1.0, 1.0]

    def test_the_collapse_ignores_the_bends_asked_for(self) -> None:
        rectified = forward_with()["without_bends"]
        squashed = forward_with(hidden_activation="sigmoid")["without_bends"]

        assert rectified == squashed


class TestTheBlock:
    def test_three_rows_go_up_as_three_by_two_then_three_by_three_then_three_by_one(
        self,
    ) -> None:
        body = block_with([[1, 2], [0, 0], [-1, 0.5]])

        hidden, output = body["layers"]
        assert body["n_rows"] == 3
        assert hidden["inputs_shape"] == [3, 2]
        assert hidden["scores_shape"] == [3, 3]
        assert output["inputs_shape"] == [3, 3]
        assert output["scores_shape"] == [3, 1]

    def test_the_worked_row_is_the_first_row_of_the_block(self) -> None:
        body = block_with([[1, 2], [0, 0], [-1, 0.5]])

        hidden, output = body["layers"]
        assert hidden["scores"] == [
            [3.0, -1.0, 1.0],
            [0.0, 0.0, 0.0],
            [-0.5, -1.5, 1.5],
        ]
        assert hidden["outputs"] == [[3.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]
        assert output["inputs"] == hidden["outputs"]
        assert body["outputs"] == [5.0, 0.0, 3.0]

    def test_a_block_of_one_row_is_the_ordinary_case(self) -> None:
        body = block_with([[1, 2]])

        assert body["layers"][0]["inputs_shape"] == [1, 2]
        assert body["outputs"] == [5.0]

    def test_a_row_of_width_three_is_refused_by_the_first_layer(self) -> None:
        response = client.post(
            BLOCK,
            json={
                "rows": [[1, 2, 3]],
                "hidden": WORKED_HIDDEN,
                "output": WORKED_OUTPUT,
            },
        )

        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "ShapeMismatchError"
        assert body["detail"] == "this layer reads (2,), got a block arranged (3,)"

    def test_a_row_of_width_one_is_refused_the_same_way(self) -> None:
        response = client.post(
            BLOCK,
            json={"rows": [[1]], "hidden": WORKED_HIDDEN, "output": WORKED_OUTPUT},
        )

        assert response.status_code == 422
        assert response.json()["detail"] == (
            "this layer reads (2,), got a block arranged (1,)"
        )

    def test_no_rows_at_all_is_refused_at_the_door(self) -> None:
        response = client.post(
            BLOCK, json={"rows": [], "hidden": WORKED_HIDDEN, "output": WORKED_OUTPUT}
        )

        assert response.status_code == 422

    def test_seven_rows_is_more_than_the_page_allows(self) -> None:
        response = client.post(
            BLOCK,
            json={
                "rows": [[1, 2]] * 7,
                "hidden": WORKED_HIDDEN,
                "output": WORKED_OUTPUT,
            },
        )

        assert response.status_code == 422


class TestTheTwoRoutes:
    def test_a_two_by_three_layer_agrees_to_the_last_bit(self) -> None:
        response = client.post(ROUTES, json={"n_inputs": 2, "n_neurons": 3, "seed": 0})

        assert response.status_code == 200
        body = response.json()
        assert body["score_gap"] == 0.0
        assert body["output_gap"] == 0.0
        assert body["n_parameters"] == 9
        assert body["first_scores_matrix"] == body["first_scores_loop"]

    def test_a_sixty_four_wide_layer_parts_in_the_last_bits(self) -> None:
        response = client.post(
            ROUTES, json={"n_inputs": 64, "n_neurons": 64, "seed": 0}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["score_gap"] == pytest.approx(7.105427357601002e-15)
        assert 0.0 < body["output_gap"] < 1e-14
        assert body["n_parameters"] == 4160

    def test_the_gap_is_never_more_than_a_few_ulps(self) -> None:
        for seed in range(4):
            body = client.post(
                ROUTES, json={"n_inputs": 32, "n_neurons": 64, "seed": seed}
            ).json()
            assert body["score_gap"] < 1e-14

    def test_a_layer_wider_than_sixty_four_is_refused(self) -> None:
        response = client.post(ROUTES, json={"n_inputs": 65, "n_neurons": 1})

        assert response.status_code == 422


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


class TestChain:
    def test_a_chain_of_three_that_holds_reports_two_seams(self) -> None:
        response = client.post(
            CHAIN,
            json={
                "layers": [
                    {"reads": 2, "answers": 3},
                    {"reads": 3, "answers": 4},
                    {"reads": 4, "answers": 1},
                ]
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["stack"] == {"reads": [2], "answers": [1]}
        assert body["n_seams"] == 2
        assert body["n_parameters"] == 9 + 16 + 5

    def test_a_broken_second_seam_is_named_as_layer_one_and_layer_two(self) -> None:
        response = client.post(
            CHAIN,
            json={
                "layers": [
                    {"reads": 2, "answers": 3},
                    {"reads": 3, "answers": 4},
                    {"reads": 5, "answers": 1},
                ]
            },
        )

        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "ShapeMismatchError"
        assert body["detail"] == "layer 1 answers with (4,) and layer 2 reads (5,)"

    def test_the_first_broken_seam_is_the_one_named(self) -> None:
        response = client.post(
            CHAIN,
            json={
                "layers": [
                    {"reads": 2, "answers": 3},
                    {"reads": 4, "answers": 4},
                    {"reads": 5, "answers": 1},
                ]
            },
        )

        assert response.status_code == 422
        assert response.json()["detail"] == (
            "layer 0 answers with (3,) and layer 1 reads (4,)"
        )

    def test_a_chain_of_one_has_no_seams_and_is_its_own_shape(self) -> None:
        response = client.post(CHAIN, json={"layers": [{"reads": 2, "answers": 3}]})

        assert response.status_code == 200
        body = response.json()
        assert body["n_seams"] == 0
        assert body["stack"] == {"reads": [2], "answers": [3]}

    def test_no_layers_at_all_is_refused_at_the_door(self) -> None:
        response = client.post(CHAIN, json={"layers": []})

        assert response.status_code == 422

    def test_seven_layers_is_more_than_the_page_allows(self) -> None:
        response = client.post(CHAIN, json={"layers": [{"reads": 2, "answers": 2}] * 7})

        assert response.status_code == 422


class TestCarvingTheCrowd:
    def test_the_crowd_is_standardised_about_the_neurons_page_means(self) -> None:
        body = carve([3])

        assert body["mean_height"] == pytest.approx(151.48)
        assert body["deviation_height"] == pytest.approx(15.613122685740992)
        assert body["mean_weight"] == pytest.approx(51.48)
        assert body["deviation_weight"] == pytest.approx(14.447477288440359)
        assert len(body["people"]) == 25

    def test_the_settings_are_pinned_and_the_line_scores_sixty_eight(self) -> None:
        body = carve([3])

        assert body["learning_rate"] == 0.5
        assert body["epochs"] == 1500
        assert body["straight_line_accuracy"] == pytest.approx(0.68)

    def test_one_rectifier_never_calls_everyone_and_can_die(self) -> None:
        body = carve([1])

        assert body["n_parameters"] == 5
        assert body["best_accuracy"] == pytest.approx(0.84)
        assert body["mean_accuracy"] == pytest.approx(0.7066666666666667)
        assert body["seeds_reaching_everyone"] == 0
        assert [run["accuracy"] for run in body["runs"]] == pytest.approx(
            [0.68, 0.84, 0.76, 0.76, 0.52, 0.68]
        )
        assert body["runs"][4]["dead_units"] == [1]
        assert round(body["runs"][4]["final_loss"], 4) == 0.6923

    def test_two_rectifiers_reach_twenty_one_of_twenty_five_from_every_start(
        self,
    ) -> None:
        body = carve([2])

        assert body["n_parameters"] == 9
        assert [run["n_correct"] for run in body["runs"]] == [21] * 6
        assert body["mean_accuracy"] == pytest.approx(0.84)

    def test_three_rectifiers_call_everyone_from_one_start_in_six(self) -> None:
        body = carve([3])

        assert body["n_parameters"] == 13
        assert body["best_seed"] == 2
        assert body["best_accuracy"] == 1.0
        assert body["seeds_reaching_everyone"] == 1
        assert body["mean_accuracy"] == pytest.approx(0.8866666666666666)
        assert round(body["runs"][2]["final_loss"], 4) == 0.0725
        assert len(body["folds"]) == 3
        assert not any(fold["dead"] for fold in body["folds"])
        assert all(
            person["is_adult"] == person["called_adult"] for person in body["people"]
        )

    def test_the_folds_are_the_first_layer_units_of_the_best_run(self) -> None:
        body = carve([3])

        first = body["folds"][0]
        assert round(first["weight_height"], 3) == -5.912
        assert round(first["weight_weight"], 3) == -2.63
        assert round(first["bias"], 3) == -1.853

    def test_six_rectifiers_call_everyone_from_four_starts_in_six(self) -> None:
        body = carve([6])

        assert body["n_parameters"] == 25
        assert body["seeds_reaching_everyone"] == 4
        assert body["mean_accuracy"] == pytest.approx(0.96)
        assert [run["accuracy"] for run in body["runs"]] == pytest.approx(
            [1.0, 1.0, 1.0, 0.84, 1.0, 0.92]
        )
        assert round(body["runs"][4]["final_loss"], 4) == 0.0388

    def test_eight_rectifiers_call_everyone_from_every_start(self) -> None:
        body = carve([8])

        assert body["n_parameters"] == 33
        assert body["seeds_reaching_everyone"] == 6
        assert body["mean_accuracy"] == 1.0
        assert body["runs"][1]["dead_units"] == [1]

    def test_two_layers_of_three_hold_the_same_twenty_five_parameters(self) -> None:
        wide = carve([6])
        deep = carve([3, 3])

        assert deep["n_parameters"] == wide["n_parameters"] == 25
        assert deep["seeds_reaching_everyone"] == 2
        assert deep["mean_accuracy"] == pytest.approx(0.8866666666666666)
        assert round(deep["runs"][2]["final_loss"], 4) == 0.0045
        assert round(deep["runs"][5]["final_loss"], 4) == 0.006
        assert deep["runs"][3]["accuracy"] == pytest.approx(0.76)
        assert deep["runs"][3]["dead_units"] == [0, 2]

    def test_two_layers_of_six_reach_everyone_every_time_at_a_far_lower_loss(
        self,
    ) -> None:
        body = carve([6, 6])

        assert body["n_parameters"] == 67
        assert body["seeds_reaching_everyone"] == 6
        assert max(run["final_loss"] for run in body["runs"]) < 0.004
        assert round(body["runs"][2]["final_loss"], 4) == 0.0013

    def test_the_region_covers_the_crowd_in_raw_units(self) -> None:
        body = carve([3])

        regions = body["regions"]
        assert regions["cells"] == 26
        assert regions["x_min"] == pytest.approx(114.75)
        assert regions["x_max"] == pytest.approx(186.25)
        assert regions["y_min"] == pytest.approx(21.05)
        assert regions["y_max"] == pytest.approx(85.95)

    def test_fewer_seeds_are_a_prefix_of_the_six(self) -> None:
        response = client.post(CARVE, json={"hidden_widths": [3], "n_seeds": 2})

        assert response.status_code == 200
        body = response.json()
        assert [run["seed"] for run in body["runs"]] == [0, 1]
        assert [run["accuracy"] for run in body["runs"]] == pytest.approx([0.92, 0.84])

    def test_a_fourth_hidden_layer_a_ninth_unit_or_a_seventh_seed_is_refused(
        self,
    ) -> None:
        assert (
            client.post(CARVE, json={"hidden_widths": [1, 1, 1, 1]}).status_code == 422
        )
        assert client.post(CARVE, json={"hidden_widths": [9]}).status_code == 422
        assert (
            client.post(CARVE, json={"hidden_widths": [3], "n_seeds": 7}).status_code
            == 422
        )


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
                "hidden": {**WORKED_HIDDEN, "weights": [[1, 1, 1], [1, -1], [-1, 1]]},
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
                    "weights": [[1, 1], [1, -1], [-1, 1], [0, 0]],
                    "biases": [0, 0, 0, 0],
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
                "hidden": {**WORKED_HIDDEN, "weights": [[1e7, 1], [1, -1], [-1, 1]]},
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
