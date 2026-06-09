"""The normalisation-layers endpoints, pinned to the page's hand arithmetic.

The page works a four-row block of two features, (1, 7), (1, 9), (7, 1) and
(23, 7), chosen so that every statistic the batch layer reads down a column is
a whole number, means of 8 and 6 and deviations of 9 and 3, and so that the
first row is whole across as well, a mean of 4, a deviation of 3 and a root
mean square of 5. The library keeps epsilon inside the root, so the API
reports the deviations as 9.0000006 and 3.0000017 rather than 9 and 3, and the
page quotes them that way. After one training pass and one step the running
figures sit a tenth of the way from mean 0, variance 1 toward the batch, so
the second column predicts by a mean of 0.6 and a deviation of 1.3416445. All
pinned.

The backward endpoint is pinned to the same block under a counting slope: the
library's passed-down block agrees with the finite difference to 2.1e-10 and
the naive form misses it by 2.04 against a largest true slope of 1.07, with
every column of the true block summing to zero. The running figures reach
within a hundredth of the batch answer at step 53 and a thousandth at 75. The
crowd experiment is pinned to the numbers the page quotes: the drift of the
hidden block, the five arrangements' training and held-out accuracy, the
epoch each loss curve first falls below 0.3, and the reparameterisation check.
"""

import math

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

NORMALISE = "/concepts/normalisation-layers/normalise"
BACKWARD = "/concepts/normalisation-layers/backward"
RUNNING = "/concepts/normalisation-layers/running-figures"
CROWD = "/concepts/normalisation-layers/crowd"

EPSILON = 1e-5

WORKED_BLOCK = [[1, 7], [1, 9], [7, 1], [23, 7]]

THREE_FEATURES = [[5, 4, 3], [7, 4, 9], [7, 8, 11], [13, 8, 17]]

TWENTY_OF_THE_CROWD = [
    [147, 41],
    [156, 53],
    [145, 57],
    [159, 57],
    [162, 61],
    [120, 25],
    [122, 28],
    [118, 24],
    [180, 80],
    [183, 83],
    [178, 78],
    [145, 45],
    [145, 55],
    [151, 45],
    [151, 55],
    [157, 45],
    [157, 55],
    [148, 50],
    [154, 50],
    [160, 50],
]


def normalised_by(layer: str, rows: list[list[float]], **changes: object) -> dict:
    response = client.post(NORMALISE, json={"rows": rows, "layer": layer, **changes})
    assert response.status_code == 200, response.text
    return response.json()


def backward_through(layer: str, rows: list[list[float]], **changes: object) -> dict:
    response = client.post(BACKWARD, json={"rows": rows, "layer": layer, **changes})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def crowd() -> dict:
    response = client.get(CROWD)
    assert response.status_code == 200
    return response.json()


class TestTheWorkedExampleUnderBatch:
    def test_the_column_means_are_eight_and_six(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        assert [entry["mean"] for entry in body["statistics"]] == [8.0, 6.0]

    def test_the_column_deviations_carry_the_epsilon(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        deviations = [entry["deviation"] for entry in body["statistics"]]
        assert deviations == [math.sqrt(81 + EPSILON), math.sqrt(9 + EPSILON)]
        assert [round(value, 7) for value in deviations] == [9.0000006, 3.0000017]

    def test_the_second_column_normalises_as_the_page_says(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        second = [round(row[1], 7) for row in body["normalised"]]
        assert second == [0.3333331, 0.9999994, -1.6666657, 0.3333331]

    def test_the_first_column_normalises_as_the_page_says(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        first = [round(row[0], 7) for row in body["normalised"]]
        assert first == [-0.7777777, -0.7777777, -0.1111111, 1.6666666]

    def test_the_answer_is_the_standardised_block_at_the_defaults(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        assert body["standardised"] == body["normalised"]

    def test_every_value_is_its_deviation_over_the_column_deviation(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        for row, raw_row in zip(body["normalised"], WORKED_BLOCK, strict=True):
            for value, raw, statistic in zip(
                row, raw_row, body["statistics"], strict=True
            ):
                expected = (raw - statistic["mean"]) / statistic["deviation"]
                assert value == pytest.approx(expected, abs=1e-12)

    def test_one_statistic_per_feature(self) -> None:
        body = normalised_by("batch", THREE_FEATURES)

        assert len(body["statistics"]) == 3
        assert [entry["mean"] for entry in body["statistics"]] == [8.0, 6.0, 10.0]
        assert [round(entry["deviation"], 7) for entry in body["statistics"]] == [
            3.0000017,
            2.0000025,
            5.000001,
        ]


class TestTheScaleAndShiftGiveBackWhatWasTaken:
    def test_the_column_deviation_and_mean_undo_the_standardisation(self) -> None:
        batch = normalised_by("batch", WORKED_BLOCK)
        undone = normalised_by(
            "batch",
            WORKED_BLOCK,
            scale=[entry["deviation"] for entry in batch["statistics"]],
            shift=[entry["mean"] for entry in batch["statistics"]],
        )

        assert undone["largest_gap_to_raw"] < 1e-15
        assert [
            [round(value, 12) for value in row] for row in undone["normalised"]
        ] == [
            [1.0, 7.0],
            [1.0, 9.0],
            [7.0, 1.0],
            [23.0, 7.0],
        ]

    def test_the_standardised_block_is_unchanged_by_the_pair(self) -> None:
        plain = normalised_by("batch", WORKED_BLOCK)
        scaled = normalised_by(
            "batch", WORKED_BLOCK, scale=[2.0, 0.5], shift=[1.0, -1.0]
        )

        assert scaled["standardised"] == plain["standardised"]
        assert [round(value, 7) for value in scaled["normalised"][0]] == [
            -0.5555555,
            -0.8333334,
        ]

    def test_the_default_answer_is_far_from_the_raw_block(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        assert body["largest_gap_to_raw"] == pytest.approx(23 - 1.6666665637860176)

    def test_the_pair_serves_the_row_layers_too(self) -> None:
        # The scale is one per feature and the row layers' statistics are one
        # per row, so the first row's own deviation and mean recover the first
        # row exactly and no other.
        plain = normalised_by("layer", WORKED_BLOCK)
        first = plain["statistics"][0]
        scaled = normalised_by(
            "layer",
            WORKED_BLOCK,
            scale=[first["deviation"], first["deviation"]],
            shift=[first["mean"], first["mean"]],
        )

        assert scaled["standardised"] == plain["standardised"]
        assert scaled["normalised"][0] == pytest.approx([1.0, 7.0], abs=1e-12)
        assert scaled["normalised"][3] != pytest.approx([23.0, 7.0], abs=1e-3)

    def test_a_pair_of_the_wrong_width_is_refused_in_words(self) -> None:
        response = client.post(
            NORMALISE, json={"rows": WORKED_BLOCK, "layer": "batch", "scale": [1, 1, 1]}
        )

        assert response.status_code == 422
        assert "one value per feature" in response.json()["detail"]


class TestPredictingAfterOneStep:
    def test_the_running_figures_moved_a_tenth_of_the_way(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        first, second = body["predicting_statistics"]
        assert first["mean"] == pytest.approx(0.8)
        assert second["mean"] == pytest.approx(0.6)
        assert first["deviation"] == pytest.approx(math.sqrt(0.9 + 0.1 * 81 + EPSILON))
        assert second["deviation"] == pytest.approx(math.sqrt(0.9 + 0.1 * 9 + EPSILON))
        assert round(first["deviation"], 7) == 3.0000017
        assert round(second["deviation"], 7) == 1.3416445

    def test_the_seven_predicts_to_four_point_seven_seven(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        assert round(body["as_predicting"][0][1], 7) == 4.7702651
        assert round(body["as_predicting"][3][0], 7) == 7.3999959

    def test_predicting_divides_by_the_running_figures(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        for row, raw_row in zip(body["as_predicting"], WORKED_BLOCK, strict=True):
            for value, raw, statistic in zip(
                row, raw_row, body["predicting_statistics"], strict=True
            ):
                expected = (raw - statistic["mean"]) / statistic["deviation"]
                assert value == pytest.approx(expected, abs=1e-12)

    def test_the_two_answers_differ(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        assert body["as_predicting"] != body["normalised"]

    def test_the_row_layers_have_no_predicting_block(self) -> None:
        for layer in ("layer", "rms"):
            body = normalised_by(layer, WORKED_BLOCK)

            assert body["as_predicting"] is None
            assert body["predicting_statistics"] is None


class TestTheWorkedExampleAcrossARow:
    def test_layer_normalisation_reads_the_first_row(self) -> None:
        body = normalised_by("layer", WORKED_BLOCK)

        first = body["statistics"][0]
        assert first["mean"] == 4.0
        assert first["deviation"] == math.sqrt(9 + EPSILON)
        assert round(first["deviation"], 7) == 3.0000017
        assert [round(value, 7) for value in body["normalised"][0]] == [
            -0.9999994,
            0.9999994,
        ]

    def test_rms_normalisation_reads_the_first_row(self) -> None:
        body = normalised_by("rms", WORKED_BLOCK)

        first = body["statistics"][0]
        assert first["mean"] is None
        assert first["deviation"] == math.sqrt(25 + EPSILON)
        assert round(first["deviation"], 7) == 5.000001
        assert [round(value, 7) for value in body["normalised"][0]] == [
            0.2,
            1.3999997,
        ]

    def test_one_statistic_per_row(self) -> None:
        for layer in ("layer", "rms"):
            body = normalised_by(layer, WORKED_BLOCK)

            assert len(body["statistics"]) == 4

    def test_every_value_is_its_deviation_over_the_row_deviation(self) -> None:
        for layer in ("layer", "rms"):
            body = normalised_by(layer, THREE_FEATURES)

            for row, raw_row, statistic in zip(
                body["normalised"], THREE_FEATURES, body["statistics"], strict=True
            ):
                centre = statistic["mean"] or 0.0
                for value, raw in zip(row, raw_row, strict=True):
                    expected = (raw - centre) / statistic["deviation"]
                    assert value == pytest.approx(expected, abs=1e-12)


class TestWhichWayTheMeanRuns:
    def test_batch_leaves_every_column_summing_to_zero(self) -> None:
        body = normalised_by("batch", THREE_FEATURES)

        for column in range(3):
            total = sum(row[column] for row in body["normalised"])
            assert total == pytest.approx(0.0, abs=1e-12)

    def test_layer_leaves_every_row_summing_to_zero(self) -> None:
        body = normalised_by("layer", THREE_FEATURES)

        for row in body["normalised"]:
            assert sum(row) == pytest.approx(0.0, abs=1e-12)

    def test_rms_does_not_centre(self) -> None:
        body = normalised_by("rms", THREE_FEATURES)

        for row in body["normalised"]:
            assert sum(row) > 0.0

    def test_a_normalised_column_has_variance_v_over_v_plus_epsilon(self) -> None:
        # The derivation's var(z) = v / (v + epsilon), 9 / 9.00001 on the
        # worked block's second column.
        body = normalised_by("batch", WORKED_BLOCK)

        second = [row[1] for row in body["normalised"]]
        mean = sum(second) / len(second)
        variance = sum((value - mean) ** 2 for value in second) / len(second)
        assert variance == pytest.approx(9 / (9 + EPSILON), abs=1e-12)
        assert variance < 1.0

    def test_a_row_answers_the_same_whatever_else_arrived(self) -> None:
        for layer in ("layer", "rms"):
            alone = normalised_by(layer, WORKED_BLOCK[:2])
            among = normalised_by(layer, WORKED_BLOCK)

            assert alone["normalised"][0] == among["normalised"][0]
            assert alone["statistics"][0] == among["statistics"][0]

    def test_a_batch_row_answers_differently_when_its_company_changes(self) -> None:
        fewer = normalised_by("batch", WORKED_BLOCK[:2])
        more = normalised_by("batch", WORKED_BLOCK)

        assert fewer["normalised"][0] != more["normalised"][0]
        assert [round(value, 7) for value in fewer["normalised"][0]] == [0.0, -0.999995]
        assert [round(value, 7) for value in more["normalised"][0]] == [
            -0.7777777,
            0.3333331,
        ]

    def test_a_constant_feature_normalises_to_zero(self) -> None:
        body = normalised_by("batch", [[5, 1], [5, 2], [5, 6]])

        assert [row[0] for row in body["normalised"]] == [0.0, 0.0, 0.0]


class TestTheCrowdUnderTheBatchLayer:
    def test_twenty_people_standardise_by_their_own_means(self) -> None:
        body = normalised_by("batch", TWENTY_OF_THE_CROWD)

        height, weight = body["statistics"]
        assert height["mean"] == pytest.approx(151.9)
        assert weight["mean"] == pytest.approx(51.85)
        assert round(height["deviation"], 4) == 17.2827
        assert round(weight["deviation"], 4) == 15.7298
        assert [round(value, 4) for value in body["normalised"][0]] == [
            -0.2835,
            -0.6898,
        ]

    def test_one_step_leaves_the_running_figures_far_from_the_crowd(self) -> None:
        body = normalised_by("batch", TWENTY_OF_THE_CROWD)

        height, weight = body["predicting_statistics"]
        assert height["mean"] == pytest.approx(15.19)
        assert weight["mean"] == pytest.approx(5.185)
        assert round(height["deviation"], 4) == 5.547
        assert round(weight["deviation"], 4) == 5.0639


class TestTheRowIsTheWrongAxisForTwoRawFeatures:
    def test_every_person_layer_normalises_to_plus_and_minus_one(self) -> None:
        body = normalised_by("layer", TWENTY_OF_THE_CROWD)

        widest = max(
            abs(abs(value) - 1.0) for row in body["normalised"] for value in row
        )
        assert widest < 3e-9
        assert all(row[0] > 0 > row[1] for row in body["normalised"])
        assert body["statistics"][0]["mean"] == 94.0
        assert round(body["statistics"][0]["deviation"], 7) == 53.0000001

    def test_rms_keeps_the_rows_apart(self) -> None:
        body = normalised_by("rms", TWENTY_OF_THE_CROWD)

        assert [round(value, 4) for value in body["normalised"][0]] == [1.3622, 0.3799]
        assert [round(value, 4) for value in body["normalised"][1]] == [1.339, 0.4549]


class TestTheBackwardPassThroughTheBatchLayer:
    def test_the_arriving_block_counts_up_by_default(self) -> None:
        body = backward_through("batch", WORKED_BLOCK)

        assert body["arriving"] == [[1, 2], [3, 4], [5, 6], [7, 8]]

    def test_the_passed_down_block_agrees_with_the_finite_difference(self) -> None:
        body = backward_through("batch", WORKED_BLOCK)

        assert body["largest_gap"] < 1e-9
        assert round(body["largest_gap"], 12) == 2.08e-10
        assert round(body["largest_true_slope"], 4) == 1.0741
        assert [[round(value, 4) for value in row] for row in body["passed_down"]] == [
            [-0.1605, -0.9259],
            [0.0617, -0.1111],
            [0.1358, -0.037],
            [-0.037, 1.0741],
        ]

    def test_the_naive_form_misses_by_twice_the_largest_slope(self) -> None:
        body = backward_through("batch", WORKED_BLOCK)

        assert round(body["largest_naive_gap"], 4) == 2.037
        assert [[round(value, 4) for value in row] for row in body["naive"]] == [
            [0.1111, 0.6667],
            [0.3333, 1.3333],
            [0.5556, 2.0],
            [0.7778, 2.6667],
        ]

    def test_the_true_columns_sum_to_zero_and_the_naive_ones_do_not(self) -> None:
        body = backward_through("batch", WORKED_BLOCK)

        assert body["largest_band_sum"] < 1e-15
        assert [round(value, 4) for value in body["naive_band_sums"]] == [
            1.7778,
            6.6667,
        ]
        assert round(body["largest_naive_band_sum"], 4) == 6.6667

    def test_the_parameter_slopes_are_sums_down_the_rows(self) -> None:
        body = backward_through("batch", WORKED_BLOCK)

        assert [round(value, 4) for value in body["scale_slope"]] == [8.0, -2.6667]
        assert body["shift_slope"] == [16.0, 20.0]
        assert [round(value, 7) for value in body["deviations"]] == [
            9.0000006,
            3.0000017,
        ]

    def test_a_supplied_arriving_block_is_used(self) -> None:
        body = backward_through(
            "batch", WORKED_BLOCK, arriving=[[1, 1], [1, 1], [1, 1], [1, 1]]
        )

        # A slope of one everywhere pulls every row alike, and the mean route
        # cancels the direct route exactly.
        assert body["largest_true_slope"] < 1e-9
        assert body["largest_gap"] < 1e-9

    def test_an_arriving_block_of_the_wrong_shape_is_refused(self) -> None:
        response = client.post(
            BACKWARD,
            json={"rows": WORKED_BLOCK, "layer": "batch", "arriving": [[1, 2], [3, 4]]},
        )

        assert response.status_code == 422


class TestTheBackwardPassAlongTheRow:
    def test_two_features_leave_nothing_to_pass_down(self) -> None:
        body = backward_through("layer", WORKED_BLOCK)

        assert body["largest_true_slope"] < 1e-6
        assert body["largest_gap"] < 1e-9
        assert round(body["largest_naive_gap"], 4) == 2.0

    def test_three_features_agree_with_the_finite_difference(self) -> None:
        body = backward_through("layer", THREE_FEATURES)

        assert body["largest_gap"] < 1e-8
        assert round(body["largest_gap"], 11) == 1.54e-09
        assert round(body["largest_true_slope"], 4) == 0.5123
        assert round(body["largest_naive_gap"], 4) == 5.3404
        assert body["largest_band_sum"] < 1e-15
        assert round(body["largest_naive_band_sum"], 4) == 14.1203
        assert [[round(value, 4) for value in row] for row in body["passed_down"]][
            1
        ] == [
            -0.5123,
            0.2049,
            0.3074,
        ]

    def test_rms_has_two_routes_and_its_rows_do_not_sum_to_zero(self) -> None:
        body = backward_through("rms", WORKED_BLOCK)

        assert body["largest_gap"] < 1e-9
        assert round(body["largest_true_slope"], 4) == 1.036
        assert round(body["largest_naive_gap"], 4) == 1.148
        assert round(body["largest_band_sum"], 4) == 0.888
        assert [round(value, 4) for value in body["band_sums"]] == [
            0.12,
            0.3504,
            0.888,
            0.2198,
        ]
        assert [[round(value, 4) for value in row] for row in body["passed_down"]] == [
            [0.14, -0.02],
            [0.3942, -0.0438],
            [-0.148, 1.036],
            [-0.0962, 0.316],
        ]


class TestTheRunningFigures:
    def test_the_figures_head_for_the_batch(self) -> None:
        response = client.post(RUNNING, json={"rows": WORKED_BLOCK, "n_steps": 200})
        assert response.status_code == 200
        body = response.json()

        assert body["momentum"] == 0.9
        assert body["batch_mean"] == [8.0, 6.0]
        assert [round(value, 7) for value in body["batch_deviation"]] == [
            9.0000006,
            3.0000017,
        ]
        first = body["steps"][0]
        assert first["running_mean"] == pytest.approx([0.8, 0.6])
        assert round(first["largest_gap"], 4) == 5.7333
        last = body["steps"][-1]
        assert last["running_mean"] == pytest.approx([8.0, 6.0], abs=1e-8)
        assert last["largest_gap"] < 1e-8

    def test_the_gap_falls_below_a_hundredth_at_step_fifty_three(self) -> None:
        body = client.post(RUNNING, json={"rows": WORKED_BLOCK, "n_steps": 200}).json()

        assert body["first_step_within_hundredth"] == 53
        assert body["first_step_within_thousandth"] == 75
        gaps = {step["step"]: round(step["largest_gap"], 4) for step in body["steps"]}
        assert gaps[10] == 1.0433
        assert gaps[20] == 0.3163
        assert gaps[44] == 0.0238

    def test_too_few_steps_report_no_arrival(self) -> None:
        body = client.post(RUNNING, json={"rows": WORKED_BLOCK, "n_steps": 10}).json()

        assert len(body["steps"]) == 10
        assert body["first_step_within_hundredth"] is None

    def test_a_lone_row_cannot_be_batch_normalised_while_training(self) -> None:
        body = client.post(RUNNING, json={"rows": WORKED_BLOCK, "n_steps": 200}).json()
        lone = body["lone_row"]

        assert lone["row"] == [1.0, 7.0]
        assert lone["training_answer"] == [0.0, 0.0]
        assert [round(value, 6) for value in lone["predicting_answer_fresh"]] == [
            0.999995,
            6.999965,
        ]
        assert [round(value, 7) for value in lone["predicting_answer_final"]] == [
            -0.7777777,
            0.3333331,
        ]
        assert [round(value, 7) for value in lone["layer_answer"]] == [
            -0.9999994,
            0.9999994,
        ]
        assert [round(value, 7) for value in lone["rms_answer"]] == [0.2, 1.3999997]

    def test_too_many_steps_are_refused(self) -> None:
        response = client.post(RUNNING, json={"rows": WORKED_BLOCK, "n_steps": 301})

        assert response.status_code == 422


class TestTheDriftOfTheHiddenBlock:
    def test_the_plain_network_moves_what_its_output_layer_reads(
        self, crowd: dict
    ) -> None:
        plain = crowd["drift"][0]

        assert plain["arrangement"] == "none"
        assert round(plain["largest_mean_shift"], 4) == 1.0897
        assert round(plain["largest_deviation_change"], 4) == 2.2777
        assert round(plain["largest_single_epoch_mean_move"], 4) == 0.1375
        assert round(plain["largest_single_epoch_deviation_move"], 4) == 0.2128
        assert plain["mean_matches_shift_gap"] is None
        assert len(plain["units"]) == 16
        assert plain["recorded_epochs"][:3] == [0, 4, 8]
        assert len(plain["recorded_epochs"]) == 101

    def test_the_third_unit_travels_furthest(self, crowd: dict) -> None:
        plain = crowd["drift"][0]
        third = plain["units"][2]

        assert round(third["means"][0], 3) == 0.439
        assert round(third["means"][-1], 3) == 1.529
        assert round(third["deviations"][0], 3) == 0.819
        assert round(third["deviations"][-1], 3) == 3.097

    def test_the_batch_layer_makes_the_mean_the_shift_and_still_moves_it(
        self, crowd: dict
    ) -> None:
        batch = crowd["drift"][1]

        assert batch["arrangement"] == "batch"
        assert batch["mean_matches_shift_gap"] < 1e-14
        # Before any step the scale is one and the shift zero, so every unit
        # the output layer reads has mean zero and a deviation of
        # sqrt(v / (v + epsilon)), one to within the epsilon.
        assert all(round(unit["means"][0], 12) == 0.0 for unit in batch["units"])
        assert all(round(unit["deviations"][0], 2) == 1.0 for unit in batch["units"])
        assert round(batch["largest_mean_shift"], 4) == 1.3204
        assert round(batch["largest_deviation_change"], 4) == 3.6403
        assert round(batch["largest_single_epoch_mean_move"], 4) == 0.3113


class TestTheFiveArrangementsHeldOut:
    def test_the_arrangements_are_reported_in_order(self, crowd: dict) -> None:
        assert [reading["arrangement"] for reading in crowd["sweep"]] == [
            "none",
            "batch",
            "layer",
            "rms",
            "weight",
        ]
        assert crowd["hidden_width"] == 16
        assert crowd["epochs"] == 400
        assert crowd["learning_rate"] == 1.0
        assert crowd["n_folds"] == 5
        assert crowd["seeds"] == [1, 2, 3, 4, 5]

    def test_the_plain_network_matches_the_dropout_page(self, crowd: dict) -> None:
        plain = crowd["sweep"][0]

        assert round(plain["training_accuracy_mean"], 3) == 0.992
        assert round(plain["held_out_accuracy_mean"], 3) == 0.744
        assert plain["held_out_accuracy_low"] == 0.64
        assert plain["held_out_accuracy_high"] == 0.84

    def test_the_batch_layer_lowered_the_held_out_score(self, crowd: dict) -> None:
        batch = crowd["sweep"][1]

        assert round(batch["training_accuracy_mean"], 3) == 0.904
        assert round(batch["held_out_accuracy_mean"], 3) == 0.64
        assert batch["held_out_accuracy_low"] == 0.6
        assert batch["held_out_accuracy_high"] == 0.68
        assert round(batch["training_loss_mean"], 4) == 0.2088

    def test_the_row_layers_and_weight_normalisation(self, crowd: dict) -> None:
        layer, rms, weight = crowd["sweep"][2:]

        assert round(layer["training_accuracy_mean"], 3) == 0.91
        assert round(layer["held_out_accuracy_mean"], 3) == 0.696
        assert round(rms["training_accuracy_mean"], 3) == 0.87
        assert round(rms["held_out_accuracy_mean"], 3) == 0.72
        assert round(weight["training_accuracy_mean"], 3) == 0.986
        assert round(weight["held_out_accuracy_mean"], 3) == 0.752
        assert (weight["held_out_accuracy_low"], weight["held_out_accuracy_high"]) == (
            0.68,
            0.8,
        )
        assert [
            round(reading["training_loss_mean"], 4) for reading in crowd["sweep"]
        ] == [
            0.0829,
            0.2088,
            0.1604,
            0.2392,
            0.0775,
        ]

    def test_every_seed_holds_every_person_out_once(self, crowd: dict) -> None:
        for reading in crowd["sweep"]:
            assert [seed["seed"] for seed in reading["per_seed"]] == [1, 2, 3, 4, 5]
            for seed in reading["per_seed"]:
                assert round(seed["held_out_accuracy"] * 25) == pytest.approx(
                    seed["held_out_accuracy"] * 25
                )


class TestTheLossCurves:
    def test_the_row_layers_reach_the_threshold_first(self, crowd: dict) -> None:
        firsts = {
            curve["arrangement"]: curve["first_epoch_below_threshold"]
            for curve in crowd["curves"]
        }

        assert crowd["loss_threshold"] == 0.3
        assert firsts == {
            "none": 77,
            "batch": 119,
            "layer": 16,
            "rms": 16,
            "weight": 77,
        }

    def test_where_each_curve_finished(self, crowd: dict) -> None:
        finals = {
            curve["arrangement"]: (
                round(curve["final_loss"], 4),
                curve["final_accuracy"],
            )
            for curve in crowd["curves"]
        }

        assert finals == {
            "none": (0.0813, 1.0),
            "batch": (0.1921, 0.92),
            "layer": (0.1041, 0.96),
            "rms": (0.4284, 0.76),
            "weight": (0.0753, 1.0),
        }

    def test_the_plain_and_weight_normalised_networks_start_identical(
        self, crowd: dict
    ) -> None:
        plain, weight = crowd["curves"][0], crowd["curves"][4]

        assert plain["readings"][0]["loss"] == weight["readings"][0]["loss"]
        assert round(plain["readings"][0]["loss"], 4) == 0.9863

    def test_only_the_batch_layer_answers_differently_by_purpose(
        self, crowd: dict
    ) -> None:
        for curve in crowd["curves"]:
            differing = [
                reading["epoch"]
                for reading in curve["readings"]
                if reading["predicting_accuracy"] != reading["training_accuracy"]
            ]
            if curve["arrangement"] == "batch":
                assert len(differing) == 88
                tenth = curve["readings"][9]
                assert (tenth["predicting_accuracy"], tenth["training_accuracy"]) == (
                    0.76,
                    0.72,
                )
            else:
                assert differing == []


class TestTheReparameterisation:
    def test_the_forward_pass_is_a_dense_layers(self, crowd: dict) -> None:
        check = crowd["reparameterisation"]

        assert check["output_gap"] == 0.0
        assert check["magnitudes"] == [1.0, 2.5, 0.5]
        assert len(check["people"]) == 5

    def test_the_gradient_agrees_with_the_finite_difference(self, crowd: dict) -> None:
        check = crowd["reparameterisation"]

        assert check["direction_gap"] < 1e-9
        assert check["magnitude_gap"] < 1e-9
        assert round(check["unprojected_gap"], 4) == 1.8286
        assert round(check["largest_direction_slope"], 4) == 2.9973
        assert check["radial_component"] < 1e-14
        assert round(check["unprojected_radial_component"], 4) == 3.6914
        assert [round(value, 4) for value in check["magnitude_slopes"]] == [
            -2.2858,
            1.4766,
            1.9604,
        ]

    def test_the_same_slope_lands_somewhere_else(self, crowd: dict) -> None:
        check = crowd["reparameterisation"]

        assert check["rate"] == 0.1
        assert round(check["destination_gap"], 4) == 0.0364
        assert [round(value, 4) for value in check["lengths_after"]] == [
            1.2286,
            2.3523,
            0.304,
        ]
        assert [round(value, 4) for value in check["dense_lengths_after"]] == [
            1.2347,
            2.382,
            0.3247,
        ]


class TestRefusals:
    def test_a_ragged_block_is_refused_in_words(self) -> None:
        response = client.post(
            NORMALISE, json={"rows": [[1, 7], [1, 9, 4], [7, 1]], "layer": "batch"}
        )

        assert response.status_code == 422
        assert "rectangle" in response.json()["detail"]
        assert "row 2" in response.json()["detail"]

    def test_a_ragged_block_is_refused_by_every_endpoint(self) -> None:
        ragged = [[1, 7], [1, 9, 4], [7, 1]]

        assert (
            client.post(BACKWARD, json={"rows": ragged, "layer": "rms"}).status_code
            == 422
        )
        assert client.post(RUNNING, json={"rows": ragged}).status_code == 422

    def test_an_unknown_layer_is_refused(self) -> None:
        response = client.post(NORMALISE, json={"rows": WORKED_BLOCK, "layer": "group"})

        assert response.status_code == 422

    def test_a_lone_row_is_refused(self) -> None:
        response = client.post(NORMALISE, json={"rows": [[1, 7]], "layer": "batch"})

        assert response.status_code == 422

    def test_a_row_of_one_value_is_refused(self) -> None:
        response = client.post(NORMALISE, json={"rows": [[1], [2]], "layer": "layer"})

        assert response.status_code == 422

    def test_too_many_features_are_refused(self) -> None:
        wide = [[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1]]
        response = client.post(NORMALISE, json={"rows": wide, "layer": "batch"})

        assert response.status_code == 422

    def test_too_many_rows_are_refused(self) -> None:
        tall = [[float(index), 1.0] for index in range(21)]
        response = client.post(NORMALISE, json={"rows": tall, "layer": "batch"})

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            NORMALISE, json={"rows": WORKED_BLOCK, "layer": "batch", "momentum": 0.5}
        )

        assert response.status_code == 422

    def test_a_non_finite_value_is_refused(self) -> None:
        response = client.post(
            NORMALISE, json={"rows": [[1, "nan"], [2, 3]], "layer": "batch"}
        )

        assert response.status_code == 422
