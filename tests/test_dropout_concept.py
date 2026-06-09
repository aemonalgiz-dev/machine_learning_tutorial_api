"""The dropout endpoints, pinned to the page's hand arithmetic and its measurements.

The page works four units holding 1, 2, 3 and 4 at a drop probability of one
half under seed 10. The first draw keeps the first and third units and
silences the other two, so with the survivors doubled the row reads 2, 0, 6,
0, while the same row passed while predicting reads 1, 2, 3, 4 untouched.
Over four hundred draws the units survive 200, 192, 190 and 204 times, which
makes their mean outputs 1.00, 1.92, 2.85 and 4.08, none further than 0.15
from the predicting answer, with the running means passing through 1.0, 2.4,
2.4, 4.8 after ten draws and 1.08, 2.04, 2.58, 3.76 after a hundred. The
page also puts a zero into a surviving unit to show the mask still says
kept.

The block endpoint pushes that row three times over through one draw, and
the three rows are thinned differently, with 2, 3 and 2 survivors. On a
rectified block with exact zeros the mask read off the outputs misfiles four
of twelve entries. The experiment on the tangled crowd is pinned to the
numbers the page quotes: training and held-out accuracy at each rate, the
unit ablations, the kept zeros, the purpose confusion and the averaging.
"""

from itertools import pairwise

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

APPLY = "/concepts/dropout/apply"
BLOCK = "/concepts/dropout/block"
EXPERIMENT = "/concepts/dropout/experiment"

WORKED_EXAMPLE = {
    "values": [1.0, 2.0, 3.0, 4.0],
    "drop_probability": 0.5,
    "random_seed": 10,
    "n_draws": 400,
}

WORKED_BLOCK = {
    "rows": [[1.0, 2.0, 3.0, 4.0]] * 3,
    "drop_probability": 0.5,
    "random_seed": 10,
}

RECTIFIED_BLOCK = {
    "rows": [[0.0, 2.0, 0.0, 4.0], [3.0, 0.0, 0.0, 1.0], [0.0, 0.0, 5.0, 0.0]],
    "drop_probability": 0.5,
    "random_seed": 10,
    "arriving": [[1.0, 2.0, 3.0, 4.0]] * 3,
}


def worked_example(**changes: object) -> dict:
    response = client.post(APPLY, json={**WORKED_EXAMPLE, **changes})
    assert response.status_code == 200
    return response.json()


def block(payload: dict, **changes: object) -> dict:
    response = client.post(BLOCK, json={**payload, **changes})
    assert response.status_code == 200
    return response.json()


@pytest.fixture(scope="module")
def experiment() -> dict:
    response = client.get(EXPERIMENT)
    assert response.status_code == 200
    return response.json()


class TestTheOneDraw:
    def test_the_scale_is_two_at_one_half(self) -> None:
        assert worked_example()["scale"] == 2.0

    def test_the_first_draw_keeps_the_first_and_third_units(self) -> None:
        body = worked_example()

        assert body["kept"] == [True, False, True, False]
        assert body["mask"] == [2.0, 0.0, 2.0, 0.0]

    def test_the_survivors_are_doubled_and_the_rest_are_zero(self) -> None:
        assert worked_example()["training_output"] == [2.0, 0.0, 6.0, 0.0]

    def test_predicting_passes_the_row_through_untouched(self) -> None:
        assert worked_example()["predicting_output"] == [1.0, 2.0, 3.0, 4.0]

    def test_the_same_seed_draws_the_same_sequence(self) -> None:
        assert worked_example() == worked_example()

    def test_the_seed_and_draw_count_default_to_the_worked_example(self) -> None:
        response = client.post(
            APPLY, json={"values": [1.0, 2.0, 3.0, 4.0], "drop_probability": 0.5}
        )

        assert response.status_code == 200
        assert response.json() == worked_example()

    def test_a_different_seed_is_a_different_draw(self) -> None:
        # Seed 3 keeps the third and fourth units instead.
        body = worked_example(random_seed=3)

        assert body["kept"] == [False, False, True, True]
        assert body["training_output"] == [0.0, 0.0, 6.0, 8.0]

    def test_a_quarter_scales_the_survivors_by_four_thirds(self) -> None:
        body = worked_example(drop_probability=0.25)

        assert abs(body["scale"] - 4.0 / 3.0) < 1e-12
        assert body["kept"] == [True, False, True, False]
        assert abs(body["training_output"][0] - 4.0 / 3.0) < 1e-12
        assert body["training_output"][2] == 4.0


class TestTheAverage:
    def test_the_units_survive_a_pinned_number_of_times(self) -> None:
        body = worked_example()

        assert body["n_draws"] == 400
        assert body["times_kept"] == [200, 192, 190, 204]

    def test_the_mean_over_four_hundred_draws(self) -> None:
        body = worked_example()

        assert [round(value, 6) for value in body["mean_output"]] == [
            1.0,
            1.92,
            2.85,
            4.08,
        ]

    def test_the_mean_is_the_survival_share_times_the_doubled_value(self) -> None:
        # The derivation's expectation with a count in place of a probability.
        body = worked_example()

        for unit_index, (count, value) in enumerate(
            zip(body["times_kept"], WORKED_EXAMPLE["values"], strict=True)
        ):
            expected = count / 400 * value * 2.0
            assert abs(body["mean_output"][unit_index] - expected) < 1e-12

    def test_the_largest_gap_is_fifteen_hundredths(self) -> None:
        assert round(worked_example()["largest_gap"], 6) == 0.15

    def test_a_single_draw_ranges_from_nothing_to_the_doubled_value(self) -> None:
        # The spread the average hides: over four hundred draws every unit
        # was silenced at least once and doubled at least once.
        body = worked_example()

        assert body["single_draw_low"] == [0.0, 0.0, 0.0, 0.0]
        assert body["single_draw_high"] == [2.0, 4.0, 6.0, 8.0]

    def test_the_running_mean_runs_from_the_first_draw_to_the_mean(self) -> None:
        body = worked_example()

        assert len(body["running_means"]) == 400
        assert body["running_means"][0] == body["training_output"]
        assert body["running_means"][-1] == body["mean_output"]

    def test_the_running_mean_after_ten_and_after_a_hundred_draws(self) -> None:
        body = worked_example()

        after_ten = [round(value, 6) for value in body["running_means"][9]]
        after_a_hundred = [round(value, 6) for value in body["running_means"][99]]
        assert after_ten == [1.0, 2.4, 2.4, 4.8]
        assert after_a_hundred == [1.08, 2.04, 2.58, 3.76]

    def test_a_shorter_run_is_a_prefix_of_a_longer_one(self) -> None:
        # One seeded layer whose generator advances between passes, so the
        # first ten draws of a long run are the ten draws of a short one.
        short = worked_example(n_draws=10)
        long = worked_example()

        assert short["running_means"] == long["running_means"][:10]
        assert short["training_output"] == long["training_output"]
        assert short["mask"] == long["mask"]


class TestTheMaskIsADraw:
    def test_a_zero_that_survived_is_reported_as_kept(self) -> None:
        # The first unit now holds zero. Its output is zero either way, and the
        # mask is the only thing that can say it survived, and it does.
        body = worked_example(values=[0.0, 2.0, 3.0, 4.0])

        assert body["training_output"] == [0.0, 0.0, 6.0, 0.0]
        assert body["kept"] == [True, False, True, False]
        assert body["mask"] == [2.0, 0.0, 2.0, 0.0]

    def test_the_survival_count_is_read_from_the_mask_not_the_output(self) -> None:
        body = worked_example(values=[0.0, 2.0, 3.0, 4.0])

        assert body["times_kept"] == [200, 192, 190, 204]
        assert body["mean_output"][0] == 0.0

    def test_no_dropping_at_all_is_the_identity(self) -> None:
        body = worked_example(drop_probability=0.0, n_draws=5)

        assert body["scale"] == 1.0
        assert body["kept"] == [True, True, True, True]
        assert body["training_output"] == body["predicting_output"]
        assert body["times_kept"] == [5, 5, 5, 5]
        assert body["largest_gap"] == 0.0


class TestTheBlock:
    def test_three_copies_of_one_row_are_thinned_three_ways(self) -> None:
        body = block(WORKED_BLOCK)

        assert body["kept"] == [
            [True, False, True, False],
            [True, False, True, True],
            [False, True, True, False],
        ]
        assert body["n_kept_per_row"] == [2, 3, 2]
        assert body["training_outputs"] == [
            [2.0, 0.0, 6.0, 0.0],
            [2.0, 0.0, 6.0, 8.0],
            [0.0, 4.0, 6.0, 0.0],
        ]

    def test_the_first_row_of_the_block_is_the_worked_example_row(self) -> None:
        # One layer, one seed, and the first row draws first, so the block's
        # first row is the single-row endpoint's first draw.
        assert (
            block(WORKED_BLOCK)["training_outputs"][0]
            == worked_example()["training_output"]
        )

    def test_predicting_leaves_every_row_untouched(self) -> None:
        assert block(WORKED_BLOCK)["predicting_outputs"] == WORKED_BLOCK["rows"]

    def test_the_blame_passed_down_is_the_arriving_block_through_the_mask(self) -> None:
        # Arriving ones by default, so what comes back is the scaled mask.
        body = block(WORKED_BLOCK)

        assert body["arriving"] == [[1.0, 1.0, 1.0, 1.0]] * 3
        assert body["passed_down"] == body["mask"]
        assert body["passed_down"] == [
            [2.0, 0.0, 2.0, 0.0],
            [2.0, 0.0, 2.0, 2.0],
            [0.0, 2.0, 2.0, 0.0],
        ]

    def test_the_layer_has_nothing_to_learn(self) -> None:
        assert block(WORKED_BLOCK)["gradient_is_none"] is True

    def test_the_backward_step_agrees_with_a_finite_difference(self) -> None:
        body = block(WORKED_BLOCK)

        assert body["largest_finite_difference_gap"] < 1e-7
        assert body["largest_finite_difference_gap"] > 0.0

    def test_a_block_with_no_zeros_misfiles_nothing(self) -> None:
        assert block(WORKED_BLOCK)["n_misfiled"] == 0

    def test_a_rectified_block_misfiles_every_kept_zero(self) -> None:
        body = block(RECTIFIED_BLOCK)

        assert body["kept"] == [
            [True, False, True, False],
            [True, False, True, True],
            [False, True, True, False],
        ]
        assert body["mask_read_from_outputs"] == [
            [False, False, False, False],
            [True, False, False, True],
            [False, False, True, False],
        ]
        assert body["misfiled"] == [
            [True, False, True, False],
            [False, False, True, False],
            [False, True, False, False],
        ]
        assert body["n_misfiled"] == 4

    def test_a_kept_zero_is_still_blamed_in_full(self) -> None:
        # The first row's first unit held zero and survived, so its output is
        # zero and its blame is the arriving 1 times the scale of 2.
        body = block(RECTIFIED_BLOCK)

        assert body["training_outputs"][0] == [0.0, 0.0, 0.0, 0.0]
        assert body["passed_down"][0] == [2.0, 0.0, 6.0, 0.0]
        assert body["passed_down"] == [
            [2.0, 0.0, 6.0, 0.0],
            [2.0, 0.0, 6.0, 8.0],
            [0.0, 4.0, 6.0, 0.0],
        ]

    def test_another_seed_thins_the_rows_another_way(self) -> None:
        body = block(WORKED_BLOCK, random_seed=11)

        assert body["kept"] == [
            [False, False, True, False],
            [False, True, False, False],
            [True, True, False, True],
        ]


class TestTheExperiment:
    def test_the_settings_are_the_ones_the_page_quotes(self, experiment: dict) -> None:
        assert experiment["hidden_width"] == 16
        assert experiment["epochs"] == 400
        assert experiment["learning_rate"] == 1.0
        assert experiment["n_folds"] == 5
        assert experiment["seeds"] == [1, 2, 3, 4, 5]
        assert len(experiment["crowd"]) == 25

    def test_the_undropped_network_memorises_its_training_rows(
        self, experiment: dict
    ) -> None:
        sweep = {
            reading["drop_probability"]: reading for reading in experiment["sweep"]
        }

        assert round(sweep[0.0]["training_accuracy_mean"], 3) == 0.992
        assert round(sweep[0.0]["held_out_accuracy_mean"], 3) == 0.744

    def test_dropout_narrows_the_gap_mostly_from_the_training_side(
        self, experiment: dict
    ) -> None:
        sweep = {
            reading["drop_probability"]: reading for reading in experiment["sweep"]
        }

        assert round(sweep[0.2]["training_accuracy_mean"], 3) == 0.950
        assert round(sweep[0.2]["held_out_accuracy_mean"], 3) == 0.752
        assert round(sweep[0.5]["training_accuracy_mean"], 3) == 0.886
        assert round(sweep[0.5]["held_out_accuracy_mean"], 3) == 0.728

    def test_the_seed_spread_swallows_the_difference_between_rates(
        self, experiment: dict
    ) -> None:
        # The honest reading: the held-out means differ by hundredths where
        # the seeds differ by tenths.
        sweep = {
            reading["drop_probability"]: reading for reading in experiment["sweep"]
        }

        assert sweep[0.0]["held_out_accuracy_low"] == 0.64
        assert sweep[0.0]["held_out_accuracy_high"] == 0.84
        assert sweep[0.2]["held_out_accuracy_low"] == 0.68
        assert sweep[0.2]["held_out_accuracy_high"] == 0.80
        assert sweep[0.5]["held_out_accuracy_low"] == 0.64
        assert sweep[0.5]["held_out_accuracy_high"] == 0.80

    def test_the_per_seed_held_out_accuracies_are_pinned(
        self, experiment: dict
    ) -> None:
        sweep = {
            reading["drop_probability"]: reading for reading in experiment["sweep"]
        }

        def held(rate: float) -> list[float]:
            return [reading["held_out_accuracy"] for reading in sweep[rate]["per_seed"]]

        assert held(0.0) == [0.84, 0.64, 0.72, 0.72, 0.80]
        assert held(0.2) == [0.80, 0.68, 0.76, 0.80, 0.72]
        assert held(0.5) == [0.72, 0.64, 0.80, 0.76, 0.72]

    def test_the_curves_hold_a_hundred_readings_each(self, experiment: dict) -> None:
        for curve in experiment["curves"]:
            assert len(curve["readings"]) == 100
            assert curve["readings"][0]["epoch"] == 4
            assert curve["readings"][-1]["epoch"] == 400

    def test_the_curves_end_where_the_page_says(self, experiment: dict) -> None:
        curves = {
            curve["drop_probability"]: curve["readings"][-1]
            for curve in experiment["curves"]
        }

        assert round(curves[0.0]["training_accuracy"], 3) == 0.990
        assert round(curves[0.0]["held_out_accuracy"], 3) == 0.840
        assert round(curves[0.2]["training_accuracy"], 3) == 0.970
        assert round(curves[0.2]["held_out_accuracy"], 3) == 0.800
        assert round(curves[0.5]["training_accuracy"], 3) == 0.890
        assert round(curves[0.5]["held_out_accuracy"], 3) == 0.720

    def test_the_thinned_loss_rises_and_falls_under_dropout_and_hardly_without(
        self, experiment: dict
    ) -> None:
        rises = {}
        for curve in experiment["curves"]:
            losses = [reading["thinned_loss"] for reading in curve["readings"]]
            rises[curve["drop_probability"]] = sum(
                1 for before, after in pairwise(losses) if after > before
            )

        assert rises[0.0] == 1
        assert rises[0.2] == 40
        assert rises[0.5] == 48

    def test_the_undropped_network_leans_on_two_units(self, experiment: dict) -> None:
        ablation = experiment["ablations"][0]

        assert ablation["drop_probability"] == 0.0
        assert ablation["intact_accuracy"] == 1.0
        assert ablation["n_dead"] == 0
        assert round(ablation["mean_drop"], 3) == 0.040
        assert round(ablation["largest_drop"], 3) == 0.240
        assert ablation["silenced"][11] == 0.84
        assert ablation["silenced"][12] == 0.76

    def test_the_dropped_network_leans_on_no_single_unit(
        self, experiment: dict
    ) -> None:
        ablation = experiment["ablations"][1]

        assert ablation["drop_probability"] == 0.5
        assert ablation["intact_accuracy"] == 0.84
        assert ablation["n_dead"] == 0
        assert ablation["silenced"] == [0.84] * 16
        assert ablation["mean_drop"] == 0.0
        assert ablation["largest_drop"] == 0.0

    def test_a_rectified_layer_hands_dropout_many_kept_zeros(
        self, experiment: dict
    ) -> None:
        kept_zeros = experiment["kept_zeros"]

        assert kept_zeros["n_readings"] == 400
        assert kept_zeros["n_zero"] == 282
        assert kept_zeros["n_zero_and_kept"] == 142

    def test_predicting_while_still_training_moves_one_person_and_one_chance(
        self, experiment: dict
    ) -> None:
        confusion = experiment["confusion"]

        assert confusion["predicting_accuracy"] == 0.84
        assert confusion["training_purpose_low"] == 0.80
        assert confusion["training_purpose_high"] == 0.84
        assert len(confusion["training_purpose_accuracies"]) == 20
        assert confusion["training_purpose_accuracies"].count(0.80) == 5
        assert confusion["training_purpose_accuracies"].count(0.84) == 15
        assert round(confusion["swing_predicting_chance"], 3) == 0.996
        assert round(confusion["swing_low_chance"], 3) == 0.292
        assert round(confusion["swing_high_chance"], 3) == 1.0

    def test_an_idle_dropout_layer_is_no_dropout_layer_to_the_last_bit(
        self, experiment: dict
    ) -> None:
        assert experiment["confusion"]["identity_control_gap"] == 0.0

    def test_the_average_settles_where_a_single_draw_does_not(
        self, experiment: dict
    ) -> None:
        averaging = experiment["averaging"]

        assert averaging["n_draws"] == 400
        assert round(averaging["largest_score_gap"], 2) == 1.39
        assert round(averaging["largest_single_draw_gap"], 1) == 62.0
        assert round(averaging["largest_predicting_score"], 1) == 61.3
        assert round(averaging["largest_chance_gap"], 3) == 0.077


class TestRefusals:
    def test_a_probability_of_one_is_refused_in_the_library_words(self) -> None:
        response = client.post(APPLY, json={**WORKED_EXAMPLE, "drop_probability": 1.0})

        assert response.status_code == 422
        assert "silences the whole layer" in response.json()["detail"]

    def test_an_empty_row_is_refused(self) -> None:
        response = client.post(APPLY, json={**WORKED_EXAMPLE, "values": []})

        assert response.status_code == 422

    def test_a_row_wider_than_the_widget_allows_is_refused(self) -> None:
        response = client.post(APPLY, json={**WORKED_EXAMPLE, "values": [1.0] * 13})

        assert response.status_code == 422

    def test_too_many_draws_are_refused(self) -> None:
        response = client.post(APPLY, json={**WORKED_EXAMPLE, "n_draws": 1001})

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(APPLY, json={**WORKED_EXAMPLE, "keep_probability": 0.5})

        assert response.status_code == 422

    def test_a_block_probability_of_one_is_refused_in_the_library_words(self) -> None:
        response = client.post(BLOCK, json={**WORKED_BLOCK, "drop_probability": 1.0})

        assert response.status_code == 422
        assert "silences the whole layer" in response.json()["detail"]

    def test_a_ragged_block_is_refused(self) -> None:
        response = client.post(
            BLOCK, json={"rows": [[1.0, 2.0], [1.0, 2.0, 3.0]], "drop_probability": 0.5}
        )

        assert response.status_code == 422

    def test_an_arriving_block_of_another_shape_is_refused(self) -> None:
        response = client.post(
            BLOCK,
            json={
                "rows": [[1.0, 2.0, 3.0]],
                "drop_probability": 0.5,
                "arriving": [[1.0, 2.0]],
            },
        )

        assert response.status_code == 422

    def test_an_empty_row_in_a_block_is_refused(self) -> None:
        response = client.post(BLOCK, json={"rows": [[]], "drop_probability": 0.5})

        assert response.status_code == 422

    def test_a_block_too_wide_or_too_tall_is_refused(self) -> None:
        wide = client.post(BLOCK, json={"rows": [[1.0] * 13], "drop_probability": 0.5})
        tall = client.post(
            BLOCK, json={"rows": [[1.0, 2.0]] * 7, "drop_probability": 0.5}
        )

        assert wide.status_code == 422
        assert tall.status_code == 422
