"""The loop endpoints, pinned to every number the training page quotes.

The page's own module already pins one recorded run. What is pinned here is
the loop around that run: what one step changes, what the step size buys and
costs, what a batch is and what a small one does to the direction, what the
training loss hides, what a seed fixes, and what the library refuses.

Two of these are worth stating up front, because they are the measurements the
page leans on hardest and both came out other than the textbook would have it.

On this chain and this start, log-loss never produces a number that is not a
number, however large the step. At a rate of a hundred the walk thrashes, its
loss rising on 77 of the 199 consecutive readings and finishing at 12.19,
having touched 0.55 on the way; nothing ever goes non-finite and nothing is
ever refused. Squared error on the same chain does run away, and it is the
objective the divergence rows use for that reason.

And the wall-clock cost of a step barely depends on how many rows it reads. A
step on one row and a step on all sixty are both dominated by the Python and
allocation around the arithmetic, so the run with sixty times as many steps
takes tens of times as long rather than the same time. The timings are
machine-dependent and are asserted only as an ordering.
"""

import math
from itertools import pairwise

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/training-a-network"
STEP = f"{BASE}/step"
RATE_SWEEP = f"{BASE}/rate-sweep"
BATCH_SWEEP = f"{BASE}/batch-sweep"
GENERALISATION = f"{BASE}/generalisation"
REPEATABILITY = f"{BASE}/repeatability"
CONTRACTS = f"{BASE}/loop-contracts"

WORKED_STEP = {
    "preset": "rings",
    "hidden_width": 6,
    "learning_rate": 0.5,
    "weight_seed": 1,
}

WORKED_RATES = {
    "preset": "rings",
    "hidden_width": 6,
    "weight_seed": 1,
    "max_epochs": 200,
    "objective": "log_loss",
    "learning_rates": [0.001, 0.01, 0.1, 0.5, 2.0, 5.0, 20.0, 100.0],
}

WORKED_BATCHES = {
    "preset": "rings",
    "hidden_width": 6,
    "weight_seed": 1,
    "learning_rate": 0.5,
    "max_epochs": 60,
    "order_seed": 3,
    "batch_sizes": [1, 5, 15, 30, 60],
}


def posted(path: str, body: dict) -> dict:
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def got(path: str) -> dict:
    response = client.get(path)
    assert response.status_code == 200, response.text
    return response.json()


class TestOneStep:
    def test_the_step_lowers_the_loss_by_the_amount_the_page_quotes(self) -> None:
        body = posted(STEP, WORKED_STEP)

        assert body["loss_before"] == pytest.approx(0.727078, abs=5e-6)
        assert body["loss_after"] == pytest.approx(0.717923, abs=5e-6)
        assert body["actual_drop"] == pytest.approx(0.009155, abs=5e-6)

    def test_the_chain_holds_twenty_five_numbers_over_sixty_rows(self) -> None:
        body = posted(STEP, WORKED_STEP)

        assert body["n_parameters"] == 25
        assert body["n_rows"] == 60
        assert sum(row["n_weights"] + row["n_biases"] for row in body["layers"]) == 25

    def test_the_rate_is_not_a_distance_and_the_response_says_what_is(self) -> None:
        # The rate multiplies the gradient, so what the step covers in the
        # twenty-five numbers is the rate times the gradient's length.
        body = posted(STEP, WORKED_STEP)

        assert body["gradient_norm"] == pytest.approx(0.139819, abs=5e-6)
        assert body["distance_moved"] == pytest.approx(
            body["learning_rate"] * body["gradient_norm"]
        )
        assert body["distance_moved"] == pytest.approx(0.069910, abs=5e-6)

    def test_the_slope_predicts_a_drop_and_the_walk_falls_short_of_it(self) -> None:
        body = posted(STEP, WORKED_STEP)

        assert body["predicted_drop"] == pytest.approx(0.009775, abs=5e-6)
        assert body["ratio"] == pytest.approx(0.936569, abs=5e-6)
        assert body["actual_drop"] < body["predicted_drop"]

    def test_the_prediction_is_exact_in_the_limit_and_poor_far_from_it(self) -> None:
        # This is the whole argument for a small step. The slope describes the
        # loss only near where it was read, so the ratio approaches one as the
        # step shrinks and falls apart as it grows.
        body = posted(STEP, WORKED_STEP)
        ratios = {row["learning_rate"]: row["ratio"] for row in body["first_order"]}

        assert ratios[0.001] == pytest.approx(0.999873, abs=5e-6)
        assert ratios[0.05] == pytest.approx(0.993644, abs=5e-6)
        assert ratios[1.0] == pytest.approx(0.873483, abs=5e-6)
        assert ratios[5.0] == pytest.approx(0.393958, abs=5e-6)

    def test_the_ladder_of_ratios_falls_all_the_way_down(self) -> None:
        body = posted(STEP, WORKED_STEP)
        ratios = [row["ratio"] for row in body["first_order"]]

        assert ratios == sorted(ratios, reverse=True)

    def test_every_rate_on_the_ladder_still_lowers_the_loss(self) -> None:
        # Falling short of the predicted drop is not the same as going uphill,
        # and on this chain even a rate of five is downhill.
        body = posted(STEP, WORKED_STEP)

        assert all(row["actual_drop"] > 0.0 for row in body["first_order"])

    def test_three_parameters_are_traced_from_before_to_after(self) -> None:
        body = posted(STEP, WORKED_STEP)
        traced = body["traced"]

        assert len(traced) == 3
        for parameter in traced:
            moved = parameter["before"] - 0.5 * parameter["slope"]
            assert parameter["after"] == pytest.approx(moved, abs=1e-12)

    def test_the_traced_hidden_weight_is_the_one_the_page_walks_through(self) -> None:
        body = posted(STEP, WORKED_STEP)
        first = body["traced"][0]

        assert first["before"] == pytest.approx(0.345584, abs=5e-6)
        assert first["slope"] == pytest.approx(0.028119, abs=5e-6)
        assert first["after"] == pytest.approx(0.331525, abs=5e-6)

    def test_a_bias_starting_at_zero_moves_against_a_negative_slope(self) -> None:
        body = posted(STEP, WORKED_STEP)
        bias = body["traced"][1]

        assert bias["before"] == 0.0
        assert bias["slope"] == pytest.approx(-0.038580, abs=5e-6)
        assert bias["after"] == pytest.approx(0.019290, abs=5e-6)

    def test_the_largest_slope_in_the_chain_is_in_the_output_layer(self) -> None:
        body = posted(STEP, WORKED_STEP)
        hidden, output = body["layers"]

        assert hidden["n_weights"] == 12 and hidden["n_biases"] == 6
        assert output["n_weights"] == 6 and output["n_biases"] == 1
        assert output["largest_weight_slope"] == pytest.approx(0.084723, abs=5e-6)
        assert body["largest_movement"] == output["largest_weight_slope"]


class TestTheStepSize:
    def test_a_step_far_too_small_makes_almost_no_progress(self) -> None:
        body = posted(RATE_SWEEP, WORKED_RATES)
        tiny = body["rows"][0]

        assert tiny["learning_rate"] == 0.001
        assert tiny["final_loss"] == pytest.approx(0.723377, abs=5e-6)
        assert body["starting_loss"] - tiny["final_loss"] == pytest.approx(
            0.003701, abs=5e-6
        )
        assert tiny["final_accuracy"] == pytest.approx(0.466667, abs=5e-6)

    def test_the_useful_range_is_wide_and_the_page_quotes_all_of_it(self) -> None:
        body = posted(RATE_SWEEP, WORKED_RATES)
        finals = {row["learning_rate"]: row["final_loss"] for row in body["rows"]}

        assert finals[0.01] == pytest.approx(0.701984, abs=5e-6)
        assert finals[0.1] == pytest.approx(0.623762, abs=5e-6)
        assert finals[0.5] == pytest.approx(0.135431, abs=5e-6)
        assert finals[2.0] == pytest.approx(0.024188, abs=5e-6)
        assert finals[5.0] == pytest.approx(0.008412, abs=5e-6)

    def test_a_step_far_too_large_thrashes_rather_than_settling(self) -> None:
        body = posted(RATE_SWEEP, WORKED_RATES)
        wild = {row["learning_rate"]: row for row in body["rows"]}[20.0]
        losses = wild["losses"]
        rises = sum(1 for before, after in pairwise(losses) if after > before)

        assert wild["final_loss"] == pytest.approx(13.216754, abs=5e-6)
        assert wild["best_loss"] == pytest.approx(0.591843, abs=5e-6)
        assert wild["best_epoch"] == 127
        assert rises == 100

    def test_the_thrashing_rate_finishes_worse_than_it_started(self) -> None:
        body = posted(RATE_SWEEP, WORKED_RATES)
        wild = {row["learning_rate"]: row for row in body["rows"]}[100.0]

        assert wild["final_loss"] > body["starting_loss"]
        assert wild["final_loss"] == pytest.approx(12.186892, abs=5e-6)
        assert wild["best_loss"] == pytest.approx(0.550848, abs=5e-6)

    def test_log_loss_never_stops_being_a_number_however_large_the_step(self) -> None:
        # Worth pinning because it is the textbook claim that did not hold
        # here. The page says so rather than showing a divergence log-loss
        # does not produce.
        body = posted(RATE_SWEEP, WORKED_RATES)

        assert all(row["first_non_finite_epoch"] is None for row in body["rows"])
        assert all(row["refused_at_epoch"] is None for row in body["rows"])
        assert all(row["epochs_run"] == 200 for row in body["rows"])

    def test_squared_error_on_the_same_chain_does_run_away(self) -> None:
        body = posted(
            RATE_SWEEP,
            {**WORKED_RATES, "objective": "squared_error", "learning_rates": [2.0]},
        )
        runaway = body["rows"][0]

        assert body["starting_loss"] == pytest.approx(0.350820, abs=5e-6)
        assert runaway["first_non_finite_epoch"] == 155
        assert runaway["refused_at_epoch"] == 157
        assert runaway["refusal"] is not None
        assert "finite" in runaway["refusal"]

    def test_a_larger_step_runs_away_sooner(self) -> None:
        body = posted(
            RATE_SWEEP,
            {
                **WORKED_RATES,
                "objective": "squared_error",
                "learning_rates": [2.0, 5.0, 20.0],
            },
        )
        epochs = [row["first_non_finite_epoch"] for row in body["rows"]]

        assert epochs == [155, 111, 77]

    def test_squared_error_at_a_smaller_step_is_perfectly_well_behaved(self) -> None:
        body = posted(
            RATE_SWEEP,
            {**WORKED_RATES, "objective": "squared_error", "learning_rates": [0.5]},
        )
        settled = body["rows"][0]

        assert settled["first_non_finite_epoch"] is None
        assert settled["final_loss"] == pytest.approx(0.004273, abs=5e-6)
        assert settled["final_accuracy"] == 1.0

    def test_the_non_finite_losses_travel_as_nulls(self) -> None:
        # JSON has no infinity a browser will parse, so a diverged epoch is a
        # gap in the curve rather than a token the widget chokes on.
        body = posted(
            RATE_SWEEP,
            {**WORKED_RATES, "objective": "squared_error", "learning_rates": [20.0]},
        )
        losses = body["rows"][0]["losses"]

        assert losses[75] is not None
        assert losses[-1] is None
        assert all(value is None or math.isfinite(value) for value in losses)

    def test_every_rate_walks_from_the_identical_start(self) -> None:
        body = posted(RATE_SWEEP, WORKED_RATES)

        assert all(
            row["losses"][0] == pytest.approx(body["starting_loss"])
            for row in body["rows"]
        )


class TestTheBatch:
    def test_the_batch_size_decides_how_many_steps_an_epoch_takes(self) -> None:
        body = posted(BATCH_SWEEP, WORKED_BATCHES)
        steps = {row["batch_size"]: row["steps_per_epoch"] for row in body["rows"]}

        assert body["n_rows"] == 60
        assert steps == {1: 60, 5: 12, 15: 4, 30: 2, 60: 1}

    def test_sixty_epochs_of_each_reach_very_different_places(self) -> None:
        body = posted(BATCH_SWEEP, WORKED_BATCHES)
        finals = {row["batch_size"]: row["final_loss"] for row in body["rows"]}

        assert finals[1] == pytest.approx(0.003141, abs=5e-6)
        assert finals[5] == pytest.approx(0.026718, abs=5e-6)
        assert finals[15] == pytest.approx(0.100265, abs=5e-6)
        assert finals[30] == pytest.approx(0.291703, abs=5e-6)
        assert finals[60] == pytest.approx(0.536594, abs=5e-6)

    def test_the_smallest_batch_gets_there_and_the_largest_does_not(self) -> None:
        body = posted(BATCH_SWEEP, WORKED_BATCHES)
        accuracies = {row["batch_size"]: row["final_accuracy"] for row in body["rows"]}

        assert accuracies[1] == 1.0
        assert accuracies[30] == pytest.approx(0.95)
        assert accuracies[60] == pytest.approx(0.766667, abs=5e-6)

    def test_the_total_steps_are_the_epochs_times_the_steps_in_one(self) -> None:
        body = posted(BATCH_SWEEP, WORKED_BATCHES)

        for row in body["rows"]:
            assert row["total_steps"] == 60 * row["steps_per_epoch"]

    def test_a_whole_batch_agrees_with_itself_exactly(self) -> None:
        # The direction a batch of every row points is the whole set's
        # direction, so the cosine between them is one to the last bit.
        body = posted(BATCH_SWEEP, WORKED_BATCHES)
        whole = {row["batch_size"]: row for row in body["agreement"]}[60]

        assert whole["mean_cosine"] == pytest.approx(1.0, abs=1e-12)
        assert whole["worst_cosine"] == pytest.approx(1.0, abs=1e-12)

    def test_a_small_batch_points_almost_anywhere(self) -> None:
        body = posted(BATCH_SWEEP, WORKED_BATCHES)
        means = {row["batch_size"]: row["mean_cosine"] for row in body["agreement"]}

        assert means[1] == pytest.approx(0.073017, abs=5e-6)
        assert means[5] == pytest.approx(0.331999, abs=5e-6)
        assert means[15] == pytest.approx(0.545584, abs=5e-6)
        assert means[30] == pytest.approx(0.779614, abs=5e-6)

    def test_the_agreement_rises_with_the_batch(self) -> None:
        body = posted(BATCH_SWEEP, WORKED_BATCHES)
        means = [row["mean_cosine"] for row in body["agreement"]]

        assert means == sorted(means)

    def test_a_single_row_can_point_the_wrong_way_entirely(self) -> None:
        body = posted(BATCH_SWEEP, WORKED_BATCHES)
        worst = {row["batch_size"]: row["worst_cosine"] for row in body["agreement"]}

        assert worst[1] == pytest.approx(-0.785021, abs=5e-6)
        assert worst[30] < 0.0
        assert worst[60] > 0.0

    def test_a_step_on_one_row_costs_nearly_what_a_step_on_sixty_costs(self) -> None:
        # An ordering rather than a figure, since the timing is the machine's.
        # The point the page makes is that sixty times the steps costs tens of
        # times the time, not one times it.
        body = posted(BATCH_SWEEP, WORKED_BATCHES)
        timed = {row["batch_size"]: row for row in body["rows"]}

        assert timed[1]["milliseconds"] > 5.0 * timed[60]["milliseconds"]
        assert timed[1]["milliseconds_per_step"] < timed[60]["milliseconds_per_step"]
        assert (
            timed[1]["milliseconds_per_step"]
            > 0.25 * (timed[60]["milliseconds_per_step"])
        )

    def test_every_curve_is_measured_on_the_whole_set(self) -> None:
        # Otherwise the rows would not be comparable, since a batch of one
        # measured on its own row says nothing about the other fifty-nine.
        body = posted(BATCH_SWEEP, WORKED_BATCHES)

        assert all(
            row["losses"][0] == pytest.approx(body["starting_loss"])
            for row in body["rows"]
        )
        assert all(len(row["losses"]) == 60 for row in body["rows"])

    def test_the_smallest_batch_is_the_only_curve_that_rises(self) -> None:
        body = posted(BATCH_SWEEP, WORKED_BATCHES)
        rises = {
            row["batch_size"]: sum(
                1 for before, after in pairwise(row["losses"]) if after > before
            )
            for row in body["rows"]
        }

        assert rises[1] == 5
        assert rises[5] == 0
        assert rises[60] == 0


class TestTheTwoCurves:
    def test_the_split_is_stratified_and_small_enough_to_memorise(self) -> None:
        body = posted(GENERALISATION, {})

        assert body["n_training"] == 14
        assert body["n_held_out"] == 46
        assert body["n_mislabelled"] == 3
        assert body["n_parameters"] == 33

    def test_the_training_loss_keeps_falling_long_after_the_other_turns(self) -> None:
        body = posted(GENERALISATION, {})

        assert body["best_held_out_loss_epoch"] == 73
        assert body["best_held_out_loss"] == pytest.approx(0.535429, abs=5e-6)
        assert body["final_training_loss"] == pytest.approx(0.075675, abs=5e-6)
        assert body["final_held_out_loss"] == pytest.approx(1.647182, abs=5e-6)

    def test_the_training_loss_falls_on_every_single_epoch_after_the_turn(self) -> None:
        # The claim the page makes is that the loss a loop can see says nothing
        # about the loss it cannot, and this is that claim as a count.
        body = posted(GENERALISATION, {})
        after = [
            reading
            for reading in body["readings"]
            if reading["epoch"] >= body["best_held_out_loss_epoch"]
        ]
        training_rises = sum(
            1
            for before, later in pairwise(after)
            if later["training_loss"] > before["training_loss"]
        )

        assert training_rises == 0
        assert after[-1]["held_out_loss"] > after[0]["held_out_loss"]

    def test_the_training_loss_falls_on_every_epoch_of_the_whole_run(self) -> None:
        body = posted(GENERALISATION, {})
        rises = sum(
            1
            for before, later in pairwise(body["readings"])
            if later["training_loss"] > before["training_loss"]
        )

        assert rises == 0

    def test_the_held_out_loss_rises_on_every_epoch_after_its_low(self) -> None:
        body = posted(GENERALISATION, {})
        after = [
            reading
            for reading in body["readings"]
            if reading["epoch"] >= body["best_held_out_loss_epoch"]
        ]
        falls = sum(
            1
            for before, later in pairwise(after)
            if later["held_out_loss"] <= before["held_out_loss"]
        )

        assert falls == 0
        assert len(after) == 528

    def test_the_clean_run_climbs_too_and_by_almost_nothing(self) -> None:
        # The honest correction to the textbook picture. The curve does rise
        # after its low, on every epoch, by a hundredth over three hundred and
        # forty-four of them.
        body = posted(GENERALISATION, {"mislabelled": 0})
        after = [
            reading
            for reading in body["readings"]
            if reading["epoch"] >= body["best_held_out_loss_epoch"]
        ]
        rises = sum(
            1
            for before, later in pairwise(after)
            if later["held_out_loss"] > before["held_out_loss"]
        )

        assert rises == len(after) - 1 == 344
        assert body["final_held_out_loss"] - body[
            "best_held_out_loss"
        ] == pytest.approx(0.0113, abs=5e-5)

    def test_the_network_learns_the_training_rows_outright(self) -> None:
        body = posted(GENERALISATION, {})

        assert body["final_training_accuracy"] == 1.0
        assert body["final_held_out_accuracy"] == pytest.approx(0.630435, abs=5e-6)
        assert body["best_held_out_accuracy"] == pytest.approx(0.760870, abs=5e-6)
        assert body["best_held_out_accuracy_epoch"] == 51

    def test_the_two_scores_disagree_about_where_to_stop(self) -> None:
        # The lowest held-out loss and the highest held-out accuracy are not
        # the same epoch, which is why the page reports both.
        body = posted(GENERALISATION, {})

        assert body["best_held_out_loss_epoch"] != body["best_held_out_accuracy_epoch"]

    def test_without_flipped_labels_the_held_out_score_merely_stops(self) -> None:
        # This is the honest version of the textbook curve. With clean labels
        # the held-out loss barely climbs at all, and the held-out accuracy
        # reached at epoch 131 is the same accuracy it still has at 600.
        body = posted(GENERALISATION, {"mislabelled": 0})

        assert body["best_held_out_loss"] == pytest.approx(0.246944, abs=5e-6)
        assert body["best_held_out_loss_epoch"] == 256
        assert body["final_held_out_loss"] == pytest.approx(0.258240, abs=5e-6)
        assert body["best_held_out_accuracy"] == pytest.approx(0.913043, abs=5e-6)
        assert body["final_held_out_accuracy"] == pytest.approx(0.913043, abs=5e-6)

    def test_the_training_loss_alone_cannot_tell_the_two_cases_apart(self) -> None:
        clean = posted(GENERALISATION, {"mislabelled": 0})
        noisy = posted(GENERALISATION, {})

        assert clean["final_training_loss"] < 0.1
        assert noisy["final_training_loss"] < 0.1
        assert noisy["final_held_out_loss"] > 6.0 * clean["final_held_out_loss"]

    def test_a_reading_is_recorded_before_the_first_step(self) -> None:
        body = posted(GENERALISATION, {"max_epochs": 5})
        epochs = [reading["epoch"] for reading in body["readings"]]

        assert epochs == [0, 1, 2, 3, 4, 5]


class TestWhatASeedFixes:
    def test_the_identical_request_lands_in_the_identical_place(self) -> None:
        body = got(REPEATABILITY)
        same = body["runs"][0]

        assert same["identical"] is True
        assert same["first_final_loss"] == pytest.approx(0.059454, abs=5e-6)

    def test_a_different_order_of_rows_changes_a_batched_walk(self) -> None:
        body = got(REPEATABILITY)
        reordered = body["runs"][1]

        assert reordered["identical"] is False
        assert reordered["first_final_loss"] == pytest.approx(0.059454, abs=5e-6)
        assert reordered["second_final_loss"] == pytest.approx(0.059036, abs=5e-6)

    def test_a_different_order_changes_nothing_when_the_batch_is_everybody(
        self,
    ) -> None:
        # Shuffling a set you are about to sum over in full is a no-op, so the
        # order seed only matters once the epoch is cut into pieces.
        body = got(REPEATABILITY)
        whole = body["runs"][2]

        assert whole["identical"] is True
        assert whole["first_final_loss"] == pytest.approx(0.536594, abs=5e-6)

    def test_four_starting_draws_reach_four_different_places(self) -> None:
        body = got(REPEATABILITY)
        finals = {start["weight_seed"]: start["final_loss"] for start in body["starts"]}

        assert finals[1] == pytest.approx(0.134309, abs=5e-6)
        assert finals[2] == pytest.approx(0.107094, abs=5e-6)
        assert finals[3] == pytest.approx(0.120911, abs=5e-6)
        assert finals[4] == pytest.approx(0.155376, abs=5e-6)

    def test_one_of_those_four_starts_does_not_get_everybody_right(self) -> None:
        body = got(REPEATABILITY)
        accuracies = {
            start["weight_seed"]: start["final_accuracy"] for start in body["starts"]
        }

        assert accuracies[1] == 1.0
        assert accuracies[4] == pytest.approx(0.966667, abs=5e-6)

    def test_a_layer_that_draws_keeps_drawing_across_the_steps(self) -> None:
        body = got(REPEATABILITY)
        kept = body["draws"][0]

        assert kept["all_alike"] is False
        assert len(kept["sheets"]) == 4
        assert kept["sheets"][0] != kept["sheets"][1]

    def test_rebuilding_that_layer_from_its_seed_repeats_one_draw(self) -> None:
        body = got(REPEATABILITY)
        rebuilt = body["draws"][1]

        assert rebuilt["all_alike"] is True
        assert all(sheet == rebuilt["sheets"][0] for sheet in rebuilt["sheets"])

    def test_stepping_such_a_layer_answers_with_the_layer_itself(self) -> None:
        # Which is why the first sequence differs and the second does not. A
        # rebuilt layer rebuilds its generator with it.
        body = got(REPEATABILITY)

        assert body["step_returns_the_same_layer"] is True


class TestTheFailureContracts:
    def test_every_row_is_probed_when_the_request_arrives(self) -> None:
        body = got(CONTRACTS)

        assert len(body["rows"]) == 14
        assert all(row["detail"] for row in body["rows"])

    @pytest.mark.parametrize(
        ("situation", "verdict"),
        [
            ("a step size of zero", "accepted"),
            ("a negative step size", "accepted"),
            ("zero epochs", "accepted"),
            ("a batch larger than the training set", "accepted"),
            ("a batch of one row", "accepted"),
            ("a batch of no rows", "refused"),
            ("a chain answering one number, targets two wide", "refused"),
            ("targets of the wrong length", "refused"),
            ("a value that is not a number in the rows", "refused"),
            ("an infinite value in the rows", "refused"),
            ("a walk that runs away, squared error at a rate of two", "refused"),
            ("two layers whose widths disagree", "refused"),
            ("asking the chain before it has been trained", "accepted"),
        ],
    )
    def test_the_verdict_the_page_prints(self, situation: str, verdict: str) -> None:
        body = got(CONTRACTS)
        found = {row["situation"]: row["verdict"] for row in body["rows"]}

        assert found[situation] == verdict

    def test_a_step_of_zero_leaves_the_loss_where_it_was(self) -> None:
        body = got(CONTRACTS)
        found = {row["situation"]: row["detail"] for row in body["rows"]}

        assert "unchanged" in found["a step size of zero"]

    def test_a_negative_step_walks_uphill(self) -> None:
        body = got(CONTRACTS)
        found = {row["situation"]: row["detail"] for row in body["rows"]}

        assert "higher" in found["a negative step size"]

    def test_asking_for_more_rows_than_exist_hands_back_what_there_is(self) -> None:
        body = got(CONTRACTS)
        found = {row["situation"]: row["detail"] for row in body["rows"]}

        assert "hands back 60" in found["a batch larger than the training set"]

    def test_the_runaway_walk_reports_the_epochs_the_page_quotes(self) -> None:
        body = got(CONTRACTS)
        found = {row["situation"]: row["detail"] for row in body["rows"]}
        watched = found["that same runaway walk, watched epoch by epoch"]

        assert "epoch 155" in watched
        assert "epoch 157" in watched

    def test_every_refusal_is_in_the_librarys_own_words(self) -> None:
        body = got(CONTRACTS)
        refused = [row for row in body["rows"] if row["verdict"] == "refused"]

        assert len(refused) == 7
        assert all(
            row["detail"].startswith(
                ("EmptyValuesError", "ShapeMismatchError", "InvalidValuesError")
            )
            for row in refused
        )


class TestTheRequestsRefuseWhatTheyShould:
    @pytest.mark.parametrize(
        ("path", "body"),
        [
            (STEP, {"learning_rate": 0.0}),
            (STEP, {"learning_rate": -0.5}),
            (STEP, {"hidden_width": 1}),
            (STEP, {"momentum": 0.9}),
            (RATE_SWEEP, {"learning_rates": [0.0]}),
            (RATE_SWEEP, {"learning_rates": [101.0]}),
            (RATE_SWEEP, {"learning_rates": []}),
            (RATE_SWEEP, {"learning_rates": [1, 2, 3, 4, 5, 6, 7, 8, 9]}),
            (RATE_SWEEP, {"objective": "hinge"}),
            (RATE_SWEEP, {"max_epochs": 0}),
            (BATCH_SWEEP, {"batch_sizes": [0]}),
            (BATCH_SWEEP, {"batch_sizes": [61]}),
            (BATCH_SWEEP, {"batch_sizes": []}),
            (BATCH_SWEEP, {"max_epochs": 121}),
            (GENERALISATION, {"n_training": 3}),
            (GENERALISATION, {"mislabelled": 7}),
            (GENERALISATION, {"max_epochs": 801}),
        ],
    )
    def test_a_request_outside_the_bounds_is_refused(
        self, path: str, body: dict
    ) -> None:
        assert client.post(path, json=body).status_code == 422

    def test_flipping_more_than_a_third_of_a_small_split_is_refused(self) -> None:
        # Three flips of six rows is not a memorisation fixture, it is a
        # differently labelled problem, so the request says so.
        refused = client.post(GENERALISATION, json={"n_training": 6, "mislabelled": 3})
        allowed = client.post(GENERALISATION, json={"n_training": 9, "mislabelled": 3})

        assert refused.status_code == 422
        assert allowed.status_code == 200

    def test_a_batch_of_one_is_permitted_and_a_batch_of_none_is_not(self) -> None:
        allowed = client.post(BATCH_SWEEP, json={"batch_sizes": [1], "max_epochs": 2})

        assert allowed.status_code == 200
        assert client.post(BATCH_SWEEP, json={"batch_sizes": [0]}).status_code == 422
