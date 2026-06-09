"""The loss-functions endpoints, pinned to the page's hand arithmetic.

The page's slider example sets the regression truth to one half, the class
label to yes, and the raw output to zero. Squared error then costs one eighth
and pulls by minus one half; absolute error costs one half and pulls by minus
one; Huber at its default threshold is inside the knee and agrees with squared
error; both cross-entropies apply a squash that lands on one half and cost the
natural logarithm of two, pulling by minus one half. Three of the five pull by
exactly the same amount, which is the page's thesis and is designed rather
than accidental. At a raw output of three the same three part company.

The batch example is the measured four under a line of slope 0.6 through
their mean point, whose misses are minus four, four, eight and minus eight
kilograms, so every batch number is whole or a clean fraction. The descent
example is the fifteen people of the ideal case with and without one weight
mistyped, 72.5 kilograms entered as 27.5. All pinned here.
"""

import math

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

MEASURE = "/concepts/loss-functions/measure"
CURVE = "/concepts/loss-functions/curve"
REGRESSION_BATCH = "/concepts/loss-functions/regression-batch"
CLASSIFICATION_BATCH = "/concepts/loss-functions/classification-batch"
SOFTMAX_ROW = "/concepts/loss-functions/softmax-row"
DESCEND_LINE = "/concepts/loss-functions/descend-line"

WORKED_SETTING = {"target": 0.5, "label": 1, "huber_threshold": 1.0}

THE_FIVE = (
    "squared_error",
    "absolute_error",
    "huber_error",
    "binary_cross_entropy",
    "softmax_cross_entropy",
)

# The measured four, and the weights a slope-0.6 line through their mean guesses.
MEASURED_FOUR_ROWS = [
    {"prediction": 74.0, "truth": 78.0},
    {"prediction": 62.0, "truth": 58.0},
    {"prediction": 71.0, "truth": 63.0},
    {"prediction": 65.0, "truth": 73.0},
]

IDEAL_PEOPLE = [
    {"x": 152, "y": 54.2},
    {"x": 155, "y": 55.4},
    {"x": 158, "y": 58.7},
    {"x": 161, "y": 60.5},
    {"x": 164, "y": 63.2},
    {"x": 167, "y": 66.2},
    {"x": 170, "y": 67.4},
    {"x": 173, "y": 70.7},
    {"x": 176, "y": 72.5},
    {"x": 179, "y": 75.2},
    {"x": 182, "y": 78.2},
    {"x": 185, "y": 79.4},
    {"x": 188, "y": 82.7},
    {"x": 191, "y": 84.5},
    {"x": 194, "y": 87.2},
]
MISTYPED_PERSON = {"x": 176, "y": 27.5}

# Sixteen people scored by (height - 152) / 5, children below 152 centimetres.
CHILD_HEIGHTS = [112, 118, 124, 130, 135, 140, 144, 148]
ADULT_HEIGHTS = [156, 160, 165, 170, 174, 178, 183, 188]


def scored_crowd(scale: float = 5.0) -> list[dict]:
    return [
        {"score": (height - 152) / scale, "label": 0} for height in CHILD_HEIGHTS
    ] + [{"score": (height - 152) / scale, "label": 1} for height in ADULT_HEIGHTS]


def readings_at(raw_output: float) -> dict:
    return client.post(
        MEASURE, json={**WORKED_SETTING, "raw_output": raw_output}
    ).json()["readings"]


class TestTheWorkedExampleAtRawOutputZero:
    def test_squared_error_costs_an_eighth_and_pulls_by_minus_a_half(self) -> None:
        reading = readings_at(0.0)["squared_error"]
        assert reading["value"] == pytest.approx(0.125)
        assert reading["gradient"] == pytest.approx(-0.5)
        assert reading["probability"] is None

    def test_absolute_error_costs_a_half_and_pulls_by_minus_one(self) -> None:
        reading = readings_at(0.0)["absolute_error"]
        assert reading["value"] == pytest.approx(0.5)
        assert reading["gradient"] == pytest.approx(-1.0)

    def test_huber_inside_its_knee_agrees_with_squared_error(self) -> None:
        readings = readings_at(0.0)
        assert readings["huber_error"]["value"] == pytest.approx(0.125)
        assert readings["huber_error"]["gradient"] == pytest.approx(-0.5)

    def test_binary_cross_entropy_costs_log_two_and_pulls_by_minus_a_half(self) -> None:
        reading = readings_at(0.0)["binary_cross_entropy"]
        assert reading["probability"] == pytest.approx(0.5)
        assert reading["value"] == pytest.approx(math.log(2.0))
        assert reading["gradient"] == pytest.approx(-0.5)

    def test_softmax_cross_entropy_agrees_with_binary_at_the_first_logit(self) -> None:
        reading = readings_at(0.0)["softmax_cross_entropy"]
        assert reading["probability"] == pytest.approx(0.5)
        assert reading["value"] == pytest.approx(math.log(2.0))
        assert reading["gradient"] == pytest.approx(-0.5)

    def test_three_of_the_five_pull_by_exactly_the_same_amount(self) -> None:
        readings = readings_at(0.0)
        pulls = {
            readings[name]["gradient"]
            for name in (
                "squared_error",
                "binary_cross_entropy",
                "softmax_cross_entropy",
            )
        }
        assert len(pulls) == 1

    def test_a_sigmoid_output_scored_by_squared_error_pulls_a_quarter_as_hard(
        self,
    ) -> None:
        reading = readings_at(0.0)["sigmoid_then_squared"]
        assert reading["probability"] == pytest.approx(0.5)
        assert reading["value"] == pytest.approx(0.125)
        assert reading["gradient"] == pytest.approx(-0.125)


class TestTheWorkedExampleAtRawOutputThree:
    def test_squared_error_costs_three_and_an_eighth(self) -> None:
        reading = readings_at(3.0)["squared_error"]
        assert reading["value"] == pytest.approx(3.125)
        assert reading["gradient"] == pytest.approx(2.5)

    def test_absolute_error_pulls_by_one_however_far(self) -> None:
        reading = readings_at(3.0)["absolute_error"]
        assert reading["value"] == pytest.approx(2.5)
        assert reading["gradient"] == pytest.approx(1.0)

    def test_huber_outside_its_knee_is_capped_at_the_threshold(self) -> None:
        reading = readings_at(3.0)["huber_error"]
        assert reading["value"] == pytest.approx(2.0)
        assert reading["gradient"] == pytest.approx(1.0)

    def test_binary_cross_entropy_has_nearly_nothing_left_to_say(self) -> None:
        reading = readings_at(3.0)["binary_cross_entropy"]
        probability = 1.0 / (1.0 + math.exp(-3.0))
        assert reading["probability"] == pytest.approx(probability)
        assert reading["value"] == pytest.approx(-math.log(probability))
        assert reading["gradient"] == pytest.approx(probability - 1.0)
        assert reading["value"] == pytest.approx(0.048587, abs=1e-6)
        assert reading["gradient"] == pytest.approx(-0.047426, abs=1e-6)

    def test_the_three_that_agreed_at_zero_part_company_here(self) -> None:
        readings = readings_at(3.0)
        assert readings["squared_error"]["gradient"] == pytest.approx(2.5)
        assert readings["binary_cross_entropy"]["gradient"] == pytest.approx(
            -0.047425873, abs=1e-6
        )


class TestTheConfidentlyWrongSide:
    """At minus three and minus five the label is still yes."""

    def test_cross_entropy_pulls_by_nearly_one_where_squared_error_pulls_by_a_half(
        self,
    ) -> None:
        readings = readings_at(-3.0)
        assert readings["binary_cross_entropy"]["value"] == pytest.approx(
            3.048587, abs=1e-6
        )
        assert readings["binary_cross_entropy"]["gradient"] == pytest.approx(
            -0.952574, abs=1e-6
        )
        assert readings["squared_error"]["gradient"] == pytest.approx(-3.5)

    def test_the_mismatched_pairing_barely_pulls_when_confidently_wrong(self) -> None:
        readings = readings_at(-5.0)
        mismatched = readings["sigmoid_then_squared"]
        canonical = readings["binary_cross_entropy"]
        assert mismatched["probability"] == pytest.approx(0.006693, abs=1e-6)
        assert mismatched["value"] == pytest.approx(0.493330, abs=1e-6)
        assert mismatched["gradient"] == pytest.approx(-0.006604, abs=1e-6)
        assert canonical["value"] == pytest.approx(5.006715, abs=1e-6)
        assert canonical["gradient"] == pytest.approx(-0.993307, abs=1e-6)
        assert abs(canonical["gradient"]) > 150 * abs(mismatched["gradient"])


class TestTheSlopeByNudging:
    """The gradient the library reports is the slope of its own value."""

    @pytest.mark.parametrize("raw_output", [-5.0, -3.0, 0.0, 3.0, 5.0])
    def test_every_reading_agrees_with_its_nudge_to_a_part_in_ten_billion(
        self, raw_output: float
    ) -> None:
        body = client.post(
            MEASURE, json={**WORKED_SETTING, "raw_output": raw_output}
        ).json()
        assert body["nudge"] == pytest.approx(1e-5)
        for name in (*THE_FIVE, "sigmoid_then_squared"):
            reading = body["readings"][name]
            assert reading["disagreement"] < 1e-9
            assert reading["slope_by_nudging"] == pytest.approx(
                reading["gradient"], abs=1e-9
            )

    def test_absolute_error_at_an_exact_tie_commits_to_zero(self) -> None:
        reading = readings_at(0.5)["absolute_error"]
        assert reading["value"] == 0.0
        assert reading["gradient"] == 0.0
        assert reading["slope_by_nudging"] == pytest.approx(0.0, abs=1e-9)

    def test_huber_exactly_at_the_knee_meets_from_both_sides(self) -> None:
        readings = readings_at(1.5)
        huber = readings["huber_error"]
        squared = readings["squared_error"]
        assert huber["value"] == pytest.approx(0.5)
        assert huber["gradient"] == pytest.approx(1.0)
        assert squared["value"] == pytest.approx(0.5)
        # The nudge straddles the knee, so the two sides have different
        # curvature and the check disagrees at the width of the nudge itself.
        assert huber["disagreement"] < 1e-5
        assert huber["disagreement"] > squared["disagreement"]

    def test_the_curve_carries_no_nudge(self) -> None:
        body = client.post(CURVE, json={**WORKED_SETTING, "n_samples": 21}).json()
        assert "slope_by_nudging" not in body["squared_error"][0]


class TestTheSharedForm:
    """Each canonical loss pulls by its own prediction minus its own truth."""

    @pytest.mark.parametrize("raw_output", [-3.0, -1.0, 0.0, 0.7, 2.0, 4.5])
    def test_squared_error_pulls_by_prediction_minus_truth(
        self, raw_output: float
    ) -> None:
        reading = readings_at(raw_output)["squared_error"]
        assert reading["gradient"] == pytest.approx(raw_output - 0.5)

    @pytest.mark.parametrize("raw_output", [-3.0, -1.0, 0.0, 0.7, 2.0, 4.5])
    def test_binary_cross_entropy_pulls_by_probability_minus_label(
        self, raw_output: float
    ) -> None:
        reading = readings_at(raw_output)["binary_cross_entropy"]
        assert reading["gradient"] == pytest.approx(reading["probability"] - 1.0)

    @pytest.mark.parametrize("raw_output", [-3.0, -1.0, 0.0, 0.7, 2.0, 4.5])
    def test_a_two_class_softmax_is_the_sigmoid(self, raw_output: float) -> None:
        readings = readings_at(raw_output)
        binary = readings["binary_cross_entropy"]
        softmax = readings["softmax_cross_entropy"]
        assert softmax["value"] == pytest.approx(binary["value"])
        assert softmax["gradient"] == pytest.approx(binary["gradient"])
        assert softmax["probability"] == pytest.approx(binary["probability"])

    def test_a_label_of_no_flips_the_classification_pull(self) -> None:
        readings = client.post(
            MEASURE,
            json={"target": 0.5, "label": 0, "huber_threshold": 1.0, "raw_output": 0.0},
        ).json()["readings"]
        assert readings["binary_cross_entropy"]["gradient"] == pytest.approx(0.5)
        assert readings["softmax_cross_entropy"]["gradient"] == pytest.approx(0.5)
        assert readings["sigmoid_then_squared"]["gradient"] == pytest.approx(0.125)


class TestWhereTheTwoClassSoftmaxAndTheSigmoidPart:
    """Their values are clipped at different floors, and only the values."""

    def test_binary_cross_entropy_saturates_at_thirty_six(self) -> None:
        at_thirty = client.post(
            CLASSIFICATION_BATCH, json={"rows": [{"score": 30.0, "label": 0}]}
        ).json()["binary_cross_entropy"]
        at_forty = client.post(
            CLASSIFICATION_BATCH, json={"rows": [{"score": 40.0, "label": 0}]}
        ).json()["binary_cross_entropy"]
        assert at_thirty["value"] == pytest.approx(30.001021, abs=1e-6)
        assert at_forty["value"] == pytest.approx(36.043653, abs=1e-6)
        assert at_forty["gradients"] == [1.0]

    def test_the_softmax_keeps_climbing_where_the_sigmoid_stopped(self) -> None:
        body = client.post(
            SOFTMAX_ROW, json={"logits": [40.0, 0.0], "true_class": 1}
        ).json()
        assert body["value"] == pytest.approx(40.0)
        assert body["gradient"] == pytest.approx([1.0, -1.0])


class TestTheCurve:
    def test_it_samples_every_loss_across_the_whole_range(self) -> None:
        body = client.post(CURVE, json={**WORKED_SETTING, "n_samples": 41}).json()
        for name in (*THE_FIVE, "sigmoid_then_squared"):
            samples = body[name]
            assert len(samples) == 41
            assert samples[0]["raw_output"] == pytest.approx(-5.0)
            assert samples[-1]["raw_output"] == pytest.approx(5.0)

    def test_the_curve_agrees_with_the_point_measurement(self) -> None:
        body = client.post(CURVE, json={**WORKED_SETTING, "n_samples": 21}).json()
        middle = body["huber_error"][10]
        assert middle["raw_output"] == pytest.approx(0.0)
        assert middle["value"] == pytest.approx(
            readings_at(0.0)["huber_error"]["value"]
        )

    def test_squared_error_is_a_bowl_and_absolute_error_a_vee(self) -> None:
        body = client.post(CURVE, json={**WORKED_SETTING, "n_samples": 101}).json()
        squared = [sample["value"] for sample in body["squared_error"]]
        absolute = [sample["gradient"] for sample in body["absolute_error"]]
        assert min(squared) == pytest.approx(0.0, abs=1e-3)
        assert {round(pull, 6) for pull in absolute} <= {-1.0, 0.0, 1.0}


class TestTheMeasuredFourAsABatch:
    """Misses of minus four, four, eight and minus eight kilograms, knee five."""

    def setup_method(self) -> None:
        self.body = client.post(
            REGRESSION_BATCH,
            json={"rows": MEASURED_FOUR_ROWS, "huber_threshold": 5.0},
        ).json()

    def test_the_misses_are_the_hand_arithmetic(self) -> None:
        assert self.body["n_rows"] == 4
        assert self.body["misses"] == [-4.0, 4.0, 8.0, -8.0]
        assert self.body["inside_knee"] == [True, True, False, False]

    def test_squared_error_costs_twenty_and_the_big_misses_pull_twice_as_hard(
        self,
    ) -> None:
        reading = self.body["squared_error"]
        assert reading["row_costs"] == [8.0, 8.0, 32.0, 32.0]
        assert reading["total_row_cost"] == 80.0
        assert reading["value"] == 20.0
        assert reading["gradients"] == [-1.0, 1.0, 2.0, -2.0]
        assert reading["pull_shares"] == pytest.approx([1 / 6, 1 / 6, 1 / 3, 1 / 3])

    def test_absolute_error_costs_six_and_every_row_pulls_alike(self) -> None:
        reading = self.body["absolute_error"]
        assert reading["row_costs"] == [4.0, 4.0, 8.0, 8.0]
        assert reading["value"] == 6.0
        assert reading["gradients"] == [-0.25, 0.25, 0.25, -0.25]
        assert reading["largest_pull_share"] == 0.25

    def test_huber_at_five_kilograms_costs_seventeen_and_three_quarters(self) -> None:
        reading = self.body["huber_error"]
        assert reading["row_costs"] == [8.0, 8.0, 27.5, 27.5]
        assert reading["value"] == 17.75
        assert reading["gradients"] == [-1.0, 1.0, 1.25, -1.25]
        assert reading["pull_shares"] == pytest.approx([2 / 9, 2 / 9, 5 / 18, 5 / 18])

    def test_doubling_the_batch_leaves_the_value_and_halves_every_pull(self) -> None:
        doubled = client.post(
            REGRESSION_BATCH,
            json={"rows": MEASURED_FOUR_ROWS * 2, "huber_threshold": 5.0},
        ).json()
        for name in ("squared_error", "absolute_error", "huber_error"):
            assert doubled[name]["value"] == self.body[name]["value"]
            assert (
                doubled[name]["total_row_cost"] == 2 * self.body[name]["total_row_cost"]
            )
            assert doubled[name]["gradients"][:4] == pytest.approx(
                [pull / 2 for pull in self.body[name]["gradients"]]
            )

    def test_a_knee_of_one_kilogram_is_absolute_error_in_disguise_here(self) -> None:
        narrow = client.post(
            REGRESSION_BATCH,
            json={"rows": MEASURED_FOUR_ROWS, "huber_threshold": 1.0},
        ).json()
        assert narrow["inside_knee"] == [False, False, False, False]
        assert narrow["huber_error"]["value"] == 5.5
        assert narrow["huber_error"]["row_costs"] == [3.5, 3.5, 7.5, 7.5]
        assert narrow["huber_error"]["gradients"] == [-0.25, 0.25, 0.25, -0.25]


class TestOneMistypedWeight:
    """The ideal fifteen, then one of them with 72.5 entered as 27.5."""

    def test_the_clean_walk_lands_on_the_closed_form(self) -> None:
        body = client.post(
            DESCEND_LINE, json={"points": IDEAL_PEOPLE, "huber_threshold": 5.0}
        ).json()
        assert body["n_rows"] == 15
        assert body["least_squares"]["slope"] == pytest.approx(0.796786, abs=1e-6)
        assert body["least_squares"]["intercept"] == pytest.approx(-67.443929, abs=1e-6)
        assert body["descent_gap"] < 1e-9
        squared = body["squared_error"]
        assert squared["starting_loss"] == pytest.approx(53.418)
        assert squared["final_loss"] == pytest.approx(0.089132, abs=1e-6)
        assert squared["final_largest_movement"] < 1e-12
        assert len(squared["loss_history"]) == 400

    def test_huber_is_squared_error_when_every_miss_is_inside_the_knee(self) -> None:
        body = client.post(
            DESCEND_LINE, json={"points": IDEAL_PEOPLE, "huber_threshold": 5.0}
        ).json()
        assert body["huber_error"]["slope"] == pytest.approx(
            body["squared_error"]["slope"]
        )
        assert body["huber_error"]["intercept"] == pytest.approx(
            body["squared_error"]["intercept"]
        )
        assert body["huber_error"]["final_loss"] == pytest.approx(
            body["squared_error"]["final_loss"]
        )

    def test_absolute_error_keeps_stepping_because_a_sign_never_shrinks(self) -> None:
        body = client.post(
            DESCEND_LINE, json={"points": IDEAL_PEOPLE, "huber_threshold": 5.0}
        ).json()
        absolute = body["absolute_error"]
        assert absolute["slope"] == pytest.approx(0.799286, abs=1e-6)
        assert absolute["intercept"] == pytest.approx(-67.876429, abs=1e-6)
        assert absolute["final_largest_movement"] == pytest.approx(0.092582, abs=1e-6)

    def test_the_outlier_moves_the_three_lines_by_very_different_amounts(self) -> None:
        body = client.post(
            DESCEND_LINE,
            json={"points": [*IDEAL_PEOPLE, MISTYPED_PERSON], "huber_threshold": 5.0},
        ).json()
        squared, absolute, huber = (
            body["squared_error"],
            body["absolute_error"],
            body["huber_error"],
        )
        assert squared["slope"] == pytest.approx(0.746407, abs=1e-6)
        assert squared["intercept"] == pytest.approx(-61.549633, abs=1e-6)
        assert huber["slope"] == pytest.approx(0.790833, abs=1e-6)
        assert huber["intercept"] == pytest.approx(-66.747500, abs=1e-6)
        assert absolute["slope"] == pytest.approx(0.798265, abs=1e-6)
        assert absolute["intercept"] == pytest.approx(-67.718224, abs=1e-6)
        at_176 = {
            name: reading["slope"] * 176 + reading["intercept"]
            for name, reading in (
                ("squared", squared),
                ("huber", huber),
                ("absolute", absolute),
            )
        }
        assert at_176["squared"] == pytest.approx(69.818, abs=1e-3)
        assert at_176["huber"] == pytest.approx(72.439, abs=1e-3)
        assert at_176["absolute"] == pytest.approx(72.776, abs=1e-3)
        assert squared["final_loss"] == pytest.approx(59.977256, abs=1e-6)
        assert huber["final_loss"] == pytest.approx(13.400674, abs=1e-6)
        assert absolute["final_loss"] == pytest.approx(3.170728, abs=1e-6)
        assert body["descent_gap"] < 1e-9

    def test_under_the_clean_line_the_mistyped_row_carries_nearly_all_the_cost(
        self,
    ) -> None:
        clean = client.post(
            DESCEND_LINE, json={"points": IDEAL_PEOPLE, "huber_threshold": 5.0}
        ).json()["least_squares"]
        rows = [
            {
                "prediction": clean["slope"] * person["x"] + clean["intercept"],
                "truth": person["y"],
            }
            for person in [*IDEAL_PEOPLE, MISTYPED_PERSON]
        ]
        body = client.post(
            REGRESSION_BATCH, json={"rows": rows, "huber_threshold": 5.0}
        ).json()
        assert body["misses"][-1] == pytest.approx(45.290357, abs=1e-6)
        squared = body["squared_error"]
        assert squared["value"] == pytest.approx(64.184075, abs=1e-6)
        assert squared["row_costs"][-1] / squared["total_row_cost"] == pytest.approx(
            0.998698, abs=1e-6
        )
        assert squared["pull_shares"][-1] == pytest.approx(0.892452, abs=1e-6)
        absolute = body["absolute_error"]
        assert absolute["value"] == pytest.approx(3.171763, abs=1e-6)
        assert absolute["pull_shares"][-1] == pytest.approx(1 / 16)
        huber = body["huber_error"]
        assert huber["value"] == pytest.approx(13.455548, abs=1e-6)
        assert huber["pull_shares"][-1] == pytest.approx(0.478109, abs=1e-6)


class TestAMislabelledAdult:
    """Sixteen people scored by a stated rule, then one adult marked a child."""

    def test_the_clean_crowd_is_all_called_correctly_at_a_small_cost(self) -> None:
        body = client.post(CLASSIFICATION_BATCH, json={"rows": scored_crowd()}).json()
        assert body["n_rows"] == 16
        assert body["accuracy"] == 1.0
        reading = body["binary_cross_entropy"]
        assert reading["value"] == pytest.approx(0.085381, abs=1e-6)
        assert reading["largest_pull_share"] == pytest.approx(0.257312, abs=1e-6)
        assert reading["row_costs"][7] == pytest.approx(0.371101, abs=1e-6)
        assert reading["row_costs"][8] == pytest.approx(0.371101, abs=1e-6)

    def test_accuracy_cannot_tell_three_scales_apart_and_the_loss_can(self) -> None:
        values = {}
        for scale in (2.5, 5.0, 10.0):
            body = client.post(
                CLASSIFICATION_BATCH, json={"rows": scored_crowd(scale)}
            ).json()
            assert body["accuracy"] == 1.0
            values[scale] = body["binary_cross_entropy"]["value"]
        assert values[2.5] == pytest.approx(0.028976, abs=1e-6)
        assert values[5.0] == pytest.approx(0.085381, abs=1e-6)
        assert values[10.0] == pytest.approx(0.191008, abs=1e-6)

    def test_one_flipped_label_takes_most_of_the_cost_and_nearly_half_the_pull(
        self,
    ) -> None:
        rows = scored_crowd()
        rows[13] = {**rows[13], "label": 0}
        body = client.post(CLASSIFICATION_BATCH, json={"rows": rows}).json()
        assert body["accuracy"] == pytest.approx(15 / 16)
        reading = body["binary_cross_entropy"]
        assert reading["value"] == pytest.approx(0.410381, abs=1e-6)
        assert reading["row_costs"][13] == pytest.approx(5.205501, abs=1e-6)
        assert reading["row_costs"][13] / reading["total_row_cost"] == pytest.approx(
            0.792785, abs=1e-6
        )
        assert reading["gradients"][13] == pytest.approx(0.062157, abs=1e-6)
        assert reading["pull_shares"][13] == pytest.approx(0.453310, abs=1e-6)


class TestOneRowOfClassScores:
    def test_scores_one_two_three_with_the_last_class_true(self) -> None:
        body = client.post(
            SOFTMAX_ROW, json={"logits": [1.0, 2.0, 3.0], "true_class": 2}
        ).json()
        assert body["probabilities"] == pytest.approx(
            [0.090031, 0.244728, 0.665241], abs=1e-6
        )
        assert body["value"] == pytest.approx(0.407606, abs=1e-6)
        assert body["gradient"] == pytest.approx(
            [0.090031, 0.244728, -0.334759], abs=1e-6
        )
        assert body["gradient_sum"] == pytest.approx(0.0, abs=1e-12)

    def test_adding_a_hundred_to_every_score_changes_nothing(self) -> None:
        plain = client.post(
            SOFTMAX_ROW, json={"logits": [1.0, 2.0, 3.0], "true_class": 2}
        ).json()
        shifted = client.post(
            SOFTMAX_ROW, json={"logits": [101.0, 102.0, 103.0], "true_class": 2}
        ).json()
        assert shifted["probabilities"] == plain["probabilities"]
        assert shifted["value"] == plain["value"]
        assert shifted["gradient"] == plain["gradient"]

    def test_the_wrong_true_class_costs_the_log_of_its_small_probability(self) -> None:
        body = client.post(
            SOFTMAX_ROW, json={"logits": [1.0, 2.0, 3.0], "true_class": 0}
        ).json()
        assert body["value"] == pytest.approx(2.407606, abs=1e-6)
        assert body["gradient"][0] == pytest.approx(-0.909969, abs=1e-6)

    def test_a_flat_row_costs_the_log_of_the_width(self) -> None:
        body = client.post(
            SOFTMAX_ROW, json={"logits": [0.0, 0.0, 0.0], "true_class": 1}
        ).json()
        assert body["value"] == pytest.approx(math.log(3.0))
        assert body["gradient"] == pytest.approx([1 / 3, -2 / 3, 1 / 3])


class TestRefusals:
    def test_a_raw_output_beyond_the_range_is_refused(self) -> None:
        response = client.post(MEASURE, json={**WORKED_SETTING, "raw_output": 6.0})
        assert response.status_code == 422

    def test_a_label_that_is_not_yes_or_no_is_refused(self) -> None:
        response = client.post(
            MEASURE,
            json={"target": 0.5, "label": 2, "huber_threshold": 1.0, "raw_output": 0.0},
        )
        assert response.status_code == 422

    def test_a_huber_threshold_of_zero_is_refused(self) -> None:
        response = client.post(
            MEASURE,
            json={"target": 0.5, "label": 1, "huber_threshold": 0.0, "raw_output": 0.0},
        )
        assert response.status_code == 422
        response = client.post(
            REGRESSION_BATCH, json={"rows": MEASURED_FOUR_ROWS, "huber_threshold": 0.0}
        )
        assert response.status_code == 422

    def test_an_unknown_field_is_refused_rather_than_ignored(self) -> None:
        response = client.post(
            MEASURE, json={**WORKED_SETTING, "raw_output": 0.0, "prediction": 0.0}
        )
        assert response.status_code == 422

    def test_too_coarse_a_sweep_is_refused(self) -> None:
        response = client.post(CURVE, json={**WORKED_SETTING, "n_samples": 5})
        assert response.status_code == 422

    def test_an_empty_batch_is_refused_at_the_door(self) -> None:
        assert client.post(REGRESSION_BATCH, json={"rows": []}).status_code == 422
        assert client.post(CLASSIFICATION_BATCH, json={"rows": []}).status_code == 422

    def test_a_batch_of_one_row_is_accepted_and_divides_by_one(self) -> None:
        body = client.post(
            REGRESSION_BATCH, json={"rows": [{"prediction": 71.0, "truth": 63.0}]}
        ).json()
        assert body["squared_error"]["value"] == 32.0
        assert body["squared_error"]["gradients"] == [8.0]
        assert body["squared_error"]["pull_shares"] == [1.0]

    def test_a_batch_that_pulls_nowhere_reports_no_shares(self) -> None:
        body = client.post(
            REGRESSION_BATCH, json={"rows": [{"prediction": 63.0, "truth": 63.0}]}
        ).json()
        assert body["absolute_error"]["pull_shares"] == [0.0]
        assert body["absolute_error"]["largest_pull_share"] == 0.0

    def test_a_class_label_of_two_is_refused_for_the_batch_too(self) -> None:
        response = client.post(
            CLASSIFICATION_BATCH, json={"rows": [{"score": 1.0, "label": 2}]}
        )
        assert response.status_code == 422

    def test_a_softmax_row_of_one_class_is_refused(self) -> None:
        response = client.post(SOFTMAX_ROW, json={"logits": [1.0], "true_class": 0})
        assert response.status_code == 422

    def test_a_true_class_beyond_the_row_is_refused_with_the_width_named(self) -> None:
        response = client.post(
            SOFTMAX_ROW, json={"logits": [1.0, 2.0], "true_class": 2}
        )
        assert response.status_code == 422
        assert "2 class scores" in response.json()["detail"]

    def test_a_negative_true_class_is_refused_rather_than_wrapped(self) -> None:
        response = client.post(
            SOFTMAX_ROW, json={"logits": [1.0, 2.0], "true_class": -1}
        )
        assert response.status_code == 422

    def test_people_all_of_one_height_are_refused_by_the_library(self) -> None:
        response = client.post(
            DESCEND_LINE,
            json={
                "points": [
                    {"x": 170, "y": 60},
                    {"x": 170, "y": 70},
                    {"x": 170, "y": 80},
                ]
            },
        )
        assert response.status_code == 422
        assert response.json()["error"] == "AllSameValuesError"

    def test_a_single_person_cannot_be_descended_through(self) -> None:
        response = client.post(DESCEND_LINE, json={"points": [{"x": 170, "y": 60}]})
        assert response.status_code == 422

    def test_too_many_epochs_are_refused(self) -> None:
        response = client.post(
            DESCEND_LINE, json={"points": IDEAL_PEOPLE, "epochs": 401}
        )
        assert response.status_code == 422
