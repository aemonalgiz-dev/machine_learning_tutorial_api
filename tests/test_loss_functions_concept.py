"""The loss-functions endpoints, pinned to the page's hand arithmetic.

The page's worked example sets the regression truth to one half, the class
label to yes, and the raw output to zero. Squared error then costs one eighth
and pulls by minus one half; absolute error costs one half and pulls by minus
one; Huber at its default threshold is inside the knee and agrees with squared
error; both cross-entropies apply a squash that lands on one half and cost the
natural logarithm of two, pulling by minus one half. Three of the five pull by
exactly the same amount, which is the page's thesis and is designed rather
than accidental. At a raw output of three the same three part company, and the
page quotes those numbers too. All pinned here.
"""

import math

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

MEASURE = "/concepts/loss-functions/measure"
CURVE = "/concepts/loss-functions/curve"

WORKED_SETTING = {"target": 0.5, "label": 1, "huber_threshold": 1.0}


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

    def test_the_three_that_agreed_at_zero_part_company_here(self) -> None:
        readings = readings_at(3.0)
        assert readings["squared_error"]["gradient"] == pytest.approx(2.5)
        assert readings["binary_cross_entropy"]["gradient"] == pytest.approx(
            -0.047425873, abs=1e-6
        )


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


class TestTheCurve:
    def test_it_samples_every_loss_across_the_whole_range(self) -> None:
        body = client.post(CURVE, json={**WORKED_SETTING, "n_samples": 41}).json()
        for name in (
            "squared_error",
            "absolute_error",
            "huber_error",
            "binary_cross_entropy",
            "softmax_cross_entropy",
        ):
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

    def test_an_unknown_field_is_refused_rather_than_ignored(self) -> None:
        response = client.post(
            MEASURE, json={**WORKED_SETTING, "raw_output": 0.0, "prediction": 0.0}
        )
        assert response.status_code == 422

    def test_too_coarse_a_sweep_is_refused(self) -> None:
        response = client.post(CURVE, json={**WORKED_SETTING, "n_samples": 5})
        assert response.status_code == 422
