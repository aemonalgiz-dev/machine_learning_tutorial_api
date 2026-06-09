"""The walk endpoints, pinned to the page's hand arithmetic.

The page walks three people, heights 165, 170 and 175 with weights 62, 68 and
68, at a rate of 0.02. Measured from the mean height of 170 the heights are
-5, 0 and 5, so starting from a flat line at zero the residuals are the
weights themselves and the first gradient is (-132, -20), whole numbers the
page works with a pencil. One step lands at a level of 2.64 and a slope of
0.4, the second at 5.1744 and 0.5333, the walk settles on pass 477 of its
500, and the line it settles on is the closed form's, slope 0.6, level 66,
intercept -36, with a loss of exactly 2. The threshold for that data is a
rate of 0.06, and the sweep shows what the library says on either side of
it, including the refusal by name. The line page's crowd, which the widget
opens on, is pinned as well. All of it here.
"""

import math
from itertools import pairwise

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.gradient_descent_regression import (
    MAX_EPOCHS,
    MAX_LEARNING_RATE,
    MAX_SWEEP_RATES,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

WALK = "/concepts/gradient-descent-regression/walk"
SWEEP = "/concepts/gradient-descent-regression/rate-sweep"

WORKED_THREE = [
    {"x": 165.0, "y": 62.0},
    {"x": 170.0, "y": 68.0},
    {"x": 175.0, "y": 68.0},
]

WORKED_RATE = 0.02

# The line page's crowd, fifteen people spread across the range.
CROWD = [
    {"x": 152, "y": 51},
    {"x": 155, "y": 61},
    {"x": 158, "y": 56},
    {"x": 161, "y": 66},
    {"x": 164, "y": 62},
    {"x": 167, "y": 71},
    {"x": 170, "y": 66},
    {"x": 173, "y": 76},
    {"x": 176, "y": 71},
    {"x": 179, "y": 81},
    {"x": 182, "y": 75},
    {"x": 185, "y": 86},
    {"x": 188, "y": 81},
    {"x": 191, "y": 91},
    {"x": 194, "y": 86},
]

SWEEP_RATES = [0.005, 0.01, 0.02, 0.05, 0.06, 0.07, 0.1, 0.2]

LIBRARY_TOLERANCE = 1e-8


def walked(points: list[dict], learning_rate: float, max_epochs: int = MAX_EPOCHS):
    response = client.post(
        WALK,
        json={
            "points": points,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestTheWorkedWalk:
    """Three people at a rate of 0.02, the page's pencil arithmetic."""

    def test_the_heights_are_measured_from_one_hundred_and_seventy(self) -> None:
        body = walked(WORKED_THREE, WORKED_RATE)

        assert body["mean_height"] == 170.0

    def test_the_walk_starts_from_a_flat_line_at_zero(self) -> None:
        body = walked(WORKED_THREE, WORKED_RATE)

        assert body["start"]["slope"] == 0.0
        assert body["start"]["level"] == 0.0
        assert body["start"]["intercept"] == 0.0

    def test_the_loss_at_the_start_is_the_mean_squared_weight(self) -> None:
        # (62^2 + 68^2 + 68^2) / 3 = 13092 / 3
        body = walked(WORKED_THREE, WORKED_RATE)

        assert body["start"]["loss"] == 4364.0

    def test_the_first_gradient_is_the_whole_number_pair(self) -> None:
        # Level: -(2/3)(62 + 68 + 68) = -132. Slope: -(2/3)(-5*62 + 5*68) = -20.
        body = walked(WORKED_THREE, WORKED_RATE)
        first = body["passes"][0]

        assert first["pass_number"] == 1
        assert first["gradient_level"] == pytest.approx(-132.0)
        assert first["gradient_slope"] == pytest.approx(-20.0)

    def test_the_first_step_lands_where_the_page_says(self) -> None:
        # 0.02 * 132 = 2.64 and 0.02 * 20 = 0.4, and the intercept the line
        # page would report is 2.64 - 0.4 * 170.
        body = walked(WORKED_THREE, WORKED_RATE)
        first = body["passes"][0]

        assert first["level"] == pytest.approx(2.64)
        assert first["slope"] == pytest.approx(0.4)
        assert first["intercept"] == pytest.approx(-65.36)
        assert first["loss"] == pytest.approx(4017.1563, abs=1e-4)
        assert first["movement"] == pytest.approx(2.64)

    def test_the_second_step_follows_from_the_first(self) -> None:
        # Residuals 61.36, 65.36, 63.36 give a level gradient of -126.72 and a
        # slope gradient of -(2/3)(5 * 2) = -6.667, so the walk lands at
        # 2.64 + 2.5344 and 0.4 + 0.1333.
        body = walked(WORKED_THREE, WORKED_RATE)
        second = body["passes"][1]

        assert second["pass_number"] == 2
        assert second["gradient_level"] == pytest.approx(-126.72)
        assert second["gradient_slope"] == pytest.approx(-6.6667, abs=1e-4)
        assert second["level"] == pytest.approx(5.1744)
        assert second["slope"] == pytest.approx(0.5333, abs=1e-4)
        assert second["intercept"] == pytest.approx(-85.4923, abs=1e-4)
        assert second["loss"] == pytest.approx(3701.8277, abs=1e-4)

    def test_the_walk_settles_on_pass_four_hundred_and_seventy_seven(self) -> None:
        body = walked(WORKED_THREE, WORKED_RATE)

        assert body["outcome"] == "converged"
        assert body["passes_run"] == 477
        assert len(body["passes"]) == 477
        assert body["passes"][-1]["movement"] < LIBRARY_TOLERANCE
        assert body["tolerance"] == LIBRARY_TOLERANCE

    def test_the_walk_ends_on_the_closed_form_line(self) -> None:
        body = walked(WORKED_THREE, WORKED_RATE)
        last = body["passes"][-1]

        assert body["closed_form"]["slope"] == pytest.approx(0.6)
        assert body["closed_form"]["level"] == pytest.approx(66.0)
        assert body["closed_form"]["intercept"] == pytest.approx(-36.0)
        assert body["closed_form"]["loss"] == pytest.approx(2.0)
        assert last["slope"] == pytest.approx(0.6, abs=1e-6)
        assert last["level"] == pytest.approx(66.0, abs=1e-5)
        assert last["loss"] == pytest.approx(2.0, abs=1e-6)

    def test_the_loss_only_falls(self) -> None:
        body = walked(WORKED_THREE, WORKED_RATE)
        losses = [body["start"]["loss"]] + [each["loss"] for each in body["passes"]]

        assert all(later <= earlier for earlier, later in pairwise(losses))

    def test_each_pass_lands_where_the_next_gradient_is_read(self) -> None:
        # The property that makes it a walk. The gradient on pass k + 1 is
        # the gradient of the loss at the line pass k landed on, which the
        # page's arithmetic for step two relies on.
        body = walked(WORKED_THREE, WORKED_RATE)
        first, second = body["passes"][:2]
        residuals = [
            point["y"] - (first["level"] + first["slope"] * (point["x"] - 170.0))
            for point in WORKED_THREE
        ]

        assert second["gradient_level"] == pytest.approx(-(2 / 3) * sum(residuals))

    def test_the_threshold_is_six_hundredths_centred(self) -> None:
        # The steep direction is the slope, with curvature 2 * variance and a
        # variance of 50 / 3, so the threshold is 2 / (100 / 3).
        body = walked(WORKED_THREE, WORKED_RATE)

        assert body["divergence_threshold"] == pytest.approx(0.06)

    def test_the_threshold_would_be_thousands_of_times_smaller_raw(self) -> None:
        body = walked(WORKED_THREE, WORKED_RATE)

        assert body["divergence_threshold_uncentred"] == pytest.approx(
            3.458e-5, rel=1e-3
        )

    def test_the_two_directions_keep_different_shares_of_their_error(self) -> None:
        # The level has curvature 2, so each pass keeps 1 - 0.04 of its error.
        # The slope has curvature 100 / 3 and keeps 1 - 2/3, a third.
        body = walked(WORKED_THREE, WORKED_RATE)

        assert body["level_factor_per_pass"] == pytest.approx(0.96)
        assert body["slope_factor_per_pass"] == pytest.approx(1 / 3)
        assert body["error_factor_per_pass"] == pytest.approx(0.96)

    def test_the_library_agrees_with_the_record(self) -> None:
        body = walked(WORKED_THREE, WORKED_RATE)
        verdict = body["library_verdict"]

        assert verdict["verdict"] == "converged"
        assert verdict["epochs_run"] == 477
        assert verdict["slope"] == pytest.approx(0.6)
        assert verdict["level"] == pytest.approx(66.0, abs=1e-5)
        assert verdict["error_name"] is None


class TestTheCrowd:
    """The line page's fifteen people, which the widget opens on."""

    def test_the_line_the_walk_is_heading_for(self) -> None:
        body = walked(CROWD, 0.005)

        assert body["mean_height"] == 173.0
        assert body["closed_form"]["slope"] == pytest.approx(0.82619, abs=1e-5)
        assert body["closed_form"]["level"] == pytest.approx(72.0)
        assert body["closed_form"]["intercept"] == pytest.approx(-70.931, abs=1e-3)
        assert body["closed_form"]["loss"] == pytest.approx(14.1248, abs=1e-4)

    def test_the_thresholds_the_page_quotes(self) -> None:
        body = walked(CROWD, 0.005)

        assert body["divergence_threshold"] == pytest.approx(0.005952, rel=1e-3)
        assert body["divergence_threshold_uncentred"] == pytest.approx(
            3.3225e-5, rel=1e-3
        )

    def test_a_safe_rate_is_still_walking_when_the_passes_run_out(self) -> None:
        body = walked(CROWD, 0.005)
        last = body["passes"][-1]

        assert body["outcome"] == "pass_limit_reached"
        assert body["passes_run"] == MAX_EPOCHS
        assert body["level_factor_per_pass"] == pytest.approx(0.99)
        assert body["slope_factor_per_pass"] == pytest.approx(-0.68)
        assert last["slope"] == pytest.approx(0.82619, abs=1e-5)
        assert last["level"] == pytest.approx(71.5269, abs=1e-4)
        assert last["loss"] == pytest.approx(14.3486, abs=1e-4)
        assert body["library_verdict"]["verdict"] == "pass_limit_reached"

    def test_one_tick_above_the_threshold_runs_away_slowly(self) -> None:
        body = walked(CROWD, 0.006)
        last = body["passes"][-1]

        assert body["outcome"] == "diverged"
        assert body["slope_factor_per_pass"] == pytest.approx(-1.016)
        assert body["passes_run"] == MAX_EPOCHS
        assert last["slope"] == pytest.approx(-2310.9, abs=0.1)
        assert body["library_verdict"]["verdict"] == "pass_limit_reached"


class TestARunawayWalk:
    def test_a_rate_above_the_threshold_is_reported_as_diverged(self) -> None:
        body = walked(WORKED_THREE, 0.2)

        assert body["outcome"] == "diverged"
        assert body["slope_factor_per_pass"] == pytest.approx(-5.6667, abs=1e-4)
        assert body["error_factor_per_pass"] == pytest.approx(5.6667, abs=1e-4)

    def test_the_record_stops_before_the_numbers_overflow(self) -> None:
        body = walked(WORKED_THREE, 0.2)
        losses = [each["loss"] for each in body["passes"]]

        assert body["passes_run"] == 13
        assert all(math.isfinite(loss) for loss in losses)
        assert losses[-1] > losses[0] > body["closed_form"]["loss"]

    def test_the_slope_swings_wider_every_pass(self) -> None:
        body = walked(WORKED_THREE, 0.2)
        misses = [abs(each["slope"] - 0.6) for each in body["passes"]]

        assert all(later > earlier for earlier, later in pairwise(misses))

    def test_the_library_refuses_the_same_rate_by_name(self) -> None:
        body = walked(WORKED_THREE, 0.2)
        verdict = body["library_verdict"]

        assert verdict["verdict"] == "refused"
        assert verdict["error_name"] == "DivergenceError"
        assert "Lower the learning rate" in verdict["message"]
        assert verdict["epochs_run"] is None
        assert verdict["slope"] is None

    def test_just_above_the_threshold_the_library_does_not_refuse(self) -> None:
        # At 0.07 the slope's error grows by a third each pass, and after five
        # hundred passes the numbers are absurd but still finite, so the
        # library reports a fit that did not converge rather than a refusal.
        body = walked(WORKED_THREE, 0.07)
        verdict = body["library_verdict"]

        assert body["outcome"] == "diverged"
        assert body["passes_run"] == 81
        assert body["slope_factor_per_pass"] == pytest.approx(-4 / 3)
        assert verdict["verdict"] == "pass_limit_reached"
        assert verdict["epochs_run"] == MAX_EPOCHS
        assert abs(verdict["slope"]) > 1e60

    def test_exactly_at_the_threshold_the_slope_hops_forever(self) -> None:
        # The slope's factor is exactly minus one, so it alternates between
        # 1.2 and 0 while the level settles at 66, and the loss it holds at
        # is (16 + 4 + 4) / 3.
        body = walked(WORKED_THREE, 0.06)
        passes = body["passes"]

        assert body["outcome"] == "pass_limit_reached"
        assert body["slope_factor_per_pass"] == pytest.approx(-1.0)
        assert body["passes_run"] == MAX_EPOCHS
        assert passes[0]["slope"] == pytest.approx(1.2)
        assert passes[1]["slope"] == pytest.approx(0.0, abs=1e-9)
        assert passes[-1]["slope"] == pytest.approx(0.0, abs=1e-9)
        assert passes[-1]["level"] == pytest.approx(66.0)
        assert passes[-1]["loss"] == pytest.approx(8.0)

    def test_the_largest_rate_is_recorded_for_six_passes(self) -> None:
        body = walked(WORKED_THREE, MAX_LEARNING_RATE)

        assert body["outcome"] == "diverged"
        assert body["passes_run"] == 6
        assert body["level_factor_per_pass"] == pytest.approx(-1.0)
        assert body["library_verdict"]["error_name"] == "DivergenceError"

    def test_a_short_budget_is_honoured(self) -> None:
        body = walked(WORKED_THREE, WORKED_RATE, max_epochs=3)

        assert body["outcome"] == "pass_limit_reached"
        assert body["passes_run"] == 3
        assert body["library_verdict"]["epochs_run"] == 3


class TestTheRateSweep:
    def swept(self) -> dict:
        response = client.post(
            SWEEP,
            json={
                "points": WORKED_THREE,
                "learning_rates": SWEEP_RATES,
                "max_epochs": MAX_EPOCHS,
            },
        )
        assert response.status_code == 200, response.text
        return response.json()

    def verdict_at(self, learning_rate: float) -> dict:
        body = self.swept()
        return next(
            each for each in body["verdicts"] if each["learning_rate"] == learning_rate
        )

    def test_one_verdict_per_rate_in_order(self) -> None:
        body = self.swept()

        assert [each["learning_rate"] for each in body["verdicts"]] == SWEEP_RATES
        assert body["divergence_threshold"] == pytest.approx(0.06)
        assert body["closed_form"]["slope"] == pytest.approx(0.6)
        assert body["closed_form"]["level"] == pytest.approx(66.0)

    def test_the_worked_rate_settles_in_the_same_number_of_passes(self) -> None:
        verdict = self.verdict_at(0.02)

        assert verdict["verdict"] == "converged"
        assert verdict["epochs_run"] == 477
        assert verdict["slope"] == pytest.approx(0.6)
        assert verdict["level"] == pytest.approx(66.0, abs=1e-5)

    def test_a_bolder_safe_rate_settles_sooner(self) -> None:
        verdict = self.verdict_at(0.05)

        assert verdict["verdict"] == "converged"
        assert verdict["epochs_run"] == 194
        assert verdict["error_factor_per_pass"] == pytest.approx(0.9)

    def test_too_timid_a_rate_is_still_walking(self) -> None:
        verdict = self.verdict_at(0.005)

        assert verdict["verdict"] == "pass_limit_reached"
        assert verdict["epochs_run"] == MAX_EPOCHS
        assert verdict["level"] == pytest.approx(65.5663, abs=1e-4)
        assert verdict["error_factor_per_pass"] == pytest.approx(0.99)

    def test_at_the_threshold_the_slope_never_settles(self) -> None:
        verdict = self.verdict_at(0.06)

        assert verdict["verdict"] == "pass_limit_reached"
        assert verdict["level"] == pytest.approx(66.0, abs=1e-5)
        assert verdict["slope"] == pytest.approx(0.0, abs=1e-9)

    def test_just_above_it_the_numbers_are_finite_and_absurd(self) -> None:
        verdict = self.verdict_at(0.07)

        assert verdict["verdict"] == "pass_limit_reached"
        assert abs(verdict["slope"]) > 1e60
        assert verdict["error_factor_per_pass"] == pytest.approx(4 / 3)

    def test_well_above_it_the_library_refuses_by_name(self) -> None:
        verdict = self.verdict_at(0.2)

        assert verdict["verdict"] == "refused"
        assert verdict["error_name"] == "DivergenceError"
        assert "Lower the learning rate" in verdict["message"]
        assert verdict["epochs_run"] is None
        assert verdict["slope"] is None
        assert verdict["level"] is None


class TestRefusals:
    def test_identical_heights_have_no_line_to_walk_toward(self) -> None:
        response = client.post(
            WALK,
            json={
                "points": [
                    {"x": 170.0, "y": 60.0},
                    {"x": 170.0, "y": 65.0},
                    {"x": 170.0, "y": 70.0},
                ],
                "learning_rate": WORKED_RATE,
                "max_epochs": MAX_EPOCHS,
            },
        )

        assert response.status_code == 422
        assert response.json()["error"] == "AllSameValuesError"

    def test_a_lone_point_is_refused(self) -> None:
        response = client.post(
            WALK,
            json={
                "points": [{"x": 170.0, "y": 68.0}],
                "learning_rate": WORKED_RATE,
                "max_epochs": MAX_EPOCHS,
            },
        )

        assert response.status_code == 422

    def test_a_rate_beyond_the_bound_is_refused(self) -> None:
        response = client.post(
            WALK,
            json={
                "points": WORKED_THREE,
                "learning_rate": MAX_LEARNING_RATE * 2,
                "max_epochs": MAX_EPOCHS,
            },
        )

        assert response.status_code == 422

    def test_a_budget_beyond_the_bound_is_refused(self) -> None:
        response = client.post(
            WALK,
            json={
                "points": WORKED_THREE,
                "learning_rate": WORKED_RATE,
                "max_epochs": MAX_EPOCHS + 1,
            },
        )

        assert response.status_code == 422

    def test_a_zero_rate_is_refused_in_the_sweep(self) -> None:
        response = client.post(
            SWEEP,
            json={
                "points": WORKED_THREE,
                "learning_rates": [0.0],
                "max_epochs": MAX_EPOCHS,
            },
        )

        assert response.status_code == 422

    def test_too_many_rates_is_refused(self) -> None:
        response = client.post(
            SWEEP,
            json={
                "points": WORKED_THREE,
                "learning_rates": [0.01] * (MAX_SWEEP_RATES + 1),
                "max_epochs": MAX_EPOCHS,
            },
        )

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            WALK,
            json={
                "points": WORKED_THREE,
                "learning_rate": WORKED_RATE,
                "max_epochs": MAX_EPOCHS,
                "epochs": 5,
            },
        )

        assert response.status_code == 422


CANDIDATE = "/concepts/gradient-descent-regression/candidate"
SURFACE = "/concepts/gradient-descent-regression/loss-surface"


class TestCandidate:
    def test_the_flat_line_at_zero_reads_the_whole_number_gradient(self) -> None:
        body = client.post(
            CANDIDATE,
            json={
                "points": WORKED_THREE,
                "level": 0,
                "slope": 0,
                "learning_rate": 0.02,
            },
        ).json()

        assert abs(body["gradient_level"] + 132) < 1e-9
        assert abs(body["gradient_slope"] + 20) < 1e-9
        assert abs(body["standing"]["loss"] - 4364) < 1e-9
        assert abs(body["next_position"]["level"] - 2.64) < 1e-9
        assert abs(body["next_position"]["slope"] - 0.4) < 1e-9

    def test_the_person_at_the_mean_height_adds_nothing_to_the_slope(self) -> None:
        body = client.post(
            CANDIDATE,
            json={
                "points": WORKED_THREE,
                "level": 0,
                "slope": 0,
                "learning_rate": 0.02,
            },
        ).json()

        middle = next(
            each for each in body["observations"] if each["centred_height"] == 0
        )
        assert middle["slope_share"] == 0
        assert middle["level_share"] != 0
        assert (
            abs(
                sum(each["level_share"] for each in body["observations"])
                - body["gradient_level"]
            )
            < 1e-9
        )

    def test_the_gradient_vanishes_at_the_closed_form(self) -> None:
        body = client.post(
            CANDIDATE,
            json={
                "points": WORKED_THREE,
                "level": 66,
                "slope": 0.6,
                "learning_rate": 0.02,
            },
        ).json()

        assert abs(body["gradient_level"]) < 1e-9
        assert abs(body["gradient_slope"]) < 1e-9
        assert abs(body["standing"]["loss"] - 2.0) < 1e-9

    def test_the_slices_pass_through_the_standing_loss(self) -> None:
        body = client.post(
            CANDIDATE,
            json={
                "points": WORKED_THREE,
                "level": 10,
                "slope": 0.2,
                "learning_rate": 0.02,
            },
        ).json()

        middle = len(body["level_slice"]["axis"]) // 2
        assert abs(body["level_slice"]["axis"][middle] - 10) < 1e-9
        assert (
            abs(body["level_slice"]["losses"][middle] - body["standing"]["loss"]) < 1e-9
        )
        assert (
            abs(body["slope_slice"]["losses"][middle] - body["standing"]["loss"]) < 1e-9
        )


class TestLossSurface:
    def test_centred_curvatures_are_the_worked_pair(self) -> None:
        body = client.post(
            SURFACE,
            json={
                "points": WORKED_THREE,
                "parameterisation": "centred",
                "learning_rate": 0.02,
                "max_epochs": 500,
            },
        ).json()

        assert abs(body["smallest_curvature"] - 2) < 1e-9
        assert abs(body["largest_curvature"] - 100 / 3) < 1e-9
        assert abs(body["divergence_threshold"] - 0.06) < 1e-9
        assert abs(body["optimum"]["first"] - 66) < 1e-9
        assert abs(body["optimum"]["second"] - 0.6) < 1e-9
        assert body["outcome"] == "converged"

    def test_raw_heights_are_a_far_narrower_valley(self) -> None:
        body = client.post(
            SURFACE,
            json={
                "points": WORKED_THREE,
                "parameterisation": "raw",
                "learning_rate": 0.02,
                "max_epochs": 500,
            },
        ).json()

        assert body["condition_number"] > 1e7
        assert body["divergence_threshold"] < 1e-4
        assert body["outcome"] == "diverged"
        assert abs(body["optimum"]["first"] + 36) < 1e-6

    def test_scaling_rounds_the_bowl(self) -> None:
        body = client.post(
            SURFACE,
            json={
                "points": WORKED_THREE,
                "parameterisation": "scaled",
                "learning_rate": 0.02,
                "max_epochs": 500,
            },
        ).json()

        assert body["condition_number"] < 2

    def test_the_lattice_is_lowest_at_the_optimum(self) -> None:
        body = client.post(
            SURFACE,
            json={
                "points": WORKED_THREE,
                "parameterisation": "centred",
                "learning_rate": 0.02,
                "max_epochs": 50,
            },
        ).json()

        lowest = min(min(row) for row in body["losses"])
        assert lowest >= body["optimum"]["loss"] - 1e-9
        assert len(body["walk"]) == 51
