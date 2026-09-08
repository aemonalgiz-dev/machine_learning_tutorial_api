"""The gradient boosting page's endpoints, pinned to the numbers it quotes.

The page works two rounds on three readings by hand, steps the fifteen-point
throw round by round at a rate of 0.3, and reads its rate, depth, stopping
and bagging comparisons off a sixty-measurement throw under the site's seeded
deal, forty-two training rows and eighteen held out. Every number the prose
quotes is pinned here, along with the refusals the failure contracts list.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import pairwise
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BOOSTING = "/concepts/ensembles/gradient-boosting"
MODEL = "/concepts/ensembles/boosting-model"
ROUNDS = "/concepts/ensembles/boosting-rounds"
CURVES = "/concepts/ensembles/boosting-curves"
VERSUS = "/concepts/ensembles/boosting-versus-bagging"

THREE_READINGS = [{"x": 1, "y": 2}, {"x": 2, "y": 6}, {"x": 3, "y": 10}]

NOISY_THROW = [
    {"x": 0.0, "y": 0.7},
    {"x": 0.29, "y": 5.2},
    {"x": 0.57, "y": 9.5},
    {"x": 0.86, "y": 13.0},
    {"x": 1.14, "y": 16.9},
    {"x": 1.43, "y": 18.1},
    {"x": 1.71, "y": 19.4},
    {"x": 2.0, "y": 21.0},
    {"x": 2.29, "y": 19.6},
    {"x": 2.57, "y": 19.1},
    {"x": 2.86, "y": 16.8},
    {"x": 3.14, "y": 13.6},
    {"x": 3.43, "y": 11.1},
    {"x": 3.71, "y": 6.2},
    {"x": 4.0, "y": 1.2},
]

SIXTY_THROW = [
    {"x": 0.11, "y": 1.7},
    {"x": 0.16, "y": 3.2},
    {"x": 0.25, "y": 5.1},
    {"x": 0.33, "y": 4.6},
    {"x": 0.46, "y": 6.5},
    {"x": 0.54, "y": 9.7},
    {"x": 0.58, "y": 9.3},
    {"x": 0.59, "y": 10.4},
    {"x": 0.64, "y": 11.9},
    {"x": 0.77, "y": 10.0},
    {"x": 0.77, "y": 12.9},
    {"x": 0.81, "y": 14.8},
    {"x": 1.04, "y": 15.1},
    {"x": 1.05, "y": 14.4},
    {"x": 1.11, "y": 17.3},
    {"x": 1.12, "y": 16.6},
    {"x": 1.21, "y": 18.4},
    {"x": 1.25, "y": 16.8},
    {"x": 1.32, "y": 15.6},
    {"x": 1.61, "y": 19.3},
    {"x": 1.64, "y": 19.0},
    {"x": 1.69, "y": 21.0},
    {"x": 1.81, "y": 20.4},
    {"x": 1.84, "y": 17.8},
    {"x": 1.89, "y": 18.5},
    {"x": 1.94, "y": 21.7},
    {"x": 2.04, "y": 21.4},
    {"x": 2.04, "y": 19.4},
    {"x": 2.05, "y": 20.4},
    {"x": 2.06, "y": 21.1},
    {"x": 2.11, "y": 21.1},
    {"x": 2.15, "y": 21.7},
    {"x": 2.16, "y": 20.7},
    {"x": 2.2, "y": 20.1},
    {"x": 2.37, "y": 19.5},
    {"x": 2.45, "y": 21.2},
    {"x": 2.49, "y": 16.0},
    {"x": 2.57, "y": 18.8},
    {"x": 2.73, "y": 18.1},
    {"x": 2.9, "y": 14.7},
    {"x": 3.0, "y": 16.4},
    {"x": 3.01, "y": 14.8},
    {"x": 3.01, "y": 17.1},
    {"x": 3.11, "y": 14.6},
    {"x": 3.15, "y": 15.4},
    {"x": 3.15, "y": 16.2},
    {"x": 3.21, "y": 14.3},
    {"x": 3.28, "y": 11.6},
    {"x": 3.31, "y": 10.2},
    {"x": 3.36, "y": 14.5},
    {"x": 3.41, "y": 11.1},
    {"x": 3.42, "y": 10.1},
    {"x": 3.45, "y": 10.9},
    {"x": 3.51, "y": 9.5},
    {"x": 3.67, "y": 8.7},
    {"x": 3.79, "y": 5.5},
    {"x": 3.8, "y": 5.3},
    {"x": 3.85, "y": 3.3},
    {"x": 3.88, "y": 4.5},
    {"x": 3.92, "y": 1.6},
]

CONSTANT_HEIGHTS = [{"x": 1, "y": 5}, {"x": 2, "y": 5}, {"x": 3, "y": 5}]


def _rounds(points, rounds, learning_rate, max_depth=1):
    return client.post(
        ROUNDS,
        json={
            "points": points,
            "rounds": rounds,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
        },
    ).json()


def _curves(points, max_rounds, configurations):
    return client.post(
        CURVES,
        json={
            "points": points,
            "max_rounds": max_rounds,
            "configurations": configurations,
        },
    ).json()


class TestThreeReadingsByHand:
    """The two rounds the page works with a pencil, and the two after them."""

    def test_the_flat_start_and_its_leftover(self) -> None:
        body = _rounds(THREE_READINGS, 4, 0.5)

        assert body["initial_prediction"] == 6.0
        assert body["residual_sum_at_start"] == 32.0
        assert body["rounds"][0]["residuals_before"] == [-4.0, 0.0, 4.0]

    def test_round_one_asks_below_one_and_a_half(self) -> None:
        first = _rounds(THREE_READINGS, 4, 0.5)["rounds"][0]

        assert first["threshold"] == 1.5
        assert first["left_value"] == -4.0
        assert first["right_value"] == 2.0
        assert first["running_predictions"] == [4.0, 7.0, 7.0]
        assert first["residual_sum_after"] == 14.0
        assert first["r_squared_after"] == 0.5625

    def test_round_two_asks_below_two_and_a_half(self) -> None:
        second = _rounds(THREE_READINGS, 4, 0.5)["rounds"][1]

        assert second["residuals_before"] == [-2.0, -1.0, 3.0]
        assert second["threshold"] == 2.5
        assert second["left_value"] == -1.5
        assert second["right_value"] == 3.0
        assert second["running_predictions"] == [3.25, 6.25, 8.5]
        assert second["residual_sum_after"] == 3.875

    def test_rounds_three_and_four_keep_shrinking_the_leftover(self) -> None:
        rounds = _rounds(THREE_READINGS, 4, 0.5)["rounds"]

        assert rounds[2]["running_predictions"] == [2.875, 5.875, 9.25]
        assert rounds[2]["residual_sum_after"] == 1.34375
        assert rounds[3]["threshold"] == 1.5
        assert rounds[3]["residual_sum_after"] == 0.482421875

    def test_the_leftover_of_a_fitted_miss_is_one_minus_the_rate_of_it(self) -> None:
        rounds = _rounds(THREE_READINGS, 2, 0.5)["rounds"]

        first_miss = rounds[0]["residuals_before"][0]
        assert rounds[1]["residuals_before"][0] == (1 - 0.5) * first_miss


class TestSteppingTheThrow:
    """The fifteen-point throw at a rate of 0.3, one stump per round."""

    def test_the_flat_start_is_the_mean_height(self) -> None:
        body = _rounds(NOISY_THROW, 3, 0.3)

        assert body["initial_prediction"] == pytest.approx(12.76)
        assert body["residual_sum_at_start"] == pytest.approx(654.356)

    def test_round_one_splits_between_the_second_and_third_reading(self) -> None:
        first = _rounds(NOISY_THROW, 3, 0.3)["rounds"][0]

        assert first["threshold"] == pytest.approx(0.43)
        assert first["left_value"] == pytest.approx(-9.81)
        assert first["right_value"] == pytest.approx(1.5092, abs=1e-4)
        assert first["residual_sum_after"] == pytest.approx(541.0935, abs=1e-3)
        assert first["r_squared_after"] == pytest.approx(0.1731, abs=1e-4)

    def test_the_next_two_thresholds_and_leftovers(self) -> None:
        rounds = _rounds(NOISY_THROW, 3, 0.3)["rounds"]

        assert rounds[1]["threshold"] == pytest.approx(3.57)
        assert rounds[1]["residual_sum_after"] == pytest.approx(434.5905, abs=1e-3)
        assert rounds[2]["threshold"] == pytest.approx(0.715)
        assert rounds[2]["residual_sum_after"] == pytest.approx(359.7725, abs=1e-3)

    def test_the_leftover_never_grows_at_a_rate_of_0_3_or_of_1(self) -> None:
        for learning_rate in (0.3, 1.0):
            rounds = _rounds(NOISY_THROW, 60, learning_rate)["rounds"]
            sums = [each["residual_sum_after"] for each in rounds]
            assert all(later <= earlier + 1e-9 for earlier, later in pairwise(sums))

    def test_the_quoted_milestones(self) -> None:
        rounds = _rounds(NOISY_THROW, 60, 0.3)["rounds"]

        assert rounds[9]["residual_sum_after"] == pytest.approx(111.5759, abs=1e-3)
        assert rounds[9]["r_squared_after"] == pytest.approx(0.8295, abs=1e-4)
        assert rounds[29]["r_squared_after"] == pytest.approx(0.985, abs=1e-3)
        assert rounds[59]["residual_sum_after"] == pytest.approx(1.8383, abs=1e-3)
        assert max(abs(v) for v in rounds[59]["residuals_before"]) == pytest.approx(
            1.2437, abs=1e-3
        )

    def test_the_replay_lands_on_the_fitted_model(self) -> None:
        rounds = _rounds(NOISY_THROW, 40, 0.3)["rounds"]
        fitted = client.post(
            MODEL,
            json={"points": NOISY_THROW, "rounds": 40, "learning_rate": 0.3},
        ).json()

        assert rounds[-1]["running_predictions"] == pytest.approx(
            fitted["predictions"], abs=1e-9
        )
        assert rounds[-1]["r_squared_after"] == pytest.approx(fitted["r_squared"])

    def test_a_deeper_member_reports_its_root_and_no_leaf_values(self) -> None:
        first = _rounds(NOISY_THROW, 2, 0.3, max_depth=3)["rounds"][0]

        assert first["threshold"] is not None
        assert first["left_value"] is None
        assert first["n_leaves"] > 2


class TestBoostingModel:
    def test_more_rounds_of_stumps_reach_the_arc(self) -> None:
        scores = {
            rounds: client.post(
                MODEL,
                json={"points": NOISY_THROW, "rounds": rounds, "learning_rate": 0.3},
            ).json()["r_squared"]
            for rounds in (1, 10, 100)
        }

        assert scores[1] == pytest.approx(0.1731, abs=1e-4)
        assert scores[10] == pytest.approx(0.8295, abs=1e-4)
        assert scores[100] == pytest.approx(0.9993, abs=1e-4)

    def test_a_hundred_stumps_hold_two_hundred_shelves(self) -> None:
        body = client.post(
            MODEL, json={"points": NOISY_THROW, "rounds": 100, "learning_rate": 0.3}
        ).json()

        assert body["total_leaves"] == 200
        assert body["initial_prediction"] == pytest.approx(12.76)

    def test_a_rate_past_one_is_refused_in_words(self) -> None:
        response = client.post(
            MODEL, json={"points": THREE_READINGS, "rounds": 5, "learning_rate": 1.5}
        )

        assert response.status_code == 422
        assert "overshoots" in response.json()["detail"]

    def test_constant_heights_fit_and_cannot_be_scored(self) -> None:
        response = client.post(
            MODEL, json={"points": CONSTANT_HEIGHTS, "rounds": 5, "learning_rate": 0.5}
        )

        assert response.status_code == 422
        assert response.json()["error"] == "UndefinedMetricError"

    def test_constant_times_leave_every_member_a_leaf(self) -> None:
        body = client.post(
            MODEL,
            json={
                "points": [{"x": 2, "y": 1}, {"x": 2, "y": 5}, {"x": 2, "y": 9}],
                "rounds": 5,
                "learning_rate": 0.5,
            },
        ).json()

        assert body["predictions"] == [5.0, 5.0, 5.0]
        assert body["r_squared"] == 0.0
        assert body["total_leaves"] == 5

    def test_two_heights_at_one_time_share_their_mean(self) -> None:
        body = client.post(
            MODEL,
            json={
                "points": [
                    {"x": 1, "y": 2},
                    {"x": 2, "y": 4},
                    {"x": 2, "y": 8},
                    {"x": 3, "y": 10},
                ],
                "rounds": 40,
                "learning_rate": 0.5,
            },
        ).json()

        assert body["predictions"][1] == pytest.approx(6.0, abs=1e-5)
        assert body["predictions"][2] == pytest.approx(6.0, abs=1e-5)
        assert body["r_squared"] == pytest.approx(0.8, abs=1e-9)

    @pytest.mark.parametrize(
        "field, value",
        [("learning_rate", 0.0), ("max_depth", 0), ("max_depth", 7), ("rounds", 301)],
    )
    def test_the_door_refuses_what_construction_would(self, field, value) -> None:
        payload = {"points": THREE_READINGS, "rounds": 5, "learning_rate": 0.5}
        payload[field] = value

        assert client.post(MODEL, json=payload).status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            MODEL,
            json={
                "points": THREE_READINGS,
                "rounds": 5,
                "learning_rate": 0.5,
                "n_rounds": 3,
            },
        )

        assert response.status_code == 422


class TestLearningCurves:
    """The sixty-measurement throw under the seeded deal."""

    RATES: ClassVar[list[dict[str, float]]] = [
        {"learning_rate": rate, "max_depth": 1} for rate in (0.05, 0.1, 0.3, 1.0)
    ]

    def test_the_deal_holds_out_eighteen_of_sixty(self) -> None:
        body = _curves(SIXTY_THROW, 5, self.RATES[:1])

        assert body["n_training"] == 42
        assert body["n_held_out"] == 18
        assert len(body["held_out_indices"]) == 18

    def test_round_zero_is_the_flat_start(self) -> None:
        curve = _curves(SIXTY_THROW, 5, self.RATES[:1])["curves"][0]

        assert curve["train_r_squared"][0] == 0.0
        assert curve["held_out_r_squared"][0] == pytest.approx(-0.0928, abs=1e-4)
        assert len(curve["train_r_squared"]) == 6

    def test_the_rate_sweep_the_page_quotes(self) -> None:
        curves = _curves(SIXTY_THROW, 300, self.RATES)["curves"]
        by_rate = {curve["learning_rate"]: curve for curve in curves}

        assert by_rate[0.05]["rounds_to_train"] == 176
        assert by_rate[0.1]["rounds_to_train"] == 86
        assert by_rate[0.3]["rounds_to_train"] == 26
        assert by_rate[1.0]["rounds_to_train"] == 14
        assert by_rate[1.0]["best_round"] == 77
        assert by_rate[1.0]["best_held_out_r_squared"] == pytest.approx(
            0.8985, abs=1e-4
        )
        assert by_rate[0.3]["best_round"] == 217
        assert by_rate[0.3]["best_held_out_r_squared"] == pytest.approx(
            0.8963, abs=1e-4
        )
        assert by_rate[0.1]["best_held_out_r_squared"] == pytest.approx(
            0.8914, abs=1e-4
        )
        assert by_rate[1.0]["final_held_out_r_squared"] == pytest.approx(
            0.8857, abs=1e-4
        )
        assert by_rate[0.3]["final_held_out_r_squared"] == pytest.approx(
            0.8951, abs=1e-4
        )

    def test_the_smallest_rate_never_reaches_the_training_threshold(self) -> None:
        curve = _curves(SIXTY_THROW, 300, [{"learning_rate": 0.02}])["curves"][0]

        assert curve["rounds_to_train"] is None
        assert curve["final_train_r_squared"] == pytest.approx(0.9001, abs=1e-4)

    def test_depth_brings_the_peak_earlier(self) -> None:
        curves = _curves(
            SIXTY_THROW,
            300,
            [{"learning_rate": 0.1, "max_depth": depth} for depth in (1, 2, 3, 4)],
        )["curves"]

        assert [curve["best_round"] for curve in curves] == [299, 97, 58, 36]
        assert [curve["rounds_to_train"] for curve in curves] == [86, 26, 19, 17]

    def test_a_deep_member_at_full_rate_explains_the_rows_in_one_round(self) -> None:
        curve = _curves(SIXTY_THROW, 5, [{"learning_rate": 1.0, "max_depth": 6}])[
            "curves"
        ][0]

        assert curve["train_r_squared"][1] == pytest.approx(0.9877, abs=1e-4)
        assert curve["held_out_r_squared"][1] == pytest.approx(0.8889, abs=1e-4)

    def test_the_stopping_round_the_page_reads(self) -> None:
        curve = _curves(SIXTY_THROW, 300, [{"learning_rate": 1.0, "max_depth": 3}])[
            "curves"
        ][0]

        assert curve["best_round"] == 5
        assert curve["best_held_out_r_squared"] == pytest.approx(0.8963, abs=1e-4)
        assert curve["final_held_out_r_squared"] == pytest.approx(0.878, abs=1e-4)
        assert curve["final_train_r_squared"] == pytest.approx(0.9937, abs=1e-4)

    def test_a_prefix_of_a_longer_fit_is_the_shorter_fit(self) -> None:
        short = _curves(SIXTY_THROW, 7, self.RATES[2:3])["curves"][0]
        long = _curves(SIXTY_THROW, 40, self.RATES[2:3])["curves"][0]

        assert short["train_r_squared"] == pytest.approx(long["train_r_squared"][:8])
        assert short["held_out_r_squared"] == pytest.approx(
            long["held_out_r_squared"][:8]
        )

    def test_too_few_points_and_too_many_configurations_are_refused(self) -> None:
        few = client.post(
            CURVES,
            json={
                "points": NOISY_THROW[:6],
                "max_rounds": 5,
                "configurations": self.RATES[:1],
            },
        )
        many = client.post(
            CURVES,
            json={
                "points": NOISY_THROW,
                "max_rounds": 5,
                "configurations": self.RATES[:1] * 7,
            },
        )

        assert few.status_code == 422
        assert many.status_code == 422

    def test_the_stepper_refuses_past_sixty_rounds(self) -> None:
        response = client.post(
            ROUNDS, json={"points": THREE_READINGS, "rounds": 61, "learning_rate": 0.5}
        )

        assert response.status_code == 422


class TestVersusBagging:
    def test_the_scores_the_page_quotes_at_fifty_members(self) -> None:
        body = client.post(
            VERSUS,
            json={"points": SIXTY_THROW, "n_members": 50, "learning_rate": 0.3},
        ).json()

        assert body["bagged_deep"]["train_r_squared"] == pytest.approx(0.9849, abs=1e-4)
        assert body["bagged_deep"]["held_out_r_squared"] == pytest.approx(
            0.8761, abs=1e-4
        )
        assert body["bagged_deep"]["out_of_bag_r_squared"] == pytest.approx(
            0.9029, abs=1e-4
        )
        assert body["bagged_stumps"]["train_r_squared"] == pytest.approx(
            0.4758, abs=1e-4
        )
        assert body["bagged_stumps"]["held_out_r_squared"] == pytest.approx(
            0.2727, abs=1e-4
        )
        assert body["boosted"]["train_r_squared"] == pytest.approx(0.9767, abs=1e-4)
        assert body["boosted"]["held_out_r_squared"] == pytest.approx(0.8832, abs=1e-4)
        assert body["boosted"]["out_of_bag_r_squared"] is None

    def test_bagged_members_see_resampled_rows_and_boosted_rounds_see_all(self) -> None:
        body = client.post(
            VERSUS,
            json={"points": SIXTY_THROW, "n_members": 50, "learning_rate": 0.3},
        ).json()

        bagged = body["bagged_deep"]["fitted_targets"]
        boosted = body["boosted"]["fitted_targets"]
        assert [each["distinct_rows"] for each in bagged] == [27, 27, 28]
        assert all(len(each["times"]) == 42 for each in bagged)
        assert [each["distinct_rows"] for each in boosted] == [42, 42, 42]
        assert set(bagged[0]["targets"]) <= set(body["training_heights"])
        sums = [sum(v * v for v in each["targets"]) for each in boosted]
        assert sums == pytest.approx([1142.69, 952.89, 784.58], abs=0.01)

    def test_a_lone_member_still_answers(self) -> None:
        response = client.post(
            VERSUS, json={"points": SIXTY_THROW, "n_members": 1, "learning_rate": 0.3}
        )

        assert response.status_code == 200
        assert response.json()["boosted"]["out_of_bag_r_squared"] is None

    def test_the_caps_hold(self) -> None:
        too_many = client.post(VERSUS, json={"points": SIXTY_THROW, "n_members": 51})
        too_few = client.post(VERSUS, json={"points": NOISY_THROW[:6], "n_members": 5})

        assert too_many.status_code == 422
        assert too_few.status_code == 422


def _every_string(answer: object) -> Iterator[str]:
    """Every string anywhere inside one answered body, keys included."""
    if isinstance(answer, str):
        yield answer
    elif isinstance(answer, dict):
        for key, value in answer.items():
            yield key
            yield from _every_string(value)
    elif isinstance(answer, list):
        for item in answer:
            yield from _every_string(item)


class TestNothingAnsweredNamesTheCode:
    """A reader sees these sentences, so none of them may name what wrote them.

    A refusal in words is the one place prose crosses from here to the page,
    and a class name, a module name or a mention of what sits underneath
    teaches a reader nothing. Every string in every body is walked rather
    than the refusals alone, so a sentence that grows such a name later fails
    here rather than in front of somebody reading the page.
    """

    FORBIDDEN: ClassVar[tuple[str, ...]] = ("library", "oop_ml", "the API")

    CALLS: ClassVar[list[tuple[str, dict]]] = [
        (BOOSTING, {"points": THREE_READINGS, "rounds": 5, "learning_rate": 0.5}),
        (BOOSTING, {"points": THREE_READINGS, "rounds": 5, "learning_rate": 1.5}),
        (MODEL, {"points": THREE_READINGS, "rounds": 5, "learning_rate": 0.5}),
        (MODEL, {"points": THREE_READINGS, "rounds": 5, "learning_rate": 1.5}),
        (MODEL, {"points": CONSTANT_HEIGHTS, "rounds": 5, "learning_rate": 0.5}),
        (ROUNDS, {"points": THREE_READINGS, "rounds": 5, "learning_rate": 0.5}),
        (ROUNDS, {"points": THREE_READINGS, "rounds": 5, "learning_rate": 1.5}),
        (
            CURVES,
            {
                "points": SIXTY_THROW,
                "max_rounds": 5,
                "configurations": [{"learning_rate": 0.3, "max_depth": 1}],
            },
        ),
        (VERSUS, {"points": SIXTY_THROW, "n_members": 5, "learning_rate": 0.3}),
    ]

    def test_no_answered_string_names_the_code(self) -> None:
        for path, request in self.CALLS:
            answered = client.post(path, json=request).json()
            for sentence in _every_string(answered):
                for name in self.FORBIDDEN:
                    assert name not in sentence, f"{path}: {sentence}"
