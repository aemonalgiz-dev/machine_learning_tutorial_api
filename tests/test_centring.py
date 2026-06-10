"""The centring on the mean page's endpoints, pinned on the page's own people.

Every number the page quotes is asserted here, so a change in the library that
moves one of them fails a test before it reaches the prose.
"""

from __future__ import annotations

import math
import re

import pytest
from fastapi.testclient import TestClient
from oop_ml.core.exceptions import (
    AllSameValuesError,
    EmptyValuesError,
    InvalidValuesError,
    MLLibError,
)

from oop_ml_api.concepts.centring import _in_plain_words
from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/centring-on-the-mean"
CROWD = [
    {"x": height, "y": weight}
    for height, weight in zip(
        [147, 156, 145, 159, 162, 120, 122, 118, 180, 183, 178],
        [41, 53, 57, 57, 61, 25, 28, 24, 80, 83, 78],
        strict=True,
    )
]
MEASURED_FOUR = [
    {"x": 180, "y": 78},
    {"x": 160, "y": 58},
    {"x": 175, "y": 63},
    {"x": 165, "y": 73},
]
IDEAL_CASE = [
    {"x": height, "y": weight}
    for height, weight in zip(
        range(130, 191, 4),
        [36, 38, 44, 44, 49, 54, 56, 61, 62, 69, 71, 74, 79, 81, 87, 88],
        strict=True,
    )
]

IDENTIFIER = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
FORBIDDEN = ("oop_ml", "__", "self.", "Error", "the library", "the API")


@pytest.fixture(scope="module")
def crowd() -> dict:
    response = client.get(f"{BASE}/crowd")
    assert response.status_code == 200
    return response.json()


def view(points: list[dict], height_origin: float = 0.0, weight_origin: float = 0.0):
    return client.post(
        f"{BASE}/origin",
        json={
            "points": points,
            "height_origin": height_origin,
            "weight_origin": weight_origin,
        },
    )


class TestTheCentringItself:
    def test_the_centres_are_the_crowd_s_means(self, crowd: dict) -> None:
        centring = crowd["centring"]
        assert centring["height_centre"] == pytest.approx(1670 / 11)
        assert centring["weight_centre"] == pytest.approx(587 / 11)
        assert centring["height_spread"] == 1.0

    def test_the_centred_columns_are_the_page_s_table(self, crowd: dict) -> None:
        centring = crowd["centring"]
        assert centring["centred_heights"][0] == pytest.approx(-4.8182, abs=1e-4)
        assert centring["centred_heights"][9] == pytest.approx(31.1818, abs=1e-4)
        assert centring["centred_weights"][0] == pytest.approx(-12.3636, abs=1e-4)
        assert centring["centred_weights"][9] == pytest.approx(29.6364, abs=1e-4)

    def test_the_mean_is_zero_to_the_last_few_bits(self, crowd: dict) -> None:
        centring = crowd["centring"]
        assert abs(centring["height_mean_after"]) < 1e-13
        assert abs(centring["weight_mean_after"]) < 1e-13
        assert centring["height_mean_after"] == pytest.approx(5.17e-15, rel=0.01)

    def test_gaps_and_spread_are_kept(self, crowd: dict) -> None:
        centring = crowd["centring"]
        assert centring["largest_gap_change"] == 0.0
        assert centring["height_variance_before"] == pytest.approx(521.7851, abs=1e-4)
        assert centring["height_variance_after"] == pytest.approx(
            centring["height_variance_before"]
        )
        assert centring["standardized_gap"] < 1e-15


class TestTheOrigin:
    def test_the_measured_four_split_in_whole_numbers(self) -> None:
        body = view(MEASURED_FOUR).json()
        assert body["centre"] == {"x": 170.0, "y": 68.0}
        assert body["second_moment_total"] == pytest.approx(33_649.0)
        assert body["variance_total"] == pytest.approx(125.0)
        assert body["location_total"] == pytest.approx(33_524.0)
        assert body["centred"]["arrow"]["angle_degrees"] == pytest.approx(45.0)
        assert body["centred"]["share"] == pytest.approx(0.8)
        assert body["uncentred"]["arrow"]["angle_degrees"] == pytest.approx(
            21.8477, abs=1e-4
        )
        assert body["mean_direction"]["angle_degrees"] == pytest.approx(
            21.8014, abs=1e-4
        )

    def test_the_measured_four_line_both_ways(self) -> None:
        body = view(MEASURED_FOUR).json()
        assert body["with_intercept"]["slope"] == pytest.approx(0.6)
        assert body["with_intercept"]["intercept"] == pytest.approx(-34.0)
        assert body["with_intercept"]["r_squared"] == pytest.approx(0.36)
        assert body["through_origin"]["slope"] == pytest.approx(0.4004, abs=1e-4)
        assert body["centred_through_origin"]["slope"] == pytest.approx(0.6)

    def test_on_the_crowd_the_uncentred_direction_points_at_the_mean(self) -> None:
        body = view(CROWD).json()
        assert body["uncentred"]["arrow"]["angle_degrees"] == pytest.approx(
            20.0869, abs=1e-4
        )
        assert body["mean_direction"]["angle_degrees"] == pytest.approx(
            19.3664, abs=1e-4
        )
        assert body["uncentred_against_mean"] == pytest.approx(0.99992, abs=1e-5)
        assert body["centred"]["arrow"]["angle_degrees"] == pytest.approx(
            42.2132, abs=1e-4
        )
        assert body["uncentred_against_centred"] == pytest.approx(0.9264, abs=1e-4)
        assert body["uncentred"]["share"] == pytest.approx(0.9946, abs=1e-4)
        assert body["centred"]["share"] == pytest.approx(0.9897, abs=1e-4)

    def test_the_squared_length_splits_into_location_and_spread(self) -> None:
        body = view(CROWD).json()
        assert body["second_moment_total"] == pytest.approx(26_849.36, abs=0.01)
        assert body["variance_total"] == pytest.approx(952.93, abs=0.01)
        assert body["location_total"] == pytest.approx(25_896.44, abs=0.01)
        assert body["second_moment_total"] == pytest.approx(
            body["variance_total"] + body["location_total"]
        )
        assert body["refit_gap"] == 0.0

    def test_the_crowd_line_through_zero(self) -> None:
        body = view(CROWD).json()
        assert body["through_origin"]["slope"] == pytest.approx(0.3634, abs=1e-4)
        assert body["through_origin"]["r_squared"] == pytest.approx(0.6155, abs=1e-4)
        assert body["with_intercept"]["slope"] == pytest.approx(0.8901, abs=1e-4)
        assert body["centred_through_origin"]["slope"] == pytest.approx(
            body["with_intercept"]["slope"], abs=1e-12
        )

    def test_an_origin_at_the_average_person_has_no_arrow_to_it(self) -> None:
        body = view(MEASURED_FOUR, 170.0, 68.0).json()
        assert body["mean_direction"] is None
        assert body["uncentred_against_mean"] is None
        assert body["location_total"] == 0.0
        assert body["uncentred_against_centred"] == pytest.approx(1.0)

    def test_the_ideal_case_is_a_strong_fit_short_of_perfect(self) -> None:
        body = view(IDEAL_CASE).json()
        assert 0.99 < body["with_intercept"]["r_squared"] < 1.0
        assert body["through_origin"]["r_squared"] < 0.7


class TestTheSweep:
    def test_the_mark_turns_only_the_uncentred_direction(self, crowd: dict) -> None:
        sweep = {row["height_origin"]: row for row in crowd["sweep"]}
        assert [round(sweep[mark]["uncentred_angle"], 2) for mark in sweep] == [
            20.09,
            28.59,
            45.32,
            54.79,
            78.91,
        ]
        assert len({row["centred_angle"] for row in crowd["sweep"]}) == 1
        assert sweep[150.0]["uncentred_share"] == pytest.approx(0.8909, abs=1e-4)


class TestTheLines:
    def test_an_intercept_is_what_centring_does(self, crowd: dict) -> None:
        lines = crowd["lines"]
        assert lines["least_squares"]["intercept"] == pytest.approx(-81.7665, abs=1e-4)
        assert lines["centred_with_intercept"]["intercept"] == pytest.approx(587 / 11)
        assert lines["centred_through_origin"]["slope"] == pytest.approx(
            lines["least_squares"]["slope"], abs=1e-12
        )
        assert lines["centred_through_origin"]["r_squared"] == pytest.approx(
            lines["least_squares"]["r_squared"]
        )

    def test_an_exempt_intercept_is_a_centred_fit_through_zero(
        self, crowd: dict
    ) -> None:
        for ridge in crowd["lines"]["ridges"]:
            assert ridge["slope_centred_through_origin"] == pytest.approx(
                ridge["slope_with_intercept"], abs=1e-12
            )
            assert ridge["slope_raw_through_origin"] == pytest.approx(0.363, abs=1e-3)
        penalties = {ridge["penalty"]: ridge for ridge in crowd["lines"]["ridges"]}
        assert penalties[1.0]["slope_with_intercept"] == pytest.approx(
            0.88992, abs=1e-5
        )
        assert penalties[100.0]["slope_with_intercept"] == pytest.approx(
            0.87484, abs=1e-5
        )


class TestTheWalks:
    def test_the_raw_walk_stops_at_its_cap_unfinished(self, crowd: dict) -> None:
        raw = next(walk for walk in crowd["walks"] if walk["reading"] == "raw")
        assert raw["passes"] == 20_000
        assert raw["converged"] is False
        assert raw["condition_number"] == pytest.approx(1_064_838, rel=1e-5)
        assert raw["ones_cosine"] == pytest.approx(0.9889, abs=1e-4)
        assert raw["level"] == pytest.approx(-1.519, abs=1e-3)
        assert raw["slope"] == pytest.approx(0.3732, abs=1e-4)

    def test_the_first_raw_pass_finds_the_line_through_zero(self, crowd: dict) -> None:
        raw = next(walk for walk in crowd["walks"] if walk["reading"] == "raw")
        first = raw["checkpoints"][0]
        assert first["passes"] == 1
        assert first["slope"] == pytest.approx(
            crowd["lines"]["through_origin"]["slope"], abs=1e-4
        )

    def test_the_centred_walk_matches_the_feature_scaling_page(
        self, crowd: dict
    ) -> None:
        centred = next(walk for walk in crowd["walks"] if walk["reading"] == "centred")
        assert centred["passes"] == 8415
        assert centred["converged"] is True
        assert centred["condition_number"] == pytest.approx(521.7851, abs=1e-4)
        assert abs(centred["ones_cosine"]) < 1e-15
        assert centred["checkpoints"][0]["slope"] == pytest.approx(
            centred["target_slope"], abs=1e-12
        )
        by_passes = {point["passes"]: point for point in centred["checkpoints"]}
        assert by_passes[1000]["level"] == pytest.approx(45.53, abs=0.01)
        assert by_passes[2000]["level"] == pytest.approx(52.21, abs=0.01)


class TestThePolynomial:
    def test_centring_removes_the_correlation_with_the_square(
        self, crowd: dict
    ) -> None:
        raw, centred = crowd["polynomial"]
        assert raw["correlation"] == pytest.approx(0.9980, abs=1e-4)
        assert centred["correlation"] == pytest.approx(-0.2290, abs=1e-4)
        assert raw["condition_number"] > 1000 * centred["condition_number"]
        assert raw["square_coefficient"] == pytest.approx(
            centred["square_coefficient"], rel=1e-9
        )
        assert raw["r_squared"] == pytest.approx(centred["r_squared"])
        assert centred["linear_coefficient"] == pytest.approx(0.9024, abs=1e-4)


class TestTheAngles:
    def test_from_zero_every_pair_looks_alike(self, crowd: dict) -> None:
        angles = crowd["angles"]
        assert angles["smallest_raw"]["first"] == 7
        assert angles["smallest_raw"]["second"] == 9
        assert angles["smallest_raw"]["value"] == pytest.approx(0.97476, abs=1e-5)
        assert angles["centred_cosines"][7][9] == pytest.approx(-0.9990, abs=1e-4)
        assert angles["smallest_centred"]["value"] == pytest.approx(-0.99964, abs=1e-5)

    def test_a_correlation_is_a_centred_cosine(self, crowd: dict) -> None:
        angles = crowd["angles"]
        assert angles["column_cosine_centred"] == pytest.approx(
            angles["correlation"], abs=1e-12
        )
        assert angles["column_cosine_raw"] == pytest.approx(0.97439, abs=1e-5)

    def test_the_votes(self, crowd: dict) -> None:
        votes = {vote["rule"]: vote for vote in crowd["angles"]["votes"]}
        assert votes["straight-line"]["raw_accuracy"] == pytest.approx(9 / 11)
        assert votes["straight-line"]["centred_accuracy"] == pytest.approx(9 / 11)
        assert votes["angle"]["raw_accuracy"] == pytest.approx(10 / 11)
        assert votes["angle"]["centred_accuracy"] == 1.0


class TestTheMethodsThatNeverNotice:
    def test_every_one_agrees(self, crowd: dict) -> None:
        invariance = crowd["invariance"]
        assert invariance["distance_gap"] < 1e-12
        assert invariance["kmeans_inertia_raw"] == pytest.approx(2952.607, abs=1e-3)
        assert invariance["kmeans_inertia_centred"] == pytest.approx(
            invariance["kmeans_inertia_raw"]
        )
        assert invariance["kmeans_same_groups"] is True
        assert invariance["tree_threshold_raw"] == 151.5
        assert invariance["tree_threshold_centred"] == pytest.approx(151.5 - 1670 / 11)
        assert invariance["tree_predictions_agree"] is True
        assert invariance["slope_gap"] < 1e-14
        assert invariance["standardized_gap"] < 1e-15


class TestTheFold:
    def test_held_out_rows_keep_the_training_zero(self, crowd: dict) -> None:
        fold = crowd["fold"]
        assert fold["training_mean"] == pytest.approx(141.125)
        assert fold["by_training"] == pytest.approx([38.875, 41.875, 36.875])
        assert fold["by_training_mean"] == pytest.approx(39.2083, abs=1e-4)
        assert fold["own_mean"] == pytest.approx(541 / 3)
        assert fold["by_own"] == pytest.approx([-1 / 3, 8 / 3, -7 / 3])


class TestTheClock:
    def test_two_bedtimes_either_side_of_midnight(self) -> None:
        body = client.post(f"{BASE}/clock", json={"hours": [23, 1]}).json()
        assert body["arithmetic_centre"] == 12.0
        assert body["arithmetic_offsets"] == [11.0, -11.0]
        assert body["circular_centre"] == 0.0
        assert body["circular_offsets"] == pytest.approx([-1.0, 1.0])
        assert body["resultant_length"] == pytest.approx(math.cos(math.pi / 12))

    def test_a_week_of_bedtimes(self) -> None:
        body = client.post(
            f"{BASE}/clock", json={"hours": [22.5, 23, 23.5, 0, 0.5, 1, 23]}
        ).json()
        assert body["arithmetic_centre"] == pytest.approx(13.357, abs=1e-3)
        assert body["circular_centre"] == pytest.approx(23.641, abs=1e-3)
        assert body["resultant_length"] == pytest.approx(0.9764, abs=1e-4)

    def test_opposite_times_have_no_circular_mean(self) -> None:
        body = client.post(f"{BASE}/clock", json={"hours": [6, 18]}).json()
        assert body["arithmetic_centre"] == 12.0
        assert body["circular_centre"] is None
        assert body["circular_offsets"] is None
        assert body["resultant_length"] < 1e-12


class TestRefusals:
    def test_a_constant_height_column_is_refused_in_plain_words(self) -> None:
        flat = [{"x": 150, "y": weight} for weight in (40, 50, 60)]
        response = view(flat)
        assert response.status_code == 422
        assert "every height" in response.json()["detail"]

    def test_a_constant_weight_column_is_refused_in_plain_words(self) -> None:
        flat = [{"x": height, "y": 50} for height in (140, 150, 160)]
        response = view(flat)
        assert response.status_code == 422
        assert "every weight" in response.json()["detail"]

    def test_a_time_outside_the_day_is_refused(self) -> None:
        response = client.post(f"{BASE}/clock", json={"hours": [25]})
        assert response.status_code == 422

    def test_unknown_fields_and_short_lists_are_refused(self) -> None:
        assert (
            client.post(
                f"{BASE}/origin", json={"points": CROWD, "centre": True}
            ).status_code
            == 422
        )
        assert view(CROWD[:2]).status_code == 422
        assert client.post(f"{BASE}/clock", json={"hours": []}).status_code == 422

    def test_no_refusal_names_anything_from_the_code(self) -> None:
        """The page shows these sentences, so no identifier may survive them."""
        sentences = [
            view([{"x": 150, "y": weight} for weight in (40, 50, 60)]).json()["detail"],
            view([{"x": height, "y": 50} for height in (140, 150, 160)]).json()[
                "detail"
            ],
            client.post(f"{BASE}/clock", json={"hours": [25]}).json()["detail"],
        ]
        refusals: list[MLLibError] = [
            EmptyValuesError("feature_values must not be empty"),
            AllSameValuesError("feature_values must not be constant (zero variance)"),
            InvalidValuesError("total variance must be positive; got 0.0"),
            MLLibError("NotFittedError raised inside some_helper"),
        ]
        sentences.extend(_in_plain_words(refusal) for refusal in refusals)

        for sentence in sentences:
            assert IDENTIFIER.search(sentence) is None, sentence
            for word in FORBIDDEN:
                assert word not in sentence, sentence
