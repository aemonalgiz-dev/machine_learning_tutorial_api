"""The principal component analysis page's endpoints, on the worked four."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

WORKED = [
    {"x": 180, "y": 78},
    {"x": 160, "y": 58},
    {"x": 175, "y": 63},
    {"x": 165, "y": 73},
]
CROWD = [
    {"x": 147, "y": 41},
    {"x": 156, "y": 53},
    {"x": 145, "y": 57},
    {"x": 159, "y": 57},
    {"x": 162, "y": 61},
    {"x": 120, "y": 25},
    {"x": 122, "y": 28},
    {"x": 118, "y": 24},
    {"x": 180, "y": 80},
    {"x": 183, "y": 83},
    {"x": 178, "y": 78},
]
ANALYZE = "/concepts/pca/analyze"
DIRECTION = "/concepts/pca/direction"
UNITS = "/concepts/pca/unit-scaling"
VERSUS = "/concepts/pca/versus-regression"
BODY = "/concepts/pca/body-measurements"


class TestAnalyze:
    def test_the_worked_four_come_out_clean(self) -> None:
        body = client.post(ANALYZE, json={"points": WORKED}).json()

        assert body["mean"] == {"x": 170.0, "y": 68.0}
        assert body["scatter"] == [[250.0, 150.0], [150.0, 250.0]]
        assert body["covariance"] == [[62.5, 37.5], [37.5, 62.5]]
        assert body["scatter_eigenvalues"] == pytest.approx([400.0, 100.0])
        assert [c["share"] for c in body["components"]] == pytest.approx([0.8, 0.2])
        assert body["cumulative_shares"] == pytest.approx([0.8, 1.0])
        assert body["deviations"][0] == {"x": 10.0, "y": 10.0}

    def test_the_first_person_lies_entirely_along_the_first_component(self) -> None:
        body = client.post(ANALYZE, json={"points": WORKED}).json()

        first = body["scores"][0]
        assert abs(first["first"]) == pytest.approx(10 * math.sqrt(2))
        assert first["second"] == pytest.approx(0.0, abs=1e-9)

    def test_the_reconstruction_error_is_the_discarded_scatter(self) -> None:
        body = client.post(ANALYZE, json={"points": WORKED}).json()

        assert body["reconstruction_error_one"] == pytest.approx(100.0)

    def test_standardising_changes_the_directions_on_the_crowd(self) -> None:
        body = client.post(ANALYZE, json={"points": CROWD}).json()

        raw = body["components"][0]
        standardised = body["standardized"]["components"][0]
        assert abs(standardised["dx"]) == pytest.approx(
            abs(standardised["dy"]), abs=1e-9
        )
        assert abs(raw["dx"]) != pytest.approx(abs(raw["dy"]), abs=1e-3)
        assert 0 < body["standardized"]["correlation"] <= 1


class TestDirection:
    def test_the_component_angle_reaches_the_component_variance(self) -> None:
        analysis = client.post(ANALYZE, json={"points": WORKED}).json()
        first = analysis["components"][0]
        angle = math.degrees(math.atan2(first["dy"], first["dx"])) % 180

        body = client.post(
            DIRECTION, json={"points": WORKED, "angle_degrees": angle}
        ).json()

        assert body["score_second_moment"] == pytest.approx(100.0)
        assert body["retained_sum_squares"] == pytest.approx(400.0)
        assert body["total_squared_residual"] == pytest.approx(100.0)
        assert body["angle_between_degrees"] == pytest.approx(0.0, abs=1e-6)
        assert body["transformed_length"] == pytest.approx(100.0)

    def test_a_longer_arrow_inflates_the_scores_and_not_the_direction(self) -> None:
        unit = client.post(
            DIRECTION, json={"points": WORKED, "angle_degrees": 45.0}
        ).json()
        doubled = client.post(
            DIRECTION, json={"points": WORKED, "angle_degrees": 45.0, "length": 2.0}
        ).json()

        assert doubled["score_variance"] == pytest.approx(4 * unit["score_variance"])
        assert doubled["unit_score_variance"] == pytest.approx(
            unit["unit_score_variance"]
        )
        assert doubled["unit"] == unit["unit"]

    def test_an_uncentred_projection_confuses_location_with_spread(self) -> None:
        centred = client.post(
            DIRECTION, json={"points": WORKED, "angle_degrees": 45.0}
        ).json()
        raw = client.post(
            DIRECTION, json={"points": WORKED, "angle_degrees": 45.0, "centre": False}
        ).json()

        assert centred["score_mean"] == pytest.approx(0.0, abs=1e-9)
        assert centred["score_second_moment"] == pytest.approx(
            centred["score_variance"]
        )
        assert raw["score_second_moment"] > 100 * raw["score_variance"]
        assert raw["pivot"] == {"x": 0.0, "y": 0.0}

    def test_the_covariance_matrix_turns_a_direction_that_is_not_its_own(self) -> None:
        body = client.post(
            DIRECTION, json={"points": WORKED, "angle_degrees": 0.0}
        ).json()

        assert body["angle_between_degrees"] > 20
        assert body["covariance"] == {"var_x": 62.5, "cov": 37.5, "var_y": 62.5}


class TestUnitScaling:
    def test_rewriting_height_in_millimetres_turns_the_first_component(self) -> None:
        body = client.post(UNITS, json={"points": CROWD}).json()

        by_unit = {variant["unit"]: variant for variant in body["variants"]}
        assert set(by_unit) == {"metres", "centimetres", "millimetres"}
        millimetres = by_unit["millimetres"]["components"][0]
        metres = by_unit["metres"]["components"][0]
        assert abs(millimetres["dx"]) > 0.99
        assert abs(metres["dy"]) > 0.99
        assert millimetres["share"] > by_unit["centimetres"]["components"][0]["share"]


class TestVersusRegression:
    def test_three_lines_and_three_residual_totals(self) -> None:
        body = client.post(VERSUS, json={"points": CROWD}).json()

        forward = body["weight_on_height"]
        backward = body["height_on_weight"]
        first = body["first_component"]
        component_slope = first["dy"] / first["dx"]
        assert forward["slope"] < component_slope < 1 / backward["slope"]
        assert body["perpendicular_residual_sum"] < forward["residual_sum"]
        assert body["perpendicular_residual_sum"] < backward["residual_sum"]


class TestBodyMeasurements:
    def test_five_components_that_account_for_everything(self) -> None:
        body = client.get(BODY).json()

        assert body["names"] == [
            "height",
            "arm_span",
            "leg_length",
            "sitting_height",
            "shoe_size",
        ]
        assert len(body["rows"]) == 60
        assert body["components"][-1]["cumulative_share"] == pytest.approx(1.0)
        assert body["components"][0]["share"] > 0.6
        assert body["reconstruction_errors"] == sorted(
            body["reconstruction_errors"], reverse=True
        )
        assert body["reconstruction_errors"][-1] == pytest.approx(0.0, abs=1e-9)
        assert body["correlation"][0][1] > 0.9
