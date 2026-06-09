"""The kernel trick endpoints, pinned to the page's clinic story.

The healthy patients sit in a band of ordinary vitals with the unwell on
every side of them, so a straight boundary cannot cut them out. The page
quotes the linear kernel at 0.708, the squared and radial kernels at 1.000,
the two routes to the squared kernel agreeing to rounding, a sigmoid table
with thirteen negative eigenvalues, a corridor 1.348 wide, a gamma sweep
that memorises past 30, and an absorbed offset released by capacity. Every
number the page quotes is pinned here.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/kernel-trick"
CLASSIFY = f"{BASE}/classify"
LIFT = f"{BASE}/lift"
IDENTITY = f"{BASE}/identity"
GRAM = f"{BASE}/gram"
MARGIN = f"{BASE}/margin"
GAMMA_SWEEP = f"{BASE}/gamma-sweep"
CAPACITY_SWEEP = f"{BASE}/capacity-sweep"
SUPPORT_ONLY = f"{BASE}/support-only"
INTERCEPT_COST = f"{BASE}/intercept-cost"
SIMILARITY = f"{BASE}/similarity-curve"

HEALTHY = [
    {"x": 36.6, "y": 68, "label": 1},
    {"x": 36.8, "y": 74, "label": 1},
    {"x": 37.0, "y": 70, "label": 1},
    {"x": 37.2, "y": 78, "label": 1},
    {"x": 36.9, "y": 64, "label": 1},
    {"x": 37.1, "y": 84, "label": 1},
    {"x": 36.7, "y": 80, "label": 1},
    {"x": 37.3, "y": 72, "label": 1},
    {"x": 36.5, "y": 76, "label": 1},
    {"x": 37.0, "y": 88, "label": 1},
]

UNWELL = [
    {"x": 35.2, "y": 48, "label": 0},
    {"x": 35.5, "y": 120, "label": 0},
    {"x": 36.0, "y": 130, "label": 0},
    {"x": 38.9, "y": 132, "label": 0},
    {"x": 39.5, "y": 120, "label": 0},
    {"x": 40.2, "y": 110, "label": 0},
    {"x": 39.8, "y": 66, "label": 0},
    {"x": 40.5, "y": 90, "label": 0},
    {"x": 35.4, "y": 90, "label": 0},
    {"x": 38.8, "y": 50, "label": 0},
    {"x": 35.8, "y": 58, "label": 0},
    {"x": 39.9, "y": 140, "label": 0},
    {"x": 35.1, "y": 72, "label": 0},
    {"x": 38.5, "y": 44, "label": 0},
]

CLINIC = HEALTHY + UNWELL

FEVER_ONLY = HEALTHY + [
    {"x": 38.9, "y": 132, "label": 0},
    {"x": 39.5, "y": 120, "label": 0},
    {"x": 40.2, "y": 110, "label": 0},
    {"x": 39.8, "y": 96, "label": 0},
    {"x": 40.5, "y": 90, "label": 0},
    {"x": 39.2, "y": 118, "label": 0},
    {"x": 39.9, "y": 140, "label": 0},
    {"x": 38.6, "y": 104, "label": 0},
]

LINEAR = {"name": "linear"}
SQUARED = {"name": "polynomial", "degree": 2, "constant": 1.0}
RADIAL = {"name": "rbf", "gamma": 1.0}


def classify(points: list[dict], kernel: dict, capacity: float = 1.0) -> dict:
    response = client.post(
        CLASSIFY, json={"points": points, "kernel": kernel, "capacity": capacity}
    )
    assert response.status_code == 200, response.json()
    return response.json()


class TestTheClinic:
    """Part 1: what a straight cut gives up, and where it is enough."""

    def test_a_straight_boundary_gets_seventeen_of_twenty_four(self) -> None:
        fit = classify(CLINIC, LINEAR)["fit"]

        assert fit["accuracy"] == pytest.approx(17 / 24)
        assert fit["n_support_vectors"] == 20
        assert fit["at_the_cap"] == 19
        assert fit["converged"] is True
        assert fit["epochs_run"] == 226

    def test_the_squared_kernel_separates_the_clinic(self) -> None:
        fit = classify(CLINIC, SQUARED)["fit"]

        assert fit["accuracy"] == 1.0
        assert fit["n_support_vectors"] == 6
        assert fit["at_the_cap"] == 3
        assert fit["epochs_run"] == 812

    def test_the_radial_kernel_separates_the_clinic(self) -> None:
        body = classify(CLINIC, RADIAL)
        fit = body["fit"]

        assert fit["accuracy"] == 1.0
        assert fit["n_support_vectors"] == 16
        assert fit["support_share"] == pytest.approx(16 / 24)
        assert fit["at_the_cap"] == 4
        assert fit["epochs_run"] == 5768
        assert body["kernel_description"] == "exp(-1.0 ||a - b|| ** 2)"

    def test_the_radial_regions_close_around_the_healthy(self) -> None:
        grid = classify(CLINIC, RADIAL)["regions"]["labels"]

        edge_cells = (
            [row[0] for row in grid] + [row[-1] for row in grid] + grid[0] + grid[-1]
        )
        assert set(edge_cells) == {0}
        assert any(1 in row for row in grid)

    def test_the_fever_only_clinic_needs_no_kernel(self) -> None:
        linear = classify(FEVER_ONLY, LINEAR)["fit"]
        squared = classify(FEVER_ONLY, SQUARED)["fit"]
        radial = classify(FEVER_ONLY, RADIAL)["fit"]

        assert linear["accuracy"] == squared["accuracy"] == radial["accuracy"] == 1.0
        assert linear["n_support_vectors"] == 2
        assert radial["n_support_vectors"] == 7

    def test_the_support_vectors_sit_at_the_corridor_edges(self) -> None:
        fit = classify(CLINIC, RADIAL)["fit"]
        decisions = fit["decision_values"]

        on_an_edge = [
            position
            for position in fit["support_positions"]
            if abs(abs(decisions[position]) - 1.0) < 1e-3
        ]
        assert len(on_an_edge) >= 12
        # The capped hypothermic patient, row 20, is inside the corridor.
        assert decisions[20] == pytest.approx(-0.3955, abs=1e-3)
        assert fit["squashed"][20] == pytest.approx(0.4024, abs=1e-3)
        assert max(fit["squashed"]) == pytest.approx(0.7935, abs=1e-3)
        assert max(decisions) == pytest.approx(1.3461, abs=1e-3)


class TestTheLift:
    """Part 2: the worked map by hand, and the two routes agreeing."""

    def test_the_plane_separates_the_lifted_clinic(self) -> None:
        body = client.post(LIFT, json={"points": CLINIC}).json()

        assert body["lifted"]["accuracy"] == 1.0
        assert body["lifted"]["n_support_vectors"] == 8
        assert body["kernel"]["n_support_vectors"] == 8
        assert body["lifted"]["epochs_run"] == body["kernel"]["epochs_run"] == 966
        assert len(body["points"]) == 24

    def test_the_two_routes_agree_to_rounding(self) -> None:
        body = client.post(LIFT, json={"points": CLINIC}).json()

        assert body["multiplier_gap"] < 1e-14
        assert body["decision_gap"] < 1e-13

    def test_the_plane_normal_has_unit_length(self) -> None:
        plane = client.post(LIFT, json={"points": CLINIC}).json()["plane"]

        assert plane["a"] ** 2 + plane["b"] ** 2 + plane["c"] ** 2 == pytest.approx(1.0)

    def test_every_lifted_point_is_squares_and_a_cross_term(self) -> None:
        body = client.post(LIFT, json={"points": CLINIC}).json()

        for point in body["points"]:
            assert point["u"] >= 0.0
            assert point["w"] >= 0.0
            assert point["v"] ** 2 == pytest.approx(2.0 * point["u"] * point["w"])


class TestTheIdentity:
    """Part 4: the 121 check and the column counts."""

    def test_both_routes_give_121(self) -> None:
        body = client.post(IDENTITY, json={"a": [1, 2], "b": [3, 4]}).json()

        assert body["dot_product"] == 11.0
        assert body["lifted_product"] == 121.0
        assert body["kernel_value"] == 121.0
        assert body["phi_a"] == pytest.approx([1.0, 2.8284271247, 4.0])
        assert body["phi_b"] == pytest.approx([9.0, 16.9705627485, 16.0])

    def test_the_identity_holds_on_another_pair(self) -> None:
        body = client.post(IDENTITY, json={"a": [0.5, -3], "b": [2, 1.5]}).json()

        assert body["lifted_product"] == pytest.approx(body["kernel_value"])
        assert body["kernel_value"] == pytest.approx((0.5 * 2 - 3 * 1.5) ** 2)

    def test_the_column_counts_the_page_quotes(self) -> None:
        counts = client.post(IDENTITY, json={"a": [1, 2], "b": [3, 4]}).json()[
            "column_counts"
        ]
        by_case = {(row["n_features"], row["degree"]): row for row in counts}

        assert by_case[(2, 2)]["with_lower_terms"] == 6
        assert by_case[(2, 2)]["pure"] == 3
        assert by_case[(100, 2)]["with_lower_terms"] == 5151
        assert by_case[(20, 5)]["with_lower_terms"] == 53130
        assert by_case[(20, 10)]["with_lower_terms"] == 30045015

    def test_a_three_vector_is_refused(self) -> None:
        response = client.post(IDENTITY, json={"a": [1, 2, 3], "b": [3, 4]})

        assert response.status_code == 422

    def test_a_non_finite_coordinate_is_refused(self) -> None:
        raw = json.dumps({"a": [1, 2], "b": ["NAN", 4]}).replace('"NAN"', "NaN")
        response = client.post(
            IDENTITY, content=raw, headers={"content-type": "application/json"}
        )

        assert response.status_code == 422
        assert "finite" in response.json()["detail"]


class TestTheGramMatrix:
    """Part 4: the classifier's whole view, and Mercer's condition."""

    def test_the_radial_table_has_a_unit_diagonal_and_no_negative_eigenvalue(
        self,
    ) -> None:
        body = client.post(GRAM, json={"points": CLINIC, "kernel": RADIAL}).json()

        assert len(body["values"]) == 24
        assert all(body["values"][i][i] == pytest.approx(1.0) for i in range(24))
        assert body["symmetric"] is True
        assert body["n_negative_eigenvalues"] == 0
        assert body["smallest_eigenvalue"] >= -1e-9
        assert body["largest_eigenvalue"] == pytest.approx(9.0739, abs=1e-3)
        assert body["ridge_accepts"] is True

    def test_the_linear_table_eigenvalues(self) -> None:
        body = client.post(GRAM, json={"points": CLINIC, "kernel": LINEAR}).json()

        assert body["largest_eigenvalue"] == pytest.approx(30.6468, abs=1e-3)
        assert body["n_negative_eigenvalues"] == 0
        assert body["eigenvalues"] == sorted(body["eigenvalues"], reverse=True)

    def test_a_sigmoid_with_a_negative_constant_is_not_a_kernel(self) -> None:
        body = client.post(
            GRAM,
            json={
                "points": CLINIC,
                "kernel": {"name": "sigmoid", "gamma": 1.0, "constant": -1.0},
            },
        ).json()

        assert body["symmetric"] is True
        assert body["n_negative_eigenvalues"] == 13
        assert body["smallest_eigenvalue"] == pytest.approx(-13.3714, abs=1e-3)
        assert body["ridge_accepts"] is False
        assert "Mercer" in body["ridge_refusal"]
        # The classifier's ascent does not refuse it; it runs and reports.
        assert body["classifier"]["accuracy"] == pytest.approx(19 / 24)
        assert body["classifier"]["n_support_vectors"] == 18


class TestTheMargin:
    """Part 3 and Part 6: the corridor, its width, and the price of a violation."""

    def test_the_hard_margin_at_capacity_ten(self) -> None:
        body = client.post(MARGIN, json={"points": FEVER_ONLY, "capacity": 10.0}).json()

        assert body["width"] == pytest.approx(1.3483, abs=1e-3)
        assert body["weights"] == pytest.approx([-1.2422, -0.8108], abs=1e-3)
        assert body["offset"] == pytest.approx(-0.1336, abs=1e-3)
        assert body["fit"]["n_support_vectors"] == 2
        assert body["fit"]["at_the_cap"] == 0
        assert body["fit"]["epochs_run"] == 1165
        decisions = body["fit"]["decision_values"]
        assert sorted(
            decisions[position] for position in body["fit"]["support_positions"]
        ) == pytest.approx([-1.0, 1.0], abs=1e-3)

    def test_the_width_is_two_over_the_weight_length(self) -> None:
        body = client.post(MARGIN, json={"points": FEVER_ONLY, "capacity": 10.0}).json()
        length = sum(weight**2 for weight in body["weights"]) ** 0.5

        assert body["width"] == pytest.approx(2.0 / length)

    def test_a_small_capacity_widens_the_corridor_and_caps_patients(self) -> None:
        body = client.post(MARGIN, json={"points": FEVER_ONLY, "capacity": 0.1}).json()

        assert body["width"] == pytest.approx(2.5383, abs=1e-3)
        assert body["fit"]["n_support_vectors"] == 9
        assert body["fit"]["at_the_cap"] == 7

    def test_at_capacity_one_the_absorbed_offset_caps_one_patient(self) -> None:
        body = client.post(MARGIN, json={"points": FEVER_ONLY, "capacity": 1.0}).json()

        assert body["width"] == pytest.approx(1.5029, abs=1e-3)
        assert body["fit"]["n_support_vectors"] == 2
        assert body["fit"]["at_the_cap"] == 1
        decisions = body["fit"]["decision_values"]
        assert min(
            decisions[position] for position in body["fit"]["support_positions"]
        ) == pytest.approx(-0.7946, abs=1e-3)

    def test_the_margin_edges_are_parallel_to_the_boundary(self) -> None:
        body = client.post(MARGIN, json={"points": FEVER_ONLY, "capacity": 10.0}).json()

        def slope(segment: dict) -> float:
            return (segment["y2"] - segment["y1"]) / (segment["x2"] - segment["x1"])

        assert slope(body["upper_edge"]) == pytest.approx(slope(body["boundary"]))
        assert slope(body["lower_edge"]) == pytest.approx(slope(body["boundary"]))


class TestTheSweeps:
    """Part 5 and Part 6: gamma from too wide to memorising, and capacity."""

    def test_the_gamma_sweep_the_page_quotes(self) -> None:
        body = client.post(GAMMA_SWEEP, json={"points": CLINIC}).json()
        by_gamma = {entry["value"]: entry for entry in body["entries"]}

        assert len(body["held_out"]) == 21
        assert sum(1 for point in body["held_out"] if point["label"] == 0) == 12

        assert by_gamma[0.01]["fit"]["accuracy"] == pytest.approx(14 / 24)
        assert by_gamma[0.01]["held_out_accuracy"] == pytest.approx(12 / 21)
        assert by_gamma[0.3]["fit"]["accuracy"] == 1.0
        assert by_gamma[0.3]["held_out_accuracy"] == pytest.approx(17 / 21)
        assert by_gamma[1.0]["held_out_accuracy"] == pytest.approx(20 / 21)
        assert by_gamma[1.0]["fit"]["n_support_vectors"] == 16
        assert by_gamma[10.0]["held_out_accuracy"] == pytest.approx(18 / 21)
        assert by_gamma[30.0]["held_out_accuracy"] == pytest.approx(16 / 21)
        assert by_gamma[100.0]["held_out_accuracy"] == pytest.approx(14 / 21)
        assert by_gamma[1000.0]["held_out_accuracy"] == pytest.approx(12 / 21)
        for gamma in (30.0, 100.0, 1000.0):
            assert by_gamma[gamma]["fit"]["accuracy"] == 1.0
            assert by_gamma[gamma]["fit"]["n_support_vectors"] == 24

    def test_the_best_held_out_score_is_a_plateau_around_one(self) -> None:
        body = client.post(GAMMA_SWEEP, json={"points": CLINIC}).json()
        best = max(entry["held_out_accuracy"] for entry in body["entries"])
        winners = {
            entry["value"]
            for entry in body["entries"]
            if entry["held_out_accuracy"] == best
        }

        assert best == pytest.approx(20 / 21)
        assert winners == {0.5, 1.0, 2.0}

    def test_the_capacity_sweep_under_the_radial_kernel(self) -> None:
        body = client.post(
            CAPACITY_SWEEP, json={"points": CLINIC, "kernel": RADIAL}
        ).json()
        by_capacity = {entry["value"]: entry for entry in body["entries"]}

        assert by_capacity[0.01]["fit"]["at_the_cap"] == 24
        assert by_capacity[0.01]["held_out_accuracy"] == pytest.approx(19 / 21)
        assert by_capacity[1.0]["fit"]["n_support_vectors"] == 16
        assert by_capacity[1.0]["fit"]["at_the_cap"] == 4
        assert by_capacity[10.0]["fit"]["n_support_vectors"] == 12
        assert by_capacity[10.0]["fit"]["at_the_cap"] == 0
        assert by_capacity[10.0]["held_out_accuracy"] == pytest.approx(20 / 21)
        assert by_capacity[100.0]["fit"]["n_support_vectors"] == 12


class TestSupportVectorsAreTheModel:
    """Part 6: refit on the support vectors alone, and drop one row each way."""

    def test_refitting_on_the_support_vectors_reproduces_the_boundary(self) -> None:
        body = client.post(
            SUPPORT_ONLY, json={"points": CLINIC, "kernel": RADIAL}
        ).json()

        assert body["full"]["n_support_vectors"] == 16
        assert len(body["support_only"]["kept_positions"]) == 16
        assert body["support_only"]["gap"] < 1e-5
        assert body["support_only"]["fit"]["accuracy"] == 1.0

    def test_dropping_a_non_support_row_moves_nothing_and_a_support_row_does(
        self,
    ) -> None:
        body = client.post(
            SUPPORT_ONLY, json={"points": CLINIC, "kernel": RADIAL}
        ).json()

        assert body["dropped_non_support"]["kept_positions"] == [
            position for position in range(24) if position != 1
        ]
        assert body["dropped_non_support"]["gap"] < 1e-6
        assert body["dropped_support"]["kept_positions"] == list(range(1, 24))
        assert body["dropped_support"]["gap"] == pytest.approx(0.3401, abs=1e-3)


class TestTheAbsorbedIntercept:
    """Part 6: the library's own fixture for the offset inside the kernel."""

    def test_capacity_releases_the_boundary_toward_the_midpoint(self) -> None:
        body = client.get(INTERCEPT_COST).json()
        by_capacity = {entry["capacity"]: entry for entry in body["entries"]}

        assert body["inputs"] == [1.0, 2.0, 3.0, 4.0]
        assert body["labels"] == [0, 0, 1, 1]
        assert by_capacity[0.1]["crossing"] == pytest.approx(0.1364, abs=1e-3)
        assert by_capacity[1.0]["crossing"] == pytest.approx(1.875, abs=1e-3)
        assert by_capacity[10.0]["crossing"] == pytest.approx(2.2308, abs=1e-3)
        assert by_capacity[100.0]["crossing"] == pytest.approx(2.5, abs=1e-3)
        assert by_capacity[1.0]["multipliers"][:3] == pytest.approx([1.0, 1.0, 1.0])
        assert by_capacity[100.0]["multipliers"][0] == 0.0
        assert by_capacity[100.0]["multipliers"][3] == 0.0


class TestTheSimilarityCurve:
    """Part 5: what near means under each kernel."""

    def test_the_curves_are_labelled_by_the_library_formulae(self) -> None:
        body = client.post(SIMILARITY, json={}).json()

        assert [curve["label"] for curve in body["curves"]] == [
            "a . b",
            "(1.0 a . b + 1.0) ** 2",
            "exp(-0.5 ||a - b|| ** 2)",
            "exp(-2.0 ||a - b|| ** 2)",
        ]
        assert body["reference"] == 1.5

    def test_the_readouts_the_page_quotes(self) -> None:
        body = client.post(SIMILARITY, json={}).json()
        positions = body["positions"]
        two_away = min(range(len(positions)), key=lambda i: abs(positions[i] + 0.5))
        at_reference = min(range(len(positions)), key=lambda i: abs(positions[i] - 1.5))

        assert body["curves"][0]["values"][two_away] == pytest.approx(-0.75)
        assert body["curves"][2]["values"][two_away] == pytest.approx(0.1353, abs=1e-3)
        assert body["curves"][3]["values"][two_away] == pytest.approx(0.0003, abs=1e-4)
        assert body["curves"][2]["values"][at_reference] == pytest.approx(1.0)
        assert body["curves"][1]["values"][at_reference] == pytest.approx(10.5625)


class TestRefusals:
    """What the endpoints will not compute, and in whose words."""

    THREE_LABELS: ClassVar[list[dict]] = CLINIC[:5] + [{"x": 37.0, "y": 60, "label": 2}]

    def test_one_class_is_refused_in_the_library_words(self) -> None:
        response = client.post(CLASSIFY, json={"points": HEALTHY, "kernel": LINEAR})

        assert response.status_code == 422
        assert response.json()["error"] == "SingleClassError"

    def test_a_lone_point_is_refused(self) -> None:
        response = client.post(CLASSIFY, json={"points": HEALTHY[:1], "kernel": LINEAR})

        assert response.status_code == 422

    def test_two_points_one_per_class_fit(self) -> None:
        body = classify(
            [{"x": 36.0, "y": 60, "label": 0}, {"x": 39.0, "y": 100, "label": 1}],
            LINEAR,
        )

        assert body["fit"]["accuracy"] == 1.0
        assert body["fit"]["n_support_vectors"] == 2

    def test_too_many_points_are_refused_with_the_limit_named(self) -> None:
        response = client.post(
            CLASSIFY, json={"points": (CLINIC * 5)[:101], "kernel": LINEAR}
        )

        assert response.status_code == 422
        assert "100" in response.json()["detail"]

    def test_a_constant_column_is_refused_by_the_standardizer(self) -> None:
        response = client.post(
            CLASSIFY,
            json={
                "points": [{"x": 37.0, "y": 60 + i, "label": i % 2} for i in range(6)],
                "kernel": LINEAR,
            },
        )

        assert response.status_code == 422
        assert response.json()["error"] == "AllSameValuesError"

    def test_a_non_finite_vital_is_refused(self) -> None:
        payload = {
            "points": CLINIC[:5] + [{"x": "NAN", "y": 60, "label": 0}],
            "kernel": LINEAR,
        }
        raw = json.dumps(payload).replace('"NAN"', "NaN")
        response = client.post(
            CLASSIFY, content=raw, headers={"content-type": "application/json"}
        )

        assert response.status_code == 422

    def test_a_third_class_is_refused(self) -> None:
        response = client.post(
            CLASSIFY, json={"points": self.THREE_LABELS, "kernel": LINEAR}
        )

        assert response.status_code == 422

    @pytest.mark.parametrize(
        "kernel",
        [
            {"name": "rbf", "gamma": 0.0},
            {"name": "polynomial", "degree": 0},
            {"name": "polynomial", "constant": -1.0},
            {"name": "cubic"},
            {"name": "rbf", "degree": 2},
        ],
    )
    def test_a_degenerate_or_unknown_kernel_is_refused(self, kernel: dict) -> None:
        response = client.post(CLASSIFY, json={"points": CLINIC, "kernel": kernel})

        assert response.status_code == 422

    def test_a_zero_capacity_is_refused(self) -> None:
        response = client.post(
            CLASSIFY, json={"points": CLINIC, "kernel": LINEAR, "capacity": 0.0}
        )

        assert response.status_code == 422

    def test_a_field_the_request_does_not_take_is_refused(self) -> None:
        response = client.post(
            CLASSIFY, json={"points": CLINIC, "kernel": LINEAR, "penalty": 1.0}
        )

        assert response.status_code == 422


class TestUnitsAndSolverSettings:
    """Part 5 and Part 6: the clinic in raw units, and the library's own defaults."""

    def test_the_playground_kernels_scored_on_the_second_clinic(self) -> None:
        assert classify(CLINIC, LINEAR)["held_out_accuracy"] == pytest.approx(15 / 21)
        assert classify(CLINIC, SQUARED)["held_out_accuracy"] == pytest.approx(16 / 21)
        assert classify(CLINIC, RADIAL)["held_out_accuracy"] == pytest.approx(20 / 21)

    def test_the_radial_kernel_on_raw_vitals_memorises(self) -> None:
        response = client.post(
            CLASSIFY, json={"points": CLINIC, "kernel": RADIAL, "standardize": False}
        )
        body = response.json()

        assert response.status_code == 200
        assert body["fit"]["accuracy"] == 1.0
        assert body["fit"]["n_support_vectors"] == 24
        assert body["held_out_accuracy"] == pytest.approx(15 / 21)

    def test_the_linear_kernel_on_raw_vitals_runs_to_the_ceiling(self) -> None:
        body = client.post(
            CLASSIFY, json={"points": CLINIC, "kernel": LINEAR, "standardize": False}
        ).json()

        assert body["fit"]["converged"] is False
        assert body["fit"]["epochs_run"] == 50_000
        assert body["fit"]["accuracy"] == pytest.approx(10 / 24)
        assert body["fit"]["at_the_cap"] == 10

    @pytest.mark.parametrize(
        ("kernel", "n_support_vectors"),
        [(LINEAR, 24), (SQUARED, 16), (RADIAL, 24)],
    )
    def test_the_library_defaults_stop_at_the_ceiling(
        self, kernel: dict, n_support_vectors: int
    ) -> None:
        body = client.post(
            CLASSIFY,
            json={"points": CLINIC, "kernel": kernel, "step": 0.001, "ceiling": 1000},
        ).json()

        assert body["fit"]["converged"] is False
        assert body["fit"]["epochs_run"] == 1000
        assert body["fit"]["n_support_vectors"] == n_support_vectors

    def test_the_gamma_plateau_the_page_quotes(self) -> None:
        body = client.post(GAMMA_SWEEP, json={"points": CLINIC}).json()
        by_gamma = {entry["value"]: entry for entry in body["entries"]}

        for gamma, kept in ((0.5, 15), (1.0, 16), (2.0, 17)):
            assert by_gamma[gamma]["held_out_accuracy"] == pytest.approx(20 / 21)
            assert by_gamma[gamma]["fit"]["n_support_vectors"] == kept

    def test_a_step_or_ceiling_outside_its_bounds_is_refused(self) -> None:
        for override in ({"step": 0.5}, {"ceiling": 0}, {"step": 0.0}):
            response = client.post(
                CLASSIFY, json={"points": CLINIC, "kernel": RADIAL, **override}
            )
            assert response.status_code == 422
