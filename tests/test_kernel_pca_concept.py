"""The kernel PCA endpoints, pinned to the page's hand arithmetic and probes.

The worked example is the PCA page's measured four, whose centred Gram matrix
under the linear kernel is two blocks with raw eigenvalues 400 and 100, which
across four rows are three times the variances 133.33 and 33.33 the PCA page
reports, shares 0.8 and 0.2, coefficients of 0.0354 on the first two rows and
0.0707 on the last two, and a first row landing at 14.14 where ordinary PCA
puts it. The running example is the ring, twelve people of one build inside
twenty-four around them in height and weight, which the radial kernel splits
along its first direction from a gamma of 0.003 to 0.017 and not at the
median-distance rule's 0.00095. The arc is the PCA page's twelve along a bend,
which the radial kernel folds rather than unrolls, five reversals along height
where ordinary PCA has none. Every figure the page quotes is pinned here, and
the raw eigenvalue is checked by two routes, the variance times n - 1 and the
reciprocal of the coefficients' squared length, because their agreeing is the
lesson the page teaches.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

ANALYZE = "/concepts/kernel-pca/analyze"
GRAM_ROUTE = "/concepts/kernel-pca/gram-route"
LIFTED = "/concepts/kernel-pca/lifted"
SWEEP = "/concepts/kernel-pca/sweep"
EDGES = "/concepts/kernel-pca/edges"

MEASURED_FOUR = [
    {"x": 180.0, "y": 78.0},
    {"x": 160.0, "y": 58.0},
    {"x": 175.0, "y": 63.0},
    {"x": 165.0, "y": 73.0},
]

FIRST_COEFFICIENT = 0.035355339059327376
"""1 / (20 sqrt 2): the unit eigenvector (1, -1, 0, 0) / sqrt 2, over sqrt 400."""

SECOND_COEFFICIENT = 0.07071067811865475
"""1 / (10 sqrt 2): the unit eigenvector (0, 0, 1, -1) / sqrt 2, over sqrt 100."""

FIRST_COORDINATE = 14.142135623730951
"""Person 1 along the first direction, 200 times 0.0354 twice over, 10 sqrt 2."""

THIRD_ROW_SECOND_COORDINATE = 7.0710678118654755
"""Person 3 along the second direction, 50 times 0.0707 twice over, 5 sqrt 2."""

ARC = [
    {"x": 120, "y": 80},
    {"x": 128, "y": 62},
    {"x": 136, "y": 49},
    {"x": 144, "y": 40},
    {"x": 152, "y": 35},
    {"x": 160, "y": 33},
    {"x": 168, "y": 35},
    {"x": 176, "y": 40},
    {"x": 184, "y": 49},
    {"x": 192, "y": 62},
    {"x": 200, "y": 80},
    {"x": 156, "y": 33},
]
"""The PCA page's twelve people along a bend."""

RING = [
    {"x": 169.7, "y": 60.3},
    {"x": 167.2, "y": 65.1},
    {"x": 164.8, "y": 69.7},
    {"x": 160.6, "y": 69.2},
    {"x": 155.7, "y": 68.4},
    {"x": 150.1, "y": 64.6},
    {"x": 150.1, "y": 61.2},
    {"x": 151.1, "y": 53.9},
    {"x": 155.5, "y": 50.7},
    {"x": 159.9, "y": 49.4},
    {"x": 164.1, "y": 52.0},
    {"x": 168.5, "y": 54.5},
    {"x": 190.0, "y": 58.8},
    {"x": 190.1, "y": 67.8},
    {"x": 185.6, "y": 74.8},
    {"x": 181.2, "y": 81.6},
    {"x": 172.3, "y": 85.9},
    {"x": 166.3, "y": 89.0},
    {"x": 159.4, "y": 89.5},
    {"x": 150.3, "y": 87.3},
    {"x": 145.8, "y": 87.5},
    {"x": 140.2, "y": 82.1},
    {"x": 132.5, "y": 75.2},
    {"x": 131.4, "y": 70.0},
    {"x": 129.8, "y": 62.2},
    {"x": 130.6, "y": 52.1},
    {"x": 133.5, "y": 44.9},
    {"x": 140.9, "y": 38.2},
    {"x": 145.6, "y": 33.9},
    {"x": 152.5, "y": 31.8},
    {"x": 160.0, "y": 30.3},
    {"x": 166.3, "y": 31.9},
    {"x": 173.5, "y": 31.7},
    {"x": 179.6, "y": 38.4},
    {"x": 186.9, "y": 45.6},
    {"x": 188.0, "y": 51.0},
]
"""The widget's ring: twelve people about ten units from (160, 60), then twenty-four about thirty."""

INNER_COUNT = 12

IDEAL_CASE = [
    {"x": 162.9, "y": 60.9},
    {"x": 161.7, "y": 63.2},
    {"x": 158.8, "y": 64.0},
    {"x": 157.1, "y": 61.5},
    {"x": 156.3, "y": 58.8},
    {"x": 158.6, "y": 57.3},
    {"x": 161.1, "y": 56.6},
    {"x": 163.7, "y": 58.0},
    {"x": 190.8, "y": 63.1},
    {"x": 189.5, "y": 75.8},
    {"x": 180.6, "y": 85.2},
    {"x": 169.1, "y": 90.2},
    {"x": 156.6, "y": 93.8},
    {"x": 144.4, "y": 89.1},
    {"x": 135.2, "y": 80.3},
    {"x": 130.3, "y": 68.9},
    {"x": 126.7, "y": 56.7},
    {"x": 131.4, "y": 44.6},
    {"x": 140.1, "y": 35.6},
    {"x": 150.2, "y": 27.4},
    {"x": 163.3, "y": 27.2},
    {"x": 175.1, "y": 31.8},
    {"x": 184.0, "y": 40.4},
    {"x": 192.1, "y": 50.3},
]
"""A tighter eight inside a wider sixteen, which splits across a far wider band."""

IDEAL_INNER_COUNT = 8

GAMMA_STOPS = [
    0.0003,
    0.0005,
    0.0008,
    0.001,
    0.0015,
    0.002,
    0.003,
    0.004,
    0.005,
    0.007,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.1,
]
"""The stops the page's sliders step through."""

RING_GAMMA = 0.005
ARC_GAMMA = 0.002


def _post(path: str, body: dict) -> dict:
    response = client.post(path, json=body)
    assert response.status_code == 200, response.json()
    return response.json()


def _analyze(points: list[dict], kernel: str, gamma: float = 1.0) -> dict:
    return _post(ANALYZE, {"points": points, "kernel": kernel, "gamma": gamma})


def _lifted(points: list[dict], kernel: str, gamma: float = 1.0, **extra) -> dict:
    return _post(LIFTED, {"points": points, "kernel": kernel, "gamma": gamma, **extra})


def _first_axis_splits(coordinates: list, inner: int) -> bool:
    """Whether every inner person sits to one side of every outer person."""
    first = [
        entry["first"] if isinstance(entry, dict) else entry[0] for entry in coordinates
    ]
    inner_values, outer_values = first[:inner], first[inner:]
    return min(inner_values) > max(outer_values) or max(inner_values) < min(
        outer_values
    )


def _reciprocal_squared_length(coefficients: list[float]) -> float:
    """The raw eigenvalue read off the coefficients alone."""
    return 1.0 / sum(value * value for value in coefficients)


def _by_height(points: list[dict], values: list[float]) -> list[float]:
    order = sorted(range(len(points)), key=lambda index: points[index]["x"])
    return [values[index] for index in order]


class TestTheMeasuredFour:
    """The PCA page's four under the linear kernel, every figure by hand."""

    def test_four_rows_so_the_divisor_is_three(self) -> None:
        body = _analyze(MEASURED_FOUR, "linear")

        assert body["n_rows"] == 4

    def test_the_raw_eigenvalues_are_four_hundred_and_one_hundred(self) -> None:
        body = _analyze(MEASURED_FOUR, "linear")

        first, second = body["components"]
        assert first["raw_eigenvalue"] == pytest.approx(400.0)
        assert second["raw_eigenvalue"] == pytest.approx(100.0)

    def test_the_variances_are_the_raw_eigenvalues_over_three(self) -> None:
        body = _analyze(MEASURED_FOUR, "linear")

        first, second = body["components"]
        assert first["variance"] == pytest.approx(400.0 / 3)
        assert second["variance"] == pytest.approx(100.0 / 3)
        assert body["total_variance"] == pytest.approx(500.0 / 3)

    def test_the_shares_are_four_fifths_and_one_fifth(self) -> None:
        body = _analyze(MEASURED_FOUR, "linear")

        first, second = body["components"]
        assert first["share"] == pytest.approx(0.8)
        assert second["share"] == pytest.approx(0.2)
        assert second["cumulative_share"] == pytest.approx(1.0)

    def test_the_raw_eigenvalue_agrees_with_the_coefficients_squared_length(
        self,
    ) -> None:
        body = _analyze(MEASURED_FOUR, "linear")

        for component in body["components"]:
            by_length = _reciprocal_squared_length(component["row_coefficients"])
            assert component["raw_eigenvalue"] == pytest.approx(by_length)

    def test_the_first_direction_is_built_from_the_first_two_people(self) -> None:
        body = _analyze(MEASURED_FOUR, "linear")

        coefficients = body["components"][0]["row_coefficients"]
        assert len(coefficients) == 4
        assert abs(coefficients[0]) == pytest.approx(FIRST_COEFFICIENT)
        assert coefficients[0] + coefficients[1] == pytest.approx(0.0, abs=1e-12)
        assert all(abs(value) < 1e-9 for value in coefficients[2:])

    def test_the_second_direction_is_built_from_people_three_and_four(self) -> None:
        body = _analyze(MEASURED_FOUR, "linear")

        coefficients = body["components"][1]["row_coefficients"]
        assert abs(coefficients[2]) == pytest.approx(SECOND_COEFFICIENT)
        assert coefficients[2] + coefficients[3] == pytest.approx(0.0, abs=1e-12)
        assert all(abs(coefficients[index]) < 1e-9 for index in (0, 1))

    def test_the_first_kernel_row_is_two_hundred_minus_two_hundred_and_zeros(
        self,
    ) -> None:
        body = _analyze(MEASURED_FOUR, "linear")

        assert body["first_row_kernel_values"] == pytest.approx(
            [200.0, -200.0, 0.0, 0.0]
        )

    def test_the_first_person_lands_at_ten_root_two(self) -> None:
        body = _analyze(MEASURED_FOUR, "linear")

        first_row = body["coordinates"][0]
        assert abs(first_row["first"]) == pytest.approx(FIRST_COORDINATE)
        assert first_row["second"] == pytest.approx(0.0, abs=1e-9)

    def test_the_third_person_lands_at_five_root_two_along_the_second(self) -> None:
        body = _analyze(MEASURED_FOUR, "linear")

        third_row = body["coordinates"][2]
        assert third_row["first"] == pytest.approx(0.0, abs=1e-9)
        assert abs(third_row["second"]) == pytest.approx(THIRD_ROW_SECOND_COORDINATE)

    def test_the_coordinate_is_the_kernel_row_against_the_coefficients(
        self,
    ) -> None:
        body = _analyze(MEASURED_FOUR, "linear")

        coefficients = body["components"][0]["row_coefficients"]
        summed = sum(
            kernel_value * coefficient
            for kernel_value, coefficient in zip(
                body["first_row_kernel_values"], coefficients, strict=True
            )
        )
        assert summed == pytest.approx(body["coordinates"][0]["first"])

    def test_the_linear_kernel_is_ordinary_pca_up_to_a_sign(self) -> None:
        body = _analyze(MEASURED_FOUR, "linear")

        for kernel, ordinary in zip(
            body["components"], body["ordinary"]["components"], strict=True
        ):
            assert kernel["variance"] == pytest.approx(ordinary["variance"])
            assert kernel["share"] == pytest.approx(ordinary["share"])

        for kernel, ordinary in zip(
            body["coordinates"], body["ordinary"]["coordinates"], strict=True
        ):
            assert abs(kernel["first"]) == pytest.approx(abs(ordinary["first"]))
            assert abs(kernel["second"]) == pytest.approx(
                abs(ordinary["second"]), abs=1e-9
            )


class TestTheGramRoute:
    """PCA by two routes on one cloud, under the linear kernel."""

    def test_the_uncentred_table_of_the_four_is_dominated_by_the_mean(self) -> None:
        body = _post(GRAM_ROUTE, {"points": MEASURED_FOUR})

        assert body["divisor"] == 3
        assert body["uncentred_gram"][0] == pytest.approx(
            [38484.0, 33324.0, 36414.0, 35394.0]
        )
        assert body["uncentred_top_eigenvalue"] == pytest.approx(134449.536, abs=1e-3)
        assert body["uncentred_top_share"] == pytest.approx(0.9989, abs=5e-5)
        # Nearly the same weight on everyone: the direction to the mean.
        eigenvector = body["uncentred_top_eigenvector"]
        assert len({value > 0 for value in eigenvector}) == 1
        assert (
            max(abs(value) for value in eigenvector)
            - min(abs(value) for value in eigenvector)
            < 0.08
        )

    def test_the_centred_table_of_the_four_is_two_blocks(self) -> None:
        body = _post(GRAM_ROUTE, {"points": MEASURED_FOUR})

        expected = [
            [200.0, -200.0, 0.0, 0.0],
            [-200.0, 200.0, 0.0, 0.0],
            [0.0, 0.0, 50.0, -50.0],
            [0.0, 0.0, -50.0, 50.0],
        ]
        for row, expected_row in zip(body["centred_gram"], expected, strict=True):
            assert row == pytest.approx(expected_row)

    def test_the_two_routes_share_their_eigenvalues_on_the_four(self) -> None:
        body = _post(GRAM_ROUTE, {"points": MEASURED_FOUR})

        assert body["scatter_matrix"][0] == pytest.approx([250.0, 150.0])
        assert body["scatter_matrix"][1] == pytest.approx([150.0, 250.0])
        assert body["scatter_eigenvalues"] == pytest.approx([400.0, 100.0])
        assert body["gram_eigenvalues"] == pytest.approx(
            [400.0, 100.0, 0.0, 0.0], abs=1e-9
        )
        assert body["ordinary_variances"] == pytest.approx([400.0 / 3, 100.0 / 3])
        assert body["largest_coordinate_gap"] < 1e-12

    def test_the_coefficients_are_perpendicular_and_not_unit_length(self) -> None:
        body = _post(GRAM_ROUTE, {"points": MEASURED_FOUR})

        assert body["coefficient_squared_lengths"] == pytest.approx([0.0025, 0.01])
        dot_products = body["coefficient_dot_products"]
        assert dot_products[0] == pytest.approx([0.0025, 0.0], abs=1e-15)
        assert dot_products[1] == pytest.approx([0.0, 0.01], abs=1e-15)
        metric = body["kernel_metric_products"]
        assert metric[0] == pytest.approx([1.0, 0.0], abs=1e-12)
        assert metric[1] == pytest.approx([0.0, 1.0], abs=1e-12)

    def test_the_first_person_as_a_query_reproduces_the_first_centred_row(self) -> None:
        body = _post(GRAM_ROUTE, {"points": MEASURED_FOUR})

        assert body["first_row_centred_against_training"] == pytest.approx(
            [200.0, -200.0, 0.0, 0.0]
        )

    def test_the_two_routes_agree_on_the_arc(self) -> None:
        body = _post(GRAM_ROUTE, {"points": ARC})

        assert body["scatter_eigenvalues"] == pytest.approx(
            [7055.879, 3316.454], abs=1e-3
        )
        assert body["gram_eigenvalues"][:2] == pytest.approx(
            body["scatter_eigenvalues"]
        )
        assert max(body["gram_eigenvalues"][2:]) < 1e-9
        assert body["ordinary_shares"] == pytest.approx([0.6803, 0.3197], abs=5e-5)
        assert body["total_variance"] == pytest.approx(942.94, abs=1e-2)
        assert body["largest_coordinate_gap"] < 1e-9
        assert body["uncentred_top_share"] == pytest.approx(0.9896, abs=5e-5)


class TestTheRing:
    """Twelve of one build inside twenty-four around them."""

    def test_ordinary_pca_leaves_the_ring_nested(self) -> None:
        body = _lifted(RING, "linear", inner_count=INNER_COUNT)

        assert [
            component["share"] for component in body["components"]
        ] == pytest.approx([0.5164, 0.4836], abs=5e-5)
        assert body["first_axis_gap"] == pytest.approx(-40.18, abs=1e-2)

    def test_the_radial_kernel_splits_the_ring_along_its_first_direction(self) -> None:
        body = _lifted(RING, "rbf", RING_GAMMA, inner_count=INNER_COUNT)

        assert body["n_rows"] == 36
        assert body["divisor"] == 35
        assert body["reach"] == pytest.approx(14.142, abs=1e-3)
        assert body["first_axis_gap"] == pytest.approx(0.634, abs=5e-4)
        assert _first_axis_splits(body["coordinates"], INNER_COUNT)
        first = [row[0] for row in body["coordinates"]]
        assert min(first[:INNER_COUNT]) == pytest.approx(0.429, abs=5e-4)
        assert max(first[:INNER_COUNT]) == pytest.approx(0.560, abs=5e-4)
        assert min(first[INNER_COUNT:]) == pytest.approx(-0.291, abs=5e-4)
        assert max(first[INNER_COUNT:]) == pytest.approx(-0.205, abs=5e-4)

    def test_the_shares_at_the_featured_gamma(self) -> None:
        body = _lifted(RING, "rbf", RING_GAMMA, inner_count=INNER_COUNT)

        first, second = body["components"]
        assert first["share"] == pytest.approx(0.1366, abs=5e-5)
        assert second["share"] == pytest.approx(0.1114, abs=5e-5)
        assert second["cumulative_share"] == pytest.approx(0.248, abs=5e-4)
        assert body["total_variance"] == pytest.approx(0.8977, abs=5e-5)
        assert body["total_kept_share"] == pytest.approx(0.248, abs=5e-4)
        assert body["rank"] == 35

    def test_the_first_direction_gives_each_group_one_sign(self) -> None:
        body = _lifted(RING, "rbf", RING_GAMMA, inner_count=INNER_COUNT)

        coefficients = body["components"][0]["row_coefficients"]
        inner = coefficients[:INNER_COUNT]
        outer = coefficients[INNER_COUNT:]
        assert len({value > 0 for value in inner}) == 1
        assert len({value > 0 for value in outer}) == 1
        assert (inner[0] > 0) != (outer[0] > 0)
        assert min(abs(value) for value in inner) == pytest.approx(0.100, abs=5e-4)
        assert max(abs(value) for value in inner) == pytest.approx(0.130, abs=5e-4)
        assert min(abs(value) for value in outer) == pytest.approx(0.048, abs=5e-4)
        assert max(abs(value) for value in outer) == pytest.approx(0.068, abs=5e-4)

    def test_the_two_scales_agree_under_the_radial_kernel_too(self) -> None:
        body = _lifted(RING, "rbf", RING_GAMMA, inner_count=INNER_COUNT)

        first = body["components"][0]
        assert first["raw_eigenvalue"] == pytest.approx(4.2923, abs=5e-5)
        assert first["raw_eigenvalue"] == pytest.approx(35 * first["variance"])
        assert first["variance"] == pytest.approx(0.1226, abs=5e-5)
        for component in body["components"]:
            by_length = _reciprocal_squared_length(component["row_coefficients"])
            assert component["raw_eigenvalue"] == pytest.approx(by_length)

    def test_the_coefficient_vectors_are_perpendicular_and_orthonormal_through_the_table(
        self,
    ) -> None:
        # The claim that they are not perpendicular in the ordinary sense did
        # not survive measurement; they are eigenvectors of one symmetric
        # table, only scaled.
        body = _lifted(RING, "rbf", RING_GAMMA, inner_count=INNER_COUNT)

        assert body["largest_coefficient_dot_product"] < 1e-12
        assert body["largest_kernel_metric_gap"] < 1e-12

    def test_person_one_is_placed_by_their_centred_kernel_row(self) -> None:
        body = _analyze(RING, "rbf", RING_GAMMA)

        assert abs(body["components"][0]["row_coefficients"][0]) == pytest.approx(
            0.1194, abs=5e-5
        )
        assert abs(body["coordinates"][0]["first"]) == pytest.approx(0.513, abs=5e-4)
        assert [
            component["share"] for component in body["ordinary"]["components"]
        ] == pytest.approx([0.5164, 0.4836], abs=5e-5)

    def test_the_kernel_row_of_person_one(self) -> None:
        body = _lifted(RING, "rbf", RING_GAMMA, inner_count=INNER_COUNT)

        row = body["uncentred_kernel"][0]
        assert row[:3] == pytest.approx([1.0, 0.864, 0.570], abs=5e-4)
        assert min(row) == pytest.approx(0.0003, abs=5e-5)

    def test_the_kept_share_per_person_with_two_directions(self) -> None:
        body = _lifted(RING, "rbf", RING_GAMMA, inner_count=INNER_COUNT)

        shares = [
            kept / lifted
            for kept, lifted in zip(
                body["kept_squared"], body["lifted_squared_deviations"], strict=True
            )
        ]
        assert min(shares) == pytest.approx(0.057, abs=5e-4)
        assert max(shares) == pytest.approx(0.537, abs=5e-4)
        assert all(
            kept <= lifted + 1e-12
            for kept, lifted in zip(
                body["kept_squared"], body["lifted_squared_deviations"], strict=True
            )
        )
        assert body["median_squared_distance"] == pytest.approx(1051.515)

    def test_keeping_every_direction_climbs_the_ladder_slowly(self) -> None:
        body = _lifted(
            RING, "rbf", RING_GAMMA, n_components=36, inner_count=INNER_COUNT
        )

        cumulative = [component["cumulative_share"] for component in body["components"]]
        assert len(cumulative) == 35
        assert (
            next(index + 1 for index, value in enumerate(cumulative) if value >= 0.5)
            == 5
        )
        assert (
            next(index + 1 for index, value in enumerate(cumulative) if value >= 0.9)
            == 13
        )
        assert cumulative[-1] == pytest.approx(1.0)
        assert body["total_kept_share"] == pytest.approx(1.0)

    def test_the_squared_kernel_keeps_the_ring_nested_until_its_fifth_direction(
        self,
    ) -> None:
        body = _lifted(
            RING, "polynomial", RING_GAMMA, n_components=5, inner_count=INNER_COUNT
        )

        shares = [component["share"] for component in body["components"]]
        assert shares[:3] == pytest.approx([0.6515, 0.3466, 0.0019], abs=5e-5)
        assert shares[4] < 1e-5
        assert body["first_axis_gap"] == pytest.approx(-63.5, abs=5e-2)
        for position in range(4):
            assert not _first_axis_splits(
                [[row[position]] for row in body["coordinates"]], INNER_COUNT
            )
        assert _first_axis_splits(
            [[row[4]] for row in body["coordinates"]], INNER_COUNT
        )


class TestTheArc:
    """The PCA page's bend, which the radial kernel folds."""

    def test_ordinary_pca_orders_the_arc_by_height_without_a_reversal(self) -> None:
        body = _lifted(ARC, "rbf", ARC_GAMMA)

        assert body["ordinary_order_reversals_along_x"] == 0
        first = _by_height(
            ARC, [entry["first"] for entry in body["ordinary_coordinates"]]
        )
        assert abs(first[0]) == pytest.approx(39.1, abs=5e-2)
        assert abs(first[-1]) == pytest.approx(40.9, abs=5e-2)

    def test_the_radial_kernel_folds_the_arc(self) -> None:
        body = _lifted(ARC, "rbf", ARC_GAMMA)

        assert body["order_reversals_along_x"] == 5
        assert body["reach"] == pytest.approx(22.36, abs=5e-3)
        assert body["rank"] == 11
        first = _by_height(ARC, [row[0] for row in body["coordinates"]])
        sign = 1.0 if first[5] > 0 else -1.0
        folded = [sign * value for value in first]
        assert folded == pytest.approx(
            [
                -0.49,
                -0.44,
                -0.07,
                0.35,
                0.58,
                0.61,
                0.58,
                0.40,
                0.05,
                -0.37,
                -0.63,
                -0.57,
            ],
            abs=5e-3,
        )
        # Both ends on one side, the middle on the other.
        assert folded[0] < 0 and folded[-1] < 0 and folded[5] > 0

    def test_the_second_direction_runs_along_the_bend(self) -> None:
        body = _lifted(ARC, "rbf", ARC_GAMMA)

        second = _by_height(ARC, [row[1] for row in body["coordinates"]])
        sign = 1.0 if second[-3] > 0 else -1.0
        assert [sign * value for value in second] == pytest.approx(
            [
                -0.36,
                -0.63,
                -0.66,
                -0.45,
                -0.15,
                0.00,
                0.14,
                0.40,
                0.59,
                0.60,
                0.40,
                0.12,
            ],
            abs=5e-3,
        )

    def test_the_shares_and_the_kernel_rows_at_the_featured_gamma(self) -> None:
        body = _lifted(ARC, "rbf", ARC_GAMMA)

        first, second = body["components"]
        assert first["share"] == pytest.approx(0.3150, abs=5e-5)
        assert second["share"] == pytest.approx(0.2705, abs=5e-5)
        assert body["total_variance"] == pytest.approx(0.7569, abs=5e-5)
        assert body["median_squared_distance"] == pytest.approx(1262.5)
        end = body["uncentred_kernel"][0]
        assert end[:3] == pytest.approx([1.0, 0.460, 0.088], abs=5e-4)
        assert end[4] == pytest.approx(0.0022, abs=5e-5)
        assert all(value < 0.001 for value in end[5:11])
        dip = body["uncentred_kernel"][5]
        assert dip[4] == pytest.approx(0.873, abs=5e-4)
        assert dip[6] == pytest.approx(0.873, abs=5e-4)
        assert dip[11] == pytest.approx(0.969, abs=5e-4)
        assert body["largest_coefficient_dot_product"] < 1e-12
        assert body["largest_kernel_metric_gap"] < 1e-12

    def test_widening_the_reach_loosens_the_fold_toward_a_rotation(self) -> None:
        body = _post(SWEEP, {"points": ARC, "gammas": GAMMA_STOPS})

        reversals = {
            entry["gamma"]: entry["order_reversals_along_x"]
            for entry in body["entries"]
        }
        assert reversals[0.0003] == 1
        assert reversals[0.0005] == 2
        assert all(reversals[gamma] == 5 for gamma in GAMMA_STOPS if gamma >= 0.0015)
        widest = body["entries"][0]
        assert widest["first_share"] == pytest.approx(0.526, abs=5e-4)
        assert widest["second_share"] == pytest.approx(0.370, abs=5e-4)
        assert body["median_squared_distance"] == pytest.approx(1262.5)


class TestTheSweep:
    """The reach swept across the ring and the ideal case."""

    SPLIT_FROM: ClassVar[float] = 0.003
    SPLIT_TO: ClassVar[float] = 0.017

    def test_the_ring_splits_from_three_thousandths_to_seventeen(self) -> None:
        body = _post(
            SWEEP,
            {
                "points": RING,
                "gammas": GAMMA_STOPS + [0.017, 0.018],
                "inner_count": INNER_COUNT,
            },
        )

        gaps = {entry["gamma"]: entry["first_axis_gap"] for entry in body["entries"]}
        splitting = sorted(gamma for gamma, gap in gaps.items() if gap > 0)
        assert splitting[0] == self.SPLIT_FROM
        assert splitting[-1] == self.SPLIT_TO
        assert gaps[0.002] == pytest.approx(-0.966, abs=5e-4)
        assert gaps[0.001] == pytest.approx(-0.992, abs=5e-4)
        assert gaps[0.004] == pytest.approx(0.645, abs=5e-4)
        assert gaps[0.02] == pytest.approx(-0.249, abs=5e-4)
        assert max(gaps.values()) == gaps[0.004]
        assert body["median_squared_distance"] == pytest.approx(1051.515)
        assert 1 / body["median_squared_distance"] == pytest.approx(0.00095, abs=5e-6)

    def test_the_lines_move_monotonically_across_the_band(self) -> None:
        body = _post(
            SWEEP, {"points": RING, "gammas": GAMMA_STOPS, "inner_count": INNER_COUNT}
        )

        entries = body["entries"]
        totals = [entry["total_variance"] for entry in entries]
        firsts = [entry["first_share"] for entry in entries]
        assert totals == sorted(totals)
        assert firsts == sorted(firsts, reverse=True)
        assert totals[0] == pytest.approx(0.292, abs=5e-4)
        assert totals[-1] == pytest.approx(0.998, abs=5e-4)
        assert firsts[0] == pytest.approx(0.428, abs=5e-4)
        assert firsts[-1] == pytest.approx(0.037, abs=5e-4)
        assert entries[0]["reach"] == pytest.approx(57.74, abs=5e-3)

    def test_the_ideal_case_holds_its_split_across_a_far_wider_band(self) -> None:
        body = _post(
            SWEEP,
            {
                "points": IDEAL_CASE,
                "gammas": GAMMA_STOPS,
                "inner_count": IDEAL_INNER_COUNT,
            },
        )

        gaps = {entry["gamma"]: entry["first_axis_gap"] for entry in body["entries"]}
        splitting = sorted(gamma for gamma, gap in gaps.items() if gap > 0)
        assert splitting[0] == 0.0015
        assert splitting[-1] == 0.1
        assert gaps[0.004] == pytest.approx(0.981, abs=5e-4)
        assert max(gaps.values()) == gaps[0.004]
        assert 1 / body["median_squared_distance"] == pytest.approx(0.00088, abs=5e-6)

    def test_the_ideal_case_is_nested_under_the_linear_kernel(self) -> None:
        body = _lifted(IDEAL_CASE, "linear", inner_count=IDEAL_INNER_COUNT)

        assert [
            component["share"] for component in body["components"]
        ] == pytest.approx([0.5068, 0.4932], abs=5e-5)
        assert body["first_axis_gap"] < 0


@pytest.fixture(scope="module")
def outcomes() -> dict[str, dict]:
    """Every edge the contracts table quotes, probed once for the module."""
    response = client.get(EDGES)
    assert response.status_code == 200, response.json()
    return {outcome["edge"]: outcome for outcome in response.json()["outcomes"]}


class TestTheEdges:
    """Every row of the contracts table, read back from the live probe."""

    def test_the_refusals(self, outcomes: dict[str, dict]) -> None:
        refused = [
            "one row",
            "three identical rows",
            "a non-finite value",
            "five directions from four rows",
            "zero directions",
            "a gamma of zero",
            "a negative polynomial constant",
            "transform before fit",
            "transform with a feature under another name",
            "transform with a feature missing",
            "a polynomial kernel of degree 200 on raw units",
            "centring a query row against itself",
        ]
        for edge in refused:
            assert not outcomes[edge]["accepted"], edge
        assert "at least two rows" in outcomes["one row"]["detail"]
        assert "at least one component" in outcomes["three identical rows"]["detail"]
        assert (
            "cannot keep 5 components from 4 rows"
            in outcomes["five directions from four rows"]["detail"]
        )
        assert (
            "non-finite"
            in outcomes["a polynomial kernel of degree 200 on raw units"]["detail"]
        )
        assert "square" in outcomes["centring a query row against itself"]["detail"]

    def test_what_is_accepted_and_the_figure_it_answers(
        self, outcomes: dict[str, dict]
    ) -> None:
        assert outcomes["two rows"]["figure"] == 1
        assert outcomes["a constant column"]["figure"] == 3
        assert outcomes["four directions from four rows on a plane"]["figure"] == 2
        assert outcomes["twelve directions from the twelve on the arc"]["figure"] == 11
        assert outcomes["transform with the features reordered"]["figure"] == 0.0

    def test_an_indefinite_kernel_is_accepted_and_clamped(
        self, outcomes: dict[str, dict]
    ) -> None:
        outcome = outcomes["a sigmoid kernel that is not a kernel"]

        assert outcome["accepted"]
        assert "5 negative eigenvalues" in outcome["detail"]
        assert outcome["figure"] == pytest.approx(-0.0365, abs=5e-5)
        assert "0.9989" in outcome["detail"]

    def test_strangers_land_at_one_spot(self, outcomes: dict[str, dict]) -> None:
        outcome = outcomes["two strangers far from everyone"]

        assert outcome["accepted"]
        assert outcome["figure"] == pytest.approx(-0.2291, abs=5e-5)
        assert "0.0e+00 apart" in outcome["detail"]

    def test_metres_change_the_kernel(self, outcomes: dict[str, dict]) -> None:
        outcome = outcomes["height in metres at the same gamma"]

        assert outcome["accepted"]
        assert outcome["figure"] == pytest.approx(0.6797, abs=5e-5)
        assert "0.3150" in outcome["detail"]


class TestRefusals:
    def test_a_lone_point_is_refused(self) -> None:
        response = client.post(
            ANALYZE, json={"points": MEASURED_FOUR[:1], "kernel": "linear"}
        )

        assert response.status_code == 422

    def test_a_line_of_points_has_no_second_direction_in_words(self) -> None:
        line = [{"x": float(step), "y": float(step)} for step in range(1, 5)]

        for path in (ANALYZE, GRAM_ROUTE):
            body = (
                {"points": line, "kernel": "linear"}
                if path == ANALYZE
                else {"points": line}
            )
            response = client.post(path, json=body)
            assert response.status_code == 422
            assert "only one direction" in response.json()["detail"]

    def test_identical_points_are_refused_by_the_library(self) -> None:
        same = [{"x": 1.0, "y": 2.0}] * 3

        response = client.post(
            LIFTED, json={"points": same, "kernel": "rbf", "gamma": 0.5}
        )

        assert response.status_code == 422
        assert "at least one component" in response.json()["detail"]

    def test_an_unknown_kernel_is_refused(self) -> None:
        response = client.post(
            ANALYZE, json={"points": MEASURED_FOUR, "kernel": "sigmoid"}
        )

        assert response.status_code == 422

    def test_a_non_positive_or_too_large_gamma_is_refused(self) -> None:
        for gamma in (0.0, 101.0):
            response = client.post(
                LIFTED, json={"points": RING, "kernel": "rbf", "gamma": gamma}
            )
            assert response.status_code == 422

    def test_more_directions_than_rows_is_refused_with_both_counts(self) -> None:
        response = client.post(
            LIFTED,
            json={"points": MEASURED_FOUR, "kernel": "linear", "n_components": 5},
        )

        assert response.status_code == 422
        assert "cannot keep 5 components from 4 rows" in response.json()["detail"]

    def test_a_group_that_leaves_nobody_outside_is_refused(self) -> None:
        response = client.post(
            LIFTED, json={"points": MEASURED_FOUR, "kernel": "linear", "inner_count": 4}
        )

        assert response.status_code == 422
        assert "outside the group" in response.json()["detail"]

    def test_a_sweep_gamma_outside_the_range_is_refused(self) -> None:
        response = client.post(SWEEP, json={"points": RING, "gammas": [0.0]})

        assert response.status_code == 422
        assert "every gamma must lie" in response.json()["detail"]

    def test_a_sweep_too_long_is_refused(self) -> None:
        response = client.post(SWEEP, json={"points": RING, "gammas": [0.01] * 61})

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        for path, body in (
            (ANALYZE, {"points": MEASURED_FOUR, "kernel": "linear", "degree": 3}),
            (GRAM_ROUTE, {"points": MEASURED_FOUR, "kernel": "linear"}),
            (SWEEP, {"points": RING, "gammas": [0.01], "extra": 1}),
        ):
            response = client.post(path, json=body)
            assert response.status_code == 422, path
