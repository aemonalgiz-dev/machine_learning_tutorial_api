"""The kernel PCA endpoint, pinned to the page's hand arithmetic.

The worked example is the library's own rotated ellipse, five points whose
centred rows are (2, 2), (-2, -2), (1, -1), (-1, 1) and (0, 0). Under the
linear kernel their centred Gram matrix is two blocks and a zero, with raw
eigenvalues 16 and 4, which across five rows are four times the variances 4
and 1, shares 0.8 and 0.2, and the first row lands at 2.8284 along the first
direction, where ordinary PCA puts it too. The opening example is two rings,
twelve points near radius one and twenty-four near radius three, which the
radial kernel splits along its first direction at gamma 0.5 and fails to split
at a tenth or at three. Every figure the page quotes is pinned here, and the
raw eigenvalue is checked by two routes, the variance times n - 1 and the
reciprocal of the coefficients' squared length, because their agreeing is the
lesson the page teaches.
"""

from typing import ClassVar

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

ANALYZE = "/concepts/kernel-pca/analyze"

WORKED_FIVE = [
    {"x": 12.0, "y": 102.0},
    {"x": 8.0, "y": 98.0},
    {"x": 11.0, "y": 99.0},
    {"x": 9.0, "y": 101.0},
    {"x": 10.0, "y": 100.0},
]

FIRST_COEFFICIENT = 0.17677669529663687
"""1 / (4 sqrt 2): the unit eigenvector (1, -1, 0, 0, 0) / sqrt 2, over sqrt 16."""

SECOND_COEFFICIENT = 0.35355339059327373
"""1 / (2 sqrt 2): the unit eigenvector (0, 0, 1, -1, 0) / sqrt 2, over sqrt 4."""

FIRST_COORDINATE = 2.8284271247461903
"""Row 1 along the first direction, 8 times 0.1768 twice over, which is 2 sqrt 2."""

THIRD_ROW_SECOND_COORDINATE = 1.4142135623730951
"""Row 3 along the second direction, 2 times 0.3536 twice over, which is sqrt 2."""

RING_POINTS = [
    {"x": 0.968, "y": 0.029},
    {"x": 0.717, "y": 0.511},
    {"x": 0.477, "y": 0.968},
    {"x": 0.060, "y": 0.915},
    {"x": -0.432, "y": 0.839},
    {"x": -0.990, "y": 0.459},
    {"x": -0.995, "y": 0.119},
    {"x": -0.894, "y": -0.612},
    {"x": -0.452, "y": -0.930},
    {"x": -0.008, "y": -1.058},
    {"x": 0.414, "y": -0.802},
    {"x": 0.847, "y": -0.551},
    {"x": 3.000, "y": -0.118},
    {"x": 3.012, "y": 0.779},
    {"x": 2.561, "y": 1.481},
    {"x": 2.124, "y": 2.160},
    {"x": 1.233, "y": 2.589},
    {"x": 0.633, "y": 2.898},
    {"x": -0.058, "y": 2.946},
    {"x": -0.966, "y": 2.729},
    {"x": -1.424, "y": 2.748},
    {"x": -1.979, "y": 2.207},
    {"x": -2.750, "y": 1.523},
    {"x": -2.859, "y": 0.995},
    {"x": -3.023, "y": 0.225},
    {"x": -2.936, "y": -0.794},
    {"x": -2.652, "y": -1.512},
    {"x": -1.906, "y": -2.179},
    {"x": -1.437, "y": -2.613},
    {"x": -0.750, "y": -2.824},
    {"x": -0.001, "y": -2.971},
    {"x": 0.632, "y": -2.809},
    {"x": 1.353, "y": -2.834},
    {"x": 1.957, "y": -2.160},
    {"x": 2.688, "y": -1.444},
    {"x": 2.800, "y": -0.896},
]
"""The widget's rings: twelve points near radius one, then twenty-four near three."""

INNER_RING_COUNT = 12


def _analyze(points: list[dict], kernel: str, gamma: float = 1.0) -> dict:
    response = client.post(
        ANALYZE, json={"points": points, "kernel": kernel, "gamma": gamma}
    )
    assert response.status_code == 200, response.json()
    return response.json()


def _first_axis_splits_the_rings(coordinates: list[dict]) -> bool:
    """Whether every inner point sits to one side of every outer point."""
    first = [entry["first"] for entry in coordinates]
    inner = first[:INNER_RING_COUNT]
    outer = first[INNER_RING_COUNT:]
    return min(inner) > max(outer) or max(inner) < min(outer)


def _reciprocal_squared_length(coefficients: list[float]) -> float:
    """The raw eigenvalue read off the coefficients alone.

    The library divides the unit eigenvector by the square root of the raw
    eigenvalue, so the coefficients' squared length is one over that value.
    """
    return 1.0 / sum(value * value for value in coefficients)


class TestTheWorkedFive:
    """The rotated ellipse under the linear kernel, every figure by hand."""

    def test_five_rows_so_the_divisor_is_four(self) -> None:
        body = _analyze(WORKED_FIVE, "linear")

        assert body["n_rows"] == 5

    def test_the_raw_eigenvalues_are_sixteen_and_four(self) -> None:
        body = _analyze(WORKED_FIVE, "linear")

        first, second = body["components"]
        assert abs(first["raw_eigenvalue"] - 16.0) < 1e-9
        assert abs(second["raw_eigenvalue"] - 4.0) < 1e-9

    def test_the_variances_are_the_raw_eigenvalues_over_four(self) -> None:
        body = _analyze(WORKED_FIVE, "linear")

        first, second = body["components"]
        assert abs(first["variance"] - 4.0) < 1e-9
        assert abs(second["variance"] - 1.0) < 1e-9

    def test_the_total_variance_is_five(self) -> None:
        body = _analyze(WORKED_FIVE, "linear")

        assert abs(body["total_variance"] - 5.0) < 1e-9

    def test_the_shares_are_four_fifths_and_one_fifth(self) -> None:
        body = _analyze(WORKED_FIVE, "linear")

        first, second = body["components"]
        assert abs(first["share"] - 0.8) < 1e-9
        assert abs(second["share"] - 0.2) < 1e-9

    def test_the_raw_eigenvalue_agrees_with_the_coefficients_squared_length(
        self,
    ) -> None:
        # Two routes to the same number: the reported variance times n - 1,
        # and one over the squared length of the coefficients, which is what
        # dividing a unit eigenvector by sqrt(raw) leaves behind.
        body = _analyze(WORKED_FIVE, "linear")

        for component in body["components"]:
            by_length = _reciprocal_squared_length(component["row_coefficients"])
            assert abs(component["raw_eigenvalue"] - by_length) < 1e-9

    def test_the_first_direction_is_built_from_the_first_two_rows(self) -> None:
        # The eigenvector for 16 is (1, -1, 0, 0, 0) / sqrt 2, and dividing by
        # sqrt 16 leaves 0.1768 on the first two rows and nothing else.
        body = _analyze(WORKED_FIVE, "linear")

        coefficients = body["components"][0]["row_coefficients"]
        assert len(coefficients) == 5
        assert abs(abs(coefficients[0]) - FIRST_COEFFICIENT) < 1e-9
        assert abs(coefficients[0] + coefficients[1]) < 1e-9
        assert all(abs(value) < 1e-9 for value in coefficients[2:])

    def test_the_second_direction_is_built_from_rows_three_and_four(self) -> None:
        body = _analyze(WORKED_FIVE, "linear")

        coefficients = body["components"][1]["row_coefficients"]
        assert abs(abs(coefficients[2]) - SECOND_COEFFICIENT) < 1e-9
        assert abs(coefficients[2] + coefficients[3]) < 1e-9
        assert all(abs(coefficients[index]) < 1e-9 for index in (0, 1, 4))

    def test_the_first_kernel_row_is_eight_minus_eight_and_zeros(self) -> None:
        body = _analyze(WORKED_FIVE, "linear")

        kernel_row = body["first_row_kernel_values"]
        assert len(kernel_row) == 5
        assert abs(kernel_row[0] - 8.0) < 1e-9
        assert abs(kernel_row[1] + 8.0) < 1e-9
        assert all(abs(value) < 1e-9 for value in kernel_row[2:])

    def test_the_third_row_kernel_values_are_two_minus_two_and_zeros(self) -> None:
        # The endpoint reports the first row's kernel values, so send the
        # points with row 3 first to read the block of twos.
        reordered = [WORKED_FIVE[2], WORKED_FIVE[3]] + WORKED_FIVE[:2] + WORKED_FIVE[4:]

        body = _analyze(reordered, "linear")

        kernel_row = body["first_row_kernel_values"]
        assert abs(kernel_row[0] - 2.0) < 1e-9
        assert abs(kernel_row[1] + 2.0) < 1e-9
        assert all(abs(value) < 1e-9 for value in kernel_row[2:])

    def test_the_first_row_lands_at_two_root_two(self) -> None:
        body = _analyze(WORKED_FIVE, "linear")

        first_row = body["coordinates"][0]
        assert abs(abs(first_row["first"]) - FIRST_COORDINATE) < 1e-9
        assert abs(first_row["second"]) < 1e-9

    def test_the_third_row_lands_at_root_two_along_the_second(self) -> None:
        body = _analyze(WORKED_FIVE, "linear")

        third_row = body["coordinates"][2]
        assert abs(third_row["first"]) < 1e-9
        assert abs(abs(third_row["second"]) - THIRD_ROW_SECOND_COORDINATE) < 1e-9

    def test_the_coordinate_is_the_kernel_row_against_the_coefficients(
        self,
    ) -> None:
        # The mechanism's sum, checked on the page's numbers: 8 times 0.1768
        # plus -8 times -0.1768 is 2.8284.
        body = _analyze(WORKED_FIVE, "linear")

        coefficients = body["components"][0]["row_coefficients"]
        summed = sum(
            kernel_value * coefficient
            for kernel_value, coefficient in zip(
                body["first_row_kernel_values"], coefficients, strict=True
            )
        )
        assert abs(summed - body["coordinates"][0]["first"]) < 1e-9

    def test_the_linear_kernel_is_ordinary_pca_up_to_a_sign(self) -> None:
        body = _analyze(WORKED_FIVE, "linear")

        for kernel, ordinary in zip(
            body["components"], body["ordinary"]["components"], strict=True
        ):
            assert abs(kernel["variance"] - ordinary["variance"]) < 1e-9
            assert abs(kernel["share"] - ordinary["share"]) < 1e-9

        for kernel, ordinary in zip(
            body["coordinates"], body["ordinary"]["coordinates"], strict=True
        ):
            assert abs(abs(kernel["first"]) - abs(ordinary["first"])) < 1e-9
            assert abs(abs(kernel["second"]) - abs(ordinary["second"])) < 1e-9


class TestTheRings:
    """Two rings that no straight axis can separate."""

    SPLITTING_GAMMA: ClassVar[float] = 0.5
    TOO_WIDE_GAMMA: ClassVar[float] = 0.1
    TOO_SHARP_GAMMA: ClassVar[float] = 3.0

    def test_the_radial_kernel_splits_the_rings_along_its_first_axis(self) -> None:
        body = _analyze(RING_POINTS, "rbf", self.SPLITTING_GAMMA)

        assert _first_axis_splits_the_rings(body["coordinates"])

    def test_the_shares_at_a_half_are_point_one_three_seven_and_point_one_one_one(
        self,
    ) -> None:
        body = _analyze(RING_POINTS, "rbf", self.SPLITTING_GAMMA)

        first, second = body["components"]
        assert abs(first["share"] - 0.137) < 5e-4
        assert abs(second["share"] - 0.111) < 5e-4
        assert abs(body["total_variance"] - 0.898) < 5e-4

    def test_the_first_direction_pulls_one_ring_against_the_other(self) -> None:
        # Every inner row carries one sign and every outer row the other, so
        # the direction really is inner rows one way, outer rows the other.
        body = _analyze(RING_POINTS, "rbf", self.SPLITTING_GAMMA)

        coefficients = body["components"][0]["row_coefficients"]
        assert len(coefficients) == len(RING_POINTS)
        inner = coefficients[:INNER_RING_COUNT]
        outer = coefficients[INNER_RING_COUNT:]
        assert len({value > 0 for value in inner}) == 1
        assert len({value > 0 for value in outer}) == 1
        assert (inner[0] > 0) != (outer[0] > 0)

    def test_the_two_scales_agree_under_the_radial_kernel_too(self) -> None:
        body = _analyze(RING_POINTS, "rbf", self.SPLITTING_GAMMA)

        assert body["n_rows"] == 36
        for component in body["components"]:
            assert abs(component["raw_eigenvalue"] - 35 * component["variance"]) < 1e-9
            by_length = _reciprocal_squared_length(component["row_coefficients"])
            assert abs(component["raw_eigenvalue"] - by_length) < 1e-9

    def test_too_wide_a_kernel_keeps_the_rings_nested(self) -> None:
        body = _analyze(RING_POINTS, "rbf", self.TOO_WIDE_GAMMA)

        first, second = body["components"]
        assert abs(first["share"] - 0.297) < 5e-4
        assert abs(second["share"] - 0.275) < 5e-4
        assert not _first_axis_splits_the_rings(body["coordinates"])

    def test_too_sharp_a_kernel_loses_the_split_again(self) -> None:
        body = _analyze(RING_POINTS, "rbf", self.TOO_SHARP_GAMMA)

        first, second = body["components"]
        assert abs(first["share"] - 0.057) < 5e-4
        assert abs(second["share"] - 0.0535) < 5e-4
        assert abs(body["total_variance"] - 0.983) < 5e-4
        assert not _first_axis_splits_the_rings(body["coordinates"])

    def test_the_split_holds_from_three_tenths_to_one_point_seven(self) -> None:
        for gamma in (0.3, 1.7):
            assert _first_axis_splits_the_rings(
                _analyze(RING_POINTS, "rbf", gamma)["coordinates"]
            )
        for gamma in (0.2, 1.8):
            assert not _first_axis_splits_the_rings(
                _analyze(RING_POINTS, "rbf", gamma)["coordinates"]
            )

    def test_the_squared_kernel_leaves_the_rings_nested_along_its_first(
        self,
    ) -> None:
        body = _analyze(RING_POINTS, "polynomial", self.SPLITTING_GAMMA)

        assert not _first_axis_splits_the_rings(body["coordinates"])

    def test_the_linear_kernel_only_rotates_the_rings(self) -> None:
        body = _analyze(RING_POINTS, "linear")

        first, second = body["components"]
        assert abs(first["share"] - 0.516) < 5e-4
        assert abs(second["share"] - 0.484) < 5e-4
        assert not _first_axis_splits_the_rings(body["coordinates"])

        for kernel, ordinary in zip(
            body["coordinates"], body["ordinary"]["coordinates"], strict=True
        ):
            assert abs(abs(kernel["first"]) - abs(ordinary["first"])) < 1e-9
            assert abs(abs(kernel["second"]) - abs(ordinary["second"])) < 1e-9

    def test_ordinary_pca_cannot_split_them_whatever_the_kernel(self) -> None:
        body = _analyze(RING_POINTS, "rbf", self.SPLITTING_GAMMA)

        first, second = body["ordinary"]["components"]
        assert abs(first["share"] - 0.516) < 5e-4
        assert abs(second["share"] - 0.484) < 5e-4
        assert not _first_axis_splits_the_rings(body["ordinary"]["coordinates"])


class TestRefusals:
    def test_a_lone_point_is_refused(self) -> None:
        response = client.post(
            ANALYZE, json={"points": WORKED_FIVE[:1], "kernel": "linear"}
        )

        assert response.status_code == 422

    def test_a_line_of_points_has_no_second_direction_in_words(self) -> None:
        line = [{"x": float(step), "y": float(step)} for step in range(1, 5)]

        response = client.post(ANALYZE, json={"points": line, "kernel": "linear"})

        assert response.status_code == 422
        assert "only one direction" in response.json()["detail"]

    def test_identical_points_are_refused_by_the_library(self) -> None:
        same = [{"x": 1.0, "y": 2.0}] * 3

        response = client.post(
            ANALYZE, json={"points": same, "kernel": "rbf", "gamma": 0.5}
        )

        assert response.status_code == 422
        assert "at least one component" in response.json()["detail"]

    def test_an_unknown_kernel_is_refused(self) -> None:
        response = client.post(
            ANALYZE, json={"points": WORKED_FIVE, "kernel": "sigmoid"}
        )

        assert response.status_code == 422

    def test_a_non_positive_gamma_is_refused(self) -> None:
        response = client.post(
            ANALYZE, json={"points": RING_POINTS, "kernel": "rbf", "gamma": 0.0}
        )

        assert response.status_code == 422

    def test_a_gamma_past_the_ceiling_is_refused(self) -> None:
        response = client.post(
            ANALYZE, json={"points": RING_POINTS, "kernel": "rbf", "gamma": 101.0}
        )

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            ANALYZE,
            json={"points": WORKED_FIVE, "kernel": "linear", "degree": 3},
        )

        assert response.status_code == 422
