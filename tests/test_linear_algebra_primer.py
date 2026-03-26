"""The linear algebra primer's endpoints, pinned to the prose's hand arithmetic.

The primer works its numbers by hand, the 3-4-5 length, the dot product of 24,
the eigen directions of [[2, 1], [1, 2]], and these tests assert the endpoints
return exactly those numbers, so the page and the widgets cannot drift apart.
"""

import math
from typing import ClassVar

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

VECTORS = "/primers/linear-algebra/vectors"
TRANSFORM = "/primers/linear-algebra/transform"


class TestVectors:
    def test_the_three_four_five_length(self) -> None:
        body = client.post(
            VECTORS,
            json={"first": {"x": 3, "y": 4}, "second": {"x": 4, "y": 3}},
        ).json()

        assert body["length_first"] == 5.0
        assert body["length_second"] == 5.0

    def test_the_worked_dot_product(self) -> None:
        # (3, 4) . (4, 3) = 12 + 12 = 24, and the cosine is 24 / 25.
        body = client.post(
            VECTORS,
            json={"first": {"x": 3, "y": 4}, "second": {"x": 4, "y": 3}},
        ).json()

        assert body["dot"] == 24.0
        assert abs(body["cosine"] - 0.96) < 1e-12

    def test_perpendicular_vectors_dot_to_zero(self) -> None:
        body = client.post(
            VECTORS,
            json={"first": {"x": 3, "y": 4}, "second": {"x": -4, "y": 3}},
        ).json()

        assert body["dot"] == 0.0
        assert body["cosine"] == 0.0

    def test_distance_is_the_length_of_the_difference(self) -> None:
        # From (1, 2) to (4, 6) the difference is (3, 4), so the distance is 5.
        body = client.post(
            VECTORS,
            json={"first": {"x": 1, "y": 2}, "second": {"x": 4, "y": 6}},
        ).json()

        assert body["distance"] == 5.0

    def test_a_zero_vector_has_no_cosine(self) -> None:
        body = client.post(
            VECTORS,
            json={"first": {"x": 0, "y": 0}, "second": {"x": 4, "y": 3}},
        ).json()

        assert body["cosine"] is None

    def test_an_out_of_range_coordinate_is_refused(self) -> None:
        response = client.post(
            VECTORS,
            json={"first": {"x": 11, "y": 0}, "second": {"x": 1, "y": 1}},
        )

        assert response.status_code == 422


class TestTransform:
    WORKED: ClassVar[dict[str, int]] = {"a": 2, "b": 1, "c": 1, "d": 2}

    def test_the_basis_images_are_the_columns(self) -> None:
        body = client.post(TRANSFORM, json=self.WORKED).json()

        assert body["basis_first"] == {"x": 2.0, "y": 1.0}
        assert body["basis_second"] == {"x": 1.0, "y": 2.0}

    def test_the_far_corner_of_the_unit_square_lands_at_three_three(self) -> None:
        # (1, 1) is an eigen direction of the worked matrix with factor 3.
        body = client.post(TRANSFORM, json=self.WORKED).json()

        assert body["unit_square"][2] == {"x": 3.0, "y": 3.0}

    def test_the_worked_matrix_has_eigen_factors_three_and_one(self) -> None:
        body = client.post(TRANSFORM, json=self.WORKED).json()

        assert body["has_real_eigen"] is True
        first, second = body["eigen"]
        assert abs(first["value"] - 3.0) < 1e-9
        assert abs(second["value"] - 1.0) < 1e-9
        # The stretched direction lies along (1, 1) and the kept one along
        # (1, -1), both reported at unit length.
        assert abs(first["x"] - first["y"]) < 1e-9
        assert abs(second["x"] + second["y"]) < 1e-9
        assert abs(math.hypot(first["x"], first["y"]) - 1.0) < 1e-9

    def test_a_rotation_has_no_real_eigen_direction(self) -> None:
        body = client.post(TRANSFORM, json={"a": 0, "b": -1, "c": 1, "d": 0}).json()

        assert body["has_real_eigen"] is False
        assert body["eigen"] == []

    def test_a_shear_reports_its_one_direction_once(self) -> None:
        body = client.post(TRANSFORM, json={"a": 1, "b": 1, "c": 0, "d": 1}).json()

        assert len(body["eigen"]) == 1
        assert abs(body["eigen"][0]["value"] - 1.0) < 1e-9
        assert abs(body["eigen"][0]["y"]) < 1e-6

    def test_grid_lines_come_back_one_per_line(self) -> None:
        body = client.post(TRANSFORM, json=self.WORKED).json()

        # Seven vertical and seven horizontal lines for extent three.
        assert len(body["grid_lines"]) == 14

    def test_an_oversized_entry_is_refused(self) -> None:
        response = client.post(TRANSFORM, json={"a": 99, "b": 0, "c": 0, "d": 1})

        assert response.status_code == 422
