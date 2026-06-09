"""The three surface endpoints behind the site's three-dimensional pictures.

The loss surface's floor must sit exactly where the regression page's hand
arithmetic put it, slope 0.8, intercept -68 and an RSS of 16, and every
sampled cell must sit at or above that floor, which is what being the minimum
means. The plane fit must recover its coefficients and predict its own lattice
consistently, and the lifted clinic must be separated by a genuinely flat
plane at accuracy 1. The lift endpoint chooses its own map rather than taking
a kernel name, so the request carries the patients and nothing else, and the
accuracy sits inside the summary of the fit that drew the plane.
"""

from typing import ClassVar

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

LOSS = "/concepts/simple-linear-regression/loss-surface"
PLANE = "/concepts/multiple-polynomial-regression/fit-plane"
LIFT = "/concepts/kernel-trick/lift"

FIVE_PEOPLE = [
    {"x": 160, "y": 58},
    {"x": 165, "y": 66},
    {"x": 170, "y": 68},
    {"x": 175, "y": 74},
    {"x": 180, "y": 74},
]

CLINIC = [
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


class TestLossSurface:
    def test_the_floor_is_the_worked_fit(self) -> None:
        body = client.post(LOSS, json={"points": FIVE_PEOPLE}).json()

        minimum = body["minimum"]
        assert abs(minimum["slope"] - 0.8) < 1e-9
        assert abs(minimum["intercept"] + 68.0) < 1e-9
        assert abs(minimum["rss"] - 16.0) < 1e-9

    def test_no_sampled_cell_dips_below_the_floor(self) -> None:
        body = client.post(LOSS, json={"points": FIVE_PEOPLE}).json()

        floor = body["minimum"]["rss"]
        assert all(cell >= floor - 1e-9 for row in body["rss"] for cell in row)

    def test_the_lattice_is_square_and_centred(self) -> None:
        body = client.post(LOSS, json={"points": FIVE_PEOPLE}).json()

        assert len(body["slopes"]) == 21
        assert len(body["intercepts"]) == 21
        assert len(body["rss"]) == 21
        middle = body["slopes"][10]
        assert abs(middle - body["minimum"]["slope"]) < 1e-9


class TestPlaneFit:
    PEOPLE: ClassVar[list[dict]] = [
        {"height": 160, "age": 30, "weight": 60},
        {"height": 170, "age": 30, "weight": 66},
        {"height": 180, "age": 30, "weight": 72},
        {"height": 160, "age": 60, "weight": 66},
        {"height": 170, "age": 60, "weight": 72},
        {"height": 180, "age": 60, "weight": 78},
    ]

    def test_the_clean_plane_is_recovered(self) -> None:
        # weight = 0.6 * height + 0.2 * age - 42 exactly on these six people.
        body = client.post(PLANE, json={"people": self.PEOPLE}).json()

        assert abs(body["height_coefficient"] - 0.6) < 1e-9
        assert abs(body["age_coefficient"] - 0.2) < 1e-9
        assert abs(body["intercept"] + 42.0) < 1e-9
        assert abs(body["r_squared"] - 1.0) < 1e-12

    def test_the_surface_is_the_plane_at_every_corner(self) -> None:
        body = client.post(PLANE, json={"people": self.PEOPLE}).json()

        for row_index, age in enumerate(body["ages_axis"]):
            for column_index, height in enumerate(body["heights_axis"]):
                expected = 0.6 * height + 0.2 * age - 42.0
                assert abs(body["surface"][row_index][column_index] - expected) < 1e-9

    def test_a_fitted_weight_comes_back_for_every_person(self) -> None:
        people = [
            {"height": 160, "age": 20, "weight": 60},
            {"height": 170, "age": 30, "weight": 70},
            {"height": 180, "age": 40, "weight": 75},
            {"height": 190, "age": 50, "weight": 85},
        ]
        body = client.post(PLANE, json={"people": people}).json()

        assert len(body["fitted"]) == len(people)
        for person, fitted in zip(people, body["fitted"], strict=True):
            expected = (
                body["intercept"]
                + body["height_coefficient"] * person["height"]
                + body["age_coefficient"] * person["age"]
            )
            assert abs(fitted - expected) < 1e-9

    def test_too_few_people_for_a_plane_is_refused(self) -> None:
        response = client.post(PLANE, json={"people": self.PEOPLE[:2]})

        assert response.status_code == 422


class TestLift:
    def test_a_flat_plane_separates_the_lifted_clinic(self) -> None:
        body = client.post(LIFT, json={"points": CLINIC}).json()

        assert body["lifted"]["accuracy"] == 1.0

    def test_every_point_sits_on_its_side_of_the_plane(self) -> None:
        body = client.post(LIFT, json={"points": CLINIC}).json()

        plane = body["plane"]
        for point in body["points"]:
            side = (
                plane["a"] * point["u"]
                + plane["b"] * point["v"]
                + plane["c"] * point["w"]
                - plane["d"]
            )
            assert (side > 0) == (point["label"] == 1)

    def test_the_lift_is_the_worked_mapping(self) -> None:
        # Every lifted point obeys v-squared equal to twice u times w, the
        # signature of (x^2, sqrt(2)xy, y^2), so the lift really is the page's.
        body = client.post(LIFT, json={"points": CLINIC}).json()

        for point in body["points"]:
            assert abs(point["v"] ** 2 - 2.0 * point["u"] * point["w"]) < 1e-9
