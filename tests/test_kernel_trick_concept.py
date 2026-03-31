"""The kernel trick endpoint, pinned to the page's clinic story.

The healthy patients sit in a band of ordinary vitals with the unwell on
every side of them, so a straight boundary cannot cut them out, and the page
quotes the linear kernel stalling near 0.7 while the squared and radial
kernels separate the clinic perfectly. Pinned here.
"""

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

CLASSIFY = "/concepts/kernel-trick/classify"

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


class TestKernelTrick:
    def test_a_straight_boundary_cannot_cut_out_the_surrounded(self) -> None:
        body = client.post(CLASSIFY, json={"points": CLINIC, "kernel": "linear"}).json()

        assert body["accuracy"] < 0.8

    def test_the_squared_kernel_separates_the_clinic(self) -> None:
        body = client.post(
            CLASSIFY, json={"points": CLINIC, "kernel": "polynomial"}
        ).json()

        assert body["accuracy"] == 1.0

    def test_the_radial_kernel_separates_the_clinic(self) -> None:
        body = client.post(CLASSIFY, json={"points": CLINIC, "kernel": "rbf"}).json()

        assert body["accuracy"] == 1.0

    def test_the_radial_regions_close_around_the_healthy(self) -> None:
        # The healthy region is bounded: walk any row of the lattice and the
        # healthy cells, where there are any, are flanked by unwell cells
        # rather than running to either edge.
        body = client.post(CLASSIFY, json={"points": CLINIC, "kernel": "rbf"}).json()

        grid = body["regions"]["labels"]
        edge_cells = (
            [row[0] for row in grid] + [row[-1] for row in grid] + grid[0] + grid[-1]
        )
        assert set(edge_cells) == {0}
        assert any(1 in row for row in grid)

    def test_an_unknown_kernel_is_refused(self) -> None:
        response = client.post(CLASSIFY, json={"points": CLINIC, "kernel": "sigmoid"})

        assert response.status_code == 422
