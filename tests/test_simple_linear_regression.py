"""The simple-linear-regression endpoint, end to end through HTTP.

These go through the real ASGI app with a test client, so they exercise the
request parsing, the library call, and the response shape together -- the whole
translation the API exists to be. The numbers are checked against a line the
points were generated from, so a wrong slope or a mislabelled residual fails
here rather than in the browser.
"""

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())


def points_on_a_line(slope: float, intercept: float, xs: list[float]) -> list[dict]:
    return [{"x": x, "y": slope * x + intercept} for x in xs]


class TestFitting:
    def test_it_recovers_the_line_the_points_lie_on(self) -> None:
        response = client.post(
            "/concepts/simple-linear-regression/fit",
            json={"points": points_on_a_line(2.0, 1.0, [0.0, 1.0, 2.0, 3.0, 4.0])},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["slope"] == 2.0
        assert body["intercept"] == 1.0
        assert body["r_squared"] == 1.0

    def test_the_line_spans_the_observed_x_range(self) -> None:
        response = client.post(
            "/concepts/simple-linear-regression/fit",
            json={"points": points_on_a_line(2.0, 1.0, [1.0, 5.0, 3.0])},
        )

        line = response.json()["line"]
        assert line["x_start"] == 1.0
        assert line["x_end"] == 5.0
        assert line["y_start"] == 3.0
        assert line["y_end"] == 11.0

    def test_residuals_are_zero_on_collinear_points(self) -> None:
        response = client.post(
            "/concepts/simple-linear-regression/fit",
            json={"points": points_on_a_line(-1.0, 4.0, [0.0, 1.0, 2.0, 3.0])},
        )

        assert all(abs(residual) < 1e-9 for residual in response.json()["residuals"])

    def test_residuals_are_signed_and_one_per_point(self) -> None:
        """A point above the fitted line has a positive residual, below negative."""
        response = client.post(
            "/concepts/simple-linear-regression/fit",
            json={
                "points": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 1.0, "y": 0.0},
                    {"x": 2.0, "y": 2.0},
                    {"x": 3.0, "y": 2.0},
                ]
            },
        )

        residuals = response.json()["residuals"]
        assert len(residuals) == 4
        assert any(residual > 0 for residual in residuals)
        assert any(residual < 0 for residual in residuals)
