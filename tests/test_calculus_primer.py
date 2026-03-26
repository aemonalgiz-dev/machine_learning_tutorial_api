"""The calculus primer: gradient descent on a fixed curve, drawn for the widget.

The endpoint's whole job is to produce a trajectory that shows three things: a
gentle rate settles at the bottom, the starting side decides which valley a
two-valley curve falls into, and a rate that is too large runs away. Each is a
test here.
"""

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app
from oop_ml_api.primers.calculus import CURVE_SAMPLES, MAX_LEARNING_RATE

client = TestClient(create_app())

DESCEND = "/primers/calculus/descend"


class TestDescent:
    def test_a_gentle_rate_settles_at_the_bowl_minimum(self) -> None:
        response = client.post(
            DESCEND,
            json={
                "function": "bowl",
                "start": 5.0,
                "learning_rate": 0.3,
                "steps": 100,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["outcome"] == "converged"
        # 0.5 (x - 1)^2 + 2 has its minimum at x = 1.
        assert abs(body["minimum"]["x"] - 1.0) < 1e-2
        assert abs(body["minimum"]["y"] - 2.0) < 1e-2

    def test_the_slope_falls_toward_zero_along_the_walk(self) -> None:
        response = client.post(
            DESCEND,
            json={"function": "bowl", "start": 5.0, "learning_rate": 0.3},
        )

        path = response.json()["path"]
        assert abs(path[-1]["slope"]) < abs(path[0]["slope"])
        assert abs(path[-1]["slope"]) < 1e-2

    def test_the_starting_side_picks_the_valley(self) -> None:
        left = client.post(
            DESCEND,
            json={"function": "valley", "start": -3.0, "learning_rate": 0.1},
        ).json()
        right = client.post(
            DESCEND,
            json={"function": "valley", "start": 3.0, "learning_rate": 0.1},
        ).json()

        # The two-valley curve has a minimum either side of a hill at the origin.
        assert left["path"][-1]["x"] < 0.0
        assert right["path"][-1]["x"] > 0.0

    def test_a_rate_that_is_too_large_diverges(self) -> None:
        response = client.post(
            DESCEND,
            json={"function": "bowl", "start": 5.0, "learning_rate": 3.0},
        )

        body = response.json()
        assert body["outcome"] == "diverged"
        assert body["minimum"] is None


class TestTangent:
    def test_the_slope_is_zero_at_the_bowl_minimum(self) -> None:
        # 0.5 (x - 1)^2 + 2 turns around at x = 1, where its slope is zero.
        body = client.post(
            "/primers/calculus/tangent", json={"function": "bowl", "x": 1.0}
        ).json()

        assert abs(body["point"]["slope"]) < 1e-9

    def test_the_slope_is_negative_on_the_way_down(self) -> None:
        body = client.post(
            "/primers/calculus/tangent", json={"function": "bowl", "x": -2.0}
        ).json()

        assert body["point"]["slope"] < 0.0

    def test_the_slope_is_positive_on_the_way_up(self) -> None:
        body = client.post(
            "/primers/calculus/tangent", json={"function": "bowl", "x": 4.0}
        ).json()

        assert body["point"]["slope"] > 0.0

    def test_a_point_dragged_off_the_curve_is_pulled_back_on(self) -> None:
        body = client.post(
            "/primers/calculus/tangent", json={"function": "bowl", "x": 999.0}
        ).json()

        assert body["point"]["x"] == 6.0  # the bowl's right edge


class TestDrawing:
    def test_the_curve_spans_its_window(self) -> None:
        body = client.post(
            DESCEND,
            json={"function": "bowl", "start": 0.0, "learning_rate": 0.3},
        ).json()

        assert len(body["curve"]) == CURVE_SAMPLES
        window = body["window"]
        assert window["x_min"] < window["x_max"]
        assert window["y_min"] < window["y_max"]


class TestRefusals:
    def test_an_unknown_function_is_refused(self) -> None:
        response = client.post(
            DESCEND,
            json={"function": "spiral", "start": 0.0, "learning_rate": 0.3},
        )

        assert response.status_code == 422

    def test_a_non_positive_rate_is_refused(self) -> None:
        response = client.post(
            DESCEND,
            json={"function": "bowl", "start": 0.0, "learning_rate": 0.0},
        )

        assert response.status_code == 422

    def test_a_rate_above_the_ceiling_is_refused(self) -> None:
        response = client.post(
            DESCEND,
            json={
                "function": "bowl",
                "start": 0.0,
                "learning_rate": MAX_LEARNING_RATE + 1.0,
            },
        )

        assert response.status_code == 422
