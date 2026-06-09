"""The bouncer at the door: limits and the library-error mapping.

Two things carry the API's safety. The input bounds refuse a request before it
becomes a compute job, and the error handler turns every typed library refusal
into a 422 the website can show a learner. Both are tested here because both are
the point of the layer -- the endpoint's happy path is the small part.
"""

from fastapi.testclient import TestClient

from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS
from oop_ml_api.main import create_app

client = TestClient(create_app())

FIT = "/concepts/simple-linear-regression/fit"


class TestInputLimits:
    def test_fewer_than_two_points_is_refused(self) -> None:
        response = client.post(FIT, json={"points": [{"x": 1.0, "y": 2.0}]})

        assert response.status_code == 422

    def test_too_many_points_is_refused_before_computing(self) -> None:
        many = [{"x": float(i), "y": float(i)} for i in range(MAX_POINTS + 1)]

        response = client.post(FIT, json={"points": many})

        assert response.status_code == 422

    def test_an_out_of_range_value_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={
                "points": [
                    {"x": 0.0, "y": 0.0},
                    {"x": MAX_MAGNITUDE * 10, "y": 1.0},
                ]
            },
        )

        assert response.status_code == 422


class TestLibraryErrorMapping:
    def test_a_degenerate_fit_becomes_a_typed_422(self) -> None:
        """Every x identical: no line is determined, and the library says so.

        The API does not anticipate this case; the library raises its own typed
        error and the shared handler turns it into a 422 carrying that error's
        name and message, which is exactly what a learner needs to see.
        """
        response = client.post(
            FIT,
            json={
                "points": [
                    {"x": 3.0, "y": 1.0},
                    {"x": 3.0, "y": 2.0},
                    {"x": 3.0, "y": 5.0},
                ]
            },
        )

        assert response.status_code == 422
        body = response.json()
        assert "error" in body and "detail" in body

    def test_malformed_json_is_a_422_not_a_500(self) -> None:
        response = client.post(FIT, json={"points": [{"x": 1.0}]})

        assert response.status_code == 422

    def test_a_nan_in_the_request_is_a_readable_422_not_a_500(self) -> None:
        """A NaN is refused before computing, and the refusal must be writable.

        The default validation handler echoes the offending value, and a NaN
        cannot be written as JSON, so the echo used to fail and the learner saw
        a 500. The shared handler now answers with the location and the message
        and never the value.
        """
        response = client.post(
            FIT,
            content='{"points": [{"x": NaN, "y": 1.0}, {"x": 2.0, "y": 3.0}]}',
            headers={"content-type": "application/json"},
        )

        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "ValidationError"
        assert isinstance(body["detail"], str)
        assert "points 0 x" in body["detail"]


class TestMeta:
    def test_health_is_ok(self) -> None:
        assert client.get("/health").json() == {"status": "ok"}
