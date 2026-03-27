"""The statistics primer's endpoints, pinned to the prose's hand arithmetic.

The primer summarises the same five people the regression page fitted, so the
numbers here are the site's running example. Mean height 170, mean weight 68,
variances 50 and 35.2, covariance 40, and a correlation whose square is the
regression page's R-squared of 0.909.
"""

import math
from typing import ClassVar

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

SUMMARIZE = "/primers/statistics/summarize"
SAMPLE = "/primers/statistics/sample"

FIVE_PEOPLE = [
    {"x": 160, "y": 58},
    {"x": 165, "y": 66},
    {"x": 170, "y": 68},
    {"x": 175, "y": 74},
    {"x": 180, "y": 74},
]


class TestSummarize:
    WORKED: ClassVar[dict] = {"points": FIVE_PEOPLE}

    def test_the_means_are_the_worked_averages(self) -> None:
        body = client.post(SUMMARIZE, json=self.WORKED).json()

        assert body["mean_x"] == 170.0
        assert body["mean_y"] == 68.0

    def test_the_variances_use_the_population_divisor(self) -> None:
        # 250 / 5 and 176 / 5, the sums the regression page worked by hand.
        body = client.post(SUMMARIZE, json=self.WORKED).json()

        assert body["variance_x"] == 50.0
        assert abs(body["variance_y"] - 35.2) < 1e-12
        assert abs(body["standard_deviation_x"] - math.sqrt(50.0)) < 1e-12

    def test_the_covariance_is_forty(self) -> None:
        body = client.post(SUMMARIZE, json=self.WORKED).json()

        assert body["covariance"] == 40.0

    def test_the_correlation_squares_to_the_regression_r_squared(self) -> None:
        body = client.post(SUMMARIZE, json=self.WORKED).json()

        correlation = body["correlation"]
        assert abs(correlation - 40.0 / math.sqrt(1760.0)) < 1e-12
        assert abs(correlation**2 - 10.0 / 11.0) < 1e-12

    def test_a_constant_column_has_no_correlation(self) -> None:
        body = client.post(
            SUMMARIZE,
            json={
                "points": [
                    {"x": 170, "y": 58},
                    {"x": 170, "y": 66},
                    {"x": 170, "y": 74},
                ]
            },
        ).json()

        assert body["standard_deviation_x"] == 0.0
        assert body["correlation"] is None

    def test_an_empty_cloud_is_refused(self) -> None:
        response = client.post(SUMMARIZE, json={"points": []})

        assert response.status_code == 422


class TestSample:
    def test_a_press_is_reproducible_from_its_seed(self) -> None:
        first = client.post(SAMPLE, json={"draw": 20, "seed": 7}).json()
        second = client.post(SAMPLE, json={"draw": 20, "seed": 7}).json()

        assert first["values"] == second["values"]

    def test_the_draws_accumulate_and_the_bins_count_them_all(self) -> None:
        opening = client.post(SAMPLE, json={"draw": 10, "seed": 3}).json()
        extended = client.post(
            SAMPLE, json={"values": opening["values"], "draw": 15, "seed": 4}
        ).json()

        assert extended["count"] == 25
        assert sum(each["count"] for each in extended["bins"]) == 25
        assert extended["values"][:10] == opening["values"]

    def test_many_draws_settle_near_the_true_mean(self) -> None:
        body = client.post(SAMPLE, json={"draw": 100, "seed": 11}).json()

        assert body["true_mean"] == 170.0
        assert abs(body["mean"] - 170.0) < 3.0

    def test_the_total_is_capped_rather_than_overdrawn(self) -> None:
        first = client.post(SAMPLE, json={"draw": 100, "seed": 1}).json()
        values = first["values"]
        for seed in range(2, 6):
            body = client.post(
                SAMPLE, json={"values": values, "draw": 100, "seed": seed}
            ).json()
            values = body["values"]

        capped = client.post(
            SAMPLE, json={"values": values, "draw": 100, "seed": 9}
        ).json()

        assert capped["count"] == 500

    def test_an_oversized_draw_is_refused(self) -> None:
        response = client.post(SAMPLE, json={"draw": 101, "seed": 1})

        assert response.status_code == 422
