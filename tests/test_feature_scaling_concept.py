"""The feature-scaling endpoint, pinned to the page's hand arithmetic.

The page works five values, 9, 19, 24, 29 and 39, chosen so that every centre
and spread the library reads off them is a whole number, mean 24, standard
deviation 10, smallest 9, range 30, largest magnitude 39, median 24, quartiles
19 and 29 so an interquartile range of 10, and root mean square 26. The 39
scales to 1.5, 1.0, 1.0, 1.5 and 1.5 under the five methods. Replacing it with
99 leaves the middle three sorted values where they were, so the robust centre
and spread cannot move while the min-max and max-abs spreads triple. All pinned.
"""

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

SCALE = "/concepts/feature-scaling/scale"

TIDY_SAMPLE = [9, 19, 24, 29, 39]

WITH_AN_OUTLIER = [9, 19, 24, 29, 99]

METHODS = ("standardize", "min_max", "max_abs", "robust", "root_mean_square")


def scalings_for(values: list[float]) -> dict:
    response = client.post(SCALE, json={"values": values})
    assert response.status_code == 200
    return response.json()["scalings"]


class TestTheWorkedExample:
    def test_every_method_comes_back_by_name(self) -> None:
        scalings = scalings_for(TIDY_SAMPLE)

        assert tuple(scalings) == METHODS

    def test_standardizing_reads_the_mean_and_standard_deviation(self) -> None:
        scaling = scalings_for(TIDY_SAMPLE)["standardize"]

        assert scaling["centre"] == 24.0
        assert scaling["spread"] == 10.0

    def test_min_max_reads_the_smallest_value_and_the_range(self) -> None:
        scaling = scalings_for(TIDY_SAMPLE)["min_max"]

        assert scaling["centre"] == 9.0
        assert scaling["spread"] == 30.0

    def test_max_abs_reads_zero_and_the_largest_magnitude(self) -> None:
        scaling = scalings_for(TIDY_SAMPLE)["max_abs"]

        assert scaling["centre"] == 0.0
        assert scaling["spread"] == 39.0

    def test_robust_reads_the_median_and_the_interquartile_range(self) -> None:
        scaling = scalings_for(TIDY_SAMPLE)["robust"]

        assert scaling["centre"] == 24.0
        assert scaling["spread"] == 10.0

    def test_root_mean_square_reads_zero_and_the_root_mean_square(self) -> None:
        scaling = scalings_for(TIDY_SAMPLE)["root_mean_square"]

        assert scaling["centre"] == 0.0
        assert scaling["spread"] == 26.0

    def test_the_thirty_nine_lands_where_the_page_says(self) -> None:
        scalings = scalings_for(TIDY_SAMPLE)

        landed = [scalings[method]["scaled"][4] for method in METHODS]
        assert landed == [1.5, 1.0, 1.0, 1.5, 1.5]

    def test_the_whole_column_is_scaled_in_input_order(self) -> None:
        scalings = scalings_for(TIDY_SAMPLE)

        assert scalings["standardize"]["scaled"] == [-1.5, -0.5, 0.0, 0.5, 1.5]
        assert scalings["robust"]["scaled"] == [-1.5, -0.5, 0.0, 0.5, 1.5]
        assert scalings["min_max"]["scaled"] == pytest.approx(
            [0.0, 1 / 3, 0.5, 2 / 3, 1.0]
        )

    def test_the_order_supplied_is_the_order_returned(self) -> None:
        shuffled = [29, 9, 39, 24, 19]
        scalings = scalings_for(shuffled)

        assert scalings["standardize"]["scaled"] == [0.5, -1.5, 1.5, 0.0, -0.5]
        assert scalings["min_max"]["scaled"][2] == 1.0
        assert scalings["min_max"]["scaled"][1] == 0.0


class TestWhatEachMethodPromises:
    def test_standardized_values_have_mean_zero_and_variance_one(self) -> None:
        scaled = scalings_for([3, 7, 8, 12, 20, 31])["standardize"]["scaled"]

        mean = sum(scaled) / len(scaled)
        variance = sum((value - mean) ** 2 for value in scaled) / len(scaled)
        assert mean == pytest.approx(0.0, abs=1e-12)
        assert variance == pytest.approx(1.0)

    def test_min_max_fills_the_unit_interval_exactly(self) -> None:
        scaled = scalings_for([3, 7, 8, 12, 20, 31])["min_max"]["scaled"]

        assert min(scaled) == 0.0
        assert max(scaled) == 1.0

    def test_max_abs_keeps_a_zero_at_zero_and_the_extreme_at_one(self) -> None:
        scaled = scalings_for([-8, 0, 2, 4])["max_abs"]["scaled"]

        assert scaled == [-1.0, 0.0, 0.25, 0.5]

    def test_root_mean_square_agrees_with_standardizing_on_a_centred_column(
        self,
    ) -> None:
        scalings = scalings_for([-3, -1, 0, 1, 3])

        assert scalings["root_mean_square"]["spread"] == pytest.approx(
            scalings["standardize"]["spread"]
        )
        assert scalings["root_mean_square"]["scaled"] == pytest.approx(
            scalings["standardize"]["scaled"]
        )


class TestWhenTheScaleLies:
    def test_the_outlier_moves_the_mean_to_thirty_six(self) -> None:
        scaling = scalings_for(WITH_AN_OUTLIER)["standardize"]

        assert scaling["centre"] == 36.0
        assert scaling["spread"] > 30.0

    def test_the_outlier_triples_the_min_max_spread(self) -> None:
        scaling = scalings_for(WITH_AN_OUTLIER)["min_max"]

        assert scaling["centre"] == 9.0
        assert scaling["spread"] == 90.0

    def test_the_outlier_sets_the_max_abs_spread(self) -> None:
        scaling = scalings_for(WITH_AN_OUTLIER)["max_abs"]

        assert scaling["spread"] == 99.0

    def test_the_robust_centre_and_spread_do_not_move_at_all(self) -> None:
        tidy = scalings_for(TIDY_SAMPLE)["robust"]
        outlying = scalings_for(WITH_AN_OUTLIER)["robust"]

        assert (tidy["centre"], tidy["spread"]) == (24.0, 10.0)
        assert (outlying["centre"], outlying["spread"]) == (24.0, 10.0)
        assert outlying["scaled"][:4] == tidy["scaled"][:4]
        assert outlying["scaled"][4] == 7.5

    def test_the_four_tidy_values_are_squashed_into_the_first_third(self) -> None:
        scalings = scalings_for(WITH_AN_OUTLIER)

        assert scalings["min_max"]["scaled"][3] == pytest.approx(2 / 9)
        assert scalings["max_abs"]["scaled"][3] == pytest.approx(29 / 99)
        assert all(value < 1 / 3 for value in scalings["min_max"]["scaled"][:4])
        assert all(value < 1 / 3 for value in scalings["max_abs"]["scaled"][:4])


class TestRefusals:
    def test_all_the_same_values_are_refused_by_the_library(self) -> None:
        response = client.post(SCALE, json={"values": [7, 7, 7, 7]})

        assert response.status_code == 422
        assert response.json()["error"] == "AllSameValuesError"

    def test_seven_of_eight_identical_is_refused_for_the_robust_spread(self) -> None:
        # Both quartiles land inside the repeated run, so the interquartile range
        # is zero while the column still varies. The library refuses, by name.
        response = client.post(SCALE, json={"values": [5, 5, 5, 5, 5, 5, 5, 8]})

        assert response.status_code == 422
        assert response.json()["error"] == "AllSameValuesError"

    def test_a_lone_value_is_refused(self) -> None:
        response = client.post(SCALE, json={"values": [24]})

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(SCALE, json={"values": TIDY_SAMPLE, "method": "robust"})

        assert response.status_code == 422
