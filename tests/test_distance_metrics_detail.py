"""The distance-metrics page's further endpoints, pinned to what the page quotes.

The axioms endpoint puts three people of one height through the four checks
and cosine fails the triangle inequality on them, the two legs summing to
0.00199 against a direct 0.00396, while the five others pass. The between
endpoint's Euclidean routes agree at 5 on the worked pair and part at
coordinates near a million, where the raw expansion answers 0.0 for a gap
of a millionth. Rescaling one person's whole row leaves cosine alone and
rescaling one column does not, which is the claim the old page had backwards.
The units endpoint shows weight in grams handing the arithmetic to one
column, 0.512 down to a millionth, and weight in tonnes reordering the three
nearest. The models endpoint finds three, three and two groups under the
three p-norms at a radius of twenty, and the dimensions endpoint's contrast
falls from 3248 in one feature to 0.238 in two hundred. All pinned.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BETWEEN = "/concepts/distance-metrics/between"
AXIOMS = "/concepts/distance-metrics/axioms"
UNITS = "/concepts/distance-metrics/units"
MODELS = "/concepts/distance-metrics/models"
DIMENSIONS = "/concepts/distance-metrics/dimensions"
NEIGHBOURS = "/concepts/distance-metrics/neighbours"

ALL_SIX = ("euclidean", "manhattan", "chebyshev", "cosine", "hamming", "canberra")

# The k-nearest page's borderline crowd, five children and six adults.
CROWD = [
    {"x": 147, "y": 41, "label": 0},
    {"x": 156, "y": 53, "label": 1},
    {"x": 145, "y": 57, "label": 0},
    {"x": 159, "y": 57, "label": 1},
    {"x": 162, "y": 61, "label": 1},
    {"x": 120, "y": 25, "label": 0},
    {"x": 122, "y": 28, "label": 0},
    {"x": 118, "y": 24, "label": 0},
    {"x": 180, "y": 80, "label": 1},
    {"x": 183, "y": 83, "label": 1},
    {"x": 178, "y": 78, "label": 1},
]
PLAIN = [{"x": person["x"], "y": person["y"]} for person in CROWD]
QUERY = {"x": 150, "y": 45}
NEAREST = {"x": 147, "y": 41}

# Three people of one height, weights 45, 52 and 60, so every p-norm makes
# the two legs sum to the direct route exactly and cosine does not.
SAME_HEIGHT = {
    "first": {"x": 150, "y": 45},
    "second": {"x": 150, "y": 60},
    "third": {"x": 150, "y": 52},
}


def pair(
    first: dict[str, float], second: dict[str, float], **rest: Any
) -> dict[str, Any]:
    response = client.post(BETWEEN, json={"first": first, "second": second, **rest})
    assert response.status_code == 200, response.text
    return response.json()


def by_metric(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {entry["metric"]: entry for entry in entries}


class TestBetweenFurther:
    def test_an_order_below_one_is_accepted_and_is_not_a_metric(self) -> None:
        body = pair(QUERY, NEAREST, order=0.5)

        assert body["at_order"] == pytest.approx(13.9282, abs=1e-4)
        assert body["at_order"] > body["distances"]["manhattan"]

    def test_an_order_of_three_sits_between_euclidean_and_chebyshev(self) -> None:
        body = pair(QUERY, NEAREST, order=3)

        assert body["at_order"] == pytest.approx(4.4979, abs=1e-4)
        assert 4.0 < body["at_order"] < 5.0

    def test_no_order_asked_for_means_none_back(self) -> None:
        assert pair(QUERY, NEAREST)["at_order"] is None

    @pytest.mark.parametrize("order", [0, -1, 200])
    def test_an_order_outside_the_bounds_is_refused(self, order: float) -> None:
        response = client.post(
            BETWEEN, json={"first": QUERY, "second": NEAREST, "order": order}
        )

        assert response.status_code == 422

    def test_the_three_euclidean_routes_agree_on_the_worked_pair(self) -> None:
        routes = pair(QUERY, NEAREST)["euclidean_routes"]

        assert routes["direct"] == 5.0
        assert routes["library"] == 5.0
        assert routes["expanded_raw"] == pytest.approx(5.0, abs=1e-12)

    def test_the_raw_expansion_loses_a_gap_of_a_millionth_near_a_million(self) -> None:
        far = {"x": 999999.0, "y": 999999.0}
        body = pair(far, {"x": 999999.000001, "y": 999999.0})
        routes = body["euclidean_routes"]

        assert routes["direct"] == pytest.approx(1.0e-6, rel=1e-4)
        assert routes["expanded_raw"] == 0.0
        assert routes["library"] == routes["direct"]

    def test_the_raw_expansion_loses_a_gap_of_a_thousandth_too(self) -> None:
        far = {"x": 999999.0, "y": 999999.0}
        routes = pair(far, {"x": 999999.001, "y": 999999.0})["euclidean_routes"]

        assert routes["direct"] == pytest.approx(1.0e-3, rel=1e-6)
        assert routes["expanded_raw"] == 0.0
        assert routes["library"] == routes["direct"]

    def test_a_gap_of_one_near_a_million_survives_every_route(self) -> None:
        far = {"x": 999999.0, "y": 999999.0}
        routes = pair(far, {"x": 1000000.0, "y": 999999.0})["euclidean_routes"]

        assert routes == {"direct": 1.0, "expanded_raw": 1.0, "library": 1.0}

    @pytest.mark.parametrize("factor", [2, 10])
    def test_scaling_a_whole_row_leaves_cosine_alone_and_nothing_else(
        self, factor: int
    ) -> None:
        plain = pair(QUERY, NEAREST)["distances"]
        scaled = pair(QUERY, {"x": NEAREST["x"] * factor, "y": NEAREST["y"] * factor})[
            "distances"
        ]

        assert scaled["cosine"] == plain["cosine"]
        assert scaled["euclidean"] > 20 * plain["euclidean"]
        assert scaled["canberra"] > 10 * plain["canberra"]

    def test_scaling_a_row_by_two_is_the_pinned_numbers(self) -> None:
        scaled = pair(QUERY, {"x": 294, "y": 82})["distances"]

        assert scaled["euclidean"] == pytest.approx(148.6775, abs=1e-4)
        assert scaled["manhattan"] == 181.0
        assert scaled["chebyshev"] == 144.0
        assert scaled["cosine"] == pytest.approx(0.000189, abs=1e-6)
        assert scaled["canberra"] == pytest.approx(0.6157, abs=1e-4)

    def test_weight_in_grams_changes_cosine_and_not_canberra(self) -> None:
        kilograms = pair(QUERY, NEAREST)["distances"]
        grams = pair({"x": 150, "y": 45000}, {"x": 147, "y": 41000})["distances"]

        assert grams["euclidean"] == pytest.approx(4000.0011, abs=1e-4)
        assert grams["manhattan"] == 4003.0
        assert grams["chebyshev"] == 4000.0
        assert grams["cosine"] == pytest.approx(3.176e-8, rel=1e-3)
        assert grams["cosine"] < kilograms["cosine"] / 1000
        assert grams["canberra"] == kilograms["canberra"]
        assert grams["hamming"] == kilograms["hamming"]

    def test_hamming_reads_codes_and_ignores_how_far_apart_they_are(self) -> None:
        two_apart = pair({"x": 0, "y": 5}, {"x": 2, "y": 5})["distances"]
        one_apart = pair({"x": 0, "y": 5}, {"x": 1, "y": 5})["distances"]

        assert two_apart["hamming"] == 0.5
        assert one_apart["hamming"] == 0.5
        assert two_apart["euclidean"] == 2.0
        assert one_apart["euclidean"] == 1.0
        assert two_apart["canberra"] == 1.0
        assert two_apart["cosine"] == pytest.approx(0.0715, abs=1e-4)

    def test_hamming_counts_a_rounding_error_as_a_disagreement(self) -> None:
        body = pair({"x": 0.3, "y": 5}, {"x": 0.1 + 0.2, "y": 5})["distances"]

        assert body["hamming"] == 0.5
        assert body["euclidean"] == pytest.approx(5.55e-17, rel=1e-2)

    def test_canberra_reads_a_gap_against_the_size_of_its_values(self) -> None:
        from_zero = pair({"x": 0, "y": 100}, {"x": 1, "y": 101})
        tiny = pair({"x": 0.001, "y": 100}, {"x": 0.002, "y": 101})
        from_one = pair({"x": 1, "y": 100}, {"x": 2, "y": 101})

        assert from_zero["working"]["canberra_terms"] == pytest.approx(
            [1.0, 0.004975], abs=1e-6
        )
        assert from_zero["distances"]["canberra"] == pytest.approx(1.00498, abs=1e-5)
        assert tiny["working"]["canberra_terms"] == pytest.approx(
            [1 / 3, 0.004975], abs=1e-6
        )
        assert from_one["working"]["canberra_terms"] == pytest.approx(
            [1 / 3, 0.004975], abs=1e-6
        )
        assert tiny["distances"]["canberra"] == from_one["distances"]["canberra"]
        assert from_zero["distances"]["euclidean"] == pytest.approx(1.4142, abs=1e-4)

    def test_canberra_reads_zero_against_zero_as_agreement(self) -> None:
        body = pair({"x": 0, "y": 0}, {"x": 0, "y": 1})

        assert body["working"]["canberra_terms"] == [0.0, 1.0]
        assert body["distances"]["canberra"] == 1.0

    def test_cosine_cannot_see_a_difference_of_size(self) -> None:
        body = pair({"x": 1, "y": 1}, {"x": 100, "y": 100})["distances"]

        assert body["cosine"] == pytest.approx(0.0, abs=1e-15)
        assert body["euclidean"] == pytest.approx(140.0071, abs=1e-4)
        assert body["manhattan"] == 198.0

    def test_cosine_reaches_two_only_for_opposite_rows(self) -> None:
        body = pair({"x": 3, "y": 4}, {"x": -3, "y": -4})["distances"]

        assert body["cosine"] == 2.0
        assert body["canberra"] == 2.0
        assert body["euclidean"] == 10.0


class TestAxioms:
    def test_every_metric_puts_a_person_at_zero_from_themselves(self) -> None:
        metrics = by_metric(client.post(AXIOMS, json=SAME_HEIGHT).json()["metrics"])

        assert [metric for metric in metrics] == list(ALL_SIX)
        for metric in ALL_SIX:
            assert metrics[metric]["self_distance"] == 0.0

    def test_every_metric_answers_the_same_both_ways(self) -> None:
        metrics = by_metric(client.post(AXIOMS, json=SAME_HEIGHT).json()["metrics"])

        for metric in ALL_SIX:
            assert metrics[metric]["forward"] == metrics[metric]["backward"]

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
    def test_three_people_in_a_line_make_the_legs_sum_exactly(
        self, metric: str
    ) -> None:
        entry = by_metric(client.post(AXIOMS, json=SAME_HEIGHT).json()["metrics"])[
            metric
        ]

        assert entry["forward"] == 15.0
        assert (entry["leg_one"], entry["leg_two"]) == (7.0, 8.0)
        assert entry["via"] == 15.0
        assert entry["triangle_holds"] is True

    def test_cosine_fails_the_triangle_inequality_on_the_same_three(self) -> None:
        entry = by_metric(client.post(AXIOMS, json=SAME_HEIGHT).json()["metrics"])[
            "cosine"
        ]

        assert entry["forward"] == pytest.approx(0.0039623, abs=1e-7)
        assert entry["leg_one"] == pytest.approx(0.0008922, abs=1e-7)
        assert entry["leg_two"] == pytest.approx(0.0010951, abs=1e-7)
        assert entry["via"] == pytest.approx(0.0019873, abs=1e-7)
        assert entry["via"] < entry["forward"]
        assert entry["triangle_holds"] is False

    def test_hamming_and_canberra_pass_on_the_same_three(self) -> None:
        metrics = by_metric(client.post(AXIOMS, json=SAME_HEIGHT).json()["metrics"])

        assert metrics["hamming"]["forward"] == 0.5
        assert metrics["hamming"]["via"] == 1.0
        assert metrics["hamming"]["triangle_holds"] is True
        assert metrics["canberra"]["forward"] == pytest.approx(0.142857, abs=1e-6)
        assert metrics["canberra"]["via"] == pytest.approx(0.143594, abs=1e-6)
        assert metrics["canberra"]["triangle_holds"] is True

    def test_only_cosine_puts_the_origin_away_from_itself(self) -> None:
        metrics = by_metric(client.post(AXIOMS, json=SAME_HEIGHT).json()["metrics"])

        assert metrics["cosine"]["origin_self_distance"] == 1.0
        for metric in ALL_SIX:
            if metric != "cosine":
                assert metrics[metric]["origin_self_distance"] == 0.0

    def test_cosine_passes_on_a_triple_that_is_not_angularly_between(self) -> None:
        body = client.post(
            AXIOMS,
            json={"first": QUERY, "second": {"x": 180, "y": 80}, "third": NEAREST},
        ).json()
        entry = by_metric(body["metrics"])["cosine"]

        assert entry["forward"] == pytest.approx(0.0080242, abs=1e-7)
        assert entry["via"] == pytest.approx(0.0108612, abs=1e-7)
        assert entry["triangle_holds"] is True

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(AXIOMS, json={**SAME_HEIGHT, "fourth": QUERY})

        assert response.status_code == 422


def units(metric: str, **rest: Any) -> dict[str, Any]:
    response = client.post(
        UNITS, json={"points": CROWD, "query": QUERY, "metric": metric, "k": 3, **rest}
    )
    assert response.status_code == 200, response.text
    return response.json()


def treatment(body: dict[str, Any], name: str) -> dict[str, Any]:
    return next(entry for entry in body["treatments"] if entry["name"] == name)


def chosen(entry: dict[str, Any]) -> list[tuple[int, float]]:
    return [(each["index"], each["distance"]) for each in entry["neighbours"]]


class TestUnits:
    def test_the_three_treatments_come_back_in_order(self) -> None:
        body = units("euclidean")

        assert [entry["name"] for entry in body["treatments"]] == [
            "raw",
            "weight_rescaled",
            "standardised",
        ]
        assert body["weight_factor"] == 1000.0
        assert body["metric"] == "euclidean"

    def test_raw_is_the_borderline_case_with_height_doing_half_the_arithmetic(
        self,
    ) -> None:
        raw = treatment(units("euclidean"), "raw")

        assert chosen(raw) == [(0, 5.0), (1, 10.0), (2, 13.0)]
        assert raw["prediction"] == 0
        assert raw["height_share_of_squared_gaps"] == pytest.approx(0.5117, abs=1e-4)

    def test_weight_in_grams_hands_the_arithmetic_to_one_column(self) -> None:
        grams = treatment(units("euclidean"), "weight_rescaled")

        assert grams["query"] == {"x": 150.0, "y": 45000.0}
        assert [index for index, _ in chosen(grams)] == [0, 1, 2]
        assert chosen(grams)[0][1] == pytest.approx(4000.0011, abs=1e-4)
        assert grams["height_share_of_squared_gaps"] == pytest.approx(1.05e-6, rel=1e-2)

    def test_weight_in_tonnes_reorders_the_three_nearest(self) -> None:
        tonnes = treatment(units("euclidean", weight_factor=0.001), "weight_rescaled")

        assert [index for index, _ in chosen(tonnes)] == [0, 2, 1]
        assert [distance for _, distance in chosen(tonnes)] == pytest.approx(
            [3.000003, 5.000014, 6.000005], abs=1e-6
        )
        assert tonnes["height_share_of_squared_gaps"] == pytest.approx(
            0.999999, abs=1e-6
        )

    def test_standardised_keeps_the_three_at_a_common_scale(self) -> None:
        standardised = treatment(units("euclidean"), "standardised")

        assert standardised["query"]["x"] == pytest.approx(-0.0796, abs=1e-4)
        assert standardised["query"]["y"] == pytest.approx(-0.4028, abs=1e-4)
        assert [index for index, _ in chosen(standardised)] == [0, 1, 2]
        assert [distance for _, distance in chosen(standardised)] == pytest.approx(
            [0.2332, 0.4663, 0.6180], abs=1e-4
        )
        assert standardised["height_share_of_squared_gaps"] == pytest.approx(
            0.4641, abs=1e-4
        )
        assert standardised["prediction"] == 0

    def test_cosine_in_grams_squeezes_every_distance_toward_zero(self) -> None:
        body = units("cosine")
        raw = treatment(body, "raw")
        grams = treatment(body, "weight_rescaled")

        assert raw["prediction"] == 1
        assert [index for index, _ in chosen(raw)] == [0, 1, 3]
        assert [index for index, _ in chosen(grams)] == [0, 1, 3]
        assert chosen(grams)[0][1] == pytest.approx(3.176e-8, rel=1e-3)
        assert max(each["distance"] for each in grams["ranked"]) == pytest.approx(
            1.253e-6, rel=1e-3
        )
        assert max(each["distance"] for each in raw["ranked"]) == pytest.approx(
            0.009011, rel=1e-3
        )
        assert [each["index"] for each in grams["ranked"]] != [
            each["index"] for each in raw["ranked"]
        ]

    def test_cosine_standardised_changes_its_answer(self) -> None:
        standardised = treatment(units("cosine"), "standardised")

        assert [index for index, _ in chosen(standardised)] == [0, 5, 7]
        assert standardised["prediction"] == 0
        assert chosen(standardised)[0][1] == pytest.approx(0.010545, abs=1e-6)

    def test_canberra_does_not_notice_the_units(self) -> None:
        body = units("canberra")

        assert chosen(treatment(body, "raw")) == chosen(
            treatment(body, "weight_rescaled")
        )
        assert chosen(treatment(body, "raw"))[0][1] == pytest.approx(0.056613, abs=1e-6)

    def test_a_constant_column_is_refused_by_the_standardizer(self) -> None:
        constant = [
            {"x": 150, "y": person["y"], "label": person["label"]} for person in CROWD
        ]
        response = client.post(
            UNITS,
            json={"points": constant, "query": QUERY, "metric": "euclidean", "k": 3},
        )

        assert response.status_code == 422
        assert response.json()["error"] == "AllSameValuesError"

    def test_a_factor_outside_the_bounds_is_refused(self) -> None:
        response = client.post(
            UNITS,
            json={
                "points": CROWD,
                "query": QUERY,
                "metric": "euclidean",
                "k": 3,
                "weight_factor": 0,
            },
        )

        assert response.status_code == 422


def models(
    radius: float, points: list[dict[str, float]] = PLAIN, **rest: Any
) -> dict[str, Any]:
    response = client.post(MODELS, json={"points": points, "radius": radius, **rest})
    assert response.status_code == 200, response.text
    return response.json()


class TestModels:
    def test_at_a_radius_of_twenty_the_three_p_norms_disagree_about_the_count(
        self,
    ) -> None:
        density = by_metric(models(20)["density"])

        assert (
            density["euclidean"]["n_clusters"],
            density["euclidean"]["n_noise"],
        ) == (3, 0)
        assert (
            density["manhattan"]["n_clusters"],
            density["manhattan"]["n_noise"],
        ) == (3, 0)
        assert (
            density["chebyshev"]["n_clusters"],
            density["chebyshev"]["n_noise"],
        ) == (2, 0)
        assert density["chebyshev"]["labels"] == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
        assert density["euclidean"]["labels"] == [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]

    def test_at_a_radius_of_thirty_euclidean_fuses_and_manhattan_does_not(self) -> None:
        density = by_metric(models(30)["density"])

        assert density["euclidean"]["n_clusters"] == 1
        assert density["chebyshev"]["n_clusters"] == 1
        assert density["manhattan"]["n_clusters"] == 3

    def test_at_a_radius_of_five_manhattan_calls_six_people_noise(self) -> None:
        density = by_metric(models(5)["density"])

        assert (
            density["manhattan"]["n_clusters"],
            density["manhattan"]["n_noise"],
        ) == (2, 6)
        assert (
            density["euclidean"]["n_clusters"],
            density["euclidean"]["n_noise"],
        ) == (3, 2)

    def test_the_other_three_read_the_radius_in_their_own_units(self) -> None:
        density = by_metric(models(20)["density"])

        for metric in ("cosine", "hamming", "canberra"):
            assert density[metric]["n_clusters"] == 1
            assert density[metric]["n_noise"] == 0

    def test_cosine_needs_a_radius_of_a_thousandth_to_see_two_groups(self) -> None:
        density = by_metric(models(0.001)["density"])["cosine"]

        assert (density["n_clusters"], density["n_noise"]) == (2, 1)
        assert density["labels"] == [-1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]

    def test_single_linkage_cuts_the_far_clump_off_under_four_metrics(self) -> None:
        linkage = by_metric(models(20)["single_linkage"])

        for metric in ("euclidean", "manhattan", "chebyshev", "canberra"):
            assert linkage[metric]["labels"] == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
        assert linkage["euclidean"]["first_merge"] == pytest.approx(2.2361, abs=1e-4)
        assert linkage["euclidean"]["last_merge"] == pytest.approx(23.3452, abs=1e-4)
        assert linkage["manhattan"]["first_merge"] == 3.0
        assert linkage["manhattan"]["last_merge"] == 33.0

    def test_single_linkage_under_cosine_cuts_by_direction(self) -> None:
        linkage = by_metric(models(20)["single_linkage"])["cosine"]

        assert linkage["labels"] == [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
        assert linkage["last_merge"] == pytest.approx(0.001076, abs=1e-6)

    def test_the_mean_wins_on_squares_and_the_median_on_absolutes(self) -> None:
        centre = models(20)["centre_argument"]

        assert centre["mean"] == pytest.approx(53.3636, abs=1e-4)
        assert centre["median"] == 57.0
        assert centre["squared_at_mean"] == pytest.approx(4742.5455, abs=1e-4)
        assert centre["squared_at_median"] == 4888.0
        assert centre["absolute_at_mean"] == pytest.approx(191.6364, abs=1e-4)
        assert centre["absolute_at_median"] == 188.0
        assert centre["squared_at_mean"] < centre["squared_at_median"]
        assert centre["absolute_at_median"] < centre["absolute_at_mean"]

    def test_ward_with_manhattan_is_refused_at_construction(self) -> None:
        ward = models(20)["ward_with_manhattan"]

        assert ward["refused"] is True
        assert "Euclidean" in ward["message"]

    def test_one_person_is_refused_by_the_linkage_cut(self) -> None:
        response = client.post(MODELS, json={"points": PLAIN[:1], "radius": 15})

        assert response.status_code == 422
        assert response.json()["error"] == "TooFewValuesError"

    def test_two_people_cut_at_two_groups_make_no_merge(self) -> None:
        body = models(15, points=PLAIN[:2])
        linkage = by_metric(body["single_linkage"])["euclidean"]

        assert linkage["labels"] == [0, 1]
        assert linkage["first_merge"] is None
        assert linkage["last_merge"] is None
        assert by_metric(body["density"])["euclidean"]["n_clusters"] == 1

    def test_a_radius_of_zero_is_refused(self) -> None:
        response = client.post(MODELS, json={"points": PLAIN, "radius": 0})

        assert response.status_code == 422


class TestDimensions:
    def test_the_default_draw_is_two_hundred_rows_at_seed_zero(self) -> None:
        body = client.post(DIMENSIONS, json={}).json()

        assert (body["n_points"], body["seed"]) == (200, 0)
        assert body["dimensions"] == [1, 2, 5, 10, 50, 200]
        assert [entry["metric"] for entry in body["metrics"]] == list(ALL_SIX)

    def test_euclidean_contrast_falls_from_thousands_to_a_quarter(self) -> None:
        euclidean = by_metric(client.post(DIMENSIONS, json={}).json()["metrics"])[
            "euclidean"
        ]
        means = [each["relative_contrast"] for each in euclidean["contrasts"]]

        assert means == pytest.approx(
            [3248.04, 36.99, 4.955, 2.090, 0.5557, 0.2376], rel=1e-3
        )
        assert means == sorted(means, reverse=True)
        assert euclidean["contrasts"][-1]["median_relative_contrast"] == pytest.approx(
            0.2348, abs=1e-4
        )

    def test_at_two_hundred_features_cosine_keeps_the_most_contrast(self) -> None:
        metrics = by_metric(client.post(DIMENSIONS, json={}).json()["metrics"])
        at_two_hundred = {
            metric: metrics[metric]["contrasts"][-1]["median_relative_contrast"]
            for metric in ALL_SIX
        }

        assert at_two_hundred["cosine"] == pytest.approx(0.5398, abs=1e-4)
        assert at_two_hundred["manhattan"] == pytest.approx(0.2974, abs=1e-4)
        assert at_two_hundred["canberra"] == pytest.approx(0.2655, abs=1e-4)
        assert at_two_hundred["euclidean"] == pytest.approx(0.2348, abs=1e-4)
        assert at_two_hundred["chebyshev"] == pytest.approx(0.1925, abs=1e-4)
        assert at_two_hundred["hamming"] == 0.0

    def test_the_mean_and_median_part_where_a_nearest_row_is_almost_on_top(
        self,
    ) -> None:
        cosine = by_metric(client.post(DIMENSIONS, json={}).json()["metrics"])["cosine"]
        in_two = cosine["contrasts"][1]

        assert in_two["dimensions"] == 2
        assert in_two["relative_contrast"] > 1e6
        assert in_two["median_relative_contrast"] > 1e5
        assert cosine["contrasts"][0]["relative_contrast"] == 0.0

    def test_hamming_on_continuous_rows_has_no_contrast_at_all(self) -> None:
        hamming = by_metric(client.post(DIMENSIONS, json={}).json()["metrics"])[
            "hamming"
        ]

        for each in hamming["contrasts"]:
            assert (each["nearest"], each["farthest"]) == (1.0, 1.0)
            assert each["relative_contrast"] == 0.0

    def test_the_same_seed_gives_the_same_answer(self) -> None:
        first = client.post(DIMENSIONS, json={"n_points": 50, "seed": 4}).json()
        second = client.post(DIMENSIONS, json={"n_points": 50, "seed": 4}).json()

        assert first == second

    @pytest.mark.parametrize(
        "payload", [{"n_points": 5}, {"n_points": 301}, {"seed": -1}]
    )
    def test_a_draw_outside_the_bounds_is_refused(
        self, payload: dict[str, int]
    ) -> None:
        assert client.post(DIMENSIONS, json=payload).status_code == 422


class TestContracts:
    def test_a_constant_column_is_accepted_by_the_neighbour_model(self) -> None:
        constant = [
            {"x": 150, "y": person["y"], "label": person["label"]} for person in CROWD
        ]
        body = client.post(
            NEIGHBOURS,
            json={"points": constant, "query": QUERY, "metric": "euclidean", "k": 3},
        ).json()

        assert [each["distance"] for each in body["neighbours"]] == pytest.approx(
            [4.0, 8.0, 12.0]
        )
        assert body["prediction"] == 0

    def test_a_query_on_a_stored_person_is_at_zero(self) -> None:
        body = client.post(
            NEIGHBOURS,
            json={"points": CROWD, "query": NEAREST, "metric": "euclidean", "k": 3},
        ).json()

        assert body["neighbours"][0] == {"index": 0, "distance": 0.0, "label": 0}

    def test_a_duplicated_person_votes_twice_at_the_same_distance(self) -> None:
        body = client.post(
            NEIGHBOURS,
            json={
                "points": CROWD + [CROWD[0]],
                "query": QUERY,
                "metric": "euclidean",
                "k": 3,
            },
        ).json()

        assert [(each["index"], each["distance"]) for each in body["neighbours"]] == [
            (0, 5.0),
            (11, 5.0),
            (1, 10.0),
        ]

    def test_a_query_at_the_origin_is_one_from_everyone_under_cosine(self) -> None:
        body = client.post(
            NEIGHBOURS,
            json={
                "points": CROWD,
                "query": {"x": 0, "y": 0},
                "metric": "cosine",
                "k": 3,
            },
        ).json()

        assert [each["distance"] for each in body["neighbours"]] == [1.0, 1.0, 1.0]

    def test_a_query_at_the_origin_is_two_from_everyone_under_canberra(self) -> None:
        body = client.post(
            NEIGHBOURS,
            json={
                "points": CROWD,
                "query": {"x": 0, "y": 0},
                "metric": "canberra",
                "k": 3,
            },
        ).json()

        assert [each["distance"] for each in body["neighbours"]] == [2.0, 2.0, 2.0]

    def test_a_coordinate_beyond_a_million_is_refused_at_the_door(self) -> None:
        response = client.post(
            NEIGHBOURS,
            json={
                "points": CROWD,
                "query": {"x": 2e6, "y": 1},
                "metric": "euclidean",
                "k": 1,
            },
        )

        assert response.status_code == 422

    def test_one_person_is_one_class_and_is_refused(self) -> None:
        response = client.post(
            NEIGHBOURS,
            json={"points": CROWD[:1], "query": QUERY, "metric": "euclidean", "k": 1},
        )

        assert response.status_code == 422
        assert response.json()["error"] == "SingleClassError"
