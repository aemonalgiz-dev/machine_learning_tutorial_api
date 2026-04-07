"""The distance-metrics endpoints, pinned to the page's hand arithmetic.

The page measures the k-nearest page's query (150, 45) against its nearest
stored person (147, 41), a gap of (3, 4), so Euclidean, Manhattan and
Chebyshev come out 5, 7 and 4, Hamming is 1 because both features differ, and
cosine and Canberra are the two small numbers the working reaches by hand. The
same gap swept through the p-norm orders slides from 7 down to 4. The unit
shapes are counted in lattice cells, 317 for the circle, 221 for the diamond
and 441 for the square, and those three counts do not move when the centre
does, where the other three do. The crowd under cosine swaps one neighbour and
flips the vote at k of 3, and at k of 5 admits a child from the far clump.
Under Hamming every distance is exactly one. All pinned.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

UNIT_CIRCLE = "/concepts/distance-metrics/unit-circle"
NEIGHBOURS = "/concepts/distance-metrics/neighbours"
BETWEEN = "/concepts/distance-metrics/between"

P_NORMS = ("euclidean", "manhattan", "chebyshev")
ALL_SIX = (*P_NORMS, "cosine", "hamming", "canberra")

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

QUERY = {"x": 150, "y": 45}
NEAREST = {"x": 147, "y": 41}
FAR_CHILD_INDEX = 6


def shape(metric: str, centre: dict[str, float] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"metric": metric}
    if centre is not None:
        payload["centre"] = centre
    response = client.post(UNIT_CIRCLE, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def cell(body: dict[str, Any], x: float, y: float) -> tuple[float, bool, bool]:
    """The distance, inside flag and boundary flag at lattice cell (x, y)."""
    column = round((x - body["x_min"]) * 10)
    row = round((y - body["y_min"]) * 10)
    return (
        body["distances"][row][column],
        body["inside"][row][column],
        body["on_boundary"][row][column],
    )


def crowd_under(metric: str, k: int = 3) -> dict[str, Any]:
    response = client.post(
        NEIGHBOURS, json={"points": CROWD, "query": QUERY, "metric": metric, "k": k}
    )
    assert response.status_code == 200, response.text
    return response.json()


def pair(first: dict[str, float], second: dict[str, float]) -> dict[str, Any]:
    response = client.post(BETWEEN, json={"first": first, "second": second})
    assert response.status_code == 200, response.text
    return response.json()


class TestUnitCircle:
    def test_the_lattice_is_forty_one_cells_from_minus_two_to_two(self) -> None:
        body = shape("euclidean")

        assert body["cells"] == 41
        assert (body["x_min"], body["x_max"]) == (-2.0, 2.0)
        assert (body["y_min"], body["y_max"]) == (-2.0, 2.0)
        for grid in ("distances", "inside", "on_boundary"):
            assert len(body[grid]) == 41
            assert all(len(row) == 41 for row in body[grid])

    def test_the_diamond_circle_and_square_nest(self) -> None:
        assert shape("manhattan")["inside_count"] == 221
        assert shape("euclidean")["inside_count"] == 317
        assert shape("chebyshev")["inside_count"] == 441

    def test_how_many_cells_land_exactly_one_unit_away(self) -> None:
        assert shape("euclidean")["boundary_count"] == 12
        assert shape("manhattan")["boundary_count"] == 40
        assert shape("chebyshev")["boundary_count"] == 80

    def test_the_boundary_cells_count_as_one_unit_away(self) -> None:
        assert cell(shape("euclidean"), 0.6, 0.8) == (1.0, True, True)
        assert cell(shape("manhattan"), 0.3, 0.7) == (1.0, True, True)
        assert cell(shape("chebyshev"), 1.0, 1.0) == (1.0, True, True)

    def test_the_corner_is_inside_the_square_and_outside_the_circle(self) -> None:
        distance, inside, on_boundary = cell(shape("euclidean"), 1.0, 1.0)

        assert distance == pytest.approx(1.4142135623730951)
        assert inside is False
        assert on_boundary is False

    @pytest.mark.parametrize("metric", P_NORMS)
    @pytest.mark.parametrize("centre", [{"x": 1.0, "y": 1.0}, {"x": -1.0, "y": 0.5}])
    def test_the_p_norm_shapes_do_not_move_with_the_centre(
        self, metric: str, centre: dict[str, float]
    ) -> None:
        moved = shape(metric, centre)
        at_origin = shape(metric)

        assert moved["inside_count"] == at_origin["inside_count"]
        assert moved["boundary_count"] == at_origin["boundary_count"]

    def test_the_origin_has_no_direction_under_cosine(self) -> None:
        body = shape("cosine")

        assert body["inside_count"] == 1681
        assert body["boundary_count"] == 1681
        assert all(value == 1.0 for row in body["distances"] for value in row)

    def test_hamming_from_the_origin_is_a_cross_at_a_half(self) -> None:
        body = shape("hamming")

        assert body["inside_count"] == 1681
        assert body["boundary_count"] == 1600
        assert cell(body, 0.0, 0.0) == (0.0, True, False)
        assert cell(body, 1.0, 0.0) == (0.5, True, False)
        assert cell(body, 0.0, 0.5) == (0.5, True, False)
        assert cell(body, 0.5, 0.5) == (1.0, True, True)
        assert cell(body, 1.7, -1.3) == (1.0, True, True)

    def test_canberra_from_the_origin_is_the_two_axes(self) -> None:
        body = shape("canberra")

        assert body["inside_count"] == 81
        assert body["boundary_count"] == 80
        assert cell(body, 0.0, 0.0) == (0.0, True, False)
        assert cell(body, 0.5, 0.0) == (1.0, True, True)
        assert cell(body, 0.0, 1.0) == (1.0, True, True)
        assert cell(body, 0.5, 0.5) == (2.0, False, False)

    def test_moving_the_centre_reshapes_the_other_three(self) -> None:
        one_one = {"x": 1.0, "y": 1.0}

        assert shape("cosine", one_one)["inside_count"] == 861
        assert shape("canberra", one_one)["inside_count"] == 413
        assert shape("hamming", one_one)["inside_count"] == 1681

    def test_cosine_around_one_one_is_a_half_plane(self) -> None:
        body = shape("cosine", {"x": 1.0, "y": 1.0})

        # The library documents that a row against its own direction lands an
        # epsilon from zero rather than on it.
        distance, inside, on_boundary = cell(body, 2.0, 2.0)
        assert distance == pytest.approx(0.0, abs=1e-12)
        assert inside is True
        assert on_boundary is False
        assert cell(body, 1.0, -1.0) == (1.0, True, True)
        assert cell(body, 0.0, 0.0) == (1.0, True, True)
        distance, inside, on_boundary = cell(body, -1.0, -1.0)
        assert distance == pytest.approx(2.0)
        assert inside is False
        assert on_boundary is False

    def test_canberra_around_one_one_reaches_the_axes_and_stops(self) -> None:
        body = shape("canberra", {"x": 1.0, "y": 1.0})

        assert cell(body, 0.0, 1.0) == (1.0, True, True)
        assert cell(body, 2.0, 1.0) == (pytest.approx(1 / 3), True, False)
        assert cell(body, 0.0, 0.0) == (2.0, False, False)

    def test_the_centre_defaults_to_the_origin(self) -> None:
        body = shape("manhattan")

        assert body["centre"] == {"x": 0.0, "y": 0.0}
        assert body["metric"] == "manhattan"

    def test_a_misspelt_metric_is_refused(self) -> None:
        response = client.post(UNIT_CIRCLE, json={"metric": "euclidian"})

        assert response.status_code == 422

    def test_a_centre_outside_the_lattice_is_refused(self) -> None:
        response = client.post(
            UNIT_CIRCLE, json={"metric": "euclidean", "centre": {"x": 5.0, "y": 0.0}}
        )

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(UNIT_CIRCLE, json={"metric": "euclidean", "radius": 2})

        assert response.status_code == 422


class TestNeighbours:
    @pytest.mark.parametrize(
        ("metric", "distances"),
        [
            ("euclidean", [5.0, 10.0, 13.0]),
            ("manhattan", [7.0, 14.0, 17.0]),
            ("chebyshev", [4.0, 8.0, 12.0]),
        ],
    )
    def test_the_three_p_norms_consult_the_same_people(
        self, metric: str, distances: list[float]
    ) -> None:
        body = crowd_under(metric)

        assert [each["index"] for each in body["neighbours"]] == [0, 1, 2]
        assert [each["distance"] for each in body["neighbours"]] == distances
        assert [each["label"] for each in body["neighbours"]] == [0, 1, 0]
        assert body["prediction"] == 0
        assert (body["votes_for_zero"], body["votes_for_one"]) == (2, 1)

    def test_cosine_swaps_a_neighbour_and_flips_the_vote(self) -> None:
        body = crowd_under("cosine")

        assert [each["index"] for each in body["neighbours"]] == [0, 1, 3]
        assert [each["label"] for each in body["neighbours"]] == [0, 1, 1]
        assert body["neighbours"][0]["distance"] == pytest.approx(0.000189, abs=1e-6)
        assert body["neighbours"][1]["distance"] == pytest.approx(0.000650, abs=1e-6)
        assert body["neighbours"][2]["distance"] == pytest.approx(0.001392, abs=1e-6)
        assert body["prediction"] == 1
        assert (body["votes_for_zero"], body["votes_for_one"]) == (1, 2)

    def test_cosine_at_five_admits_a_child_from_the_far_clump(self) -> None:
        body = crowd_under("cosine", k=5)

        chosen = {each["index"]: each for each in body["neighbours"]}
        assert FAR_CHILD_INDEX in chosen
        assert chosen[FAR_CHILD_INDEX]["distance"] == pytest.approx(0.002168, abs=1e-6)
        assert CROWD[FAR_CHILD_INDEX] == {"x": 122, "y": 28, "label": 0}

    def test_euclidean_at_five_never_reaches_the_far_clump(self) -> None:
        body = crowd_under("euclidean", k=5)

        assert [each["index"] for each in body["neighbours"]] == [0, 1, 2, 3, 4]
        assert body["prediction"] == 1

    def test_hamming_puts_everyone_at_exactly_one(self) -> None:
        body = crowd_under("hamming")

        assert [each["distance"] for each in body["neighbours"]] == [1.0, 1.0, 1.0]
        assert [each["distance"] for each in body["ranked"]] == [1.0] * len(CROWD)
        assert {value for row in body["regions"]["labels"] for value in row} == {0}

    def test_canberra_keeps_the_same_three_in_the_same_order(self) -> None:
        body = crowd_under("canberra")

        assert [each["index"] for each in body["neighbours"]] == [0, 1, 2]
        assert body["neighbours"][0]["distance"] == pytest.approx(0.056613, abs=1e-6)
        assert body["neighbours"][1]["distance"] == pytest.approx(0.101240, abs=1e-6)
        assert body["neighbours"][2]["distance"] == pytest.approx(0.134596, abs=1e-6)
        assert body["prediction"] == 0

    @pytest.mark.parametrize("metric", ALL_SIX)
    def test_the_ranking_holds_everyone_nearest_first_with_the_voters_flagged(
        self, metric: str
    ) -> None:
        body = crowd_under(metric)

        ranked = body["ranked"]
        assert len(ranked) == len(CROWD)
        assert sorted(each["index"] for each in ranked) == list(range(len(CROWD)))
        distances = [each["distance"] for each in ranked]
        assert distances == sorted(distances)
        assert [each["index"] for each in ranked if each["chosen"]] == [
            each["index"] for each in body["neighbours"]
        ]
        assert body["metric"] == metric

    def test_the_regions_are_the_shared_lattice(self) -> None:
        body = crowd_under("euclidean")

        assert body["regions"]["cells"] == 26
        assert {value for row in body["regions"]["labels"] for value in row} == {0, 1}

    def test_more_voters_than_people_is_refused_by_the_library(self) -> None:
        response = client.post(
            NEIGHBOURS,
            json={"points": CROWD[:2], "query": QUERY, "metric": "euclidean", "k": 3},
        )

        assert response.status_code == 422
        assert response.json()["error"] == "TooFewValuesError"

    def test_a_crowd_of_one_class_is_refused_by_the_library(self) -> None:
        children = [each for each in CROWD if each["label"] == 0]
        response = client.post(
            NEIGHBOURS,
            json={"points": children, "query": QUERY, "metric": "euclidean", "k": 3},
        )

        assert response.status_code == 422
        assert response.json()["error"] == "SingleClassError"

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            NEIGHBOURS,
            json={
                "points": CROWD,
                "query": QUERY,
                "metric": "euclidean",
                "k": 3,
                "order": 2,
            },
        )

        assert response.status_code == 422


class TestBetween:
    def test_the_gap_is_three_by_four_and_the_p_norms_follow(self) -> None:
        body = pair(QUERY, NEAREST)

        assert (body["working"]["gap_x"], body["working"]["gap_y"]) == (3.0, 4.0)
        assert body["distances"]["euclidean"] == 5.0
        assert body["distances"]["manhattan"] == 7.0
        assert body["distances"]["chebyshev"] == 4.0

    def test_hamming_counts_both_features_as_differing(self) -> None:
        body = pair(QUERY, NEAREST)

        assert body["distances"]["hamming"] == 1.0
        assert body["working"]["differing_features"] == 2
        assert body["working"]["feature_count"] == 2

    def test_cosine_is_one_minus_a_normalised_dot_product(self) -> None:
        body = pair(QUERY, NEAREST)
        working = body["working"]

        assert working["dot_product"] == 23895.0
        assert working["first_length"] == pytest.approx(156.6046, abs=1e-4)
        assert working["second_length"] == pytest.approx(152.6106, abs=1e-4)
        assert working["cosine_similarity"] == pytest.approx(0.99981, abs=1e-5)
        assert body["distances"]["cosine"] == pytest.approx(0.00019, abs=1e-5)
        assert body["distances"]["cosine"] == pytest.approx(
            1.0 - working["cosine_similarity"]
        )

    def test_canberra_is_the_two_fractions_summed(self) -> None:
        body = pair(QUERY, NEAREST)
        terms = body["working"]["canberra_terms"]

        assert terms == [pytest.approx(3 / 297), pytest.approx(4 / 86)]
        assert terms[0] == pytest.approx(0.0101, abs=1e-4)
        assert terms[1] == pytest.approx(0.0465, abs=1e-4)
        assert body["distances"]["canberra"] == pytest.approx(0.0566, abs=1e-4)
        assert body["distances"]["canberra"] == pytest.approx(sum(terms))

    def test_the_orders_slide_from_manhattan_to_chebyshev(self) -> None:
        body = pair(QUERY, NEAREST)

        assert [each["order"] for each in body["orders"]] == [
            1.0,
            1.5,
            2.0,
            4.0,
            10.0,
            None,
        ]
        distances = [each["distance"] for each in body["orders"]]
        assert distances[0] == 7.0
        assert distances[1] == pytest.approx(5.5843, abs=1e-4)
        assert distances[2] == 5.0
        assert distances[3] == pytest.approx(4.2846, abs=1e-4)
        assert distances[4] == pytest.approx(4.0220, abs=1e-4)
        assert distances[5] == 4.0
        assert distances == sorted(distances, reverse=True)

    def test_the_sweep_ends_agree_with_the_named_metrics(self) -> None:
        body = pair(QUERY, NEAREST)

        by_order = {each["order"]: each["distance"] for each in body["orders"]}
        assert by_order[1.0] == body["distances"]["manhattan"]
        assert by_order[2.0] == body["distances"]["euclidean"]
        assert by_order[None] == body["distances"]["chebyshev"]

    def test_a_zero_row_is_one_from_everything_under_cosine(self) -> None:
        body = pair({"x": 0, "y": 0}, {"x": 3, "y": 4})

        assert body["distances"]["cosine"] == 1.0
        assert body["working"]["cosine_similarity"] == 0.0
        assert body["distances"]["euclidean"] == 5.0
        assert body["distances"]["canberra"] == 2.0

    def test_a_row_against_itself_is_zero_under_all_six(self) -> None:
        body = pair({"x": 3, "y": 4}, {"x": 3, "y": 4})

        assert set(body["distances"].values()) == {0.0}
        assert body["working"]["differing_features"] == 0
        assert body["working"]["canberra_terms"] == [0.0, 0.0]

    def test_a_scaled_copy_is_zero_under_cosine_alone(self) -> None:
        body = pair({"x": 1, "y": 1}, {"x": 100, "y": 100})

        assert body["distances"]["cosine"] == pytest.approx(0.0, abs=1e-12)
        assert body["distances"]["euclidean"] == pytest.approx(99 * 2**0.5)
        assert body["distances"]["hamming"] == 1.0

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            BETWEEN, json={"first": QUERY, "second": NEAREST, "metric": "cosine"}
        )

        assert response.status_code == 422
