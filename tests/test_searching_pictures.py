"""The searching page's endpoints, pinned to the numbers the page quotes.

Every figure here comes from the reference network, which is trained once per
process and shared, and from a collection drawn under a fixed seed, so a rerun
reproduces them exactly. The tolerances are there for the last bits of the
arithmetic, not for a retrain that could land somewhere else.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())
BASE = "/concepts/searching-a-collection-of-pictures"

IDENTIFIER = re.compile(r"[A-Z][a-z]+[A-Z]\w*|\w+_\w+|Error\b|oop_ml|library")
"""What a class name, an attribute or a credit looks like inside a sentence."""


def _get(path: str) -> dict:
    response = client.get(BASE + path)
    assert response.status_code == 200, response.text
    return response.json()


def _search(**request: object) -> dict:
    response = client.post(BASE + "/search", json=request)
    assert response.status_code == 200, response.text
    return response.json()


def _row(rows: list[dict], key: str, value: int) -> dict:
    return next(row for row in rows if row[key] == value)


class TestOverview:
    def test_the_collection_is_the_stated_draw(self) -> None:
        body = _get("/overview")

        assert body["collection_seed"] == 11
        assert body["collection_size"] == 4000
        assert body["per_kind"] == 1000
        assert body["n_queries"] == 240
        assert body["dimension"] == 16
        assert body["kinds"] == ["cross", "square", "disc", "bar"]
        assert body["network_held_out_accuracy"] == pytest.approx(0.9708333, abs=1e-6)

    def test_precision_at_each_k(self) -> None:
        rows = _get("/overview")["precision"]

        assert [row["k"] for row in rows] == [1, 5, 10, 20]
        assert [row["precision"] for row in rows] == pytest.approx(
            [0.9833333, 0.9791667, 0.9775, 0.9683333], abs=1e-6
        )
        assert [row["queries_all_same_kind"] for row in rows] == [236, 228, 222, 207]

    def test_precision_by_kind_and_the_vote(self) -> None:
        body = _get("/overview")

        assert [row["precision"] for row in body["by_kind"]] == pytest.approx(
            [0.9683333, 0.98, 0.9766667, 0.985], abs=1e-6
        )
        assert body["vote_accuracy"] == pytest.approx(0.9875)
        assert body["chance_precision"] == pytest.approx(0.25)

    def test_the_worked_distance_can_be_redone_by_hand(self) -> None:
        worked = _get("/overview")["worked"]

        assert worked["query"] == 0
        assert worked["neighbour"] == 2072
        assert worked["query_kind"] == worked["neighbour_kind"] == "cross"
        assert len(worked["coordinates"]) == 16
        redone = sum(pair["gap"] ** 2 for pair in worked["coordinates"])
        assert redone == pytest.approx(worked["sum_of_squares_from_rounded"], abs=1e-9)
        assert worked["sum_of_squares_from_rounded"] == pytest.approx(
            0.278769, abs=1e-6
        )
        assert worked["distance_from_rounded"] == pytest.approx(0.5279858, abs=1e-6)
        assert worked["distance"] == pytest.approx(0.5277642, abs=1e-6)
        assert worked["second_distance"] == pytest.approx(0.5990537, abs=1e-6)
        assert worked["farthest_distance"] == pytest.approx(5.3239978, abs=1e-6)

    def test_the_four_queries_whose_nearest_is_another_kind(self) -> None:
        strangers = _get("/overview")["strangers"]

        assert [
            (one["query"], one["query_kind"], one["neighbour_kind"])
            for one in strangers
        ] == [
            (16, "cross", "disc"),
            (35, "bar", "disc"),
            (140, "cross", "disc"),
            (185, "square", "bar"),
        ]
        assert [one["same_kind_in_ten"] for one in strangers] == [3, 2, 6, 0]

    def test_the_cost_of_the_exact_answer(self) -> None:
        body = _get("/overview")

        assert body["distances_per_query"] == 4000
        assert body["multiply_adds_per_query"] == 64000
        assert body["distances_for_every_query"] == 960000

    def test_the_neighbour_search_agrees_with_ranking_every_distance(self) -> None:
        assert _get("/overview")["search_agrees_with_distance_ranking"] is True


class TestSearch:
    def test_the_worked_query_is_served_by_its_own_cell(self) -> None:
        body = _search(query=0, k=10, n_cells=32, probes=1)

        assert body["recall"] == 1.0
        assert body["exact_precision"] == 1.0
        assert body["index_distances"] == 112
        assert body["touched_share"] == pytest.approx(0.02)
        assert body["probed"][0]["size"] == 80
        assert body["kth_distance"] == pytest.approx(0.7691233, abs=1e-6)
        assert body["next_distance"] == pytest.approx(0.8107918, abs=1e-6)

    def test_the_exact_answer_is_nearest_first_and_the_index_counts_honestly(
        self,
    ) -> None:
        body = _search(query=16, k=10, n_cells=64, probes=3)

        distances = [one["distance"] for one in body["exact"]]
        assert distances == sorted(distances)
        assert len(body["exact"]) == 10
        assert all(one["in_exact_answer"] for one in body["exact"])
        assert body["exact_distances"] == 4000
        read = round(body["touched_share"] * 4000)
        assert body["index_distances"] == 64 + read
        assert sum(cell["size"] for cell in body["probed"]) == read

    def test_searching_every_cell_is_the_exact_answer(self) -> None:
        body = _search(query=35, k=20, n_cells=16, probes=16)

        assert body["recall"] == 1.0
        assert [one["position"] for one in body["indexed"]] == [
            one["position"] for one in body["exact"]
        ]

    def test_more_cells_searched_than_exist_is_refused_in_plain_words(self) -> None:
        response = client.post(
            BASE + "/search", json={"query": 0, "k": 10, "n_cells": 16, "probes": 17}
        )

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "16 cells" in detail
        assert not IDENTIFIER.search(detail)

    @pytest.mark.parametrize(
        "request_body",
        [
            {"query": 240},
            {"query": -1},
            {"k": 0},
            {"k": 51},
            {"n_cells": 20},
            {"probes": 0},
            {"surprise": 1},
        ],
    )
    def test_malformed_requests_are_refused(self, request_body: dict) -> None:
        response = client.post(BASE + "/search", json=request_body)

        assert response.status_code == 422


class TestTradeOff:
    def test_the_cells_at_each_count(self) -> None:
        fits = _get("/trade-off")["cell_counts"]

        assert [
            (fit["n_cells"], fit["smallest_cell"], fit["largest_cell"]) for fit in fits
        ] == [
            (16, 98, 513),
            (32, 44, 356),
            (64, 16, 269),
        ]
        assert all(fit["empty_cells"] == 0 for fit in fits)

    def test_thirty_two_cells(self) -> None:
        rows = _row(_get("/trade-off")["cell_counts"], "n_cells", 32)["rows"]

        one, two, three, four, six = (
            _row(rows, "probes", count) for count in (1, 2, 3, 4, 6)
        )
        assert one["recall"] == pytest.approx(0.8329167, abs=1e-6)
        assert one["touched_share"] == pytest.approx(0.0424667, abs=1e-6)
        assert one["queries_with_every_neighbour"] == 121
        assert one["worst_recall"] == 0.0
        assert one["distances_computed"] == pytest.approx(201.8666667, abs=1e-6)
        assert two["recall"] == pytest.approx(0.9583333, abs=1e-6)
        assert two["touched_share"] == pytest.approx(0.0814313, abs=1e-6)
        assert two["queries_with_every_neighbour"] == 196
        assert two["distances_computed"] == pytest.approx(357.725, abs=1e-6)
        assert three["recall"] == pytest.approx(0.9825, abs=1e-6)
        assert three["touched_share"] == pytest.approx(0.1147333, abs=1e-6)
        assert four["recall"] == pytest.approx(0.9958333, abs=1e-6)
        assert four["touched_share"] == pytest.approx(0.1467115, abs=1e-6)
        assert four["queries_with_every_neighbour"] == 234
        assert six["recall"] == 1.0
        assert six["touched_share"] == pytest.approx(0.2005344, abs=1e-6)

    def test_the_question_one_cell_serves_worst(self) -> None:
        rows = _row(_get("/trade-off")["cell_counts"], "n_cells", 32)["rows"]
        assert _row(rows, "probes", 1)["worst_query"] == 46

        recalls = [
            _search(query=46, k=10, n_cells=32, probes=probes)["recall"]
            for probes in (1, 2, 3)
        ]
        assert recalls == [0.0, 0.5, 1.0]
        body = _search(query=46, k=10, n_cells=32, probes=3)
        assert body["query_kind"] == "disc"
        assert [cell["size"] for cell in body["probed"]] == [181, 235, 321]
        assert [cell["holds_true_neighbours"] for cell in body["probed"]] == [0, 5, 5]
        assert [cell["centre_distance"] for cell in body["probed"]] == pytest.approx(
            [0.8160, 0.8877, 0.8929], abs=5e-5
        )
        assert body["kth_distance"] == pytest.approx(0.4853, abs=5e-5)

    def test_questions_land_in_the_large_cells(self) -> None:
        fits = _get("/trade-off")["cell_counts"]

        assert [fit["size_weighted_share"] for fit in fits] == pytest.approx(
            [0.0770141, 0.0412265, 0.0221815], abs=1e-6
        )
        for fit in fits:
            assert fit["size_weighted_share"] > 1 / fit["n_cells"]
        assert [fit["iterations"] for fit in fits] == [50, 46, 26]

    def test_equal_reading_favours_more_cells(self) -> None:
        fits = _get("/trade-off")["cell_counts"]
        near_eight_percent = [
            _row(_row(fits, "n_cells", 16)["rows"], "probes", 1),
            _row(_row(fits, "n_cells", 32)["rows"], "probes", 2),
            _row(_row(fits, "n_cells", 64)["rows"], "probes", 4),
        ]

        assert [row["recall"] for row in near_eight_percent] == pytest.approx(
            [0.8741667, 0.9583333, 0.9841667], abs=1e-6
        )
        assert [row["touched_share"] for row in near_eight_percent] == pytest.approx(
            [0.0800583, 0.0814313, 0.0811917], abs=1e-6
        )
        assert [
            row["distances_computed"] for row in near_eight_percent
        ] == pytest.approx([336.2333333, 357.725, 388.7666667], abs=1e-6)

    def test_searching_every_cell_recovers_everything(self) -> None:
        for fit in _get("/trade-off")["cell_counts"]:
            last = fit["rows"][-1]
            assert last["probes"] == fit["n_cells"]
            assert last["recall"] == 1.0
            assert last["touched_share"] == 1.0
            assert last["distances_computed"] == 4000 + fit["n_cells"]

    def test_hashing_at_twelve_bits(self) -> None:
        table = _row(_get("/trade-off")["hashing"], "bits", 12)

        assert table["possible_buckets"] == 4096
        assert table["occupied_buckets"] == 341
        assert table["largest_bucket"] == 274
        assert [row["buckets_probed"] for row in table["rows"]] == [1, 13, 79, 299, 794]
        assert [row["recall"] for row in table["rows"]] == pytest.approx(
            [0.4583333, 0.7970833, 0.9408333, 0.9829167, 0.9945833], abs=1e-6
        )
        assert [row["touched_share"] for row in table["rows"]] == pytest.approx(
            [0.0215385, 0.0735542, 0.1464219, 0.2347417, 0.3445021], abs=1e-6
        )

    def test_hashing_at_eight_and_sixteen_bits(self) -> None:
        hashing = _get("/trade-off")["hashing"]
        eight = _row(hashing, "bits", 8)
        sixteen = _row(hashing, "bits", 16)

        assert (eight["occupied_buckets"], eight["largest_bucket"]) == (117, 481)
        assert (sixteen["occupied_buckets"], sixteen["largest_bucket"]) == (659, 138)
        assert eight["rows"][0]["recall"] == pytest.approx(0.6125, abs=1e-6)
        assert eight["rows"][1]["recall"] == pytest.approx(0.9179167, abs=1e-6)
        assert sixteen["rows"][3]["recall"] == pytest.approx(0.96125, abs=1e-6)
        assert sixteen["rows"][3]["touched_share"] == pytest.approx(0.1408083, abs=1e-6)
        assert sixteen["rows"][4]["buckets_probed"] == 2517

    def test_the_two_exact_answers_overlap(self) -> None:
        overlap = _get("/trade-off")["cosine_euclidean_overlap"]

        assert overlap == pytest.approx(0.8016667, abs=1e-6)


class TestHashing:
    def test_the_worked_codes(self) -> None:
        body = _get("/hashing")

        assert body["bits"] == 12
        assert body["query_code"] == "101011000000"
        assert [
            (one["position"], one["kind"], one["code"], one["bits_agreeing"])
            for one in body["others"]
        ] == [
            (2072, "cross", "111011000000", 11),
            (2461, "square", "000110000110", 6),
            (1915, "bar", "000101010111", 4),
        ]
        assert [one["angle_degrees"] for one in body["others"]] == pytest.approx(
            [10.0877187, 80.0271880, 118.2928964], abs=1e-6
        )

    def test_the_agreement_probability_holds(self) -> None:
        body = _get("/hashing")

        assert body["check_planes"] == 2000
        assert body["check_pairs"] == 200
        assert body["mean_gap"] == pytest.approx(0.0058909, abs=1e-6)
        assert body["largest_gap"] == pytest.approx(0.0232060, abs=1e-6)
        assert body["standard_error"] == pytest.approx(0.0111803, abs=1e-6)
        assert body["mean_gap"] < body["standard_error"]

    def test_the_vectors_sit_off_the_origin(self) -> None:
        body = _get("/hashing")

        assert body["all_positive_coordinates"] == 11
        assert body["mean_vector_length"] == pytest.approx(3.0252709, abs=1e-6)


class TestConcentration:
    def test_nearest_over_farthest(self) -> None:
        ratios = _get("/concentration")["ratios"]

        assert [(row["label"], row["dimension"]) for row in ratios] == [
            ("picture vectors", 16),
            ("raw pixels", 256),
            ("uniform random", 2),
            ("uniform random", 16),
            ("uniform random", 64),
            ("uniform random", 256),
            ("uniform random", 1024),
        ]
        assert [row["mean_ratio"] for row in ratios] == pytest.approx(
            [
                0.0774826,
                0.2242936,
                0.0072254,
                0.3230157,
                0.5982071,
                0.7797872,
                0.8841730,
            ],
            abs=1e-6,
        )

    def test_the_same_index_on_random_vectors(self) -> None:
        series = _get("/concentration")["index_rows"]
        pictures, sixteen, sixty_four = series

        assert (
            pictures["dimension"],
            sixteen["dimension"],
            sixty_four["dimension"],
        ) == (16, 16, 64)
        assert _row(pictures["rows"], "probes", 2)["recall"] == pytest.approx(
            0.9583333, abs=1e-6
        )
        assert _row(sixteen["rows"], "probes", 2)["recall"] == pytest.approx(
            0.48375, abs=1e-6
        )
        assert _row(sixty_four["rows"], "probes", 2)["recall"] == pytest.approx(
            0.2408333, abs=1e-6
        )
        assert _row(sixteen["rows"], "probes", 16)["recall"] == pytest.approx(
            0.97625, abs=1e-6
        )
        assert _row(sixty_four["rows"], "probes", 16)["recall"] == pytest.approx(
            0.8325, abs=1e-6
        )
        assert _row(sixty_four["rows"], "probes", 24)["recall"] == pytest.approx(
            0.9620833, abs=1e-6
        )
        assert _row(sixty_four["rows"], "probes", 24)["touched_share"] == pytest.approx(
            0.7534583, abs=1e-6
        )

    def test_how_many_directions_hold_the_spread(self) -> None:
        body = _get("/concentration")

        assert body["components_for_ninety"] == 6
        assert body["uniform_components_for_ninety"] == 15
        assert body["spread_shares"][0] == pytest.approx(0.4867601, abs=1e-6)
        assert body["uniform_spread_shares"][0] == pytest.approx(0.0697455, abs=1e-6)


class TestMap:
    def test_every_picture_is_placed_with_its_cells(self) -> None:
        body = _get("/map")

        assert len(body["points"]) == 4000
        assert len(body["queries"]) == 240
        assert [count["n_cells"] for count in body["cell_counts"]] == [16, 32, 64]
        assert all(len(point["cells"]) == 3 for point in body["points"])
        assert body["kept_share"] == pytest.approx(0.7083530, abs=1e-6)

    def test_the_centres_carry_the_cell_sizes(self) -> None:
        body = _get("/map")

        for count in body["cell_counts"]:
            assert sum(centre["size"] for centre in count["centres"]) == 4000


class TestSeeds:
    def test_the_reference_seed_matches_the_overview(self) -> None:
        body = _get("/seeds/0")

        assert body["precision_at_k"] == pytest.approx(0.9775)
        assert body["precision_at_one"] == pytest.approx(0.9833333, abs=1e-6)
        assert body["index_recall"] == pytest.approx(0.9583333, abs=1e-6)
        assert body["network_held_out_accuracy"] == pytest.approx(0.9708333, abs=1e-6)

    @pytest.mark.parametrize(
        ("weight_seed", "accuracy", "at_one", "at_ten", "recall"),
        [
            (1, 0.9208333, 0.9916667, 0.9645833, 0.9404167),
            (2, 0.9666667, 0.9916667, 0.9779167, 0.945),
        ],
    )
    def test_other_starting_weights(
        self,
        weight_seed: int,
        accuracy: float,
        at_one: float,
        at_ten: float,
        recall: float,
    ) -> None:
        body = _get(f"/seeds/{weight_seed}")

        assert body["network_held_out_accuracy"] == pytest.approx(accuracy, abs=1e-6)
        assert body["precision_at_one"] == pytest.approx(at_one, abs=1e-6)
        assert body["precision_at_k"] == pytest.approx(at_ten, abs=1e-6)
        assert body["index_recall"] == pytest.approx(recall, abs=1e-6)

    def test_a_seed_that_was_never_trained_is_refused(self) -> None:
        assert client.get(BASE + "/seeds/3").status_code == 422


class TestNothingVisibleNamesTheCode:
    def test_every_string_the_widgets_render_is_plain(self) -> None:
        strings: list[str] = []

        def gather(value: object) -> None:
            if isinstance(value, str):
                strings.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    gather(item)
            elif isinstance(value, list):
                for item in value:
                    gather(item)

        for path in ("/overview", "/trade-off", "/hashing", "/concentration"):
            gather(_get(path))
        gather(_search(query=16, k=10, n_cells=32, probes=2))

        for text in strings:
            assert not IDENTIFIER.search(text), text
