"""The convolutional networks page's endpoints, on the shared picture collection.

Every number the page quotes is pinned here. The training is deterministic, so
held-out accuracies are pinned as counts out of the 240 held-out pictures and
the rest to tolerances a retrain on the same seed reproduces exactly. The
reports share their caches with the running process, so each arrangement is
trained once for the whole module.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app
from oop_ml_api.picture_network import accuracy_of, reference_network, reference_split

client = TestClient(create_app())

PREFIX = "/concepts/convolutional-networks"
HELD_OUT = 240

HELD_OUT_CORRECT = {
    ("reference", 0): 233,
    ("reference", 1): 221,
    ("reference", 2): 232,
    ("without_pooling", 0): 210,
    ("without_pooling", 1): 197,
    ("without_pooling", 2): 196,
    ("one_round", 0): 215,
    ("one_round", 1): 214,
    ("one_round", 2): 206,
    ("dense", 0): 163,
    ("dense", 1): 175,
    ("dense", 2): 159,
}
"""Held-out pictures named correctly after epoch 40, per arrangement and seed."""

IDENTIFIER = re.compile(r"[A-Z][a-z]+[A-Z]\w*|\w+_\w+|Error\b")
"""A class name or a parameter name, which no refusal on the page may carry."""


def report(variant: str, seed: int) -> dict:
    response = client.get(
        f"{PREFIX}/network", params={"variant": variant, "seed": seed}
    )
    assert response.status_code == 200
    return response.json()


class TestTheReferenceFit:
    def test_the_repeated_loop_is_the_shared_network_to_the_last_bit(self) -> None:
        body = report("reference", 0)
        shared = reference_network()
        _, held_out = reference_split()

        assert body["final_loss"] == shared.history[-1].loss
        assert body["held_out_accuracy"] == accuracy_of(shared.stack, held_out)
        assert [entry["loss"] for entry in body["history"]] == [
            record.loss for record in shared.history
        ]

    def test_the_curve_the_training_section_quotes(self) -> None:
        body = report("reference", 0)
        history = body["history"]

        assert body["n_training"] == 240
        assert body["n_held_out"] == 240
        assert history[0]["loss"] == pytest.approx(1.3691, abs=1e-4)
        assert history[0]["held_out_accuracy"] == pytest.approx(83 / 240)
        assert body["final_loss"] == pytest.approx(0.0190, abs=1e-4)
        first_perfect = next(
            entry["epoch"] for entry in history if entry["training_accuracy"] == 1.0
        )
        assert first_perfect == 33
        assert history[32]["loss"] == pytest.approx(0.0347, abs=1e-4)
        assert body["best_epoch"] == 36

    @pytest.mark.parametrize(
        ("seed", "best", "best_epoch", "final_loss"),
        [(0, 233, 36, 0.0190), (1, 225, 27, 0.0133), (2, 232, 25, 0.0122)],
    )
    def test_three_starts(
        self, seed: int, best: int, best_epoch: int, final_loss: float
    ) -> None:
        body = report("reference", seed)

        assert round(body["best_held_out_accuracy"] * HELD_OUT) == best
        assert body["best_epoch"] == best_epoch
        assert body["final_loss"] == pytest.approx(final_loss, abs=1e-4)
        assert body["training_accuracy"] == 1.0


class TestTheArrangements:
    @pytest.mark.parametrize(("variant", "seed"), sorted(HELD_OUT_CORRECT))
    def test_held_out_accuracy(self, variant: str, seed: int) -> None:
        body = report(variant, seed)

        assert body["held_out_accuracy"] * HELD_OUT == pytest.approx(
            HELD_OUT_CORRECT[(variant, seed)]
        )

    @pytest.mark.parametrize(
        ("variant", "parameters", "multiply_adds"),
        [
            ("reference", 2468, 29760),
            ("without_pooling", 33188, 115776),
            ("one_round", 4220, 13376),
            ("dense", 2541, 2512),
        ],
    )
    def test_what_each_costs(
        self, variant: str, parameters: int, multiply_adds: int
    ) -> None:
        body = report(variant, 0)

        assert body["n_parameters"] == parameters
        assert body["n_multiply_adds"] == multiply_adds

    def test_the_dense_network_does_not_fit_its_own_training_half(self) -> None:
        assert [
            round(report("dense", seed)["training_accuracy"] * 240)
            for seed in (0, 1, 2)
        ] == [
            220,
            230,
            227,
        ]
        for variant in ("reference", "without_pooling", "one_round"):
            assert report(variant, 0)["training_accuracy"] == 1.0

    def test_the_layers_of_the_network_and_how_far_each_sees(self) -> None:
        layers = report("reference", 0)["layers"]

        assert [layer["parameters"] for layer in layers] == [40, 0, 296, 0, 0, 2064, 68]
        assert [layer["field"] for layer in layers[:4]] == [3, 4, 8, 10]
        assert [layer["step"] for layer in layers[:4]] == [1, 2, 2, 4]
        assert layers[4]["answers"] == [128]

    def test_without_pooling_the_second_convolution_sees_five(self) -> None:
        layers = report("without_pooling", 0)["layers"]

        assert [layer["field"] for layer in layers[:2]] == [3, 5]
        assert layers[2]["answers"] == [2048]
        assert layers[3]["parameters"] == 32784


class TestShifting:
    @pytest.mark.parametrize(
        ("variant", "changed"),
        [
            ("reference", [0.0665, 0.1151, 0.0549]),
            ("without_pooling", [0.1679, 0.2334, 0.2492]),
            ("one_round", [0.1193, 0.1489, 0.1985]),
            ("dense", [0.3506, 0.3083, 0.3791]),
        ],
    )
    def test_one_pixel_changes_the_call_on_this_share(
        self, variant: str, changed: list[float]
    ) -> None:
        readings = [report(variant, seed)["shifts"][0] for seed in (0, 1, 2)]

        assert [reading["n_pictures"] for reading in readings] == [947] * 3
        assert [reading["calls_changed"] for reading in readings] == pytest.approx(
            changed, abs=1e-4
        )

    def test_three_pixels_on_the_network_and_without_pooling(self) -> None:
        network = [report("reference", seed)["shifts"][2] for seed in (0, 1, 2)]
        bare = [report("without_pooling", seed)["shifts"][2] for seed in (0, 1, 2)]

        assert [reading["calls_changed"] for reading in network] == pytest.approx(
            [0.12, 0.1986, 0.1186], abs=1e-4
        )
        assert [reading["calls_changed"] for reading in bare] == pytest.approx(
            [0.3371, 0.4057, 0.3843], abs=1e-4
        )
        assert network[0]["n_pictures"] == 700

    def test_the_moved_pictures_are_still_mostly_named(self) -> None:
        first = report("reference", 0)["shifts"][0]

        assert first["shifted_accuracy"] == pytest.approx(0.9388, abs=1e-4)
        assert first["accuracy_before"] == pytest.approx(0.9704, abs=1e-4)
        assert report("reference", 0)["shifts"][1]["calls_changed"] == pytest.approx(
            0.0812, abs=1e-4
        )


class TestTheFirstFilters:
    def test_seed_zero(self) -> None:
        filters = report("reference", 0)["filters"]

        assert [entry["closest"] for entry in filters] == [
            "brighter below",
            "brighter to the right",
            "brighter to the upper left",
            "brighter to the lower right",
        ]
        assert [entry["closeness"] for entry in filters] == pytest.approx(
            [0.675, 0.543, 0.814, 0.794], abs=1e-3
        )
        assert [entry["closeness_at_start"] for entry in filters] == pytest.approx(
            [0.309, 0.552, 0.594, 0.754], abs=1e-3
        )
        assert [entry["moved"] for entry in filters] == pytest.approx(
            [1.332, 0.014, 1.858, 0.09], abs=1e-3
        )
        assert [entry["active_share"] for entry in filters] == pytest.approx(
            [0.4632, 0.0008, 0.2719, 0.0226], abs=1e-4
        )
        assert [entry["weight_total"] for entry in filters] == pytest.approx(
            [3.256, -3.395, 3.766, -0.738], abs=1e-3
        )
        assert [entry["bias"] for entry in filters] == pytest.approx(
            [-0.589, -0.123, -0.963, -0.096], abs=1e-3
        )

    def test_how_many_filters_are_nearly_silent_at_each_seed(self) -> None:
        assert [report("reference", seed)["n_nearly_silent"] for seed in (0, 1, 2)] == [
            2,
            0,
            1,
        ]

    def test_the_random_baseline(self) -> None:
        baseline = report("reference", 0)["edge_baseline"]

        assert baseline["n_draws"] == 2000
        assert baseline["median"] == pytest.approx(0.4655, abs=1e-4)
        assert baseline["ninetieth_percentile"] == pytest.approx(0.7241, abs=1e-4)

    def test_the_dense_network_has_no_filters(self) -> None:
        assert report("dense", 0)["filters"] == []


class TestMaps:
    def test_one_picture_of_each_kind_and_what_it_was_called(self) -> None:
        body = client.get(f"{PREFIX}/maps").json()

        assert [picture["kind"] for picture in body["pictures"]] == [
            "cross",
            "square",
            "disc",
            "bar",
        ]
        assert [picture["called"] for picture in body["pictures"]] == [
            "cross",
            "square",
            "disc",
            "bar",
        ]
        assert [len(depth["maps"]) for depth in body["pictures"][0]["depths"]] == [
            4,
            4,
            8,
            8,
        ]
        assert [depth["side"] for depth in body["pictures"][0]["depths"]] == [
            16,
            8,
            8,
            4,
        ]
        assert len(body["pictures"][0]["vector"]) == 16

    def test_the_disc_is_nearly_called_a_cross(self) -> None:
        disc = client.get(f"{PREFIX}/maps").json()["pictures"][2]

        assert disc["probabilities"][2] == pytest.approx(0.5158, abs=1e-4)
        assert disc["probabilities"][0] == pytest.approx(0.4678, abs=1e-4)

    def test_the_worked_cell(self) -> None:
        worked = client.get(f"{PREFIX}/maps").json()["worked"]

        assert (worked["filter_index"], worked["row"], worked["column"]) == (2, 11, 11)
        assert worked["product_total"] == pytest.approx(3.6014, abs=1e-4)
        assert worked["bias"] == pytest.approx(-0.9630, abs=1e-4)
        assert worked["score"] == pytest.approx(2.6384, abs=1e-4)
        assert worked["output"] == worked["score"]
        assert worked["pool_window"] == [[0.5139, 2.3726], [0.0, 2.6384]]
        assert worked["pooled"] == pytest.approx(worked["score"])
        total = sum(sum(row) for row in worked["products"])
        assert total == pytest.approx(worked["product_total"], abs=1e-3)


class TestReach:
    @pytest.mark.parametrize(
        ("depth", "row", "column", "field", "in_window", "moved"),
        [
            ("first_convolution", 8, 8, 3, 9, 9),
            ("first_pooling", 4, 4, 4, 16, 14),
            ("second_convolution", 4, 4, 8, 64, 63),
            ("second_pooling", 2, 2, 10, 100, 92),
            ("second_pooling", 0, 0, 10, 49, 45),
        ],
    )
    def test_the_geometry_and_the_poking_agree(
        self,
        depth: str,
        row: int,
        column: int,
        field: int,
        in_window: int,
        moved: int,
    ) -> None:
        body = client.post(
            f"{PREFIX}/reach",
            json={"kind": "cross", "depth": depth, "row": row, "column": column},
        ).json()

        assert body["field"] == field
        assert body["n_in_window"] == in_window
        assert body["n_moved"] == moved
        assert body["n_moved_outside"] == 0

    def test_the_corner_unit_reads_past_the_picture(self) -> None:
        body = client.post(
            f"{PREFIX}/reach",
            json={"kind": "cross", "depth": "second_pooling", "row": 0, "column": 0},
        ).json()

        assert (body["top"], body["left"], body["bottom"], body["right"]) == (
            -3,
            -3,
            6,
            6,
        )

    def test_a_unit_past_the_map_is_refused_in_plain_words(self) -> None:
        response = client.post(
            f"{PREFIX}/reach",
            json={"kind": "cross", "depth": "second_pooling", "row": 5, "column": 0},
        )

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "4 by 4" in detail
        assert IDENTIFIER.search(detail) is None

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(f"{PREFIX}/reach", json={"kind": "ring"})

        assert response.status_code == 422


class TestUnseen:
    def test_rings_are_given_to_the_known_kinds(self) -> None:
        body = client.get(f"{PREFIX}/unseen").json()

        assert body["n_rings"] == 60
        assert body["ring_calls"] == [9, 48, 3, 0]
        assert body["ring_mean_confidence"] == pytest.approx(0.8614, abs=1e-4)
        assert body["ring_share_above_ninety"] == pytest.approx(38 / 60)
        assert body["held_out_mean_confidence"] == pytest.approx(0.9484, abs=1e-4)
        assert body["held_out_share_above_ninety"] == pytest.approx(203 / 240)

    def test_a_blank_picture_is_called_a_bar(self) -> None:
        body = client.get(f"{PREFIX}/unseen").json()

        assert body["blank_called"] == "bar"
        assert body["blank_probabilities"][3] == pytest.approx(0.8921, abs=1e-4)

    def test_only_sides_that_halve_to_four_can_be_read(self) -> None:
        sides = {
            entry["side"]: entry
            for entry in client.get(f"{PREFIX}/unseen").json()["sides"]
        }

        assert sides[16]["flattened"] == 128
        assert sides[12]["flattened"] == 72
        assert sides[20]["flattened"] == 200
        assert sides[32]["flattened"] == 512
        assert sides[17]["readable"] is True
        assert sides[17]["first_drops"] == 1
        assert [side for side, entry in sides.items() if entry["readable"]] == [16, 17]


class TestRefusals:
    def test_a_seed_outside_the_three(self) -> None:
        response = client.get(f"{PREFIX}/network", params={"seed": 5})

        assert response.status_code == 422

    def test_an_arrangement_that_does_not_exist(self) -> None:
        response = client.get(f"{PREFIX}/network", params={"variant": "wide"})

        assert response.status_code == 422

    def test_no_label_the_page_draws_carries_an_identifier(self) -> None:
        for variant in ("reference", "without_pooling", "one_round", "dense"):
            body = report(variant, 0)
            texts = [body["label"]] + [layer["label"] for layer in body["layers"]]
            texts += [entry["closest"] for entry in body["filters"]]
            for text in texts:
                assert IDENTIFIER.search(text) is None, text
        maps = client.get(f"{PREFIX}/maps").json()
        for picture in maps["pictures"]:
            for depth in picture["depths"]:
                assert IDENTIFIER.search(depth["label"]) is None
