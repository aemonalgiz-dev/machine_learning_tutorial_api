"""The a-vector-for-a-picture page's endpoints, pinned at the numbers it quotes.

Every network here is the shared reference network or the same design under
weight seeds 1 and 2, all deterministic, so a retrain reproduces these values
exactly and the tolerances only absorb the page's rounding to four places.
The last class asserts that no identifier reaches a reader through any string
an endpoint returns.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/a-vector-for-a-picture"
FOUR = 5e-5


def _get(path: str) -> dict:
    response = client.get(BASE + path)
    assert response.status_code == 200, response.text
    return response.json()


def _representation(body: dict, name: str) -> dict:
    return next(row for row in body["representations"] if row["name"] == name)


class TestWalk:
    def test_the_network_layer_by_layer(self) -> None:
        body = _get("/walk")

        assert [row["answer_shape"] for row in body["layers"]] == [
            [4, 16, 16],
            [4, 8, 8],
            [8, 8, 8],
            [8, 4, 4],
            [128],
            [16],
            [4],
        ]
        assert [row["n_parameters"] for row in body["layers"]] == [
            40,
            0,
            296,
            0,
            0,
            2064,
            68,
        ]
        assert body["n_parameters"] == 2468
        assert body["n_training"] == 240
        assert body["n_held_out"] == 240
        assert body["per_kind_held_out"] == 60
        assert body["epochs"] == 40

    def test_how_well_it_names(self) -> None:
        body = _get("/walk")

        assert body["held_out_accuracy"] == pytest.approx(0.9708, abs=FOUR)
        assert body["training_accuracy"] == 1.0

    def test_the_four_pictures(self) -> None:
        body = _get("/walk")
        kinds = [picture["kind"] for picture in body["pictures"]]

        assert kinds == ["cross", "square", "disc", "bar"]
        assert [picture["position"] for picture in body["pictures"]] == [0, 1, 2, 3]
        assert all(len(picture["vector"]) == 16 for picture in body["pictures"])
        assert all(
            abs(value) < 1.0
            for picture in body["pictures"]
            for value in picture["vector"]
        )

    def test_the_disc_is_named_by_a_whisker(self) -> None:
        disc = _get("/walk")["pictures"][2]

        assert disc["called"] == "disc"
        assert disc["probabilities"][2] == pytest.approx(0.5158, abs=FOUR)
        assert disc["probabilities"][0] == pytest.approx(0.4678, abs=FOUR)

    def test_the_worked_cosine(self) -> None:
        worked = _get("/walk")["worked"]

        assert worked["dot"] == pytest.approx(3.2608, abs=FOUR)
        assert worked["first_length"] == pytest.approx(3.0017, abs=FOUR)
        assert worked["second_length"] == pytest.approx(3.6335, abs=FOUR)
        assert worked["cosine"] == pytest.approx(
            worked["dot"] / (worked["first_length"] * worked["second_length"])
        )
        assert worked["cosine"] == pytest.approx(0.2990, abs=FOUR)

    def test_the_four_by_four_tables(self) -> None:
        body = _get("/walk")

        assert body["cosines"][0][2] == pytest.approx(0.4199, abs=FOUR)
        assert body["cosines"][1][3] == pytest.approx(-0.3454, abs=FOUR)
        assert body["pixel_gaps"][0][2] == pytest.approx(4.0616, abs=FOUR)
        assert body["pixel_gaps"][1][3] == pytest.approx(6.2299, abs=FOUR)


class TestSphere:
    def test_every_picture_and_every_ring_is_placed(self) -> None:
        body = _get("/sphere")

        assert body["dimension"] == 16
        assert body["n_pictures"] == 240
        assert body["n_rings"] == 60
        for view in body["views"]:
            assert len(view["points"]) == 300
            for point in view["points"]:
                length = (point["x"] ** 2 + point["y"] ** 2 + point["z"] ** 2) ** 0.5
                assert length == pytest.approx(1.0)

    def test_the_trained_and_untrained_arrangements(self) -> None:
        trained, untrained = _get("/sphere")["views"]

        assert trained["network"] == "trained"
        assert trained["kept_share"] == pytest.approx(0.8115, abs=FOUR)
        assert trained["within"] == pytest.approx(0.7588, abs=FOUR)
        assert trained["across"] == pytest.approx(0.0022, abs=FOUR)
        assert trained["nearest_own_kind"] == pytest.approx(233 / 240)
        assert untrained["kept_share"] == pytest.approx(0.5925, abs=FOUR)
        assert untrained["within"] == pytest.approx(0.7394, abs=FOUR)
        assert untrained["across"] == pytest.approx(0.7325, abs=FOUR)
        assert untrained["nearest_own_kind"] == pytest.approx(189 / 240)


class TestSeedZero:
    def test_pixels_against_the_vector(self) -> None:
        body = _get("/seed/0")

        assert _representation(body, "pixels, by distance")[
            "nearest_own_kind"
        ] == pytest.approx(194 / 240)
        assert _representation(body, "pixels, by angle")[
            "nearest_own_kind"
        ] == pytest.approx(196 / 240)
        assert _representation(body, "pixels less their own mean, by angle")[
            "nearest_own_kind"
        ] == pytest.approx(212 / 240)
        assert _representation(body, "hidden 16, trained")[
            "nearest_own_kind"
        ] == pytest.approx(233 / 240)
        assert _representation(body, "hidden 16, trained, by distance")[
            "nearest_own_kind"
        ] == pytest.approx(233 / 240)

    def test_pixel_cosines_barely_tell_kinds_apart(self) -> None:
        pixels = _representation(_get("/seed/0"), "pixels, by angle")

        assert pixels["within"] == pytest.approx(0.5937, abs=FOUR)
        assert pixels["across"] == pytest.approx(0.5799, abs=FOUR)

    def test_the_three_layers(self) -> None:
        body = _get("/seed/0")

        assert _representation(body, "flattened 128, trained")[
            "nearest_own_kind"
        ] == pytest.approx(231 / 240)
        assert _representation(body, "scores 4, trained")[
            "nearest_own_kind"
        ] == pytest.approx(229 / 240)
        flattened = _representation(body, "flattened 128, trained")
        assert flattened["within"] == pytest.approx(0.5143, abs=FOUR)
        assert flattened["across"] == pytest.approx(0.4062, abs=FOUR)

    def test_the_untrained_network(self) -> None:
        body = _get("/seed/0")

        assert body["untrained_accuracy"] == pytest.approx(0.2125, abs=FOUR)
        assert _representation(body, "flattened 128, untrained")[
            "nearest_own_kind"
        ] == pytest.approx(201 / 240)
        assert _representation(body, "hidden 16, untrained")[
            "nearest_own_kind"
        ] == pytest.approx(189 / 240)
        assert _representation(body, "scores 4, untrained")[
            "nearest_own_kind"
        ] == pytest.approx(97 / 240)

    def test_which_kinds_are_confused(self) -> None:
        body = _get("/seed/0")

        assert body["vector_confusion"] == [
            [57, 1, 2, 0],
            [0, 60, 0, 0],
            [3, 0, 57, 0],
            [0, 0, 1, 59],
        ]
        assert body["pixel_confusion"] == [
            [45, 1, 11, 3],
            [6, 44, 3, 7],
            [9, 0, 51, 0],
            [6, 0, 0, 54],
        ]

    def test_where_the_rings_land(self) -> None:
        rings = _get("/seed/0")["rings"]

        assert rings["called"] == [9, 48, 3, 0]
        assert rings["nearest_by_vector"] == [1, 15, 0, 0, 44]
        assert rings["nearest_by_pixels"] == [0, 9, 4, 1, 46]
        assert rings["nearest_by_untrained"] == [2, 9, 14, 1, 34]
        assert rings["ring_to_ring"] == pytest.approx(0.7388, abs=FOUR)
        assert rings["ring_to_kind"][1] == pytest.approx(0.7294, abs=FOUR)
        assert rings["gathered"] == {
            "pixels, by distance": 46,
            "pixels less their own mean, by angle": 50,
            "flattened 128, trained": 50,
            "hidden 16, trained": 44,
            "scores 4, trained": 41,
            "hidden 16, untrained": 34,
        }

    def test_seed_zero_is_the_reference(self) -> None:
        body = _get("/seed/0")

        assert body["held_out_accuracy"] == pytest.approx(0.9708, abs=FOUR)
        assert body["against_reference"] is None


class TestOtherSeeds:
    @pytest.mark.parametrize(
        ("seed", "accuracy", "hidden", "flattened", "scores", "untrained"),
        [
            (1, 0.9208, 230, 234, 219, 180),
            (2, 0.9667, 235, 237, 230, 186),
        ],
    )
    def test_every_seed_tells_the_same_story(
        self,
        seed: int,
        accuracy: float,
        hidden: int,
        flattened: int,
        scores: int,
        untrained: int,
    ) -> None:
        body = _get(f"/seed/{seed}")

        assert body["held_out_accuracy"] == pytest.approx(accuracy, abs=FOUR)
        assert _representation(body, "hidden 16, trained")[
            "nearest_own_kind"
        ] == pytest.approx(hidden / 240)
        assert _representation(body, "flattened 128, trained")[
            "nearest_own_kind"
        ] == pytest.approx(flattened / 240)
        assert _representation(body, "scores 4, trained")[
            "nearest_own_kind"
        ] == pytest.approx(scores / 240)
        assert _representation(body, "hidden 16, untrained")[
            "nearest_own_kind"
        ] == pytest.approx(untrained / 240)

    @pytest.mark.parametrize(
        ("seed", "gathered", "called_square"),
        [
            (
                1,
                {
                    "flattened 128, trained": 55,
                    "hidden 16, trained": 45,
                    "scores 4, trained": 32,
                },
                38,
            ),
            (
                2,
                {
                    "flattened 128, trained": 55,
                    "hidden 16, trained": 43,
                    "scores 4, trained": 29,
                },
                37,
            ),
        ],
    )
    def test_rings_under_other_seeds(
        self, seed: int, gathered: dict[str, int], called_square: int
    ) -> None:
        rings = _get(f"/seed/{seed}")["rings"]

        for name, count in gathered.items():
            assert rings["gathered"][name] == count
        assert rings["called"][1] == called_square

    @pytest.mark.parametrize(
        ("seed", "same_picture", "same_nearest", "same_kind"),
        [(1, -0.1387, 40 / 240, 225 / 240), (2, 0.0187, 45 / 240, 230 / 240)],
    )
    def test_coordinates_do_not_carry_between_seeds(
        self, seed: int, same_picture: float, same_nearest: float, same_kind: float
    ) -> None:
        against = _get(f"/seed/{seed}")["against_reference"]

        assert against["same_picture_cosine"] == pytest.approx(same_picture, abs=FOUR)
        assert against["same_nearest_picture"] == pytest.approx(same_nearest)
        assert against["same_nearest_kind"] == pytest.approx(same_kind)


class TestNeighbours:
    def test_the_disc_both_ways(self) -> None:
        response = client.post(BASE + "/neighbours", json={"position": 2})
        body = response.json()

        assert response.status_code == 200
        assert body["kind"] == "disc"
        assert body["pool_size"] == 240
        assert [row["kind"] for row in body["by_vector"]] == [
            "cross",
            "disc",
            "disc",
            "disc",
        ]
        assert body["by_vector"][0]["cosine"] == pytest.approx(0.9061, abs=FOUR)
        assert [row["kind"] for row in body["by_pixels"]] == [
            "cross",
            "cross",
            "disc",
            "cross",
        ]
        assert body["by_pixels"][0]["gap"] == pytest.approx(1.9275, abs=FOUR)

    def test_the_misses_each_way(self) -> None:
        body = client.post(BASE + "/neighbours", json={"position": 0}).json()

        assert len(body["pixel_misses"]) == 46
        assert body["vector_misses"] == [2, 14, 35, 52, 54, 68, 140]
        fixed = [
            miss for miss in body["pixel_misses"] if miss not in body["vector_misses"]
        ]
        assert len(fixed) == 43

    def test_a_ring_is_named_a_square(self) -> None:
        body = client.post(
            BASE + "/neighbours", json={"collection": "rings", "position": 0}
        ).json()

        assert body["kind"] == "ring"
        assert body["called"] == "square"
        assert body["pool_size"] == 300
        assert body["by_vector"][0]["kind"] == "ring"

    def test_the_query_is_never_its_own_neighbour(self) -> None:
        body = client.post(
            BASE + "/neighbours", json={"position": 5, "n_results": 8}
        ).json()

        assert all(
            not (row["collection"] == "held_out" and row["position"] == 5)
            for row in body["by_vector"] + body["by_pixels"]
        )
        assert len(body["by_vector"]) == 8

    def test_an_untrained_network_can_be_asked(self) -> None:
        body = client.post(
            BASE + "/neighbours", json={"position": 0, "network": "untrained"}
        ).json()

        assert len(body["vector_misses"]) == 240 - 189

    def test_a_ring_that_is_not_there_is_refused_in_plain_words(self) -> None:
        response = client.post(
            BASE + "/neighbours", json={"collection": "rings", "position": 70}
        )

        assert response.status_code == 422
        assert response.json()["detail"] == (
            "there are 60 rings, numbered 0 to 59, so there is no ring 70"
        )

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(BASE + "/neighbours", json={"picture": 3})

        assert response.status_code == 422


class TestEdges:
    def test_what_a_length_is(self) -> None:
        lengths = _get("/edges")["lengths"]

        assert lengths["shortest"] == pytest.approx(1.4867, abs=FOUR)
        assert lengths["longest"] == pytest.approx(3.9429, abs=FOUR)
        assert lengths["mean"] == pytest.approx(2.9769, abs=FOUR)
        assert lengths["bound"] == 4.0
        assert lengths["saturated_share"] == pytest.approx(0.2437, abs=FOUR)
        assert lengths["confidence_correlation"] == pytest.approx(0.4355, abs=FOUR)

    def test_every_dark_blank_is_the_same_picture(self) -> None:
        black, dim, background, bright = _get("/edges")["blanks"]

        assert black["lit_after_first_convolution"] == 0
        assert dim["lit_after_first_convolution"] == 0
        assert dim["cosine_to_all_black"] == pytest.approx(1.0)
        assert black["called"] == "bar"
        assert black["probabilities"][3] == pytest.approx(0.8921, abs=FOUR)
        assert black["cosine_to_kind"][3] == pytest.approx(0.7629, abs=FOUR)
        assert background["lit_after_first_convolution"] == 464
        assert background["cosine_to_all_black"] == pytest.approx(0.6402, abs=FOUR)
        assert bright["called"] == "bar"
        assert bright["probabilities"][1] == pytest.approx(0.4309, abs=FOUR)

    def test_a_dimmed_picture_moves(self) -> None:
        sweeps = _get("/edges")["brightness"]
        cross = sweeps[0]["steps"]
        by_factor = {step["factor"]: step for step in cross}

        assert sweeps[0]["kind"] == "cross"
        assert by_factor[1.0]["cosine_to_original"] == pytest.approx(1.0)
        assert by_factor[0.5]["cosine_to_original"] == pytest.approx(0.5697, abs=FOUR)
        assert by_factor[0.25]["cosine_to_original"] == pytest.approx(-0.1928, abs=FOUR)
        assert by_factor[0.25]["called"] == "bar"
        assert by_factor[2.0]["cosine_to_original"] == pytest.approx(0.9084, abs=FOUR)
        assert all(sweep["steps"][0]["called"] == "bar" for sweep in sweeps)


IDENTIFIER = re.compile(r"[a-z][A-Z]|[A-Za-z]_[A-Za-z]|\boop_ml\b|library|\bAPI\b")


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        # A key with a space in it is a label a widget shows, so it is read
        # too; a key without one is a field name the page never displays. The
        # collection a picture came from is a request value the widgets map
        # to their own words, so it is not page text either.
        shown_keys = [key for key in value if " " in key]
        return shown_keys + [
            text
            for key, item in value.items()
            if key != "collection"
            for text in _strings(item)
        ]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


class TestNothingNamesTheCode:
    @pytest.mark.parametrize("path", ["/walk", "/sphere", "/seed/0", "/edges"])
    def test_no_identifier_in_a_fixed_answer(self, path: str) -> None:
        for text in _strings(_get(path)):
            assert not IDENTIFIER.search(text), text

    def test_no_identifier_in_a_neighbour_answer_or_refusal(self) -> None:
        answered = client.post(
            BASE + "/neighbours", json={"collection": "rings", "position": 3}
        ).json()
        refused = client.post(
            BASE + "/neighbours", json={"collection": "rings", "position": 99}
        ).json()

        for text in _strings(answered) + [refused["detail"]]:
            assert not IDENTIFIER.search(text), text
