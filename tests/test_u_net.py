"""The U-Net page's endpoints, on the shared picture collection.

Every number the page quotes is pinned here. Training is deterministic, so the
figures are pinned to four places, which a retrain on the same seed reproduces
exactly. The reports share their caches with the running process, so each of
the six networks is trained once for the whole module.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.u_net import (
    backward,
    boundary_of,
    built_u,
    picture_threshold,
    scored,
)
from oop_ml_api.main import create_app
from oop_ml_api.picture_network import reference_split

client = TestClient(create_app())

PREFIX = "/concepts/u-net"

IDENTIFIER = re.compile(r"[A-Z][a-z]+[A-Z]\w*|\w+_\w+|Error\b")
"""A class name or a parameter name, which no label on the page may carry."""

HELD_OUT = {
    (True, 0): (0.9992, 0.9917, 0.9962, 1.0000),
    (True, 1): (0.9994, 0.9951, 0.9972, 1.0000),
    (True, 2): (0.9994, 0.9949, 0.9974, 0.9999),
    (False, 0): (0.9491, 0.6228, 0.7581, 0.9975),
    (False, 1): (0.9056, 0.1630, 0.6176, 0.9787),
    (False, 2): (0.9345, 0.5588, 0.7051, 0.9928),
}
"""Pixels right, mean overlap, boundary pixels right and the rest right."""


def report(skip: bool, seed: int) -> dict:
    response = client.get(f"{PREFIX}/network", params={"skip": skip, "seed": seed})
    assert response.status_code == 200
    return response.json()


class TestMeasuringAnAnswer:
    def test_the_boundary_is_both_sides_of_the_edge_and_not_the_frame(self) -> None:
        mask = np.zeros((1, 5, 5), dtype=bool)
        mask[0, 1:4, 1:4] = True

        boundary = boundary_of(mask)[0]

        assert boundary[2, 2] == np.False_
        assert boundary[1, 1] == np.True_
        assert boundary[0, 1] == np.True_
        assert boundary[0, 0] == np.False_
        assert int(boundary.sum()) == 8 + 12

    def test_the_overlap_is_intersection_over_union(self) -> None:
        truth = np.zeros((1, 4, 4), dtype=bool)
        truth[0, :2, :2] = True
        called = np.zeros((1, 4, 4), dtype=bool)
        called[0, :2, 1:3] = True

        result = scored(called, truth)

        assert result.mean_overlap == pytest.approx(2 / 6)
        assert result.missed == 2
        assert result.invented == 2
        assert result.pixel_accuracy == pytest.approx(12 / 16)

    def test_otsu_splits_two_levels_between_them(self) -> None:
        picture = np.array([[0.1, 0.1, 0.1], [0.1, 0.9, 0.9], [0.1, 0.1, 0.9]])

        assert picture_threshold(picture) == pytest.approx(0.5)


class TestTheWalkThroughTheJoins:
    def test_the_walk_agrees_with_a_finite_difference(self) -> None:
        body = client.get(f"{PREFIX}/gradient-check").json()

        assert body["n_pictures"] == 4
        assert len(body["layers"]) == 6
        for layer in body["layers"]:
            assert layer["largest_discrepancy"] < 1e-7
            assert layer["largest_slope"] > 0.5
        assert body["pictures_largest_discrepancy"] < 1e-7

    def test_forgetting_the_skip_share_is_caught_below_the_joins_only(self) -> None:
        layers = client.get(f"{PREFIX}/gradient-check").json()["layers"]

        forgetting = [layer["largest_discrepancy_forgetting_skip"] for layer in layers]
        assert forgetting[0] == pytest.approx(6.8380, abs=1e-4)
        assert forgetting[1] == pytest.approx(4.0537, abs=1e-4)
        assert all(value < 1e-7 for value in forgetting[2:])
        assert layers[0]["largest_slope"] == pytest.approx(9.5049, abs=1e-4)

    def test_the_blame_that_crosses_the_skip(self) -> None:
        body = client.get(f"{PREFIX}/gradient-check").json()
        start = body["blame_at_start"]
        after = body["blame_after_training"]

        assert start[0]["skip_length"] == pytest.approx(1.8615, abs=1e-4)
        assert start[0]["deep_length"] == pytest.approx(6.5513, abs=1e-4)
        assert start[1]["skip_length"] == pytest.approx(5.3331, abs=1e-4)
        assert start[1]["deep_length"] == pytest.approx(5.7689, abs=1e-4)
        assert after[0]["skip_length"] == pytest.approx(0.5731, abs=1e-4)
        assert after[0]["deep_length"] == pytest.approx(0.5874, abs=1e-4)
        assert after[1]["skip_length"] == pytest.approx(0.3114, abs=1e-4)
        assert after[1]["deep_length"] == pytest.approx(0.5200, abs=1e-4)

    def test_without_skips_there_is_no_share_to_add(self) -> None:
        _, held_out = reference_split()
        chosen = held_out.subset([0, 1])
        targets = chosen.masks.reshape(2, -1).astype(float)
        network = built_u(False, 0)

        with_addition = backward(network, chosen.pictures, targets)
        without_addition = backward(
            network, chosen.pictures, targets, add_skip_share=False
        )

        assert with_addition.skip_blame == {}
        assert np.array_equal(with_addition.to_pictures, without_addition.to_pictures)


class TestTheArrangements:
    @pytest.mark.parametrize(("skip", "seed"), sorted(HELD_OUT))
    def test_held_out_scores(self, skip: bool, seed: int) -> None:
        held = report(skip, seed)["held_out"]
        pixels, overlap, boundary, interior = HELD_OUT[(skip, seed)]

        assert held["pixel_accuracy"] == pytest.approx(pixels, abs=1e-4)
        assert held["mean_overlap"] == pytest.approx(overlap, abs=1e-4)
        assert held["boundary_accuracy"] == pytest.approx(boundary, abs=1e-4)
        assert held["interior_accuracy"] == pytest.approx(interior, abs=1e-4)

    def test_the_counts_of_pixels_and_where_they_went_wrong(self) -> None:
        with_skip = report(True, 0)["held_out"]
        without = report(False, 0)["held_out"]

        assert with_skip["n_boundary"] == 12433
        assert with_skip["n_interior"] == 49007
        assert (with_skip["boundary_wrong"], with_skip["interior_wrong"]) == (47, 1)
        assert (without["boundary_wrong"], without["interior_wrong"]) == (3007, 123)
        assert (without["missed"], without["invented"]) == (1760, 1370)

    @pytest.mark.parametrize(
        ("skip", "parameters", "up"), [(True, 2521, 1596), (False, 1801, 876)]
    )
    def test_the_parameters_of_each_part(
        self, skip: bool, parameters: int, up: int
    ) -> None:
        body = report(skip, 0)

        assert body["n_parameters"] == parameters
        assert body["parameters_by_part"] == {
            "the way down": 336,
            "the middle": 584,
            "the way up": up,
            "the answer": 5,
        }
        assert [layer["parameters"] for layer in body["layers"]][-5:] == [
            1160 if skip else 584,
            0,
            436 if skip else 292,
            5,
            0,
        ]

    def test_the_joins_read_the_skip_channels(self) -> None:
        layers = report(True, 0)["layers"]

        assert layers[6]["reads"] == [16, 8, 8]
        assert layers[8]["reads"] == [12, 16, 16]
        assert [layer["reads_skip"] for layer in layers].count(True) == 2

    def test_the_curve_with_the_skips(self) -> None:
        history = report(True, 0)["history"]

        assert len(history) == 20
        assert history[0]["loss_per_pixel"] == pytest.approx(0.3126, abs=1e-4)
        assert history[9]["held_out_overlap"] == pytest.approx(0.9866, abs=1e-4)
        assert history[-1]["loss_per_pixel"] == pytest.approx(0.0023, abs=1e-4)

    def test_the_curve_without_them_falls_back_to_nothing(self) -> None:
        history = report(False, 1)["history"]

        assert history[17]["held_out_overlap"] == pytest.approx(0.5558, abs=1e-4)
        assert history[18]["held_out_overlap"] == pytest.approx(0.0403, abs=1e-4)
        assert report(False, 2)["history"][10]["held_out_overlap"] == 0.0

    @pytest.mark.parametrize(
        ("skip", "overlaps"),
        [
            (True, [0.9975, 0.9960, 0.9961, 0.9772]),
            (False, [0.6499, 0.4147, 0.7692, 0.6575]),
        ],
    )
    def test_each_kind_at_seed_zero(self, skip: bool, overlaps: list[float]) -> None:
        by_kind = report(skip, 0)["by_kind"]

        assert [entry["kind"] for entry in by_kind] == [
            "cross",
            "square",
            "disc",
            "bar",
        ]
        assert [entry["mean_overlap"] for entry in by_kind] == pytest.approx(
            overlaps, abs=1e-4
        )

    def test_the_worked_pixel_and_the_worked_overlap(self) -> None:
        with_skip = report(True, 0)
        without = report(False, 0)

        pixel = with_skip["worked_pixel"]
        assert (pixel["row"], pixel["column"], pixel["truth"]) == (5, 10, 1)
        assert pixel["score"] == pytest.approx(7.0074, abs=1e-4)
        assert pixel["probability"] == pytest.approx(0.99910, abs=1e-5)
        assert pixel["loss_term"] == pytest.approx(-np.log(pixel["probability"]))
        assert without["worked_pixel"]["score"] == pytest.approx(0.0832, abs=1e-4)
        assert without["worked_pixel"]["loss_term"] == pytest.approx(0.6524, abs=1e-4)

        overlap = without["worked_overlap"]
        assert (
            overlap["truth_pixels"],
            overlap["called_pixels"],
            overlap["intersection"],
            overlap["union"],
        ) == (28, 23, 19, 32)
        assert (overlap["n_boundary"], overlap["boundary_right"]) == (52, 39)
        assert with_skip["worked_overlap"]["overlap"] == 1.0


class TestTheRings:
    @pytest.mark.parametrize(
        ("skip", "overlaps"),
        [(True, [0.9906, 0.9700, 0.9946]), (False, [0.4867, 0.0481, 0.4507])],
    )
    def test_the_overlap_on_a_kind_never_seen(
        self, skip: bool, overlaps: list[float]
    ) -> None:
        values = [report(skip, seed)["rings"]["mean_overlap"] for seed in (0, 1, 2)]

        assert values == pytest.approx(overlaps, abs=1e-4)

    def test_where_the_ring_errors_fall(self) -> None:
        with_skip = [report(True, seed)["ring_errors"] for seed in (0, 1, 2)]
        without = report(False, 0)["ring_errors"]

        assert with_skip[0]["n_hole"] == 2046
        assert with_skip[0]["n_ring"] == 2257
        assert [entry["ring_missed"] for entry in with_skip] == [20, 70, 11]
        assert [entry["hole_called_shape"] for entry in with_skip] == [0, 2, 0]
        assert (without["hole_called_shape"], without["ring_missed"]) == (245, 908)


class TestBaselines:
    def test_knowing_the_kind_says_almost_nothing_about_where(self) -> None:
        body = client.get(f"{PREFIX}/baselines").json()
        by_name = {entry["name"]: entry["scores"] for entry in body["baselines"]}

        assert body["mask_share"] == pytest.approx(0.1108, abs=1e-4)
        assert body["classifier_accuracy"] == pytest.approx(233 / 240)
        assert by_name["the kind, from the classifier"][
            "mean_overlap"
        ] == pytest.approx(0.0807, abs=1e-4)
        assert by_name["the true kind"]["mean_overlap"] == pytest.approx(
            0.0803, abs=1e-4
        )
        assert by_name["the true kind"]["pixel_accuracy"] == pytest.approx(
            0.8976, abs=1e-4
        )

    def test_the_classifier_keeps_an_arrangement_only_until_it_flattens(self) -> None:
        shapes = client.get(f"{PREFIX}/baselines").json()["classifier_shapes"]

        assert [entry["spatial"] for entry in shapes] == [True] * 4 + [False] * 3
        assert shapes[3]["answers"] == [8, 4, 4]

    def test_a_threshold_is_the_obvious_alternative_and_it_wins(self) -> None:
        body = client.get(f"{PREFIX}/baselines").json()
        by_name = {entry["name"]: entry["scores"] for entry in body["baselines"]}

        assert body["best_cut"] == pytest.approx(0.41)
        assert body["cuts_tried"] == 41
        fixed = by_name["one cut for every picture"]
        assert fixed["mean_overlap"] == pytest.approx(0.9834, abs=1e-4)
        assert (fixed["boundary_wrong"], fixed["interior_wrong"]) == (60, 45)
        otsu = by_name["a cut for each picture"]
        assert otsu["mean_overlap"] == pytest.approx(0.99987, abs=1e-5)
        assert (otsu["boundary_wrong"], otsu["interior_wrong"]) == (0, 1)
        assert body["ring_picture_threshold"]["mean_overlap"] == 1.0
        assert body["ring_cut"]["mean_overlap"] == pytest.approx(0.9796, abs=1e-4)


class TestLarger:
    def test_the_same_weights_read_a_picture_twice_the_size(self) -> None:
        body = client.get(f"{PREFIX}/larger").json()

        assert body["side"] == 32
        assert len(body["probability"]) == 32
        assert body["scores"]["mean_overlap"] == 1.0
        assert body["changed_pixels"] == 0

    def test_a_blank_picture_is_given_no_shape(self) -> None:
        body = client.get(f"{PREFIX}/larger").json()

        assert body["blank_called_with_skip"] == 0
        assert body["blank_largest_probability"] == pytest.approx(0.0044, abs=1e-4)

    def test_only_sides_divisible_by_four_join(self) -> None:
        sides = {
            entry["side"]: entry
            for entry in client.get(f"{PREFIX}/larger").json()["sides"]
        }

        joined = [
            side
            for side, entry in sides.items()
            if entry["half_size_join_defined"] and entry["full_size_join_defined"]
        ]
        assert joined == [8, 12, 16, 20, 24, 32]
        assert sides[18]["after_first_pooling"] == 9
        assert sides[18]["repeated_once"] == 8
        assert sides[30]["answer_side"] == 28


class TestRefusals:
    def test_a_seed_outside_the_three(self) -> None:
        response = client.get(f"{PREFIX}/network", params={"seed": 5})

        assert response.status_code == 422

    def test_a_skip_that_is_not_a_yes_or_no(self) -> None:
        response = client.get(f"{PREFIX}/network", params={"skip": "sometimes"})

        assert response.status_code == 422

    def test_no_label_the_page_draws_carries_an_identifier(self) -> None:
        texts: list[str] = []
        for skip in (True, False):
            body = report(skip, 0)
            texts.append(body["label"])
            texts += [layer["label"] for layer in body["layers"]]
            texts += [layer["part"] for layer in body["layers"]]
            texts += [entry["kind_words"] for entry in body["by_kind"]]
            texts += [entry["kind_words"] for entry in body["examples"]]
        baselines = client.get(f"{PREFIX}/baselines").json()
        texts += [entry["name"] for entry in baselines["baselines"]]
        texts += [entry["description"] for entry in baselines["baselines"]]
        texts += [entry["label"] for entry in baselines["classifier_shapes"]]
        check = client.get(f"{PREFIX}/gradient-check").json()
        texts += [entry["label"] for entry in check["layers"]]
        texts += [entry["label"] for entry in check["blame_at_start"]]
        for text in texts:
            assert IDENTIFIER.search(text) is None, text
