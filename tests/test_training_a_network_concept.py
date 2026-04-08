"""The training-a-network endpoint, pinned to what one recorded run does.

The page's claim is that a hidden layer buys something a straight boundary
cannot have, and both presets are chosen so that claim is measurable rather
than rhetorical. On the rings the logistic page's model manages 0.583 and the
network reaches 1.000; on exclusive-or the straight boundary manages 0.500,
which is what guessing achieves, and the network still reaches 1.000.

The other thing pinned here is that one request is one run. The losses come
from a single walk, so they fall rather than wander, and the recorded regions
are snapshots of that same walk rather than separate fits.
"""

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

TRAIN = "/concepts/training-a-network/train"

WORKED = {
    "preset": "rings",
    "hidden_width": 6,
    "learning_rate": 0.5,
    "max_epochs": 200,
    "weight_seed": 1,
}


def trained(**overrides) -> dict:
    response = client.post(TRAIN, json={**WORKED, **overrides})
    assert response.status_code == 200, response.text
    return response.json()


class TestTheRecordedRun:
    def test_one_reading_per_epoch_in_order(self) -> None:
        body = trained()
        epochs = [reading["epoch"] for reading in body["readings"]]

        assert epochs == list(range(1, 201))

    def test_the_loss_falls_over_the_run(self) -> None:
        body = trained()

        assert body["starting_loss"] > body["final_loss"]
        assert body["starting_loss"] == pytest.approx(0.7271, abs=5e-4)
        assert body["final_loss"] == pytest.approx(0.1343, abs=5e-4)

    def test_the_first_reading_is_the_untrained_loss(self) -> None:
        # The backward pass measures on the way forward, so an epoch's loss
        # belongs to the stack as it stood before that epoch's step.
        body = trained()

        assert body["readings"][0]["loss"] == pytest.approx(body["starting_loss"])

    def test_the_walk_settles_rather_than_wandering(self) -> None:
        # One run, so the losses are comparable to each other. Over the last
        # half of the run every reading is below every reading of the first
        # tenth, which a sequence of separate fits would not guarantee.
        body = trained()
        losses = [reading["loss"] for reading in body["readings"]]

        assert max(losses[100:]) < min(losses[:20])

    def test_the_movement_is_reported_and_shrinks(self) -> None:
        body = trained()
        movements = [reading["largest_movement"] for reading in body["readings"]]

        assert all(movement >= 0.0 for movement in movements)
        assert movements[-1] < movements[0]

    def test_the_regions_are_snapshots_of_the_same_run(self) -> None:
        body = trained()
        epochs = [record["epoch"] for record in body["regions"]]

        assert epochs[0] == 0
        assert epochs[-1] == 200
        assert epochs == sorted(epochs)
        assert len(epochs) == 8

    def test_a_region_is_a_square_lattice_of_calls(self) -> None:
        body = trained()
        grid = body["regions"][-1]["regions"]

        assert grid["cells"] == 26
        assert len(grid["labels"]) == 26
        assert all(len(row) == 26 for row in grid["labels"])
        assert {value for row in grid["labels"] for value in row} <= {0, 1}

    def test_the_untrained_region_is_not_the_trained_one(self) -> None:
        body = trained()

        assert (
            body["regions"][0]["regions"]["labels"]
            != (body["regions"][-1]["regions"]["labels"])
        )


class TestWhatTheHiddenLayerBuys:
    def test_the_network_separates_the_rings_and_a_line_cannot(self) -> None:
        body = trained(preset="rings")

        assert body["final_accuracy"] == 1.0
        assert body["straight_line_accuracy"] == pytest.approx(0.583, abs=5e-4)

    def test_the_network_separates_exclusive_or_and_a_line_guesses(self) -> None:
        body = trained(preset="exclusive_or")

        assert body["final_accuracy"] == 1.0
        assert body["straight_line_accuracy"] == pytest.approx(0.5)

    def test_the_network_is_small_enough_to_count(self) -> None:
        # Two into six with six biases, then six into one with one bias.
        body = trained()

        assert body["n_parameters"] == 2 * 6 + 6 + 6 * 1 + 1

    def test_a_wider_hidden_layer_holds_more_parameters(self) -> None:
        narrow = trained(hidden_width=2)
        wide = trained(hidden_width=8)

        assert narrow["n_parameters"] == 2 * 2 + 2 + 2 + 1
        assert wide["n_parameters"] > narrow["n_parameters"]

    def test_both_presets_draw_sixty_people_in_two_classes(self) -> None:
        for preset in ("rings", "exclusive_or"):
            points = trained(preset=preset, max_epochs=1)["points"]

            assert len(points) == 60
            assert sum(1 for point in points if point["label"] == 1) == 30


class TestTheRunIsReproducible:
    def test_the_same_request_gives_the_same_run(self) -> None:
        first = trained()
        second = trained()

        assert first["readings"] == second["readings"]
        assert first["points"] == second["points"]

    def test_a_different_weight_seed_changes_the_walk(self) -> None:
        first = trained(weight_seed=1)
        second = trained(weight_seed=2)

        assert first["starting_loss"] != second["starting_loss"]
        assert first["points"] == second["points"]

    def test_the_people_do_not_depend_on_the_weight_seed(self) -> None:
        # The draw of the people is fixed so the page can quote what it shows;
        # only the starting weights move with the seed.
        assert trained(weight_seed=1)["points"] == trained(weight_seed=9)["points"]


class TestRefusals:
    def test_an_unknown_preset_is_refused(self) -> None:
        response = client.post(TRAIN, json={**WORKED, "preset": "spirals"})

        assert response.status_code == 422

    def test_too_narrow_a_hidden_layer_is_refused(self) -> None:
        response = client.post(TRAIN, json={**WORKED, "hidden_width": 1})

        assert response.status_code == 422

    def test_too_long_a_run_is_refused(self) -> None:
        response = client.post(TRAIN, json={**WORKED, "max_epochs": 301})

        assert response.status_code == 422

    def test_a_rate_of_zero_is_refused(self) -> None:
        response = client.post(TRAIN, json={**WORKED, "learning_rate": 0.0})

        assert response.status_code == 422

    def test_a_field_the_request_does_not_have_is_refused(self) -> None:
        response = client.post(TRAIN, json={**WORKED, "momentum": 0.9})

        assert response.status_code == 422
