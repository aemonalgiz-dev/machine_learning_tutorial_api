"""The restricted Boltzmann machine endpoints, pinned to the page's figures.

The page stores the Hopfield page's T, L and cross on a five-by-five grid,
written as 0 and 1, and learns them with three hidden units over five hundred
epochs. Before any learning the free energy of every row is minus three times
the natural logarithm of two, -2.0794, to within the noise in the starting
weights, and the readout shows -2.08, -2.10 and -2.10. After learning each
shape has its own code over the three hidden units, the reconstruction error
is 0.0057, the three free energies are -24.50, -24.78 and -23.88, and the T
with the same five cells flipped that the Hopfield page's first scramble flips
reads as the T's code, rebuilds as the T, and sits at -11.82, higher than
every shape the machine was shown. All pinned.
"""

import math

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

FIT = "/concepts/rbm/fit"
RECONSTRUCT = "/concepts/rbm/reconstruct"

HIDDEN_UNITS = 3
EPOCHS = 500


def shape(rows: list[str]) -> list[int]:
    return [1 if cell == "#" else 0 for row in rows for cell in row]


T_SHAPE = shape(["#####", "..#..", "..#..", "..#..", "..#.."])
L_SHAPE = shape(["#....", "#....", "#....", "#....", "#####"])
CROSS_SHAPE = shape(["#...#", ".#.#.", "..#..", ".#.#.", "#...#"])

THREE_SHAPES = [T_SHAPE, L_SHAPE, CROSS_SHAPE]

# The widget's damage button flips the same five cells the Hopfield page's
# first scramble flips.
DAMAGED_CELLS = [5, 7, 8, 23, 24]


def flipped_at(pattern: list[int], cells: list[int]) -> list[int]:
    return [
        1 - value if index in cells else value for index, value in enumerate(pattern)
    ]


DAMAGED_T = flipped_at(T_SHAPE, DAMAGED_CELLS)


def fitted(
    patterns: list[list[int]], n_hidden_units: int = HIDDEN_UNITS, epochs: int = EPOCHS
) -> dict:
    response = client.post(
        FIT,
        json={
            "patterns": patterns,
            "n_hidden_units": n_hidden_units,
            "max_epochs": epochs,
        },
    )
    assert response.status_code == 200
    return response.json()


def reconstructed(
    patterns: list[list[int]],
    probe: list[int],
    n_hidden_units: int = HIDDEN_UNITS,
    epochs: int = EPOCHS,
) -> dict:
    response = client.post(
        RECONSTRUCT,
        json={
            "patterns": patterns,
            "n_hidden_units": n_hidden_units,
            "max_epochs": epochs,
            "probe": probe,
        },
    )
    assert response.status_code == 200
    return response.json()


def rounded(values: list[float]) -> list[int]:
    return [round(value) for value in values]


class TestBeforeLearning:
    """The one figure that can be checked by hand, minus n times ln 2."""

    def test_the_starting_free_energies_are_the_hand_figure_to_two_decimals(
        self,
    ) -> None:
        body = fitted(THREE_SHAPES)

        starting = [pattern["initial_free_energy"] for pattern in body["patterns"]]
        assert [round(value, 2) for value in starting] == [-2.08, -2.10, -2.10]

    def test_the_starting_free_energies_sit_within_the_weight_noise(self) -> None:
        # -3 ln 2 exactly would need zero weights. The library starts from
        # normal noise of spread 0.01, so the gap is the noise and nothing else.
        body = fitted(THREE_SHAPES)

        for pattern in body["patterns"]:
            assert abs(pattern["initial_free_energy"] - (-3 * math.log(2.0))) < 0.05

    @pytest.mark.parametrize("n_hidden_units", range(1, 9))
    def test_the_hand_figure_scales_with_the_hidden_count(
        self, n_hidden_units: int
    ) -> None:
        body = fitted(THREE_SHAPES, n_hidden_units=n_hidden_units, epochs=1)

        for pattern in body["patterns"]:
            assert (
                abs(pattern["initial_free_energy"] - (-n_hidden_units * math.log(2.0)))
                < 0.1
            )

    def test_the_probe_starts_at_the_same_figure_as_the_shapes(self) -> None:
        body = reconstructed(THREE_SHAPES, DAMAGED_T)

        assert round(body["probe"]["initial_free_energy"], 2) == -2.07
        assert abs(body["probe"]["initial_free_energy"] - (-3 * math.log(2.0))) < 0.05


class TestTheWorkedExample:
    """Three shapes, three hidden units, five hundred epochs, seed zero."""

    def test_the_walk_runs_every_epoch_and_does_not_settle(self) -> None:
        body = fitted(THREE_SHAPES)

        assert body["epochs_run"] == 500
        assert body["converged"] is False

    def test_the_learned_state_has_the_stated_widths(self) -> None:
        body = fitted(THREE_SHAPES)

        assert body["n_visible_units"] == 25
        assert body["n_hidden_units"] == 3
        assert len(body["weights"]) == 25
        assert all(len(row) == 3 for row in body["weights"])
        assert len(body["visible_bias"]) == 25
        assert len(body["hidden_bias"]) == 3

    def test_each_shape_has_its_own_code_over_the_hidden_units(self) -> None:
        body = fitted(THREE_SHAPES)

        codes = [
            rounded(pattern["hidden_probabilities"]) for pattern in body["patterns"]
        ]
        assert codes == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]

    def test_the_reconstruction_error_is_the_quoted_figure(self) -> None:
        body = fitted(THREE_SHAPES)

        assert round(body["reconstruction_error"], 4) == 0.0057

    def test_every_shape_is_rebuilt_to_within_a_hundredth(self) -> None:
        body = fitted(THREE_SHAPES)

        for pattern in body["patterns"]:
            assert pattern["reconstruction_error"] < 0.01

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_every_shape_rounds_back_to_itself(self, index: int) -> None:
        body = fitted(THREE_SHAPES)

        assert rounded(body["patterns"][index]["reconstruction"]) == THREE_SHAPES[index]

    def test_the_free_energies_after_learning(self) -> None:
        body = fitted(THREE_SHAPES)

        learned = [pattern["free_energy"] for pattern in body["patterns"]]
        assert [round(value, 2) for value in learned] == [-24.50, -24.78, -23.88]

    def test_learning_lowers_every_shape_far_below_its_start(self) -> None:
        body = fitted(THREE_SHAPES)

        for pattern in body["patterns"]:
            assert pattern["free_energy"] < pattern["initial_free_energy"] - 20.0

    def test_one_epoch_leaves_the_machine_near_where_it_started(self) -> None:
        # The first update barely moves the small starting weights, so the
        # reconstruction is still a grey half everywhere and the error is
        # close to the quarter that a uniform guess earns on binary cells.
        body = fitted(THREE_SHAPES, epochs=1)

        assert body["epochs_run"] == 1
        assert abs(body["reconstruction_error"] - 0.25) < 0.01
        assert body["converged"] is False
        for pattern in body["patterns"]:
            assert abs(pattern["free_energy"] - pattern["initial_free_energy"]) < 0.5


class TestOneHiddenUnit:
    """Too narrow to hold three shapes apart, so every shape gets one code."""

    def test_every_shape_gets_the_same_code(self) -> None:
        body = fitted(THREE_SHAPES, n_hidden_units=1)

        codes = [
            rounded(pattern["hidden_probabilities"]) for pattern in body["patterns"]
        ]
        assert codes == [[1], [1], [1]]

    def test_every_shape_comes_back_as_the_same_grey_average(self) -> None:
        body = fitted(THREE_SHAPES, n_hidden_units=1)

        first, second, third = (
            pattern["reconstruction"] for pattern in body["patterns"]
        )
        assert (
            max(
                abs(first_cell - second_cell)
                for first_cell, second_cell in zip(first, second, strict=True)
            )
            < 0.05
        )
        assert (
            max(
                abs(first_cell - third_cell)
                for first_cell, third_cell in zip(first, third, strict=True)
            )
            < 0.05
        )
        assert body["reconstruction_error"] > 0.15


class TestTheDamagedProbe:
    """The T with five cells flipped, pushed through the fitted machine."""

    def test_the_probe_reads_as_the_t(self) -> None:
        body = reconstructed(THREE_SHAPES, DAMAGED_T)

        assert rounded(body["probe"]["hidden_probabilities"]) == [1, 1, 0]

    def test_the_probe_rebuilds_as_the_t(self) -> None:
        body = reconstructed(THREE_SHAPES, DAMAGED_T)

        assert rounded(body["probe"]["reconstruction"]) == T_SHAPE

    def test_the_damaged_cells_are_pulled_back_toward_the_t(self) -> None:
        # Cell 7 is a stem cell the damage switched off, and comes back near
        # one; the other four were switched on, and come back near zero.
        body = reconstructed(THREE_SHAPES, DAMAGED_T)

        rebuilt = body["probe"]["reconstruction"]
        assert rebuilt[7] > 0.85
        assert all(rebuilt[cell] < 0.1 for cell in (5, 8, 23, 24))

    def test_the_probe_sits_higher_in_free_energy_than_every_shape(self) -> None:
        body = reconstructed(THREE_SHAPES, DAMAGED_T)

        assert round(body["probe"]["free_energy"], 2) == -11.82
        assert body["probe"]["free_energy"] > max(body["stored_free_energies"])

    def test_the_stored_free_energies_agree_with_the_fit_endpoint(self) -> None:
        fit_body = fitted(THREE_SHAPES)
        body = reconstructed(THREE_SHAPES, DAMAGED_T)

        assert body["stored_free_energies"] == [
            pattern["free_energy"] for pattern in fit_body["patterns"]
        ]
        assert body["epochs_run"] == fit_body["epochs_run"]
        assert body["reconstruction_error"] == fit_body["reconstruction_error"]

    def test_the_probe_is_far_from_its_own_reconstruction(self) -> None:
        # Five of twenty-five cells are moved most of the way, so the probe's
        # own error is about five over twenty-five.
        body = reconstructed(THREE_SHAPES, DAMAGED_T)

        assert abs(body["probe"]["reconstruction_error"] - 0.176) < 0.005

    def test_an_undamaged_shape_is_its_own_reconstruction(self) -> None:
        fit_body = fitted(THREE_SHAPES)
        body = reconstructed(THREE_SHAPES, T_SHAPE)

        assert body["probe"] == fit_body["patterns"][0]


class TestOneShape:
    def test_a_single_pattern_can_be_learned(self) -> None:
        body = fitted([T_SHAPE])

        assert len(body["patterns"]) == 1
        assert rounded(body["patterns"][0]["reconstruction"]) == T_SHAPE


class TestRefusals:
    def test_ragged_patterns_are_refused_in_words(self) -> None:
        response = client.post(
            FIT,
            json={
                "patterns": [T_SHAPE, T_SHAPE[:24]],
                "n_hidden_units": HIDDEN_UNITS,
                "max_epochs": EPOCHS,
            },
        )

        assert response.status_code == 422
        assert "pattern 2 holds 24 values" in response.json()["detail"]

    def test_a_probe_of_the_wrong_width_is_refused_in_words(self) -> None:
        response = client.post(
            RECONSTRUCT,
            json={
                "patterns": THREE_SHAPES,
                "n_hidden_units": HIDDEN_UNITS,
                "max_epochs": EPOCHS,
                "probe": T_SHAPE[:24],
            },
        )

        assert response.status_code == 422
        assert "the probe holds 24 values" in response.json()["detail"]

    def test_a_value_other_than_zero_or_one_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={
                "patterns": [[2, *T_SHAPE[1:]], L_SHAPE],
                "n_hidden_units": HIDDEN_UNITS,
                "max_epochs": EPOCHS,
            },
        )

        assert response.status_code == 422

    def test_a_probability_is_refused_even_though_the_library_would_take_it(
        self,
    ) -> None:
        response = client.post(
            FIT,
            json={
                "patterns": [[0.5, *T_SHAPE[1:]], L_SHAPE],
                "n_hidden_units": HIDDEN_UNITS,
                "max_epochs": EPOCHS,
            },
        )

        assert response.status_code == 422

    def test_too_many_patterns_are_refused_before_computing(self) -> None:
        response = client.post(
            FIT,
            json={
                "patterns": [T_SHAPE] * 13,
                "n_hidden_units": HIDDEN_UNITS,
                "max_epochs": EPOCHS,
            },
        )

        assert response.status_code == 422

    def test_a_pattern_too_narrow_for_a_grid_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={"patterns": [[1, 0, 1]], "n_hidden_units": 1, "max_epochs": 10},
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("n_hidden_units", [0, 9])
    def test_a_hidden_layer_outside_the_bounds_is_refused(
        self, n_hidden_units: int
    ) -> None:
        response = client.post(
            FIT,
            json={
                "patterns": THREE_SHAPES,
                "n_hidden_units": n_hidden_units,
                "max_epochs": EPOCHS,
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("epochs", [0, 501])
    def test_a_walk_outside_the_bounds_is_refused(self, epochs: int) -> None:
        response = client.post(
            FIT,
            json={
                "patterns": THREE_SHAPES,
                "n_hidden_units": HIDDEN_UNITS,
                "max_epochs": epochs,
            },
        )

        assert response.status_code == 422

    def test_an_unknown_field_is_refused_rather_than_dropped(self) -> None:
        response = client.post(
            FIT,
            json={
                "patterns": THREE_SHAPES,
                "n_hidden_units": HIDDEN_UNITS,
                "max_epochs": EPOCHS,
                "learning_rate": 0.5,
            },
        )

        assert response.status_code == 422
