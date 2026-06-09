"""The Hopfield recall endpoint, pinned to the page's hand arithmetic.

The page works four units and two stored patterns, (+1, +1, -1, -1) and
(+1, +1, -1, +1), whose weights come out as 0.5, -0.5 and -0.5 with a zero row
for the unit the two patterns disagree on, and hands the network the first
pattern with its first unit flipped. One unit moves on the first pass, the
energy falls from 0.5 to -1.5, and the second pass moves nothing. The widget
stores a T, an L and a cross on a five-by-five grid at a load of 0.12, where
the first scramble of five cells falls back into the T in two passes, and
storing three more shapes raises the load to 0.24, where the same scramble
settles into a state nobody stored. All pinned.
"""

from itertools import pairwise
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

RECALL = "/concepts/hopfield/recall"

FIRST_PATTERN = [1, 1, -1, -1]
SECOND_PATTERN = [1, 1, -1, 1]
FIRST_UNIT_FLIPPED = [-1, 1, -1, -1]


def shape(rows: list[str]) -> list[int]:
    return [1 if cell == "#" else -1 for row in rows for cell in row]


T_SHAPE = shape(["#####", "..#..", "..#..", "..#..", "..#.."])
L_SHAPE = shape(["#....", "#....", "#....", "#....", "#####"])
CROSS_SHAPE = shape(["#...#", ".#.#.", "..#..", ".#.#.", "#...#"])
SQUARE_SHAPE = shape(["#####", "#...#", "#...#", "#...#", "#####"])
Z_SHAPE = shape(["#####", "...#.", "..#..", ".#...", "#####"])
DIAMOND_SHAPE = shape(["..#..", ".#.#.", "#...#", ".#.#.", "..#.."])

THREE_SHAPES = [T_SHAPE, L_SHAPE, CROSS_SHAPE]
SIX_SHAPES = [*THREE_SHAPES, SQUARE_SHAPE, Z_SHAPE, DIAMOND_SHAPE]

# The widget's first scramble, the first five distinct cells the browser's
# seeded sequence produces.
FIRST_SCRAMBLE = [7, 24, 23, 8, 5]


def flipped_at(pattern: list[int], cells: list[int]) -> list[int]:
    return [-value if index in cells else value for index, value in enumerate(pattern)]


def recalled(patterns: list[list[int]], probe: list[int]) -> dict:
    response = client.post(RECALL, json={"patterns": patterns, "probe": probe})
    assert response.status_code == 200
    return response.json()


class TestTheWorkedExample:
    def test_the_weights_are_the_hand_worked_matrix(self) -> None:
        body = recalled([FIRST_PATTERN, SECOND_PATTERN], FIRST_UNIT_FLIPPED)

        assert body["weights"] == [
            [0.0, 0.5, -0.5, 0.0],
            [0.5, 0.0, -0.5, 0.0],
            [-0.5, -0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]

    def test_the_load_is_one_half(self) -> None:
        body = recalled([FIRST_PATTERN, SECOND_PATTERN], FIRST_UNIT_FLIPPED)

        assert body["n_units"] == 4
        assert body["n_stored_patterns"] == 2
        assert body["load"] == 0.5

    def test_the_first_pass_moves_one_unit_and_drops_the_energy_by_two(
        self,
    ) -> None:
        body = recalled([FIRST_PATTERN, SECOND_PATTERN], FIRST_UNIT_FLIPPED)

        first = body["passes"][0]
        assert body["initial_energy"] == 0.5
        assert first["pass_number"] == 1
        assert first["units_changed"] == 1
        assert first["energy_before"] == 0.5
        assert first["energy_after"] == -1.5
        assert first["state"] == FIRST_PATTERN

    def test_the_second_pass_moves_nothing_and_the_walk_is_over(self) -> None:
        body = recalled([FIRST_PATTERN, SECOND_PATTERN], FIRST_UNIT_FLIPPED)

        assert len(body["passes"]) == 2
        assert body["passes"][1]["units_changed"] == 0
        assert body["settled"] is True
        assert body["is_fixed_point"] is True

    def test_the_network_settles_into_the_first_pattern(self) -> None:
        body = recalled([FIRST_PATTERN, SECOND_PATTERN], FIRST_UNIT_FLIPPED)

        assert body["settled_state"] == FIRST_PATTERN
        assert body["settled_into"] == {"pattern_index": 0, "flipped": False}


class TestThreeShapes:
    """The widget's opening state, a T, an L and a cross at a load of 0.12."""

    def test_the_load_is_three_over_twenty_five(self) -> None:
        body = recalled(THREE_SHAPES, T_SHAPE)

        assert body["load"] == pytest.approx(0.12)

    def test_the_weight_between_the_first_two_cells_is_minus_one_twenty_fifth(
        self,
    ) -> None:
        # Both lit in the T, one lit and one dark in the L and in the cross,
        # so the three products are +1, -1 and -1, summing to -1 over 25.
        body = recalled(THREE_SHAPES, T_SHAPE)

        assert body["weights"][0][1] == pytest.approx(-0.04)
        assert body["weights"][1][0] == pytest.approx(-0.04)

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_every_stored_shape_is_a_resting_state(self, index: int) -> None:
        body = recalled(THREE_SHAPES, THREE_SHAPES[index])

        assert len(body["passes"]) == 1
        assert body["passes"][0]["units_changed"] == 0
        assert body["is_fixed_point"] is True
        assert body["settled_into"] == {"pattern_index": index, "flipped": False}

    def test_the_resting_energy_of_the_t(self) -> None:
        body = recalled(THREE_SHAPES, T_SHAPE)

        assert body["initial_energy"] == pytest.approx(-11.2)

    def test_the_first_scramble_of_the_t_falls_back_into_the_t(self) -> None:
        body = recalled(THREE_SHAPES, flipped_at(T_SHAPE, FIRST_SCRAMBLE))

        assert body["initial_energy"] == pytest.approx(-3.36)
        assert len(body["passes"]) == 2
        assert body["passes"][0]["units_changed"] == 5
        assert body["passes"][0]["energy_after"] == pytest.approx(-11.2)
        assert body["settled_state"] == T_SHAPE
        assert body["settled_into"] == {"pattern_index": 0, "flipped": False}

    def test_the_energy_never_rises_along_the_walk(self) -> None:
        body = recalled(THREE_SHAPES, flipped_at(T_SHAPE, FIRST_SCRAMBLE))

        energies = [body["initial_energy"]] + [
            recall_pass["energy_after"] for recall_pass in body["passes"]
        ]
        assert all(later <= earlier + 1e-9 for earlier, later in pairwise(energies))

    def test_a_probe_more_than_half_wrong_lands_in_the_negative(self) -> None:
        # Fifteen of the twenty-five cells flipped is closer to the T's
        # negation than to the T, and the negation is a resting state too.
        body = recalled(THREE_SHAPES, flipped_at(T_SHAPE, list(range(15))))

        assert body["settled_into"] == {"pattern_index": 0, "flipped": True}
        assert body["settled_state"] == [-value for value in T_SHAPE]


class TestSixShapes:
    """Store three more and the same scramble falls somewhere nobody stored."""

    SPURIOUS_STATE: ClassVar[list[int]] = shape(
        ["#####", ".....", "..#..", ".....", "#####"]
    )

    def test_the_load_is_six_over_twenty_five(self) -> None:
        body = recalled(SIX_SHAPES, T_SHAPE)

        assert body["load"] == pytest.approx(0.24)

    @pytest.mark.parametrize("index", range(6))
    def test_every_stored_shape_is_still_a_resting_state(self, index: int) -> None:
        body = recalled(SIX_SHAPES, SIX_SHAPES[index])

        assert body["is_fixed_point"] is True
        assert body["settled_into"] == {"pattern_index": index, "flipped": False}

    def test_the_first_scramble_of_the_t_settles_where_no_pattern_lives(
        self,
    ) -> None:
        body = recalled(SIX_SHAPES, flipped_at(T_SHAPE, FIRST_SCRAMBLE))

        assert body["settled"] is True
        assert body["is_fixed_point"] is True
        assert body["settled_into"] == {"pattern_index": None, "flipped": False}
        assert body["settled_state"] == self.SPURIOUS_STATE

    def test_the_spurious_state_sits_as_deep_as_the_z(self) -> None:
        scrambled = recalled(SIX_SHAPES, flipped_at(T_SHAPE, FIRST_SCRAMBLE))
        z_at_rest = recalled(SIX_SHAPES, Z_SHAPE)

        assert scrambled["initial_energy"] == pytest.approx(-7.04)
        assert len(scrambled["passes"]) == 2
        assert scrambled["passes"][0]["energy_after"] == pytest.approx(-14.24)
        assert z_at_rest["initial_energy"] == pytest.approx(-14.24)


class TestRefusals:
    def test_ragged_patterns_are_refused_in_words(self) -> None:
        response = client.post(
            RECALL,
            json={
                "patterns": [FIRST_PATTERN, [1, 1, -1, -1, 1]],
                "probe": FIRST_PATTERN,
            },
        )

        assert response.status_code == 422
        assert "pattern 2 holds 5 values" in response.json()["detail"]

    def test_a_value_outside_minus_one_and_one_is_refused_in_words(self) -> None:
        response = client.post(
            RECALL,
            json={"patterns": [[1, 1, 0, -1], SECOND_PATTERN], "probe": FIRST_PATTERN},
        )

        assert response.status_code == 422
        assert "must hold only -1 and 1" in response.json()["detail"]

    def test_a_probe_with_a_stray_value_is_refused_in_words(self) -> None:
        response = client.post(
            RECALL,
            json={"patterns": [FIRST_PATTERN, SECOND_PATTERN], "probe": [1, 2, -1, -1]},
        )

        assert response.status_code == 422
        assert "must hold only -1 and 1" in response.json()["detail"]

    def test_a_probe_of_the_wrong_width_is_refused(self) -> None:
        response = client.post(
            RECALL,
            json={
                "patterns": [FIRST_PATTERN, SECOND_PATTERN],
                "probe": [1, 1, -1, -1, 1],
            },
        )

        assert response.status_code == 422
        assert "4 units" in response.json()["detail"]

    def test_too_many_patterns_are_refused_before_computing(self) -> None:
        response = client.post(
            RECALL, json={"patterns": [T_SHAPE] * 9, "probe": T_SHAPE}
        )

        assert response.status_code == 422

    def test_a_pattern_too_narrow_to_be_a_network_is_refused(self) -> None:
        response = client.post(RECALL, json={"patterns": [[1]], "probe": [1]})

        assert response.status_code == 422
