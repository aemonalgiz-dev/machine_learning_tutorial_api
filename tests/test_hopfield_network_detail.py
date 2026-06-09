"""The Hopfield page's further endpoints, pinned to what the page quotes.

The storage endpoint assembles the four-cell matrix from its two outer
products and agrees with the library. The unit-by-unit walk repairs the
four-cell probe on its first visit, dropping the energy from 0.5 to -1.5, and
on the scrambled T under three shapes makes exactly five flips, each lowering
the energy by twice the weighted sum the cell read. The two-unit oscillator
alternates forever under the synchronous rule and settles on the first sweep
under the asynchronous one, into whichever memory the first-visited unit
chooses. The path from the T to its negation climbs to 1.44 and returns to
-11.2 at both ends; the odd mixture of the three shapes is a fixed point at
-7.84 that nobody stored. Capacity at twenty-five units is exact at three
patterns and 0.683 at six, and at a hundred units it is 0.94 at a load of
0.10 and 0.49 at 0.20. The self connection stops the four-cell probe moving
at a self weight of 1.0, and shifts every energy by the same constant. The
contracts come back as the page's table says they do.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

STORAGE = "/concepts/hopfield/storage"
WALK = "/concepts/hopfield/walk-by-unit"
STATES = "/concepts/hopfield/states"
CAPACITY = "/concepts/hopfield/capacity"
SELF_CONNECTION = "/concepts/hopfield/self-connection"
CONTRACTS = "/concepts/hopfield/contracts"

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
FIRST_SCRAMBLE = [7, 24, 23, 8, 5]
TWO_UNIT_PATTERN = [1, -1]


def flipped_at(pattern: list[int], cells: list[int]) -> list[int]:
    return [-value if index in cells else value for index, value in enumerate(pattern)]


def posted(path: str, body: dict) -> dict:
    response = client.post(path, json=body)
    assert response.status_code == 200, response.json()
    return response.json()


class TestStorage:
    def test_the_two_outer_products_are_the_definition(self) -> None:
        body = posted(STORAGE, {"patterns": [FIRST_PATTERN, SECOND_PATTERN]})

        first, second = body["contributions"]
        assert first["products"][0] == [1, 1, -1, -1]
        assert first["products"][3] == [-1, -1, 1, 1]
        assert second["products"][3] == [1, 1, -1, 1]

    def test_the_sum_keeps_a_diagonal_the_library_clears(self) -> None:
        body = posted(STORAGE, {"patterns": [FIRST_PATTERN, SECOND_PATTERN]})

        assert body["summed"][0] == [0.5, 0.5, -0.5, 0.0]
        assert body["summed"][3] == [0.0, 0.0, 0.0, 0.5]
        assert body["weights"][0] == [0.0, 0.5, -0.5, 0.0]
        assert body["weights"][3] == [0.0, 0.0, 0.0, 0.0]
        assert body["has_self_connections"] is False

    def test_the_summed_matrix_with_the_diagonal_cleared_is_the_library_matrix(
        self,
    ) -> None:
        body = posted(STORAGE, {"patterns": THREE_SHAPES})

        for row, (summed, weights) in enumerate(
            zip(body["summed"], body["weights"], strict=True)
        ):
            for column, (candidate, actual) in enumerate(
                zip(summed, weights, strict=True)
            ):
                expected = 0.0 if row == column else candidate
                assert actual == pytest.approx(expected)

    def test_the_diagonal_before_clearing_is_the_load(self) -> None:
        body = posted(STORAGE, {"patterns": THREE_SHAPES})

        assert body["summed"][0][0] == pytest.approx(0.12)
        assert body["load"] == pytest.approx(0.12)
        assert body["weights"][0][1] == pytest.approx(-0.04)


class TestTheWalkByUnit:
    def test_the_four_cell_probe_is_repaired_on_the_first_visit(self) -> None:
        body = posted(
            WALK,
            {"patterns": [FIRST_PATTERN, SECOND_PATTERN], "probe": FIRST_UNIT_FLIPPED},
        )

        first = body["visits"][0]
        assert first["unit"] == 0
        assert first["weighted_sum"] == 1.0
        assert first["flipped"] is True
        assert first["value_before"] == -1
        assert first["value_after"] == 1
        assert first["energy_before"] == 0.5
        assert first["energy_after"] == -1.5

    def test_the_four_cell_walk_is_two_sweeps_of_four_visits(self) -> None:
        body = posted(
            WALK,
            {"patterns": [FIRST_PATTERN, SECOND_PATTERN], "probe": FIRST_UNIT_FLIPPED},
        )

        assert len(body["visits"]) == 8
        assert sum(visit["flipped"] for visit in body["visits"]) == 1
        assert [pass_["units_changed"] for pass_ in body["passes"]] == [1, 0]
        assert body["stopped_because"] == "settled"
        assert body["settled_into"] == {"pattern_index": 0, "flipped": False}

    def test_cell_four_reads_exactly_zero_and_is_left_alone(self) -> None:
        body = posted(
            WALK,
            {"patterns": [FIRST_PATTERN, SECOND_PATTERN], "probe": FIRST_UNIT_FLIPPED},
        )

        fourth = [visit for visit in body["visits"] if visit["unit"] == 3]
        assert all(visit["weighted_sum"] == 0.0 for visit in fourth)
        assert all(visit["flipped"] is False for visit in fourth)

    def test_the_scrambled_t_makes_five_flips_under_three_shapes(self) -> None:
        body = posted(
            WALK,
            {"patterns": THREE_SHAPES, "probe": flipped_at(T_SHAPE, FIRST_SCRAMBLE)},
        )

        flips = [visit for visit in body["visits"] if visit["flipped"]]
        assert len(body["visits"]) == 50
        assert [flip["unit"] for flip in flips] == [8, 7, 23, 5, 24]
        assert [flip["visit_number"] for flip in flips] == [11, 12, 14, 19, 20]
        assert body["initial_energy"] == pytest.approx(-3.36)
        assert body["passes"][-1]["energy_after"] == pytest.approx(-11.2)

    def test_every_flip_lowers_the_energy_by_twice_the_weighted_sum(self) -> None:
        body = posted(
            WALK,
            {"patterns": THREE_SHAPES, "probe": flipped_at(T_SHAPE, FIRST_SCRAMBLE)},
        )

        for visit in body["visits"]:
            drop = visit["energy_before"] - visit["energy_after"]
            if visit["flipped"]:
                assert drop == pytest.approx(2 * abs(visit["weighted_sum"]))
            else:
                assert drop == pytest.approx(0.0)

    def test_the_energy_never_rises_between_visits(self) -> None:
        body = posted(
            WALK,
            {"patterns": THREE_SHAPES, "probe": flipped_at(T_SHAPE, FIRST_SCRAMBLE)},
        )

        energies = [body["initial_energy"]] + [
            visit["energy_after"] for visit in body["visits"]
        ]
        assert all(later <= earlier + 1e-9 for earlier, later in pairwise(energies))

    def test_the_probe_cells_read_sums_pointing_back_to_the_t(self) -> None:
        body = posted(
            WALK,
            {"patterns": THREE_SHAPES, "probe": flipped_at(T_SHAPE, FIRST_SCRAMBLE)},
        )

        sums = body["passes"][0]["weighted_sums"]
        assert sums[7] == pytest.approx(0.48)
        assert sums[24] == pytest.approx(-0.48)
        assert sums[23] == pytest.approx(-0.72)
        assert sums[8] == pytest.approx(-0.72)
        assert sums[5] == pytest.approx(-0.72)

    def test_under_six_shapes_the_same_scramble_settles_somewhere_unstored(
        self,
    ) -> None:
        body = posted(
            WALK, {"patterns": SIX_SHAPES, "probe": flipped_at(T_SHAPE, FIRST_SCRAMBLE)}
        )

        flips = [visit for visit in body["visits"] if visit["flipped"]]
        assert [flip["unit"] for flip in flips] == [17, 20, 8, 5, 21]
        assert body["initial_energy"] == pytest.approx(-7.04)
        assert body["passes"][-1]["energy_after"] == pytest.approx(-14.24)
        assert body["settled_into"] == {"pattern_index": None, "flipped": False}
        assert body["is_fixed_point"] is True


class TestTheTwoUpdateRules:
    def test_synchronous_update_oscillates_with_period_two(self) -> None:
        body = posted(
            WALK,
            {
                "patterns": [TWO_UNIT_PATTERN],
                "probe": [1, 1],
                "update_rule": "synchronous",
                "max_passes": 6,
            },
        )

        assert body["visits"] == []
        assert [pass_["state"] for pass_ in body["passes"]] == [
            [-1, -1],
            [1, 1],
            [-1, -1],
            [1, 1],
            [-1, -1],
            [1, 1],
        ]
        assert all(pass_["energy_after"] == 0.5 for pass_ in body["passes"])
        assert body["settled"] is False
        assert body["stopped_because"] == "pass limit reached"
        assert body["oscillation_period"] == 2

    @pytest.mark.parametrize(
        ("seed", "expected"),
        [(0, [-1, 1]), (1, [-1, 1]), (2, [-1, 1]), (3, [1, -1]), (4, [1, -1])],
    )
    def test_asynchronous_update_settles_at_once_into_the_memory_the_first_unit_picks(
        self, seed: int, expected: list[int]
    ) -> None:
        body = posted(
            WALK, {"patterns": [TWO_UNIT_PATTERN], "probe": [1, 1], "random_seed": seed}
        )

        assert body["settled_state"] == expected
        assert body["visits"][0]["flipped"] is True
        assert body["visits"][1]["flipped"] is False
        assert body["passes"][0]["energy_after"] == -0.5
        assert body["oscillation_period"] is None

    def test_synchronous_update_settles_on_the_scrambled_t_anyway(self) -> None:
        body = posted(
            WALK,
            {
                "patterns": THREE_SHAPES,
                "probe": flipped_at(T_SHAPE, FIRST_SCRAMBLE),
                "update_rule": "synchronous",
            },
        )

        assert [pass_["units_changed"] for pass_ in body["passes"]] == [5, 0]
        assert body["settled_into"] == {"pattern_index": 0, "flipped": False}

    def test_synchronous_update_on_the_half_reversed_t_finds_a_different_rest(
        self,
    ) -> None:
        body = posted(
            WALK,
            {
                "patterns": THREE_SHAPES,
                "probe": flipped_at(T_SHAPE, list(range(15))),
                "update_rule": "synchronous",
            },
        )

        assert [pass_["units_changed"] for pass_ in body["passes"]] == [16, 12, 5, 0]
        assert body["passes"][-1]["energy_after"] == pytest.approx(-8.8)
        assert body["settled_into"] == {"pattern_index": None, "flipped": False}


class TestStates:
    def test_the_path_to_the_negation_climbs_and_returns_to_the_same_depth(
        self,
    ) -> None:
        path = [flipped_at(T_SHAPE, list(range(count))) for count in range(26)]
        body = posted(STATES, {"patterns": THREE_SHAPES, "states": path})

        energies = [report["energy"] for report in body["reports"]]
        assert energies[0] == pytest.approx(-11.2)
        assert energies[25] == pytest.approx(-11.2)
        assert max(energies) == pytest.approx(1.44)
        assert energies.index(max(energies)) == 12
        assert body["reports"][0]["is_fixed_point"] is True
        assert body["reports"][25]["is_fixed_point"] is True
        assert body["reports"][25]["equals"] == {"pattern_index": 0, "flipped": True}
        assert all(not report["is_fixed_point"] for report in body["reports"][1:25])

    def test_the_path_to_the_l_has_a_ridge_between_two_wells(self) -> None:
        differing = [index for index in range(25) if T_SHAPE[index] != L_SHAPE[index]]
        path = [
            flipped_at(T_SHAPE, differing[:count])
            for count in range(len(differing) + 1)
        ]
        body = posted(STATES, {"patterns": THREE_SHAPES, "states": path})

        energies = [report["energy"] for report in body["reports"]]
        assert len(differing) == 14
        assert energies[0] == pytest.approx(-11.2)
        assert energies[14] == pytest.approx(-11.2)
        assert max(energies) == pytest.approx(-3.52)
        assert body["reports"][14]["equals"] == {"pattern_index": 1, "flipped": False}

    def test_the_stored_shapes_energies(self) -> None:
        body = posted(STATES, {"patterns": THREE_SHAPES, "states": [T_SHAPE]})

        assert body["stored_energies"] == pytest.approx([-11.2, -11.2, -11.04])

    def test_the_odd_mixture_is_a_fixed_point_nobody_stored(self) -> None:
        mixture = [
            1 if first + second + third > 0 else -1
            for first, second, third in zip(T_SHAPE, L_SHAPE, CROSS_SHAPE, strict=True)
        ]
        body = posted(STATES, {"patterns": THREE_SHAPES, "states": [mixture]})

        report = body["reports"][0]
        assert report["is_fixed_point"] is True
        assert report["units_wanting_to_move"] == 0
        assert report["energy"] == pytest.approx(-7.84)
        assert report["agreements"] == [18, 18, 20]
        assert report["equals"] == {"pattern_index": None, "flipped": False}

    def test_under_six_shapes_the_spurious_rest_is_as_deep_as_the_z(self) -> None:
        spurious = shape(["#####", ".....", "..#..", ".....", "#####"])
        body = posted(STATES, {"patterns": SIX_SHAPES, "states": [spurious]})

        assert body["stored_energies"] == pytest.approx(
            [-11.36, -12.64, -13.76, -15.52, -14.24, -11.52]
        )
        assert body["reports"][0]["energy"] == pytest.approx(-14.24)
        assert body["reports"][0]["is_fixed_point"] is True

    def test_too_many_states_are_refused_before_computing(self) -> None:
        response = client.post(
            STATES, json={"patterns": THREE_SHAPES, "states": [T_SHAPE] * 65}
        )

        assert response.status_code == 422


class TestCapacity:
    @pytest.fixture(scope="class")
    def body(self) -> dict:
        response = client.get(CAPACITY)
        assert response.status_code == 200
        return response.json()

    def test_two_sizes_are_swept(self, body: dict) -> None:
        assert [sweep["n_units"] for sweep in body["sweeps"]] == [25, 100]
        assert [sweep["n_flipped"] for sweep in body["sweeps"]] == [5, 10]
        assert all(sweep["n_sets"] == 10 for sweep in body["sweeps"])

    def test_the_grid_is_exact_at_three_patterns_and_not_at_six(
        self, body: dict
    ) -> None:
        points = {point["n_patterns"]: point for point in body["sweeps"][0]["points"]}

        assert points[3]["exact_share"] == pytest.approx(1.0)
        assert points[3]["corrupted_share"] == pytest.approx(0.9)
        assert points[6]["exact_share"] == pytest.approx(0.683, abs=5e-4)
        assert points[6]["corrupted_share"] == pytest.approx(0.383, abs=5e-4)
        assert points[12]["exact_share"] == pytest.approx(0.117, abs=5e-4)

    def test_a_hundred_units_erodes_gradually_around_the_literature_figure(
        self, body: dict
    ) -> None:
        points = {point["n_patterns"]: point for point in body["sweeps"][1]["points"]}

        assert points[5]["exact_share"] == pytest.approx(1.0)
        assert points[10]["exact_share"] == pytest.approx(0.94)
        assert points[14]["exact_share"] == pytest.approx(0.721, abs=5e-4)
        assert points[20]["exact_share"] == pytest.approx(0.49)
        assert points[20]["corrupted_share"] == pytest.approx(0.4)
        assert points[30]["exact_share"] == pytest.approx(0.037, abs=5e-4)
        assert points[40]["corrupted_share"] == pytest.approx(0.0)

    def test_the_fixed_point_share_is_the_exact_share_asked_without_settling(
        self, body: dict
    ) -> None:
        for sweep in body["sweeps"]:
            for point in sweep["points"]:
                assert point["fixed_point_share"] == pytest.approx(
                    point["exact_share"], abs=0.06
                )

    def test_the_six_shapes_recall_better_than_six_random_patterns(
        self, body: dict
    ) -> None:
        trials = {trial["name"]: trial for trial in body["comparisons"]}

        assert trials["six shapes"]["mean_absolute_overlap"] == pytest.approx(
            0.1947, abs=5e-4
        )
        assert trials["six random patterns"]["mean_absolute_overlap"] == pytest.approx(
            0.1573, abs=5e-4
        )
        assert trials["six shapes"]["fixed_point_share"] == pytest.approx(1.0)
        assert trials["six random patterns"]["fixed_point_share"] == pytest.approx(
            0.7167, abs=5e-4
        )
        assert trials["six shapes"]["recovered_share"] == pytest.approx(
            0.6167, abs=5e-4
        )
        assert trials["six random patterns"]["recovered_share"] == pytest.approx(0.38)

    def test_the_three_shapes_overlap_less_than_three_random_patterns(
        self, body: dict
    ) -> None:
        trials = {trial["name"]: trial for trial in body["comparisons"]}

        assert trials["three shapes"]["mean_absolute_overlap"] == pytest.approx(
            0.0667, abs=5e-4
        )
        assert trials["three random patterns"][
            "mean_absolute_overlap"
        ] == pytest.approx(0.12)
        assert trials["three shapes"]["recovered_share"] == pytest.approx(
            0.9833, abs=5e-4
        )
        assert trials["three random patterns"]["recovered_share"] == pytest.approx(
            0.955
        )


class TestSelfConnection:
    @pytest.mark.parametrize(
        ("self_weight", "first_sum", "moving"),
        [(0.0, 1.0, [0]), (0.5, 0.5, [0]), (1.0, 0.0, []), (2.0, -1.0, [])],
    )
    def test_the_self_weight_votes_for_staying_put(
        self, self_weight: float, first_sum: float, moving: list[int]
    ) -> None:
        body = posted(
            SELF_CONNECTION,
            {
                "patterns": [FIRST_PATTERN, SECOND_PATTERN],
                "probe": FIRST_UNIT_FLIPPED,
                "self_weight": self_weight,
            },
        )

        assert body["weighted_sums_with"][0] == pytest.approx(first_sum)
        assert body["units_wanting_to_move_with"] == moving
        assert body["units_wanting_to_move_without"] == [0]
        assert body["largest_row_sum"] == pytest.approx(1.0)
        assert body["has_self_connections"] is (self_weight > 0.0)

    def test_the_energy_shift_is_the_same_constant_for_every_state(self) -> None:
        body = posted(
            SELF_CONNECTION,
            {
                "patterns": [FIRST_PATTERN, SECOND_PATTERN],
                "probe": FIRST_UNIT_FLIPPED,
                "self_weight": 1.0,
            },
        )

        assert body["probe_energy_without"] == 0.5
        assert body["probe_energy_with"] == -1.5
        assert body["energy_shift"] == pytest.approx(-0.5 * 1.0 * 4)

    def test_on_the_scrambled_t_a_self_weight_of_one_freezes_every_cell(self) -> None:
        frozen = posted(
            SELF_CONNECTION,
            {
                "patterns": THREE_SHAPES,
                "probe": flipped_at(T_SHAPE, FIRST_SCRAMBLE),
                "self_weight": 1.0,
            },
        )
        half = posted(
            SELF_CONNECTION,
            {
                "patterns": THREE_SHAPES,
                "probe": flipped_at(T_SHAPE, FIRST_SCRAMBLE),
                "self_weight": 0.5,
            },
        )

        assert len(frozen["units_wanting_to_move_without"]) == 5
        assert len(half["units_wanting_to_move_with"]) == 3
        assert frozen["units_wanting_to_move_with"] == []
        assert frozen["largest_row_sum"] == pytest.approx(1.44)
        assert frozen["settled_into_without"] == {"pattern_index": 0, "flipped": False}


class TestContracts:
    @pytest.fixture(scope="class")
    def probes(self) -> dict[str, dict]:
        response = client.get(CONTRACTS)
        assert response.status_code == 200
        return {probe["edge"]: probe for probe in response.json()["probes"]}

    @pytest.mark.parametrize(
        ("edge", "error"),
        [
            ("no units at all", "EmptyValuesError"),
            ("one unit", "TooFewValuesError"),
            ("a value of 0 in a pattern", "InvalidValuesError"),
            ("a non-finite value", "InvalidValuesError"),
            ("units holding different numbers of patterns", "NonEqualArrayLengthError"),
            ("two units with one name", "NonUniqueFeaturesError"),
            ("recall before fit", "NotFittedError"),
            ("a probe of the wrong width", "InvalidValuesError"),
            ("a table missing a fitted unit", "InvalidValuesError"),
            ("a table with an unknown unit", "InvalidValuesError"),
            ("a pass limit below one", "ValidationError"),
            ("an unknown keyword", "ValidationError"),
            ("an asymmetric matrix handed to the container", "InvalidValuesError"),
        ],
    )
    def test_the_refusals_are_typed(
        self, probes: dict[str, dict], edge: str, error: str
    ) -> None:
        assert probes[edge]["outcome"] == "refused"
        assert probes[edge]["error"] == error

    @pytest.mark.parametrize(
        "edge",
        [
            "one pattern",
            "a unit lit in every pattern",
            "more patterns than units",
            "the same pattern stored twice",
            "a pattern and its negation stored together",
            "a table with the units reordered",
            "a probe that is already at rest",
            "a weighted sum of exactly zero",
            "synchronous update on an oscillating probe",
            "a self-connected matrix handed to the container",
        ],
    )
    def test_the_acceptances_are_accepted(
        self, probes: dict[str, dict], edge: str
    ) -> None:
        assert probes[edge]["outcome"] == "accepted"
        assert probes[edge]["error"] is None

    def test_the_details_say_what_the_page_says(self, probes: dict[str, dict]) -> None:
        assert "5 of the 6" in probes["more patterns than units"]["detail"]
        assert "exactly doubled" in probes["the same pattern stored twice"]["detail"]
        assert (
            "storing the first pattern twice"
            in probes["a pattern and its negation stored together"]["detail"]
        )
        assert (
            "pass limit reached"
            in probes["synchronous update on an oscillating probe"]["detail"]
        )
        assert "cell [4]" in probes["a weighted sum of exactly zero"]["detail"]
        assert "unit_3" in probes["a value of 0 in a pattern"]["detail"]
        assert (
            "has_self_connections is True"
            in probes["a self-connected matrix handed to the container"]["detail"]
        )

    def test_every_probe_has_exactly_one_outcome(self, probes: dict[str, dict]) -> None:
        assert len(probes) == 23
        for probe in probes.values():
            assert (probe["outcome"] == "refused") == (probe["error"] is not None)


class TestWhatThePageQuotes:
    """The numbers the page's prose reads off the widgets, pinned by step."""

    def test_step_three_the_three_named_weights(self) -> None:
        body = posted(STORAGE, {"patterns": THREE_SHAPES})

        assert body["weights"][0][1] == pytest.approx(-0.04)
        assert body["weights"][0][4] == pytest.approx(0.04)
        assert body["weights"][0][20] == pytest.approx(0.04)

    def test_step_ten_the_probe_sums_and_the_margin_of_the_undamaged_cells(
        self,
    ) -> None:
        body = posted(
            WALK,
            {"patterns": THREE_SHAPES, "probe": flipped_at(T_SHAPE, FIRST_SCRAMBLE)},
        )

        sums = body["passes"][0]["weighted_sums"]
        undamaged = [
            abs(sums[cell]) for cell in range(25) if cell not in FIRST_SCRAMBLE
        ]
        assert min(undamaged) == pytest.approx(0.24)

    def test_step_eleven_each_cell_reads_the_state_the_previous_one_left(self) -> None:
        body = posted(
            WALK,
            {"patterns": THREE_SHAPES, "probe": flipped_at(T_SHAPE, FIRST_SCRAMBLE)},
        )

        by_visit = {visit["visit_number"]: visit for visit in body["visits"]}
        assert by_visit[12]["unit"] == 7
        assert by_visit[12]["weighted_sum"] == pytest.approx(0.56)
        assert by_visit[19]["unit"] == 5
        assert by_visit[19]["weighted_sum"] == pytest.approx(-0.96)
        assert by_visit[29]["unit"] == 7
        assert by_visit[29]["weighted_sum"] == pytest.approx(0.96)
        assert by_visit[43]["unit"] == 5
        assert by_visit[43]["weighted_sum"] == pytest.approx(-1.04)

    def test_step_thirteen_the_five_drops_add_to_the_whole_descent(self) -> None:
        body = posted(
            WALK,
            {"patterns": THREE_SHAPES, "probe": flipped_at(T_SHAPE, FIRST_SCRAMBLE)},
        )

        flips = [visit for visit in body["visits"] if visit["flipped"]]
        assert [round(flip["weighted_sum"], 2) for flip in flips] == [
            -0.72,
            0.56,
            -0.72,
            -0.96,
            -0.96,
        ]
        assert [round(flip["energy_after"], 2) for flip in flips] == [
            -4.8,
            -5.92,
            -7.36,
            -9.28,
            -11.2,
        ]
        assert sum(
            flip["energy_before"] - flip["energy_after"] for flip in flips
        ) == pytest.approx(7.84)

    def test_step_fifteen_the_path_to_the_cross(self) -> None:
        differing = [
            index for index in range(25) if T_SHAPE[index] != CROSS_SHAPE[index]
        ]
        path = [
            flipped_at(T_SHAPE, differing[:count])
            for count in range(len(differing) + 1)
        ]
        body = posted(STATES, {"patterns": THREE_SHAPES, "states": path})

        energies = [report["energy"] for report in body["reports"]]
        assert len(differing) == 12
        assert max(energies) == pytest.approx(-5.28)
        assert energies.index(max(energies)) == 6
        assert body["reports"][12]["equals"] == {"pattern_index": 2, "flipped": False}

    def test_step_sixteen_the_reversed_t_reads_the_same_sums_reversed(self) -> None:
        body = posted(
            STATES,
            {
                "patterns": THREE_SHAPES,
                "states": [T_SHAPE, [-value for value in T_SHAPE]],
            },
        )

        upright, reversed_ = body["reports"]
        assert upright["weighted_sums"][:5] == pytest.approx(
            [0.8, 0.96, 0.96, 0.96, 1.04]
        )
        assert reversed_["weighted_sums"][:5] == pytest.approx(
            [-0.8, -0.96, -0.96, -0.96, -1.04]
        )
        assert reversed_["agreements"][0] == 0
        assert min(abs(value) for value in upright["weighted_sums"]) == pytest.approx(
            0.72
        )

    def test_step_sixteen_eighteen_reversed_cells_fall_into_the_negation(self) -> None:
        response = client.post(
            "/concepts/hopfield/recall",
            json={
                "patterns": THREE_SHAPES,
                "probe": flipped_at(T_SHAPE, list(range(18))),
            },
        )

        body = response.json()
        assert body["initial_energy"] == pytest.approx(-1.12)
        assert body["passes"][0]["units_changed"] == 7
        assert body["settled_into"] == {"pattern_index": 0, "flipped": True}

    def test_step_seventeen_the_mixture_is_shallow_as_well_as_high(self) -> None:
        mixture = [
            1 if first + second + third > 0 else -1
            for first, second, third in zip(T_SHAPE, L_SHAPE, CROSS_SHAPE, strict=True)
        ]
        body = posted(STATES, {"patterns": THREE_SHAPES, "states": [mixture]})

        assert sum(1 for value in mixture if value > 0) == 6
        assert min(
            abs(value) for value in body["reports"][0]["weighted_sums"]
        ) == pytest.approx(0.16)

    def test_step_eighteen_the_six_shape_rest_and_what_it_shares(self) -> None:
        walk = posted(
            WALK, {"patterns": SIX_SHAPES, "probe": flipped_at(T_SHAPE, FIRST_SCRAMBLE)}
        )
        body = posted(
            STATES, {"patterns": SIX_SHAPES, "states": [walk["settled_state"]]}
        )

        flipped_units = [visit["unit"] for visit in walk["visits"] if visit["flipped"]]
        assert sum(unit in FIRST_SCRAMBLE for unit in flipped_units) == 2
        assert body["reports"][0]["agreements"] == [19, 17, 15, 18, 23, 10]
        assert walk["settled_state"] == shape(
            ["#####", ".....", "..#..", ".....", "#####"]
        )

    def test_step_twenty_the_grid_is_exact_through_four_patterns(self) -> None:
        body = client.get(CAPACITY).json()
        points = {point["n_patterns"]: point for point in body["sweeps"][0]["points"]}

        assert all(
            points[count]["fixed_point_share"] == pytest.approx(1.0)
            for count in (1, 2, 3, 4)
        )
        assert points[4]["corrupted_share"] == pytest.approx(0.85)
        assert points[5]["exact_share"] == pytest.approx(0.86)
        assert points[5]["corrupted_share"] == pytest.approx(0.62)

    def test_step_twenty_one_the_hundred_unit_points_the_page_names(self) -> None:
        body = client.get(CAPACITY).json()
        points = {point["n_patterns"]: point for point in body["sweeps"][1]["points"]}

        assert points[8]["exact_share"] == pytest.approx(1.0)
        assert points[12]["exact_share"] == pytest.approx(0.958, abs=5e-4)
        assert points[12]["exact_share"] > points[10]["exact_share"]
        assert points[25]["exact_share"] == pytest.approx(0.14)

    def test_step_twenty_four_all_at_once_settles_somewhere_worse(self) -> None:
        half = flipped_at(T_SHAPE, list(range(15)))
        synchronous = posted(
            WALK,
            {"patterns": THREE_SHAPES, "probe": half, "update_rule": "synchronous"},
        )
        asynchronous = posted(WALK, {"patterns": THREE_SHAPES, "probe": half})

        assert (
            sum(1 for probe, truth in zip(half, T_SHAPE, strict=True) if probe == truth)
            == 10
        )
        assert synchronous["initial_energy"] == pytest.approx(0.48)
        assert [round(pass_["energy_after"], 2) for pass_ in synchronous["passes"]] == [
            -5.28,
            -5.44,
            -8.8,
            -8.8,
        ]
        assert synchronous["is_fixed_point"] is True
        assert [pass_["units_changed"] for pass_ in asynchronous["passes"]] == [10, 0]
        assert asynchronous["passes"][0]["energy_after"] == pytest.approx(-11.2)
        assert asynchronous["settled_into"] == {"pattern_index": 0, "flipped": True}

    @pytest.mark.parametrize("seed", [5, 6, 7])
    def test_step_twenty_three_the_further_seeds_the_page_names(
        self, seed: int
    ) -> None:
        body = posted(
            WALK, {"patterns": [TWO_UNIT_PATTERN], "probe": [1, 1], "random_seed": seed}
        )

        assert body["settled_state"] == ([-1, 1] if seed == 7 else [1, -1])
        assert len(body["passes"]) == 2
