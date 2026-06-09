"""The seven further endpoints the generative page's later parts measure.

The figures pinned here are the ones the page quotes. The replayed learning
step is checked against a real fit of the same length under the same seed,
because the replay is the one place the API does arithmetic of its own and a
drift there would put a number on the page that no fit ever produced. The
contract table is checked twice over, once for what each edge did and once for
whether any name out of the code survived the rewriting into plain words.
"""

import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

FIT = "/concepts/rbm/fit"
STEP = "/concepts/rbm/step"
STATES = "/concepts/rbm/states"
EPOCH_SWEEP = "/concepts/rbm/epoch-sweep"
HIDDEN_SWEEP = "/concepts/rbm/hidden-sweep"
SEEDS = "/concepts/rbm/seeds"
AGAINST_MEMORY = "/concepts/rbm/against-memory"
CONTRACTS = "/concepts/rbm/contracts"

HIDDEN_UNITS = 3
EPOCHS = 500
LEARNING_RATE = 0.1


def shape(rows: list[str]) -> list[int]:
    return [1 if cell == "#" else 0 for row in rows for cell in row]


T_SHAPE = shape(["#####", "..#..", "..#..", "..#..", "..#.."])
L_SHAPE = shape(["#....", "#....", "#....", "#....", "#####"])
CROSS_SHAPE = shape(["#...#", ".#.#.", "..#..", ".#.#.", "#...#"])
SQUARE_SHAPE = shape(["#####", "#...#", "#...#", "#...#", "#####"])

THREE_SHAPES = [T_SHAPE, L_SHAPE, CROSS_SHAPE]

DAMAGED_CELLS = [5, 7, 8, 23, 24]
DAMAGED_T = [
    1 - value if index in DAMAGED_CELLS else value
    for index, value in enumerate(T_SHAPE)
]
NEGATED_T = [1 - value for value in T_SHAPE]

# Thirteen lit cells drawn once at random and written down, so that the page
# and this file put the identical grid to the machine.
SCATTER = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1]

# The two four-cell rows the page traces its learning step on, agreeing on the
# first three cells and differing on the last.
FOUR_CELL_PAIR = [[1, 1, 0, 0], [1, 1, 0, 1]]
FOUR_CELL_HIDDEN_UNITS = 2


def posted(path: str, body: dict) -> dict:
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def stepped(patterns: list[list[int]], n_hidden_units: int, n_steps: int) -> dict:
    return posted(
        STEP,
        {
            "patterns": patterns,
            "n_hidden_units": n_hidden_units,
            "n_steps": n_steps,
        },
    )


def fitted(patterns: list[list[int]], n_hidden_units: int, max_epochs: int) -> dict:
    return posted(
        FIT,
        {
            "patterns": patterns,
            "n_hidden_units": n_hidden_units,
            "max_epochs": max_epochs,
        },
    )


class TestTheReplayIsTheFit:
    """The step endpoint rebuilds the walk, so it has to arrive where fit does."""

    @pytest.mark.parametrize("n_steps", [1, 2, 3, 5])
    def test_the_replayed_weights_are_the_fitted_weights(self, n_steps: int) -> None:
        replayed = stepped(THREE_SHAPES, HIDDEN_UNITS, n_steps)
        learned = fitted(THREE_SHAPES, HIDDEN_UNITS, n_steps)

        assert replayed["steps"][-1]["weights_after"] == learned["weights"]

    @pytest.mark.parametrize("n_steps", [1, 3, 7])
    def test_the_replayed_biases_are_the_fitted_biases(self, n_steps: int) -> None:
        replayed = stepped(FOUR_CELL_PAIR, FOUR_CELL_HIDDEN_UNITS, n_steps)
        learned = fitted(FOUR_CELL_PAIR, FOUR_CELL_HIDDEN_UNITS, n_steps)

        assert replayed["steps"][-1]["visible_bias_after"] == learned["visible_bias"]
        assert replayed["steps"][-1]["hidden_bias_after"] == learned["hidden_bias"]

    def test_a_short_walk_is_the_start_of_a_long_one(self) -> None:
        # Which is what lets every sweep on this page refit rather than
        # remember: the fit at ten passes is the first ten passes of the fit
        # at five hundred, because both draw from the same seeded stream.
        long_walk = stepped(THREE_SHAPES, HIDDEN_UNITS, 10)
        short_walk = fitted(THREE_SHAPES, HIDDEN_UNITS, 4)

        assert long_walk["steps"][3]["weights_after"] == short_walk["weights"]


class TestTheFourCellStep:
    """The learning step traced on two rows of four cells, at two hidden units."""

    def test_the_two_counts_are_measured_on_different_blocks(self) -> None:
        first = stepped(FOUR_CELL_PAIR, FOUR_CELL_HIDDEN_UNITS, 1)["steps"][0]

        assert first["data_visible_means"] == [1.0, 1.0, 0.0, 0.5]
        assert first["chain_visible_means"] == [0.0, 0.5, 0.5, 1.0]
        assert first["chain_visible"] == [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]

    def test_the_first_cell_is_on_in_both_rows_and_off_in_both_drawn_rows(
        self,
    ) -> None:
        first = stepped(FOUR_CELL_PAIR, FOUR_CELL_HIDDEN_UNITS, 1)["steps"][0]

        assert round(first["data_correlations"][0][0], 4) == 0.5035
        assert first["chain_correlations"][0][0] == 0.0

    def test_every_weight_change_is_the_rate_times_the_difference(self) -> None:
        first = stepped(FOUR_CELL_PAIR, FOUR_CELL_HIDDEN_UNITS, 1)["steps"][0]

        for row in range(4):
            for column in range(FOUR_CELL_HIDDEN_UNITS):
                expected = LEARNING_RATE * (
                    first["data_correlations"][row][column]
                    - first["chain_correlations"][row][column]
                )
                assert first["weight_change"][row][column] == pytest.approx(expected)

    def test_the_bias_change_is_the_same_subtraction_on_the_means(self) -> None:
        first = stepped(FOUR_CELL_PAIR, FOUR_CELL_HIDDEN_UNITS, 1)["steps"][0]

        # The first cell is lit in both rows and dark in both drawn rows, so
        # its change is a tenth of one exactly; the last is half and one.
        assert first["visible_bias_after"] == [0.1, 0.05, -0.05, -0.05]

    def test_the_hidden_biases_barely_move_on_the_first_step(self) -> None:
        # Every hidden unit starts near a half for every row at both ends of
        # the chain, so the two means very nearly cancel.
        first = stepped(FOUR_CELL_PAIR, FOUR_CELL_HIDDEN_UNITS, 1)["steps"][0]

        assert max(abs(value) for value in first["hidden_bias_after"]) < 0.001

    def test_the_weights_move_from_the_noise_the_seed_drew(self) -> None:
        replayed = stepped(FOUR_CELL_PAIR, FOUR_CELL_HIDDEN_UNITS, 1)
        first = replayed["steps"][0]

        assert round(replayed["starting_weights"][0][0], 4) == 0.0013
        assert round(first["weights_after"][0][0], 4) == 0.0516
        assert first["weights_after"][0][0] == pytest.approx(
            replayed["starting_weights"][0][0] + first["weight_change"][0][0]
        )

    def test_the_tenth_step_moves_nothing_because_the_chain_drew_the_data(
        self,
    ) -> None:
        replayed = stepped(FOUR_CELL_PAIR, FOUR_CELL_HIDDEN_UNITS, 12)
        tenth = replayed["steps"][9]

        assert tenth["chain_visible"] == [
            [float(value) for value in row] for row in FOUR_CELL_PAIR
        ]
        assert tenth["largest_movement"] == 0.0
        assert tenth["weights_after"] == replayed["steps"][8]["weights_after"]

    def test_the_walk_stops_there_and_calls_itself_settled(self) -> None:
        learned = fitted(FOUR_CELL_PAIR, FOUR_CELL_HIDDEN_UNITS, EPOCHS)

        assert learned["epochs_run"] == 10
        assert learned["converged"] is True
        assert round(learned["reconstruction_error"], 4) == 0.1598

    def test_and_it_stops_with_the_rows_still_badly_rebuilt(self) -> None:
        learned = fitted(FOUR_CELL_PAIR, FOUR_CELL_HIDDEN_UNITS, EPOCHS)

        assert round(learned["patterns"][0]["reconstruction"][0], 2) == 0.65


class TestTheStepOnTheShapes:
    """The same rule on twenty-five cells, where the first step is one pass."""

    def test_the_first_step_moves_a_weight_by_the_stated_amount(self) -> None:
        replayed = stepped(THREE_SHAPES, HIDDEN_UNITS, 1)
        first = replayed["steps"][0]

        assert round(first["data_correlations"][0][0], 4) == 0.4967
        assert round(first["chain_correlations"][0][0], 4) == 0.1681
        assert round(replayed["starting_weights"][0][0], 5) == 0.00126
        assert round(first["weights_after"][0][0], 5) == 0.03412

    def test_no_parameter_moves_further_than_the_rate(self) -> None:
        replayed = stepped(THREE_SHAPES, HIDDEN_UNITS, 8)

        for one in replayed["steps"]:
            assert one["largest_movement"] <= LEARNING_RATE + 1e-12


class TestHowTheMachineScoresARow:
    """The ordering the page quotes, over rows it was shown and rows it was not."""

    @pytest.fixture(scope="class")
    def scored(self) -> dict:
        return posted(
            STATES,
            {
                "patterns": THREE_SHAPES,
                "n_hidden_units": HIDDEN_UNITS,
                "max_epochs": EPOCHS,
                "states": [
                    T_SHAPE,
                    DAMAGED_T,
                    SCATTER,
                    SQUARE_SHAPE,
                    NEGATED_T,
                ],
            },
        )

    def test_the_free_energies_are_the_quoted_figures(self, scored: dict) -> None:
        assert [round(row["free_energy"], 2) for row in scored["states"]] == [
            -24.50,
            -11.82,
            -3.71,
            -0.77,
            2.06,
        ]

    def test_the_ordering_runs_from_stored_to_reversed(self, scored: dict) -> None:
        energies = [row["free_energy"] for row in scored["states"]]

        assert energies == sorted(energies)

    def test_a_stored_pattern_reports_the_same_thing_from_either_list(
        self, scored: dict
    ) -> None:
        assert scored["states"][0] == scored["stored"][0]

    def test_the_damaged_row_reads_as_the_shape_it_came_from(
        self, scored: dict
    ) -> None:
        assert [
            round(value) for value in scored["states"][1]["hidden_probabilities"]
        ] == [round(value) for value in scored["stored"][0]["hidden_probabilities"]]

    def test_the_unseen_square_reads_as_no_stored_code(self, scored: dict) -> None:
        stored_codes = [
            [round(value) for value in row["hidden_probabilities"]]
            for row in scored["stored"]
        ]
        square = [round(value) for value in scored["states"][3]["hidden_probabilities"]]

        assert square == [0, 1, 0]
        assert square not in stored_codes

    def test_the_unseen_square_comes_back_as_most_of_the_l(self, scored: dict) -> None:
        # Its one raised hidden unit is the one the L raises, so what returns
        # is the L's stem and foot rather than the square's outline.
        rebuilt = scored["states"][3]["reconstruction"]
        stem_and_foot = [0, 5, 10, 15, 20, 21, 22, 23, 24]

        assert min(rebuilt[cell] for cell in stem_and_foot) > 0.4
        assert max(rebuilt[cell] for cell in (6, 8, 11, 13, 16, 18)) < 0.1

    def test_and_the_rest_of_its_top_row_hangs_around_a_half(
        self, scored: dict
    ) -> None:
        rebuilt = scored["states"][3]["reconstruction"]

        assert all(0.4 < rebuilt[cell] < 0.5 for cell in (1, 2, 3))

    def test_every_row_starts_at_the_same_shallow_figure(self, scored: dict) -> None:
        starting = [row["initial_free_energy"] for row in scored["states"]]

        assert max(starting) - min(starting) < 0.1


class TestWatchingTheDamageComeOut:
    """The damaged shape rebuilt at eleven walk lengths from the same seed."""

    @pytest.fixture(scope="class")
    def swept(self) -> dict:
        return posted(
            EPOCH_SWEEP,
            {
                "patterns": THREE_SHAPES,
                "n_hidden_units": HIDDEN_UNITS,
                "probe": DAMAGED_T,
                "target": T_SHAPE,
                "epoch_counts": [1, 10, 25, 50, 100, 150, 200, 300, 400, 500],
            },
        )

    def test_the_first_hundred_and_fifty_passes_barely_move_the_answer(
        self, swept: dict
    ) -> None:
        early = swept["points"][:6]

        assert all(point["cells_matching_target"] <= 18 for point in early)
        assert all(point["distance_to_target"] > 0.15 for point in early)

        by_length = {point["max_epochs"]: point for point in swept["points"]}
        assert round(by_length[1]["stored_reconstruction_error"], 4) == 0.2437
        assert by_length[100]["cells_matching_target"] == 18
        assert by_length[150]["cells_matching_target"] == 17

    def test_then_it_collapses(self, swept: dict) -> None:
        by_length = {point["max_epochs"]: point for point in swept["points"]}

        assert by_length[200]["cells_matching_target"] == 24
        assert by_length[300]["cells_matching_target"] == 25
        assert round(by_length[300]["distance_to_target"], 4) == 0.0380
        assert round(by_length[500]["distance_to_target"], 4) == 0.0061

    def test_the_probe_ends_reading_as_the_shape_it_was_damaged_from(
        self, swept: dict
    ) -> None:
        last = swept["points"][-1]

        assert [round(value) for value in last["probe_hidden_probabilities"]] == [
            1,
            1,
            0,
        ]

    def test_the_probe_sits_above_every_stored_shape_once_learning_has_bitten(
        self, swept: dict
    ) -> None:
        for point in swept["points"]:
            if point["max_epochs"] >= 100:
                assert point["probe_free_energy"] > max(point["stored_free_energies"])

    def test_the_ordering_is_noise_before_that(self, swept: dict) -> None:
        # At one pass every row scores within a tenth of every other, so which
        # is above which says nothing, and the damaged shape happens to be the
        # lowest of the four.
        first = swept["points"][0]
        every = [first["probe_free_energy"], *first["stored_free_energies"]]

        assert max(every) - min(every) < 0.3
        assert first["probe_free_energy"] < max(first["stored_free_energies"])

    def test_a_row_the_machine_never_saw_is_not_repaired_into_itself(self) -> None:
        swept = posted(
            EPOCH_SWEEP,
            {
                "patterns": THREE_SHAPES,
                "n_hidden_units": HIDDEN_UNITS,
                "probe": SQUARE_SHAPE,
                "target": SQUARE_SHAPE,
                "epoch_counts": [1, 100, 500],
            },
        )
        last = swept["points"][-1]

        assert last["cells_matching_target"] == 17
        assert round(last["distance_to_target"], 4) == 0.2263


class TestHowManyHiddenUnitsAreEnough:
    """Eight widths, refitted from the same seed on the same three shapes."""

    @pytest.fixture(scope="class")
    def swept(self) -> dict:
        return posted(
            HIDDEN_SWEEP,
            {
                "patterns": THREE_SHAPES,
                "max_epochs": EPOCHS,
                "hidden_counts": [1, 2, 3, 4, 5, 6, 7, 8],
            },
        )

    def test_one_unit_cannot_keep_three_shapes_apart(self, swept: dict) -> None:
        narrowest = swept["points"][0]

        assert narrowest["n_distinct_codes"] == 1
        assert narrowest["patterns_rebuilt_exactly"] == 0
        assert round(narrowest["reconstruction_error"], 4) == 0.1700

    def test_two_units_separate_the_shapes_and_still_lose_one(
        self, swept: dict
    ) -> None:
        two = swept["points"][1]

        assert two["n_distinct_codes"] == 3
        assert two["patterns_rebuilt_exactly"] == 2
        assert round(two["reconstruction_error"], 4) == 0.0351

    def test_three_units_rebuild_all_three(self, swept: dict) -> None:
        three = swept["points"][2]

        assert three["codes"] == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
        assert three["patterns_rebuilt_exactly"] == 3
        assert round(three["reconstruction_error"], 4) == 0.0057

    def test_wider_is_not_reliably_better(self, swept: dict) -> None:
        # Five units score worse than three and than four, which is why the
        # page reports the sweep rather than a rule about width.
        by_width = {point["n_hidden_units"]: point for point in swept["points"]}

        assert round(by_width[5]["reconstruction_error"], 4) == 0.0165
        assert by_width[5]["reconstruction_error"] > by_width[3]["reconstruction_error"]
        assert by_width[5]["reconstruction_error"] > by_width[4]["reconstruction_error"]

    def test_the_widest_scores_lowest_and_says_nothing_more(self, swept: dict) -> None:
        by_width = {point["n_hidden_units"]: point for point in swept["points"]}

        assert round(by_width[7]["reconstruction_error"], 4) == 0.0024
        assert by_width[7]["n_distinct_codes"] == 3
        assert by_width[3]["n_distinct_codes"] == 3


class TestWhatASeedFixes:
    """Four seeds, and the first of them fitted twice over."""

    @pytest.fixture(scope="class")
    def across_seeds(self) -> dict:
        return posted(
            SEEDS,
            {
                "patterns": THREE_SHAPES,
                "n_hidden_units": HIDDEN_UNITS,
                "max_epochs": EPOCHS,
                "seeds": [0, 1, 2, 3],
            },
        )

    def test_the_same_seed_twice_gives_the_identical_machine(
        self, across_seeds: dict
    ) -> None:
        assert across_seeds["repeat_weight_gap"] == 0.0
        assert across_seeds["repeat_error_gap"] == 0.0

    def test_a_different_seed_gives_a_different_machine(
        self, across_seeds: dict
    ) -> None:
        assert round(across_seeds["across_seed_weight_gap"], 3) == 5.250

    def test_every_seed_still_keeps_the_three_shapes_apart(
        self, across_seeds: dict
    ) -> None:
        assert all(run["codes_all_distinct"] for run in across_seeds["runs"])

    def test_which_unit_means_what_is_not_fixed(self, across_seeds: dict) -> None:
        first, second = across_seeds["runs"][0], across_seeds["runs"][1]

        assert first["codes"][0] == [1, 1, 0]
        assert second["codes"][0] == [0, 1, 1]

    def test_how_well_it_learns_moves_a_little_and_not_much(
        self, across_seeds: dict
    ) -> None:
        errors = [run["reconstruction_error"] for run in across_seeds["runs"]]

        assert round(min(errors), 4) == 0.0057
        assert round(max(errors), 4) == 0.0215
        assert round(across_seeds["error_spread"], 4) == 0.0157


class TestBesideTheMemoryThatStoresExactly:
    """The same three shapes given to both models, and the same rows put to both."""

    @pytest.fixture(scope="class")
    def compared(self) -> dict:
        return posted(
            AGAINST_MEMORY,
            {
                "patterns": THREE_SHAPES,
                "n_hidden_units": HIDDEN_UNITS,
                "max_epochs": EPOCHS,
                "states": [T_SHAPE, NEGATED_T, DAMAGED_T, SQUARE_SHAPE, SCATTER],
            },
        )

    def test_the_memory_scores_a_row_and_its_reversal_identically(
        self, compared: dict
    ) -> None:
        assert compared["largest_memory_reversal_gap"] == 0.0
        for report in compared["reports"]:
            assert report["memory_energy"] == report["reversed_memory_energy"]

    def test_the_machine_does_not(self, compared: dict) -> None:
        assert round(compared["largest_reversal_gap"], 2) == 26.57

    def test_the_shape_and_its_reversal_are_twenty_six_apart(
        self, compared: dict
    ) -> None:
        upright, reversed_row = compared["reports"][0], compared["reports"][1]

        assert round(upright["free_energy"], 2) == -24.50
        assert round(reversed_row["free_energy"], 2) == 2.06
        assert upright["memory_energy"] == reversed_row["memory_energy"] == -11.20

    def test_the_memory_settles_the_reversal_into_the_inverted_shape(
        self, compared: dict
    ) -> None:
        reversed_row = compared["reports"][1]

        assert reversed_row["settled_into"] == 0
        assert reversed_row["settled_inverted"] is True

    def test_both_models_repair_the_damaged_shape(self, compared: dict) -> None:
        damaged = compared["reports"][2]

        assert damaged["settled"] == T_SHAPE
        assert damaged["rebuilt_rounds_to"] == 0

    def test_the_unseen_square_is_where_they_part(self, compared: dict) -> None:
        square = compared["reports"][3]

        assert square["settled"] == L_SHAPE
        assert square["settled_into"] == 1
        assert square["rebuilt_rounds_to"] is None

    def test_the_memory_always_answers_with_something_it_stored(
        self, compared: dict
    ) -> None:
        for report in compared["reports"]:
            assert report["settled_into"] is not None

    def test_the_machine_answers_two_of_the_five_with_no_stored_shape(
        self, compared: dict
    ) -> None:
        rounding = [report["rebuilt_rounds_to"] for report in compared["reports"]]

        assert rounding.count(None) == 2
        assert len(rounding) - rounding.count(None) == 3


# --- The contract table, and the words it reaches the page in ---------------

EXPECTED_EDGES = (
    "a cell holding 2",
    "a cell holding −1",
    "a cell holding 0.5",
    "a cell holding a value that is not a number",
    "one pattern only",
    "patterns of differing widths",
    "no patterns at all",
    "more hidden units than cells",
    "no hidden units",
    "no learning passes",
    "a learning rate of zero",
    "a negative learning rate",
    "a weight read before learning",
    "a rebuild asked for before learning",
    "a probe of the wrong width",
    "a probe naming a cell the machine never saw",
    "a probe with its cells reordered",
    "hidden units handed in where cells were wanted",
    "a cell that is dark in every pattern",
    "a setting the machine does not have",
)

ACCEPTED_EDGES = frozenset(
    {
        "a cell holding 0.5",
        "one pattern only",
        "more hidden units than cells",
        "a learning rate of zero",
        "a probe with its cells reordered",
        "a cell that is dark in every pattern",
    }
)

SNAKE_CASE = re.compile(r"[a-z]+_[a-z]+")
"""A name written the way code writes it, which the page never shows."""

CAMEL_CASE = re.compile(r"\b[A-Z][a-z]+[A-Z][A-Za-z]*")
"""A class name, which the page never shows either."""

FORBIDDEN_WORDS = (
    "the library",
    "this library",
    "the API",
    "the package",
    "the backend",
    "oop_ml",
    "Error",
)
"""Ways of crediting or naming the code, which belong nowhere a reader looks."""


@pytest.fixture(scope="module")
def contract_probes() -> list[dict]:
    response = client.get(CONTRACTS)
    assert response.status_code == 200
    return response.json()["probes"]


class TestTheContractTable:
    def test_every_edge_the_page_lists_is_probed_in_order(
        self, contract_probes: list[dict]
    ) -> None:
        assert tuple(probe["edge"] for probe in contract_probes) == EXPECTED_EDGES

    def test_each_edge_was_accepted_or_refused_as_the_page_says(
        self, contract_probes: list[dict]
    ) -> None:
        for probe in contract_probes:
            expected = "accepted" if probe["edge"] in ACCEPTED_EDGES else "refused"
            assert probe["outcome"] == expected, probe["edge"]

    def test_every_probe_says_something(self, contract_probes: list[dict]) -> None:
        for probe in contract_probes:
            assert probe["detail"].strip(), probe["edge"]

    @pytest.mark.parametrize("field", ["edge", "detail"])
    def test_no_name_out_of_the_code_survives_the_rewriting(
        self, contract_probes: list[dict], field: str
    ) -> None:
        for probe in contract_probes:
            sentence = probe[field]
            assert not SNAKE_CASE.search(sentence), sentence
            assert not CAMEL_CASE.search(sentence), sentence
            for word in FORBIDDEN_WORDS:
                assert word not in sentence, sentence

    def test_a_value_outside_the_interval_is_refused_in_words(
        self, contract_probes: list[dict]
    ) -> None:
        refusal = _detail_for(contract_probes, "a cell holding 2")

        assert refusal.startswith("cell 1 must lie in [0, 1]")

    def test_a_value_between_is_read_as_a_mean(
        self, contract_probes: list[dict]
    ) -> None:
        detail = _detail_for(contract_probes, "a cell holding 0.5")

        assert "read as a mean" in detail

    def test_the_two_widths_are_reported_as_counts(
        self, contract_probes: list[dict]
    ) -> None:
        detail = _detail_for(contract_probes, "a probe of the wrong width")

        assert detail == (
            "expected exactly the 4 cells it learned from, and was handed 3"
        )

    def test_reordering_the_cells_changes_nothing(
        self, contract_probes: list[dict]
    ) -> None:
        detail = _detail_for(contract_probes, "a probe with its cells reordered")

        assert "matched by name rather than by position" in detail
        assert "differ by 0.0" in detail

    def test_a_zero_rate_settles_at_once(self, contract_probes: list[dict]) -> None:
        detail = _detail_for(contract_probes, "a learning rate of zero")

        assert "stops after 1 pass" in detail

    def test_the_table_is_the_same_on_a_second_request(self) -> None:
        first = client.get(CONTRACTS).json()
        second = client.get(CONTRACTS).json()

        assert first == second


def _detail_for(probes: list[dict], edge: str) -> str:
    return next(probe["detail"] for probe in probes if probe["edge"] == edge)


class TestTheNewEndpointsRefuse:
    def test_a_ragged_pattern_list_is_refused_by_every_endpoint(self) -> None:
        for path, extra in (
            (STEP, {"n_steps": 1}),
            (STATES, {"max_epochs": 10, "states": [T_SHAPE]}),
            (
                AGAINST_MEMORY,
                {"max_epochs": 10, "states": [T_SHAPE]},
            ),
        ):
            response = client.post(
                path,
                json={
                    "patterns": [T_SHAPE, T_SHAPE[:24]],
                    "n_hidden_units": HIDDEN_UNITS,
                    **extra,
                },
            )

            assert response.status_code == 422, path
            assert "pattern 2 holds 24 values" in response.json()["detail"]

    def test_a_scored_row_of_the_wrong_width_is_refused_in_words(self) -> None:
        response = client.post(
            STATES,
            json={
                "patterns": THREE_SHAPES,
                "n_hidden_units": HIDDEN_UNITS,
                "max_epochs": 10,
                "states": [T_SHAPE, T_SHAPE[:24]],
            },
        )

        assert response.status_code == 422
        assert "row 2 holds 24 values" in response.json()["detail"]

    def test_a_probe_of_the_wrong_width_is_refused_by_the_sweep(self) -> None:
        response = client.post(
            EPOCH_SWEEP,
            json={
                "patterns": THREE_SHAPES,
                "n_hidden_units": HIDDEN_UNITS,
                "probe": T_SHAPE[:24],
                "target": T_SHAPE,
                "epoch_counts": [10],
            },
        )

        assert response.status_code == 422
        assert "row 1 holds 24 values" in response.json()["detail"]

    @pytest.mark.parametrize("n_steps", [0, 13])
    def test_a_replay_outside_the_bounds_is_refused(self, n_steps: int) -> None:
        response = client.post(
            STEP,
            json={
                "patterns": THREE_SHAPES,
                "n_hidden_units": HIDDEN_UNITS,
                "n_steps": n_steps,
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize(
        ("path", "body"),
        [
            (
                HIDDEN_SWEEP,
                {
                    "patterns": THREE_SHAPES,
                    "max_epochs": 10,
                    "hidden_counts": [9],
                },
            ),
            (
                HIDDEN_SWEEP,
                {"patterns": THREE_SHAPES, "max_epochs": 10, "hidden_counts": []},
            ),
            (
                SEEDS,
                {
                    "patterns": THREE_SHAPES,
                    "n_hidden_units": HIDDEN_UNITS,
                    "max_epochs": 10,
                    "seeds": [-1],
                },
            ),
            (
                SEEDS,
                {
                    "patterns": THREE_SHAPES,
                    "n_hidden_units": HIDDEN_UNITS,
                    "max_epochs": 10,
                    "seeds": [0, 1, 2, 3, 4, 5, 6],
                },
            ),
        ],
    )
    def test_a_sweep_outside_the_bounds_is_refused(self, path: str, body: dict) -> None:
        assert client.post(path, json=body).status_code == 422

    def test_an_unknown_field_is_refused_rather_than_dropped(self) -> None:
        response = client.post(
            STEP,
            json={
                "patterns": THREE_SHAPES,
                "n_hidden_units": HIDDEN_UNITS,
                "n_steps": 1,
                "learning_rate": 0.5,
            },
        )

        assert response.status_code == 422
