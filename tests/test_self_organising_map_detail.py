"""The six endpoints beyond the fit, pinned to what the map page quotes.

The fit endpoint has its own module. This one covers the parts of a fit the
resting map hides, and every number here is one the page prints in prose or
draws in a widget.

The page traces one presentation by hand on a chain of three cells placed at
round numbers, (120, 30), (150, 50) and (180, 70), and presents the person at
(156, 53) at a rate of a half and a reach of one. It says the middle cell wins,
that the two beside it score 0.6065 each, that the winner halves its distance
from 6.708 to 3.354, and that the left cell closes from 42.720 to 29.765 and
the right from 29.411 to 20.492, so every cell ends nearer the person than it
began. At a reach of zero it says only the winner moves.

It reads the two schedules on a three by three grid over a hundred epochs and
says the rate falls from a half to a hundredth, the reach from 1.5 cells to a
quarter of one, that a cell one step from the winner takes 0.8007 of its step
on the first epoch and 0.000335 on the last, and that all nine cells move on
the first epoch against one on the last, the count falling to five at epoch 90
and to one at epoch 99. On a chain of six it says the count goes 6, then 5 at
epoch 80, 3 at epoch 90 and 1 at epoch 100.

It holds each of the two decays still in turn on the eighteen people along the
arch, on a chain of six. With both shrinking the map moves 0.2703 on its last
epoch, quantises at 7.9669 and arranges at 0.9860. Holding the step at a half
leaves it moving 13.5400. Holding the reach wide leaves it settled at 0.0278
and arranged at 0.9979 while quantising at 22.7539, because every cell rests in
one knot. Holding both leaves it moving 17.9446 and quantising at 21.3833.

It compares the eleven people fitted with the reach and without it. On a chain
of four the arranged map correlates grid distance against people distance at
0.9902 and quantises at 5.1435, while the same map with the reach switched off
correlates at −0.1773 and quantises better, at 3.5932. On a three by three grid
over the same eleven, which is a square laid over a line, the correlation falls
to 0.5308, and two cells at opposite corners rest 11.344 apart while two cells
one step apart rest 28.622 apart.

It says a chain with no reach is the grouping method, agreeing about who goes
with whom on the eleven people at three cells with the prototypes 0.0233 apart,
and it says the agreement is not a promise, because at four cells the two put
different people together. It fits the eleven people from six seeds and says
every seed groups them the same way while only the first reproduces the first
fit exactly, and that two of the six lay the arrangement out backwards.

Every row of the contract table is the edges endpoint's, run rather than
remembered.
"""

from itertools import pairwise
from typing import Any

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/self-organising-map"
PRESENTATION = f"{BASE}/presentation"
SCHEDULES = f"{BASE}/schedules"
ABLATION = f"{BASE}/ablation"
TOPOLOGY = f"{BASE}/topology"
VERSUS = f"{BASE}/versus-k-means"
SEEDS = f"{BASE}/seeds"
EDGES = f"{BASE}/edges"

CROWD = [
    {"x": 147, "y": 41},
    {"x": 156, "y": 53},
    {"x": 145, "y": 57},
    {"x": 159, "y": 57},
    {"x": 162, "y": 61},
    {"x": 120, "y": 25},
    {"x": 122, "y": 28},
    {"x": 118, "y": 24},
    {"x": 180, "y": 80},
    {"x": 183, "y": 83},
    {"x": 178, "y": 78},
]
"""The eleven people the page works on, with their labels taken away."""

ARCH = [
    {"x": 108, "y": 30},
    {"x": 110, "y": 39},
    {"x": 112, "y": 51},
    {"x": 116, "y": 57},
    {"x": 120, "y": 68},
    {"x": 126, "y": 71},
    {"x": 132, "y": 79},
    {"x": 138, "y": 80},
    {"x": 147, "y": 85},
    {"x": 153, "y": 82},
    {"x": 160, "y": 82},
    {"x": 168, "y": 76},
    {"x": 176, "y": 75},
    {"x": 180, "y": 65},
    {"x": 187, "y": 60},
    {"x": 188, "y": 48},
    {"x": 192, "y": 41},
    {"x": 193, "y": 30},
]
"""Eighteen people along an arch, the shape a chain of cells can follow."""

HAND_PLACED_CHAIN = [
    {"x": 120, "y": 30},
    {"x": 150, "y": 50},
    {"x": 180, "y": 70},
]
"""Three cells at round numbers, so one presentation checks with a calculator."""

PRESENTED_PERSON = {"x": 156, "y": 53}
"""The person the page shows to that chain, nearest the middle cell."""


def posted(path: str, body: dict[str, Any]) -> dict[str, Any]:
    response = client.post(path, json=body)
    assert response.status_code == 200, response.json()
    return response.json()


def presented(rate: float, radius: float) -> dict[str, Any]:
    return posted(
        PRESENTATION,
        {
            "units": HAND_PLACED_CHAIN,
            "grid_width": 3,
            "grid_height": 1,
            "person": PRESENTED_PERSON,
            "rate": rate,
            "radius": radius,
        },
    )


def cell_at(report: dict[str, Any], row: int, column: int) -> dict[str, Any]:
    return next(
        cell
        for cell in report["cells"]
        if cell["row"] == row and cell["column"] == column
    )


def pair_between(
    report: dict[str, Any], first: tuple[int, int], second: tuple[int, int]
) -> dict[str, Any]:
    return next(
        pair
        for pair in report["pairs"]
        if (pair["first_row"], pair["first_column"]) == first
        and (pair["second_row"], pair["second_column"]) == second
    )


class TestOnePresentation:
    """One person shown once to three cells placed by hand."""

    def test_the_middle_cell_wins(self) -> None:
        body = presented(0.5, 1.0)

        assert (body["winner_row"], body["winner_column"]) == (0, 1)
        assert [unit["is_winner"] for unit in body["units"]] == [False, True, False]

    def test_the_two_cells_beside_the_winner_score_the_same(self) -> None:
        body = presented(0.5, 1.0)

        assert [unit["score"] for unit in body["units"]] == [
            pytest.approx(0.6065, abs=0.00005),
            1.0,
            pytest.approx(0.6065, abs=0.00005),
        ]

    def test_the_winner_closes_half_its_distance(self) -> None:
        winner = presented(0.5, 1.0)["units"][1]

        assert (winner["after_x"], winner["after_y"]) == (153.0, 51.5)
        assert winner["distance_before"] == pytest.approx(6.708, abs=0.0005)
        assert winner["distance_after"] == pytest.approx(3.354, abs=0.0005)

    def test_the_left_cell_moves_furthest_though_it_took_the_smaller_share(
        self,
    ) -> None:
        # Its share of the step is 0.6065 against the winner's 1.0, and its own
        # gap to the person is over six times the winner's, so the product is
        # nearly four times the winner's move.
        units = presented(0.5, 1.0)["units"]

        assert units[0]["moved"] == pytest.approx(12.956, abs=0.0005)
        assert units[1]["moved"] == pytest.approx(3.354, abs=0.0005)
        assert units[2]["moved"] == pytest.approx(8.919, abs=0.0005)

    def test_every_cell_ends_nearer_the_person_than_it_began(self) -> None:
        body = presented(0.5, 1.0)

        assert [round(unit["distance_before"], 3) for unit in body["units"]] == [
            42.720,
            6.708,
            29.411,
        ]
        assert [round(unit["distance_after"], 3) for unit in body["units"]] == [
            29.765,
            3.354,
            20.492,
        ]
        for unit in body["units"]:
            assert unit["distance_after"] < unit["distance_before"]

    def test_all_three_cells_count_as_moving_at_a_reach_of_one(self) -> None:
        assert presented(0.5, 1.0)["moving_units"] == 3

    def test_at_no_reach_only_the_winner_moves_and_the_winner_moves_the_same(
        self,
    ) -> None:
        wide, none = presented(0.5, 1.0), presented(0.5, 0.0)

        assert none["moving_units"] == 1
        assert [unit["score"] for unit in none["units"]] == [0.0, 1.0, 0.0]
        assert [unit["moved"] for unit in none["units"]] == [
            0.0,
            pytest.approx(3.354, abs=0.0005),
            0.0,
        ]
        assert none["units"][1]["after_x"] == wide["units"][1]["after_x"]
        assert none["units"][1]["after_y"] == wide["units"][1]["after_y"]

    def test_a_grid_the_cells_do_not_fill_is_refused(self) -> None:
        response = client.post(
            PRESENTATION,
            json={
                "units": HAND_PLACED_CHAIN,
                "grid_width": 2,
                "grid_height": 2,
                "person": PRESENTED_PERSON,
                "rate": 0.5,
                "radius": 1.0,
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("rate", [0.0, 1.5])
    def test_a_rate_outside_its_range_is_refused(self, rate: float) -> None:
        response = client.post(
            PRESENTATION,
            json={
                "units": HAND_PLACED_CHAIN,
                "grid_width": 3,
                "grid_height": 1,
                "person": PRESENTED_PERSON,
                "rate": rate,
                "radius": 1.0,
            },
        )

        assert response.status_code == 422


class TestTheSchedules:
    """The two curves read epoch by epoch, and how many cells still move."""

    def test_the_rate_and_the_reach_at_both_ends_of_the_walk(self) -> None:
        body = posted(SCHEDULES, {"grid_width": 3, "grid_height": 3, "max_epochs": 100})

        first, last = body["epochs"][0], body["epochs"][-1]
        assert (first["rate"], first["radius"]) == (0.5, 1.5)
        assert (last["rate"], last["radius"]) == (0.01, 0.25)
        assert body["n_units"] == 9
        assert (body["middle_row"], body["middle_column"]) == (1, 1)

    def test_a_neighbouring_cell_takes_four_fifths_of_the_step_and_then_none(
        self,
    ) -> None:
        body = posted(SCHEDULES, {"grid_width": 3, "grid_height": 3, "max_epochs": 100})

        first, last = body["epochs"][0], body["epochs"][-1]
        assert first["neighbour_score"] == pytest.approx(0.8007, abs=0.00005)
        assert first["diagonal_score"] == pytest.approx(0.6412, abs=0.00005)
        assert last["neighbour_score"] == pytest.approx(0.000335, abs=0.0000005)
        assert last["diagonal_score"] < 1e-6

    def test_nine_cells_move_on_the_first_epoch_and_one_on_the_last(self) -> None:
        body = posted(SCHEDULES, {"grid_width": 3, "grid_height": 3, "max_epochs": 100})

        counts = [epoch["moving_units"] for epoch in body["epochs"]]
        assert (counts[0], counts[-1]) == (9, 1)
        assert [
            (epoch["epoch"], epoch["moving_units"])
            for previous, epoch in zip(body["epochs"], body["epochs"][1:], strict=False)
            if epoch["moving_units"] != previous["moving_units"]
        ] == [(90, 5), (99, 1)]

    def test_the_chain_of_six_sheds_cells_in_its_own_pattern(self) -> None:
        body = posted(SCHEDULES, {"grid_width": 6, "grid_height": 1, "max_epochs": 100})

        assert body["n_units"] == 6
        assert body["epochs"][0]["moving_units"] == 6
        assert [
            (epoch["epoch"], epoch["moving_units"])
            for previous, epoch in zip(body["epochs"], body["epochs"][1:], strict=False)
            if epoch["moving_units"] != previous["moving_units"]
        ] == [(80, 5), (90, 3), (100, 1)]

    def test_the_count_never_rises(self) -> None:
        body = posted(SCHEDULES, {"grid_width": 4, "grid_height": 4, "max_epochs": 60})

        counts = [epoch["moving_units"] for epoch in body["epochs"]]
        assert all(later <= earlier for earlier, later in pairwise(counts))

    def test_both_curves_are_read_against_the_budget_rather_than_the_epoch(
        self,
    ) -> None:
        # A twenty-epoch walk is not the first twenty epochs of a hundred; it is
        # the same walk stretched, so its last epoch holds the same rate and the
        # same reach as the hundredth of the longer one, and its tenth epoch is
        # already halfway down both curves where the longer walk's tenth has
        # barely begun to fall.
        short = posted(SCHEDULES, {"grid_width": 3, "grid_height": 3, "max_epochs": 20})
        long = posted(SCHEDULES, {"grid_width": 3, "grid_height": 3, "max_epochs": 100})

        assert short["epochs"][-1]["rate"] == long["epochs"][-1]["rate"]
        assert short["epochs"][-1]["radius"] == long["epochs"][-1]["radius"]
        assert short["epochs"][9]["rate"] < long["epochs"][9]["rate"]
        assert short["epochs"][9]["radius"] < long["epochs"][9]["radius"]


class TestHoldingOneDecayStill:
    """The arch fitted four times, one curve held still at a time."""

    def ablated(self) -> dict[str, dict[str, Any]]:
        body = posted(
            ABLATION,
            {"points": ARCH, "grid_width": 6, "grid_height": 1, "max_epochs": 100},
        )
        return {run["name"]: run for run in body["runs"]}

    def test_the_four_runs_are_named_and_flagged(self) -> None:
        runs = self.ablated()

        assert [
            (run["rate_decays"], run["radius_decays"]) for run in runs.values()
        ] == [
            (True, True),
            (False, True),
            (True, False),
            (False, False),
        ]

    def test_with_both_shrinking_the_map_settles_and_arranges(self) -> None:
        run = self.ablated()["both shrink"]

        assert run["final_movement"] == pytest.approx(0.2703, abs=0.00005)
        assert run["quantisation_error"] == pytest.approx(7.9669, abs=0.00005)
        assert run["correlation"] == pytest.approx(0.9860, abs=0.00005)

    def test_holding_the_step_leaves_the_map_still_travelling(self) -> None:
        # Fifty times the movement of the settled run, across a cloud about
        # eighty-five wide, which is a map reporting where it happened to be
        # when the budget ran out.
        runs = self.ablated()

        assert runs["the step held at a half"]["final_movement"] == pytest.approx(
            13.5400, abs=0.00005
        )
        assert (
            runs["the step held at a half"]["final_movement"]
            > 40 * runs["both shrink"]["final_movement"]
        )

    def test_holding_the_step_still_quantises_well_which_is_the_honest_part(
        self,
    ) -> None:
        # It is lower than the settled run's, and that is luck about where the
        # walk stopped rather than a better map, which is why the movement is
        # reported beside it.
        runs = self.ablated()

        assert runs["the step held at a half"]["quantisation_error"] == pytest.approx(
            6.9227, abs=0.00005
        )
        assert (
            runs["the step held at a half"]["quantisation_error"]
            < runs["both shrink"]["quantisation_error"]
        )

    def test_holding_the_reach_wide_settles_beautifully_and_describes_nobody(
        self,
    ) -> None:
        run = self.ablated()["the reach held wide"]

        assert run["final_movement"] == pytest.approx(0.0278, abs=0.00005)
        assert run["correlation"] == pytest.approx(0.9979, abs=0.00005)
        assert run["quantisation_error"] == pytest.approx(22.7539, abs=0.00005)

    def test_the_reach_held_wide_leaves_every_cell_in_one_knot(self) -> None:
        # The six cells of the settled run span 74.7 of the arch; held at the
        # starting reach of three, the same six span 31.2, which is why the
        # quantisation error is nearly three times as large.
        runs = self.ablated()

        assert self.widest_gap(runs["the reach held wide"]) == pytest.approx(
            31.168, abs=0.0005
        )
        assert self.widest_gap(runs["both shrink"]) == pytest.approx(74.698, abs=0.0005)

    @staticmethod
    def widest_gap(run: dict[str, Any]) -> float:
        return max(
            ((here["x"] - there["x"]) ** 2 + (here["y"] - there["y"]) ** 2) ** 0.5
            for here in run["units"]
            for there in run["units"]
        )

    def test_holding_both_fails_in_both_ways_at_once(self) -> None:
        run = self.ablated()["neither shrinks"]

        assert run["final_movement"] == pytest.approx(17.9446, abs=0.00005)
        assert run["quantisation_error"] == pytest.approx(21.3833, abs=0.00005)


class TestWhatTheReachBuys:
    """The same people fitted with the neighbourhood and without it."""

    def chain_of_four(self) -> dict[str, Any]:
        return posted(
            TOPOLOGY,
            {"points": CROWD, "grid_width": 4, "grid_height": 1, "max_epochs": 100},
        )

    def test_the_arranged_chain_correlates_the_two_distances_at_point_nine_nine(
        self,
    ) -> None:
        body = self.chain_of_four()

        assert body["organised"]["correlation"] == pytest.approx(0.9902, abs=0.00005)

    def test_the_same_chain_without_the_reach_carries_no_arrangement(self) -> None:
        body = self.chain_of_four()

        assert body["without_neighbourhood"]["correlation"] == pytest.approx(
            -0.1773, abs=0.00005
        )

    def test_the_unarranged_chain_quantises_better(self) -> None:
        # The arrangement is paid for, and this is the price. Every cell spent
        # the wide-reach epochs being dragged toward people its neighbours won.
        body = self.chain_of_four()

        assert body["organised"]["quantisation_error"] == pytest.approx(
            5.1435, abs=0.00005
        )
        assert body["without_neighbourhood"]["quantisation_error"] == pytest.approx(
            3.5932, abs=0.00005
        )

    def test_the_arranged_chain_runs_from_the_tall_end_to_the_short_end(self) -> None:
        body = self.chain_of_four()

        assert [cell["people"] for cell in body["organised"]["cells"]] == [
            [8, 9, 10],
            [1, 3, 4],
            [0, 2],
            [5, 6, 7],
        ]

    def test_the_arranged_chain_rests_where_the_page_says_it_does(self) -> None:
        cells = self.chain_of_four()["organised"]["cells"]

        assert [(round(cell["x"], 1), round(cell["y"], 1)) for cell in cells] == [
            (177.0, 76.8),
            (160.3, 59.0),
            (144.1, 46.6),
            (123.2, 28.5),
        ]

    def test_writing_the_heights_in_millimetres_hardly_moves_this_map(self) -> None:
        # The textbook caution is that a distance-based method follows whichever
        # column carries the larger numbers, and on these eleven people it does
        # not bite, because height and weight run together so closely that
        # stretching one barely changes which cell is nearest.
        millimetres = [{"x": person["x"] * 10, "y": person["y"]} for person in CROWD]
        stretched = posted(
            TOPOLOGY,
            {
                "points": millimetres,
                "grid_width": 4,
                "grid_height": 1,
                "max_epochs": 100,
            },
        )["organised"]

        assert stretched["correlation"] == pytest.approx(0.9892, abs=0.00005)
        assert [cell["people"] for cell in stretched["cells"]] == [
            cell["people"] for cell in self.chain_of_four()["organised"]["cells"]
        ]

    def test_the_arranged_chain_reads_in_order_along_both_measurements(self) -> None:
        cells = self.chain_of_four()["organised"]["cells"]

        heights = [cell["x"] for cell in cells]
        weights = [cell["y"] for cell in cells]
        assert heights == sorted(heights, reverse=True)
        assert weights == sorted(weights, reverse=True)

    def test_the_unarranged_chain_does_not(self) -> None:
        cells = self.chain_of_four()["without_neighbourhood"]["cells"]

        heights = [cell["x"] for cell in cells]
        assert heights != sorted(heights)
        assert heights != sorted(heights, reverse=True)

    def test_one_step_on_the_grid_is_not_one_distance_in_the_data(self) -> None:
        # Even where the arrangement is as good as it gets, the three pairs a
        # single step apart rest 20.4, 24.4 and 27.7 apart, which is a spread of
        # more than a third about their own middle.
        report = self.chain_of_four()["organised"]

        adjacent = sorted(
            pair["weight_distance"]
            for pair in report["pairs"]
            if pair["grid_distance"] == 1.0
        )
        assert adjacent == [
            pytest.approx(20.407, abs=0.0005),
            pytest.approx(24.364, abs=0.0005),
            pytest.approx(27.717, abs=0.0005),
        ]
        assert adjacent[-1] / adjacent[0] == pytest.approx(1.358, abs=0.0005)

    def test_three_steps_on_the_grid_is_not_three_times_one_step(self) -> None:
        report = self.chain_of_four()["organised"]

        assert pair_between(report, (0, 0), (0, 3))["weight_distance"] == pytest.approx(
            72.332, abs=0.0005
        )


class TestASquareGridOverALine:
    """Three by three over eleven people who lie along one direction."""

    def both_ways(self) -> dict[str, Any]:
        return posted(
            TOPOLOGY,
            {"points": CROWD, "grid_width": 3, "grid_height": 3, "max_epochs": 100},
        )

    def test_the_square_pays_for_its_arrangement_as_the_chain_does(self) -> None:
        body = self.both_ways()

        assert body["organised"]["quantisation_error"] == pytest.approx(
            2.8341, abs=0.00005
        )
        assert body["without_neighbourhood"]["quantisation_error"] == pytest.approx(
            0.7117, abs=0.00005
        )

    def square(self) -> dict[str, Any]:
        return posted(
            TOPOLOGY,
            {"points": CROWD, "grid_width": 3, "grid_height": 3, "max_epochs": 100},
        )["organised"]

    def test_the_correlation_falls_to_a_half_because_the_grid_has_to_fold(
        self,
    ) -> None:
        assert self.square()["correlation"] == pytest.approx(0.5308, abs=0.00005)

    def test_the_two_pairs_furthest_apart_on_the_grid_are_nothing_alike(self) -> None:
        report = self.square()

        assert pair_between(report, (0, 0), (2, 2))["weight_distance"] == pytest.approx(
            78.831, abs=0.0005
        )
        assert pair_between(report, (0, 2), (2, 0))["weight_distance"] == pytest.approx(
            11.344, abs=0.0005
        )

    def test_a_pair_one_step_apart_can_rest_further_than_a_pair_three_steps_apart(
        self,
    ) -> None:
        report = self.square()

        assert pair_between(report, (1, 0), (1, 1))["weight_distance"] == pytest.approx(
            28.622, abs=0.0005
        )
        assert (
            pair_between(report, (1, 0), (1, 1))["weight_distance"]
            > pair_between(report, (0, 2), (2, 0))["weight_distance"]
        )

    def test_the_closest_pair_in_the_data_is_a_diagonal_pair_on_the_grid(self) -> None:
        report = self.square()
        closest = min(report["pairs"], key=lambda pair: pair["weight_distance"])

        assert closest["grid_distance"] == pytest.approx(1.4142, abs=0.00005)
        assert closest["weight_distance"] == pytest.approx(4.757, abs=0.0005)

    def test_a_cell_that_won_nobody_is_reported_rather_than_hidden(self) -> None:
        report = self.square()

        empty = [cell for cell in report["cells"] if cell["n_people"] == 0]
        assert len(empty) == 3
        for cell in empty:
            assert cell["people"] == []
            assert cell["mean_x"] is None
            assert cell["mean_y"] is None

    def test_a_cell_that_won_somebody_reports_their_average(self) -> None:
        cell = cell_at(self.square(), 0, 0)

        assert cell["n_people"] == 3
        assert cell["mean_x"] == pytest.approx(120.0)
        assert cell["mean_y"] == pytest.approx(25.6667, abs=0.0005)


class TestAgainstTheGroupingMethod:
    """A chain with no reach, beside the method it collapses into."""

    def test_three_cells_and_three_groups_put_the_same_people_together(self) -> None:
        body = posted(VERSUS, {"points": CROWD, "n_units": 3, "max_epochs": 100})

        assert body["same_partition"] is True
        assert body["matched_one_to_one"] is True

    def test_the_prototypes_land_in_the_same_places(self) -> None:
        body = posted(VERSUS, {"points": CROWD, "n_units": 3, "max_epochs": 100})

        assert body["largest_prototype_gap"] == pytest.approx(0.0233, abs=0.00005)

    def test_they_agree_about_the_grouping_and_about_no_label(self) -> None:
        body = posted(VERSUS, {"points": CROWD, "n_units": 3, "max_epochs": 100})

        assert body["map_labels"] != body["kmeans_labels"]
        assert body["same_partition"] is True

    def test_the_two_quantise_the_same_people_equally_well(self) -> None:
        body = posted(VERSUS, {"points": CROWD, "n_units": 3, "max_epochs": 100})

        assert body["map_quantisation_error"] == pytest.approx(5.1887, abs=0.00005)
        assert body["kmeans_quantisation_error"] == pytest.approx(5.1912, abs=0.00005)

    def test_the_agreement_is_not_a_promise(self) -> None:
        # Four cells over the same eleven people, and the two rules part
        # company. The map visits one person at a time and takes a capped walk
        # under a decaying step; the grouping method reassigns everybody at
        # once and jumps each centre to its group's mean. Both are correct and
        # they found different arrangements.
        body = posted(VERSUS, {"points": CROWD, "n_units": 4, "max_epochs": 100})

        assert body["same_partition"] is False
        assert body["largest_prototype_gap"] == pytest.approx(8.0623, abs=0.00005)

    def test_and_at_four_cells_the_online_walk_quantised_better(self) -> None:
        body = posted(VERSUS, {"points": CROWD, "n_units": 4, "max_epochs": 100})

        assert body["map_quantisation_error"] == pytest.approx(3.5932, abs=0.00005)
        assert body["kmeans_quantisation_error"] == pytest.approx(3.6373, abs=0.00005)
        assert body["map_quantisation_error"] < body["kmeans_quantisation_error"]

    def test_the_arch_agrees_at_four_cells(self) -> None:
        body = posted(VERSUS, {"points": ARCH, "n_units": 4, "max_epochs": 100})

        assert body["same_partition"] is True
        assert body["largest_prototype_gap"] == pytest.approx(0.0535, abs=0.00005)

    def test_more_cells_than_people_is_refused_here_because_of_the_other_method(
        self,
    ) -> None:
        response = client.post(
            VERSUS, json={"points": CROWD[:2], "n_units": 3, "max_epochs": 10}
        )

        assert response.status_code == 422
        assert "grouping method" in response.json()["detail"]


class TestWhatASeedFixes:
    """The same eleven people fitted from six seeds."""

    def seeded(self) -> list[dict[str, Any]]:
        return posted(
            SEEDS,
            {
                "points": CROWD,
                "grid_width": 4,
                "grid_height": 1,
                "max_epochs": 100,
                "n_seeds": 6,
            },
        )["fits"]

    def test_every_seed_puts_the_same_people_together(self) -> None:
        assert all(fit["same_partition_as_first"] for fit in self.seeded())

    def test_only_the_first_seed_reproduces_the_first_fit_exactly(self) -> None:
        fits = self.seeded()

        assert [fit["identical_to_first"] for fit in fits] == [
            True,
            False,
            False,
            False,
            False,
            False,
        ]

    def test_a_reseeded_fit_can_lay_the_arrangement_out_backwards(self) -> None:
        # Two of the six read short to tall along the chain and four read tall
        # to short, which is why nothing on the page asserts which cell holds
        # whom, only that neighbouring cells hold similar people.
        fits = self.seeded()

        assert [fit["reading_order"] == [0, 1, 2, 3] for fit in fits] == [
            False,
            True,
            True,
            False,
            True,
            True,
        ]

    def test_every_seed_arranges_the_chain_just_as_well(self) -> None:
        for fit in self.seeded():
            assert fit["correlation"] == pytest.approx(0.9902, abs=0.0005)

    def test_every_seed_quantises_the_people_just_as_well(self) -> None:
        for fit in self.seeded():
            assert fit["quantisation_error"] == pytest.approx(5.147, abs=0.005)

    def test_a_repeated_request_answers_identically(self) -> None:
        first = self.seeded()
        second = self.seeded()

        assert first == second


class TestTheEdges:
    """Every row of the page's contract table, probed rather than remembered."""

    def outcomes(self) -> dict[str, dict[str, Any]]:
        response = client.get(EDGES)
        assert response.status_code == 200
        return {row["edge"]: row for row in response.json()["outcomes"]}

    def test_every_row_the_table_prints_is_here(self) -> None:
        assert set(self.outcomes()) == {
            "a grid with more cells than there are people",
            "twelve groups asked of the grouping method, for contrast",
            "a grid of one cell",
            "a walk of no epochs",
            "a reach wider than the grid, held there",
            "a reach of zero",
            "a column that never changes",
            "a value that is not a number",
            "an unfitted map asked to answer",
            "a query carrying a column the fit never saw",
            "a query missing one of the fitted columns",
            "a query with the fitted columns in the other order",
            "a query carrying one name twice",
            "people who already sit on their cells",
        }

    @pytest.mark.parametrize(
        "edge",
        [
            "a grid with more cells than there are people",
            "a grid of one cell",
            "a reach wider than the grid, held there",
            "a reach of zero",
            "a column that never changes",
            "a query with the fitted columns in the other order",
            "people who already sit on their cells",
        ],
    )
    def test_the_accepted_rows_are_accepted(self, edge: str) -> None:
        assert self.outcomes()[edge]["accepted"] is True

    @pytest.mark.parametrize(
        "edge",
        [
            "twelve groups asked of the grouping method, for contrast",
            "a walk of no epochs",
            "a value that is not a number",
            "an unfitted map asked to answer",
            "a query carrying a column the fit never saw",
            "a query missing one of the fitted columns",
            "a query carrying one name twice",
        ],
    )
    def test_the_refused_rows_are_refused(self, edge: str) -> None:
        assert self.outcomes()[edge]["accepted"] is False

    def test_thirty_six_cells_over_eleven_people_leaves_ten_winners(self) -> None:
        row = self.outcomes()["a grid with more cells than there are people"]

        assert row["figure"] == 10.0

    def test_the_grouping_method_refuses_the_same_request_by_name(self) -> None:
        row = self.outcomes()[
            "twelve groups asked of the grouping method, for contrast"
        ]

        assert row["detail"] == "cannot find 12 groups in 11 rows"

    def test_one_cell_rests_within_a_twentieth_of_the_mean_of_everybody(self) -> None:
        row = self.outcomes()["a grid of one cell"]

        assert row["figure"] == pytest.approx(0.055, abs=0.0005)

    def test_a_reach_that_never_shrinks_leaves_a_knot_barely_one_unit_wide(
        self,
    ) -> None:
        row = self.outcomes()["a reach wider than the grid, held there"]

        assert row["figure"] == pytest.approx(1.1586, abs=0.00005)

    def test_a_flat_column_is_kept_exactly(self) -> None:
        row = self.outcomes()["a column that never changes"]

        assert row["figure"] == 0.0

    def test_columns_are_matched_by_name_rather_than_by_position(self) -> None:
        row = self.outcomes()["a query with the fitted columns in the other order"]

        assert row["figure"] == 0.0

    def test_people_already_on_their_cells_stop_the_walk_at_once(self) -> None:
        row = self.outcomes()["people who already sit on their cells"]

        assert row["figure"] == 0.0
        assert "1 epoch of the 100" in row["detail"]

    @pytest.mark.parametrize(
        ("edge", "fragment"),
        [
            ("a value that is not a number", "finite"),
            ("an unfitted map asked to answer", "must be fit"),
            ("a query carrying a column the fit never saw", "expected exactly"),
            ("a query missing one of the fitted columns", "expected exactly"),
            ("a query carrying one name twice", "must be unique"),
            ("a walk of no epochs", "greater than or equal to 1"),
        ],
    )
    def test_each_refusal_says_what_was_wrong(self, edge: str, fragment: str) -> None:
        assert fragment in self.outcomes()[edge]["detail"]

    def test_the_edges_answer_the_same_way_twice(self) -> None:
        assert self.outcomes() == self.outcomes()

    def test_no_sentence_names_anything_from_the_code(self) -> None:
        """The website prints these sentences, so no identifier may survive.

        Two of the refusals arrive naming a parameter and a class, and they
        are rewritten on the way out, since a reader learning what a map is
        has no use for either. This is what stops one growing back.
        """
        forbidden = (
            "feature_values",
            "SelfOrganisingMap",
            "KMeans",
            "Error",
            "oop_ml",
            "self.",
            "__",
        )
        for edge, row in self.outcomes().items():
            for identifier in forbidden:
                assert identifier not in row["detail"], f"{edge}: {row['detail']}"
