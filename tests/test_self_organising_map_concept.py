"""The self-organising map endpoint, pinned to the page's claims.

The page drapes a one-by-two map over the k-means page's eight people in two
tight clumps and says the two units rest with one in each clump, each within
three of its clump's mean, (121, 26) and (181, 80), and neither on it, because
the neighbourhood pulled the two units toward each other until the radius had
decayed. It says the four short people are won by one unit and the four tall
by the other, that the walk runs its full budget of a hundred epochs, that the
map moved 0.107 on its last epoch, and that the quantisation error is 3.28.
It also says a one-by-one map rests on the grand mean (151, 53), and that
doubling the budget brings the units nearer their means. On the crowd it says
a three-by-three map puts the short clump in one corner and the tall clump in
the opposite one, and it says the walk runs its full budget except when every
person already sits on their unit, when the library stops after one epoch.
The equation block quotes the two schedules, a rate from a half to a
hundredth and a radius from half the longest side to a quarter step. All
pinned.
"""

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.self_organising_map import (
    LEARNING_RATE,
    RADIUS_END,
    _radius_schedule,
)
from oop_ml_api.limits import MAX_POINTS
from oop_ml_api.main import create_app

client = TestClient(create_app())

FIT = "/concepts/self-organising-map/fit"

TWO_GROUPS = [
    {"x": 118, "y": 24},
    {"x": 120, "y": 25},
    {"x": 122, "y": 28},
    {"x": 124, "y": 27},
    {"x": 178, "y": 78},
    {"x": 180, "y": 80},
    {"x": 182, "y": 83},
    {"x": 184, "y": 79},
]

SHORT_MEAN = (121.0, 26.0)
TALL_MEAN = (181.0, 80.0)
GRAND_MEAN = (151.0, 53.0)

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


def organised(points: list[dict], width: int, height: int, epochs: int) -> dict:
    response = client.post(
        FIT,
        json={
            "points": points,
            "grid_width": width,
            "grid_height": height,
            "max_epochs": epochs,
        },
    )
    assert response.status_code == 200
    return response.json()


def nearest_unit(units: list[dict], mean: tuple[float, float]) -> dict:
    return min(
        units, key=lambda unit: abs(unit["x"] - mean[0]) + abs(unit["y"] - mean[1])
    )


class TestTheWorkedExample:
    """A one-by-two map over the eight people, for a hundred epochs."""

    def test_one_unit_rests_in_each_clump(self) -> None:
        body = organised(TWO_GROUPS, 2, 1, 100)

        short_unit = nearest_unit(body["units"], SHORT_MEAN)
        tall_unit = nearest_unit(body["units"], TALL_MEAN)
        assert short_unit is not tall_unit
        assert abs(short_unit["x"] - SHORT_MEAN[0]) < 3.0
        assert abs(short_unit["y"] - SHORT_MEAN[1]) < 3.0
        assert abs(tall_unit["x"] - TALL_MEAN[0]) < 3.0
        assert abs(tall_unit["y"] - TALL_MEAN[1]) < 3.0

    def test_each_unit_rests_near_its_mean_and_not_on_it(self) -> None:
        # The neighbourhood pulled each unit toward the other clump for as long
        # as the radius lasted, so each rests on the side of its mean that
        # faces the other, and more than one away from it.
        body = organised(TWO_GROUPS, 2, 1, 100)

        short_unit = nearest_unit(body["units"], SHORT_MEAN)
        tall_unit = nearest_unit(body["units"], TALL_MEAN)
        assert 1.0 < short_unit["x"] - SHORT_MEAN[0]
        assert 1.0 < short_unit["y"] - SHORT_MEAN[1]
        assert 1.0 < TALL_MEAN[0] - tall_unit["x"]
        assert 1.0 < TALL_MEAN[1] - tall_unit["y"]

    def test_the_resting_units_to_one_decimal(self) -> None:
        body = organised(TWO_GROUPS, 2, 1, 100)

        short_unit = nearest_unit(body["units"], SHORT_MEAN)
        tall_unit = nearest_unit(body["units"], TALL_MEAN)
        assert short_unit["x"] == pytest.approx(123.0, abs=0.05)
        assert short_unit["y"] == pytest.approx(27.8, abs=0.05)
        assert tall_unit["x"] == pytest.approx(179.1, abs=0.05)
        assert tall_unit["y"] == pytest.approx(78.2, abs=0.05)

    def test_the_short_people_share_one_unit_and_the_tall_the_other(self) -> None:
        body = organised(TWO_GROUPS, 2, 1, 100)

        winners = [(winner["row"], winner["column"]) for winner in body["winners"]]
        assert len(set(winners[:4])) == 1
        assert len(set(winners[4:])) == 1
        assert winners[0] != winners[4]

    def test_the_short_people_are_won_by_the_unit_in_their_clump(self) -> None:
        body = organised(TWO_GROUPS, 2, 1, 100)

        short_unit = nearest_unit(body["units"], SHORT_MEAN)
        first_winner = body["winners"][0]
        assert (first_winner["row"], first_winner["column"]) == (
            short_unit["row"],
            short_unit["column"],
        )

    def test_the_walk_runs_its_full_budget(self) -> None:
        body = organised(TWO_GROUPS, 2, 1, 100)

        assert body["epochs_run"] == 100

    def test_the_final_movement_is_a_tenth(self) -> None:
        body = organised(TWO_GROUPS, 2, 1, 100)

        assert body["final_movement"] == pytest.approx(0.107, abs=0.0005)

    def test_the_quantisation_error_is_three_point_two_eight(self) -> None:
        body = organised(TWO_GROUPS, 2, 1, 100)

        assert body["quantisation_error"] == pytest.approx(3.28, abs=0.005)

    def test_a_longer_walk_rests_nearer_the_means(self) -> None:
        # Both schedules are fractions of the budget, so doubling it doubles
        # the epochs spent with the neighbourhood gone, and each unit has twice
        # as many small steps to drift onto its own clump.
        hundred = organised(TWO_GROUPS, 2, 1, 100)
        two_hundred = organised(TWO_GROUPS, 2, 1, 200)

        for mean in (SHORT_MEAN, TALL_MEAN):
            before = nearest_unit(hundred["units"], mean)
            after = nearest_unit(two_hundred["units"], mean)
            assert abs(after["x"] - mean[0]) < abs(before["x"] - mean[0])
            assert abs(after["y"] - mean[1]) < abs(before["y"] - mean[1])
            assert abs(after["x"] - mean[0]) < 1.0
            assert abs(after["y"] - mean[1]) < 1.0

    def test_a_single_unit_rests_on_the_grand_mean(self) -> None:
        # With no neighbour to pull it anywhere, a lone unit is an online
        # k-means centre and settles on the mean of everyone.
        body = organised(TWO_GROUPS, 1, 1, 100)

        assert len(body["units"]) == 1
        assert body["units"][0]["x"] == pytest.approx(GRAND_MEAN[0], abs=0.5)
        assert body["units"][0]["y"] == pytest.approx(GRAND_MEAN[1], abs=0.5)
        assert all(winner == {"row": 0, "column": 0} for winner in body["winners"])


class TestTheGrid:
    def test_the_units_come_back_in_row_major_order(self) -> None:
        body = organised(CROWD, 3, 2, 50)

        assert body["grid_width"] == 3
        assert body["grid_height"] == 2
        assert [(unit["row"], unit["column"]) for unit in body["units"]] == [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
        ]

    def test_every_person_is_won_by_a_unit_on_the_grid(self) -> None:
        body = organised(CROWD, 3, 3, 100)

        assert len(body["winners"]) == len(CROWD)
        for winner in body["winners"]:
            assert 0 <= winner["row"] < 3
            assert 0 <= winner["column"] < 3

    def test_the_quantisation_error_is_the_mean_distance_to_the_winner(self) -> None:
        body = organised(CROWD, 2, 2, 100)

        by_position = {(unit["row"], unit["column"]): unit for unit in body["units"]}
        total = 0.0
        for point, winner in zip(CROWD, body["winners"], strict=True):
            unit = by_position[(winner["row"], winner["column"])]
            total += (
                (point["x"] - unit["x"]) ** 2 + (point["y"] - unit["y"]) ** 2
            ) ** 0.5

        assert body["quantisation_error"] == pytest.approx(total / len(CROWD))

    def test_the_epoch_budget_is_what_stops_the_walk(self) -> None:
        body = organised(CROWD, 2, 2, 7)

        assert body["epochs_run"] == 7

    def test_people_already_on_their_units_stop_the_walk_at_once(self) -> None:
        # The library's early stop asks for no unit to have moved further than
        # its tolerance in a whole epoch. A rate ending at a hundredth never
        # produces that on its own, so the one way to see it is people who
        # already sit on the unit that wins them.
        body = organised([{"x": 150, "y": 50}] * 3, 2, 1, 100)

        assert body["epochs_run"] == 1
        assert body["final_movement"] == 0.0

    def test_the_crowd_takes_opposite_corners_on_a_three_by_three_map(self) -> None:
        body = organised(CROWD, 3, 3, 100)

        winners = [(winner["row"], winner["column"]) for winner in body["winners"]]
        short_winners = set(winners[5:8])
        tall_winners = set(winners[8:11])
        assert len(short_winners) == 1
        assert len(tall_winners) == 1
        short_corner = short_winners.pop()
        tall_corner = tall_winners.pop()
        assert short_corner in {(0, 0), (0, 2), (2, 0), (2, 2)}
        assert tall_corner == (2 - short_corner[0], 2 - short_corner[1])


class TestTheSchedules:
    """The two curves the page's equation block quotes are the module's constants."""

    def test_the_rate_falls_from_a_half_to_a_hundredth(self) -> None:
        assert LEARNING_RATE.start == 0.5
        assert LEARNING_RATE.end == 0.01

    @pytest.mark.parametrize(
        ("width", "height", "start"), [(2, 1, 1.0), (1, 1, 0.5), (3, 6, 3.0)]
    )
    def test_the_radius_falls_from_half_the_longest_side_to_a_quarter_step(
        self, width: int, height: int, start: float
    ) -> None:
        schedule = _radius_schedule(width, height)

        assert schedule.start == start
        assert schedule.end == RADIUS_END == 0.25


class TestRefusals:
    def test_a_lone_person_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={
                "points": TWO_GROUPS[:1],
                "grid_width": 1,
                "grid_height": 1,
                "max_epochs": 10,
            },
        )

        assert response.status_code == 422

    def test_too_many_people_are_refused_before_computing(self) -> None:
        many = [
            {"x": float(person_number), "y": float(person_number)}
            for person_number in range(MAX_POINTS + 1)
        ]

        response = client.post(
            FIT,
            json={
                "points": many,
                "grid_width": 2,
                "grid_height": 2,
                "max_epochs": 10,
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("side", [0, 7])
    def test_a_grid_side_outside_one_to_six_is_refused(self, side: int) -> None:
        response = client.post(
            FIT,
            json={
                "points": TWO_GROUPS,
                "grid_width": side,
                "grid_height": 1,
                "max_epochs": 10,
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("epochs", [0, 201])
    def test_an_epoch_budget_outside_one_to_two_hundred_is_refused(
        self, epochs: int
    ) -> None:
        response = client.post(
            FIT,
            json={
                "points": TWO_GROUPS,
                "grid_width": 2,
                "grid_height": 1,
                "max_epochs": epochs,
            },
        )

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={
                "points": TWO_GROUPS,
                "grid_width": 2,
                "grid_height": 1,
                "max_epochs": 10,
                "radius": 1.0,
            },
        )

        assert response.status_code == 422
