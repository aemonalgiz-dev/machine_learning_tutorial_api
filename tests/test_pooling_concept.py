"""The pooling endpoints, pinned to the page's hand arithmetic.

The page works a four by four picture under a window of two and a stride of
two, chosen so the four windows sum to 12, 8, 4 and 16 and each holds one
largest value. A maximum answers [[5, 4], [2, 7]] and an average answers
[[3, 2], [1, 4]]. With a slope of one arriving at every answer, the maximum
hands the whole of it to the four winners and nothing to the other twelve
cells, and the average hands every cell a quarter. At a stride of one the
windows overlap and the library adds, so the cell holding 4 wins three windows
and shows 3, and an interior cell under an average shows four quarters. A
bright column nudged one cell sideways inside its windows leaves the pooled
map untouched, and nudged across a boundary moves it one cell along. The
shares of a single window sum to one only to within float64, side 7 comes to
0.9999999999999999, sixteen of the first thirty-two sides miss, and the worst
is side 31. All pinned.
"""

from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

APPLY = "/concepts/pooling/apply"
SHARE_TOTALS = "/concepts/pooling/share-totals"

WORKED_PICTURE = [
    [1, 5, 1, 3],
    [4, 2, 4, 0],
    [0, 1, 3, 1],
    [2, 1, 7, 5],
]

STROKE_SIDE = 8
STROKE_VALUE = 9.0


def pooled_by(
    kind: str,
    picture: list[list[float]] = WORKED_PICTURE,
    window: int = 2,
    stride: int = 2,
) -> dict:
    response = client.post(
        APPLY,
        json={"picture": picture, "window": window, "stride": stride, "kind": kind},
    )
    assert response.status_code == 200
    return response.json()


def share_totals_of(kind: str, largest_side: int) -> dict:
    response = client.post(
        SHARE_TOTALS, json={"kind": kind, "largest_side": largest_side}
    )
    assert response.status_code == 200
    return response.json()


def vertical_stroke(column: int) -> list[list[float]]:
    """An eight by eight picture lit along one column and dark elsewhere."""
    return [
        [STROKE_VALUE if each == column else 0.0 for each in range(STROKE_SIDE)]
        for _ in range(STROKE_SIDE)
    ]


class TestTheWorkedExampleUnderMax:
    def test_the_pooled_map_keeps_each_window_s_largest(self) -> None:
        body = pooled_by("max")

        assert body["pooled"] == [[5.0, 4.0], [2.0, 7.0]]
        assert (body["pooled_height"], body["pooled_width"]) == (2, 2)

    def test_the_blame_is_one_at_the_four_winners_and_zero_elsewhere(self) -> None:
        body = pooled_by("max")

        assert body["blame"] == [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
        ]

    def test_four_of_sixteen_cells_receive_any_correction(self) -> None:
        body = pooled_by("max")

        assert body["n_corrected"] == 4
        assert body["height"] * body["width"] == 16
        assert sum(sum(row) for row in body["corrected"]) == 4

    def test_each_window_names_the_cell_that_won_it(self) -> None:
        body = pooled_by("max")

        winners = [
            (entry["winner"]["row"], entry["winner"]["column"])
            for entry in body["windows"]
        ]
        assert winners == [(0, 1), (1, 2), (3, 0), (3, 2)]

    def test_the_windows_sit_at_the_output_position_times_the_stride(self) -> None:
        body = pooled_by("max")

        corners = [(entry["top"], entry["left"]) for entry in body["windows"]]
        assert corners == [(0, 0), (0, 2), (2, 0), (2, 2)]
        assert [entry["answer"] for entry in body["windows"]] == [5.0, 4.0, 2.0, 7.0]

    def test_every_window_s_shares_total_exactly_one(self) -> None:
        body = pooled_by("max")

        assert [entry["share_total"] for entry in body["windows"]] == [1.0] * 4


class TestTheWorkedExampleUnderAverage:
    def test_the_pooled_map_is_each_window_s_mean(self) -> None:
        body = pooled_by("average")

        assert body["pooled"] == [[3.0, 2.0], [1.0, 4.0]]

    def test_the_four_windows_sum_to_twelve_eight_four_and_sixteen(self) -> None:
        body = pooled_by("average")

        sums = [4 * entry["answer"] for entry in body["windows"]]
        assert sums == [12.0, 8.0, 4.0, 16.0]

    def test_every_cell_receives_a_quarter(self) -> None:
        body = pooled_by("average")

        assert body["blame"] == [[0.25] * 4 for _ in range(4)]

    def test_all_sixteen_cells_receive_correction(self) -> None:
        body = pooled_by("average")

        assert body["n_corrected"] == 16
        assert all(all(row) for row in body["corrected"])

    def test_an_average_names_no_winner(self) -> None:
        body = pooled_by("average")

        assert [entry["winner"] for entry in body["windows"]] == [None] * 4

    def test_a_window_of_two_s_shares_total_exactly_one(self) -> None:
        body = pooled_by("average")

        assert [entry["share_total"] for entry in body["windows"]] == [1.0] * 4


class TestTheGeometry:
    """The pooled extent is (extent - window) // stride + 1 on each side."""

    def test_a_stride_of_two_pools_four_to_two(self) -> None:
        body = pooled_by("max", window=2, stride=2)

        assert (body["pooled_height"], body["pooled_width"]) == (2, 2)
        assert len(body["windows"]) == 4

    def test_a_stride_of_one_pools_four_to_three(self) -> None:
        body = pooled_by("max", window=2, stride=1)

        assert (body["pooled_height"], body["pooled_width"]) == (3, 3)
        assert len(body["windows"]) == 9

    def test_a_window_the_size_of_the_picture_is_one_answer(self) -> None:
        body = pooled_by("max", window=4, stride=4)

        assert body["pooled"] == [[7.0]]
        assert body["blame"] == [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]

    def test_a_window_that_does_not_divide_the_side_stops_early(self) -> None:
        picture = [[float(row * 5 + column) for column in range(5)] for row in range(5)]
        body = pooled_by("max", picture=picture, window=2, stride=2)

        assert (body["pooled_height"], body["pooled_width"]) == (2, 2)
        assert body["pooled"] == [[6.0, 8.0], [16.0, 18.0]]
        # The fifth row and column belong to no window and receive nothing.
        assert body["blame"][4] == [0.0] * 5
        assert [row[4] for row in body["blame"]] == [0.0] * 5

    def test_a_rectangle_is_pooled_as_a_rectangle(self) -> None:
        body = pooled_by("average", picture=[[1, 2, 3, 4], [5, 6, 7, 8]])

        assert (body["height"], body["width"]) == (2, 4)
        assert body["pooled"] == [[3.5, 5.5]]

    def test_the_answer_echoes_the_window_and_stride_it_used(self) -> None:
        body = pooled_by("average", window=3, stride=1)

        assert (body["window"], body["stride"]) == (3, 1)
        assert body["kind"] == "average"


class TestOverlappingWindows:
    """At a stride of one the windows overlap and the blame accumulates."""

    def test_the_nine_maxima(self) -> None:
        body = pooled_by("max", stride=1)

        assert body["pooled"] == [[5.0, 5.0, 4.0], [4.0, 4.0, 4.0], [2.0, 7.0, 7.0]]

    def test_the_cell_holding_four_wins_three_windows_and_shows_three(self) -> None:
        body = pooled_by("max", stride=1)

        assert body["blame"] == [
            [0.0, 2.0, 0.0, 0.0],
            [1.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 2.0, 0.0],
        ]
        assert body["blame"][1][2] == 3.0
        assert body["n_corrected"] == 5

    def test_an_average_hands_an_interior_cell_four_quarters(self) -> None:
        body = pooled_by("average", stride=1)

        assert body["pooled"] == [
            [3.0, 3.0, 2.0],
            [1.75, 2.5, 2.0],
            [1.0, 3.0, 4.0],
        ]
        assert body["blame"] == [
            [0.25, 0.5, 0.5, 0.25],
            [0.5, 1.0, 1.0, 0.5],
            [0.5, 1.0, 1.0, 0.5],
            [0.25, 0.5, 0.5, 0.25],
        ]
        assert body["n_corrected"] == 16

    def test_the_total_blame_equals_the_number_of_windows(self) -> None:
        # The shares inside every window sum to one, so a slope of one per
        # window hands down exactly one per window, whatever the stride and
        # whichever kind.
        for kind in ("max", "average"):
            for stride in (1, 2):
                body = pooled_by(kind, stride=stride)
                handed_down = sum(sum(row) for row in body["blame"])
                assert handed_down == pytest.approx(len(body["windows"]))


class TestTheTieRule:
    def test_a_tied_maximum_goes_to_the_first_in_row_major_order(self) -> None:
        body = pooled_by("max", picture=[[3, 3], [1, 3]], window=2, stride=2)

        assert body["pooled"] == [[3.0]]
        assert body["blame"] == [[1.0, 0.0], [0.0, 0.0]]
        assert body["windows"][0]["winner"] == {"row": 0, "column": 0}

    def test_a_flat_patch_trains_only_the_top_left_of_each_window(self) -> None:
        body = pooled_by("max", picture=[[5] * 4 for _ in range(4)])

        assert body["pooled"] == [[5.0, 5.0], [5.0, 5.0]]
        assert body["blame"] == [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        assert body["n_corrected"] == 4


class TestAShiftInsideTheWindow:
    """A bright column nudged one cell along, under a window of two at two."""

    def test_the_stroke_pools_to_one_bright_column(self) -> None:
        body = pooled_by("max", picture=vertical_stroke(2))

        assert body["pooled"] == [[0.0, 9.0, 0.0, 0.0]] * 4

    def test_a_nudge_that_stays_inside_the_windows_changes_nothing(self) -> None:
        for kind in ("max", "average"):
            before = pooled_by(kind, picture=vertical_stroke(2))
            after = pooled_by(kind, picture=vertical_stroke(3))

            assert after["pooled"] == before["pooled"]

    def test_an_average_halves_the_stroke_and_keeps_it_still(self) -> None:
        before = pooled_by("average", picture=vertical_stroke(2))
        after = pooled_by("average", picture=vertical_stroke(3))

        assert before["pooled"] == [[0.0, 4.5, 0.0, 0.0]] * 4
        assert after["pooled"] == before["pooled"]

    def test_a_nudge_across_a_boundary_moves_the_answer_one_cell(self) -> None:
        body = pooled_by("max", picture=vertical_stroke(4))

        assert body["pooled"] == [[0.0, 0.0, 9.0, 0.0]] * 4


class TestShareTotals:
    def test_side_seven_comes_to_one_less_a_bit(self) -> None:
        body = share_totals_of("average", 8)

        seven = body["totals"][6]
        assert seven["side"] == 7
        assert seven["positions"] == 49
        assert seven["total"] == 0.9999999999999999
        assert seven["exact"] is False

    def test_the_widget_s_seven_by_seven_window_reads_the_same_total(self) -> None:
        body = pooled_by(
            "average", picture=[[0.0] * 7 for _ in range(7)], window=7, stride=7
        )

        assert len(body["windows"]) == 1
        assert body["windows"][0]["share_total"] == 0.9999999999999999
        assert body["n_corrected"] == 49

    def test_seven_is_the_only_inexact_side_of_the_first_eight(self) -> None:
        body = share_totals_of("average", 8)

        assert [entry["exact"] for entry in body["totals"]] == [
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
        ]
        assert body["n_inexact"] == 1
        assert body["worst_side"] == 7

    def test_sixteen_of_the_first_thirty_two_sides_miss(self) -> None:
        body = share_totals_of("average", 32)

        assert len(body["totals"]) == 32
        assert body["n_inexact"] == 16
        inexact = [entry["side"] for entry in body["totals"] if not entry["exact"]]
        assert inexact == [
            7,
            10,
            12,
            13,
            14,
            17,
            19,
            20,
            21,
            23,
            24,
            25,
            26,
            28,
            29,
            31,
        ]

    def test_the_worst_is_side_thirty_one_at_four_point_four_e_minus_sixteen(
        self,
    ) -> None:
        body = share_totals_of("average", 32)

        assert body["worst_side"] == 31
        assert body["worst_gap"] == 2.0**-51
        assert body["worst_gap"] == pytest.approx(4.4e-16, abs=0.05e-16)
        assert body["totals"][30]["total"] == 1.0000000000000004

    def test_no_side_misses_by_more_than_the_worst(self) -> None:
        body = share_totals_of("average", 32)

        for entry in body["totals"]:
            assert abs(entry["total"] - 1.0) <= 2.0**-51

    def test_a_maximum_s_shares_total_exactly_one_at_every_side(self) -> None:
        body = share_totals_of("max", 32)

        assert all(entry["total"] == 1.0 for entry in body["totals"])
        assert body["n_inexact"] == 0
        assert (body["worst_side"], body["worst_gap"]) == (1, 0.0)


class TestRefusals:
    RAGGED: ClassVar[list[list[int]]] = [[1, 2, 3], [4, 5], [6, 7, 8]]

    def test_a_window_larger_than_the_picture_is_refused_by_the_library(self) -> None:
        response = client.post(
            APPLY,
            json={"picture": WORKED_PICTURE, "window": 5, "stride": 1, "kind": "max"},
        )

        assert response.status_code == 422
        assert response.json()["error"] == "ShapeMismatchError"
        assert "does not fit" in response.json()["detail"]

    def test_a_ragged_picture_is_refused_in_words(self) -> None:
        response = client.post(
            APPLY,
            json={"picture": self.RAGGED, "window": 2, "stride": 1, "kind": "max"},
        )

        assert response.status_code == 422
        assert "rectangle" in response.json()["detail"]

    def test_a_window_of_zero_is_refused(self) -> None:
        response = client.post(
            APPLY,
            json={"picture": WORKED_PICTURE, "window": 0, "stride": 1, "kind": "max"},
        )

        assert response.status_code == 422

    def test_a_stride_of_zero_is_refused(self) -> None:
        response = client.post(
            APPLY,
            json={"picture": WORKED_PICTURE, "window": 2, "stride": 0, "kind": "max"},
        )

        assert response.status_code == 422

    def test_a_picture_wider_than_eight_is_refused(self) -> None:
        response = client.post(
            APPLY,
            json={
                "picture": [[0.0] * 9 for _ in range(9)],
                "window": 2,
                "stride": 2,
                "kind": "max",
            },
        )

        assert response.status_code == 422

    def test_an_empty_picture_is_refused(self) -> None:
        response = client.post(
            APPLY, json={"picture": [], "window": 1, "stride": 1, "kind": "max"}
        )

        assert response.status_code == 422

    def test_an_unknown_kind_is_refused(self) -> None:
        response = client.post(
            APPLY,
            json={"picture": WORKED_PICTURE, "window": 2, "stride": 2, "kind": "sum"},
        )

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            APPLY,
            json={
                "picture": WORKED_PICTURE,
                "window": 2,
                "stride": 2,
                "kind": "max",
                "padding": 1,
            },
        )

        assert response.status_code == 422

    def test_a_side_beyond_thirty_two_is_refused(self) -> None:
        response = client.post(
            SHARE_TOTALS, json={"kind": "average", "largest_side": 33}
        )

        assert response.status_code == 422

    def test_a_side_of_zero_is_refused(self) -> None:
        response = client.post(
            SHARE_TOTALS, json={"kind": "average", "largest_side": 0}
        )

        assert response.status_code == 422
