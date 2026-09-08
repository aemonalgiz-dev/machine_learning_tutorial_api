"""The pooling page's arrangement, window, shift and edge-case endpoints.

The first file pins what the layer answers and what it hands back down. This
one pins the four things the page adds around that.

The arrangement is settled before any data arrives, so the geometry endpoint
builds a layer and reads its shape: an eight by eight picture under a window of
two at a stride of two answers four by four, and under a stride of three
answers three by three while leaving rows 2 and 5 outside every window. The
convolution page's own map, eight filters of 26 by 26, holds 5408 numbers and
comes out of a window of two at a stride of two holding 1352, a quarter of
them.

One window is enough to show what the two kinds differ by. The window 1, 5, 4,
2 answers 5 under a maximum, which owes the whole of it to one position, and 3
under an average, which owes a quarter to each.

A bright column nudged one cell inside its windows changes 16 of the picture's
64 cells and none of the pooled map's 16; nudged across a boundary it changes 8
of them, under both kinds.

And the edge cases are probed rather than described. Thirteen of the sixteen
are refused, every one of them in the library's own words, including the bare
integer where extents are wanted. Three are accepted, and the fourth thing that
is accepted has a section of the page to itself: two layers of one shape built
from different windows and strides take each other's responses and hand down
the same total of 10, and six of the sixteen cells receive something different.
"""

from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

GEOMETRY = "/concepts/pooling/geometry"
WINDOW = "/concepts/pooling/window"
SHIFT = "/concepts/pooling/shift"
EDGE_CASES = "/concepts/pooling/edge-cases"

PATCH = [
    [1, 5, 1, 3],
    [4, 2, 4, 0],
    [0, 1, 3, 1],
    [2, 1, 7, 5],
]

STROKE_SIDE = 8
STROKE_VALUE = 9.0


def geometry_of(
    height: int,
    width: int,
    windows: list[int],
    strides: list[int],
    channels: int = 1,
) -> dict:
    response = client.post(
        GEOMETRY,
        json={
            "channels": channels,
            "height": height,
            "width": width,
            "windows": windows,
            "strides": strides,
        },
    )
    assert response.status_code == 200
    return response.json()


def only_row(body: dict) -> dict:
    assert len(body["rows"]) == 1
    return body["rows"][0]


def window_of(values: list[list[float]]) -> dict:
    response = client.post(WINDOW, json={"window": values})
    assert response.status_code == 200
    return response.json()


def summary_for(body: dict, kind: str) -> dict:
    return next(entry for entry in body["summaries"] if entry["kind"] == kind)


def vertical_stroke(column: int) -> list[list[float]]:
    """An eight by eight picture lit along one column and dark elsewhere."""
    return [
        [STROKE_VALUE if each == column else 0.0 for each in range(STROKE_SIDE)]
        for _ in range(STROKE_SIDE)
    ]


def edge_cases() -> dict:
    response = client.get(EDGE_CASES)
    assert response.status_code == 200
    return response.json()


def case_named(body: dict, name: str) -> dict:
    return next(entry for entry in body["cases"] if entry["name"] == name)


class TestTheArrangementBeforeAnyData:
    """Every extent is known at construction, so the endpoint runs nothing."""

    def test_eight_by_eight_at_two_and_two_answers_four_by_four(self) -> None:
        row = only_row(geometry_of(8, 8, [2], [2]))

        assert row["reads"] == [1, 8, 8]
        assert row["answers"] == [1, 4, 4]
        assert (row["n_inputs"], row["n_outputs"]) == (64, 16)
        assert row["kept_share"] == 0.25
        assert row["values_dropped"] == 48

    def test_a_window_of_one_at_a_stride_of_one_changes_nothing(self) -> None:
        row = only_row(geometry_of(8, 8, [1], [1]))

        assert row["answers"] == row["reads"]
        assert row["n_outputs"] == row["n_inputs"] == 64
        assert row["n_unvisited_cells"] == 0

    def test_a_stride_below_the_window_overlaps_and_shrinks_less(self) -> None:
        row = only_row(geometry_of(8, 8, [2], [1]))

        assert row["overlaps"] is True
        assert row["answers"] == [1, 7, 7]
        assert row["n_outputs"] == 49

    @pytest.mark.parametrize(
        ("window", "stride", "answers", "unvisited", "cells"),
        [
            (2, 2, [1, 4, 4], [], 0),
            (2, 3, [1, 3, 3], [2, 5], 28),
            (3, 2, [1, 3, 3], [7], 15),
            (3, 3, [1, 2, 2], [6, 7], 28),
            (1, 2, [1, 4, 4], [1, 3, 5, 7], 48),
        ],
    )
    def test_the_positions_a_sweep_never_reaches(
        self,
        window: int,
        stride: int,
        answers: list[int],
        unvisited: list[int],
        cells: int,
    ) -> None:
        row = only_row(geometry_of(8, 8, [window], [stride]))

        assert row["answers"] == answers
        assert row["unvisited_rows"] == unvisited
        assert row["unvisited_columns"] == unvisited
        assert row["n_unvisited_cells"] == cells

    def test_a_five_by_five_leaves_its_last_row_and_column_out(self) -> None:
        row = only_row(geometry_of(5, 5, [2], [2]))

        assert row["answers"] == [1, 2, 2]
        assert row["unvisited_rows"] == [4]
        assert row["unvisited_columns"] == [4]
        assert row["n_unvisited_cells"] == 9

    def test_the_channel_count_passes_straight_through(self) -> None:
        row = only_row(geometry_of(26, 26, [2], [2], channels=8))

        assert row["reads"] == [8, 26, 26]
        assert row["answers"] == [8, 13, 13]

    def test_the_convolution_page_s_map_comes_out_a_quarter_the_size(self) -> None:
        row = only_row(geometry_of(26, 26, [2], [2], channels=8))

        assert row["n_inputs"] == 5408
        assert row["n_outputs"] == 1352
        assert row["values_dropped"] == 4056
        assert row["kept_share"] == 0.25

    def test_the_same_map_under_a_window_of_three_at_a_stride_of_two(self) -> None:
        row = only_row(geometry_of(26, 26, [3], [2], channels=8))

        assert row["answers"] == [8, 12, 12]
        assert row["n_outputs"] == 1152
        assert row["unvisited_rows"] == [25]
        assert row["n_unvisited_cells"] == 51

    def test_both_kinds_agree_on_every_arrangement(self) -> None:
        body = geometry_of(8, 8, [1, 2, 3, 4], [1, 2, 3, 4])

        assert len(body["rows"]) == 16
        assert all(row["kinds_agree"] for row in body["rows"])

    def test_a_window_that_does_not_fit_is_a_refused_row_not_a_refused_sweep(
        self,
    ) -> None:
        body = geometry_of(4, 4, [2, 5], [1])

        first, second = body["rows"]
        assert first["refused"] is False
        assert second["refused"] is True
        assert second["detail"] == "a window of 5 does not fit in a picture 4 by 4"
        assert second["answers"] is None


class TestOneWindowUnderBothKinds:
    """The window 1, 5, 4, 2, which is the patch's top left."""

    TOP_LEFT: ClassVar[list[list[float]]] = [[1, 5], [4, 2]]

    def test_the_window_is_echoed_with_its_total(self) -> None:
        body = window_of(self.TOP_LEFT)

        assert body["side"] == 2
        assert body["positions"] == 4
        assert body["total"] == 12.0

    def test_a_maximum_keeps_five_and_owes_it_to_one_position(self) -> None:
        summary = summary_for(window_of(self.TOP_LEFT), "max")

        assert summary["answer"] == 5.0
        assert summary["shares"] == [[0.0, 1.0], [0.0, 0.0]]
        assert summary["winner"] == {"row": 0, "column": 1}
        assert summary["n_receiving"] == 1

    def test_an_average_keeps_three_and_owes_a_quarter_to_each(self) -> None:
        summary = summary_for(window_of(self.TOP_LEFT), "average")

        assert summary["answer"] == 3.0
        assert summary["shares"] == [[0.25, 0.25], [0.25, 0.25]]
        assert summary["winner"] is None
        assert summary["n_receiving"] == 4

    def test_both_kinds_shares_total_one(self) -> None:
        body = window_of(self.TOP_LEFT)

        assert [entry["share_total"] for entry in body["summaries"]] == [1.0, 1.0]

    def test_a_window_of_three_hands_out_ninths(self) -> None:
        body = window_of([[1, 5, 1], [4, 2, 4], [0, 1, 3]])
        average = summary_for(body, "average")

        assert body["positions"] == 9
        assert average["answer"] == pytest.approx(21 / 9)
        assert average["shares"][0][0] == pytest.approx(1 / 9)
        assert average["share_total"] == 1.0
        assert average["n_receiving"] == 9

    def test_a_tied_window_names_the_first_position_in_row_major_order(self) -> None:
        summary = summary_for(window_of([[3, 3], [1, 3]]), "max")

        assert summary["answer"] == 3.0
        assert summary["winner"] == {"row": 0, "column": 0}
        assert summary["n_receiving"] == 1

    def test_a_window_of_one_gives_that_one_position_everything(self) -> None:
        body = window_of([[4]])

        assert summary_for(body, "max")["shares"] == [[1.0]]
        assert summary_for(body, "average")["shares"] == [[1.0]]

    def test_a_maximum_separates_two_windows_a_mean_cannot(self) -> None:
        # One strong reading against four moderate ones. Both average 2, which
        # is the page's argument that a mean discards the peak, and their
        # maxima are 8 and 2, which is its argument that a maximum keeps it.
        strong = window_of([[0, 0], [0, 8]])
        moderate = window_of([[2, 2], [2, 2]])

        assert summary_for(strong, "average")["answer"] == 2.0
        assert summary_for(moderate, "average")["answer"] == 2.0
        assert summary_for(strong, "max")["answer"] == 8.0
        assert summary_for(moderate, "max")["answer"] == 2.0

    def test_a_window_that_is_not_square_is_refused_in_words(self) -> None:
        response = client.post(WINDOW, json={"window": [[1, 2, 3], [4, 5, 6]]})

        assert response.status_code == 422
        assert "square" in response.json()["detail"]

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(WINDOW, json={"window": [[1]], "kind": "max"})

        assert response.status_code == 422


class TestWhatASidewaysNudgeDoes:
    """A bright column moved right, pooled at a window of two and a stride of two."""

    def request_body(self, shifts: list[int]) -> dict:
        response = client.post(
            SHIFT,
            json={
                "picture": vertical_stroke(2),
                "window": 2,
                "stride": 2,
                "shifts": shifts,
            },
        )
        assert response.status_code == 200
        return response.json()

    def maps_for(self, body: dict, kind: str) -> list[dict]:
        return next(entry for entry in body["by_kind"] if entry["kind"] == kind)["maps"]

    def test_sixty_four_cells_pool_to_sixteen(self) -> None:
        body = self.request_body([0])

        assert body["n_picture_cells"] == 64
        assert body["n_pooled_cells"] == 16
        assert (body["pooled_height"], body["pooled_width"]) == (4, 4)

    def test_a_nudge_inside_the_windows_moves_the_picture_and_not_the_map(self) -> None:
        for kind in ("max", "average"):
            inside = self.maps_for(self.request_body([0, 1]), kind)[1]

            assert inside["shift"] == 1
            assert inside["picture_cells_changed"] == 16
            assert inside["pooled_cells_changed"] == 0
            assert inside["largest_pooled_change"] == 0.0

    def test_a_nudge_across_a_boundary_moves_eight_of_the_sixteen(self) -> None:
        for kind in ("max", "average"):
            across = self.maps_for(self.request_body([0, 2]), kind)[1]

            assert across["shift"] == 2
            assert across["picture_cells_changed"] == 16
            assert across["pooled_cells_changed"] == 8

    def test_the_maximum_carries_the_stroke_and_the_average_halves_it(self) -> None:
        body = self.request_body([0])

        assert self.maps_for(body, "max")[0]["pooled"][0] == [0.0, 9.0, 0.0, 0.0]
        assert self.maps_for(body, "average")[0]["pooled"][0] == [0.0, 4.5, 0.0, 0.0]

    def test_the_bright_column_of_the_map_moves_one_cell_along(self) -> None:
        across = self.maps_for(self.request_body([2]), "max")[0]

        assert across["pooled"][0] == [0.0, 0.0, 9.0, 0.0]
        assert across["largest_pooled_change"] == 9.0

    def test_more_shifts_than_the_endpoint_allows_are_refused(self) -> None:
        response = client.post(
            SHIFT,
            json={
                "picture": PATCH,
                "window": 2,
                "stride": 2,
                "shifts": list(range(9)),
            },
        )

        assert response.status_code == 422


class TestTheEdgesProbed:
    def test_thirteen_are_refused_and_three_accepted(self) -> None:
        body = edge_cases()

        assert len(body["cases"]) == 16
        assert (body["n_refused"], body["n_accepted"]) == (13, 3)

    def test_every_refusal_is_one_of_the_library_s_own(self) -> None:
        body = edge_cases()

        assert all(case["library_error"] for case in body["cases"] if case["raised"])

    @pytest.mark.parametrize(
        ("name", "error", "fragment"),
        [
            ("a window larger than the picture", "ShapeMismatchError", "does not fit"),
            (
                "a bare integer where extents are wanted",
                "ShapeMismatchError",
                "not a sequence of extents",
            ),
            ("two extents rather than three", "ShapeMismatchError", "2 extents"),
            ("a window of zero", "InvalidValuesError", "at least 1 wide"),
            ("a stride of zero", "InvalidValuesError", "at least 1"),
            ("an extent of zero", "InvalidValuesError", "height must be at least 1"),
            (
                "a flag where a whole number is wanted",
                "InvalidValuesError",
                "not a bool",
            ),
            ("a block holding no rows", "EmptyValuesError", "at least one row"),
            ("a picture with a missing value", "InvalidValuesError", "finite"),
            (
                "a block of the wrong arrangement",
                "ShapeMismatchError",
                "got a block arranged (1, 5, 5)",
            ),
            (
                "an arriving block of the wrong arrangement",
                "ShapeMismatchError",
                "answers with (1, 2, 2)",
            ),
            (
                "an arriving block with the wrong number of rows",
                "ShapeMismatchError",
                "holds 1 rows and the arriving block holds 3",
            ),
            (
                "a response from a layer of another shape",
                "ShapeMismatchError",
                "came from one reading (1, 6, 6)",
            ),
        ],
    )
    def test_each_refusal_names_itself(
        self, name: str, error: str, fragment: str
    ) -> None:
        case = case_named(edge_cases(), name)

        assert case["raised"] is True
        assert case["error"] == error
        assert fragment in case["detail"]

    def test_a_bare_integer_is_refused_in_the_library_s_own_words(self) -> None:
        case = case_named(edge_cases(), "a bare integer where extents are wanted")

        assert case["library_error"] is True
        assert case["detail"] == (
            "a pooling layer reads (channels, height, width), got 4, which is "
            "not a sequence of extents"
        )

    def test_a_stride_larger_than_the_window_is_accepted(self) -> None:
        case = case_named(edge_cases(), "a stride larger than the window")

        assert case["raised"] is False
        assert case["outcome"] == (
            "accepted, answering (1, 1, 1) and leaving 15 of 16 cells outside "
            "every window"
        )

    def test_a_window_of_one_is_accepted_and_changes_nothing(self) -> None:
        case = case_named(edge_cases(), "a window of one")

        assert case["raised"] is False
        assert "answering (1, 4, 4), which is what it read" in case["outcome"]
        assert "exactly 1" in case["outcome"]

    def test_a_step_answers_with_the_same_layer(self) -> None:
        case = case_named(edge_cases(), "a step asked of a layer with nothing to learn")

        assert case["raised"] is False
        assert case["outcome"] == "accepted, and answers with the same layer object"


class TestThePairingTheLibraryDocumentsRatherThanDefends:
    """Two layers of one shape, and a response handed to the wrong one."""

    def test_both_layers_read_and_answer_the_same_arrangement(self) -> None:
        mistake = edge_cases()["pairing_mistake"]

        assert mistake["reads"] == [1, 4, 4]
        assert mistake["answers"] == [1, 2, 2]
        assert (mistake["producer_window"], mistake["producer_stride"]) == (3, 1)
        assert (mistake["receiver_window"], mistake["receiver_stride"]) == (2, 2)

    def test_nothing_is_raised(self) -> None:
        assert edge_cases()["pairing_mistake"]["raised"] is False

    def test_the_totals_agree_and_so_nothing_looks_wrong(self) -> None:
        mistake = edge_cases()["pairing_mistake"]

        assert mistake["producer_total"] == 10.0
        assert mistake["receiver_total"] == 10.0

    def test_three_of_the_four_arriving_values_land_somewhere_else(self) -> None:
        # The fourth, the 4 in the bottom right, happens to reach the same cell
        # either way, since the picture increases towards that corner and both
        # bottom-right windows end on it. Six of the sixteen cells differ.
        mistake = edge_cases()["pairing_mistake"]

        assert mistake["arriving"] == [[1.0, 2.0], [3.0, 4.0]]
        assert mistake["producer_blame"] == [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 3.0, 4.0],
        ]
        assert mistake["receiver_blame"] == [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
        ]
        assert mistake["n_cells_differing"] == 6


class TestTheOverlapAccounting:
    """What overlap changes is where the blame concentrates, not how much of it."""

    def blame_under(self, kind: str, stride: int) -> list[list[float]]:
        response = client.post(
            "/concepts/pooling/apply",
            json={"picture": PATCH, "window": 2, "stride": stride, "kind": kind},
        )
        assert response.status_code == 200
        return response.json()["blame"]

    @pytest.mark.parametrize("kind", ["max", "average"])
    @pytest.mark.parametrize(("stride", "windows"), [(1, 9), (2, 4)])
    def test_the_total_handed_down_is_one_per_window(
        self, kind: str, stride: int, windows: int
    ) -> None:
        blame = self.blame_under(kind, stride)

        assert sum(sum(row) for row in blame) == pytest.approx(float(windows))

    @pytest.mark.parametrize(
        ("kind", "disjoint", "overlapping"),
        [("max", 1.0, 3.0), ("average", 0.25, 1.0)],
    )
    def test_overlap_concentrates_what_the_busiest_cell_receives(
        self, kind: str, disjoint: float, overlapping: float
    ) -> None:
        assert max(max(row) for row in self.blame_under(kind, 2)) == disjoint
        assert max(max(row) for row in self.blame_under(kind, 1)) == overlapping


def every_string_in(value: object) -> list[str]:
    """Every string anywhere inside a decoded body, however deeply nested."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [found for each in value.values() for found in every_string_in(each)]
    if isinstance(value, list):
        return [found for each in value for found in every_string_in(each)]
    return []


FORBIDDEN_WORDS = ("library", "oop_ml", "the api")
"""What a sentence shown to a reader may never name."""


class TestNothingReturnedNamesWhatProducedIt:
    """The website prints these sentences, so none of them may name the code.

    A reader meeting the pooling page has no reason to know what any of it is
    written in, and a sentence that says so is a sentence about the wrong
    thing. Every endpoint here is called and every string in what comes back is
    read, so wording that starts naming its owner again fails at this test
    rather than reaching a page.
    """

    @pytest.mark.parametrize(
        ("path", "payload"),
        [
            (
                "/concepts/pooling/apply",
                {"picture": PATCH, "window": 2, "stride": 2, "kind": "max"},
            ),
            (
                "/concepts/pooling/share-totals",
                {"kind": "average", "largest_side": 8},
            ),
            (
                GEOMETRY,
                {
                    "channels": 1,
                    "height": 8,
                    "width": 8,
                    "windows": [2],
                    "strides": [2],
                },
            ),
            (WINDOW, {"window": [[1, 5], [4, 2]]}),
            (
                SHIFT,
                {
                    "picture": vertical_stroke(2),
                    "window": 2,
                    "stride": 2,
                    "shifts": [0, 1],
                },
            ),
            (EDGE_CASES, None),
        ],
    )
    def test_no_sentence_an_endpoint_returns_names_the_code_behind_it(
        self, path: str, payload: dict | None
    ) -> None:
        if payload is None:
            response = client.get(path)
        else:
            response = client.post(path, json=payload)
        assert response.status_code == 200

        for sentence in every_string_in(response.json()):
            for forbidden in FORBIDDEN_WORDS:
                assert forbidden not in sentence.lower(), f"{path}: {sentence}"
