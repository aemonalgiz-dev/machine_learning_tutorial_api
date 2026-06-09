"""The codebook-quantisation endpoints, on the four pictures the page is built from.

Every number the page quotes is pinned here, so a change in the library or in
this module fails a test rather than quietly changing a sentence on the page.
The last class is the one that matters most: nothing these endpoints return may
carry an identifier, because a widget renders these strings.
"""

from __future__ import annotations

import math
import re

from fastapi.testclient import TestClient

from oop_ml_api.concepts.codebook_quantisation import (
    PICTURE_LABELS,
    TABLE_SIZES,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/codebook-quantisation"
NEAREST = f"{BASE}/nearest-code"
QUANTISE = f"{BASE}/quantise"
SWEEP = f"{BASE}/table-size-sweep"
SAME_NUMBER = f"{BASE}/same-number"
CHOOSING = f"{BASE}/choosing-the-table"


def quantise(picture: str, n_codes: int, table_from: str | None = None) -> dict:
    """One picture under one table, unwrapped."""
    body: dict[str, object] = {"picture": picture, "n_codes": n_codes}
    if table_from is not None:
        body["table_from"] = table_from
    response = client.post(QUANTISE, json=body)
    assert response.status_code == 200, response.json()
    return response.json()


def sweep_for(picture: str) -> dict:
    """One picture's row of the sweep."""
    response = client.get(SWEEP)
    assert response.status_code == 200
    return next(row for row in response.json()["pictures"] if row["picture"] == picture)


def step_at(picture: str, n_codes: int) -> dict:
    """One table size on one picture."""
    return next(
        step for step in sweep_for(picture)["steps"] if step["n_codes"] == n_codes
    )


class TestFourVectorsAgainstThreeEntries:
    def test_every_squared_distance_is_the_hand_worked_one(self) -> None:
        rows = client.get(NEAREST).json()["rows"]

        assert [row["squared_distances"] for row in rows] == [
            [1.0, 9.0, 10.0],
            [10.0, 2.0, 13.0],
            [5.0, 13.0, 2.0],
            [4.0, 4.0, 13.0],
        ]

    def test_the_answers_and_the_gaps_are_one_two_two_four(self) -> None:
        rows = client.get(NEAREST).json()["rows"]

        assert [row["code_id"] for row in rows] == [0, 1, 2, 0]
        assert [row["squared_gap"] for row in rows] == [1.0, 2.0, 2.0, 4.0]

    def test_the_rounding_error_is_the_mean_of_those_four_gaps(self) -> None:
        body = client.get(NEAREST).json()

        assert body["distortion"] == 2.25
        assert body["distortion"] == sum(
            row["squared_gap"] for row in body["rows"]
        ) / len(body["rows"])

    def test_only_the_last_vector_is_tied(self) -> None:
        rows = client.get(NEAREST).json()["rows"]

        assert [row["tied"] for row in rows] == [False, False, False, True]

    def test_listing_the_same_entries_differently_moves_only_the_tie(self) -> None:
        """The tied vector follows the lower position, so reordering moves it."""
        body = client.get(NEAREST).json()

        assert body["reordered_ids"] == [1, 0, 2, 0]
        assert body["reordered_distortion"] == body["distortion"] == 2.25


class TestThePhotograph:
    def test_it_is_a_hundred_and_twenty_eight_pieces_of_which_fifty_three_differ(
        self,
    ) -> None:
        seen = quantise("photograph", 16)

        assert seen["n_pieces"] == 128
        assert seen["n_distinct_pieces"] == 53
        assert seen["side"] == 16
        assert len(seen["pixels"]) == 16
        assert all(len(row) == 16 for row in seen["pixels"])

    def test_sixteen_entries_round_every_piece_to_within_a_hundredth(self) -> None:
        seen = quantise("photograph", 16)

        assert seen["distortion"] == 0.0004179970089908049
        assert seen["root_mean_square_gap"] == 0.014456780571600387
        assert seen["worst_squared_gap"] == 0.0024305785123966746

    def test_every_entry_of_a_table_chosen_from_it_wins_something(self) -> None:
        seen = quantise("photograph", 16)

        assert seen["n_unused"] == 0
        assert sum(seen["usage"]) == 128
        assert min(seen["usage"]) > 0

    def test_the_reconstruction_is_the_entry_each_piece_was_given(self) -> None:
        seen = quantise("photograph", 8)
        flat = [value for row in seen["reconstruction"] for value in row]

        for position, piece in enumerate(seen["pieces"]):
            entry = seen["codes"][piece["code_id"]]
            assert flat[2 * position] == entry[0]
            assert flat[2 * position + 1] == entry[1]

    def test_the_numbers_cost_four_bits_each_at_sixteen_entries(self) -> None:
        seen = quantise("photograph", 16)

        assert seen["bits_per_piece"] == 4.0
        assert seen["total_id_bits"] == 512.0

    def test_one_entry_carries_no_information_at_all(self) -> None:
        seen = quantise("photograph", 1)

        assert seen["bits_per_piece"] == 0.0
        assert seen["total_id_bits"] == 0.0
        assert {piece["code_id"] for piece in seen["pieces"]} == {0}


class TestWhatABiggerTableBuys:
    def test_the_photograph_falls_by_about_four_at_every_doubling(self) -> None:
        photograph = sweep_for("photograph")
        cuts = [
            step["cut_by"] for step in photograph["steps"] if step["cut_by"] is not None
        ]

        assert [round(cut, 2) for cut in cuts] == [4.57, 3.24, 4.10, 4.49, 7.96]

    def test_the_speckled_square_falls_by_about_two(self) -> None:
        """Pieces that fill the plane are the case the high-rate result predicts."""
        speckle = sweep_for("speckle")
        cuts = [
            step["cut_by"] for step in speckle["steps"] if step["cut_by"] is not None
        ]

        assert [round(cut, 2) for cut in cuts] == [1.74, 2.52, 2.22, 2.30, 2.63]

    def test_the_photograph_lies_near_a_line_and_the_speckle_does_not(self) -> None:
        assert round(sweep_for("photograph")["spread_ratio"], 1) == 163.6
        assert round(sweep_for("speckle")["spread_ratio"], 1) == 1.4

    def test_the_poster_gains_seventy_times_at_the_size_of_its_own_variety(
        self,
    ) -> None:
        poster = sweep_for("poster")

        assert round(step_at("poster", 4)["cut_by"], 2) == 69.81
        assert round(step_at("poster", 8)["cut_by"], 2) == 1.76
        assert poster["n_distinct_pieces"] == 93

    def test_every_size_is_a_doubling_and_the_bits_go_up_by_one(self) -> None:
        photograph = sweep_for("photograph")

        assert [step["n_codes"] for step in photograph["steps"]] == list(TABLE_SIZES)
        assert [step["bits_per_piece"] for step in photograph["steps"]] == [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ]

    def test_a_table_of_one_is_the_spread_of_the_pieces_about_their_mean(self) -> None:
        body = client.get(CHOOSING).json()

        assert body["one_entry_distortion"] == body["spread_about_the_mean"]
        assert step_at("photograph", 1)["distortion"] == body["one_entry_distortion"]


class TestThePrintedChart:
    def test_it_holds_only_five_different_pieces(self) -> None:
        assert sweep_for("chart")["n_distinct_pieces"] == 5

    def test_eight_entries_reproduce_it_exactly_and_three_never_win(self) -> None:
        step = step_at("chart", 8)

        assert step["exact"] is True
        assert step["distortion"] < 1e-12
        assert step["cut_by"] is None
        assert step["n_unused"] == 3

    def test_a_table_larger_than_its_variety_cannot_be_chosen(self) -> None:
        for n_codes in (16, 32):
            step = step_at("chart", n_codes)
            assert step["distortion"] is None
            assert "two numbers meaning one thing" in step["refused"]

    def test_the_same_refusal_reaches_a_caller_as_a_typed_answer(self) -> None:
        response = client.post(QUANTISE, json={"picture": "chart", "n_codes": 16})

        assert response.status_code == 422
        assert "two numbers meaning one thing" in response.json()["detail"]


class TestOneNumberForMoreThanOnePiece:
    def test_two_entries_give_one_number_to_pieces_far_apart(self) -> None:
        collision = client.get(SAME_NUMBER).json()["collisions"][0]

        assert collision["n_codes"] == 2
        assert collision["first"] == [0.0, 0.0]
        assert collision["second"] == [0.41, 0.45]
        assert round(collision["separation"], 4) == 0.6088
        assert [round(value, 4) for value in collision["reconstruction"]] == [
            0.2067,
            0.2366,
        ]

    def test_the_four_entry_pair_the_page_walks_through(self) -> None:
        collision = next(
            row
            for row in client.get(SAME_NUMBER).json()["collisions"]
            if row["n_codes"] == 4
        )

        assert collision["first"] == [0.35, 0.39]
        assert collision["second"] == [0.56, 0.62]
        assert [round(value, 4) for value in collision["reconstruction"]] == [
            0.449,
            0.5119,
        ]

    def test_the_widest_such_gap_narrows_as_the_table_grows(self) -> None:
        collisions = client.get(SAME_NUMBER).json()["collisions"]

        assert [collision["n_codes"] for collision in collisions] == [2, 4, 8, 16]
        separations = [collision["separation"] for collision in collisions]
        assert separations == sorted(separations, reverse=True)
        assert [round(value, 4) for value in separations] == [
            0.6088,
            0.3114,
            0.1772,
            0.0894,
        ]

    def test_a_picture_the_table_was_never_chosen_for_rounds_far_worse(self) -> None:
        body = client.get(SAME_NUMBER).json()

        assert body["elsewhere_n_codes"] == 16
        assert body["home_distortion"] == 0.0004179970089908049
        assert body["elsewhere_distortion"] == 0.10429026083806298
        assert round(body["elsewhere_ratio"], 1) == 249.5

    def test_the_best_and_the_worst_match_are_both_one_whole_number(self) -> None:
        """Nothing about either answer says how far the piece had to move."""
        body = client.get(SAME_NUMBER).json()

        assert body["elsewhere_best"]["piece"] == [0.05, 0.05]
        assert body["elsewhere_best"]["squared_gap"] == 0.001
        assert body["elsewhere_worst"]["piece"] == [0.9, 0.05]
        assert round(body["elsewhere_worst"]["squared_gap"], 6) == 0.404696
        assert isinstance(body["elsewhere_best"]["code_id"], int)
        assert isinstance(body["elsewhere_worst"]["code_id"], int)


class TestChoosingTheTable:
    def test_a_table_chosen_from_half_the_picture_leaves_entries_unused(self) -> None:
        rows = client.get(CHOOSING).json()["rows"]

        assert [row["n_codes"] for row in rows] == [2, 4, 8, 16, 32]
        assert [row["fitted_unused"] for row in rows] == [0, 0, 0, 0, 0]
        assert [row["held_out_unused"] for row in rows] == [0, 1, 2, 6, 13]

    def test_the_held_out_half_also_rounds_worse(self) -> None:
        row = next(
            row for row in client.get(CHOOSING).json()["rows"] if row["n_codes"] == 16
        )

        assert row["fitted_distortion"] == 0.0002694531249999999
        assert row["held_out_distortion"] == 0.0008267795138888884

    def test_a_picture_from_elsewhere_leaves_most_of_the_table_unused(self) -> None:
        rows = client.get(CHOOSING).json()["rows"]

        assert [row["elsewhere_unused"] for row in rows] == [0, 0, 3, 11, 27]

    def test_the_starting_draw_barely_moves_the_answer_here(self) -> None:
        """Six starts, two answers, and they differ in the sixth decimal."""
        starts = client.get(CHOOSING).json()["starts"]

        assert [start["seed"] for start in starts] == [0, 1, 2, 3, 4, 5]
        reached = sorted({round(start["distortion"], 7) for start in starts})
        assert reached == [0.0018756, 0.0018772]

    def test_adding_the_two_coordinate_gaps_sends_one_piece_elsewhere(self) -> None:
        body = client.get(CHOOSING).json()

        assert body["disagreement_size"] == 8
        assert body["n_disagreements"] == 1
        only = body["disagreements"][0]
        assert only["piece"] == [0.13, 0.3]
        assert only["straight_line_id"] == 5
        assert only["coordinate_sum_id"] == 1
        assert round(only["straight_line_distances"][5], 4) == 0.1078
        assert round(only["straight_line_distances"][1], 4) == 0.1303
        assert round(only["coordinate_sum_distances"][5], 4) == 0.148
        assert round(only["coordinate_sum_distances"][1], 4) == 0.1437

    def test_the_largest_table_the_photograph_admits_is_its_own_variety(self) -> None:
        body = client.get(CHOOSING).json()

        assert body["largest_table"] == 53
        assert round(body["largest_table_distortion"] * 1e33, 1) == 4.6
        assert body["refusal_size"] == 54
        assert "two numbers meaning one thing" in body["refusal"]
        assert body["largest_table"] == quantise("photograph", 2)["n_distinct_pieces"]


class TestTheRequestIsBounded:
    def test_an_unknown_picture_is_refused(self) -> None:
        response = client.post(QUANTISE, json={"picture": "etching", "n_codes": 4})

        assert response.status_code == 422

    def test_a_table_of_no_entries_is_refused(self) -> None:
        response = client.post(QUANTISE, json={"picture": "photograph", "n_codes": 0})

        assert response.status_code == 422

    def test_a_table_beyond_the_ceiling_is_refused(self) -> None:
        response = client.post(QUANTISE, json={"picture": "photograph", "n_codes": 500})

        assert response.status_code == 422

    def test_an_unexpected_field_is_refused(self) -> None:
        response = client.post(
            QUANTISE, json={"picture": "photograph", "n_codes": 4, "seed": 1}
        )

        assert response.status_code == 422

    def test_a_table_may_be_chosen_from_one_picture_and_used_on_another(self) -> None:
        seen = quantise("chart", 16, table_from="photograph")

        assert seen["picture"] == "chart"
        assert seen["table_from"] == "photograph"
        assert seen["distortion"] == 0.10429026083806298
        assert seen["n_unused"] == 11


IDENTIFIER = re.compile(r"\b[a-z]+_[a-z_]+\b|\b[A-Z][a-z]+[A-Z][A-Za-z]*\b")
"""Anything shaped like a field name or a class name, which a reader never meets."""


class TestNothingVisibleNamesTheCode:
    def test_the_picture_labels_are_plain_english(self) -> None:
        for label in PICTURE_LABELS.values():
            assert not IDENTIFIER.search(label)

    def test_every_refusal_a_widget_can_show_is_in_plain_words(self) -> None:
        sentences = [
            step["refused"]
            for picture in client.get(SWEEP).json()["pictures"]
            for step in picture["steps"]
            if step["refused"]
        ]
        refused = client.post(QUANTISE, json={"picture": "chart", "n_codes": 32})
        sentences.append(refused.json()["detail"])
        sentences.append(client.get(CHOOSING).json()["refusal"])

        assert sentences
        for sentence in sentences:
            assert not IDENTIFIER.search(sentence), sentence
            assert "codes" not in sentence
            assert "oop_ml" not in sentence

    def test_every_string_the_sweep_returns_is_in_plain_words(self) -> None:
        for picture in client.get(SWEEP).json()["pictures"]:
            assert not IDENTIFIER.search(picture["label"])
            assert not IDENTIFIER.search(picture["description"])

    def test_the_quantise_answer_says_nothing_a_reader_cannot_read(self) -> None:
        seen = quantise("poster", 8)

        assert not IDENTIFIER.search(seen["picture_label"])
        assert not IDENTIFIER.search(seen["picture_description"])
        assert not IDENTIFIER.search(seen["table_from_label"])


class TestTheArithmeticIsConsistent:
    def test_the_rounding_error_is_the_mean_of_the_squared_gaps(self) -> None:
        seen = quantise("speckle", 8)
        gaps = [piece["squared_gap"] for piece in seen["pieces"]]

        assert math.isclose(seen["distortion"], sum(gaps) / len(gaps), rel_tol=1e-12)

    def test_the_root_mean_square_gap_is_per_coordinate(self) -> None:
        seen = quantise("photograph", 16)

        assert math.isclose(
            seen["root_mean_square_gap"],
            math.sqrt(seen["distortion"] / 2),
            rel_tol=1e-12,
        )

    def test_the_bits_are_the_logarithm_of_the_table_size(self) -> None:
        for n_codes in (2, 4, 8, 16, 32):
            seen = quantise("photograph", n_codes)
            assert seen["bits_per_piece"] == math.log2(n_codes)
            assert seen["total_id_bits"] == 128 * math.log2(n_codes)
