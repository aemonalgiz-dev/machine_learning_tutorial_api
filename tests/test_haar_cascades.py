"""Pins for every number the Haar cascades page quotes.

The fixture is the one the module builds: twelve by twelve windows, fifty
targets and two thousand backgrounds, and a sixty-six by sixty-six scene with
three targets planted in it. Every figure below was measured rather than
derived, and the page quotes these and nothing else.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/haar-cascades"

IDENTIFIER = re.compile(r"[a-z]_[a-z]|[a-z][A-Z]|\b[A-Z][a-z]+[A-Z]")
"""Anything spelled the way code is spelled rather than the way English is."""


def _post(path: str, body: dict) -> dict:
    response = client.post(f"{BASE}{path}", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _get(path: str) -> dict:
    response = client.get(f"{BASE}{path}")
    assert response.status_code == 200, response.text
    return response.json()


class TestTheIntegralImage:
    def test_the_table_is_one_larger_on_each_side_and_opens_with_zeros(self):
        body = _post("/integral-image", {})

        assert body["picture"]["height"] == 6
        assert body["picture"]["width"] == 7
        assert len(body["table"]) == 7
        assert len(body["table"][0]) == 8
        assert body["table"][0] == [0.0] * 8
        assert [row[0] for row in body["table"]] == [0.0] * 7

    def test_the_last_entry_is_the_whole_picture(self):
        body = _post("/integral-image", {})

        assert body["table"][-1][-1] == 1176.0

    def test_four_lookups_answer_a_box_whatever_its_size(self):
        """The box three rows down and three across, worked on paper: 340
        minus 10 minus 124 plus 1 is 207."""
        body = _post("/integral-image", {"top": 1, "left": 2, "height": 3, "width": 3})

        assert body["n_lookups"] == 4
        assert len(body["corners"]) == 4
        assert sorted(corner["sign"] for corner in body["corners"]) == [-1, -1, 1, 1]
        assert [corner["value"] for corner in body["corners"]] == [
            340.0,
            10.0,
            124.0,
            1.0,
        ]
        assert body["from_the_table"] == 207.0
        assert body["from_the_pixels"] == 207.0

    def test_a_box_in_the_corner_needs_no_special_case(self):
        body = _post("/integral-image", {"top": 0, "left": 0, "height": 2, "width": 3})

        assert body["from_the_table"] == body["from_the_pixels"] == 36.0

    def test_the_two_routes_agree_on_every_box_of_the_hand_sized_picture(self):
        body = _post("/integral-image", {})

        assert body["agreement"] == {
            "picture_height": 6,
            "picture_width": 7,
            "n_rectangles": 588,
            "n_agreeing": 588,
            "largest_gap": 0.0,
        }

    def test_the_shared_scene_can_be_read_the_same_way(self):
        body = _post(
            "/integral-image",
            {"picture": "workbench", "top": 6, "left": 5, "height": 11, "width": 11},
        )

        assert body["picture"]["height"] == 48
        assert body["n_lookups"] == 4
        assert body["from_the_table"] == pytest.approx(
            body["from_the_pixels"], abs=1e-3
        )

    def test_a_box_running_off_the_picture_is_refused_in_plain_words(self):
        """The refusal a widget shows is page text, so nothing in it may be
        spelled the way code is."""
        response = client.post(
            f"{BASE}/integral-image",
            json={"top": 5, "left": 0, "height": 3, "width": 3},
        )

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "6 by 7" in detail
        assert not IDENTIFIER.search(detail)
        assert "(" not in detail

    def test_a_field_it_does_not_have_is_refused(self):
        response = client.post(f"{BASE}/integral-image", json={"box": 3})

        assert response.status_code == 422


class TestWhatAReadCosts:
    def test_the_lookup_count_does_not_grow_with_the_box(self):
        body = _get("/reading-cost")

        assert [box["n_pixels"] for box in body["boxes"]] == [1, 391, 360000]
        assert [box["n_lookups"] for box in body["boxes"]] == [4, 4, 4]
        assert body["times_larger"] == 360000

    def test_and_neither_does_the_time(self):
        """A wide margin, because a timing assertion that is tight fails for
        reasons nothing to do with what it is measuring. Measured over six
        runs, the table answered the large box in 1.01 to 1.11 times what the
        small one cost, where adding the pixels up cost 107 to 130 times."""
        body = _get("/reading-cost")

        assert body["table_ratio"] < 3.0
        assert body["pixel_ratio"] > 20.0
        assert body["reads_to_repay_the_table"] > 1.0


class TestWhatADetectorReads:
    def test_the_five_arrangements_and_which_of_them_cancel(self):
        body = _get("/arrangements")

        assert len(body["arrangements"]) == 5
        balanced = {one["label"]: one["balanced"] for one in body["arrangements"]}
        assert balanced == {
            "Two cells side by side": True,
            "Two cells stacked": True,
            "Three cells in a row": False,
            "Three cells in a column": False,
            "Four cells in a chequer": True,
        }

    def test_the_worked_readings_are_the_ones_the_page_quotes(self):
        body = _get("/arrangements")

        assert [one["value"] for one in body["worked"]] == [8.0, 0.0, 2.0, 0.0, 2.0]

    def test_the_three_cell_arrangements_read_the_brightness_as_well(self):
        """The finding the page reports rather than smooths over. One added
        cell against two subtracted ones does not cancel, so a picture with no
        contrast in it at all still moves them when the lamp moves."""
        body = _get("/arrangements")
        readings = {one["arrangement"]: one for one in body["level"]}

        assert readings["Two cells stacked"]["on_dim"] == 0.0
        assert readings["Two cells stacked"]["on_bright"] == 0.0
        assert readings["Four cells in a chequer"]["on_bright"] == 0.0
        assert readings["Three cells in a column"]["on_dim"] == -0.25
        assert readings["Three cells in a column"]["on_bright"] == -0.9
        assert readings["Three cells in a row"]["on_bright"] == -0.9

    def test_how_many_features_a_window_admits(self):
        body = _get("/arrangements")
        counts = {one["side"]: one["n_features"] for one in body["counts"]}

        assert counts[4] == 136
        assert counts[6] == 669
        assert counts[8] == 2056
        assert counts[12] == 10344
        assert counts[24] == 162336

    def test_and_that_it_grows_faster_than_the_area(self):
        body = _get("/arrangements")
        per_pixel = {one["side"]: one["features_per_pixel"] for one in body["counts"]}

        assert per_pixel[24] == 281.83
        assert body["doubling_factor"] == 15.69

    def test_the_accurate_way_of_adding_the_cells_up_is_the_wrong_one(self):
        """The second finding the page reports. Compensating the rounding over
        a run of floats, which Python's own summing has done since 3.12, moves
        1 of these 200 values by 2.2e-16, and a threshold sitting exactly on
        such a value is where a last bit decides the answer."""
        rounding = _get("/arrangements")["rounding"]

        assert rounding["n_values"] == 200
        assert rounding["n_disagreeing"] == 1
        assert rounding["largest_gap"] == pytest.approx(2.220446049250313e-16)
        assert rounding["plain"] == "-0.8391149931171447"
        assert rounding["compensated"] == "-0.8391149931171449"
        assert rounding["plain_votes_yes"] is True
        assert rounding["compensated_votes_yes"] is False


class TestTheFittedDetector:
    def test_the_shape_of_the_cascade_it_found(self):
        body = _get("/detector")

        assert body["window_side"] == 12
        assert body["n_stages"] == 3
        assert body["n_rules"] == 5
        assert body["n_candidate_features"] == 600
        assert body["n_available_features"] == 10344
        assert body["n_targets"] == 50
        assert body["n_backgrounds"] == 2000

    def test_what_each_stage_rejected_and_what_it_kept(self):
        body = _get("/detector")

        assert [one["negatives_reaching"] for one in body["stages"]] == [2000, 105, 3]
        assert [one["negatives_rejected"] for one in body["stages"]] == [1895, 102, 3]
        assert [one["positives_kept"] for one in body["stages"]] == [50, 50, 50]
        assert [one["n_rules"] for one in body["stages"]] == [2, 2, 1]

    def test_the_lowering_never_had_to_bite_on_this_fixture(self):
        """Reported rather than glossed. A stage may lower what it demands
        until it keeps every target, and here the plain vote already kept all
        fifty, so the least confident target sat above the half a plain vote
        asks for in every stage and nothing was lowered."""
        body = _get("/detector")

        assert [one["demanded_share"] for one in body["stages"]] == [0.5, 0.5, 0.5]
        assert [one["was_lowered"] for one in body["stages"]] == [False, False, False]
        assert [one["lowest_positive_confidence"] for one in body["stages"]] == [
            0.5355,
            0.5121,
            1.0,
        ]

    def test_the_first_stage_alone_would_be_a_poor_detector(self):
        """117 of 2000 fresh backgrounds get past the first stage where 1 gets
        past all three, which is the whole argument for having three."""
        held_out = _get("/detector")["held_out"]

        assert held_out["backgrounds_through_each_stage"] == [2000, 117, 3, 1]
        assert held_out["background_shares"] == [0.0585, 0.0256, 0.3333]

    def test_every_fresh_target_that_is_lost_is_lost_at_the_first_stage(self):
        held_out = _get("/detector")["held_out"]

        assert held_out["targets_through_each_stage"] == [50, 47, 47, 47]
        assert held_out["recall"] == 0.94
        assert held_out["accuracy"] == 0.998
        assert held_out["precision"] == 0.9792

    def test_the_rules_are_drawable(self):
        body = _get("/detector")
        rules = [rule for stage in body["stages"] for rule in stage["rules"]]

        assert len(rules) == 5
        assert [len(rule["cells"]) for rule in rules] == [2, 3, 3, 3, 2]
        for rule in rules:
            assert rule["voice"] > 0.0
            assert rule["direction"] in ("above", "at or below")
            assert sum(cell["sign"] for cell in rule["cells"]) in (0, -1)


class TestTheSweep:
    def test_most_windows_die_at_the_first_stage(self):
        """The load-bearing measurement of the whole page."""
        body = _get("/scan")

        assert body["windows_examined"] == 3025
        assert [one["windows_reaching"] for one in body["stages"]] == [3025, 235, 137]
        assert [one["share_of_all_windows"] for one in body["stages"]] == [
            1.0,
            0.0777,
            0.0453,
        ]

    def test_what_the_ordering_saved_four_ways(self):
        body = _get("/scan")

        assert body["rule_evaluations"] == 6657
        assert body["flat_evaluations"] == 15125
        assert body["searched_bank"] == 1_815_000
        assert body["exhaustive_bank"] == 31_290_600
        assert body["saving_over_flat"] == 2.272
        assert body["saving_over_searched_bank"] == 272.6
        assert body["saving_over_exhaustive_bank"] == 4700.4
        assert body["rules_per_window"] == 2.2007

    def test_the_expected_cost_is_the_stages_weighted_by_who_reaches_them(self):
        body = _get("/scan")
        by_hand = sum(
            one["n_rules"] * one["share_of_all_windows"] for one in body["stages"]
        )

        assert by_hand == pytest.approx(body["rules_per_window"], abs=0.001)

    def test_it_finds_the_planted_targets_and_a_great_many_other_places(self):
        body = _get("/scan")
        accepted = {(one["top"], one["left"]) for one in body["accepted"]}

        assert body["n_accepted"] == 111
        assert {(one["top"], one["left"]) for one in body["planted"]} <= accepted
        assert body["accepted_near_a_target"] == 28

    def test_the_survivors_can_be_drawn_and_match_the_counts(self):
        body = _get("/scan")

        assert [len(one) for one in body["survivors"]] == [235, 137]
        assert body["scene"]["height"] == body["scene"]["width"] == 66


class TestThreeScenesOneDetector:
    def test_a_scene_that_is_mostly_target_is_the_one_it_is_slowest_on(self):
        """The place the ordering buys least. Tiling the same ground with
        targets takes the cost from 2.16 rules a window to 2.66 and the saving
        from 2.32 to 1.88, and the ceiling is the whole cascade's five."""
        body = _get("/scenes")
        by_label = {one["label"]: one for one in body["scenes"]}

        assert body["most_rules_per_window"] == 5
        assert by_label["Nothing but ground"]["rules_per_window"] == 2.1597
        assert by_label["Three targets hidden in it"]["rules_per_window"] == 2.2007
        assert by_label["Tiled edge to edge with targets"]["rules_per_window"] == 2.6552
        assert by_label["Nothing but ground"]["saving_over_flat"] == 2.315
        assert by_label["Tiled edge to edge with targets"]["saving_over_flat"] == 1.883

    def test_the_windows_reaching_each_stage_move_with_the_scene(self):
        body = _get("/scenes")
        by_label = {one["label"]: one for one in body["scenes"]}

        assert by_label["Nothing but ground"]["windows_reaching"] == [3025, 189, 105]
        assert by_label["Tiled edge to edge with targets"]["windows_reaching"] == [
            3025,
            693,
            596,
        ]


class TestThePosesItWasNeverShown:
    def test_relit_and_turned_targets_are_never_found_at_all(self):
        """Not found less often. Found zero times out of fifty, twice over,
        where fresh upright ones are found 43 times."""
        body = _get("/poses")
        found = {one["label"]: one["n_found"] for one in body["poses"]}

        assert found["Upright, as it was trained"] == 43
        assert found["Lit from the other side"] == 0
        assert found["Turned a quarter"] == 0
        assert all(one["n_shown"] == 50 for one in body["poses"])

    def test_the_reading_behind_it_changes_sign(self):
        body = _get("/poses")
        flip = body["sign_flip"]

        assert flip["on_the_target"] == -7.1856
        assert flip["on_the_relit_target"] == 7.3344
        assert flip["on_the_target"] * flip["on_the_relit_target"] < 0.0
        assert flip["votes_yes_on_the_target"] is True
        assert flip["votes_yes_on_the_relit_target"] is False


class TestNothingTheReaderSeesIsSpelledLikeCode:
    """Every string these endpoints return is page text, so every one of them
    is held to the same rule the prose is."""

    def _strings(self, value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [one for item in value.values() for one in self._strings(item)]
        if isinstance(value, list):
            return [one for item in value for one in self._strings(item)]
        return []

    @pytest.mark.parametrize(
        "path",
        ["/reading-cost", "/arrangements", "/detector", "/scan", "/scenes", "/poses"],
    )
    def test_no_identifier_reaches_the_page(self, path):
        for sentence in self._strings(_get(path)):
            if sentence.startswith(("-0.", "0.")):
                continue
            assert not IDENTIFIER.search(sentence), sentence
            for forbidden in ("oop_ml", "the library", "the API", "the package"):
                assert forbidden not in sentence
