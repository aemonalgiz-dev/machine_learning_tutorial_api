"""The oriented-gradients endpoints, on the pictures the page is built from.

Every number the page quotes is pinned here, so a change in the library or in
this module fails a test rather than quietly changing a sentence on the page.
The last class is the one that matters most: nothing these endpoints return may
carry an identifier, because a widget renders these strings.
"""

from __future__ import annotations

import math
import re

from fastapi.testclient import TestClient

from oop_ml_api.concepts.oriented_gradients import PICTURES, RULE_LABELS
from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/histogram-of-oriented-gradients"
DESCRIBE = f"{BASE}/describe"
HAND_WORKED = f"{BASE}/hand-worked"
SHARING = f"{BASE}/vote-sharing"
LIGHTING = f"{BASE}/lighting"
TURNING = f"{BASE}/turning"
RANGE_CHOICE = f"{BASE}/range-choice"
SIZE = f"{BASE}/size"
INSIDE = f"{BASE}/inside-one-cell"


def described(**settings: object) -> dict:
    """One picture under one set of settings, unwrapped."""
    response = client.post(DESCRIBE, json=settings)
    assert response.status_code == 200, response.json()
    return response.json()


def sized(**settings: object) -> dict:
    """One arrangement's size arithmetic, unwrapped."""
    response = client.post(SIZE, json=settings)
    assert response.status_code == 200, response.json()
    return response.json()


class TestTheSceneAtTheUsualSettings:
    def test_the_grid_and_the_buckets_are_the_pedestrian_detector_s(self) -> None:
        seen = described()

        assert seen["height"] == 48
        assert seen["width"] == 48
        assert seen["n_pixels"] == 2304
        assert seen["cell_side"] == 8
        assert seen["n_cell_rows"] == 6
        assert seen["n_cell_columns"] == 6
        assert seen["n_buckets"] == 9
        assert seen["range_span_degrees"] == 180.0
        assert seen["bucket_span_degrees"] == 20.0
        assert seen["bucket_centre_degrees"] == [
            10.0,
            30.0,
            50.0,
            70.0,
            90.0,
            110.0,
            130.0,
            150.0,
            170.0,
        ]

    def test_the_finished_answer_holds_nine_hundred_numbers(self) -> None:
        seen = described()

        assert seen["n_block_rows"] == 5
        assert seen["n_block_columns"] == 5
        assert seen["n_values"] == 900
        assert seen["cells_in_the_answer"] == 100.0
        assert math.isclose(seen["description_length"], 5.0, rel_tol=1e-12)

    def test_every_pixel_votes_and_the_weights_add_up(self) -> None:
        seen = described()
        by_cell = sum(sum(row) for row in seen["cell_weights"])

        assert math.isclose(seen["total_weight"], 1064.378464, abs_tol=1e-5)
        assert math.isclose(by_cell, seen["total_weight"], rel_tol=1e-9)

    def test_the_square_s_upright_edge_splits_between_the_first_and_the_last(
        self,
    ) -> None:
        """Nought degrees is as far from the first bucket's centre as from the
        last one's, and over half a circle those two are neighbours."""
        cell = described()["cells"][1][0]
        weight = sum(cell)

        assert math.isclose(cell[0] / weight, 0.5, abs_tol=1e-9)
        assert math.isclose(cell[8] / weight, 0.5, abs_tol=1e-9)
        assert cell[1:8] == [0.0] * 7

    def test_and_the_whole_vote_rule_puts_the_same_edge_in_one_bucket(self) -> None:
        cell = described(vote_sharing="whole_to_nearest")["cells"][1][0]

        assert math.isclose(cell[0] / sum(cell), 1.0, abs_tol=1e-9)


class TestTheSixBySixPatchWorkedByHand:
    def test_the_sweep_answers_four_at_the_two_columns_beside_the_edge(self) -> None:
        seen = client.get(HAND_WORKED).json()
        by_position = {
            (reading["row"], reading["column"]): reading for reading in seen["readings"]
        }

        assert len(seen["readings"]) == 36
        for row in range(6):
            for column in range(6):
                reading = by_position[(row, column)]
                expected = 4.0 if column in (2, 3) else 0.0
                assert reading["across"] == expected
                assert reading["down"] == 0.0
                assert reading["magnitude"] == expected
                assert reading["direction_degrees"] == 0.0

    def test_each_of_the_four_cells_holds_twelve_split_in_half(self) -> None:
        seen = client.get(HAND_WORKED).json()

        assert seen["n_cell_rows"] == 2
        assert seen["n_cell_columns"] == 2
        assert seen["total_weight"] == 48.0
        assert seen["weight_per_cell"] == 12.0
        for row in seen["cells"]:
            for cell in row:
                assert cell == [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0]

    def test_the_answer_is_one_block_of_thirty_six_numbers(self) -> None:
        seen = client.get(HAND_WORKED).json()

        assert seen["n_values"] == 36
        assert math.isclose(seen["unnormalised_length"], math.sqrt(288.0), abs_tol=1e-5)
        assert seen["unnormalised"][0] == 6.0
        assert math.isclose(seen["normalised"][0], 6.0 / math.sqrt(288.0), abs_tol=1e-6)

    def test_the_whole_vote_rule_puts_each_cell_s_twelve_in_one_bucket(self) -> None:
        seen = client.get(HAND_WORKED).json()

        for row in seen["cells_whole_vote"]:
            for cell in row:
                assert cell == [12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_cutting_back_at_the_limit_leaves_this_patch_exactly_where_it_was(
        self,
    ) -> None:
        """Every large entry is the same size, so cutting them all to the limit
        and making the block a unit vector again restores every one of them."""
        seen = client.get(HAND_WORKED).json()

        assert seen["clip_limit"] == 0.2
        assert seen["clipped"] == seen["normalised"]
        assert seen["entries_above_the_limit"] == 8
        assert all(entry > 0.2 for entry in seen["clipped"] if entry > 0.0)


class TestSharingOneVoteBetweenTwoBuckets:
    def test_the_jump_is_ten_times_the_slide(self) -> None:
        seen = client.get(SHARING).json()

        assert seen["before_degrees"] == 19.0
        assert seen["after_degrees"] == 21.0
        assert math.isclose(seen["moved_whole"], 724.077344, abs_tol=1e-4)
        assert math.isclose(seen["moved_shared"], 72.407734, abs_tol=1e-4)
        assert math.isclose(seen["ratio"], 10.0, abs_tol=1e-13)
        assert seen["ratio"] != 10.0

    def test_the_whole_vote_changes_bucket_and_the_shared_one_moves_a_tenth(
        self,
    ) -> None:
        seen = client.get(SHARING).json()

        assert seen["whole_before"][0] == 1.0
        assert seen["whole_after"][1] == 1.0
        assert seen["shared_before"][:2] == [0.55, 0.45]
        assert seen["shared_after"][:2] == [0.45, 0.55]

    def test_the_sweep_covers_the_boundary_and_every_step_spends_one_vote(
        self,
    ) -> None:
        seen = client.get(SHARING).json()

        assert len(seen["steps"]) == 46
        assert seen["steps"][0]["degrees"] == 0.0
        assert seen["steps"][45]["degrees"] == 45.0
        for step in seen["steps"]:
            assert math.isclose(sum(step["whole"]), 1.0, abs_tol=1e-6)
            assert math.isclose(sum(step["shared"]), 1.0, abs_tol=1e-6)

    def test_a_ramp_pointing_along_nought_degrees_reads_as_two_half_buckets(
        self,
    ) -> None:
        seen = client.get(SHARING).json()
        flat = seen["steps"][0]

        assert flat["whole"][0] == 1.0
        assert flat["shared"][0] == 0.5
        assert flat["shared"][8] == 0.5

    def test_on_the_boundary_itself_the_whole_vote_rule_is_balanced(self) -> None:
        """At 20 degrees exactly, the ramp's direction is worked out pixel by
        pixel and lands either side of the boundary by amounts far below
        anything the picture could carry, so the whole-vote rule splits the
        cell where the shared rule reads a clean half and half."""
        seen = client.get(SHARING).json()
        boundary = seen["steps"][20]

        assert boundary["degrees"] == 20.0
        assert boundary["whole"][:2] == [0.453125, 0.546875]
        assert boundary["shared"][:2] == [0.5, 0.5]

    def test_the_scene_s_own_upright_edge_says_the_same(self) -> None:
        seen = client.get(SHARING).json()

        assert seen["upright_edge_whole"][0] == 1.0
        assert seen["upright_edge_shared"][0] == 0.5
        assert seen["upright_edge_shared"][8] == 0.5


class TestWhatAChangeOfLightingDoes:
    def test_the_pixels_move_almost_as_far_as_they_were_long(self) -> None:
        seen = client.post(LIGHTING, json={}).json()

        assert seen["n_pixels"] == 2304
        assert seen["n_pixels_changed"] == 2304
        assert math.isclose(seen["pixel_length"], 21.889438, abs_tol=1e-5)
        assert math.isclose(seen["pixel_moved"], 19.515571, abs_tol=1e-5)
        assert math.isclose(seen["pixel_share_moved"], 0.891552, abs_tol=1e-5)

    def test_the_directions_do_not_move_and_every_magnitude_scales_alike(
        self,
    ) -> None:
        seen = client.post(LIGHTING, json={}).json()

        assert seen["largest_direction_difference_degrees"] < 1e-10
        assert math.isclose(seen["smallest_magnitude_ratio"], 1.6, abs_tol=1e-8)
        assert math.isclose(seen["largest_magnitude_ratio"], 1.6, abs_tol=1e-8)

    def test_the_two_halves_of_a_lighting_change_are_bought_at_different_stages(
        self,
    ) -> None:
        """Adding a constant is already free before any rescaling, since the
        sweep's weights add to nought; multiplying is what the rescaling is
        actually for."""
        rows = {
            row["rule"]: row for row in client.post(LIGHTING, json={}).json()["rows"]
        }
        left_alone = rows["none"]

        assert math.isclose(left_alone["description_length"], 221.490322, abs_tol=1e-4)
        assert left_alone["moved_by_the_shift"] < 1e-12
        assert math.isclose(left_alone["moved_by_the_scale"], 132.894, abs_tol=1e-2)
        assert math.isclose(left_alone["moved_by_both"], 132.894, abs_tol=1e-2)

    def test_and_every_rescaling_buys_the_other_half(self) -> None:
        rows = {
            row["rule"]: row for row in client.post(LIGHTING, json={}).json()["rows"]
        }

        for rule in ("l1", "l1_square_root", "l2", "l2_clipped"):
            assert rows[rule]["moved_by_both"] < 1e-14, rule

    def test_each_rescaling_puts_a_block_on_the_scale_it_names(self) -> None:
        rows = {
            row["rule"]: row for row in client.post(LIGHTING, json={}).json()["rows"]
        }

        assert math.isclose(rows["l1"]["first_block_sum"], 1.0, abs_tol=1e-9)
        for rule in ("l1_square_root", "l2", "l2_clipped"):
            assert math.isclose(rows[rule]["first_block_length"], 1.0, abs_tol=1e-9)
        assert rows["none"]["first_block_length"] > 1.0

    def test_a_brighter_lamp_moves_the_unrescaled_answer_further(self) -> None:
        gentle = client.post(LIGHTING, json={"scale": 1.6, "shift": 0.0}).json()
        harsh = client.post(LIGHTING, json={"scale": 3.7, "shift": 12.0}).json()

        gentle_move = next(row for row in gentle["rows"] if row["rule"] == "none")[
            "moved_by_both"
        ]
        harsh_move = next(row for row in harsh["rows"] if row["rule"] == "none")[
            "moved_by_both"
        ]

        assert math.isclose(gentle_move, 132.894, abs_tol=1e-2)
        assert math.isclose(harsh_move, 598.024, abs_tol=1e-2)

    def test_a_lamp_that_cannot_be_switched_off_is_refused(self) -> None:
        assert client.post(LIGHTING, json={"scale": 0.0}).status_code == 422


class TestTheLimitIsNotACeiling:
    def test_cutting_back_lowers_the_loudest_entry_and_leaves_it_above_the_limit(
        self,
    ) -> None:
        plain = described(block_normalisation="l2")
        cut = described(block_normalisation="l2_clipped")

        assert math.isclose(plain["largest_entry"], 0.927352, abs_tol=1e-5)
        assert math.isclose(cut["largest_entry"], 0.537608, abs_tol=1e-5)
        assert cut["largest_entry"] < plain["largest_entry"]
        assert cut["largest_entry"] > cut["clip_limit"]

    def test_and_leaves_more_entries_above_the_limit_than_it_found(self) -> None:
        """Making the block a unit vector again after the cut lifts everything
        that was not cut, so the count of entries over the limit goes up."""
        plain = described(block_normalisation="l2")
        cut = described(block_normalisation="l2_clipped")

        assert plain["entries_above_the_clip_limit"] == 170
        assert cut["entries_above_the_clip_limit"] == 248


class TestATurnIsNotSurvived:
    def test_a_quarter_turn_moves_the_answer_further_than_its_own_length(
        self,
    ) -> None:
        rows = {row["n_buckets"]: row for row in client.get(TURNING).json()["rows"]}
        usual = rows[9]

        assert math.isclose(usual["description_length"], 5.0, rel_tol=1e-12)
        assert math.isclose(usual["moved_by_turning"], 5.233081, abs_tol=1e-4)
        assert usual["moved_by_relighting"] < 1e-14
        assert math.isclose(usual["turned_share"], 1.046616, abs_tol=1e-5)

    def test_the_buckets_alone_cannot_be_blamed(self) -> None:
        """At six buckets a quarter turn is a whole three buckets, and the
        answer still moves, because the cells have moved as well."""
        rows = {row["n_buckets"]: row for row in client.get(TURNING).json()["rows"]}

        assert rows[6]["bucket_span_degrees"] == 30.0
        assert rows[6]["whole_buckets_in_a_quarter_turn"] is True
        assert math.isclose(rows[6]["moved_by_turning"], 4.772876, abs_tol=1e-4)
        assert rows[6]["turned_share"] > 0.9

    def test_every_bucket_count_tells_the_same_story(self) -> None:
        for row in client.get(TURNING).json()["rows"]:
            assert row["turned_share"] > 0.8
            assert row["moved_by_relighting"] < 1e-13

    def test_no_choice_of_rescaling_rescues_it(self) -> None:
        for row in client.get(TURNING).json()["rules"]:
            assert row["turned_share"] > 0.5, row["rule"]


class TestHalfACircleAgainstTheWholeOne:
    def test_a_bar_and_its_opposite_are_the_same_thing_over_half_a_circle(
        self,
    ) -> None:
        pairs = {
            pair["angle_range"]: pair for pair in client.get(RANGE_CHOICE).json()["bar"]
        }

        assert pairs["unsigned"]["distance"] == 0.0
        assert math.isclose(pairs["unsigned"]["description_length"], 1.0, rel_tol=1e-12)

    def test_and_as_far_apart_as_they_could_be_over_the_whole_one(self) -> None:
        pairs = {
            pair["angle_range"]: pair for pair in client.get(RANGE_CHOICE).json()["bar"]
        }

        assert math.isclose(pairs["signed"]["distance"], math.sqrt(2.0), abs_tol=1e-5)
        assert pairs["signed"]["here"]["occupied_buckets"] == [0, 9]
        assert pairs["signed"]["here"]["bucket_centre_degrees"][0] == 10.0
        assert pairs["signed"]["here"]["bucket_centre_degrees"][9] == 190.0

    def test_the_bar_occupies_one_bucket_of_nine_over_half_a_circle(self) -> None:
        pairs = {
            pair["angle_range"]: pair for pair in client.get(RANGE_CHOICE).json()["bar"]
        }

        assert pairs["unsigned"]["here"]["occupied_buckets"] == [0]
        assert pairs["unsigned"]["here"]["shares"][0] == 1.0

    def test_the_whole_scene_says_the_same_as_the_bar(self) -> None:
        pairs = {
            pair["angle_range"]: pair
            for pair in client.get(RANGE_CHOICE).json()["scene"]
        }

        assert pairs["unsigned"]["distance"] < 1e-14
        assert math.isclose(pairs["signed"]["distance"], 5.98991, abs_tol=1e-4)
        assert math.isclose(pairs["signed"]["description_length"], 5.0, rel_tol=1e-12)


class TestHowLongTheAnswerIs:
    def test_the_pedestrian_window_answers_the_published_figure(self) -> None:
        seen = sized(height=128, width=64)

        assert seen["n_cell_rows"] == 16
        assert seen["n_cell_columns"] == 8
        assert seen["n_block_rows"] == 15
        assert seen["n_block_columns"] == 7
        assert seen["n_blocks"] == 105
        assert seen["n_values"] == 3780
        assert seen["cells_in_the_answer"] == 420
        assert seen["n_cells"] == 128

    def test_nine_cells_are_described_in_sixteen_cells_worth_of_numbers(self) -> None:
        seen = sized(height=24, width=24)

        assert seen["n_cells"] == 9
        assert seen["cells_in_the_answer"] == 16
        assert seen["coverage"] == [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        assert math.isclose(seen["repetition"], 16 / 9, abs_tol=1e-6)
        assert seen["uncovered"] == 0

    def test_tiling_the_blocks_describes_each_cell_exactly_once(self) -> None:
        seen = sized(height=24, width=24, block_stride_in_cells=2)

        assert seen["cells_in_the_answer"] == 4
        assert seen["n_values"] == 36

    def test_the_scene_s_own_arrangement(self) -> None:
        seen = sized(height=48, width=48)

        assert seen["n_cells"] == 36
        assert seen["cells_in_the_answer"] == 100
        assert seen["n_values"] == 900
        assert math.isclose(seen["numbers_per_pixel"], 900 / 2304, abs_tol=1e-6)

    def test_a_patch_that_does_not_divide_into_cells_is_refused(self) -> None:
        refused = client.post(SIZE, json={"height": 48, "width": 50, "cell_side": 8})

        assert refused.status_code == 422
        assert "does not divide" in refused.json()["detail"]

    def test_a_stride_wider_than_the_block_would_skip_cells_and_is_refused(
        self,
    ) -> None:
        refused = client.post(
            SIZE,
            json={"cells_per_block_side": 2, "block_stride_in_cells": 3},
        )

        assert refused.status_code == 422
        assert "never reach the answer" in refused.json()["detail"]


class TestWhatACellCannotSee:
    def test_a_square_moved_inside_its_own_cell_changes_nothing_at_all(self) -> None:
        seen = client.get(INSIDE).json()

        assert math.isclose(seen["pixels_moved_inside"], 4.242641, abs_tol=1e-5)
        assert seen["description_moved_inside"] == 0.0
        assert seen["counts_moved_inside"] == 0.0

    def test_and_the_same_move_across_a_cell_edge_changes_everything(self) -> None:
        seen = client.get(INSIDE).json()

        assert math.isclose(seen["pixels_moved_across"], 4.242641, abs_tol=1e-5)
        assert math.isclose(
            seen["description_moved_across"], math.sqrt(2.0), abs_tol=1e-5
        )
        assert math.isclose(seen["description_length"], 1.0, rel_tol=1e-12)


class TestTheOtherPicturesAndTheRefusals:
    def test_an_even_patch_has_no_direction_to_record(self) -> None:
        seen = described(picture="even")

        assert seen["total_weight"] == 0.0
        assert seen["description_length"] == 0.0
        assert seen["largest_entry"] == 0.0

    def test_relighting_the_scene_leaves_its_description_where_it_was(self) -> None:
        seen = described(picture="scene_relit")

        assert seen["distance_from_the_scene"] < 1e-14

    def test_turning_the_scene_does_not(self) -> None:
        seen = described(picture="scene_turned")

        assert math.isclose(seen["distance_from_the_scene"], 5.233081, abs_tol=1e-4)

    def test_the_scene_is_no_distance_from_itself(self) -> None:
        assert described(picture="scene")["distance_from_the_scene"] == 0.0

    def test_a_cell_side_that_does_not_divide_the_patch_is_refused(self) -> None:
        refused = client.post(DESCRIBE, json={"picture": "scene", "cell_side": 7})

        assert refused.status_code == 422
        assert "does not divide" in refused.json()["detail"]

    def test_a_cell_so_large_that_no_block_fits_is_refused(self) -> None:
        refused = client.post(
            DESCRIBE, json={"picture": "upright_edge", "cell_side": 6}
        )

        assert refused.status_code == 422
        assert "smaller than the" in refused.json()["detail"]

    def test_a_misspelled_setting_is_refused_rather_than_dropped(self) -> None:
        refused = client.post(DESCRIBE, json={"picture": "scene", "cellside": 8})

        assert refused.status_code == 422

    def test_one_bucket_would_describe_nothing_and_is_refused(self) -> None:
        assert (
            client.post(DESCRIBE, json={"picture": "scene", "n_buckets": 1}).status_code
            == 422
        )

    def test_the_signed_range_divides_the_whole_circle(self) -> None:
        seen = described(picture="bar", angle_range="signed", n_buckets=18)

        assert seen["signed"] is True
        assert seen["range_span_degrees"] == 360.0
        assert seen["bucket_span_degrees"] == 20.0
        assert seen["bucket_centre_degrees"][9] == 190.0


IDENTIFIER = re.compile(r"\b[a-z]+_[a-z_]+\b|\b[A-Z][a-z]+[A-Z][A-Za-z]*\b")
"""Anything shaped like a field name or a class name, which a reader never meets."""


class TestNothingVisibleNamesTheCode:
    def test_every_picture_is_labelled_in_plain_english(self) -> None:
        for label, blurb, _ in PICTURES.values():
            assert not IDENTIFIER.search(label), label
            assert not IDENTIFIER.search(blurb), blurb

    def test_every_rescaling_is_described_rather_than_named(self) -> None:
        for label in RULE_LABELS.values():
            assert not IDENTIFIER.search(label), label

    def test_every_refusal_a_reader_can_provoke_is_in_plain_words(self) -> None:
        sentences = [
            client.post(DESCRIBE, json={"picture": "scene", "cell_side": 7}).json()[
                "detail"
            ],
            client.post(
                DESCRIBE, json={"picture": "upright_edge", "cell_side": 6}
            ).json()["detail"],
            client.post(SIZE, json={"height": 48, "width": 50}).json()["detail"],
        ]

        assert sentences
        for sentence in sentences:
            assert not IDENTIFIER.search(sentence), sentence
            assert "oop_ml" not in sentence

    def test_the_answers_a_widget_renders_carry_no_identifier(self) -> None:
        seen = described(picture="scene_relit")
        pair = client.get(RANGE_CHOICE).json()["bar"][0]

        assert not IDENTIFIER.search(seen["picture_label"])
        assert not IDENTIFIER.search(seen["picture_description"])
        assert not IDENTIFIER.search(pair["here"]["picture_label"])
        for row in client.post(LIGHTING, json={}).json()["rows"]:
            assert not IDENTIFIER.search(row["rule_label"]), row["rule_label"]
