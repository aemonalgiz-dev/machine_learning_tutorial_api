"""The image alignment page's endpoints, on the shared workbench scene.

Two kinds of claim, as the other vision pages keep them. The quarter turn of
the bent bar is exact, so its numbers are asserted as equalities: a turned
corner reads the same bits, the same histogram, and the patch descriptor's
old failure comes out at the keypoints page's own 1.443376. The thirty-degree
turn of the scene is resampled, so nothing about it is exact, and what is
pinned there is what the page quotes: how many matches, how many right, and
that the method's inliers are exactly the right ones.

The load-bearing test is ``TestTheRansacReplayAgreesWithTheLibrary``. The
replay of every pair is the one computation this module does for itself, and
the test holds its winner to the library's answer.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/image-alignment"
ALIGN = f"{BASE}/align"
TURN_COST = f"{BASE}/turn-cost"
ORIENTATION = f"{BASE}/orientation"
DESCRIBE = f"{BASE}/describe"
RANSAC = f"{BASE}/ransac"
PYRAMID = f"{BASE}/pyramid"


def aligned(**body) -> dict:
    response = client.post(ALIGN, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def fetched(path: str) -> dict:
    response = client.get(path)
    assert response.status_code == 200, response.text
    return response.json()


class TestTheOpeningMove:
    def test_the_two_pictures_are_the_framed_scene_and_its_turned_self(self) -> None:
        body = aligned()

        assert (body["height"], body["width"]) == (64, 64)
        assert len(body["first"]) == 64 and len(body["second"]) == 64
        assert body["truth"] == {
            "scale": 1.0,
            "angle_degrees": 30.0,
            "row_shift": 22.97,
            "column_shift": -16.53,
        }

    def test_the_orb_pair_keeps_fifteen_matches_of_which_ten_are_right(self) -> None:
        body = aligned()

        assert (body["n_matches"], body["n_right"]) == (15, 10)
        assert len(body["first_keypoints"]) == 40
        assert len(body["second_keypoints"]) == 40

    def test_and_its_inliers_are_exactly_the_right_ones(self) -> None:
        """The method's verdict against the truth, which the page draws as a
        two-by-two count. Nothing right was refused and nothing wrong kept."""
        body = aligned()

        assert body["inliers_right"] == 10
        assert body["inliers_wrong"] == 0
        assert body["outliers_right"] == 0
        assert body["outliers_wrong"] == 5

    def test_the_recovered_move_is_within_a_degree_and_lines_the_pictures_up(
        self,
    ) -> None:
        body = aligned()

        assert body["angle_gap_degrees"] == 0.56
        assert body["estimate"]["scale"] == pytest.approx(1.0, abs=0.01)
        assert body["difference_before"] == 0.1755
        assert body["difference_after"] == 0.0187
        assert body["refusal"] is None

    def test_the_sift_pair_finds_more_keypoints_because_a_symmetric_corner_is_two(
        self,
    ) -> None:
        body = aligned(method="sift")

        assert body["n_matches"] == 34
        assert body["n_right"] == 17
        assert body["inliers_right"] == 17
        assert body["inliers_wrong"] == 0
        assert body["angle_gap_degrees"] == 0.14

    def test_every_match_names_keypoints_that_exist(self) -> None:
        body = aligned()

        for match in body["matches"]:
            assert 0 <= match["first"] < len(body["first_keypoints"])
            assert 0 <= match["second"] < len(body["second_keypoints"])
            assert 0.0 <= match["ratio"] <= 0.8

    def test_a_turn_with_no_shift_and_no_scale_is_still_recovered(self) -> None:
        body = aligned(angle_degrees=-45.0, row_shift=0.0, column_shift=0.0)

        assert body["refusal"] is None
        assert body["angle_gap_degrees"] < 2.0

    def test_an_out_of_range_move_is_refused_before_anything_is_computed(self) -> None:
        assert client.post(ALIGN, json={"scale": 3.0}).status_code == 422
        assert client.post(ALIGN, json={"angle_degrees": 400}).status_code == 422
        assert client.post(ALIGN, json={"unknown": 1}).status_code == 422


class TestWhatAQuarterTurnCosts:
    def test_the_patch_descriptor_fails_by_the_keypoints_pages_own_number(self) -> None:
        body = fetched(TURN_COST)

        assert len(body["rows"]) == 6
        for row in body["rows"]:
            assert row["patch_distance"] == 1.443376

    def test_and_both_oriented_descriptors_do_not_move_at_all(self) -> None:
        body = fetched(TURN_COST)

        for row in body["rows"]:
            assert row["binary_bits"] == 0.0
            assert row["histogram_distance"] == 0.0

    def test_three_convex_corners_are_bit_identical_to_each_other(self) -> None:
        """The repeated-structure case the page states rather than hides."""
        body = fetched(TURN_COST)
        by_corner = {(row["row"], row["column"]): row for row in body["rows"]}

        for corner in ((14, 14), (34, 14), (34, 22)):
            assert by_corner[corner]["binary_nearest_other"] == 0.0
            assert by_corner[corner]["histogram_nearest_other"] > 0.3
        assert by_corner[(20, 23)]["binary_nearest_other"] == 78.0


class TestTheDirections:
    def test_the_centroid_points_from_each_corner_into_the_bright_side(self) -> None:
        body = fetched(ORIENTATION)
        angles = {
            (c["row"], c["column"]): c["centroid_angle_degrees"]
            for c in body["corners"]
        }

        assert angles[(14, 14)] == 45.0
        assert angles[(14, 34)] == 135.0
        assert angles[(34, 14)] == -45.0
        assert angles[(34, 22)] == -135.0

    def test_the_histogram_has_thirty_six_bins_and_names_its_fullest(self) -> None:
        body = fetched(ORIENTATION)

        assert body["bins"] == 36
        for corner in body["corners"]:
            assert len(corner["votes"]) == 36
            assert corner["votes"][corner["fullest_bin"]] == max(corner["votes"])
            assert len(corner["patch"]) == 2 * body["radius"] + 1


class TestOneCornerDescribed:
    def test_the_turned_corner_reads_the_same_bits_and_the_same_histogram(self) -> None:
        body = fetched(DESCRIBE)

        assert body["hamming"] == 0.0
        assert body["histogram_distance"] == 0.0
        assert body["before"]["bits"] == body["after"]["bits"]
        assert len(body["before"]["bits"]) == body["pair_count"] == 256
        assert len(body["before"]["histogram"]) == 128

    def test_and_faces_a_quarter_turn_the_other_way(self) -> None:
        body = fetched(DESCRIBE)

        assert body["before"]["angle_degrees"] == 45.0
        assert body["after"]["angle_degrees"] == -45.0


class TestTheRansacReplayAgreesWithTheLibrary:
    def test_every_pair_is_tried_and_the_winner_explains_the_most(self) -> None:
        body = fetched(RANSAC)

        assert body["n_matches"] == 15
        assert body["n_pairs"] == 104
        winner = body["proposals"][body["winner"]]
        assert winner["n_agreeing"] == max(p["n_agreeing"] for p in body["proposals"])
        assert winner["n_agreeing"] == 10
        assert winner["both_right"] is True

    def test_most_pairs_explain_only_themselves(self) -> None:
        body = fetched(RANSAC)
        counts = body["agreement_counts"]

        assert counts[2] == 20
        assert counts[10] == 10
        assert sum(counts) == body["n_pairs"]

    def test_the_winner_proposes_the_move_the_library_settled_on(self) -> None:
        body = fetched(RANSAC)
        winner = body["proposals"][body["winner"]]

        assert winner["angle_degrees"] == pytest.approx(
            body["estimate"]["angle_degrees"], abs=0.1
        )
        assert winner["scale"] == pytest.approx(body["estimate"]["scale"], abs=0.01)


class TestThePyramid:
    def test_six_levels_find_three_times_the_keypoints_and_fewer_right_matches(
        self,
    ) -> None:
        body = fetched(PYRAMID)
        rows = {(row["levels"], row["factor"]): row for row in body["rows"]}

        assert rows[(1, 2.0)]["n_first_keypoints"] == 40
        assert rows[(6, 1.2)]["n_first_keypoints"] == 126
        assert rows[(1, 2.0)]["n_right"] == 10
        assert rows[(6, 1.2)]["n_right"] == 6
        assert all(row["recovered"] for row in body["rows"])
