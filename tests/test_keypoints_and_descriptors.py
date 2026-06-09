"""The keypoints and descriptors page's endpoints.

Every number the page quotes is pinned here, on the two pictures the page
works: the bright square whose answers come out round, and the workbench scene
the classical vision pages share. The claims worth reading twice are the ones
that contradict the usual telling, so they are asserted as equalities rather
than approximations: the determinant along a straight edge is exactly zero, the
smaller eigenvalue there is exactly what flat ground gives, and Harris puts an
edge below flat ground rather than above it.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from oop_ml.core.computer_vision.keypoints import Keypoint, PatchDescriptor

from oop_ml_api.concepts.keypoints_and_descriptors import (
    PATCH_SIDE,
    _dense_descriptions,
    _square,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

PREFIX = "/concepts/keypoints-and-descriptors"
DETECT = f"{PREFIX}/detect"
WHY = f"{PREFIX}/why-a-corner"
APERTURE = f"{PREFIX}/aperture"
SUPPRESSION = f"{PREFIX}/suppression"
DESCRIPTOR = f"{PREFIX}/descriptor"
MATCHING = f"{PREFIX}/matching"
SCENE = f"{PREFIX}/scene-report"
WITHOUT = f"{PREFIX}/without-corners"


class TestDetecting:
    def test_the_shared_scene_answers_forty_seven_places_out_of_two_thousand(
        self,
    ) -> None:
        body = client.post(DETECT, json={}).json()

        assert (body["height"], body["width"]) == (48, 48)
        assert body["n_candidates"] == 457
        assert body["n_keypoints"] == 47
        assert body["score_high"] == 1.8392

    def test_the_scores_and_the_picture_are_drawn_on_the_same_grid(self) -> None:
        body = client.post(DETECT, json={}).json()

        assert len(body["picture"]) == body["height"]
        assert len(body["scores"]) == body["height"]
        assert all(len(row) == body["width"] for row in body["scores"])

    def test_the_square_answers_its_four_corners_at_exactly_four(self) -> None:
        body = client.post(
            DETECT, json={"scene": "square", "minimum_strength": 0.5, "border": 0}
        ).json()

        assert [(point["row"], point["column"]) for point in body["keypoints"]] == [
            (6, 6),
            (6, 14),
            (14, 6),
            (14, 14),
        ]
        assert all(point["strength"] == 4.0 for point in body["keypoints"])

    def test_a_picture_with_one_long_edge_answers_nothing_at_all(self) -> None:
        """A horizon has no corner, and the response is exactly zero everywhere
        rather than merely small."""
        body = client.post(
            DETECT, json={"scene": "horizon", "minimum_strength": 0.0}
        ).json()

        assert body["score_high"] == 0.0
        assert body["n_candidates"] == 0
        assert body["n_keypoints"] == 0

    def test_a_shadow_corner_scores_what_an_object_corner_of_the_same_step_scores(
        self,
    ) -> None:
        """The floor carries a pale block and a dark shadow the same distance
        from the ground's brightness. All eight corners read 0.64."""
        body = client.post(
            DETECT, json={"scene": "floor", "minimum_strength": 0.5}
        ).json()

        assert body["n_keypoints"] == 8
        assert all(point["strength"] == 0.64 for point in body["keypoints"])

    def test_turning_the_scene_a_quarter_circle_keeps_every_keypoint(self) -> None:
        body = client.post(DETECT, json={"scene": "turned"}).json()

        assert body["n_keypoints"] == 47

    def test_harris_scores_an_edge_below_zero_where_the_other_measure_cannot(
        self,
    ) -> None:
        body = client.post(
            DETECT, json={"scene": "workbench", "measure": "harris"}
        ).json()

        assert body["score_low"] < 0.0
        assert client.post(DETECT, json={}).json()["score_low"] == 0.0

    def test_an_even_window_has_no_centre_pixel_and_is_refused(self) -> None:
        answer = client.post(DETECT, json={"window_side": 4})

        assert answer.status_code == 422
        assert "centre pixel" in answer.json()["detail"]

    @pytest.mark.parametrize(
        "payload",
        [
            {"measure": "harris", "sensitivity": 0.25},
            {"measure": "harris", "sensitivity": 0.0},
            {"separation": 0},
            {"border": -1},
            {"limit": 0},
            {"scene": "nowhere"},
            {"nonsense": 1},
        ],
    )
    def test_an_impossible_request_is_refused(self, payload: dict) -> None:
        assert client.post(DETECT, json=payload).status_code == 422


class TestWhyACorner:
    def test_the_energy_cannot_tell_a_corner_from_an_edge(self) -> None:
        """The trace at a corner is 8% above the trace at an edge, which is
        nothing, and that is the measured reason the measures read the
        determinant instead."""
        body = client.get(WHY).json()
        corner, edge, flat = body["probes"]

        assert corner["trace"] == pytest.approx(11.555556)
        assert edge["trace"] == pytest.approx(10.666667)
        assert flat["trace"] == 0.0
        assert body["trace_ratio"] == pytest.approx(1.083333)

    def test_the_determinant_along_a_straight_edge_is_exactly_zero(self) -> None:
        body = client.get(WHY).json()
        corner, edge, flat = body["probes"]

        assert corner["determinant"] == pytest.approx(30.222222)
        assert edge["determinant"] == 0.0
        assert flat["determinant"] == 0.0

    def test_the_smaller_eigenvalue_puts_an_edge_exactly_where_flat_ground_is(
        self,
    ) -> None:
        body = client.get(WHY).json()
        corner, edge, flat = body["probes"]

        assert corner["smallest_eigenvalue"] == 4.0
        assert edge["smallest_eigenvalue"] == 0.0
        assert flat["smallest_eigenvalue"] == 0.0

    def test_harris_puts_an_edge_below_flat_ground(self) -> None:
        body = client.get(WHY).json()
        corner, edge, flat = body["probes"]

        assert corner["harris"] == pytest.approx(24.880988)
        assert flat["harris"] == 0.0
        assert edge["harris"] == pytest.approx(-4.551111)

    def test_every_gradient_in_the_window_on_an_edge_points_one_way(self) -> None:
        """Which is why the matrix is rank one there, and the determinant is
        zero by arithmetic rather than by rounding."""
        body = client.get(WHY).json()
        edge = body["probes"][1]

        vectors = [vector for row in edge["gradients"] for vector in row]
        assert all(vector["down"] == 0.0 for vector in vectors)
        assert {vector["across"] for vector in vectors} == {4.0, 0.0}

    def test_a_single_pixel_window_finds_no_corner_anywhere(self) -> None:
        """Without the averaging every matrix is one gradient against itself,
        whose determinant is zero by construction."""
        body = client.get(WHY).json()
        by_side = {step["window_side"]: step for step in body["window_sweep"]}

        assert by_side[1]["brightest"] == 0.0
        assert by_side[1]["n_keypoints"] == 0
        assert by_side[3]["corner_score"] == 4.0
        assert by_side[3]["n_keypoints"] == 4

    def test_a_wider_window_lowers_the_corner_and_finds_more_of_them(self) -> None:
        body = client.get(WHY).json()
        scores = [step["corner_score"] for step in body["window_sweep"]][1:]
        counts = [step["n_keypoints"] for step in body["window_sweep"]][1:]

        assert scores == [4.0, 2.72, 2.040816, 1.62963]
        assert counts == [4, 12, 16, 25]

    def test_near_the_ceiling_harris_finds_nothing_and_the_corner_goes_negative(
        self,
    ) -> None:
        body = client.get(WHY).json()

        assert body["harris_sensitivity"] == 0.24
        assert body["harris_at_high_sensitivity"] == pytest.approx(-1.825185)
        assert body["harris_keypoints_at_high_sensitivity"] == 0


class TestTheAperture:
    def test_an_edge_patch_matches_five_positions_along_that_edge(self) -> None:
        body = client.get(APERTURE).json()

        assert body["edge_matches"] == 5
        assert body["corner_matches"] == 1

    def test_the_five_that_match_are_consecutive_and_the_rest_are_not_close(
        self,
    ) -> None:
        body = client.get(APERTURE).json()
        exact = [
            step["row"] for step in body["steps"] if step["distance_from_edge"] == 0.0
        ]

        assert exact == [8, 9, 10, 11, 12]

    def test_the_corner_matches_itself_and_its_own_far_corner_at_the_known_gap(
        self,
    ) -> None:
        body = client.get(APERTURE).json()
        by_row = {step["row"]: step for step in body["steps"]}

        assert by_row[6]["distance_from_corner"] == 0.0
        assert by_row[14]["distance_from_corner"] == pytest.approx(1.443376)

    def test_the_two_reference_patches_are_the_five_by_five_squares_shown(self) -> None:
        body = client.get(APERTURE).json()

        assert len(body["edge_patch"]) == PATCH_SIDE
        assert body["edge_patch"] == [[0.0, 0.0, 1.0, 1.0, 1.0]] * PATCH_SIDE
        assert body["corner_patch"][0] == [0.0] * PATCH_SIDE
        assert body["corner_patch"][4] == [0.0, 0.0, 1.0, 1.0, 1.0]


class TestSuppressionAndTheBorder:
    def test_one_corner_raises_eight_candidates_and_the_radius_takes_seven(
        self,
    ) -> None:
        body = client.get(SUPPRESSION).json()
        counts = {
            step["separation"]: step["n_keypoints"] for step in body["square_sweep"]
        }

        assert body["square_candidates"] == 32
        assert counts[1] == 32
        assert counts[2] == 4
        assert counts[7] == 4

    def test_the_same_on_the_shared_scene_is_most_of_the_raw_answer(self) -> None:
        body = client.get(SUPPRESSION).json()
        counts = {
            step["separation"]: step["n_keypoints"] for step in body["scene_sweep"]
        }

        assert body["scene_candidates"] == 457
        assert counts[1] == 446
        assert counts[2] == 101
        assert counts[3] == 47
        assert counts[7] == 16

    def test_dropping_the_reported_keypoints_leaves_the_one_corner_inside(self) -> None:
        body = client.get(SUPPRESSION).json()

        assert len(body["without_border"]) == 4
        assert [(point["row"], point["column"]) for point in body["in_order"]] == [
            (10, 10)
        ]

    def test_dropping_the_candidates_first_promotes_the_pixel_beside_a_corner(
        self,
    ) -> None:
        """The right count, plausible strengths, and every position wrong by
        one pixel, which is why the two orders are worth telling apart."""
        body = client.get(SUPPRESSION).json()
        wrong = [(point["row"], point["column"]) for point in body["out_of_order"]]

        assert wrong == [(10, 10), (3, 3), (3, 9), (9, 3)]
        assert body["shoulder_score"] == pytest.approx(3.555556)
        assert body["corner_score"] == 4.0
        assert len(body["out_of_order"]) == len(body["without_border"])


class TestTheDescriptor:
    def test_the_patch_is_taken_all_the_way_by_arithmetic_a_reader_can_check(
        self,
    ) -> None:
        body = client.get(DESCRIPTOR).json()

        assert body["patch_mean"] == 0.36
        assert body["centred_length"] == 2.4
        assert body["centred"][0] == [-0.36] * PATCH_SIDE
        assert body["unit"][0] == [-0.15] * PATCH_SIDE
        assert body["unit"][4][4] == pytest.approx(0.266667)

    def test_lighting_moves_the_description_by_rounding_and_nothing_more(self) -> None:
        body = client.get(DESCRIPTOR).json()
        distances = {item["name"]: item["distance"] for item in body["comparisons"]}

        assert distances["the same corner, ten added to every pixel"] < 2e-15
        assert distances["the same corner, every pixel multiplied by three"] < 1e-15
        assert distances["the same corner, ten added to every pixel"] > 0.0

    def test_a_shift_leaves_the_description_bit_identical(self) -> None:
        body = client.get(DESCRIPTOR).json()
        distances = {item["name"]: item["distance"] for item in body["comparisons"]}

        assert (
            distances["the same corner, the picture shifted down two and right three"]
            == 0.0
        )
        assert body["shifted_strengths_agree"]
        assert body["shifted_positions_agree"]

    def test_two_flat_patches_are_zero_apart_however_far_apart_they_are(self) -> None:
        body = client.get(DESCRIPTOR).json()
        distances = {item["name"]: item["distance"] for item in body["comparisons"]}

        assert (
            distances["flat ground outside the square, against flat ground inside it"]
            == 0.0
        )

    def test_the_scale_runs_to_two_and_a_reversed_picture_reaches_it(self) -> None:
        body = client.get(DESCRIPTOR).json()
        distances = {item["name"]: item["distance"] for item in body["comparisons"]}

        assert distances[
            "the same corner in the picture with its brightness reversed"
        ] == pytest.approx(2.0)
        assert distances["the top-right corner of the same square"] == pytest.approx(
            1.443376
        )
        assert distances["the bottom-right corner of the same square"] == pytest.approx(
            1.666667
        )

    def test_a_quarter_turn_leaves_each_corner_where_another_corner_was(self) -> None:
        """The gap is the same 1.443376 that separates a square's top-left
        corner from its top-right one, which is the honest reading of it."""
        body = client.get(DESCRIPTOR).json()

        assert body["turned_gaps"] == [pytest.approx(1.443376)] * 4


class TestTheRatioTest:
    def test_a_genuine_match_wins_at_zero_against_a_runner_up_far_behind(self) -> None:
        body = client.get(MATCHING).json()
        scene = body["scenes"][0]

        assert scene["n_kept"] == 4
        assert all(match["distance"] == 0.0 for match in scene["matches"])
        assert all(
            match["runner_up_distance"] == pytest.approx(1.443376)
            for match in scene["matches"]
        )
        assert all(match["ratio"] == 0.0 for match in scene["matches"])

    def test_and_every_winner_is_the_corner_that_moved_by_the_known_shift(self) -> None:
        body = client.get(MATCHING).json()
        scene = body["scenes"][0]

        assert all(
            (match["to_row"], match["to_column"])
            == (match["from_row"] + 4, match["from_column"] + 5)
            for match in scene["matches"]
        )

    def test_two_equally_good_answers_report_one_and_are_refused(self) -> None:
        body = client.get(MATCHING).json()
        scene = body["scenes"][1]

        assert scene["n_kept"] == 0
        assert all(match["distance"] == 0.0 for match in scene["matches"])
        assert all(match["runner_up_distance"] == 0.0 for match in scene["matches"])
        assert all(match["ratio"] == 1.0 for match in scene["matches"])

    def test_the_same_coincidence_with_the_tie_broken_computes_its_ratio(self) -> None:
        body = client.get(MATCHING).json()
        scene = body["scenes"][2]
        first = scene["matches"][0]

        assert first["distance"] == pytest.approx(0.050116)
        assert first["runner_up_distance"] == pytest.approx(0.0511)
        assert first["ratio"] == pytest.approx(0.980729)
        assert scene["n_kept"] == 0
        assert body["maximum_ratio"] == 0.8


class TestTheSharedSceneEndToEnd:
    def test_relighting_multiplies_every_score_by_the_square_of_the_lighting(
        self,
    ) -> None:
        body = client.get(SCENE).json()

        assert body["lighting_scale"] == 1.6
        assert body["score_ratio"] == pytest.approx(2.56)
        assert body["relit_score_high"] == pytest.approx(4.708352)

    def test_and_leaves_the_descriptions_where_they_were(self) -> None:
        body = client.get(SCENE).json()

        assert body["relit_descriptor_gap"] < 2e-15
        assert body["relit_descriptor_gap"] > 0.0

    def test_a_quarter_turn_puts_every_keypoint_where_the_map_says(self) -> None:
        body = client.get(SCENE).json()

        assert body["turned_positions_agree"]
        assert body["n_turned_keypoints"] == body["n_keypoints"] == 47

    def test_but_no_description_survives_that_turn(self) -> None:
        body = client.get(SCENE).json()

        assert body["turned_gap_low"] > 0.05
        assert body["turned_gap_high"] == pytest.approx(1.545328)

    def test_the_ratio_test_refuses_the_repeated_motif_and_keeps_the_rest(self) -> None:
        body = client.get(SCENE).json()

        assert body["n_matches"] == 47
        assert body["n_kept"] == 32
        assert body["n_refused"] == 15
        assert body["n_refused_on_the_repeated_motif"] == 15
        assert body["refused_ratio_low"] == pytest.approx(0.824163)

    def test_six_of_the_kept_matches_are_a_ratio_of_two_rounding_errors(self) -> None:
        """Every distance in this run is at rounding level, so a tie the test
        was meant to refuse is broken below the last bit of a number and six
        coincidences pass at ratios well under the threshold."""
        body = client.get(SCENE).json()

        assert body["n_on_the_same_position"] == 26
        assert len(body["strays"]) == 6
        assert all(stray["distance"] < 1e-15 for stray in body["strays"])
        assert all(stray["runner_up_distance"] < 1e-15 for stray in body["strays"])
        assert min(stray["ratio"] for stray in body["strays"]) == pytest.approx(
            0.572598, abs=1e-6
        )
        assert max(stray["ratio"] for stray in body["strays"]) < 0.8

    def test_most_positions_of_the_scene_cannot_say_where_they_came_from(self) -> None:
        body = client.get(SCENE).json()

        assert body["n_positions"] == 1936
        assert body["distinct_descriptions"] == 815
        assert body["n_sharing_a_description"] == 1179
        assert body["largest_identical_group"] == 86

    def test_the_cost_of_describing_everything_against_describing_the_few(
        self,
    ) -> None:
        body = client.get(SCENE).json()

        assert body["dense_numbers"] == 48400
        assert body["sparse_numbers"] == 1175
        assert body["dense_pairs"] == 3748096
        assert body["sparse_pairs"] == 2209
        assert body["dense_seconds"] > body["sparse_seconds"]

    def test_the_dense_route_describes_a_position_the_way_the_library_does(
        self,
    ) -> None:
        """The comparison is only worth anything if both sides compute the same
        description, so the one written here is checked against the library's at
        a keypoint, to the last bit."""
        picture = _square()
        keypoint = Keypoint(6, 6, 4.0)
        half = PATCH_SIDE // 2
        width = picture.width - PATCH_SIDE + 1
        row = (keypoint.row - half) * width + (keypoint.column - half)

        theirs = np.asarray(PatchDescriptor.of(picture, keypoint, PATCH_SIDE))
        mine = _dense_descriptions(picture)[row]

        assert np.array_equal(mine, theirs)


class TestWhereItIsWorseThanLookingAtEverything:
    def test_a_horizon_answers_nothing_at_all(self) -> None:
        body = client.get(WITHOUT).json()

        assert body["score_high"] == 0.0
        assert body["n_keypoints"] == 0

    def test_while_describing_every_position_recovers_the_move_exactly(self) -> None:
        """Of the 289 positions a patch fits at, 221 are flat and alike; every
        one of the 68 straddling the edge finds its partner three rows lower, at
        a distance of exactly zero."""
        body = client.get(WITHOUT).json()

        assert body["n_positions"] == 289
        assert body["n_flat_positions"] == 221
        assert body["n_informative_positions"] == 68
        assert body["n_recovering_the_shift"] == 68
        assert body["furthest_true_partner"] == 0.0
        assert body["shift_rows"] == 3


class TestNothingNamesTheCodeBehindThePage:
    """Every string a widget can render is page text, so none of it may name a
    class, an attribute or a module."""

    FORBIDDEN = ("oop_ml", "Descriptor", "Keypoint(", "CornerResponse", "numpy")

    def _strings(self, value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [text for item in value.values() for text in self._strings(item)]
        if isinstance(value, list):
            return [text for item in value for text in self._strings(item)]
        return []

    @pytest.mark.parametrize(
        "path", [WHY, APERTURE, SUPPRESSION, DESCRIPTOR, MATCHING, SCENE, WITHOUT]
    )
    def test_no_identifier_reaches_the_page(self, path: str) -> None:
        for text in self._strings(client.get(path).json()):
            for forbidden in self.FORBIDDEN:
                assert forbidden not in text, text
