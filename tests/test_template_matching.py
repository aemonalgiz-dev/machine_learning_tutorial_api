"""The template matching page's endpoints, on the shared workbench scene.

Two kinds of claim, kept apart for the reason the library's own spec keeps them
apart. The arithmetic on the six by six picture can be checked with a pencil and
is asserted exactly. The scores on the forty-eight by forty-eight scene cannot
be, and are pinned to the digit anyway, because the page quotes them and a
change in any of them is a change to what the page says.

The load-bearing test is ``TestTheRampIsWhatSeparatesTheRules``. Every other
test here would pass for a service that had only one scoring rule in it.
"""

from __future__ import annotations

import math
import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/template-matching"
SEARCH = f"{BASE}/search"
HAND_SIZED = f"{BASE}/hand-sized"
CHANGED = f"{BASE}/changed-thing"
BELIEVABILITY = f"{BASE}/believability"
COST = f"{BASE}/cost"
UNDEFINED = f"{BASE}/undefined"
RAMP = f"{BASE}/ramp"

SQUARED, CORRELATION, NORMALISED = 0, 1, 2
"""The three rules, in the order every response lists them."""


def searched(scene: str = "workbench") -> dict:
    response = client.post(SEARCH, json={"scene": scene})
    assert response.status_code == 200
    return response.json()


def rule(body: dict, which: int) -> dict:
    return body["results"][which]


class TestTheSceneAndTheTemplate:
    def test_the_scene_is_forty_eight_square_and_the_motif_seven(self) -> None:
        body = searched()

        assert (body["scene"]["height"], body["scene"]["width"]) == (48, 48)
        assert (body["template"]["height"], body["template"]["width"]) == (7, 7)
        assert body["motif_positions"] == [[26, 26], [38, 12], [39, 36]]

    def test_every_grid_says_what_black_and_white_stand_for(self) -> None:
        """A greyscale drawing is uninterpretable without the two ends of its
        scale, and the scene, the template and each surface run over different
        ranges."""
        body = searched()
        grids = [body["scene"], body["template"]] + [
            result["surface"] for result in body["results"]
        ]

        for grid in grids:
            assert grid["darkest"] < grid["brightest"]
            assert len(grid["rows"]) == grid["height"]
            assert all(len(row) == grid["width"] for row in grid["rows"])

    def test_the_surface_is_smaller_than_the_scene_by_the_template(self) -> None:
        """Only positions where the template fits entirely are scored, so the
        surface is 48 - 7 + 1 on each side."""
        body = searched()

        for result in body["results"]:
            assert (result["surface"]["height"], result["surface"]["width"]) == (42, 42)

    def test_the_three_lightings_are_the_same_drawing(self) -> None:
        """The motif sits in the same three places whatever the lamp is doing,
        which is what makes a difference in the answer attributable."""
        for scene in ("workbench", "unlit", "relit"):
            assert searched(scene)["motif_positions"] == [[26, 26], [38, 12], [39, 36]]

    def test_a_scene_nobody_drew_is_refused(self) -> None:
        assert client.post(SEARCH, json={"scene": "moonlight"}).status_code == 422

    def test_a_misspelled_keyword_is_refused_rather_than_ignored(self) -> None:
        assert (
            client.post(SEARCH, json={"scene": "workbench", "rule": "any"}).status_code
            == 422
        )


class TestTheRampIsWhatSeparatesTheRules:
    """The load-bearing tests. One scene, three rules, three different winners,
    and only one of them a place the thing being looked for actually is."""

    def test_the_squared_differences_pick_the_least_brightly_lit_copy(self) -> None:
        """(38, 12) is the leftmost of the three copies and so the one the ramp
        has added least to, which is why it is nearest the unlit template."""
        result = rule(searched(), SQUARED)

        assert (result["best"]["row"], result["best"]["column"]) == (38, 12)
        assert result["best"]["score"] == pytest.approx(0.4571706655)
        assert [
            position["score"] for position in result["at_the_motifs"]
        ] == pytest.approx([1.6869397918, 0.4571706655, 3.044477139])
        assert result["higher_is_better"] is False

    def test_plain_correlation_picks_the_flat_bright_inside_of_the_disc(self) -> None:
        """(9, 33) holds no cross at all. It is a patch of the disc's flat
        interior on the bright side of the ramp, and it beats every one of the
        three real copies, the brightest of which reaches 27.1932."""
        result = rule(searched(), CORRELATION)

        assert (result["best"]["row"], result["best"]["column"]) == (9, 33)
        assert result["best"]["score"] == pytest.approx(27.9307148936)
        assert [
            position["score"] for position in result["at_the_motifs"]
        ] == pytest.approx([25.4276425532, 22.9558978723, 27.1931744681])
        assert result["best"]["score"] > max(
            position["score"] for position in result["at_the_motifs"]
        )

    def test_and_it_ranks_the_three_copies_by_how_brightly_they_are_lit(self) -> None:
        """The three are the same drawing, so a rule that ranked them at all is
        ranking the lamp. Left to right they are 22.9559, 25.4276 and 27.1932."""
        result = rule(searched(), CORRELATION)
        scores = [position["score"] for position in result["at_the_motifs"]]

        assert scores[1] < scores[0] < scores[2]

    def test_the_normalised_rule_picks_a_real_copy(self) -> None:
        result = rule(searched(), NORMALISED)

        assert [result["best"]["row"], result["best"]["column"]] in searched()[
            "motif_positions"
        ]
        assert result["best"]["score"] == pytest.approx(0.9991504412)

    def test_and_scores_all_three_copies_exactly_alike(self) -> None:
        """The same drawing under three different lamps is the same deviation
        pattern three times, so the cosine is the same number three times."""
        result = rule(searched(), NORMALISED)
        scores = [position["score"] for position in result["at_the_motifs"]]

        assert len(set(scores)) == 1

    def test_the_flat_bright_patch_that_won_the_multiplication_scores_nothing(
        self,
    ) -> None:
        """Correlation's winner has no deviation pattern the cross agrees with,
        so the rule that looks at deviation gives it essentially zero."""
        body = searched()
        surface = rule(body, NORMALISED)["surface"]["rows"]

        assert abs(surface[9][33]) < 1e-9

    def test_taking_the_ramp_away_makes_the_squared_differences_exact(self) -> None:
        """Without the lamp the three copies are the template, pixel for pixel,
        so the literal reading of the distance between two pictures is zero at
        all three."""
        result = rule(searched("unlit"), SQUARED)

        assert [position["score"] for position in result["at_the_motifs"]] == [
            0.0,
            0.0,
            0.0,
        ]

    def test_but_correlation_still_prefers_a_shape_that_is_not_the_motif(self) -> None:
        """The square's corner scores 21.5748 where an exact copy of the
        template scores 20.3076, so the rule can be beaten by something bigger
        and brighter even with the ramp gone."""
        result = rule(searched("unlit"), CORRELATION)

        assert (result["best"]["row"], result["best"]["column"]) == (6, 5)
        assert result["best"]["score"] == pytest.approx(21.5748)
        assert result["perfect_score"] == pytest.approx(20.3076)
        assert result["best"]["score"] > result["perfect_score"]

    def test_turning_the_lamp_up_moves_two_of_the_three_answers(self) -> None:
        """Brightened, the squared differences answer a patch of empty ground
        and score the three real copies between 21.899 and 40.2612, where the
        winner scores 4.7145."""
        result = rule(searched("relit"), SQUARED)

        assert (result["best"]["row"], result["best"]["column"]) == (0, 19)
        assert result["best"]["score"] == pytest.approx(4.7144903957)
        assert min(
            position["score"] for position in result["at_the_motifs"]
        ) == pytest.approx(21.8990397121)

    def test_and_leaves_the_normalised_answer_where_it_was(self) -> None:
        body = searched("relit")
        plain = searched("workbench")

        assert rule(body, NORMALISED)["best"] == rule(plain, NORMALISED)["best"]
        assert [
            position["score"] for position in rule(body, NORMALISED)["at_the_motifs"]
        ] == pytest.approx(
            [position["score"] for position in rule(plain, NORMALISED)["at_the_motifs"]]
        )


class TestTheRampItself:
    def test_the_three_copies_differ_in_brightness_and_not_in_pattern(self) -> None:
        body = client.get(RAMP).json()

        assert body["patch_means"] == pytest.approx(
            [0.7495961789, 0.6602344768, 0.8134259661]
        )
        assert body["scores_are_identical"] is True

    def test_the_ramp_inside_a_patch_is_perpendicular_to_the_centred_motif(
        self,
    ) -> None:
        """Which is why it does not change the numerator at all. The motif is
        the same left to right and a straight ramp is its own negative left to
        right, so the two multiply to nothing."""
        body = client.get(RAMP).json()

        assert abs(body["ramp_dotted_with_template"]) < 1e-15

    def test_so_the_score_falls_below_one_by_exactly_the_length_it_added(
        self,
    ) -> None:
        """A perpendicular addition leaves the agreement alone and lengthens the
        patch, so the cosine is one over the square root of one plus the ratio
        of the two lengths squared, which is where 0.99915 comes from."""
        body = client.get(RAMP).json()

        assert body["predicted_cosine"] == pytest.approx(
            body["normalised_scores"][0], abs=1e-15
        )
        assert body["normalised_scores"][0] < 1.0

    def test_the_position_that_beat_the_copies_is_flat_and_bright(self) -> None:
        """Row 9, column 33 lies wholly inside the disc, so its pixels run only
        from 0.9906 to 1.0289, which is the ramp and nothing else."""
        body = client.get(RAMP).json()

        assert body["decoy_at"] == [9, 33]
        assert body["decoy_darkest"] == pytest.approx(0.9906382979)
        assert body["decoy_brightest"] == pytest.approx(1.0289361702)
        assert body["decoy_mean"] == pytest.approx(1.009787234)

    def test_the_multiplication_read_that_patch_s_brightness_and_nothing_else(
        self,
    ) -> None:
        """A flat patch of brightness b scores b times the sum of the template's
        pixels, and here the two agree to the last bit, so the rule extracted no
        information about shape at all."""
        body = client.get(RAMP).json()

        assert body["template_sum"] == pytest.approx(27.66)
        assert body["decoy_correlation_if_it_were_flat"] == body["decoy_correlation"]
        assert body["decoy_correlation"] == pytest.approx(27.9307148936)

    def test_while_a_genuine_copy_is_its_own_squares_plus_the_light_on_it(
        self,
    ) -> None:
        """20.3076 from the cross agreeing with itself and 5.1200 from the ramp,
        which is less than a patch averaging 1.0098 delivers."""
        body = client.get(RAMP).json()

        assert body["template_sum_of_squares"] == pytest.approx(20.3076)
        assert body["middle_copy_correlation"] == pytest.approx(25.4276425532)
        assert body["middle_copy_correlation"] - body[
            "template_sum_of_squares"
        ] == pytest.approx(5.1200425532)

    def test_relighting_moves_no_score_by_more_than_rounding(self) -> None:
        """Every pixel multiplied by 1.6 and shifted by 0.15, over all 1,764
        positions, and the largest change to any score is 1.7e-15."""
        body = client.get(RAMP).json()

        assert body["largest_score_change_when_relit"] < 1e-14


class TestTheHandSizedSearch:
    def test_sixteen_positions_and_a_hundred_and_forty_four_pixel_reads(self) -> None:
        body = client.get(HAND_SIZED).json()

        assert body["n_positions"] == 16
        assert body["n_pixels_read"] == 144
        assert len(body["cells"]) == 16

    def test_the_first_row_of_scores_can_be_checked_with_a_pencil(self) -> None:
        """Four positions. The squared differences run 3, 0, 3, 6; correlation
        runs 0, 3, 3, 3; the normalised rule runs 0, 1, 0.5, 0."""
        body = client.get(HAND_SIZED).json()
        first = [cell for cell in body["cells"] if cell["row"] == 0]

        assert [cell["scores"][SQUARED] for cell in first] == [3.0, 0.0, 3.0, 6.0]
        assert [cell["scores"][CORRELATION] for cell in first] == [0.0, 3.0, 3.0, 3.0]
        assert [cell["scores"][NORMALISED] for cell in first] == [0.0, 1.0, 0.5, 0.0]

    def test_every_row_of_the_picture_answers_the_same_four_numbers(self) -> None:
        """The picture is the same six pixels repeated down, so a reader who has
        checked one row has checked all four."""
        body = client.get(HAND_SIZED).json()

        by_row = [
            [cell["scores"] for cell in body["cells"] if cell["row"] == row]
            for row in range(4)
        ]
        assert by_row[0] == by_row[1] == by_row[2] == by_row[3]

    def test_correlation_cannot_separate_the_copy_from_two_wrong_answers(self) -> None:
        """The heart of the example. The exact copy, a patch holding half the
        edge and a patch of flat brightness all score exactly 3, so the rule
        that multiplies has no preference among them at all and the position it
        reports is decided by which was read first."""
        body = client.get(HAND_SIZED).json()

        assert [
            (position["column"], position["score"])
            for position in body["correlation_ties"]
        ] == [(1, 3.0), (2, 3.0), (3, 3.0)]
        assert body["best"][CORRELATION]["column"] == 1

    def test_while_the_other_two_rules_separate_them_cleanly(self) -> None:
        body = client.get(HAND_SIZED).json()
        first = [cell for cell in body["cells"] if cell["row"] == 0]

        assert len({cell["scores"][SQUARED] for cell in first}) == 3
        assert len({cell["scores"][NORMALISED] for cell in first}) == 3

    def test_the_perfect_score_of_each_rule_is_stated(self) -> None:
        """Two are constants and the third is not, because correlation has no
        scale of its own: an exact copy scores the template's own sum of
        squares, which here is 3."""
        body = client.get(HAND_SIZED).json()

        assert body["perfect_scores"] == [0.0, 3.0, 1.0]


class TestTheThingChanged:
    def test_an_unaltered_copy_is_the_control(self) -> None:
        body = client.get(CHANGED).json()
        control = body["cases"][0]

        assert control["change"] == "Unaltered"
        assert control["at_the_truth"]["score"] == pytest.approx(1.0)
        assert control["pixels_away"] == 0.0

    @pytest.mark.parametrize(
        "index, at_the_truth, best_score, pixels_away",
        [
            (1, 0.0384615385, 0.5204164999, 4.0),
            (2, -0.1217948718, 0.6590820437, 3.1622776602),
            (3, 0.0384615385, 0.5204164999, 4.0),
            (4, 0.1649711326, 0.7806247498, 5.0990195136),
        ],
    )
    def test_a_turned_or_enlarged_copy_is_not_found_and_a_wrong_answer_is_given(
        self, index: int, at_the_truth: float, best_score: float, pixels_away: float
    ) -> None:
        """The score at the position the thing occupies falls to near nothing,
        and below nothing at half a turn, while the position reported instead is
        several pixels away and scores high enough to look like something."""
        case = client.get(CHANGED).json()["cases"][index]

        assert case["at_the_truth"]["score"] == pytest.approx(at_the_truth)
        assert case["best"]["score"] == pytest.approx(best_score)
        assert case["pixels_away"] == pytest.approx(pixels_away)

    def test_a_doubled_l_holds_an_exact_unscaled_copy_of_itself(self) -> None:
        """Doubling an L puts the original L back, pixel for pixel, at the join
        of the two arms."""
        body = client.get(CHANGED).json()

        assert body["doubled_corner_contains_the_original"] is True
        assert body["doubled_corner_copy_at"] == [2, 1]

    def test_so_failing_on_size_can_be_a_perfect_score_in_the_wrong_place(
        self,
    ) -> None:
        """The thing starts at row 3, column 3, and scores 0.3162 there. The
        answer given is row 5, column 4, at a flawless 1.0."""
        corner = client.get(CHANGED).json()["doubled_corner"]

        assert corner["at_the_truth"]["score"] == pytest.approx(0.316227766)
        assert (corner["best"]["row"], corner["best"]["column"]) == (5, 4)
        assert corner["best"]["score"] == 1.0

    def test_and_the_confidence_check_endorses_that_wrong_answer(self) -> None:
        """The sharpest version of the finding. The winner stands 2.0917 clear
        of anything else, which passes the threshold, so nothing anywhere says
        the object is not where it was reported."""
        body = client.get(CHANGED).json()

        assert body["doubled_corner_ratio"] == pytest.approx(2.0916500663)
        assert body["doubled_corner_believable"] is True


class TestBelievability:
    def test_the_two_ranges_do_not_touch(self) -> None:
        """Absent runs 1.0179 to 1.3654 and present 1.8722 to 2.5797, over six
        textures with the same glyph drawn in or left out."""
        body = client.get(BELIEVABILITY).json()

        assert body["absent_low"] == pytest.approx(1.0178536684)
        assert body["absent_high"] == pytest.approx(1.3654107102)
        assert body["present_low"] == pytest.approx(1.8721764059)
        assert body["present_high"] == pytest.approx(2.579681204)
        assert body["absent_high"] < body["threshold"] < body["present_low"]

    def test_the_scores_alone_come_nowhere_near_separating_the_two_cases(self) -> None:
        """A picture containing nothing reaches 0.5341, which is high enough to
        look like something to anyone who was hoping."""
        body = client.get(BELIEVABILITY).json()

        assert body["absent_score_high"] == pytest.approx(0.5341377003)
        assert all(
            seed["present_best_score"] == pytest.approx(1.0) for seed in body["seeds"]
        )

    def test_a_three_by_three_template_defeats_the_ratio(self) -> None:
        """Not the ratio's fault. Nine pixels are matched by chance at 0.7752 on
        average in a picture that contains nothing, so four of six genuine
        matches fail to stand clear."""
        body = client.get(BELIEVABILITY).json()
        small = next(row for row in body["sizes"] if row["side"] == 3)

        assert small["mean_chance_agreement"] == pytest.approx(0.7752271896)
        assert small["highest_chance_agreement"] == pytest.approx(0.8843952055)
        assert small["present_below_threshold"] == 4

    def test_where_twenty_five_pixels_are_matched_by_chance_much_less_often(
        self,
    ) -> None:
        body = client.get(BELIEVABILITY).json()
        medium = next(row for row in body["sizes"] if row["side"] == 5)

        assert medium["mean_chance_agreement"] == pytest.approx(0.5046505237)
        assert medium["present_below_threshold"] == 0

    def test_chance_agreement_falls_with_every_pixel_added(self) -> None:
        body = client.get(BELIEVABILITY).json()
        means = [row["mean_chance_agreement"] for row in body["sizes"]]

        assert means[0] > means[1] > means[2]

    def test_a_template_too_large_for_its_picture_has_no_second_candidate(self) -> None:
        """Seven by seven in fifteen by fifteen leaves a nine by nine surface,
        and no position in it is seven away from the middle, so there is nothing
        to compare the winner against and the question cannot be asked."""
        body = client.get(BELIEVABILITY).json()
        large = next(row for row in body["sizes"] if row["side"] == 7)

        assert large["has_a_second_candidate"] is False
        assert large["present_low"] is None
        assert large["mean_chance_agreement"] == pytest.approx(0.3326493412)

    def test_the_workbench_is_reported_as_not_believable_because_it_holds_three(
        self,
    ) -> None:
        """Honest rather than wrong. The question the ratio asks is whether the
        winner is the only good position, and with three identical copies the
        answer is no, so the ratio is exactly one."""
        result = rule(searched(), NORMALISED)

        assert result["believability"]["ratio"] == 1.0
        assert result["believability"]["believable"] is False
        assert result["believability"]["runner_up"]["score"] == pytest.approx(
            result["best"]["score"]
        )

    def test_while_the_rule_that_chose_worse_reports_more_confidence(self) -> None:
        """Which is where the recommended rule is worse than the one it
        replaced. The squared differences chose one of the three copies and
        stand 3.69 clear, so on this scene the better rule gives the less usable
        confidence answer."""
        body = searched()

        assert rule(body, SQUARED)["believability"]["ratio"] == pytest.approx(
            3.6899563319
        )
        assert rule(body, SQUARED)["believability"]["believable"] is True
        assert rule(body, NORMALISED)["believability"]["believable"] is False


class TestWhatItCosts:
    def test_the_two_small_searches_report_what_they_actually_did(self) -> None:
        body = client.get(COST).json()
        small, scene = body["rows"][0], body["rows"][1]

        assert (small["n_positions"], small["n_pixels_read"]) == (16, 144)
        assert (scene["n_positions"], scene["n_pixels_read"]) == (1764, 86436)

    def test_the_count_is_every_position_the_template_fits_at(self) -> None:
        body = client.get(COST).json()

        for row in body["rows"]:
            side = row["picture_side"] - row["template_side"] + 1
            assert row["n_positions"] == side * side
            assert (
                row["n_pixels_read"] == row["n_positions"] * row["template_side"] ** 2
            )

    def test_a_realistic_size_reads_eight_hundred_million_pixels(self) -> None:
        """Sixty-four by sixty-four on five hundred and twelve by five hundred
        and twelve, for one object at one angle at one size."""
        body = client.get(COST).json()
        realistic = body["largest_row"]

        assert realistic["n_positions"] == 201601
        assert realistic["n_pixels_read"] == 825757696

    def test_and_searching_angles_and_sizes_multiplies_that(self) -> None:
        body = client.get(COST).json()

        assert body["n_angles"] * body["n_scales"] == 60
        assert body["n_pixels_read_over_angles_and_scales"] == 49545461760


class TestWhereTheQuotientStopsMeaningAnything:
    def test_a_flat_patch_of_twenty_five_pixels_comes_back_exactly_zero(self) -> None:
        """Twenty-five copies of 0.12 average to 0.12 exactly, so the deviations
        are exactly nothing and the stated convention applies without rounding."""
        body = client.get(UNDEFINED).json()

        assert body["flat_patch_at_25_pixels"] == 0.0

    def test_but_nine_copies_of_the_same_number_do_not_average_to_it_exactly(
        self,
    ) -> None:
        """So that quotient is rounding divided by rounding. It stays tiny only
        because what is left over is a constant, and a constant is perpendicular
        to anything centred."""
        body = client.get(UNDEFINED).json()

        assert body["flat_patch_at_9_pixels"] != 0.0
        assert abs(body["flat_patch_at_9_pixels"]) < 1e-15

    def test_a_patch_that_genuinely_varies_by_nothing_much_is_scored_confidently(
        self,
    ) -> None:
        """The case with no protection. A patch whose brightest and darkest
        differ by 1.4e-17 has a real direction, so the quotient is a real
        cosine, and 0.7454 is reported for a pattern no instrument could
        measure."""
        body = client.get(UNDEFINED).json()

        assert body["barely_varying_patch_spread"] < 1e-16
        assert body["barely_varying_patch_score"] == pytest.approx(0.7453559925)

    def test_the_geometry_that_leaves_no_second_candidate(self) -> None:
        """Seven by seven in fifteen by fifteen leaves a nine by nine surface,
        and the winner of that search lies at most 4 rows or columns from any
        other position, where two patches have to be 7 apart before they stop
        overlapping. So every candidate is the winner shifted, and there is
        nothing to set it against."""
        body = client.get(UNDEFINED).json()

        assert body["no_second_candidate_surface_side"] == (
            body["no_second_candidate_picture_side"]
            - body["no_second_candidate_template_side"]
            + 1
        )
        assert body["furthest_reach"] == 4
        assert body["furthest_reach"] < body["no_second_candidate_template_side"]
        assert body["second_candidate_exists"] is False


class TestNothingOnThePageNamesTheCodeBehindIt:
    """A reader has no reason to meet an identifier, so no string any of these
    endpoints returns may carry one."""

    IDENTIFIER = re.compile(r"[a-z]+_[a-z]+|[a-z][A-Z]|\b[A-Z][a-z]+[A-Z]")

    def _every_string(self, value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [
                text for key, item in value.items() for text in self._every_string(item)
            ]
        if isinstance(value, list):
            return [text for item in value for text in self._every_string(item)]
        return []

    def test_no_response_string_carries_an_identifier(self) -> None:
        bodies = [searched(scene) for scene in ("workbench", "unlit", "relit")]
        bodies += [
            client.get(path).json()
            for path in (HAND_SIZED, CHANGED, BELIEVABILITY, COST, UNDEFINED, RAMP)
        ]

        for body in bodies:
            for text in self._every_string(body):
                assert not self.IDENTIFIER.search(text), text

    def test_the_rules_are_named_the_way_a_textbook_names_them(self) -> None:
        body = client.get(HAND_SIZED).json()

        assert body["rules"] == [
            "Sum of squared differences",
            "Correlation",
            "Normalised cross-correlation",
        ]

    def test_no_response_number_is_a_shape_a_page_cannot_print(self) -> None:
        """A ratio with no second candidate to divide by is the absence of a
        competitor rather than a large number, and travels as a flag."""
        bodies = [searched(scene) for scene in ("workbench", "unlit", "relit")]

        for body in bodies:
            for result in body["results"]:
                believability = result["believability"]
                assert believability is not None
                assert math.isfinite(believability["ratio"])
                assert believability["is_infinite"] is False
