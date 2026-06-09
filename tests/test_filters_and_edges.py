"""The filters-and-edges endpoints, pinned to the numbers the page quotes.

The page works the shared six by six picture, whose rows are all
``0 0 0 1 1 1``, so its one edge sits between columns two and three and every
answer below can be checked on paper. A central difference answers 0.5 at both
columns two and three; Sobel answers 4 there, which is the sum of its positive
weights, since each of them meets a one and each negative weight meets a zero.
Reversing the weights, which is the whole difference between correlation and
convolution, turns both answers negative and leaves everything else alone.

At the border the four rules part company exactly where they should. Repeating
the edge pixel invents nothing at either end. Treating the outside as black
invents nothing at the dark left border and a full step at the bright right one.
Reading from the opposite side invents a step at both. Answering only where the
weights fit answers a four by four picture instead of a six by six one.

The page's central claim is measured on the shared scene against the shared
relighting. A brightness threshold of 0.6 labels all 2304 pixels of the scene
correctly and disagrees with itself about 877 of them once the lamp is turned
up, while every gradient magnitude is multiplied by the same 1.6 and no
direction moves by more than 1.1e-14 radians.
"""

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.filters_and_edges import (
    BRIGHTNESS_THRESHOLD,
    EDGE_RULE_LABELS,
    NOISE_LEVEL,
    OPERATOR_LABELS,
    SCENE_LABELS,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/filters-and-edges"
SCENE = f"{BASE}/scene"
FIELD = f"{BASE}/field"
HAND_SWEEP = f"{BASE}/hand-sweep"
BORDERS = f"{BASE}/borders"
OPERATORS = f"{BASE}/operators"
THRESHOLDS = f"{BASE}/thresholds"

HAND_ROW = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
"""Every row of the shared six by six picture."""

POSITIVE_WEIGHT_TOTALS = {
    "central_difference": 0.5,
    "prewitt": 3.0,
    "sobel": 4.0,
    "scharr": 16.0,
}
"""What each operator answers on a clean step, which is that same total."""


def get(path: str) -> dict:
    response = client.get(path)
    assert response.status_code == 200
    return response.json()


def post(path: str, body: dict) -> dict:
    response = client.post(path, json=body)
    assert response.status_code == 200
    return response.json()


class TestTheScene:
    def test_all_three_pictures_are_forty_eight_square(self):
        scene = get(SCENE)

        assert scene["height"] == 48
        assert scene["width"] == 48
        for name in ("scene", "without_ramp", "relit"):
            assert len(scene[name]) == 48
            assert all(len(row) == 48 for row in scene[name])

    def test_the_ramp_is_what_separates_the_two_scenes(self):
        scene = get(SCENE)

        assert scene["ramp_across"] == pytest.approx(0.30)
        assert scene["ramp_per_column"] == pytest.approx(0.30 / 47)
        assert scene["without_ramp"][0][0] == pytest.approx(0.12)
        assert scene["scene"][0][47] == pytest.approx(0.42)

    def test_one_threshold_labels_the_scene_perfectly(self):
        """Every value between the brightest ground and the darkest shape does."""
        lighting = get(SCENE)["lighting"]

        assert lighting["brightest_ground"] == pytest.approx(0.42)
        assert lighting["darkest_shape"] == pytest.approx(0.8119148936170213)
        assert lighting["brightest_ground"] < BRIGHTNESS_THRESHOLD
        assert BRIGHTNESS_THRESHOLD < lighting["darkest_shape"]
        assert lighting["scene_reading"]["wrongly_labelled"] == 0
        assert lighting["scene_reading"]["kept"] == lighting["n_shape_pixels"] == 375

    def test_and_the_same_threshold_fails_on_the_relit_scene(self):
        """The window that works moves clear of where it was."""
        lighting = get(SCENE)["lighting"]

        assert lighting["relit_brightest_ground"] == pytest.approx(0.822)
        assert lighting["relit_darkest_shape"] == pytest.approx(1.449063829787234)
        assert lighting["relit_brightest_ground"] > lighting["darkest_shape"]
        assert lighting["relit_reading"]["kept"] == 1252
        assert lighting["relit_reading"]["wrongly_labelled"] == 877
        assert lighting["n_disagreeing"] == 877
        assert lighting["share_disagreeing"] == pytest.approx(877 / 2304)

    def test_the_gradient_is_scaled_by_the_lamp_and_not_turned_by_it(self):
        lighting = get(SCENE)["lighting"]

        assert lighting["magnitude_ratio_low"] == pytest.approx(1.6, abs=1e-12)
        assert lighting["magnitude_ratio_high"] == pytest.approx(1.6, abs=1e-12)
        assert lighting["largest_direction_move"] < 1.1e-14
        assert lighting["n_sharp_pixels"] == 450
        assert lighting["largest_direction_move_where_sharp"] < 1e-15

    def test_the_brightness_moved_a_long_way_at_every_pixel(self):
        lighting = get(SCENE)["lighting"]

        assert lighting["mean_brightness_move"] == pytest.approx(0.376453125)
        assert lighting["largest_brightness_move"] == pytest.approx(0.7788510638297872)


class TestTheField:
    def test_the_scene_answers_a_map_of_the_same_size(self):
        field = post(FIELD, {})

        assert field["answer_height"] == 48
        assert field["answer_width"] == 48
        for name in ("horizontal", "vertical", "magnitude", "direction"):
            assert len(field[name]) == 48
            assert all(len(row) == 48 for row in field[name])

    def test_keeping_only_the_valid_positions_answers_a_smaller_map(self):
        field = post(FIELD, {"edge_rule": "keep_valid"})

        assert field["answer_height"] == 46
        assert field["answer_width"] == 46

    def test_the_edge_direction_is_the_gradient_turned_a_right_angle(self):
        field = post(FIELD, {"scene": "hand"})

        assert field["direction"][2][2] == pytest.approx(0.0)
        assert field["edge_direction"][2][2] == pytest.approx(1.5708, abs=1e-4)

    def test_a_scene_nothing_draws_is_refused_at_the_door(self):
        response = client.post(FIELD, json={"scene": "a photograph"})

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self):
        response = client.post(FIELD, json={"operator": "sobel", "sharpen": True})

        assert response.status_code == 422

    @pytest.mark.parametrize("operator", sorted(POSITIVE_WEIGHT_TOTALS))
    def test_every_operator_sweeps_every_scene(self, operator):
        field = post(FIELD, {"operator": operator, "scene": "relit"})

        assert field["operator_label"] == OPERATOR_LABELS[operator]
        assert field["scene_label"] == SCENE_LABELS["relit"]
        assert field["largest_magnitude"] > 0.0


class TestTheHandWorkedSweep:
    def test_the_picture_is_the_shared_six_by_six_one(self):
        sweep = post(HAND_SWEEP, {})

        assert sweep["picture"] == [HAND_ROW] * 6

    def test_a_central_difference_answers_a_half_at_both_columns(self):
        """The step sits between two columns, so both of them report it."""
        sweep = post(HAND_SWEEP, {})

        assert sweep["weights"] == [[-0.5, 0.0, 0.5]]
        assert sweep["answer"][0] == [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
        assert sweep["answer"] == [[0.0, 0.0, 0.5, 0.5, 0.0, 0.0]] * 6

    def test_every_answer_carries_the_window_that_produced_it(self):
        sweep = post(HAND_SWEEP, {})

        first = sweep["cells"][0]
        assert first["row"] == 0 and first["column"] == 0
        assert first["left"] == -1
        assert first["patch"] == [[0.0, 0.0, 0.0]]
        assert first["answer"] == 0.0

        at_the_edge = sweep["cells"][2]
        assert at_the_edge["patch"] == [[0.0, 0.0, 1.0]]
        assert at_the_edge["products"] == [[-0.0, 0.0, 0.5]]
        assert at_the_edge["answer"] == 0.5

    def test_reversing_the_weights_reverses_every_sign(self):
        """Which is the whole difference between correlation and convolution."""
        forward = post(HAND_SWEEP, {})
        backward = post(HAND_SWEEP, {"reversed_weights": True})

        assert backward["weights"] == [[0.5, 0.0, -0.5]]
        assert backward["answer"] == [
            [-value for value in row] for row in forward["answer"]
        ]

    @pytest.mark.parametrize(
        "operator, expected", sorted(POSITIVE_WEIGHT_TOTALS.items())
    )
    def test_each_operator_answers_the_sum_of_its_positive_weights(
        self, operator, expected
    ):
        sweep = post(HAND_SWEEP, {"operator": operator})

        assert sweep["positive_weight_total"] == expected
        assert sweep["weight_total"] == 0.0
        assert sweep["answer"][2][2] == expected
        assert sweep["answer"][2][3] == expected
        assert sweep["answer"][2][0] == 0.0

    def test_the_vertical_grid_finds_nothing_in_a_picture_with_no_rows_differing(self):
        sweep = post(HAND_SWEEP, {"operator": "sobel", "vertical": True})

        assert sweep["weights"] == [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ]
        assert sweep["answer"] == [[0.0] * 6] * 6

    def test_keeping_only_valid_positions_loses_a_ring_of_pixels(self):
        sweep = post(HAND_SWEEP, {"operator": "sobel", "edge_rule": "keep_valid"})

        assert sweep["answer_height"] == 4
        assert sweep["answer_width"] == 4
        assert sweep["pad_rows"] == 0 and sweep["pad_columns"] == 0
        assert sweep["answer"] == [[0.0, 4.0, 4.0, 0.0]] * 4


class TestTheBorderRules:
    def test_every_rule_is_reported_with_a_label_a_reader_can_act_on(self):
        borders = get(BORDERS)

        assert len(borders["readings"]) == 4
        for reading in borders["readings"]:
            assert reading["label"] == EDGE_RULE_LABELS[reading["edge_rule"]]
            assert reading["note"]

    def test_repeating_the_edge_pixel_invents_nothing(self):
        reading = self._reading("extend")

        assert reading["averaged_middle_row"] == [0.0, 0.0, 0.3333, 0.6667, 1.0, 1.0]
        assert reading["magnitude_middle_row"] == [0.0, 0.0, 4.0, 4.0, 0.0, 0.0]
        assert reading["invented_at"] == []

    def test_reading_from_the_opposite_side_invents_an_edge_at_both_ends(self):
        reading = self._reading("wrap")

        assert reading["averaged_middle_row"][0] == pytest.approx(1 / 3, abs=1e-4)
        assert reading["magnitude_middle_row"] == [4.0, 0.0, 4.0, 4.0, 0.0, 4.0]
        assert reading["invented_at"] == [0, 5]

    def test_treating_the_outside_as_black_invents_one_at_the_bright_end_only(self):
        """It happens to be right on the left of this picture and wrong on the right."""
        reading = self._reading("pad_with_zero")

        assert reading["averaged_middle_row"][0] == 0.0
        assert reading["magnitude_middle_row"] == [0.0, 0.0, 4.0, 4.0, 0.0, 4.0]
        assert reading["invented_at"] == [5]

    def test_answering_only_where_the_weights_fit_answers_a_smaller_picture(self):
        reading = self._reading("keep_valid")

        assert reading["answer_height"] == 4
        assert reading["answer_width"] == 4
        assert reading["invented_at"] == []

    def test_the_two_that_agree_here_are_not_the_same_rule(self):
        """Both answer zero at the left border and they part company at the right."""
        extended = self._reading("extend")
        padded = self._reading("pad_with_zero")

        assert extended["magnitude_middle_row"][0] == padded["magnitude_middle_row"][0]
        assert extended["magnitude_middle_row"][5] != padded["magnitude_middle_row"][5]

    @staticmethod
    def _reading(edge_rule: str) -> dict:
        borders = get(BORDERS)
        return next(
            reading
            for reading in borders["readings"]
            if reading["edge_rule"] == edge_rule
        )


class TestTheOperators:
    def test_each_operator_answers_the_sum_of_its_positive_weights(self):
        readings = {
            reading["operator"]: reading for reading in get(OPERATORS)["readings"]
        }

        for operator, expected in POSITIVE_WEIGHT_TOTALS.items():
            assert readings[operator]["positive_weight_total"] == expected
            assert readings[operator]["clean_step_answer"] == pytest.approx(expected)

    def test_every_operator_sums_to_zero_so_a_flat_region_answers_nothing(self):
        for reading in get(OPERATORS)["readings"]:
            assert reading["weight_total"] == 0.0

    def test_every_operator_agrees_about_the_direction_of_a_clean_step(self):
        for reading in get(OPERATORS)["readings"]:
            assert reading["clean_step_direction"] == pytest.approx(0.0)
            assert reading["clean_step_edge_direction"] == pytest.approx(
                1.5708, abs=1e-4
            )

    def test_on_the_diagonal_bar_they_part_company_by_a_quarter_of_a_degree(self):
        """And by a factor of twenty-six in how big the answer is."""
        operators = get(OPERATORS)

        assert operators["bar_angle_spread"] == pytest.approx(0.2838, abs=1e-3)
        assert operators["bar_magnitude_ratio"] == pytest.approx(25.94, abs=0.01)
        for reading in operators["readings"]:
            assert reading["bar_angle_error"] < 1.0

    def test_on_a_blurred_straight_edge_the_ordering_is_the_published_one(self):
        readings = {
            reading["operator"]: reading["sweep"]
            for reading in get(OPERATORS)["readings"]
        }

        worst = {
            operator: sweep["blurred_worst_error"]
            for operator, sweep in readings.items()
        }
        assert worst["scharr"] == pytest.approx(0.2784, abs=1e-3)
        assert worst["sobel"] == pytest.approx(1.1630, abs=1e-3)
        assert worst["prewitt"] == pytest.approx(3.1038, abs=1e-3)
        assert worst["central_difference"] == pytest.approx(4.1300, abs=1e-3)
        assert (
            worst["scharr"]
            < worst["sobel"]
            < worst["prewitt"]
            < worst["central_difference"]
        )

    def test_and_on_a_hard_one_pixel_step_that_ordering_does_not_survive(self):
        """Every one of them is out by nine degrees and Scharr is the worst."""
        readings = {
            reading["operator"]: reading["sweep"]
            for reading in get(OPERATORS)["readings"]
        }

        assert readings["sobel"]["hard_mean_error"] == pytest.approx(9.2367, abs=1e-3)
        assert readings["scharr"]["hard_mean_error"] == pytest.approx(10.8913, abs=1e-3)
        assert (
            readings["scharr"]["hard_mean_error"] > readings["sobel"]["hard_mean_error"]
        )
        assert readings["central_difference"]["hard_worst_error"] == pytest.approx(40.0)

    def test_every_drawn_angle_is_reported_for_every_operator(self):
        operators = get(OPERATORS)

        assert operators["angles"] == list(range(0, 90, 5))
        for reading in operators["readings"]:
            assert len(reading["sweep"]["blurred"]) == len(operators["angles"])
            assert reading["sweep"]["blurred"][0]["error"] == pytest.approx(0.0)

    def test_the_cost_is_counted_as_well_as_timed(self):
        operators = get(OPERATORS)

        assert operators["n_operations_per_pixel"] == 34
        assert operators["n_operations_for_scene"] == 34 * 48 * 48
        scene_cost = next(cost for cost in operators["costs"] if cost["side"] == 48)
        assert scene_cost["n_pixels"] == 2304
        assert scene_cost["n_multiplies"] == 9 * 2304
        assert scene_cost["n_additions"] == 8 * 2304
        assert scene_cost["seconds_per_sweep"] > 0.0

    def test_the_per_pixel_cost_falls_as_the_picture_grows(self):
        """Most of the time on a small picture is setting the sweep up."""
        costs = {cost["side"]: cost for cost in get(OPERATORS)["costs"]}

        assert costs[1024]["nanoseconds_per_pixel"] < costs[48]["nanoseconds_per_pixel"]


class TestTheThreshold:
    def test_the_two_populations_are_read_off_the_ramp_free_scene(self):
        thresholds = get(THRESHOLDS)

        assert thresholds["n_pixels"] == 2304
        assert thresholds["n_edge_pixels"] == 450
        assert thresholds["n_flat_pixels"] == 1854
        assert thresholds["n_faint_pixels"] == 98
        assert thresholds["noise_level"] == NOISE_LEVEL

    def test_with_no_noise_a_wide_window_of_thresholds_is_perfect(self):
        thresholds = get(THRESHOLDS)

        assert thresholds["lowest_clean_threshold"] == pytest.approx(0.05106, abs=1e-4)
        assert thresholds["highest_clean_threshold"] == pytest.approx(0.898, abs=1e-3)
        assert (
            thresholds["lowest_clean_threshold"] < thresholds["highest_clean_threshold"]
        )

    def test_and_the_noise_closes_that_window_entirely(self):
        """Flat ground now reaches past the faintest real edge, so they overlap."""
        thresholds = get(THRESHOLDS)

        assert thresholds["noisy_flat_high"] == pytest.approx(1.0157, abs=1e-3)
        assert thresholds["noisy_flat_high"] > thresholds["highest_clean_threshold"]

    def test_no_threshold_both_keeps_every_edge_and_rejects_every_flat_pixel(self):
        thresholds = get(THRESHOLDS)

        assert all(
            row["flat_kept"] > 0 or row["edge_lost"] > 0 for row in thresholds["rows"]
        )
        assert thresholds["best_wrong"] == 40
        assert thresholds["best_threshold"] == pytest.approx(0.9)

    def test_the_two_ends_of_the_sweep_fail_in_opposite_directions(self):
        rows = {row["threshold"]: row for row in get(THRESHOLDS)["rows"]}

        assert rows[0.4]["edge_lost"] == 0
        assert rows[0.4]["flat_kept"] == 693
        assert rows[1.1]["flat_kept"] == 0
        assert rows[1.1]["edge_lost"] == 63

    def test_the_faint_half_is_what_a_high_threshold_loses(self):
        rows = {row["threshold"]: row for row in get(THRESHOLDS)["rows"]}

        assert rows[1.2]["faint_lost"] == 72
        assert rows[1.2]["edge_lost"] == 75


class TestNothingVisibleNamesTheCodeBehindThePage:
    """Every string a widget can render, checked for identifiers.

    A message that grows a class name or an attribute name later fails here
    rather than reaching a reader, which is the rule the page is held to.
    """

    FORBIDDEN = (
        "oop_ml",
        "the library",
        "this library",
        "the API",
        "the package",
        "GradientField",
        "GradientOperator",
        "EdgeRule",
        "Picture",
        "swept",
        "edge_rule",
        "numpy",
    )

    def _strings(self, payload: object) -> list[str]:
        if isinstance(payload, str):
            return [payload]
        if isinstance(payload, dict):
            return [
                found
                for key, value in payload.items()
                for found in self._strings(value)
                if key not in ("operator", "edge_rule", "scene")
            ]
        if isinstance(payload, list):
            return [found for value in payload for found in self._strings(value)]
        return []

    @pytest.mark.parametrize("path", [SCENE, BORDERS, OPERATORS, THRESHOLDS])
    def test_no_identifier_survives_into_a_readable_string(self, path):
        for sentence in self._strings(get(path)):
            for identifier in self.FORBIDDEN:
                assert identifier not in sentence, sentence

    def test_the_same_holds_of_the_two_posted_endpoints(self):
        for payload in (post(FIELD, {}), post(HAND_SWEEP, {})):
            for sentence in self._strings(payload):
                for identifier in self.FORBIDDEN:
                    assert identifier not in sentence, sentence
