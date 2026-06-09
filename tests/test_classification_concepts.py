"""The three classification endpoints, pinned to the pages' worked numbers.

The k-NN page walks a query whose nearest three neighbours sit at distances 5,
10 and 13, all 3-4-5 triangles, voting child at k of 3 and adult at k of 5.
The trees page works the root Gini of 60 over 121 by hand and quotes the root
question, is height below 151.5. The logistic page reads a boundary near 4.24
hours off a fixed overlapping class. All of that is pinned here.
"""

import math
from itertools import pairwise
from typing import ClassVar

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

LOGISTIC = "/concepts/logistic-regression/fit"
KNN = "/concepts/k-nearest-neighbours/classify"
TREE = "/concepts/decision-trees/fit"

# Eleven people, five children and six adults, the worked set both the k-NN
# and the trees pages load.
PEOPLE = [
    {"x": 147, "y": 41, "label": 0},
    {"x": 156, "y": 53, "label": 1},
    {"x": 145, "y": 57, "label": 0},
    {"x": 159, "y": 57, "label": 1},
    {"x": 162, "y": 61, "label": 1},
    {"x": 120, "y": 25, "label": 0},
    {"x": 122, "y": 28, "label": 0},
    {"x": 118, "y": 24, "label": 0},
    {"x": 180, "y": 80, "label": 1},
    {"x": 183, "y": 83, "label": 1},
    {"x": 178, "y": 78, "label": 1},
]

QUERY = {"x": 150, "y": 45}

# Twelve study outcomes with an overlap, six fails and six passes.
OUTCOMES = [
    {"x": 1, "label": 0},
    {"x": 1.5, "label": 0},
    {"x": 2, "label": 0},
    {"x": 3, "label": 0},
    {"x": 3.5, "label": 0},
    {"x": 5, "label": 0},
    {"x": 4, "label": 1},
    {"x": 4.5, "label": 1},
    {"x": 5.5, "label": 1},
    {"x": 6, "label": 1},
    {"x": 7, "label": 1},
    {"x": 8, "label": 1},
]


class TestLogistic:
    def test_the_boundary_sits_between_the_classes(self) -> None:
        body = client.post(LOGISTIC, json={"points": OUTCOMES}).json()

        assert abs(body["boundary"] - 4.235) < 0.01
        assert body["slope"] > 0

    def test_the_overlap_pair_is_what_the_fit_misses(self) -> None:
        # Ten of twelve are called correctly; the fail at 5 hours and the
        # pass at 4 sit on the wrong sides of the boundary.
        body = client.post(LOGISTIC, json={"points": OUTCOMES}).json()

        assert abs(body["accuracy"] - 10 / 12) < 1e-9

    def test_the_curve_is_a_probability(self) -> None:
        body = client.post(LOGISTIC, json={"points": OUTCOMES}).json()

        heights = [each["y"] for each in body["curve"]]
        assert all(0.0 <= value <= 1.0 for value in heights)
        assert heights[0] < 0.1 and heights[-1] > 0.9

    def test_a_hot_climb_balloons_the_slope(self) -> None:
        # The gradient is bounded, so nothing overflows, though at a hot rate
        # the strides fly past the top and the slope grows with the epochs
        # instead of settling.
        calm = client.post(
            LOGISTIC, json={"points": OUTCOMES, "learning_rate": 0.1}
        ).json()
        hot = client.post(
            LOGISTIC, json={"points": OUTCOMES, "learning_rate": 40.0}
        ).json()

        assert hot["slope"] > 10 * calm["slope"]

    def test_a_rate_past_the_cap_is_refused(self) -> None:
        response = client.post(
            LOGISTIC, json={"points": OUTCOMES, "learning_rate": 51.0}
        )

        assert response.status_code == 422

    def test_a_single_class_is_refused(self) -> None:
        response = client.post(
            LOGISTIC,
            json={"points": [{"x": 1, "label": 0}, {"x": 2, "label": 0}]},
        )

        assert response.status_code == 422


class TestKNearest:
    def test_the_three_nearest_sit_at_the_worked_distances(self) -> None:
        body = client.post(KNN, json={"points": PEOPLE, "query": QUERY, "k": 3}).json()

        distances = [each["distance"] for each in body["neighbours"]]
        assert distances == [5.0, 10.0, 13.0]

    def test_three_neighbours_vote_child(self) -> None:
        body = client.post(KNN, json={"points": PEOPLE, "query": QUERY, "k": 3}).json()

        assert body["votes_for_zero"] == 2
        assert body["votes_for_one"] == 1
        assert body["prediction"] == 0

    def test_five_neighbours_flip_the_answer_to_adult(self) -> None:
        body = client.post(KNN, json={"points": PEOPLE, "query": QUERY, "k": 5}).json()

        assert body["votes_for_one"] == 3
        assert body["prediction"] == 1

    def test_more_voters_than_people_is_refused_in_words(self) -> None:
        response = client.post(KNN, json={"points": PEOPLE[:3], "query": QUERY, "k": 5})

        assert response.status_code == 422
        assert "voters" in response.json()["detail"]

    def test_the_regions_cover_the_lattice(self) -> None:
        body = client.post(KNN, json={"points": PEOPLE, "query": QUERY, "k": 3}).json()

        grid = body["regions"]
        assert len(grid["labels"]) == grid["cells"]
        assert all(len(row) == grid["cells"] for row in grid["labels"])


class TestDecisionTree:
    def test_the_worked_root_question_is_height_below_151_and_a_half(self) -> None:
        body = client.post(TREE, json={"points": PEOPLE, "max_depth": 3}).json()

        assert body["tree"]["question"] == "height < 151.5?"

    def test_the_clean_set_is_solved_by_one_question(self) -> None:
        body = client.post(TREE, json={"points": PEOPLE, "max_depth": 3}).json()

        assert body["depth"] == 1
        assert body["n_leaves"] == 2
        assert body["accuracy"] == 1.0
        assert body["tree"]["left"]["kind"] == "leaf"
        assert body["tree"]["left"]["label"] == 0
        assert body["tree"]["right"]["label"] == 1

    def test_a_capped_depth_is_respected(self) -> None:
        body = client.post(TREE, json={"points": PEOPLE, "max_depth": 1}).json()

        assert body["depth"] == 1

    def test_a_single_class_is_refused_in_words(self) -> None:
        children = [each for each in PEOPLE if each["label"] == 0]
        response = client.post(TREE, json={"points": children, "max_depth": 2})

        assert response.status_code == 422
        assert "nothing to separate" in response.json()["detail"]


class TestLogisticBalance:
    """The gaps the balance chart draws, and the condition they satisfy."""

    def test_the_gaps_cancel_at_a_calm_rate(self) -> None:
        points = [
            {"x": 1.0, "label": 0},
            {"x": 2.0, "label": 0},
            {"x": 3.0, "label": 0},
            {"x": 4.0, "label": 1},
            {"x": 5.0, "label": 0},
            {"x": 6.0, "label": 1},
            {"x": 7.0, "label": 1},
            {"x": 8.0, "label": 1},
        ]
        response = client.post(
            "/concepts/logistic-regression/fit",
            json={"points": points, "learning_rate": 0.1},
        )

        body = response.json()
        assert len(body["gaps"]) == len(points)
        assert abs(body["gap_total"]) < 0.05
        assert body["gap_total"] == sum(body["gaps"])


class TestRootCandidates:
    """The sweep the trees page's gain chart draws, pinned to the clean crowd."""

    CLEAN_CROWD: ClassVar[list[dict]] = [
        {"x": 147.0, "y": 41.0, "label": 0},
        {"x": 156.0, "y": 53.0, "label": 1},
        {"x": 145.0, "y": 57.0, "label": 0},
        {"x": 159.0, "y": 57.0, "label": 1},
        {"x": 162.0, "y": 61.0, "label": 1},
        {"x": 120.0, "y": 25.0, "label": 0},
        {"x": 122.0, "y": 28.0, "label": 0},
        {"x": 118.0, "y": 24.0, "label": 0},
        {"x": 180.0, "y": 80.0, "label": 1},
        {"x": 183.0, "y": 83.0, "label": 1},
        {"x": 178.0, "y": 78.0, "label": 1},
    ]

    def search(self) -> dict:
        response = client.post(
            "/concepts/decision-trees/root-candidates",
            json={"points": self.CLEAN_CROWD},
        )
        assert response.status_code == 200
        return response.json()

    def test_the_winner_matches_the_worked_example(self) -> None:
        body = self.search()
        assert body["best_feature"] == "height"
        assert body["best_threshold"] == 151.5
        assert round(body["best_gain"], 3) == 0.496

    def test_no_candidate_beats_the_winner(self) -> None:
        body = self.search()
        assert all(
            candidate["gain"] <= body["best_gain"] + 1e-12
            for candidate in body["candidates"]
        )

    def test_both_features_were_scanned(self) -> None:
        features = {candidate["feature"] for candidate in self.search()["candidates"]}
        assert features == {"height", "weight"}


class TestTreeWalk:
    """One visitor routed down the small three-leaf tree, pinned end to end."""

    THREE_LEAF_CROWD: ClassVar[list[dict]] = [
        {"x": 118.0, "y": 24.0, "label": 0},
        {"x": 122.0, "y": 28.0, "label": 0},
        {"x": 120.0, "y": 25.0, "label": 0},
        {"x": 132.0, "y": 32.0, "label": 0},
        {"x": 128.0, "y": 30.0, "label": 0},
        {"x": 135.0, "y": 34.0, "label": 0},
        {"x": 146.0, "y": 59.0, "label": 0},
        {"x": 140.0, "y": 63.0, "label": 0},
        {"x": 148.0, "y": 67.0, "label": 0},
        {"x": 161.0, "y": 42.0, "label": 0},
        {"x": 168.0, "y": 45.0, "label": 0},
        {"x": 159.0, "y": 57.0, "label": 1},
        {"x": 162.0, "y": 61.0, "label": 1},
        {"x": 178.0, "y": 66.0, "label": 1},
        {"x": 180.0, "y": 78.0, "label": 1},
        {"x": 183.0, "y": 83.0, "label": 1},
    ]

    def walk(self, height: float, weight: float) -> dict:
        response = client.post(
            "/concepts/decision-trees/walk",
            json={
                "points": self.THREE_LEAF_CROWD,
                "max_depth": 2,
                "visitor": {"x": height, "y": weight},
            },
        )
        assert response.status_code == 200
        return response.json()

    def test_the_tree_is_the_three_leaf_shape(self) -> None:
        body = self.walk(125.0, 30.0)
        root = body["tree"]
        assert root["feature"] == "height"
        assert root["threshold"] == 153.5
        assert root["left"]["kind"] == "answer"
        assert root["left"]["samples"] == 9
        assert root["right"]["feature"] == "weight"
        assert root["right"]["threshold"] == 51.0

    def test_a_stocky_child_is_never_asked_about_weight(self) -> None:
        body = self.walk(147.0, 62.0)
        assert len(body["steps"]) == 1
        assert body["steps"][0]["feature"] == "height"
        assert body["steps"][0]["direction"] == "left"
        assert body["prediction"] == 0

    def test_a_slender_teenager_needs_both_questions(self) -> None:
        body = self.walk(166.0, 46.0)
        directions = [(step["feature"], step["direction"]) for step in body["steps"]]
        assert directions == [("height", "right"), ("weight", "left")]
        assert body["prediction"] == 0

    def test_an_adult_goes_right_twice(self) -> None:
        body = self.walk(180.0, 74.0)
        directions = [(step["feature"], step["direction"]) for step in body["steps"]]
        assert directions == [("height", "right"), ("weight", "right")]
        assert body["prediction"] == 1

    def test_the_landing_leaf_carries_the_prediction(self) -> None:
        body = self.walk(166.0, 46.0)

        def find(node: dict, wanted: int) -> dict | None:
            if node["id"] == wanted:
                return node
            for side in ("left", "right"):
                child = node[side]
                if child is not None:
                    found = find(child, wanted)
                    if found is not None:
                        return found
            return None

        leaf = find(body["tree"], body["leaf_id"])
        assert leaf is not None
        assert leaf["kind"] == "answer"
        assert leaf["label"] == body["prediction"]


EVALUATE = "/concepts/logistic-regression/evaluate"
LOGISTIC_WALK = "/concepts/logistic-regression/walk"


class TestLogisticEvaluate:
    def _fitted(self) -> dict:
        return client.post(
            LOGISTIC, json={"points": OUTCOMES, "learning_rate": 0.5}
        ).json()

    def test_the_fit_reports_its_own_convergence(self) -> None:
        body = self._fitted()

        assert body["converged"] is True
        assert body["epochs_run"] < 10_000
        assert abs(body["odds_multiplier"] - math.exp(body["slope"])) < 1e-9
        assert abs(body["log_loss"] + body["log_likelihood"] / len(OUTCOMES)) < 1e-12

    def test_the_straight_line_leaves_the_probability_band(self) -> None:
        line = self._fitted()["straight_line"]
        at_one_hour = line["intercept"] + line["slope"] * 1
        at_eight_hours = line["intercept"] + line["slope"] * 8

        assert at_one_hour < 0
        assert at_eight_hours > 1

    def test_the_fitted_curve_scores_the_same_through_evaluate(self) -> None:
        fitted = self._fitted()
        body = client.post(
            EVALUATE,
            json={
                "points": OUTCOMES,
                "slope": fitted["slope"],
                "intercept": fitted["intercept"],
            },
        ).json()

        assert abs(body["accuracy"] - fitted["accuracy"]) < 1e-12
        assert abs(body["log_loss"] - fitted["log_loss"]) < 1e-9
        assert abs(body["gap_total"]) < 1e-3
        assert abs(body["weighted_gap_total"]) < 1e-2

    def test_lowering_the_threshold_changes_the_decisions_without_refitting(
        self,
    ) -> None:
        fitted = self._fitted()
        at_half = client.post(
            EVALUATE,
            json={
                "points": OUTCOMES,
                "slope": fitted["slope"],
                "intercept": fitted["intercept"],
                "threshold": 0.5,
            },
        ).json()
        at_three_tenths = client.post(
            EVALUATE,
            json={
                "points": OUTCOMES,
                "slope": fitted["slope"],
                "intercept": fitted["intercept"],
                "threshold": 0.3,
            },
        ).json()

        assert at_half["accuracy"] < at_three_tenths["accuracy"]
        assert at_half["log_loss"] == at_three_tenths["log_loss"]
        assert at_three_tenths["boundary"] < at_half["boundary"]
        assert (
            at_three_tenths["confusion"]["false_negatives"]
            < at_half["confusion"]["false_negatives"]
        )

    def test_a_flatter_curve_makes_the_same_decisions_at_a_worse_loss(self) -> None:
        fitted = self._fitted()
        sharp = client.post(
            EVALUATE,
            json={
                "points": OUTCOMES,
                "slope": fitted["slope"],
                "intercept": fitted["intercept"],
            },
        ).json()
        flat = client.post(
            EVALUATE,
            json={
                "points": OUTCOMES,
                "slope": fitted["slope"] / 2,
                "intercept": fitted["intercept"] / 2,
            },
        ).json()

        assert [each["predicted"] for each in sharp["outcomes"]] == [
            each["predicted"] for each in flat["outcomes"]
        ]
        assert flat["log_loss"] > sharp["log_loss"]


class TestLogisticWalk:
    def test_the_climb_lowers_the_loss_and_records_every_pass(self) -> None:
        body = client.post(
            LOGISTIC_WALK,
            json={"points": OUTCOMES, "learning_rate": 0.5, "max_epochs": 100},
        ).json()

        assert len(body["passes"]) == 100
        # A bold first stride at this rate overshoots, so the first pass can
        # sit above the start; the climb as a whole still comes down.
        assert body["passes"][-1]["log_loss"] < body["start_log_loss"]
        assert body["passes"][-1]["log_loss"] < body["passes"][0]["log_loss"]
        assert abs(body["start_log_loss"] - math.log(2)) < 1e-12

    def test_separated_classes_keep_steepening_after_accuracy_is_perfect(self) -> None:
        separated = [{"x": x, "label": 0} for x in (1, 1.5, 2, 2.5, 3, 3.5)] + [
            {"x": x, "label": 1} for x in (5.5, 6, 6.5, 7, 7.5, 8)
        ]
        body = client.post(
            LOGISTIC_WALK,
            json={"points": separated, "learning_rate": 0.5, "max_epochs": 1000},
        ).json()

        first_perfect = next(each for each in body["passes"] if each["accuracy"] == 1.0)
        last = body["passes"][-1]
        assert last["slope"] > first_perfect["slope"] * 2
        assert last["log_loss"] < first_perfect["log_loss"]
        assert body["converged"] is False

    def test_the_lattice_holds_the_start_and_the_end(self) -> None:
        body = client.post(
            LOGISTIC_WALK,
            json={"points": OUTCOMES, "learning_rate": 0.5, "max_epochs": 50},
        ).json()

        assert body["intercept_axis"][0] <= 0 <= body["intercept_axis"][-1]
        assert body["slope_axis"][0] <= 0 <= body["slope_axis"][-1]
        assert (
            body["intercept_axis"][0]
            <= body["passes"][-1]["intercept"]
            <= body["intercept_axis"][-1]
        )
        assert (
            body["slope_axis"][0]
            <= body["passes"][-1]["slope"]
            <= body["slope_axis"][-1]
        )


SPLIT_INSPECTOR = "/concepts/decision-trees/split-inspector"
GROWTH = "/concepts/decision-trees/growth"
DEPTH_SWEEP = "/concepts/decision-trees/depth-sweep"
VERSUS_LOGISTIC = "/concepts/decision-trees/versus-logistic"

TREE_CROWD = [
    {"x": 147, "y": 41, "label": 0},
    {"x": 156, "y": 53, "label": 1},
    {"x": 145, "y": 57, "label": 0},
    {"x": 159, "y": 57, "label": 1},
    {"x": 162, "y": 61, "label": 1},
    {"x": 120, "y": 25, "label": 0},
    {"x": 122, "y": 28, "label": 0},
    {"x": 118, "y": 24, "label": 0},
    {"x": 180, "y": 80, "label": 1},
    {"x": 183, "y": 83, "label": 1},
    {"x": 178, "y": 78, "label": 1},
]

MUDDLED_CROWD = TREE_CROWD + [
    {"x": 145, "y": 45, "label": 0},
    {"x": 145, "y": 55, "label": 1},
    {"x": 151, "y": 45, "label": 1},
    {"x": 151, "y": 55, "label": 0},
    {"x": 157, "y": 45, "label": 0},
    {"x": 157, "y": 55, "label": 1},
    {"x": 148, "y": 50, "label": 1},
    {"x": 154, "y": 50, "label": 0},
    {"x": 160, "y": 50, "label": 1},
    {"x": 147, "y": 58, "label": 0},
    {"x": 153, "y": 58, "label": 1},
    {"x": 150, "y": 42, "label": 1},
    {"x": 156, "y": 42, "label": 0},
    {"x": 143, "y": 50, "label": 0},
]


class TestSplitInspector:
    def test_the_winning_question_scores_the_whole_root_impurity(self) -> None:
        body = client.post(
            SPLIT_INSPECTOR,
            json={"points": TREE_CROWD, "feature": "height", "threshold": 151.5},
        ).json()

        assert abs(body["parent"]["gini"] - 60 / 121) < 1e-12
        assert body["left"]["gini"] == 0.0
        assert body["right"]["gini"] == 0.0
        assert abs(body["gain"] - 60 / 121) < 1e-12
        assert body["left"]["children"] == 5 and body["right"]["adults"] == 6

    def test_a_poor_question_earns_little(self) -> None:
        body = client.post(
            SPLIT_INSPECTOR,
            json={"points": TREE_CROWD, "feature": "height", "threshold": 121},
        ).json()

        assert body["left"]["n_samples"] == 2
        assert body["right"]["children"] == 3 and body["right"]["adults"] == 6
        assert abs(body["right"]["gini"] - (1 - (3 / 9) ** 2 - (6 / 9) ** 2)) < 1e-12
        assert 0 < body["gain"] < 0.2

    def test_the_candidates_sit_between_distinct_values(self) -> None:
        body = client.post(
            SPLIT_INSPECTOR,
            json={"points": TREE_CROWD, "feature": "height", "threshold": 151.5},
        ).json()

        assert 151.5 in body["candidates"]
        assert len(body["candidates"]) == 10
        assert body["candidates"] == sorted(body["candidates"])


class TestGrowth:
    def test_the_clean_crowd_grows_one_split(self) -> None:
        body = client.post(GROWTH, json={"points": TREE_CROWD}).json()

        assert body["n_splits"] == 1
        assert body["n_leaves"] == 2
        root = body["nodes"][0]
        assert root["feature"] == "height" and abs(root["threshold"] - 151.5) < 1e-9
        assert abs(root["gain"] - 60 / 121) < 1e-9
        assert body["nodes"][root["left_id"]]["children"] == 5
        assert body["nodes"][root["right_id"]]["adults"] == 6

    def test_every_leaf_owns_a_rectangle_inside_the_window(self) -> None:
        body = client.post(
            GROWTH, json={"points": MUDDLED_CROWD, "max_depth": 3}
        ).json()

        for node in body["nodes"]:
            x_low, x_high, y_low, y_high = node["bounds"]
            assert body["x_min"] <= x_low < x_high <= body["x_max"]
            assert body["y_min"] <= y_low < y_high <= body["y_max"]
        leaves = [node for node in body["nodes"] if node["split_order"] is None]
        assert len(leaves) == body["n_leaves"]
        assert sum(node["children"] + node["adults"] for node in leaves) == len(
            MUDDLED_CROWD
        )

    def test_a_leaf_minimum_holds_growth_back(self) -> None:
        free = client.post(
            GROWTH, json={"points": MUDDLED_CROWD, "max_depth": 6}
        ).json()
        held = client.post(
            GROWTH,
            json={"points": MUDDLED_CROWD, "max_depth": 6, "min_samples_leaf": 4},
        ).json()

        assert held["n_leaves"] < free["n_leaves"]
        assert all(
            node["children"] + node["adults"] >= 4
            for node in held["nodes"]
            if node["split_order"] is None
        )


class TestDepthSweep:
    def test_training_accuracy_never_falls_with_depth(self) -> None:
        body = client.post(
            DEPTH_SWEEP, json={"points": MUDDLED_CROWD, "max_depth": 6}
        ).json()

        train = [each["train_accuracy"] for each in body["scores"]]
        assert all(later >= earlier - 1e-12 for earlier, later in pairwise(train))
        assert len(body["held_out_indices"]) == int(len(MUDDLED_CROWD) * 0.3)

    def test_the_held_out_score_stops_climbing_before_the_training_score(self) -> None:
        body = client.post(
            DEPTH_SWEEP, json={"points": MUDDLED_CROWD, "max_depth": 6}
        ).json()

        held = [each["held_out_accuracy"] for each in body["scores"]]
        train = [each["train_accuracy"] for each in body["scores"]]
        assert train[-1] > held[-1]


class TestVersusLogistic:
    def test_both_models_map_the_same_window(self) -> None:
        body = client.post(
            VERSUS_LOGISTIC, json={"points": TREE_CROWD, "max_depth": 2}
        ).json()

        assert body["tree_regions"]["cells"] == body["logistic_regions"]["cells"]
        assert body["tree_regions"]["x_min"] == body["logistic_regions"]["x_min"]
        assert body["tree_accuracy"] == 1.0
        assert body["logistic_accuracy"] == 1.0


class TestRootWithNothingToGain:
    XOR_CROWD: ClassVar[list[dict]] = [
        {"x": 130, "y": 30, "label": 0},
        {"x": 135, "y": 35, "label": 0},
        {"x": 170, "y": 70, "label": 0},
        {"x": 175, "y": 75, "label": 0},
        {"x": 130, "y": 70, "label": 1},
        {"x": 135, "y": 75, "label": 1},
        {"x": 170, "y": 30, "label": 1},
        {"x": 175, "y": 35, "label": 1},
    ]

    def test_no_root_question_earns_anything_on_a_crossed_crowd(self) -> None:
        body = client.post(
            "/concepts/decision-trees/root-candidates", json={"points": self.XOR_CROWD}
        ).json()

        assert body["best_feature"] is None
        assert max(each["gain"] for each in body["candidates"]) < 1e-12

    def test_the_tree_then_stays_a_single_leaf(self) -> None:
        body = client.post(
            GROWTH, json={"points": self.XOR_CROWD, "max_depth": 3}
        ).json()

        assert body["n_splits"] == 0
        assert body["n_leaves"] == 1
        assert body["accuracy"] == 0.5
