"""The three classification endpoints, pinned to the pages' worked numbers.

The k-NN page walks a query whose nearest three neighbours sit at distances 5,
10 and 13, all 3-4-5 triangles, voting child at k of 3 and adult at k of 5.
The trees page works the root Gini of 60 over 121 by hand and quotes the root
question, is height below 151.5. The logistic page reads a boundary near 4.24
hours off a fixed overlapping class. All of that is pinned here.
"""

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
