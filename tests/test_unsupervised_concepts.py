"""The k-means and PCA endpoints, pinned to the pages' hand arithmetic.

The k-means page works eight people in two tight groups whose resting centres
are the plain group means, (121, 26) and (181, 80), with an inertia of exactly
64 summed by hand. The PCA page works four people whose deviation sums build
the matrix [[250, 150], [150, 250]], whose kept directions are the linear
algebra primer's (1, 1) and (1, -1) with shares of 0.8 and 0.2. All pinned.
"""

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

KMEANS = "/concepts/k-means/cluster"
PCA = "/concepts/pca/analyze"

TWO_GROUPS = [
    {"x": 118, "y": 24},
    {"x": 120, "y": 25},
    {"x": 122, "y": 28},
    {"x": 124, "y": 27},
    {"x": 178, "y": 78},
    {"x": 180, "y": 80},
    {"x": 182, "y": 83},
    {"x": 184, "y": 79},
]

FOUR_PEOPLE = [
    {"x": 180, "y": 78},
    {"x": 160, "y": 58},
    {"x": 175, "y": 63},
    {"x": 165, "y": 73},
]


class TestKMeans:
    def test_the_resting_centres_are_the_group_means(self) -> None:
        body = client.post(KMEANS, json={"points": TWO_GROUPS, "k": 2}).json()

        centres = sorted([(each["x"], each["y"]) for each in body["centres"]])
        assert centres == [(121.0, 26.0), (181.0, 80.0)]

    def test_the_inertia_is_the_hand_worked_sixty_four(self) -> None:
        body = client.post(KMEANS, json={"points": TWO_GROUPS, "k": 2}).json()

        assert body["inertia"] == 64.0

    def test_the_grouping_separates_the_two_clumps(self) -> None:
        body = client.post(KMEANS, json={"points": TWO_GROUPS, "k": 2}).json()

        labels = body["labels"]
        assert len(set(labels[:4])) == 1
        assert len(set(labels[4:])) == 1
        assert labels[0] != labels[4]

    def test_one_group_holds_everyone(self) -> None:
        body = client.post(KMEANS, json={"points": TWO_GROUPS, "k": 1}).json()

        assert set(body["labels"]) == {0}
        assert len(body["centres"]) == 1

    def test_inertia_never_rises_with_more_groups(self) -> None:
        inertias = [
            client.post(KMEANS, json={"points": TWO_GROUPS, "k": k}).json()["inertia"]
            for k in (1, 2, 4)
        ]

        assert inertias[0] >= inertias[1] >= inertias[2]

    def test_more_groups_than_people_is_refused_in_words(self) -> None:
        response = client.post(KMEANS, json={"points": TWO_GROUPS[:3], "k": 5})

        assert response.status_code == 422
        assert "conjured" in response.json()["detail"]


class TestPca:
    def test_the_directions_are_the_primer_pair(self) -> None:
        body = client.post(PCA, json={"points": FOUR_PEOPLE}).json()

        first, second = body["components"]
        # The long direction lies along (1, 1) and the short along (1, -1),
        # reported at unit length.
        assert abs(abs(first["dx"]) - abs(first["dy"])) < 1e-9
        assert abs(abs(second["dx"]) - abs(second["dy"])) < 1e-9
        assert abs(first["dx"] * first["dy"]) > 0.49  # same signs
        assert second["dx"] * second["dy"] < 0  # opposite signs

    def test_the_shares_are_four_fifths_and_one_fifth(self) -> None:
        body = client.post(PCA, json={"points": FOUR_PEOPLE}).json()

        first, second = body["components"]
        assert abs(first["share"] - 0.8) < 1e-9
        assert abs(second["share"] - 0.2) < 1e-9

    def test_the_mean_is_the_cloud_centre(self) -> None:
        body = client.post(PCA, json={"points": FOUR_PEOPLE}).json()

        assert body["mean"] == {"x": 170.0, "y": 68.0}

    def test_reconstructions_sit_on_the_long_diagonal(self) -> None:
        # Flattened to one component, every rebuilt person satisfies
        # weight - 68 = height - 70, the line through the mean along (1, 1).
        body = client.post(PCA, json={"points": FOUR_PEOPLE}).json()

        for point in body["reconstructions"]:
            assert abs((point["y"] - 68.0) - (point["x"] - 170.0)) < 1e-9

    def test_a_lone_point_is_refused(self) -> None:
        response = client.post(PCA, json={"points": [{"x": 170, "y": 68}]})

        assert response.status_code == 422
