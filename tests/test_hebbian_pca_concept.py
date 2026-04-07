"""The Hebbian principal components endpoint, pinned to the page's claims.

The page hands the PCA page's four people to a unit obeying Oja's rule and
says that on those four the rule starts at a rate of 0.0025, that after one
epoch its first direction sits 23.1 degrees off the diagonal (1, 1) with the
learned vector 1.04 long, that after twenty epochs it sits a single degree off,
and that after the default two hundred it lies within two millionths of a
degree of the diagonal with the second direction within a hundredth of a degree
of (1, -1). It says the shares come back 0.8 and 0.2 to six decimals, that the
learned vectors are 1.000000 long with nothing normalising them, that the worst
orthogonality is about 0.00004, and that the walk stops itself at epoch 91 of
the 200 it was allowed. The eigen twin is the PCA page's own answer. On the
crowd it says the first direction lands within a hundredth of a degree while
the second comes back eight degrees off at length 0.43 with a worst
orthogonality of 0.14 after running the full budget, the honest report of a
direction that learns at a pace set by the one percent of the spread it is
chasing. All pinned.
"""

from itertools import pairwise

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.limits import MAX_POINTS
from oop_ml_api.main import create_app

client = TestClient(create_app())

LEARN = "/concepts/hebbian-pca/learn"

DEFAULT_EPOCHS = 200

FOUR_PEOPLE = [
    {"x": 180, "y": 78},
    {"x": 160, "y": 58},
    {"x": 175, "y": 63},
    {"x": 165, "y": 73},
]

CROWD = [
    {"x": 147, "y": 41},
    {"x": 156, "y": 53},
    {"x": 145, "y": 57},
    {"x": 159, "y": 57},
    {"x": 162, "y": 61},
    {"x": 120, "y": 25},
    {"x": 122, "y": 28},
    {"x": 118, "y": 24},
    {"x": 180, "y": 80},
    {"x": 183, "y": 83},
    {"x": 178, "y": 78},
]

DIAGONAL = 0.5**0.5
"""The coordinate of (1, 1) scaled to unit length, and of (1, -1)."""


def learned(points: list[dict], epochs: int) -> dict:
    response = client.post(LEARN, json={"points": points, "max_epochs": epochs})
    assert response.status_code == 200
    return response.json()


class TestTheWorkedExample:
    """The PCA page's four people, for the default two hundred epochs."""

    def test_the_starting_rate_is_a_half_over_the_longest_deviation_squared(
        self,
    ) -> None:
        # The longest deviation is (10, 10), of squared length 200.
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        assert body["starting_rate"] == pytest.approx(0.0025)

    def test_the_mean_is_the_cloud_centre(self) -> None:
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        assert body["mean"] == {"x": 170.0, "y": 68.0}

    def test_the_first_direction_is_the_diagonal_to_two_millionths_of_a_degree(
        self,
    ) -> None:
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        assert body["angles_degrees"][0] < 2e-6

    def test_the_first_direction_reads_as_the_pca_page_reports_it(self) -> None:
        # The same sign as the eigensolver's, so the learned vector reads
        # (0.71, 0.71) as the PCA page's own answer does.
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        first = body["hebbian"][0]
        assert first["dx"] == pytest.approx(DIAGONAL, abs=1e-6)
        assert first["dy"] == pytest.approx(DIAGONAL, abs=1e-6)

    def test_the_second_direction_is_the_other_diagonal_to_a_hundredth(
        self,
    ) -> None:
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        second = body["hebbian"][1]
        assert body["angles_degrees"][1] < 0.01
        assert abs(second["dx"]) == pytest.approx(DIAGONAL, abs=1e-4)
        assert abs(second["dy"]) == pytest.approx(DIAGONAL, abs=1e-4)
        assert second["dx"] * second["dy"] < 0

    def test_the_shares_are_four_fifths_and_one_fifth_to_six_decimals(self) -> None:
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        first, second = body["hebbian"]
        assert first["share"] == pytest.approx(0.8, abs=1e-6)
        assert second["share"] == pytest.approx(0.2, abs=1e-6)

    def test_the_learned_vectors_have_length_one_to_six_decimals(self) -> None:
        # Nothing normalises them. The length is the rule's own doing.
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        for direction in body["hebbian"]:
            assert direction["length"] == pytest.approx(1.0, abs=1e-6)

    def test_the_worst_orthogonality_is_forty_three_millionths(self) -> None:
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        assert body["worst_orthogonality"] == pytest.approx(4.3e-5, abs=5e-7)
        assert body["worst_orthogonality"] < 1e-4

    def test_the_walk_stops_itself_at_epoch_ninety_one(self) -> None:
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        assert body["epochs_run"] == 91

    def test_the_eigen_twin_is_the_pca_pages_answer(self) -> None:
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        first, second = body["eigen"]
        assert abs(first["dx"]) == pytest.approx(DIAGONAL, abs=1e-9)
        assert abs(first["dy"]) == pytest.approx(DIAGONAL, abs=1e-9)
        assert first["dx"] * first["dy"] > 0
        assert second["dx"] * second["dy"] < 0
        assert first["share"] == pytest.approx(0.8, abs=1e-9)
        assert second["share"] == pytest.approx(0.2, abs=1e-9)

    def test_the_variances_agree_with_the_eigenvalues(self) -> None:
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        for direction, twin in zip(body["hebbian"], body["eigen"], strict=True):
            assert direction["variance"] == pytest.approx(twin["variance"], rel=1e-6)


class TestTheSwing:
    """The bold line swinging into the faint one as the budget grows."""

    def test_after_one_epoch_the_first_direction_is_twenty_three_degrees_off(
        self,
    ) -> None:
        body = learned(FOUR_PEOPLE, 1)

        assert body["angles_degrees"][0] == pytest.approx(23.1, abs=0.05)
        assert body["epochs_run"] == 1

    def test_after_one_epoch_the_learned_vector_is_four_percent_long(self) -> None:
        # One pass of kicks from a random start leaves the length over one,
        # and the subtraction has not yet had time to pull it back.
        body = learned(FOUR_PEOPLE, 1)

        assert body["hebbian"][0]["length"] == pytest.approx(1.04, abs=0.005)

    def test_after_twenty_epochs_the_first_direction_is_a_degree_off(self) -> None:
        body = learned(FOUR_PEOPLE, 20)

        assert body["angles_degrees"][0] == pytest.approx(1.0, abs=0.05)

    def test_the_angle_only_falls_as_the_budget_grows(self) -> None:
        # Down to a hundred epochs the fall is strict. Past that the angle is
        # the residual a finite walk leaves, millionths of a degree either
        # way, and two such residuals are not ordered.
        angles = [
            learned(FOUR_PEOPLE, epochs)["angles_degrees"][0]
            for epochs in (1, 5, 10, 20, 50, 100)
        ]

        assert all(later < earlier for earlier, later in pairwise(angles))
        assert angles[-1] < 1e-5
        assert learned(FOUR_PEOPLE, 200)["angles_degrees"][0] < 1e-5

    def test_every_budget_is_its_own_fit(self) -> None:
        # The rate falls across whatever budget it is given, so a walk of
        # twenty epochs is not the first twenty of a walk of two hundred, and
        # the two do not share a length at epoch twenty.
        twenty = learned(FOUR_PEOPLE, 20)
        two_hundred = learned(FOUR_PEOPLE, 200)

        assert twenty["epochs_run"] == 20
        assert two_hundred["epochs_run"] < 200
        assert twenty["hebbian"][1]["length"] != pytest.approx(
            two_hundred["hebbian"][1]["length"], abs=1e-3
        )


class TestTheCrowd:
    """The second direction learns at a pace set by the spread it chases."""

    def test_the_first_direction_lands_within_a_hundredth_of_a_degree(self) -> None:
        body = learned(CROWD, DEFAULT_EPOCHS)

        assert body["angles_degrees"][0] < 0.01
        assert body["hebbian"][0]["length"] == pytest.approx(1.0, abs=1e-5)

    def test_the_second_direction_comes_back_short_and_eight_degrees_off(
        self,
    ) -> None:
        # The eigen twin holds one percent of the spread, so the rule moves
        # toward it a hundred times more slowly than toward the first.
        body = learned(CROWD, DEFAULT_EPOCHS)

        assert body["eigen"][1]["share"] == pytest.approx(0.010, abs=0.0005)
        assert body["angles_degrees"][1] == pytest.approx(8.2, abs=0.1)
        assert body["hebbian"][1]["length"] == pytest.approx(0.43, abs=0.01)
        assert body["worst_orthogonality"] == pytest.approx(0.14, abs=0.01)

    def test_the_walk_runs_its_full_budget(self) -> None:
        body = learned(CROWD, DEFAULT_EPOCHS)

        assert body["epochs_run"] == DEFAULT_EPOCHS

    def test_the_angle_is_taken_modulo_the_sign(self) -> None:
        # The eigensolver's first direction and the rule's point opposite
        # ways on the crowd, and the angle between them is still nothing.
        body = learned(CROWD, DEFAULT_EPOCHS)

        first, twin = body["hebbian"][0], body["eigen"][0]
        assert first["dx"] * twin["dx"] < 0
        assert body["angles_degrees"][0] < 0.01


class TestRefusals:
    def test_a_lone_person_is_refused(self) -> None:
        response = client.post(
            LEARN, json={"points": FOUR_PEOPLE[:1], "max_epochs": 10}
        )

        assert response.status_code == 422

    def test_too_many_people_are_refused_before_computing(self) -> None:
        many = [{"x": float(i), "y": float(i)} for i in range(MAX_POINTS + 1)]

        response = client.post(LEARN, json={"points": many, "max_epochs": 10})

        assert response.status_code == 422

    @pytest.mark.parametrize("epochs", [0, 501])
    def test_an_epoch_budget_outside_one_to_five_hundred_is_refused(
        self, epochs: int
    ) -> None:
        response = client.post(
            LEARN, json={"points": FOUR_PEOPLE, "max_epochs": epochs}
        )

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            LEARN, json={"points": FOUR_PEOPLE, "max_epochs": 10, "seed": 3}
        )

        assert response.status_code == 422

    def test_a_cloud_with_no_spread_is_refused_by_the_library(self) -> None:
        # Every person on one spot has no direction to find, and the library
        # says so before the rate is ever set from the longest deviation.
        response = client.post(
            LEARN,
            json={"points": [{"x": 170, "y": 68}] * 3, "max_epochs": 10},
        )

        assert response.status_code == 422
        assert response.json()["error"] == "InvalidValuesError"
