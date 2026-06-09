"""The Hebbian principal components endpoints, pinned to the page's claims.

The page hands the PCA page's four people to a unit obeying Oja's rule and
quotes what happened, step by step and epoch by epoch, and the numbers it
quotes are pinned here. Three endpoints are under test. ``learn`` is the
library's fit beside the eigensolver's, ``walk`` the same rule stepped one
person at a time so the page can show the weight vector turning, and
``sweeps`` the fit repeated across budgets, rates, seeds and schedules.

The claims that matter most are the ones a reader could check by hand: the
first four steps from a start along height, the exact identity that says what
one step does to the squared length, and that the trace lands bit for bit
where the library's own fit lands. After those come the measurements the page
reports rather than argues, that the plain rule's length runs away while its
direction is right, that a constant rate settles on the four people and
wobbles on the crowd, that two independent units find one direction, that an
uncentred walk learns where the cloud is rather than how it is shaped, and
that a circular cloud sends every seed somewhere different.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
from fastapi.testclient import TestClient
from oop_ml import ExponentialDecaySchedule, Feature, HebbianPrincipalComponents
from oop_ml.core.exceptions import (
    AllSameValuesError,
    DivergenceError,
    InvalidValuesError,
    NotFittedError,
    TooFewValuesError,
)

from oop_ml_api.limits import MAX_POINTS
from oop_ml_api.main import create_app

client = TestClient(create_app())

LEARN = "/concepts/hebbian-pca/learn"
WALK = "/concepts/hebbian-pca/walk"
SWEEPS = "/concepts/hebbian-pca/sweeps"

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

IDEAL_CASE = [
    {"x": 152, "y": 51},
    {"x": 155, "y": 52},
    {"x": 158, "y": 56.5},
    {"x": 161, "y": 58.5},
    {"x": 164, "y": 62},
    {"x": 167, "y": 66},
    {"x": 170, "y": 67},
    {"x": 173, "y": 71.5},
    {"x": 176, "y": 73.5},
    {"x": 179, "y": 77},
    {"x": 182, "y": 81},
    {"x": 185, "y": 82},
    {"x": 188, "y": 86.5},
    {"x": 191, "y": 88.5},
    {"x": 194, "y": 92},
]

CIRCLE = [
    {"x": 180.0, "y": 68.0},
    {"x": 177.0711, "y": 75.0711},
    {"x": 170.0, "y": 78.0},
    {"x": 162.9289, "y": 75.0711},
    {"x": 160.0, "y": 68.0},
    {"x": 162.9289, "y": 60.9289},
    {"x": 170.0, "y": 58.0},
    {"x": 177.0711, "y": 60.9289},
]
"""Eight people evenly round a circle about (170, 68), so the eigenvalues tie."""

DIAGONAL = 0.5**0.5
"""The coordinate of (1, 1) scaled to unit length, and of (1, -1)."""


def learned(points: list[dict], epochs: int, **extra: object) -> dict:
    response = client.post(
        LEARN, json={"points": points, "max_epochs": epochs, **extra}
    )
    assert response.status_code == 200
    return response.json()


def walked(points: list[dict], **extra: object) -> dict:
    response = client.post(WALK, json={"points": points, **extra})
    assert response.status_code == 200
    return response.json()


def swept(points: list[dict]) -> dict:
    response = client.post(SWEEPS, json={"points": points})
    assert response.status_code == 200
    return response.json()


def angle_between(first: dict, second: dict) -> float:
    """The angle in degrees between two recorded weight vectors."""
    alignment = abs(first["dx"] * second["dx"] + first["dy"] * second["dy"])
    return math.degrees(
        math.acos(min(1.0, alignment / (first["length"] * second["length"])))
    )


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
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        first = body["hebbian"][0]
        assert first["dx"] == pytest.approx(DIAGONAL, abs=1e-6)
        assert first["dy"] == pytest.approx(DIAGONAL, abs=1e-6)

    def test_the_second_direction_is_the_other_diagonal_to_a_hundredth(
        self,
    ) -> None:
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        second = body["hebbian"][1]
        assert body["angles_degrees"][1] == pytest.approx(0.0024, abs=0.0002)
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

    def test_the_walk_stops_itself_at_epoch_ninety_one(self) -> None:
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        assert body["epochs_run"] == 91
        assert body["converged"] is True

    def test_the_kept_variance_slightly_exceeds_the_total(self) -> None:
        # The two directions overlap by the worst orthogonality, and the
        # overlap is counted twice, so the kept total is a little over.
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        assert body["total_variance"] == pytest.approx(166.6667, abs=1e-3)
        assert body["kept_variance"] > body["total_variance"]
        assert body["kept_variance"] - body["total_variance"] < 2e-7

    def test_the_eigen_twin_is_the_pca_pages_answer(self) -> None:
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        first, second = body["eigen"]
        assert abs(first["dx"]) == pytest.approx(DIAGONAL, abs=1e-9)
        assert first["dx"] * first["dy"] > 0
        assert second["dx"] * second["dy"] < 0
        assert first["share"] == pytest.approx(0.8, abs=1e-9)
        assert second["share"] == pytest.approx(0.2, abs=1e-9)
        assert first["variance"] == pytest.approx(133.3333, abs=1e-3)
        assert second["variance"] == pytest.approx(33.3333, abs=1e-3)

    def test_the_variances_agree_with_the_eigenvalues(self) -> None:
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        for direction, twin in zip(body["hebbian"], body["eigen"], strict=True):
            assert direction["variance"] == pytest.approx(twin["variance"], rel=1e-6)

    def test_the_scores_along_the_two_routes_agree(self) -> None:
        # Person 1 lies entirely along the first direction and person 3
        # entirely along the second, by both routes.
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS)

        rule, eigen = body["learned_scores"], body["eigen_scores"]
        assert rule[0][0] == pytest.approx(14.1421, abs=1e-4)
        assert abs(eigen[0][0]) == pytest.approx(14.1421, abs=1e-4)
        assert abs(rule[0][1]) < 1e-3
        assert eigen[0][1] == pytest.approx(0.0, abs=1e-9)
        assert abs(rule[2][0]) < 1e-12
        assert abs(rule[2][1]) == pytest.approx(7.0711, abs=1e-4)


class TestTheSwing:
    """The bold line swinging into the faint one as the budget grows."""

    @pytest.mark.parametrize(
        ("epochs", "angle"),
        [(1, 23.13), (5, 16.44), (10, 6.89), (20, 1.01), (50, 0.0027)],
    )
    def test_the_first_angle_at_each_budget(self, epochs: int, angle: float) -> None:
        body = learned(FOUR_PEOPLE, epochs)

        assert body["angles_degrees"][0] == pytest.approx(angle, rel=0.02)
        assert body["epochs_run"] == epochs

    def test_after_one_epoch_the_learned_vector_is_four_percent_long(self) -> None:
        body = learned(FOUR_PEOPLE, 1)

        assert body["hebbian"][0]["length"] == pytest.approx(1.0404, abs=0.001)

    def test_after_one_epoch_the_kept_variance_overcounts_by_ten(self) -> None:
        # Two directions twelve percent from perpendicular overlap enough
        # to claim 176 of a total of 167.
        body = learned(FOUR_PEOPLE, 1)

        assert body["worst_orthogonality"] == pytest.approx(0.120, abs=0.002)
        assert body["kept_variance"] == pytest.approx(176.29, abs=0.05)

    def test_the_angle_only_falls_as_the_budget_grows(self) -> None:
        angles = [
            learned(FOUR_PEOPLE, epochs)["angles_degrees"][0]
            for epochs in (1, 5, 10, 20, 50, 100)
        ]

        assert all(later < earlier for earlier, later in pairwise(angles))
        assert angles[-1] < 1e-5

    def test_every_budget_is_its_own_fit(self) -> None:
        twenty = learned(FOUR_PEOPLE, 20)
        two_hundred = learned(FOUR_PEOPLE, 200)

        assert twenty["epochs_run"] == 20
        assert two_hundred["epochs_run"] < 200
        assert twenty["hebbian"][1]["length"] != pytest.approx(
            two_hundred["hebbian"][1]["length"], abs=1e-3
        )


class TestTheRateOnLearn:
    """A multiplier on the starting rate, up to the library's refusal."""

    def test_twice_the_rate_settles_in_thirty_one_epochs(self) -> None:
        body = learned(FOUR_PEOPLE, DEFAULT_EPOCHS, rate_multiplier=2.0)

        assert body["epochs_run"] == 31
        assert body["angles_degrees"][0] < 1e-5

    def test_four_times_the_rate_is_refused_as_a_diverged_walk(self) -> None:
        response = client.post(
            LEARN,
            json={"points": FOUR_PEOPLE, "max_epochs": 200, "rate_multiplier": 4.0},
        )

        assert response.status_code == 422
        assert response.json()["error"] == "DivergenceError"
        assert "epoch 4" in response.json()["detail"]

    def test_the_crowd_survives_four_times_and_not_eight(self) -> None:
        four = learned(CROWD, DEFAULT_EPOCHS, rate_multiplier=4.0)
        eight = client.post(
            LEARN, json={"points": CROWD, "max_epochs": 200, "rate_multiplier": 8.0}
        )

        assert four["angles_degrees"][1] == pytest.approx(0.144, abs=0.005)
        assert four["hebbian"][1]["length"] == pytest.approx(0.995, abs=0.002)
        assert eight.status_code == 422
        assert eight.json()["error"] == "DivergenceError"


class TestTheStepByHand:
    """One epoch from a start along height, which the page works by hand."""

    def test_the_four_steps_in_the_seeded_order(self) -> None:
        body = walked(FOUR_PEOPLE, max_epochs=1, start="along_height")
        steps = body["steps"]

        assert body["starting_rate"] == pytest.approx(0.0025)
        assert body["initial"][0]["dx"] == 1.0
        assert body["initial"][0]["dy"] == 0.0
        assert body["initial"][0]["angle_degrees"] == pytest.approx(45.0)
        assert [step["row"] for step in steps] == [3, 1, 0, 2]

        first = steps[0]
        assert first["presented"] == {"x": -5.0, "y": 5.0}
        assert first["outputs"] == [-5.0]
        assert first["residuals"] == [{"x": 0.0, "y": 5.0}]
        assert first["updates"] == [{"x": 0.0, "y": -0.0625}]
        assert first["weights"][0]["dx"] == 1.0
        assert first["weights"][0]["dy"] == -0.0625
        assert first["weights"][0]["angle_degrees"] == pytest.approx(48.58, abs=0.01)

        second = steps[1]
        assert second["presented"] == {"x": -10.0, "y": -10.0}
        assert second["outputs"] == [-9.375]
        assert second["residuals"][0]["x"] == pytest.approx(-0.625)
        assert second["residuals"][0]["y"] == pytest.approx(-10.5859, abs=1e-4)
        assert second["weights"][0]["dx"] == pytest.approx(1.0146, abs=1e-4)
        assert second["weights"][0]["dy"] == pytest.approx(0.1856, abs=1e-4)
        assert second["weights"][0]["angle_degrees"] == pytest.approx(34.63, abs=0.01)

        third = steps[2]
        assert third["presented"] == {"x": 10.0, "y": 10.0}
        assert third["outputs"][0] == pytest.approx(12.0026, abs=1e-4)
        assert third["weights"][0]["length"] == pytest.approx(1.0376, abs=1e-4)
        assert third["weights"][0]["angle_degrees"] == pytest.approx(21.19, abs=0.01)

        last = steps[3]
        assert last["weights"][0]["length"] == pytest.approx(1.0372, abs=1e-4)
        assert last["weights"][0]["angle_degrees"] == pytest.approx(23.61, abs=0.01)
        assert body["epochs"][0]["movement"] == pytest.approx(0.3783, abs=1e-4)

    def test_the_squared_length_identity_holds_at_every_step(self) -> None:
        # |w'|^2 = |w|^2 + 2 rate y^2 (1 - |w|^2) + rate^2 y^2 |r|^2, exactly.
        body = walked(FOUR_PEOPLE, max_epochs=1, start="along_height")
        before = 1.0

        for step in body["steps"]:
            output = step["outputs"][0]
            residual = step["residuals"][0]
            rate = step["rate"]
            first_order = 2 * rate * output**2 * (1 - before)
            second_order = (
                rate**2 * output**2 * (residual["x"] ** 2 + residual["y"] ** 2)
            )
            after = step["weights"][0]["length"] ** 2

            assert after == pytest.approx(
                before + first_order + second_order, abs=1e-12
            )
            before = after

    def test_the_third_step_is_where_the_subtraction_first_bites(self) -> None:
        # The page quotes the third step's two terms: the subtraction pulls
        # the squared length down by 0.046 and the second-order kick pushes
        # it up by 0.059.
        body = walked(FOUR_PEOPLE, max_epochs=1, start="along_height")
        third = body["steps"][2]
        before = body["steps"][1]["weights"][0]["length"] ** 2
        output = third["outputs"][0]
        residual = third["residuals"][0]

        assert before == pytest.approx(1.0640, abs=1e-4)
        assert 2 * third["rate"] * output**2 * (1 - before) == pytest.approx(
            -0.0461, abs=1e-4
        )
        assert third["rate"] ** 2 * output**2 * (
            residual["x"] ** 2 + residual["y"] ** 2
        ) == pytest.approx(0.0587, abs=1e-4)

    def test_plain_hebb_lengthens_at_every_step(self) -> None:
        body = walked(FOUR_PEOPLE, rule="hebb", max_epochs=1, start="along_height")
        steps = body["steps"]

        assert steps[0]["updates"] == [{"x": 0.0625, "y": -0.0625}]
        assert steps[0]["weights"][0]["length"] == pytest.approx(1.0643, abs=1e-4)
        lengths = [step["weights"][0]["length"] for step in steps]
        assert all(later > earlier for earlier, later in pairwise([1.0, *lengths]))
        assert lengths[-1] == pytest.approx(1.8254, abs=1e-4)
        assert steps[-1]["weights"][0]["angle_degrees"] == pytest.approx(
            29.36, abs=0.01
        )


class TestPlainHebbAgainstOja:
    """What twenty epochs do to the length under each rule."""

    def test_plain_hebb_at_a_constant_rate_runs_away_along_the_right_direction(
        self,
    ) -> None:
        body = walked(
            FOUR_PEOPLE, rule="hebb", max_epochs=20, start="along_height", decay=False
        )
        lengths = [epoch["weights"][0]["length"] for epoch in body["epochs"]]

        assert body["diverged_at_epoch"] is None
        assert lengths[0] == pytest.approx(1.825, abs=0.001)
        assert lengths[4] == pytest.approx(40.84, abs=0.01)
        assert lengths[9] == pytest.approx(2351.3, abs=0.5)
        assert lengths[19] == pytest.approx(7.82e6, rel=0.001)
        assert body["epochs"][-1]["weights"][0]["angle_degrees"] < 0.001

    def test_plain_hebb_at_a_falling_rate_still_lengthens_every_epoch(self) -> None:
        body = walked(FOUR_PEOPLE, rule="hebb", max_epochs=20, start="along_height")
        lengths = [epoch["weights"][0]["length"] for epoch in body["epochs"]]

        assert all(later > earlier for earlier, later in pairwise(lengths))
        assert lengths[-1] == pytest.approx(41.99, abs=0.01)
        assert body["epochs"][-1]["weights"][0]["angle_degrees"] == pytest.approx(
            2.94, abs=0.01
        )

    def test_oja_at_a_constant_rate_settles_at_length_one_in_thirteen_epochs(
        self,
    ) -> None:
        body = walked(
            FOUR_PEOPLE, rule="oja", max_epochs=20, start="along_height", decay=False
        )
        lengths = [epoch["weights"][0]["length"] for epoch in body["epochs"]]
        angles = [epoch["weights"][0]["angle_degrees"] for epoch in body["epochs"]]

        assert lengths[:4] == pytest.approx([1.0372, 1.0084, 1.0009, 1.0001], abs=1e-4)
        assert lengths[-1] == pytest.approx(1.0, abs=1e-9)
        assert angles[:4] == pytest.approx([23.61, 8.32, 2.66, 0.84], abs=0.01)
        assert angles[-1] < 1e-6
        assert body["converged_at_epoch"] == 13

    def test_oja_on_one_unit_is_sangers_rule_on_one_unit_to_the_last_bit(
        self,
    ) -> None:
        oja = walked(FOUR_PEOPLE, rule="oja", max_epochs=20)
        sanger = walked(FOUR_PEOPLE, rule="sanger", max_epochs=20)

        assert oja["steps"] == sanger["steps"]

    def test_the_trace_lands_where_the_librarys_fit_lands(self) -> None:
        # The same seed, the same order, the same schedule, so the twenty-epoch
        # trace of two units is the twenty-epoch fit, to the last bit.
        trace = walked(FOUR_PEOPLE, rule="sanger", n_components=2, max_epochs=20)
        fit = learned(FOUR_PEOPLE, 20)

        for state, direction in zip(
            trace["epochs"][-1]["weights"], fit["hebbian"], strict=True
        ):
            assert state["dx"] / state["length"] == pytest.approx(
                direction["dx"], abs=1e-14
            )
            assert state["dy"] / state["length"] == pytest.approx(
                direction["dy"], abs=1e-14
            )
            assert state["length"] == pytest.approx(direction["length"], abs=1e-14)
        assert trace["worst_orthogonality"] == pytest.approx(
            fit["worst_orthogonality"], abs=1e-14
        )


class TestTheRateOnTheWalk:
    """Multiples of the starting rate, from settling to overflowing."""

    def test_twice_the_rate_settles_in_three_epochs(self) -> None:
        body = walked(
            FOUR_PEOPLE,
            rule="oja",
            max_epochs=20,
            start="along_height",
            decay=False,
            rate_multiplier=2.0,
        )

        assert body["converged_at_epoch"] == 3
        assert body["epochs"][-1]["weights"][0]["length"] == pytest.approx(
            0.9994, abs=1e-4
        )

    def test_three_times_the_rate_leaves_the_length_bouncing(self) -> None:
        body = walked(
            FOUR_PEOPLE,
            rule="oja",
            max_epochs=20,
            start="along_height",
            decay=False,
            rate_multiplier=3.0,
        )
        lengths = [epoch["weights"][0]["length"] for epoch in body["epochs"]]

        assert body["converged_at_epoch"] is None
        assert min(lengths) < 0.3
        assert max(lengths) > 1.2

    def test_eight_times_the_rate_overflows_on_the_third_epoch(self) -> None:
        body = walked(
            FOUR_PEOPLE,
            rule="oja",
            max_epochs=20,
            start="along_height",
            decay=False,
            rate_multiplier=8.0,
        )
        lengths = [step["weights"][0]["length"] for step in body["steps"]]

        assert body["diverged_at_epoch"] == 3
        assert lengths[:4] == pytest.approx([1.118, 1.677, 9.48, 210.1], rel=0.01)
        assert len(body["steps"]) == 8


class TestTheConstantRate:
    """Whether the rate has to fall, measured on two clouds."""

    def test_on_the_four_a_constant_rate_settles_and_sooner(self) -> None:
        # Every one of the four lies exactly on an eigen direction, so the
        # fixed point is exact for each row and nothing chases anything.
        sweeps = swept(FOUR_PEOPLE)
        by_budget = {fit["max_epochs"]: fit for fit in sweeps["constant"]}

        assert by_budget[200]["epochs_run"] == 46
        assert by_budget[200]["converged"] is True
        assert by_budget[200]["angles_degrees"][0] < 1e-6
        assert by_budget[20]["angles_degrees"][0] < 1e-4
        assert {fit["max_epochs"]: fit for fit in sweeps["epochs"]}[20][
            "angles_degrees"
        ][0] == pytest.approx(1.01, abs=0.01)

    def test_on_the_crowd_a_constant_rate_never_settles(self) -> None:
        sweeps = swept(CROWD)

        assert all(fit["converged"] is False for fit in sweeps["constant"])
        last = {fit["max_epochs"]: fit for fit in sweeps["constant"]}[200]
        assert last["angles_degrees"][0] == pytest.approx(0.386, abs=0.005)

    def test_on_the_crowd_the_falling_rate_settles_the_first_direction(self) -> None:
        decaying = walked(CROWD, max_epochs=20)
        constant = walked(CROWD, max_epochs=20, decay=False)

        last_decaying = [
            step["weights"][0]["angle_degrees"]
            for step in decaying["steps"]
            if step["epoch"] == 20
        ]
        last_constant = [
            step["weights"][0]["angle_degrees"]
            for step in constant["steps"]
            if step["epoch"] == 20
        ]
        assert max(last_decaying) < 0.011
        assert min(last_constant) > 0.1
        assert max(last_constant) > 0.7

    def test_on_the_crowd_the_constant_rate_finds_the_second_direction_better(
        self,
    ) -> None:
        # The falling rate starves the slow unit, which is the trade the page
        # has to be plain about.
        sweeps = swept(CROWD)
        decaying = {fit["max_epochs"]: fit for fit in sweeps["epochs"]}[200]
        constant = {fit["max_epochs"]: fit for fit in sweeps["constant"]}[200]

        assert decaying["angles_degrees"][1] == pytest.approx(8.2, abs=0.1)
        assert decaying["lengths"][1] == pytest.approx(0.426, abs=0.005)
        assert constant["angles_degrees"][1] == pytest.approx(0.072, abs=0.005)
        assert constant["lengths"][1] == pytest.approx(1.003, abs=0.002)


class TestTwoUnits:
    """Independent units find one direction; deflation separates them."""

    def test_two_independent_units_on_the_four_close_to_five_degrees(self) -> None:
        body = walked(FOUR_PEOPLE, rule="oja", n_components=2, max_epochs=20)
        gaps = [angle_between(*epoch["weights"]) for epoch in body["epochs"]]

        assert gaps[0] == pytest.approx(76.3, abs=0.1)
        assert all(later < earlier for earlier, later in pairwise(gaps))
        assert gaps[-1] == pytest.approx(5.25, abs=0.01)
        assert body["worst_orthogonality"] == pytest.approx(0.996, abs=0.001)

    def test_two_independent_units_on_the_crowd_coincide(self) -> None:
        body = walked(CROWD, rule="oja", n_components=2, max_epochs=20)

        assert angle_between(*body["epochs"][-1]["weights"]) < 1e-3
        assert body["worst_orthogonality"] > 0.99999

    def test_deflation_holds_the_second_unit_off_the_first(self) -> None:
        body = walked(FOUR_PEOPLE, rule="sanger", n_components=2, max_epochs=20)
        gaps = [angle_between(*epoch["weights"]) for epoch in body["epochs"]]

        assert gaps[-1] == pytest.approx(76.5, abs=0.1)
        assert body["epochs"][-1]["weights"][1]["angle_degrees"] == pytest.approx(
            14.5, abs=0.1
        )
        assert body["epochs"][-1]["weights"][1]["length"] == pytest.approx(
            0.890, abs=0.001
        )


class TestTheCrowd:
    """The second direction learns at a pace set by the spread it chases."""

    def test_the_first_direction_lands_within_a_hundredth_of_a_degree(self) -> None:
        body = learned(CROWD, DEFAULT_EPOCHS)

        assert body["angles_degrees"][0] == pytest.approx(0.0031, abs=0.0005)
        assert body["hebbian"][0]["length"] == pytest.approx(1.0, abs=1e-5)

    def test_the_second_direction_comes_back_short_and_eight_degrees_off(
        self,
    ) -> None:
        body = learned(CROWD, DEFAULT_EPOCHS)

        assert body["eigen"][1]["share"] == pytest.approx(0.0103, abs=0.0002)
        assert body["angles_degrees"][1] == pytest.approx(8.2, abs=0.1)
        assert body["hebbian"][1]["length"] == pytest.approx(0.43, abs=0.01)
        assert body["worst_orthogonality"] == pytest.approx(0.14, abs=0.01)
        assert body["epochs_run"] == DEFAULT_EPOCHS
        assert body["converged"] is False

    def test_the_angle_is_taken_modulo_the_sign(self) -> None:
        body = learned(CROWD, DEFAULT_EPOCHS)

        first, twin = body["hebbian"][0], body["eigen"][0]
        assert first["dx"] * twin["dx"] < 0
        assert body["angles_degrees"][0] < 0.01

    def test_the_first_scores_agree_and_the_second_do_not(self) -> None:
        body = learned(CROWD, DEFAULT_EPOCHS)

        for rule, eigen in zip(
            body["learned_scores"], body["eigen_scores"], strict=True
        ):
            assert abs(rule[0]) == pytest.approx(abs(eigen[0]), abs=0.002)
        assert body["learned_scores"][4][1] == pytest.approx(2.98, abs=0.01)
        assert body["eigen_scores"][4][1] == pytest.approx(1.19, abs=0.01)

    def test_the_ideal_case_leaves_the_second_direction_twenty_degrees_off(
        self,
    ) -> None:
        body = learned(IDEAL_CASE, DEFAULT_EPOCHS)

        assert body["eigen"][1]["share"] == pytest.approx(0.00074, abs=0.00002)
        assert body["angles_degrees"][0] < 0.001
        assert body["angles_degrees"][1] == pytest.approx(20.5, abs=0.1)
        assert body["hebbian"][1]["length"] == pytest.approx(0.180, abs=0.002)


class TestTheSweeps:
    """The library's fit repeated across budgets, rates and seeds."""

    def test_the_epoch_sweep_on_the_four(self) -> None:
        by_budget = {fit["max_epochs"]: fit for fit in swept(FOUR_PEOPLE)["epochs"]}

        assert list(by_budget) == [1, 2, 3, 5, 10, 20, 50, 100, 200]
        assert by_budget[1]["angles_degrees"][0] == pytest.approx(23.13, abs=0.01)
        assert by_budget[10]["angles_degrees"][0] == pytest.approx(6.89, abs=0.01)
        assert by_budget[100]["angles_degrees"][0] < 1e-5
        assert by_budget[200]["epochs_run"] == 91
        assert by_budget[200]["shares"] == pytest.approx([0.8, 0.2], abs=1e-6)

    def test_the_rate_sweep_on_the_four(self) -> None:
        by_multiplier = {
            fit["rate_multiplier"]: fit for fit in swept(FOUR_PEOPLE)["rates"]
        }

        assert by_multiplier[0.1]["angles_degrees"][0] == pytest.approx(2.04, abs=0.01)
        assert by_multiplier[0.5]["angles_degrees"][0] < 1e-5
        assert by_multiplier[1.0]["epochs_run"] == 91
        assert by_multiplier[2.0]["epochs_run"] == 31
        assert by_multiplier[4.0]["diverged"] is True
        assert by_multiplier[4.0]["angles_degrees"] is None
        assert by_multiplier[8.0]["diverged"] is True

    def test_the_rate_sweep_on_the_crowd_diverges_only_at_eight(self) -> None:
        by_multiplier = {fit["rate_multiplier"]: fit for fit in swept(CROWD)["rates"]}

        assert by_multiplier[4.0]["diverged"] is False
        assert by_multiplier[8.0]["diverged"] is True

    def test_every_seed_finds_the_diagonal_on_the_four(self) -> None:
        seeds = swept(FOUR_PEOPLE)["seeds"]

        assert [fit["seed"] for fit in seeds] == [11, 1, 2, 3, 4]
        assert all(fit["angle_degrees"] < 1e-4 for fit in seeds)
        assert all(fit["angle_to_first_seed_degrees"] < 1e-4 for fit in seeds)
        assert all(14 <= fit["epochs_run"] <= 22 for fit in seeds)

    def test_every_seed_finds_the_crowds_first_direction(self) -> None:
        seeds = swept(CROWD)["seeds"]

        assert all(fit["angle_degrees"] < 0.03 for fit in seeds)


class TestTheCircle:
    """Equal eigenvalues, where no direction is the answer."""

    def test_the_eigensolver_picks_the_axes_and_halves_the_variance(self) -> None:
        body = learned(CIRCLE, DEFAULT_EPOCHS)

        assert body["eigen"][0]["share"] == pytest.approx(0.5, abs=1e-9)
        assert body["eigen"][1]["share"] == pytest.approx(0.5, abs=1e-9)
        assert abs(body["eigen"][0]["dx"]) == pytest.approx(1.0, abs=1e-9)

    def test_the_rule_settles_at_length_one_somewhere_else(self) -> None:
        body = learned(CIRCLE, DEFAULT_EPOCHS)

        assert body["angles_degrees"][0] == pytest.approx(80.1, abs=0.1)
        assert body["hebbian"][0]["share"] == pytest.approx(0.5, abs=1e-9)
        assert body["hebbian"][1]["share"] == pytest.approx(0.5, abs=1e-9)
        assert body["hebbian"][0]["length"] == pytest.approx(1.0005, abs=1e-4)
        assert body["worst_orthogonality"] < 1e-3
        assert body["converged"] is False

    def test_every_seed_lands_somewhere_different(self) -> None:
        seeds = swept(CIRCLE)["seeds"]
        gaps = [fit["angle_to_first_seed_degrees"] for fit in seeds[1:]]

        assert gaps == pytest.approx([21.4, 8.6, 56.9, 38.6], abs=0.1)
        assert all(fit["length"] == pytest.approx(1.0005, abs=1e-4) for fit in seeds)


class TestUncentred:
    """What the rule learns when the rows are positions rather than deviations."""

    def test_on_the_four_it_learns_the_direction_to_the_mean(self) -> None:
        body = walked(FOUR_PEOPLE, max_epochs=20, centre=False)
        final = body["epochs"][-1]["weights"][0]

        assert body["starting_rate"] == pytest.approx(1.30e-5, rel=0.01)
        assert final["angle_to_mean_degrees"] == pytest.approx(0.083, abs=0.005)
        assert final["angle_degrees"] == pytest.approx(23.12, abs=0.01)
        assert final["length"] == pytest.approx(1.0, abs=1e-4)

    def test_on_the_crowd_the_same(self) -> None:
        body = walked(CROWD, max_epochs=20, centre=False)
        final = body["epochs"][-1]["weights"][0]

        assert final["angle_to_mean_degrees"] == pytest.approx(0.74, abs=0.01)
        assert final["angle_degrees"] == pytest.approx(22.1, abs=0.1)


class TestRefusals:
    def test_a_lone_person_is_refused(self) -> None:
        response = client.post(
            LEARN, json={"points": FOUR_PEOPLE[:1], "max_epochs": 10}
        )

        assert response.status_code == 422

    def test_two_people_are_enough_for_one_direction(self) -> None:
        body = learned(FOUR_PEOPLE[:2], DEFAULT_EPOCHS)

        assert body["angles_degrees"][0] < 1e-5
        assert body["hebbian"][0]["share"] == pytest.approx(1.0, abs=1e-9)
        assert body["eigen"][1]["variance"] == pytest.approx(0.0, abs=1e-9)

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

    def test_a_zero_rate_multiplier_is_refused(self) -> None:
        response = client.post(
            LEARN,
            json={"points": FOUR_PEOPLE, "max_epochs": 10, "rate_multiplier": 0.0},
        )

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            LEARN, json={"points": FOUR_PEOPLE, "max_epochs": 10, "seed": 3}
        )

        assert response.status_code == 422

    def test_a_cloud_with_no_spread_is_refused_by_the_eigensolver_first(
        self,
    ) -> None:
        # Every person on one spot has no direction to find. The endpoint
        # fits the eigensolver first, so its refusal is the one that comes
        # back; the rule's own fit refuses the same cloud by another name.
        response = client.post(
            LEARN,
            json={"points": [{"x": 170, "y": 68}] * 3, "max_epochs": 10},
        )

        assert response.status_code == 422
        assert response.json()["error"] == "InvalidValuesError"

        with pytest.raises(AllSameValuesError):
            HebbianPrincipalComponents().fit(
                [Feature("height", [170.0] * 3), Feature("weight", [68.0] * 3)]
            )

    def test_a_constant_column_fits_and_the_second_direction_shrinks(self) -> None:
        # Height is constant, so the first direction is pure weight and the
        # second unit is handed nothing to explain and only ever decays.
        body = learned([{"x": 170, "y": y} for y in (58, 63, 73, 78)], DEFAULT_EPOCHS)

        assert body["angles_degrees"][0] < 1e-9
        assert abs(body["hebbian"][0]["dy"]) == pytest.approx(1.0, abs=1e-9)
        assert body["eigen"][1]["variance"] == 0.0
        assert body["hebbian"][1]["length"] == pytest.approx(0.243, abs=0.002)
        assert body["angles_degrees"][1] == pytest.approx(22.4, abs=0.1)

    def test_a_non_finite_coordinate_is_refused(self) -> None:
        response = client.post(
            LEARN,
            content='{"points": [{"x": NaN, "y": 1}, {"x": 2, "y": 3}], "max_epochs": 10}',
            headers={"content-type": "application/json"},
        )

        assert response.status_code == 422

    def test_the_trace_refuses_a_budget_over_twenty_and_a_third_unit(self) -> None:
        over = client.post(WALK, json={"points": FOUR_PEOPLE, "max_epochs": 21})
        third = client.post(WALK, json={"points": FOUR_PEOPLE, "n_components": 3})
        unknown = client.post(WALK, json={"points": FOUR_PEOPLE, "rule": "power"})

        assert over.status_code == 422
        assert third.status_code == 422
        assert unknown.status_code == 422

    def test_the_sweeps_refuse_more_than_thirty_people(self) -> None:
        many = [{"x": 100 + i, "y": 40 + i * 0.5} for i in range(31)]

        response = client.post(SWEEPS, json={"points": many})

        assert response.status_code == 422

    def test_millimetres_change_the_rate_and_not_the_first_answer(self) -> None:
        body = learned(
            [{"x": person["x"] * 10, "y": person["y"]} for person in FOUR_PEOPLE],
            DEFAULT_EPOCHS,
        )

        assert body["starting_rate"] == pytest.approx(4.95e-5, rel=0.01)
        assert body["angles_degrees"][0] < 0.01


class TestTheLibrarysOwnContracts:
    """The refusals the page's contract table quotes that no endpoint reaches."""

    heights = Feature("height", [180.0, 160.0, 175.0, 165.0])
    weights = Feature("weight", [78.0, 58.0, 63.0, 73.0])
    schedule = ExponentialDecaySchedule(start=0.0025, end=0.000025)

    def test_a_third_direction_from_two_features_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            HebbianPrincipalComponents(n_components=3).fit([self.heights, self.weights])

    def test_reading_before_fitting_is_refused_by_name(self) -> None:
        with pytest.raises(NotFittedError):
            _ = HebbianPrincipalComponents().directions
        with pytest.raises(NotFittedError):
            HebbianPrincipalComponents().transform([self.heights, self.weights])

    def test_the_librarys_default_rate_overflows_on_centimetres(self) -> None:
        with pytest.raises(DivergenceError, match="epoch 3"):
            HebbianPrincipalComponents(random_seed=11).fit([self.heights, self.weights])

    def test_one_row_is_refused(self) -> None:
        with pytest.raises(TooFewValuesError):
            HebbianPrincipalComponents().fit(
                [Feature("height", [180.0]), Feature("weight", [78.0])]
            )

    def test_transform_matches_by_name_and_refuses_a_stranger(self) -> None:
        fitted = HebbianPrincipalComponents(
            learning_rate=self.schedule, random_seed=11
        ).fit([self.heights, self.weights])

        reordered = fitted.transform([self.weights, self.heights])
        ordered = fitted.transform([self.heights, self.weights])
        assert list(reordered[0].values) == list(ordered[0].values)

        with pytest.raises(InvalidValuesError):
            fitted.transform([self.heights, Feature("mass", [78.0, 58.0, 63.0, 73.0])])
        with pytest.raises(InvalidValuesError):
            fitted.transform([self.heights])

    def test_one_feature_fits_with_one_direction_carrying_everything(self) -> None:
        fitted = HebbianPrincipalComponents(
            learning_rate=self.schedule, random_seed=11
        ).fit([self.heights])

        assert fitted.directions.variance_shares == pytest.approx((1.0,))
