"""The evaluation endpoints, end to end through HTTP.

The page's whole argument is a set of numbers, the training score that only
climbs, the held-out score that turns around and collapses, and the fold
spread that widens exactly where the mean stops deserving trust. These tests
pin that argument on the fixed seeded split so the prose and the widgets
cannot drift from what the API serves.
"""

from itertools import pairwise

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

NOISY_THROW: list[dict] = [
    {"x": 0.0, "y": 0.7},
    {"x": 0.29, "y": 5.2},
    {"x": 0.57, "y": 9.5},
    {"x": 0.86, "y": 13.0},
    {"x": 1.14, "y": 16.9},
    {"x": 1.43, "y": 18.1},
    {"x": 1.71, "y": 19.4},
    {"x": 2.0, "y": 21.0},
    {"x": 2.29, "y": 19.6},
    {"x": 2.57, "y": 19.1},
    {"x": 2.86, "y": 16.8},
    {"x": 3.14, "y": 13.6},
    {"x": 3.43, "y": 11.1},
    {"x": 3.71, "y": 6.2},
    {"x": 4.0, "y": 1.2},
]


class TestSplitFit:
    """One degree fitted on the training share and judged twice."""

    def test_the_seeded_split_is_stable(self) -> None:
        response = client.post(
            "/concepts/evaluation/split-fit",
            json={"points": NOISY_THROW, "degree": 2},
        )

        body = response.json()
        assert body["held_out_indices"] == [0, 1, 10, 12]

    def test_an_honest_degree_scores_well_on_both_shares(self) -> None:
        response = client.post(
            "/concepts/evaluation/split-fit",
            json={"points": NOISY_THROW, "degree": 2},
        )

        body = response.json()
        assert round(body["train_r_squared"], 3) == 0.995
        assert round(body["held_out_r_squared"], 3) == 0.989

    def test_a_wild_degree_flatters_only_the_share_it_saw(self) -> None:
        response = client.post(
            "/concepts/evaluation/split-fit",
            json={"points": NOISY_THROW, "degree": 9},
        )

        body = response.json()
        assert body["train_r_squared"] > 0.99
        assert body["held_out_r_squared"] < -1000.0

    def test_a_degree_the_training_share_cannot_pin_is_refused(self) -> None:
        response = client.post(
            "/concepts/evaluation/split-fit",
            json={"points": NOISY_THROW[:8], "degree": 6},
        )

        assert response.status_code == 422
        assert "training share" in response.json()["detail"]


class TestDegreeCurve:
    """The two score curves the gap chart draws."""

    def curve(self) -> dict:
        response = client.post(
            "/concepts/evaluation/degree-curve", json={"points": NOISY_THROW}
        )
        assert response.status_code == 200
        return response.json()

    def test_the_training_score_never_falls(self) -> None:
        scores = self.curve()["train_scores"]
        assert all(later >= earlier - 1e-9 for earlier, later in pairwise(scores))

    def test_the_held_out_score_turns_around(self) -> None:
        body = self.curve()
        held = dict(zip(body["degrees"], body["held_out_scores"], strict=True))
        assert held[1] < 0.0
        assert held[2] > 0.98
        assert held[9] < -1000.0

    def test_both_curves_share_the_degrees(self) -> None:
        body = self.curve()
        assert body["degrees"] == list(range(1, 10))
        assert len(body["train_scores"]) == len(body["held_out_scores"]) == 9


class TestValidationCurve:
    """Five folds per degree, and the spread that prices the mean."""

    def validations(self) -> list[dict]:
        response = client.post(
            "/concepts/evaluation/validation-curve", json={"points": NOISY_THROW}
        )
        assert response.status_code == 200
        return response.json()["validations"]

    def by_degree(self) -> dict[int, dict]:
        return {entry["degree"]: entry for entry in self.validations()}

    def test_the_honest_degrees_agree_with_the_split(self) -> None:
        entries = self.by_degree()
        assert round(entries[2]["mean_r_squared"], 3) == 0.989
        assert entries[2]["spread"] < 0.05

    def test_a_wild_degree_is_priced_by_its_spread(self) -> None:
        entries = self.by_degree()
        assert entries[8]["mean_r_squared"] < 0.0
        assert entries[8]["spread"] > 100 * entries[2]["spread"]

    def test_every_degree_the_folds_can_pin_is_present(self) -> None:
        degrees = [entry["degree"] for entry in self.validations()]
        assert degrees == list(range(1, 10))


class TestSimilarityCurve:
    """The kernel chart's four curves, labelled by the library itself."""

    def test_the_labels_are_the_kernels_own_descriptions(self) -> None:
        response = client.post("/concepts/kernel-trick/similarity-curve", json={})

        labels = [curve["label"] for curve in response.json()["curves"]]
        assert labels[0] == "a . b"
        assert "** 2" in labels[1]
        assert labels[2].startswith("exp(")
        assert labels[3].startswith("exp(")

    def test_only_the_radial_curves_forget_the_far_away(self) -> None:
        response = client.post("/concepts/kernel-trick/similarity-curve", json={})

        body = response.json()
        far_left = 0
        linear = body["curves"][0]["values"]
        assert abs(linear[far_left]) > 1.0
        for curve in body["curves"][2:]:
            assert abs(curve["values"][far_left]) < 1e-6

    def test_the_radial_curves_peak_at_the_reference(self) -> None:
        response = client.post("/concepts/kernel-trick/similarity-curve", json={})

        body = response.json()
        positions = body["positions"]
        for curve in body["curves"][2:]:
            peak_position = positions[curve["values"].index(max(curve["values"]))]
            assert abs(peak_position - body["reference"]) < 0.06
