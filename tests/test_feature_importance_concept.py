"""The feature-importance endpoint, pinned to the page's quoted numbers.

The page works one draw of the parity puzzle, three hundred rows under seed 4,
which is the library's own parity fixture, with the forest and the scrambles
seeded at zero as the library's specs seed them. The lone tree scores 0.537 on
its rows and 0.497 held out, and both measures hand the distractor most of the
explanation, 0.807 by impurity and 0.710 by scrambling. The forest scores
0.993 and 0.960, impurity still hands the distractor 0.519, and scrambling
drops it to 0.018. The tree's four splits and both models' scramble ladders
are the page's hand arithmetic, and both normalise to the library's shares.
"""

from functools import cache
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

MEASURE = "/concepts/feature-importance/measure"

WORKED = {"n_rows": 300, "seed": 4}

FIT_ORDER = ["first", "second", "distractor"]


@cache
def measured() -> dict:
    """The worked draw, fitted once for the whole module."""
    response = client.post(MEASURE, json=WORKED)
    assert response.status_code == 200
    return response.json()


def share_of(shares: list[dict], name: str) -> float:
    return next(entry["share"] for entry in shares if entry["name"] == name)


def normalised_drops(shuffles: list[dict]) -> list[float]:
    clamped = [max(0.0, step["drop"]) for step in shuffles]
    total = sum(clamped)
    return [value / total for value in clamped]


class TestTheDraw:
    def test_the_features_come_in_the_fit_order(self) -> None:
        body = measured()

        assert body["n_rows"] == 300
        assert body["feature_names"] == FIT_ORDER
        for report in (body["lone_tree"], body["forest"]):
            for reading in ("impurity", "permutation", "shuffles"):
                assert [entry["name"] for entry in report[reading]] == FIT_ORDER
        assert [entry["name"] for entry in body["lone_tree"]["credits"]] == FIT_ORDER

    def test_every_reading_sums_to_one(self) -> None:
        body = measured()

        for report in (body["lone_tree"], body["forest"]):
            for reading in ("impurity", "permutation"):
                assert sum(
                    entry["share"] for entry in report[reading]
                ) == pytest.approx(1.0)


class TestTheLoneTree:
    """The model that does not work, on which the two measures agree."""

    def test_the_accuracies_the_page_quotes(self) -> None:
        tree = measured()["lone_tree"]

        assert tree["training_accuracy"] == pytest.approx(0.5367, abs=5e-4)
        assert tree["held_out_accuracy"] == pytest.approx(0.4967, abs=5e-4)
        assert tree["intact_score"] == tree["training_accuracy"]

    def test_both_measures_hand_the_distractor_most_of_the_explanation(self) -> None:
        tree = measured()["lone_tree"]

        assert share_of(tree["impurity"], "distractor") == pytest.approx(
            0.807, abs=5e-4
        )
        assert share_of(tree["permutation"], "distractor") == pytest.approx(
            0.710, abs=5e-4
        )
        assert share_of(tree["impurity"], "distractor") > 0.6
        assert share_of(tree["permutation"], "distractor") > 0.6

    def test_the_rest_goes_to_second_and_first_earns_nothing(self) -> None:
        tree = measured()["lone_tree"]

        assert share_of(tree["impurity"], "second") == pytest.approx(0.193, abs=5e-4)
        assert share_of(tree["permutation"], "second") == pytest.approx(0.290, abs=5e-4)
        assert share_of(tree["impurity"], "first") == 0.0
        assert share_of(tree["permutation"], "first") == 0.0

    def test_both_measures_name_the_distractor(self) -> None:
        tree = measured()["lone_tree"]

        assert tree["leading_by_impurity"] == "distractor"
        assert tree["leading_by_permutation"] == "distractor"

    def test_the_four_splits_the_page_sums_by_hand(self) -> None:
        splits = measured()["lone_tree"]["splits"]

        assert [
            (split["feature"], split["depth"], split["n_rows"]) for split in splits
        ] == [
            ("distractor", 0, 300),
            ("distractor", 1, 295),
            ("second", 2, 12),
            ("distractor", 2, 283),
        ]
        assert [split["gain"] for split in splits] == pytest.approx(
            [0.008362, 0.005087, 0.115741, 0.006362], abs=5e-6
        )
        assert [split["credit"] for split in splits] == pytest.approx(
            [2.5086, 1.5006, 1.3889, 1.8006], abs=5e-5
        )

    def test_each_credit_is_rows_times_gain(self) -> None:
        for split in measured()["lone_tree"]["splits"]:
            assert split["credit"] == pytest.approx(split["n_rows"] * split["gain"])

    def test_the_credits_summed_by_feature(self) -> None:
        tree = measured()["lone_tree"]

        assert [credit["credit"] for credit in tree["credits"]] == pytest.approx(
            [0.0, 1.3889, 5.8098], abs=5e-5
        )
        assert tree["credit_total"] == pytest.approx(7.1987, abs=5e-5)
        assert tree["credit_total"] == pytest.approx(
            sum(split["credit"] for split in tree["splits"])
        )

    def test_the_credits_normalise_to_the_impurity_shares(self) -> None:
        tree = measured()["lone_tree"]

        for credit, share in zip(tree["credits"], tree["impurity"], strict=True):
            assert credit["name"] == share["name"]
            assert share["share"] == pytest.approx(
                credit["credit"] / tree["credit_total"]
            )

    def test_the_scramble_ladder_the_page_quotes(self) -> None:
        tree = measured()["lone_tree"]

        assert [step["shuffled_score"] for step in tree["shuffles"]] == pytest.approx(
            [0.5367, 0.5233, 0.5040], abs=5e-4
        )
        assert [step["drop"] for step in tree["shuffles"]] == pytest.approx(
            [0.0, 0.0133, 0.0327], abs=5e-4
        )
        assert tree["drop_total"] == pytest.approx(0.0460, abs=5e-4)


class TestTheForest:
    """The model that works, on which the two measures part company."""

    def test_the_accuracies_the_page_quotes(self) -> None:
        forest = measured()["forest"]

        assert forest["training_accuracy"] == pytest.approx(0.9933, abs=5e-4)
        assert forest["held_out_accuracy"] == pytest.approx(0.960, abs=5e-4)
        assert forest["intact_score"] == forest["training_accuracy"]

    def test_impurity_still_credits_the_distractor_with_half(self) -> None:
        forest = measured()["forest"]

        assert share_of(forest["impurity"], "distractor") == pytest.approx(
            0.519, abs=5e-4
        )
        assert share_of(forest["impurity"], "distractor") > 0.4
        assert share_of(forest["impurity"], "first") == pytest.approx(0.202, abs=5e-4)
        assert share_of(forest["impurity"], "second") == pytest.approx(0.280, abs=5e-4)

    def test_the_distractor_outscores_the_two_real_columns_combined(self) -> None:
        impurity = measured()["forest"]["impurity"]

        assert share_of(impurity, "distractor") > share_of(
            impurity, "first"
        ) + share_of(impurity, "second")

    def test_scrambling_drops_the_distractor_to_almost_nothing(self) -> None:
        forest = measured()["forest"]

        assert share_of(forest["permutation"], "distractor") == pytest.approx(
            0.018, abs=5e-4
        )
        assert share_of(forest["permutation"], "distractor") < 0.1
        assert share_of(forest["permutation"], "first") == pytest.approx(
            0.479, abs=5e-4
        )
        assert share_of(forest["permutation"], "second") == pytest.approx(
            0.504, abs=5e-4
        )

    def test_the_two_measures_disagree_about_the_leader(self) -> None:
        forest = measured()["forest"]

        assert forest["leading_by_impurity"] == "distractor"
        assert forest["leading_by_permutation"] == "second"

    def test_the_scramble_ladder_the_page_quotes(self) -> None:
        forest = measured()["forest"]

        assert [step["shuffled_score"] for step in forest["shuffles"]] == pytest.approx(
            [0.538, 0.5147, 0.9767], abs=5e-4
        )
        assert [step["drop"] for step in forest["shuffles"]] == pytest.approx(
            [0.4553, 0.4787, 0.0167], abs=5e-4
        )
        assert forest["drop_total"] == pytest.approx(0.9507, abs=5e-4)


class TestTheWalkAgreesWithTheLibrary:
    """The endpoint's recorded scrambles are the library's own draws."""

    @pytest.mark.parametrize("model", ["lone_tree", "forest"])
    def test_the_clamped_drops_normalise_to_the_permutation_shares(
        self, model: str
    ) -> None:
        report = measured()[model]

        assert normalised_drops(report["shuffles"]) == pytest.approx(
            [entry["share"] for entry in report["permutation"]]
        )

    @pytest.mark.parametrize("model", ["lone_tree", "forest"])
    def test_the_drop_total_is_the_clamped_sum(self, model: str) -> None:
        report = measured()[model]

        assert report["drop_total"] == pytest.approx(
            sum(max(0.0, step["drop"]) for step in report["shuffles"])
        )


class TestOtherDraws:
    """A fresh draw keeps the story, and the same draw answers the same."""

    OTHER_DRAW: ClassVar[dict] = {"n_rows": 200, "seed": 11}

    def test_the_same_request_answers_identically(self) -> None:
        first = client.post(MEASURE, json=self.OTHER_DRAW).json()
        second = client.post(MEASURE, json=self.OTHER_DRAW).json()

        assert first == second

    def test_the_forest_beats_the_lone_tree_held_out(self) -> None:
        body = client.post(MEASURE, json=self.OTHER_DRAW).json()

        assert (
            body["forest"]["held_out_accuracy"] > body["lone_tree"]["held_out_accuracy"]
        )


class TestRefusals:
    def test_too_few_rows_is_refused(self) -> None:
        response = client.post(MEASURE, json={"n_rows": 199, "seed": 4})

        assert response.status_code == 422

    def test_too_many_rows_is_refused(self) -> None:
        response = client.post(MEASURE, json={"n_rows": 401, "seed": 4})

        assert response.status_code == 422

    def test_a_seed_past_the_ceiling_is_refused(self) -> None:
        response = client.post(MEASURE, json={"n_rows": 300, "seed": 10000})

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(MEASURE, json={**WORKED, "n_members": 5})

        assert response.status_code == 422

    def test_a_forest_member_that_never_split_is_refused_in_words(self) -> None:
        """Seed 8 hands member 16 a column its root finds no gain in."""
        response = client.post(MEASURE, json={"n_rows": 300, "seed": 8})

        assert response.status_code == 422
        assert "grew no split" in response.json()["detail"]
        assert "member 16" in response.json()["detail"]
