"""The feature-importance endpoints, pinned to the page's quoted numbers.

The page works one draw of the parity puzzle, three hundred rows under seed 4,
which is the library's own parity fixture, with the forest and the scrambles
seeded at zero as the library's specs seed them. The lone tree scores 0.537 on
its rows and 0.497 held out, and both measures hand the distractor most of the
explanation, 0.807 by impurity and 0.710 by scrambling. The forest scores
0.993 and 0.960, impurity still hands the distractor 0.519, and scrambling
drops it to 0.018 on the training rows and to exactly zero held out.

The ledger grows the lone tree on the site's tangled crowd, eight splits, and
the running totals it reports assemble the shares 0.777 and 0.223 a split at
a time. The forest beside it is the random-forests page's seed-7 committee,
whose impurity shares 0.527 and 0.473 that page already quotes.

The drawn crowd is two hundred people with a raffle ticket and a second
reading of height beside their measurements, fitted six ways. Its numbers
are the page's evidence for the biases, and are pinned here so the page and
the fixture cannot drift apart.
"""

from functools import cache
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

MEASURE = "/concepts/feature-importance/measure"
LEDGER = "/concepts/feature-importance/ledger"
DRAWN = "/concepts/feature-importance/drawn-crowd"
EDGES = "/concepts/feature-importance/edges"

WORKED = {"n_rows": 300, "seed": 4}

FIT_ORDER = ["first", "second", "distractor"]

TANGLED_CROWD = [
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


@cache
def measured() -> dict:
    """The worked parity draw, fitted once for the whole module."""
    response = client.post(MEASURE, json=WORKED)
    assert response.status_code == 200
    return response.json()


@cache
def ledgered(max_features: int = 1) -> dict:
    """The tangled crowd's ledger, at the forest's feature restriction."""
    response = client.post(
        LEDGER, json={"points": TANGLED_CROWD, "max_features": max_features}
    )
    assert response.status_code == 200
    return response.json()


@cache
def drawn() -> dict:
    """The drawn crowd, computed once."""
    response = client.get(DRAWN)
    assert response.status_code == 200
    return response.json()


def arrangement(name: str) -> dict:
    return next(one for one in drawn()["arrangements"] if one["name"] == name)


def share_of(shares: list[dict], name: str) -> float:
    return next(entry["share"] for entry in shares if entry["name"] == name)


def drop_of(reading: dict, name: str) -> float:
    return next(step["drop"] for step in reading["steps"] if step["name"] == name)


def count_of(counts: list[dict], name: str) -> int:
    return next(entry["count"] for entry in counts if entry["name"] == name)


def normalised_drops(reading: dict) -> list[float]:
    clamped = [max(0.0, step["drop"]) for step in reading["steps"]]
    total = sum(clamped)
    return [value / total for value in clamped]


class TestTheDraw:
    def test_the_features_come_in_the_fit_order(self) -> None:
        body = measured()

        assert body["n_rows"] == 300
        assert body["feature_names"] == FIT_ORDER
        for report in (body["lone_tree"], body["forest"]):
            assert [entry["name"] for entry in report["impurity"]] == FIT_ORDER
            for rows in ("training", "held_out"):
                assert [entry["name"] for entry in report[rows]["steps"]] == FIT_ORDER
        assert [entry["name"] for entry in body["lone_tree"]["credits"]] == FIT_ORDER

    def test_every_reported_reading_sums_to_one(self) -> None:
        body = measured()

        for report in (body["lone_tree"], body["forest"]):
            assert sum(entry["share"] for entry in report["impurity"]) == pytest.approx(
                1.0
            )
            for rows in ("training", "held_out"):
                assert sum(
                    entry["share"] for entry in report[rows]["shares"]
                ) == pytest.approx(1.0)


class TestTheLoneTree:
    """The model that does not work, on which the two measures agree."""

    def test_the_accuracies_the_page_quotes(self) -> None:
        tree = measured()["lone_tree"]

        assert tree["training_accuracy"] == pytest.approx(0.5367, abs=5e-4)
        assert tree["held_out_accuracy"] == pytest.approx(0.4967, abs=5e-4)
        assert tree["training"]["intact_score"] == tree["training_accuracy"]
        assert tree["held_out"]["intact_score"] == tree["held_out_accuracy"]

    def test_both_measures_hand_the_distractor_most_of_the_explanation(self) -> None:
        tree = measured()["lone_tree"]

        assert share_of(tree["impurity"], "distractor") == pytest.approx(
            0.807, abs=5e-4
        )
        assert share_of(tree["training"]["shares"], "distractor") == pytest.approx(
            0.710, abs=5e-4
        )
        assert tree["leading_by_impurity"] == "distractor"
        assert tree["training"]["leading"] == "distractor"

    def test_the_rest_goes_to_second_and_first_earns_nothing(self) -> None:
        tree = measured()["lone_tree"]

        assert share_of(tree["impurity"], "second") == pytest.approx(0.193, abs=5e-4)
        assert share_of(tree["training"]["shares"], "second") == pytest.approx(
            0.290, abs=5e-4
        )
        assert share_of(tree["impurity"], "first") == 0.0
        assert share_of(tree["training"]["shares"], "first") == 0.0

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

    def test_the_credits_summed_by_feature_normalise_to_the_shares(self) -> None:
        tree = measured()["lone_tree"]

        assert [credit["credit"] for credit in tree["credits"]] == pytest.approx(
            [0.0, 1.3889, 5.8098], abs=5e-5
        )
        assert tree["credit_total"] == pytest.approx(7.1987, abs=5e-5)
        for credit, share in zip(tree["credits"], tree["impurity"], strict=True):
            assert share["share"] == pytest.approx(
                credit["credit"] / tree["credit_total"]
            )

    def test_the_training_scramble_ladder_the_page_quotes(self) -> None:
        reading = measured()["lone_tree"]["training"]

        assert [step["shuffled_score"] for step in reading["steps"]] == pytest.approx(
            [0.5367, 0.5233, 0.5040], abs=5e-4
        )
        assert [step["drop"] for step in reading["steps"]] == pytest.approx(
            [0.0, 0.0133, 0.0327], abs=5e-4
        )
        assert reading["drop_total"] == pytest.approx(0.0460, abs=5e-4)

    def test_held_out_a_column_can_leave_the_tree_better_off(self) -> None:
        """At chance there is little to lose, and one drop comes back negative."""
        reading = measured()["lone_tree"]["held_out"]

        assert drop_of(reading, "second") == pytest.approx(-0.0080, abs=5e-4)
        assert drop_of(reading, "distractor") == pytest.approx(0.0113, abs=5e-4)
        assert reading["drop_total"] == pytest.approx(0.0113, abs=5e-4)
        assert share_of(reading["shares"], "second") == 0.0
        assert share_of(reading["shares"], "distractor") == 1.0
        assert reading["refusal"] is None


class TestTheForest:
    """The model that works, on which the two measures part company."""

    def test_the_accuracies_the_page_quotes(self) -> None:
        forest = measured()["forest"]

        assert forest["training_accuracy"] == pytest.approx(0.9933, abs=5e-4)
        assert forest["held_out_accuracy"] == pytest.approx(0.960, abs=5e-4)

    def test_impurity_still_credits_the_distractor_with_half(self) -> None:
        impurity = measured()["forest"]["impurity"]

        assert share_of(impurity, "distractor") == pytest.approx(0.519, abs=5e-4)
        assert share_of(impurity, "first") == pytest.approx(0.202, abs=5e-4)
        assert share_of(impurity, "second") == pytest.approx(0.280, abs=5e-4)
        assert share_of(impurity, "distractor") > share_of(
            impurity, "first"
        ) + share_of(impurity, "second")

    def test_scrambling_the_training_rows_drops_the_distractor_to_almost_nothing(
        self,
    ) -> None:
        reading = measured()["forest"]["training"]

        assert [step["shuffled_score"] for step in reading["steps"]] == pytest.approx(
            [0.538, 0.5147, 0.9767], abs=5e-4
        )
        assert [step["drop"] for step in reading["steps"]] == pytest.approx(
            [0.4553, 0.4787, 0.0167], abs=5e-4
        )
        assert reading["drop_total"] == pytest.approx(0.9507, abs=5e-4)
        assert share_of(reading["shares"], "distractor") == pytest.approx(
            0.018, abs=5e-4
        )
        assert share_of(reading["shares"], "first") == pytest.approx(0.479, abs=5e-4)
        assert share_of(reading["shares"], "second") == pytest.approx(0.504, abs=5e-4)

    def test_scrambling_the_held_out_rows_drops_it_to_exactly_zero(self) -> None:
        reading = measured()["forest"]["held_out"]

        assert reading["intact_score"] == pytest.approx(0.960, abs=5e-4)
        assert [step["drop"] for step in reading["steps"]] == pytest.approx(
            [0.4687, 0.4500, -0.0040], abs=5e-4
        )
        assert reading["drop_total"] == pytest.approx(0.9187, abs=5e-4)
        assert share_of(reading["shares"], "distractor") == 0.0
        assert share_of(reading["shares"], "first") == pytest.approx(0.510, abs=5e-4)
        assert share_of(reading["shares"], "second") == pytest.approx(0.490, abs=5e-4)

    def test_the_two_measures_disagree_about_the_leader(self) -> None:
        forest = measured()["forest"]

        assert forest["leading_by_impurity"] == "distractor"
        assert forest["training"]["leading"] == "second"
        assert forest["held_out"]["leading"] == "first"


class TestTheWalkAgreesWithTheLibrary:
    """The endpoint's recorded scrambles are the library's own draws."""

    @pytest.mark.parametrize("model", ["lone_tree", "forest"])
    @pytest.mark.parametrize("rows", ["training", "held_out"])
    def test_the_clamped_drops_normalise_to_the_shares(
        self, model: str, rows: str
    ) -> None:
        reading = measured()[model][rows]

        assert normalised_drops(reading) == pytest.approx(
            [entry["share"] for entry in reading["shares"]]
        )
        assert reading["drop_total"] == pytest.approx(
            sum(max(0.0, step["drop"]) for step in reading["steps"])
        )


class TestTheLedger:
    """The site's crowd, one split at a time."""

    def test_the_tree_the_page_steps_through(self) -> None:
        tree = ledgered()["tree"]

        assert (tree["depth"], tree["n_leaves"], tree["n_splits"]) == (5, 9, 8)
        assert tree["training_accuracy"] == 1.0
        assert [
            (
                split["order"],
                split["feature"],
                split["threshold"],
                split["depth"],
                split["n_rows"],
            )
            for split in tree["splits"]
        ] == [
            (1, "height", 147.5, 0, 25),
            (2, "weight", 52.5, 1, 9),
            (3, "height", 158.0, 1, 16),
            (4, "weight", 56.0, 2, 3),
            (5, "height", 150.5, 2, 10),
            (6, "weight", 51.5, 3, 8),
            (7, "height", 152.5, 4, 4),
            (8, "height", 152.0, 4, 4),
        ]

    def test_the_gains_and_credits_the_page_quotes(self) -> None:
        splits = ledgered()["tree"]["splits"]

        assert [split["gain"] for split in splits] == pytest.approx(
            [0.1881, 0.0494, 0.0750, 0.4444, 0.0800, 0.1250, 0.3750, 0.3750], abs=5e-5
        )
        assert [split["credit"] for split in splits] == pytest.approx(
            [4.7022, 0.4444, 1.2000, 1.3333, 0.8000, 1.0000, 1.5000, 1.5000], abs=5e-5
        )
        assert splits[0]["impurity"] == pytest.approx(0.4992, abs=5e-5)
        for split in splits:
            assert split["credit"] == pytest.approx(split["n_rows"] * split["gain"])

    def test_the_largest_gain_earns_the_fourth_largest_credit(self) -> None:
        """Split 4 buys 0.444 on three people; the root buys 0.188 on twenty-five."""
        splits = ledgered()["tree"]["splits"]

        assert max(splits, key=lambda split: split["gain"])["order"] == 4
        assert max(splits, key=lambda split: split["credit"])["order"] == 1

    def test_the_running_totals_end_at_the_shares(self) -> None:
        tree = ledgered()["tree"]

        first, second, last = tree["splits"][0], tree["splits"][1], tree["splits"][-1]
        assert [share["share"] for share in first["running_shares"]] == [1.0, 0.0]
        assert [share["share"] for share in second["running_shares"]] == pytest.approx(
            [0.9136, 0.0864], abs=5e-5
        )
        assert [
            credit["credit"] for credit in last["running_credits"]
        ] == pytest.approx([9.7022, 2.7778], abs=5e-5)
        assert [credit["credit"] for credit in tree["credits"]] == pytest.approx(
            [9.7022, 2.7778], abs=5e-5
        )
        assert tree["credit_total"] == pytest.approx(12.48, abs=5e-5)
        assert [share["share"] for share in tree["impurity"]] == pytest.approx(
            [0.7774, 0.2226], abs=5e-5
        )
        assert [share["share"] for share in last["running_shares"]] == pytest.approx(
            [share["share"] for share in tree["impurity"]]
        )

    def test_each_running_share_is_the_running_credit_over_the_running_total(
        self,
    ) -> None:
        for split in ledgered()["tree"]["splits"]:
            total = sum(credit["credit"] for credit in split["running_credits"])
            for credit, share in zip(
                split["running_credits"], split["running_shares"], strict=True
            ):
                assert share["share"] == pytest.approx(credit["credit"] / total)

    def test_the_memorising_tree_scrambled_on_its_own_rows(self) -> None:
        reading = ledgered()["tree"]["training"]

        assert reading["intact_score"] == 1.0
        assert [step["drop"] for step in reading["steps"]] == pytest.approx(
            [0.408, 0.232], abs=5e-4
        )
        assert [share["share"] for share in reading["shares"]] == pytest.approx(
            [0.6375, 0.3625], abs=5e-4
        )

    def test_the_forest_is_the_random_forests_pages_seed_seven_committee(self) -> None:
        forest = ledgered()["forest"]

        assert (forest["n_members"], forest["max_features"]) == (25, 1)
        assert forest["training_accuracy"] == pytest.approx(0.96, abs=5e-4)
        assert forest["out_of_bag_accuracy"] == pytest.approx(0.52, abs=5e-4)
        assert [credit["credit"] for credit in forest["roots"]] == [8.0, 17.0]
        assert [share["share"] for share in forest["impurity"]] == pytest.approx(
            [0.5271, 0.4729], abs=5e-4
        )
        assert [
            share["share"] for share in forest["training"]["shares"]
        ] == pytest.approx([0.5949, 0.4051], abs=5e-4)

    def test_the_forest_averages_its_members_shares(self) -> None:
        forest = ledgered()["forest"]

        heights = [shares[0]["share"] for shares in forest["member_shares"]]
        assert len(heights) == 25
        assert min(heights) == pytest.approx(0.179, abs=5e-4)
        assert max(heights) == 1.0
        assert sum(heights) / len(heights) == pytest.approx(
            share_of(forest["impurity"], "height")
        )

    def test_offering_both_features_is_bagging_and_leans_on_height(self) -> None:
        forest = ledgered(max_features=2)["forest"]

        assert [credit["credit"] for credit in forest["roots"]] == [20.0, 5.0]
        assert forest["out_of_bag_accuracy"] == pytest.approx(0.64, abs=5e-4)
        assert [share["share"] for share in forest["impurity"]] == pytest.approx(
            [0.7111, 0.2889], abs=5e-4
        )
        assert [
            share["share"] for share in forest["training"]["shares"]
        ] == pytest.approx([0.75, 0.25], abs=5e-4)


class TestTheDrawnCrowd:
    """Two hundred people, a raffle ticket, a second reading of height."""

    def test_the_crowd_and_its_columns(self) -> None:
        body = drawn()

        assert (body["n_rows"], body["n_held_out"]) == (200, 200)
        assert body["column_names"] == ["height", "weight", "ticket", "height_again"]
        assert body["adult_share"] == pytest.approx(0.465)
        correlations = {
            entry["name"]: entry["credit"] for entry in body["correlations_with_height"]
        }
        assert correlations["weight"] == pytest.approx(0.898, abs=5e-4)
        assert correlations["ticket"] == pytest.approx(-0.131, abs=5e-4)
        assert correlations["height_again"] == pytest.approx(0.996, abs=5e-4)
        assert [one["name"] for one in body["arrangements"]] == [
            "measurements",
            "with_ticket",
            "with_twin",
            "all_four",
            "weight_only",
            "ticket_only",
        ]

    def test_three_questions_get_three_answers_on_the_measurements_forest(self) -> None:
        forest = arrangement("measurements")["forest"]

        assert (
            count_of(forest["split_counts"], "height"),
            count_of(forest["split_counts"], "weight"),
        ) == (240, 272)
        assert share_of(forest["impurity"], "weight") == pytest.approx(0.521, abs=5e-4)
        assert forest["leading_by_impurity"] == "weight"
        assert share_of(forest["held_out"]["shares"], "height") == pytest.approx(
            0.957, abs=5e-4
        )
        assert forest["held_out"]["leading"] == "height"
        assert forest["training_accuracy"] == 1.0
        assert forest["held_out_accuracy"] == pytest.approx(0.88, abs=5e-4)
        assert forest["out_of_bag_accuracy"] == pytest.approx(0.89, abs=5e-4)

    def test_the_lone_tree_memorises_and_its_two_readings_differ_by_the_rows(
        self,
    ) -> None:
        tree = arrangement("measurements")["tree"]

        assert (tree["n_splits"], tree["depth"]) == (16, 7)
        assert tree["training_accuracy"] == 1.0
        assert tree["held_out_accuracy"] == pytest.approx(0.845, abs=5e-4)
        assert [share["share"] for share in tree["impurity"]] == pytest.approx(
            [0.8965, 0.1035], abs=5e-4
        )
        assert [
            share["share"] for share in tree["training"]["shares"]
        ] == pytest.approx([0.8903, 0.1097], abs=5e-4)
        assert drop_of(tree["held_out"], "weight") == pytest.approx(-0.03, abs=5e-4)
        assert [share["share"] for share in tree["held_out"]["shares"]] == [1.0, 0.0]

    def test_the_lone_tree_never_asks_about_the_ticket(self) -> None:
        tree = arrangement("with_ticket")["tree"]

        assert count_of(tree["split_counts"], "ticket") == 0
        assert share_of(tree["impurity"], "ticket") == 0.0
        assert tree["held_out_accuracy"] == pytest.approx(0.845, abs=5e-4)

    def test_the_forest_offered_one_column_at_a_time_credits_the_ticket(self) -> None:
        forest = arrangement("with_ticket")["forest"]

        assert count_of(forest["split_counts"], "ticket") == 196
        assert count_of(forest["split_counts"], "height") == 189
        assert forest["n_splits"] == 559
        assert share_of(forest["impurity"], "ticket") == pytest.approx(0.1065, abs=5e-4)
        assert share_of(forest["training"]["shares"], "ticket") == pytest.approx(
            0.0836, abs=5e-4
        )
        assert drop_of(forest["held_out"], "ticket") == pytest.approx(-0.012, abs=5e-4)
        assert share_of(forest["held_out"]["shares"], "ticket") == 0.0
        assert forest["held_out_accuracy"] == pytest.approx(0.87, abs=5e-4)

    def test_a_second_reading_of_height_splits_the_credit_and_hides_the_first(
        self,
    ) -> None:
        alone = arrangement("measurements")["forest"]
        twinned = arrangement("with_twin")["forest"]

        assert share_of(twinned["impurity"], "height") == pytest.approx(
            0.2566, abs=5e-4
        )
        assert share_of(twinned["impurity"], "height_again") == pytest.approx(
            0.5103, abs=5e-4
        )
        assert drop_of(alone["held_out"], "height") == pytest.approx(0.36, abs=5e-4)
        assert drop_of(twinned["held_out"], "height") == pytest.approx(0.148, abs=5e-4)
        assert drop_of(twinned["held_out"], "height_again") == pytest.approx(
            0.184, abs=5e-4
        )
        assert drop_of(twinned["held_out"], "height") + drop_of(
            twinned["held_out"], "height_again"
        ) < drop_of(alone["held_out"], "height")
        assert share_of(twinned["held_out"]["shares"], "height") == pytest.approx(
            0.434, abs=5e-4
        )
        assert twinned["held_out_accuracy"] == pytest.approx(0.875, abs=5e-4)

    def test_the_lone_tree_with_the_twin_leans_on_one_of_them_held_out(self) -> None:
        tree = arrangement("with_twin")["tree"]

        assert count_of(tree["split_counts"], "height_again") == 5
        assert share_of(tree["impurity"], "height_again") == pytest.approx(
            0.1007, abs=5e-4
        )
        assert drop_of(tree["held_out"], "height_again") == pytest.approx(
            -0.001, abs=5e-4
        )
        assert share_of(tree["held_out"]["shares"], "height") == 1.0

    def test_all_four_columns_offered_two_at_a_time(self) -> None:
        forest = arrangement("all_four")["forest"]

        assert arrangement("all_four")["max_features"] == 2
        assert count_of(forest["split_counts"], "ticket") == 77
        assert share_of(forest["impurity"], "ticket") == pytest.approx(0.0431, abs=5e-4)
        assert share_of(forest["held_out"]["shares"], "ticket") == 0.0
        assert share_of(forest["held_out"]["shares"], "height") == pytest.approx(
            0.609, abs=5e-4
        )
        assert forest["held_out_accuracy"] == pytest.approx(0.87, abs=5e-4)

    def test_weight_alone_predicts_usefully_with_no_role_in_the_rule(self) -> None:
        weight_only = arrangement("weight_only")

        assert weight_only["tree"]["held_out_accuracy"] == pytest.approx(
            0.755, abs=5e-4
        )
        assert weight_only["forest"]["held_out_accuracy"] == pytest.approx(
            0.76, abs=5e-4
        )
        assert weight_only["forest"]["held_out"]["shares"] == [
            {"name": "weight", "share": 1.0}
        ]
        assert drop_of(weight_only["forest"]["held_out"], "weight") == pytest.approx(
            0.254, abs=5e-4
        )

    def test_the_ticket_alone_is_memorised_and_worth_nothing_held_out(self) -> None:
        ticket_only = arrangement("ticket_only")

        tree = ticket_only["tree"]
        assert (tree["n_splits"], tree["depth"]) == (109, 21)
        assert tree["training_accuracy"] == 1.0
        assert tree["held_out_accuracy"] == pytest.approx(0.5, abs=5e-4)
        assert drop_of(tree["training"], "ticket") == pytest.approx(0.5, abs=5e-4)
        assert tree["training"]["shares"] == [{"name": "ticket", "share": 1.0}]
        assert drop_of(tree["held_out"], "ticket") == pytest.approx(-0.008, abs=5e-4)
        assert tree["held_out"]["shares"] is None
        assert "No feature earned anything" in tree["held_out"]["refusal"]

        forest = ticket_only["forest"]
        assert forest["held_out_accuracy"] == pytest.approx(0.505, abs=5e-4)
        assert forest["out_of_bag_accuracy"] == pytest.approx(0.42, abs=5e-4)
        assert forest["held_out"]["shares"] is None

    def test_refitting_without_the_ticket_costs_nothing_held_out(self) -> None:
        with_ticket = arrangement("with_ticket")
        without = arrangement("measurements")

        assert (
            with_ticket["tree"]["held_out_accuracy"]
            == without["tree"]["held_out_accuracy"]
        )
        assert without["forest"]["held_out_accuracy"] - with_ticket["forest"][
            "held_out_accuracy"
        ] == pytest.approx(0.01, abs=5e-4)

    def test_the_same_request_answers_identically(self) -> None:
        assert client.get(DRAWN).json() == drawn()


class TestTheEdges:
    """What the library does at each boundary, as the page's table reports."""

    EXPECTED: ClassVar[dict[str, tuple[str, str | None]]] = {
        "a feature of no values": ("refused", "EmptyValuesError"),
        "one row": ("refused", "SingleClassError"),
        "one class only": ("refused", "SingleClassError"),
        "a non-finite value": ("refused", "InvalidValuesError"),
        "every column constant, then the split reading": (
            "refused",
            "InvalidValuesError",
        ),
        "one constant column beside two real ones": ("accepted", None),
        "every column constant, then the scramble reading": (
            "refused",
            "InvalidValuesError",
        ),
        "zero scramble repeats": ("refused", "ValidationError"),
        "the split reading before fit": ("refused", "NotFittedError"),
        "the scramble reading before fit": ("refused", "NotFittedError"),
        "a scramble with a column renamed": ("refused", "InvalidValuesError"),
        "a scramble with the columns reordered": ("accepted", None),
        "a share asked for by a name the fit never saw": (
            "refused",
            "InvalidValuesError",
        ),
        "two columns of one name": ("refused", "NonUniqueFeaturesError"),
        "more features per split than there are features": ("accepted", None),
        "a committee whose members have no splits to read": (
            "refused",
            "InvalidValuesError",
        ),
    }

    def test_every_edge_behaves_as_the_page_says(self) -> None:
        probes = {probe["edge"]: probe for probe in client.get(EDGES).json()["probes"]}

        assert set(probes) == set(self.EXPECTED)
        for edge, (outcome, error) in self.EXPECTED.items():
            assert (probes[edge]["outcome"], probes[edge]["error"]) == (
                outcome,
                error,
            ), edge

    def test_the_details_the_page_quotes(self) -> None:
        probes = {
            probe["edge"]: probe["detail"]
            for probe in client.get(EDGES).json()["probes"]
        }

        assert "nothing to discriminate between" in probes["one row"]
        assert (
            "No feature earned anything"
            in probes["every column constant, then the split reading"]
        )
        assert probes["one constant column beside two real ones"].endswith(
            "share is 0.0"
        )
        assert (
            "expected features height, weight; got stature, weight"
            in probes["a scramble with a column renamed"]
        )
        assert probes["a scramble with the columns reordered"].endswith(
            "weight 0.000, height 1.000"
        )
        assert (
            "Known features: height, weight"
            in probes["a share asked for by a name the fit never saw"]
        )
        assert (
            "use PermutationImportance instead"
            in probes["a committee whose members have no splits to read"]
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
        assert client.post(MEASURE, json={"n_rows": 199, "seed": 4}).status_code == 422

    def test_too_many_rows_is_refused(self) -> None:
        assert client.post(MEASURE, json={"n_rows": 401, "seed": 4}).status_code == 422

    def test_a_seed_past_the_ceiling_is_refused(self) -> None:
        assert (
            client.post(MEASURE, json={"n_rows": 300, "seed": 10000}).status_code == 422
        )

    def test_an_unknown_field_is_refused(self) -> None:
        assert client.post(MEASURE, json={**WORKED, "n_members": 5}).status_code == 422

    def test_a_forest_member_that_never_split_is_refused_in_words(self) -> None:
        """Seed 8 hands member 16 a column its root finds no gain in."""
        response = client.post(MEASURE, json={"n_rows": 300, "seed": 8})

        assert response.status_code == 422
        assert "grew no split" in response.json()["detail"]
        assert "member 16" in response.json()["detail"]

    def test_a_ledger_of_one_class_is_refused_by_the_library(self) -> None:
        one_class = [{**point, "label": 1} for point in TANGLED_CROWD]
        response = client.post(LEDGER, json={"points": one_class})

        assert response.status_code == 422
        assert response.json()["error"] == "SingleClassError"

    def test_a_ledger_of_too_few_people_is_refused(self) -> None:
        assert (
            client.post(LEDGER, json={"points": TANGLED_CROWD[:3]}).status_code == 422
        )

    def test_a_ledger_too_deep_or_too_wide_is_refused(self) -> None:
        assert (
            client.post(
                LEDGER, json={"points": TANGLED_CROWD, "max_depth": 9}
            ).status_code
            == 422
        )
        assert (
            client.post(
                LEDGER, json={"points": TANGLED_CROWD, "max_features": 3}
            ).status_code
            == 422
        )
        assert (
            client.post(
                LEDGER, json={"points": TANGLED_CROWD, "n_members": 51}
            ).status_code
            == 422
        )

    def test_a_shallow_ledger_stops_where_it_is_told(self) -> None:
        body = client.post(
            LEDGER, json={"points": TANGLED_CROWD, "max_depth": 1}
        ).json()

        assert body["tree"]["n_splits"] == 1
        assert body["tree"]["splits"][0]["feature"] == "height"
        assert [share["share"] for share in body["tree"]["impurity"]] == [1.0, 0.0]
