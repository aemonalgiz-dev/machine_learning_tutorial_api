"""The grid-search endpoints, pinned to the page's hand arithmetic.

The page works twelve people measured by height and weight. Under the fixed
seeds three are held out, the other nine are dealt into three folds, and two
neighbours wins among one to five, its second fold scoring 1 - 2.5/182 by
hand. The honest score on the three held-out people is lower than the
winner's own number, and the winner refitted and scored on the searched rows
is higher still. Two copies of one setting through one unseeded search tie
exactly, a misspelt dial is refused by name, a two-dial grid costs the
product of its ranges times the folds, the noise section quotes one draw of
the library's optimism measurement with its nested correction, and the
nested loop on the twelve gives a lower figure than the flat search. All
pinned.
"""

from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.grid_search import DRAWS, NOISE_CANDIDATES, NOISE_SEED
from oop_ml_api.main import create_app

client = TestClient(create_app())

SEARCH = "/concepts/grid-search/search"
SAME_FOLDS = "/concepts/grid-search/same-folds"
SPACE = "/concepts/grid-search/space"
GRID = "/concepts/grid-search/grid"
OPTIMISM = "/concepts/grid-search/optimism"
NESTED = "/concepts/grid-search/nested"

TWELVE_PEOPLE = [
    {"x": 150, "y": 50},
    {"x": 153, "y": 52},
    {"x": 157, "y": 55},
    {"x": 159, "y": 57},
    {"x": 162, "y": 61},
    {"x": 165, "y": 63},
    {"x": 168, "y": 66},
    {"x": 171, "y": 71},
    {"x": 174, "y": 72},
    {"x": 177, "y": 76},
    {"x": 180, "y": 78},
    {"x": 183, "y": 82},
]

FIVE_CANDIDATES = {
    "points": TWELVE_PEOPLE,
    "n_neighbours_from": 1,
    "n_neighbours_to": 5,
    "n_folds": 3,
}


def searched() -> dict:
    response = client.post(SEARCH, json=FIVE_CANDIDATES)
    assert response.status_code == 200
    return response.json()


class TestTheWorkedSearch:
    """Five neighbour counts on twelve people, the page's hand arithmetic."""

    # The deal the page describes: the held-out people are the ones 162, 174
    # and 183 centimetres tall, and the nine searched people fall into three
    # folds of three, each fold listed in the order its predictions come back.
    HELD_OUT: ClassVar[list[int]] = [4, 8, 11]
    FOLD_MEMBERS: ClassVar[list[list[int]]] = [[9, 0, 2], [1, 5, 7], [6, 3, 10]]
    FOLD_OF: ClassVar[list[int | None]] = [0, 1, 0, 2, None, 1, 2, 1, None, 0, 2, None]

    def test_the_held_out_people_are_the_three_the_page_names(self) -> None:
        assert searched()["held_out_indices"] == self.HELD_OUT

    def test_the_deal_is_the_one_the_page_describes(self) -> None:
        body = searched()

        assert body["fold_members"] == self.FOLD_MEMBERS
        assert body["fold_of"] == self.FOLD_OF

    def test_the_second_fold_at_two_neighbours_is_the_hand_worked_score(
        self,
    ) -> None:
        # The second fold judges the people at 153, 165 and 171 centimetres,
        # whose weights 52, 63 and 71 average 62, so the total sum of squares
        # is 100 + 1 + 81 = 182. Their two nearest searched neighbours predict
        # 52.5, 61.5 and 71, squared misses 0.25, 2.25 and 0, a residual sum
        # of 2.5.
        two_neighbours = searched()["candidates"][1]

        assert two_neighbours["n_neighbours"] == 2
        assert two_neighbours["fold_predictions"][1] == [52.5, 61.5, 71.0]
        assert two_neighbours["fold_scores"][1] == pytest.approx(1 - 2.5 / 182)

    def test_the_other_two_folds_at_two_neighbours(self) -> None:
        # First fold, the people at 177, 150 and 157 centimetres weighing 76,
        # 50 and 55, predicted 74.5, 54.5 and 54.5; third fold, the people at
        # 168, 159 and 180 weighing 66, 57 and 78, predicted 67, 53.5 and 73.5.
        two_neighbours = searched()["candidates"][1]

        assert two_neighbours["fold_predictions"][0] == [74.5, 54.5, 54.5]
        assert two_neighbours["fold_predictions"][2] == [67.0, 53.5, 73.5]
        assert two_neighbours["fold_scores"][0] == pytest.approx(0.9402, abs=5e-5)
        assert two_neighbours["fold_scores"][2] == pytest.approx(0.8491, abs=5e-5)

    def test_every_candidate_score_is_the_mean_of_its_folds(self) -> None:
        for candidate in searched()["candidates"]:
            assert len(candidate["fold_scores"]) == 3
            assert candidate["score"] == pytest.approx(
                sum(candidate["fold_scores"]) / 3
            )

    def test_the_five_candidate_scores_the_page_quotes(self) -> None:
        body = searched()

        assert [candidate["n_neighbours"] for candidate in body["candidates"]] == [
            1,
            2,
            3,
            4,
            5,
        ]
        assert [candidate["score"] for candidate in body["candidates"]] == (
            pytest.approx([0.8944, 0.9252, 0.7695, 0.6010, 0.2448], abs=5e-5)
        )

    def test_the_training_score_at_one_neighbour_is_exactly_one(self) -> None:
        # Every person's nearest neighbour among the searched rows is that
        # person, so a fit that chose the dial by its own score would always
        # choose one neighbour.
        scores = [candidate["training_score"] for candidate in searched()["candidates"]]

        assert scores[0] == 1.0
        assert scores == pytest.approx([1.0, 0.9758, 0.9560, 0.9258, 0.8564], abs=5e-5)
        assert scores == sorted(scores, reverse=True)

    def test_two_neighbours_wins_and_the_ranking_keeps_the_losers(self) -> None:
        body = searched()

        assert body["best_n_neighbours"] == 2
        assert body["best_score"] == pytest.approx(0.9252, abs=5e-5)
        assert body["ranked_n_neighbours"] == [2, 1, 3, 4, 5]
        assert body["n_candidates"] == 5
        assert body["n_fits"] == 15

    def test_the_spread_and_the_mean_candidate(self) -> None:
        body = searched()
        scores = [candidate["score"] for candidate in body["candidates"]]

        assert body["score_spread"] == pytest.approx(max(scores) - min(scores))
        assert body["score_spread"] == pytest.approx(0.6804, abs=5e-5)
        assert body["mean_candidate_score"] == pytest.approx(sum(scores) / 5)
        assert body["mean_candidate_score"] == pytest.approx(0.6870, abs=5e-5)

    def test_the_honest_score_sits_between_the_two_scores_it_must_not_be(
        self,
    ) -> None:
        # The winner refitted on all nine searched people and scored on the
        # three held out earns 0.8720; scored on the nine it was fitted to it
        # earns 0.9758, which is the training score wearing the wrong name.
        body = searched()

        assert body["honest_score"] == pytest.approx(0.8720, abs=5e-5)
        assert body["same_rows_score"] == pytest.approx(0.9758, abs=5e-5)
        assert body["honest_score"] < body["best_score"] < body["same_rows_score"]

    def test_the_one_neighbour_probe_scores_its_own_rows_at_exactly_one(
        self,
    ) -> None:
        response = client.post(SEARCH, json={**FIVE_CANDIDATES, "n_neighbours_to": 1})

        assert response.status_code == 200
        assert response.json()["same_rows_score"] == 1.0
        assert response.json()["honest_score"] == pytest.approx(0.7825, abs=5e-5)

    def test_more_neighbours_than_a_training_fold_holds_is_refused(self) -> None:
        response = client.post(SEARCH, json={**FIVE_CANDIDATES, "n_neighbours_to": 7})

        assert response.status_code == 422
        assert "7 neighbours" in response.json()["detail"]

    def test_an_empty_range_of_neighbour_counts_is_refused(self) -> None:
        response = client.post(
            SEARCH,
            json={**FIVE_CANDIDATES, "n_neighbours_from": 4, "n_neighbours_to": 2},
        )

        assert response.status_code == 422
        assert "empty" in response.json()["detail"]

    def test_an_unknown_field_is_refused_at_the_door(self) -> None:
        response = client.post(SEARCH, json={**FIVE_CANDIDATES, "n_neighbors": 3})

        assert response.status_code == 422

    def test_more_folds_than_searched_rows_is_refused(self) -> None:
        response = client.post(SEARCH, json={**FIVE_CANDIDATES, "n_folds": 10})

        assert response.status_code == 422
        assert "10 folds" in response.json()["detail"]

    def test_a_fold_of_one_person_is_refused_for_having_no_spread(self) -> None:
        # Four people leave one held out and three searched, so three folds
        # hold one person each and a person has no spread to explain.
        response = client.post(
            SEARCH, json={**FIVE_CANDIDATES, "points": TWELVE_PEOPLE[:4]}
        )

        assert response.status_code == 422
        assert "zero variance" in response.json()["detail"]

    def test_two_people_is_refused_before_a_fold_can_be_dealt(self) -> None:
        response = client.post(
            SEARCH,
            json={**FIVE_CANDIDATES, "points": TWELVE_PEOPLE[:2], "n_folds": 2},
        )

        assert response.status_code == 422
        assert "got 1" in response.json()["detail"]

    def test_a_constant_weight_is_refused_and_a_constant_height_is_not(self) -> None:
        flat_weights = [{"x": person["x"], "y": 60} for person in TWELVE_PEOPLE]
        flat_heights = [{"x": 170, "y": person["y"]} for person in TWELVE_PEOPLE]

        refused = client.post(SEARCH, json={**FIVE_CANDIDATES, "points": flat_weights})
        accepted = client.post(
            SEARCH,
            json={**FIVE_CANDIDATES, "points": flat_heights, "n_neighbours_to": 2},
        )

        assert refused.status_code == 422
        assert accepted.status_code == 200
        assert accepted.json()["best_score"] < 0.0


class TestTheSameFolds:
    """Two copies of one setting tie exactly; fresh deals do not."""

    def test_two_copies_through_one_unseeded_search_tie_to_the_last_bit(
        self,
    ) -> None:
        response = client.post(
            SAME_FOLDS,
            json={"points": TWELVE_PEOPLE, "n_neighbours": 2, "n_folds": 3},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["scores_tie"] is True
        assert body["pinned_scores"][0] == body["pinned_scores"][1]

    def test_the_same_setting_on_six_fresh_deals_spreads_widely(self) -> None:
        body = client.post(
            SAME_FOLDS,
            json={"points": TWELVE_PEOPLE, "n_neighbours": 2, "n_folds": 3},
        ).json()

        assert body["fresh_deal_seeds"] == [0, 1, 2, 3, 4, 5]
        assert body["fresh_deal_scores"] == pytest.approx(
            [0.7909, 0.5832, 0.6013, 0.9252, -9.0911, 0.4413], abs=5e-5
        )
        assert body["fresh_deal_spread"] == pytest.approx(10.0163, abs=5e-5)

    def test_the_seeded_deal_the_search_uses_is_among_the_fresh_ones(self) -> None:
        # Seed 3 is the page's fold seed, so that deal's score is the search's
        # own score for two neighbours.
        body = client.post(
            SAME_FOLDS,
            json={"points": TWELVE_PEOPLE, "n_neighbours": 2, "n_folds": 3},
        ).json()

        assert body["fresh_deal_scores"][3] == pytest.approx(searched()["best_score"])

    def test_a_setting_larger_than_a_training_fold_is_refused(self) -> None:
        response = client.post(
            SAME_FOLDS,
            json={"points": TWELVE_PEOPLE, "n_neighbours": 9, "n_folds": 3},
        )

        assert response.status_code == 422


class TestBuildingTheSpace:
    """A misspelt dial is refused before any data is seen."""

    def test_the_right_name_builds_one_candidate_per_value(self) -> None:
        body = client.post(
            SPACE, json={"parameter_name": "n_neighbours", "values": [1, 2, 3]}
        ).json()

        assert body["declared_fields"] == ["metric", "n_neighbours"]
        assert body["parameter_names"] == ["n_neighbours"]
        assert body["n_candidates"] == 3
        assert [one["built_n_neighbours"] for one in body["candidates"]] == [1, 2, 3]

    def test_a_misspelt_name_is_refused_and_the_message_names_the_fields(
        self,
    ) -> None:
        response = client.post(
            SPACE, json={"parameter_name": "n_neighbors", "values": [1, 2, 3]}
        )

        assert response.status_code == 422
        assert response.json()["error"] == "InvalidValuesError"
        assert "'n_neighbors'" in response.json()["detail"]
        assert "'n_neighbours'" in response.json()["detail"]

    def test_an_empty_range_is_refused(self) -> None:
        response = client.post(
            SPACE, json={"parameter_name": "n_neighbours", "values": []}
        )

        assert response.status_code == 422
        assert "empty" in response.json()["detail"]

    def test_a_value_the_constructor_refuses_is_caught_when_rebuilt(self) -> None:
        response = client.post(
            SPACE, json={"parameter_name": "n_neighbours", "values": [0, 1]}
        )

        assert response.status_code == 422
        assert "greater than 0" in response.json()["detail"]


class TestTheTwoDialGrid:
    """Degree times penalty, every combination, on the searched nine."""

    THREE_BY_FOUR: ClassVar[dict] = {
        "points": TWELVE_PEOPLE,
        "degrees": [1, 2, 3],
        "penalties": [0.01, 0.1, 1.0, 10.0],
        "n_folds": 3,
    }

    def test_the_grid_is_the_product_and_the_fits_are_that_times_the_folds(
        self,
    ) -> None:
        body = client.post(GRID, json=self.THREE_BY_FOUR).json()

        assert body["n_candidates"] == 12
        assert body["n_fits"] == 36
        assert len(body["cells"]) == 12
        assert [cell["degree"] for cell in body["cells"][:4]] == [1, 1, 1, 1]
        assert [cell["penalty"] for cell in body["cells"][:4]] == [0.01, 0.1, 1.0, 10.0]

    def test_the_winner_the_page_quotes(self) -> None:
        body = client.post(GRID, json=self.THREE_BY_FOUR).json()

        assert body["best_degree"] == 1
        assert body["best_penalty"] == 0.1
        assert body["best_score"] == pytest.approx(0.9883, abs=5e-5)
        assert body["worst_score"] == pytest.approx(0.5304, abs=5e-5)
        assert body["score_spread"] == pytest.approx(0.4579, abs=5e-5)
        assert body["honest_score"] == pytest.approx(0.9957, abs=5e-5)

    def test_a_bigger_grid_costs_proportionally_more(self) -> None:
        body = client.post(
            GRID,
            json={
                **self.THREE_BY_FOUR,
                "degrees": [1, 2, 3, 4],
                "penalties": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            },
        ).json()

        assert body["n_candidates"] == 24
        assert body["n_fits"] == 72
        assert body["best_degree"] == 1
        assert body["best_penalty"] == 0.1

    def test_a_degree_or_penalty_outside_the_page_is_refused_at_the_door(
        self,
    ) -> None:
        degree = client.post(GRID, json={**self.THREE_BY_FOUR, "degrees": [5]})
        penalty = client.post(GRID, json={**self.THREE_BY_FOUR, "penalties": [0.0]})

        assert degree.status_code == 422
        assert penalty.status_code == 422

    def test_a_constant_height_is_refused_by_the_scaler(self) -> None:
        response = client.post(
            GRID,
            json={
                **self.THREE_BY_FOUR,
                "points": [{"x": 170, "y": person["y"]} for person in TWELVE_PEOPLE],
            },
        )

        assert response.status_code == 422
        assert response.json()["error"] == "AllSameValuesError"


def quoted_draw() -> dict:
    response = client.post(OPTIMISM, json={"seed": NOISE_SEED})
    assert response.status_code == 200
    return response.json()


class TestTheNoiseOptimism:
    """The draw the page quotes, on a target nothing can learn."""

    def test_the_winner_reports_a_positive_score_on_pure_noise(self) -> None:
        body = quoted_draw()

        assert body["target"] == "noise"
        assert body["n_candidates"] == NOISE_CANDIDATES == 25
        assert body["best_n_neighbours"] == 17
        assert body["best_score"] > 0.0
        assert body["best_score"] == pytest.approx(0.0146, abs=5e-5)

    def test_the_mean_candidate_and_the_re_score_the_page_quotes(self) -> None:
        body = quoted_draw()

        assert body["mean_candidate_score"] == pytest.approx(-0.1738, abs=5e-5)
        assert body["refolded_score"] == pytest.approx(-0.0915, abs=5e-5)
        assert body["score_spread"] == pytest.approx(1.3448, abs=5e-5)

    def test_the_optimism_is_the_winner_minus_its_re_score(self) -> None:
        body = quoted_draw()

        assert body["optimism"] == pytest.approx(
            body["best_score"] - body["refolded_score"]
        )
        assert body["optimism"] == pytest.approx(0.1061, abs=5e-5)

    def test_the_gap_averaged_over_twelve_draws(self) -> None:
        body = quoted_draw()

        assert body["n_draws"] == DRAWS == 12
        assert body["mean_optimism_over_draws"] == pytest.approx(0.0728, abs=5e-5)

    def test_the_nested_loop_reports_a_negative_figure_on_the_same_draw(
        self,
    ) -> None:
        body = quoted_draw()

        assert body["nested_score"] == pytest.approx(-0.1222, abs=5e-5)
        assert body["nested_score"] < 0.0 < body["best_score"]
        assert body["nested_winners"] == [24, 22, 13, 21, 23]

    def test_twenty_five_candidates_one_per_neighbour_count(self) -> None:
        candidates = quoted_draw()["candidates"]

        assert [one["n_neighbours"] for one in candidates] == list(
            range(1, NOISE_CANDIDATES + 1)
        )
        for candidate in candidates:
            assert len(candidate["fold_scores"]) == 5
            assert candidate["score"] == pytest.approx(
                sum(candidate["fold_scores"]) / 5
            )

    def test_the_optimism_grows_with_the_candidate_count(self) -> None:
        means = [
            client.post(
                OPTIMISM, json={"seed": NOISE_SEED, "n_candidates": count}
            ).json()["mean_optimism_over_draws"]
            for count in (2, 5, 10, 25)
        ]

        assert means == pytest.approx([0.0153, 0.0479, 0.0500, 0.0728], abs=5e-5)
        assert means == sorted(means)

    def test_two_candidates_on_this_draw_land_the_wrong_side_of_zero(self) -> None:
        # One draw's optimism can be negative; the mean over twelve is not.
        body = client.post(
            OPTIMISM, json={"seed": NOISE_SEED, "n_candidates": 2}
        ).json()

        assert body["optimism"] == pytest.approx(-0.0878, abs=5e-5)
        assert body["mean_optimism_over_draws"] > 0.0

    def test_a_target_with_signal_shrinks_the_optimism_without_removing_it(
        self,
    ) -> None:
        body = client.post(
            OPTIMISM, json={"seed": NOISE_SEED, "target": "signal"}
        ).json()

        assert body["best_n_neighbours"] == 3
        assert body["best_score"] == pytest.approx(0.7386, abs=5e-5)
        assert body["refolded_score"] == pytest.approx(0.7050, abs=5e-5)
        assert body["optimism"] == pytest.approx(0.0337, abs=5e-5)
        assert body["mean_optimism_over_draws"] == pytest.approx(0.0185, abs=5e-5)
        assert body["nested_score"] == pytest.approx(0.7225, abs=5e-5)
        assert body["score_spread"] == pytest.approx(0.3226, abs=5e-5)

    def test_another_draw_is_a_different_picture(self) -> None:
        response = client.post(OPTIMISM, json={"seed": NOISE_SEED + 1})

        assert response.status_code == 200
        assert response.json()["seed"] == NOISE_SEED + 1
        assert response.json()["best_score"] != quoted_draw()["best_score"]

    def test_a_seed_or_count_beyond_the_bound_is_refused(self) -> None:
        assert client.post(OPTIMISM, json={"seed": 10_001}).status_code == 422
        assert (
            client.post(OPTIMISM, json={"seed": 1, "n_candidates": 26}).status_code
            == 422
        )


class TestTheNestedLoop:
    """A second layer of folds round the whole search, on the twelve."""

    def test_every_outer_fold_searched_and_scored_on_rows_it_never_saw(
        self,
    ) -> None:
        body = client.post(NESTED, json={"points": TWELVE_PEOPLE}).json()

        assert [fold["held_out_indices"] for fold in body["outer_folds"]] == [
            [5, 7, 8],
            [0, 1, 4],
            [6, 9, 10],
            [2, 3, 11],
        ]
        assert [fold["best_n_neighbours"] for fold in body["outer_folds"]] == [
            1,
            1,
            1,
            1,
        ]
        assert [fold["honest_score"] for fold in body["outer_folds"]] == (
            pytest.approx([-0.0274, 0.2718, 0.3105, 0.9094], abs=5e-5)
        )

    def test_the_nested_figure_is_below_the_flat_search_and_says_so(self) -> None:
        body = client.post(NESTED, json={"points": TWELVE_PEOPLE}).json()

        assert body["mean_honest_score"] == pytest.approx(0.3661, abs=5e-5)
        assert body["honest_spread"] == pytest.approx(0.9368, abs=5e-5)
        assert body["flat_best_n_neighbours"] == 2
        assert body["flat_best_score"] == pytest.approx(0.7932, abs=5e-5)
        assert body["mean_honest_score"] < body["flat_best_score"]
        assert body["n_candidates"] == 5
        assert body["n_fits"] == 4 * (5 * 3 + 1) == 64

    def test_an_outer_fold_of_one_person_is_refused(self) -> None:
        response = client.post(
            NESTED,
            json={
                "points": TWELVE_PEOPLE[:5],
                "n_outer_folds": 5,
                "n_inner_folds": 3,
                "n_neighbours_to": 2,
            },
        )

        assert response.status_code == 422
