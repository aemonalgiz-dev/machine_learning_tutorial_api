"""The grid-search endpoints, pinned to the page's hand arithmetic.

The page works twelve people along a line. Under the fixed seeds three are
held out, the other nine are dealt into three folds, and two neighbours wins
among one to five, its second fold scoring 1 - 5.25/86 by hand. The honest
score on the three held-out people is 1 - 18.5/(218/3), which is lower than
the winner's own number. The noise section quotes one draw of the library's
optimism measurement, a winner reporting a positive R squared on a target
nothing can learn, and the gap averaged over twelve draws. All pinned.
"""

from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.grid_search import DRAWS, NOISE_CANDIDATES, NOISE_SEED
from oop_ml_api.main import create_app

client = TestClient(create_app())

SEARCH = "/concepts/grid-search/search"
OPTIMISM = "/concepts/grid-search/optimism"

TWELVE_ALONG_A_LINE = [
    {"x": 0.5, "y": 2},
    {"x": 1.4, "y": 3},
    {"x": 2.6, "y": 6},
    {"x": 3.1, "y": 5},
    {"x": 4.7, "y": 10},
    {"x": 5.2, "y": 11},
    {"x": 6.8, "y": 13},
    {"x": 7.3, "y": 16},
    {"x": 8.9, "y": 17},
    {"x": 9.6, "y": 20},
    {"x": 10.4, "y": 21},
    {"x": 11.5, "y": 22},
]

FIVE_CANDIDATES = {
    "points": TWELVE_ALONG_A_LINE,
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

    # The deal the page describes: the held-out people sit at x = 4.7, 8.9 and
    # 11.5, and the nine searched people fall into three folds of three.
    HELD_OUT: ClassVar[list[int]] = [4, 8, 11]
    FOLD_OF: ClassVar[list[int | None]] = [0, 1, 0, 2, None, 1, 2, 1, None, 0, 2, None]

    def test_the_held_out_people_are_the_three_the_page_names(self) -> None:
        assert searched()["held_out_indices"] == self.HELD_OUT

    def test_the_deal_is_the_one_the_page_describes(self) -> None:
        assert searched()["fold_of"] == self.FOLD_OF

    def test_the_second_fold_at_two_neighbours_is_the_hand_worked_score(
        self,
    ) -> None:
        # The second fold judges x = 1.4, 5.2 and 7.3, whose targets 3, 11 and
        # 16 average 10, so the total sum of squares is 49 + 1 + 36 = 86. The
        # two nearest searched neighbours predict 4, 9 and 16.5, squared
        # errors 1, 4 and 0.25, a residual sum of 5.25.
        two_neighbours = searched()["candidates"][1]

        assert two_neighbours["n_neighbours"] == 2
        assert two_neighbours["fold_scores"][1] == pytest.approx(1 - 5.25 / 86)

    def test_the_other_two_folds_at_two_neighbours(self) -> None:
        # First fold, residuals 4 + 4 + 2.25 against a total of 536/3; third
        # fold, residuals 0.25 + 0.25 + 9 against a total of 128.
        fold_scores = searched()["candidates"][1]["fold_scores"]

        assert fold_scores[0] == pytest.approx(1 - 10.25 / (536 / 3))
        assert fold_scores[2] == pytest.approx(1 - 9.5 / 128)

    def test_every_candidate_score_is_the_mean_of_its_folds(self) -> None:
        for candidate in searched()["candidates"]:
            assert len(candidate["fold_scores"]) == 3
            assert candidate["score"] == pytest.approx(
                sum(candidate["fold_scores"]) / 3
            )

    def test_the_five_candidate_scores_the_page_quotes(self) -> None:
        scores = [candidate["score"] for candidate in searched()["candidates"]]

        assert [
            candidate["n_neighbours"] for candidate in searched()["candidates"]
        ] == [
            1,
            2,
            3,
            4,
            5,
        ]
        assert scores == pytest.approx(
            [0.9115, 0.9358, 0.7869, 0.6725, 0.3092], abs=5e-5
        )

    def test_two_neighbours_wins(self) -> None:
        body = searched()

        assert body["best_n_neighbours"] == 2
        assert body["best_score"] == pytest.approx(
            (1 - 10.25 / (536 / 3) + 1 - 5.25 / 86 + 1 - 9.5 / 128) / 3
        )

    def test_the_spread_is_the_winner_minus_the_worst(self) -> None:
        body = searched()
        scores = [candidate["score"] for candidate in body["candidates"]]

        assert body["score_spread"] == pytest.approx(max(scores) - min(scores))
        assert body["score_spread"] == pytest.approx(0.6266, abs=5e-5)

    def test_the_mean_candidate_score(self) -> None:
        body = searched()
        scores = [candidate["score"] for candidate in body["candidates"]]

        assert body["mean_candidate_score"] == pytest.approx(sum(scores) / 5)
        assert body["mean_candidate_score"] == pytest.approx(0.7232, abs=5e-5)

    def test_the_honest_score_is_the_hand_worked_three_person_score(self) -> None:
        # The winner refitted on all nine searched people predicts 8, 20.5 and
        # 20.5 for the held-out targets 10, 17 and 22, a residual sum of 18.5
        # against a total sum of squares of 218/3 around their mean 49/3.
        body = searched()

        assert body["honest_score"] == pytest.approx(1 - 18.5 / (218 / 3))
        assert body["honest_score"] == pytest.approx(0.7454, abs=5e-5)
        assert body["honest_score"] < body["best_score"]

    def test_more_neighbours_than_a_training_fold_holds_is_refused(self) -> None:
        response = client.post(SEARCH, json={**FIVE_CANDIDATES, "n_neighbours_to": 7})

        assert response.status_code == 422
        assert "neighbours" in response.json()["detail"]

    def test_an_empty_range_of_neighbour_counts_is_refused(self) -> None:
        response = client.post(
            SEARCH,
            json={**FIVE_CANDIDATES, "n_neighbours_from": 4, "n_neighbours_to": 2},
        )

        assert response.status_code == 422
        assert "empty" in response.json()["detail"]

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(SEARCH, json={**FIVE_CANDIDATES, "n_neighbors": 3})

        assert response.status_code == 422


def quoted_draw() -> dict:
    response = client.post(OPTIMISM, json={"seed": NOISE_SEED})
    assert response.status_code == 200
    return response.json()


class TestTheNoiseOptimism:
    """The draw the page quotes, on a target nothing can learn."""

    def test_the_winner_reports_a_positive_score_on_pure_noise(self) -> None:
        body = quoted_draw()

        assert body["best_n_neighbours"] == 17
        assert body["best_score"] > 0.0
        assert body["best_score"] == pytest.approx(0.0146, abs=5e-5)

    def test_the_mean_candidate_and_the_re_score_the_page_quotes(self) -> None:
        body = quoted_draw()

        assert body["mean_candidate_score"] == pytest.approx(-0.1738, abs=5e-5)
        assert body["refolded_score"] == pytest.approx(-0.0915, abs=5e-5)

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
        assert body["mean_optimism_over_draws"] > 0.0

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

    def test_the_spread_is_the_winner_minus_the_worst(self) -> None:
        body = quoted_draw()
        scores = [candidate["score"] for candidate in body["candidates"]]

        assert body["score_spread"] == pytest.approx(max(scores) - min(scores))
        assert body["mean_candidate_score"] == pytest.approx(sum(scores) / len(scores))

    def test_another_draw_is_a_different_picture(self) -> None:
        response = client.post(OPTIMISM, json={"seed": NOISE_SEED + 1})

        assert response.status_code == 200
        assert response.json()["seed"] == NOISE_SEED + 1
        assert response.json()["best_score"] != quoted_draw()["best_score"]

    def test_a_seed_beyond_the_bound_is_refused(self) -> None:
        response = client.post(OPTIMISM, json={"seed": 10_001})

        assert response.status_code == 422
