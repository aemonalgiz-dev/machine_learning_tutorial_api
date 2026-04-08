"""The pipelines endpoints, pinned to the numbers the page quotes.

The leak is measured on the family of thirty seeds starting at zero, at the
three row counts the widget draws. On the standardizer, which never reads the
target, the gap at thirty rows is -0.0076 with twelve of thirty seeds
flattered, a coin toss rather than a bias. On a column chosen by its
correlation with a target of pure noise the gap is +0.2888 with twenty-seven
of thirty flattered, and it shrinks with rows without changing sign. The degree
search picks degree 2 on the thrown ball at 0.9116, with degree 1 at -1.2250,
and finds nothing to choose between on a straight line, a spread of 0.0047.
Every one of those numbers is pinned here.
"""

from functools import cache
from itertools import pairwise
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

LEAK = "/concepts/pipelines/leak"
DEGREE_SEARCH = "/concepts/pipelines/degree-search"

FIRST_SEED = 0
ROW_COUNTS = (30, 60, 200)

THROWN_BALL = [
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

STRAIGHT_LINE = [
    {"x": 0.0, "y": 3.3},
    {"x": 0.5, "y": 3.55},
    {"x": 1.0, "y": 4.6},
    {"x": 1.5, "y": 4.85},
    {"x": 2.0, "y": 6.2},
    {"x": 2.5, "y": 6.75},
    {"x": 3.0, "y": 7.4},
    {"x": 3.5, "y": 8.55},
    {"x": 4.0, "y": 8.7},
    {"x": 4.5, "y": 9.85},
    {"x": 5.0, "y": 10.7},
    {"x": 5.5, "y": 11.05},
    {"x": 6.0, "y": 12.0},
    {"x": 6.5, "y": 13.15},
    {"x": 7.0, "y": 13.4},
]


@cache
def leak_at(n_rows: int, seed: int = FIRST_SEED) -> dict:
    """One leak measurement, fetched once per row count and reused."""
    response = client.post(LEAK, json={"n_rows": n_rows, "seed": seed})
    assert response.status_code == 200
    return response.json()


def searched(points: list[dict]) -> dict:
    response = client.post(DEGREE_SEARCH, json={"points": points})
    assert response.status_code == 200
    return response.json()


class TestTheStandardizerLeak:
    """A transformer blind to the target, on the family of seeds from zero."""

    EXPECTED: ClassVar[dict[int, tuple[float, float, float, int]]] = {
        30: (0.583649, 0.591200, -0.007552, 12),
        60: (0.819573, 0.820463, -0.000890, 10),
        200: (0.940255, 0.940357, -0.000102, 15),
    }

    @pytest.mark.parametrize("n_rows", ROW_COUNTS)
    def test_the_two_arrangements_score_as_quoted(self, n_rows: int) -> None:
        scaling = leak_at(n_rows)["scaling"]
        outside, inside, flattered, seeds = self.EXPECTED[n_rows]

        assert scaling["mean_outside"] == pytest.approx(outside, abs=1e-6)
        assert scaling["mean_inside"] == pytest.approx(inside, abs=1e-6)
        assert scaling["flattered_by"] == pytest.approx(flattered, abs=1e-6)
        assert scaling["seeds_flattered"] == seeds

    def test_the_gap_at_thirty_rows_is_a_coin_toss(self) -> None:
        """Under a hundredth in size, and twelve of thirty seeds is no bias."""
        scaling = leak_at(30)["scaling"]

        assert abs(scaling["flattered_by"]) < 0.01
        assert 10 <= scaling["seeds_flattered"] <= 20

    def test_the_gap_shrinks_in_size_as_the_rows_grow(self) -> None:
        sizes = [
            abs(leak_at(n_rows)["scaling"]["flattered_by"]) for n_rows in ROW_COUNTS
        ]

        assert all(later < earlier for earlier, later in pairwise(sizes))

    def test_thirty_seeds_are_run_and_the_means_are_theirs(self) -> None:
        body = leak_at(30)
        scaling = body["scaling"]

        assert body["n_seeds"] == 30
        assert body["first_seed"] == FIRST_SEED
        assert len(scaling["outside_scores"]) == 30
        assert len(scaling["inside_scores"]) == 30
        assert scaling["mean_outside"] == pytest.approx(
            sum(scaling["outside_scores"]) / 30
        )
        assert scaling["mean_inside"] == pytest.approx(
            sum(scaling["inside_scores"]) / 30
        )
        assert scaling["flattered_by"] == pytest.approx(
            scaling["mean_outside"] - scaling["mean_inside"]
        )

    def test_seeds_flattered_counts_the_seeds_where_outside_won(self) -> None:
        scaling = leak_at(30)["scaling"]

        counted = sum(
            outside > inside
            for outside, inside in zip(
                scaling["outside_scores"], scaling["inside_scores"], strict=True
            )
        )
        assert scaling["seeds_flattered"] == counted

    def test_the_settings_are_the_ones_the_page_names(self) -> None:
        body = leak_at(30)

        assert body["n_rows"] == 30
        assert body["n_folds"] == 5
        assert body["n_neighbours"] == 5
        assert body["n_noise_columns"] == 20

    def test_another_family_of_seeds_gives_another_family_of_scores(self) -> None:
        first = leak_at(30, seed=0)["scaling"]
        second = leak_at(30, seed=30)["scaling"]

        assert first["outside_scores"] != second["outside_scores"]
        assert abs(second["flattered_by"]) < 0.02


class TestTheTargetConsultingLeak:
    """A column chosen by its correlation with a target of pure noise."""

    EXPECTED: ClassVar[dict[int, tuple[float, float, float, int]]] = {
        30: (-0.597887, -0.886705, 0.288818, 27),
        60: (-0.269592, -0.425194, 0.155602, 27),
        200: (-0.198506, -0.239192, 0.040686, 18),
    }

    @pytest.mark.parametrize("n_rows", ROW_COUNTS)
    def test_the_two_arrangements_score_as_quoted(self, n_rows: int) -> None:
        selection = leak_at(n_rows)["selection"]
        outside, inside, flattered, seeds = self.EXPECTED[n_rows]

        assert selection["mean_outside"] == pytest.approx(outside, abs=1e-6)
        assert selection["mean_inside"] == pytest.approx(inside, abs=1e-6)
        assert selection["flattered_by"] == pytest.approx(flattered, abs=1e-6)
        assert selection["seeds_flattered"] == seeds

    def test_the_gap_is_systematic_where_the_standardizer_s_was_not(self) -> None:
        body = leak_at(30)

        assert body["selection"]["flattered_by"] > 0.25
        assert body["selection"]["seeds_flattered"] >= 25
        assert body["selection"]["flattered_by"] > body["scaling"]["flattered_by"]

    def test_the_gap_shrinks_with_rows_and_keeps_its_sign(self) -> None:
        gaps = [leak_at(n_rows)["selection"]["flattered_by"] for n_rows in ROW_COUNTS]

        assert all(gap > 0.0 for gap in gaps)
        assert all(later < earlier for earlier, later in pairwise(gaps))

    def test_nothing_is_there_to_find_in_either_arrangement(self) -> None:
        """Both means sit below zero, so the leak flatters a mirage, not a model."""
        selection = leak_at(30)["selection"]

        assert selection["mean_outside"] < 0.0
        assert selection["mean_inside"] < 0.0


class TestTheConfigurationIsNeverFitted:
    """One pipeline object, handed to one hundred and fifty fold fits."""

    def test_the_pipeline_comes_back_fitted_to_the_last_fold(self) -> None:
        assert leak_at(30)["pipeline_fitted_after_folds"] is True

    def test_its_steps_and_model_come_back_untouched(self) -> None:
        assert leak_at(30)["configuration_fitted_after_folds"] is False


class TestTheDegreeSearch:
    """Whole pipeline steps as the candidates, never a step-and-field string."""

    def test_the_thrown_ball_chooses_degree_two(self) -> None:
        body = searched(THROWN_BALL)

        assert body["degrees"] == [1, 2, 3, 4, 5]
        assert body["best_degree"] == 2
        assert body["n_candidates"] == 5
        assert body["n_folds"] == 5
        assert body["penalty"] == 0.1

    def test_the_thrown_ball_scores_as_quoted(self) -> None:
        body = searched(THROWN_BALL)

        expected = [-1.225044, 0.911600, 0.894787, 0.834413, 0.830524]
        for score, wanted in zip(body["scores"], expected, strict=True):
            assert score == pytest.approx(wanted, abs=1e-6)
        assert body["best_score"] == pytest.approx(0.911600, abs=1e-6)
        assert body["score_spread"] == pytest.approx(2.136644, abs=1e-6)

    def test_the_straight_line_finds_nothing_to_choose_between(self) -> None:
        body = searched(STRAIGHT_LINE)

        expected = [0.970257, 0.971766, 0.973388, 0.974645, 0.974932]
        for score, wanted in zip(body["scores"], expected, strict=True):
            assert score == pytest.approx(wanted, abs=1e-6)
        assert body["best_degree"] == 5
        assert body["best_score"] == pytest.approx(0.974932, abs=1e-6)
        assert body["score_spread"] == pytest.approx(0.004675, abs=1e-6)
        assert body["score_spread"] < 0.01

    def test_the_best_score_is_the_largest_and_the_spread_is_the_range(self) -> None:
        body = searched(THROWN_BALL)

        assert body["best_score"] == max(body["scores"])
        assert body["score_spread"] == pytest.approx(
            max(body["scores"]) - min(body["scores"])
        )


class TestWhatIsRefused:
    def test_too_few_rows_for_the_leak_is_refused(self) -> None:
        response = client.post(LEAK, json={"n_rows": 29, "seed": 0})

        assert response.status_code == 422

    def test_too_many_rows_for_the_leak_is_refused(self) -> None:
        response = client.post(LEAK, json={"n_rows": 201, "seed": 0})

        assert response.status_code == 422

    def test_a_seed_past_the_ceiling_is_refused(self) -> None:
        response = client.post(LEAK, json={"n_rows": 30, "seed": 10_001})

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(LEAK, json={"n_rows": 30, "seed": 0, "n_seeds": 3})

        assert response.status_code == 422

    def test_too_few_points_to_deal_five_folds_is_refused_in_the_library_s_words(
        self,
    ) -> None:
        response = client.post(DEGREE_SEARCH, json={"points": THROWN_BALL[:4]})

        assert response.status_code == 422
        assert response.json()["error"] == "TooFewValuesError"

    def test_a_fold_of_one_row_is_refused_in_the_library_s_words(self) -> None:
        """Eight points deal folds of one row, whose R squared has no variance."""
        response = client.post(DEGREE_SEARCH, json={"points": THROWN_BALL[:8]})

        assert response.status_code == 422
        assert response.json()["error"] == "UndefinedMetricError"
