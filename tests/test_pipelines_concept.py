"""The pipelines endpoints, pinned to the numbers the page quotes.

The chain is followed on the grid-search page's twelve people with three
held out, the standardizer's centre and spread on the training nine, what
the held-out heights became, and every check on the configuration
afterwards. The leak is measured on the family of thirty seeds starting at
zero, at the three crowd sizes the widget draws. On the standardizer, which
never reads the target, the gap at thirty people is -0.0017 with fourteen of
thirty seeds flattered, a coin toss rather than a bias. On a column chosen by
its correlation with a target of pure noise the gap is +0.2888 with
twenty-seven of thirty flattered, and it shrinks with rows without changing
sign. The search keeps degree 1 at a penalty of 0.1 on the twelve, the two
orders of the steps agree at a penalty of zero and part at every other, and
every edge in the page's table comes back as the page says. Every one of those
numbers is pinned here.
"""

from functools import cache
from itertools import pairwise
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

ANATOMY = "/concepts/pipelines/anatomy"
LEAK = "/concepts/pipelines/leak"
SEARCH = "/concepts/pipelines/search"
ORDER = "/concepts/pipelines/order"
CONTRACTS = "/concepts/pipelines/contracts"

FIRST_SEED = 0
ROW_COUNTS = (30, 60, 200)
PAGE_SEED = 4
PAGE_FOLDS = 4
MIDDLE_THREE = [1, 5, 9]
TALLEST_THREE = [9, 10, 11]

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

IDEAL_CASE = [
    {"x": 152, "y": 54.2},
    {"x": 155, "y": 55.4},
    {"x": 158, "y": 58.7},
    {"x": 161, "y": 60.5},
    {"x": 164, "y": 63.2},
    {"x": 167, "y": 66.2},
    {"x": 170, "y": 67.4},
    {"x": 173, "y": 70.7},
    {"x": 176, "y": 72.5},
    {"x": 179, "y": 75.2},
    {"x": 182, "y": 78.2},
    {"x": 185, "y": 79.4},
    {"x": 188, "y": 82.7},
    {"x": 191, "y": 84.5},
    {"x": 194, "y": 87.2},
]


@cache
def anatomy_holding_out(*held_out: int) -> dict:
    """The chain followed once per held-out set and reused."""
    response = client.post(
        ANATOMY,
        json={
            "points": TWELVE_PEOPLE,
            "held_out": list(held_out),
            "n_folds": PAGE_FOLDS,
            "seed": PAGE_SEED,
        },
    )
    assert response.status_code == 200
    return response.json()


@cache
def leak_at(n_rows: int, seed: int = FIRST_SEED) -> dict:
    """One leak measurement, fetched once per row count and reused."""
    response = client.post(LEAK, json={"n_rows": n_rows, "seed": seed})
    assert response.status_code == 200
    return response.json()


def searched(points: list[dict]) -> dict:
    response = client.post(
        SEARCH,
        json={
            "points": points,
            "degrees": [1, 2, 3, 4],
            "penalties": [0.01, 0.1, 1, 10],
            "n_folds": PAGE_FOLDS,
            "seed": PAGE_SEED,
        },
    )
    assert response.status_code == 200
    return response.json()


@cache
def ordered() -> dict:
    response = client.post(
        ORDER,
        json={
            "points": TWELVE_PEOPLE,
            "penalties": [0, 0.01, 0.1, 1, 10, 100],
            "held_out": MIDDLE_THREE,
            "n_folds": PAGE_FOLDS,
            "seed": PAGE_SEED,
        },
    )
    assert response.status_code == 200
    return response.json()


@cache
def contracts() -> dict:
    response = client.get(CONTRACTS)
    assert response.status_code == 200
    return {probe["edge"]: probe for probe in response.json()["probes"]}


class TestTheChainFollowed:
    """Nine train, three held out, and what each step learned."""

    def test_the_steps_run_in_order_and_the_second_reads_the_first_s_output(
        self,
    ) -> None:
        steps = anatomy_holding_out(*MIDDLE_THREE)["steps"]

        assert [step["name"] for step in steps] == ["terms", "scaler"]
        assert steps[0]["reads"] == ["height"]
        assert steps[0]["produces"] == ["height", "height^2"]
        assert steps[0]["learned"] == []
        assert steps[1]["reads"] == ["height", "height^2"]
        assert [scaling["name"] for scaling in steps[1]["learned"]] == [
            "height",
            "height^2",
        ]

    def test_the_standardizer_learned_the_training_nine(self) -> None:
        learned = anatomy_holding_out(*MIDDLE_THREE)["steps"][1]["learned"]

        assert learned[0]["mean"] == pytest.approx(167.1111, abs=1e-4)
        assert learned[0]["standard_deviation"] == pytest.approx(10.3757, abs=1e-4)
        assert learned[1]["mean"] == pytest.approx(28033.7778, abs=1e-4)
        assert learned[1]["standard_deviation"] == pytest.approx(3467.3875, abs=1e-4)

    def test_the_held_out_heights_are_standardized_by_the_training_numbers(
        self,
    ) -> None:
        held_out = anatomy_holding_out(*MIDDLE_THREE)["held_out"]

        assert [person["height"] for person in held_out] == [153.0, 165.0, 177.0]
        standardized = [person["standardized_height"] for person in held_out]
        assert standardized == pytest.approx([-1.3600, -0.2035, 0.9531], abs=1e-4)
        predicted = [person["predicted_weight"] for person in held_out]
        assert predicted == pytest.approx([52.75, 63.66, 74.98], abs=0.01)

    def test_the_scores_are_as_quoted(self) -> None:
        body = anatomy_holding_out(*MIDDLE_THREE)

        assert body["held_out_r2"] == pytest.approx(0.9930, abs=1e-4)
        assert body["training_r2"] == pytest.approx(0.9917, abs=1e-4)
        assert body["intercept"] == pytest.approx(65.78, abs=0.01)
        coefficients = {entry["name"]: entry["value"] for entry in body["coefficients"]}
        assert coefficients["height"] == pytest.approx(4.821, abs=1e-3)
        assert coefficients["height^2"] == pytest.approx(4.853, abs=1e-3)

    def test_the_tallest_three_lie_past_the_training_people(self) -> None:
        body = anatomy_holding_out(*TALLEST_THREE)
        learned = body["steps"][1]["learned"][0]

        assert learned["mean"] == pytest.approx(162.1111, abs=1e-4)
        assert learned["standard_deviation"] == pytest.approx(7.6659, abs=1e-4)
        standardized = [person["standardized_height"] for person in body["held_out"]]
        assert standardized == pytest.approx([1.942, 2.334, 2.725], abs=1e-3)
        predicted = [person["predicted_weight"] for person in body["held_out"]]
        assert predicted == pytest.approx([74.71, 77.61, 80.53], abs=0.01)
        assert body["held_out_r2"] == pytest.approx(0.7871, abs=1e-4)
        assert body["training_r2"] == pytest.approx(0.9872, abs=1e-4)

    def test_one_person_cannot_be_standardized_alone(self) -> None:
        alone = anatomy_holding_out(*MIDDLE_THREE)["alone"]

        assert alone["error"] == "AllSameValuesError"

    def test_a_chain_with_no_steps_is_the_bare_model(self) -> None:
        assert anatomy_holding_out(*MIDDLE_THREE)["no_steps_gap"] == 0.0


class TestTheConfigurationIsNeverFitted:
    """Every check on the one object, after one fit and after four folds."""

    def test_the_configuration_stays_unfitted_and_the_copies_are_fitted(
        self,
    ) -> None:
        body = anatomy_holding_out(*MIDDLE_THREE)

        assert body["configuration_model_fitted"] is False
        assert body["configuration_steps_fitted"] == [False, False]
        assert body["fitted_model_fitted"] is True
        assert body["fitted_parts_are_separate"] is True

    def test_predict_learns_nothing(self) -> None:
        body = anatomy_holding_out(*MIDDLE_THREE)

        assert body["mean_before_predict"] == body["mean_after_predict"]

    def test_a_refit_starts_from_scratch(self) -> None:
        assert anatomy_holding_out(*MIDDLE_THREE)["refit_gap"] == 0.0

    def test_a_failed_refit_leaves_the_previous_fit_whole(self) -> None:
        body = anatomy_holding_out(*MIDDLE_THREE)

        assert body["failed_refit"]["error"] == "AllSameValuesError"
        assert body["still_fitted_after_failed_refit"] is True
        assert body["predictions_after_failed_refit_gap"] == 0.0

    def test_four_folds_give_four_independent_fits(self) -> None:
        body = anatomy_holding_out(*MIDDLE_THREE)

        assert body["n_folds"] == PAGE_FOLDS
        assert body["fold_seed"] == PAGE_SEED
        assert [fold["gap_to_fresh_fit"] for fold in body["folds"]] == [0.0] * 4
        scores = [fold["r2_score"] for fold in body["folds"]]
        assert scores == pytest.approx([0.9986, 0.9916, 0.9234, 0.9894], abs=1e-4)
        assert body["mean_fold_r2"] == pytest.approx(0.9758, abs=1e-4)
        assert body["configuration_fitted_after_folds"] is False
        assert body["pipeline_fitted_after_folds"] is True


class TestTheStandardizerLeak:
    """A transformer blind to the target, on the family of seeds from zero."""

    EXPECTED: ClassVar[dict[int, tuple[float, float, float, int, float]]] = {
        30: (0.481280, 0.482982, -0.001702, 14, 0.1019),
        60: (0.766758, 0.766392, 0.000365, 15, 0.0256),
        200: (0.863534, 0.863411, 0.000123, 18, 0.0037),
    }

    @pytest.mark.parametrize("n_rows", ROW_COUNTS)
    def test_the_two_arrangements_score_as_quoted(self, n_rows: int) -> None:
        scaling = leak_at(n_rows)["scaling"]
        outside, inside, flattered, seeds, largest = self.EXPECTED[n_rows]

        assert scaling["mean_outside"] == pytest.approx(outside, abs=1e-6)
        assert scaling["mean_inside"] == pytest.approx(inside, abs=1e-6)
        assert scaling["flattered_by"] == pytest.approx(flattered, abs=1e-6)
        assert scaling["seeds_flattered"] == seeds
        assert scaling["largest_gap"] == pytest.approx(largest, abs=1e-4)

    def test_the_gap_at_thirty_people_is_a_coin_toss(self) -> None:
        scaling = leak_at(30)["scaling"]

        assert abs(scaling["flattered_by"]) < 0.01
        assert 10 <= scaling["seeds_flattered"] <= 20

    def test_the_largest_single_seed_gap_shrinks_as_the_people_grow(self) -> None:
        sizes = [leak_at(n_rows)["scaling"]["largest_gap"] for n_rows in ROW_COUNTS]

        assert all(later < earlier for earlier, later in pairwise(sizes))

    def test_thirty_seeds_are_run_and_the_means_are_theirs(self) -> None:
        body = leak_at(30)
        scaling = body["scaling"]

        assert body["n_seeds"] == 30
        assert body["first_seed"] == FIRST_SEED
        assert body["training_share"] == pytest.approx(0.8)
        assert len(scaling["outside_scores"]) == 30
        assert len(scaling["inside_scores"]) == 30
        assert scaling["mean_outside"] == pytest.approx(
            sum(scaling["outside_scores"]) / 30
        )
        assert scaling["flattered_by"] == pytest.approx(
            scaling["mean_outside"] - scaling["mean_inside"]
        )
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

    def test_the_next_family_of_seeds_is_a_coin_toss_too(self) -> None:
        second = leak_at(30, seed=30)["scaling"]

        assert second["flattered_by"] == pytest.approx(-0.010363, abs=1e-6)
        assert second["seeds_flattered"] == 14

    def test_one_chain_object_served_every_fold(self) -> None:
        body = leak_at(30)

        assert body["pipeline_fitted_after_folds"] is True
        assert body["configuration_fitted_after_folds"] is False


class TestTheTargetConsultingLeak:
    """A column chosen by its correlation with a target of pure noise."""

    EXPECTED: ClassVar[dict[int, tuple[float, float, float, int, float]]] = {
        30: (-0.597887, -0.886705, 0.288818, 27, 0.7329),
        60: (-0.269592, -0.425194, 0.155602, 27, 0.4518),
        200: (-0.198506, -0.239192, 0.040686, 18, 0.1972),
    }

    @pytest.mark.parametrize("n_rows", ROW_COUNTS)
    def test_the_two_arrangements_score_as_quoted(self, n_rows: int) -> None:
        selection = leak_at(n_rows)["selection"]
        outside, inside, flattered, seeds, largest = self.EXPECTED[n_rows]

        assert selection["mean_outside"] == pytest.approx(outside, abs=1e-6)
        assert selection["mean_inside"] == pytest.approx(inside, abs=1e-6)
        assert selection["flattered_by"] == pytest.approx(flattered, abs=1e-6)
        assert selection["seeds_flattered"] == seeds
        assert selection["largest_gap"] == pytest.approx(largest, abs=1e-4)

    def test_the_gap_is_systematic_where_the_standardizer_s_was_not(self) -> None:
        body = leak_at(30)

        assert body["selection"]["flattered_by"] > 0.25
        assert body["selection"]["seeds_flattered"] >= 25
        assert body["selection"]["flattered_by"] > body["scaling"]["flattered_by"]

    def test_the_gap_shrinks_with_people_and_keeps_its_sign(self) -> None:
        gaps = [leak_at(n_rows)["selection"]["flattered_by"] for n_rows in ROW_COUNTS]

        assert all(gap > 0.0 for gap in gaps)
        assert all(later < earlier for earlier, later in pairwise(gaps))

    def test_the_next_family_of_seeds_keeps_the_sign(self) -> None:
        second = leak_at(30, seed=30)["selection"]

        assert second["flattered_by"] == pytest.approx(0.340943, abs=1e-6)
        assert second["seeds_flattered"] == 25

    def test_nothing_is_there_to_find_in_either_arrangement(self) -> None:
        selection = leak_at(30)["selection"]

        assert selection["mean_outside"] < 0.0
        assert selection["mean_inside"] < 0.0


class TestTheSearch:
    """Whole step lists and whole models as the candidates."""

    def test_the_twelve_keep_degree_one_at_a_penalty_of_a_tenth(self) -> None:
        body = searched(TWELVE_PEOPLE)

        assert body["best_degree"] == 1
        assert body["best_penalty"] == 0.1
        assert body["best_score"] == pytest.approx(0.9796, abs=1e-4)
        assert body["score_spread"] == pytest.approx(0.3639, abs=1e-4)
        assert body["n_candidates"] == 16
        assert body["n_folds"] == PAGE_FOLDS
        assert body["n_fits"] == 64

    def test_the_cells_the_page_quotes(self) -> None:
        cells = {
            (cell["degree"], cell["penalty"]): cell["score"]
            for cell in searched(TWELVE_PEOPLE)["cells"]
        }

        assert len(cells) == 16
        assert cells[(1, 0.01)] == pytest.approx(0.9796, abs=1e-4)
        assert cells[(2, 0.1)] == pytest.approx(0.9790, abs=1e-4)
        assert cells[(2, 1.0)] == pytest.approx(0.9758, abs=1e-4)
        assert cells[(1, 10.0)] == pytest.approx(0.6157, abs=1e-4)
        assert cells[(4, 10.0)] == pytest.approx(0.9081, abs=1e-4)

    def test_the_best_score_is_the_largest_and_the_spread_is_the_range(self) -> None:
        body = searched(TWELVE_PEOPLE)
        scores = [cell["score"] for cell in body["cells"]]

        assert body["best_score"] == max(scores)
        assert body["score_spread"] == pytest.approx(max(scores) - min(scores))

    def test_the_ideal_case_chooses_a_penalty_and_not_a_degree(self) -> None:
        body = searched(IDEAL_CASE)

        assert body["best_degree"] == 1
        assert body["best_penalty"] == 0.1
        assert body["best_score"] == pytest.approx(0.9976, abs=1e-4)
        assert body["score_spread"] == pytest.approx(0.2828, abs=1e-4)

    def test_the_fields_and_the_two_refusals(self) -> None:
        body = searched(TWELVE_PEOPLE)

        assert body["model_hyperparameters"] == ["fit_intercept", "penalty"]
        assert body["chain_hyperparameters"] == ["model", "steps"]
        assert body["degree_on_model"]["error"] == "InvalidValuesError"
        assert "'degree'" in body["degree_on_model"]["detail"]
        assert body["misspelt_field"]["error"] == "InvalidValuesError"
        assert "'stepz'" in body["misspelt_field"]["detail"]


class TestTheOrderOfTheSteps:
    """The same two steps both ways round."""

    def test_at_a_penalty_of_zero_the_two_orders_agree(self) -> None:
        at_zero = ordered()["sweep"][0]

        assert at_zero["penalty"] == 0.0
        assert at_zero["expand_then_scale_score"] == pytest.approx(0.9579, abs=1e-4)
        assert at_zero["scale_then_expand_score"] == pytest.approx(
            at_zero["expand_then_scale_score"], abs=1e-9
        )
        assert at_zero["held_out_gap"] < 1e-9
        assert ordered()["least_squares_gap"] < 1e-9

    def test_at_every_positive_penalty_the_chain_s_own_order_scores_higher(
        self,
    ) -> None:
        positive = [entry for entry in ordered()["sweep"] if entry["penalty"] > 0]

        assert len(positive) == 5
        for entry in positive:
            assert entry["expand_then_scale_score"] > entry["scale_then_expand_score"]
            assert entry["held_out_gap"] > 0.01

    def test_the_sweep_as_quoted(self) -> None:
        sweep = {entry["penalty"]: entry for entry in ordered()["sweep"]}

        assert sweep[0.01]["expand_then_scale_score"] == pytest.approx(0.9764, abs=1e-4)
        assert sweep[0.01]["scale_then_expand_score"] == pytest.approx(0.9581, abs=1e-4)
        assert sweep[0.1]["expand_then_scale_score"] == pytest.approx(0.9790, abs=1e-4)
        assert sweep[0.1]["scale_then_expand_score"] == pytest.approx(0.9597, abs=1e-4)
        assert sweep[1.0]["expand_then_scale_score"] == pytest.approx(0.9758, abs=1e-4)
        assert sweep[1.0]["scale_then_expand_score"] == pytest.approx(0.9523, abs=1e-4)
        assert sweep[1.0]["held_out_gap"] == pytest.approx(0.735, abs=1e-3)
        assert sweep[10.0]["expand_then_scale_score"] == pytest.approx(0.8105, abs=1e-4)
        assert sweep[10.0]["scale_then_expand_score"] == pytest.approx(0.5539, abs=1e-4)
        assert sweep[10.0]["held_out_gap"] == pytest.approx(2.334, abs=1e-3)

    def test_the_columns_each_order_handed_the_model(self) -> None:
        body = ordered()

        assert body["shown_penalty"] == 1.0
        own = {column["name"]: column for column in body["expand_then_scale_columns"]}
        reverse = {
            column["name"]: column for column in body["scale_then_expand_columns"]
        }
        for column in own.values():
            assert column["mean"] == pytest.approx(0.0, abs=1e-9)
            assert column["standard_deviation"] == pytest.approx(1.0, abs=1e-9)
        assert own["height"]["coefficient"] == pytest.approx(4.821, abs=1e-3)
        assert own["height^2"]["coefficient"] == pytest.approx(4.853, abs=1e-3)
        assert reverse["height^2"]["mean"] == pytest.approx(1.0, abs=1e-9)
        assert reverse["height^2"]["standard_deviation"] == pytest.approx(
            0.932, abs=1e-3
        )
        assert reverse["height"]["coefficient"] == pytest.approx(9.191, abs=1e-3)
        assert reverse["height^2"]["coefficient"] == pytest.approx(0.241, abs=1e-3)


class TestTheContracts:
    """Every edge in the page's table, as the page says."""

    @pytest.mark.parametrize(
        ("edge", "error"),
        [
            ("no features at all", "EmptyValuesError"),
            ("one training person", "AllSameValuesError"),
            ("a constant height column", "AllSameValuesError"),
            ("a height that is not a number", "InvalidValuesError"),
            ("a target shorter than the features", "NonEqualArrayLengthError"),
            (
                "a degree the training people cannot support, 9 on 8 of them",
                "TooFewValuesError",
            ),
            ("predict before fit", "NotFittedError"),
            ("read the fitted steps before fit", "NotFittedError"),
            (
                "predict with a feature missing, from a chain fitted on height and age",
                "InvalidValuesError",
            ),
            (
                "predict with a feature the fit never saw, met by the standardizer first",
                "InvalidValuesError",
            ),
            ("a transformer handed in as the model", "ValidationError"),
            ("a classifier handed in as a regression chain's model", "ValidationError"),
            ("two steps with one name", "NonUniqueFeaturesError"),
            ("a blank step name", "InvalidValuesError"),
            ("ask for a step the chain does not run", "InvalidValuesError"),
            ("an unknown constructor keyword", "ValidationError"),
            ("a search over a misspelt chain field", "InvalidValuesError"),
            ("a search asking the ridge model for a degree", "InvalidValuesError"),
            ("four people dealt into five folds", "TooFewValuesError"),
            ("a classification chain with one class in the target", "SingleClassError"),
            ("a binary-only classifier in a classification chain", "ValidationError"),
        ],
    )
    def test_what_is_refused(self, edge: str, error: str) -> None:
        probe = contracts()[edge]

        assert probe["outcome"] == "refused"
        assert probe["error"] == error

    @pytest.mark.parametrize(
        "edge",
        [
            "no steps at all",
            "predict with a feature the fit never saw, met by the expansion first",
            "predict with the features reordered",
            "a step already fitted on other people, handed in",
            "a two-class chain fitted on the twelve",
        ],
    )
    def test_what_is_accepted(self, edge: str) -> None:
        assert contracts()[edge]["outcome"] == "accepted"

    def test_a_model_as_a_step_escapes_as_a_bare_failure(self) -> None:
        probe = contracts()["a model handed in as a step"]

        assert probe["outcome"] == "escaped"
        assert probe["error"] == "AttributeError"

    def test_a_handed_in_fitted_step_is_refitted_from_scratch(self) -> None:
        result = contracts()["a step already fitted on other people, handed in"][
            "result"
        ]

        assert "166.5833" in result
        assert "200.0" in result

    def test_every_edge_the_page_tabulates_is_probed(self) -> None:
        assert len(contracts()) == 27


class TestWhatTheEndpointsRefuse:
    def test_too_few_rows_for_the_leak_is_refused(self) -> None:
        assert client.post(LEAK, json={"n_rows": 29, "seed": 0}).status_code == 422

    def test_too_many_rows_for_the_leak_is_refused(self) -> None:
        assert client.post(LEAK, json={"n_rows": 201, "seed": 0}).status_code == 422

    def test_a_seed_past_the_ceiling_is_refused(self) -> None:
        assert client.post(LEAK, json={"n_rows": 30, "seed": 10_001}).status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(LEAK, json={"n_rows": 30, "seed": 0, "n_seeds": 3})

        assert response.status_code == 422

    def test_a_held_out_index_past_the_people_is_refused(self) -> None:
        response = client.post(
            ANATOMY, json={"points": TWELVE_PEOPLE, "held_out": [12]}
        )

        assert response.status_code == 422

    def test_holding_out_all_but_one_is_refused(self) -> None:
        response = client.post(
            ANATOMY, json={"points": TWELVE_PEOPLE, "held_out": list(range(11))}
        )

        assert response.status_code == 422

    def test_too_few_people_for_the_folds_is_refused_in_the_library_s_words(
        self,
    ) -> None:
        response = client.post(
            SEARCH,
            json={"points": TWELVE_PEOPLE[:3], "degrees": [1, 2], "penalties": [0.1]},
        )

        assert response.status_code == 422
        assert response.json()["error"] == "TooFewValuesError"

    def test_a_degree_past_the_ceiling_is_refused(self) -> None:
        response = client.post(
            SEARCH, json={"points": TWELVE_PEOPLE, "degrees": [7], "penalties": [0.1]}
        )

        assert response.status_code == 422

    def test_a_negative_penalty_is_refused(self) -> None:
        response = client.post(
            ORDER, json={"points": TWELVE_PEOPLE, "penalties": [-1], "held_out": [1]}
        )

        assert response.status_code == 422
