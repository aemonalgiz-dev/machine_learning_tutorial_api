"""The standard score page's endpoints, pinned on the orchard.

The orchard is twenty-four apple trees measured by trunk girth in millimetres
and by a season's water in metres, with the fruit each bore in kilograms. The
two predictor columns are recorded in units a thousand apart, so every claim the
page makes about what a unit does to a fit is asserted here: what the two
columns teach, what the five saplings come to when the sum is done by hand,
which of six model families the unit moved and which it left alone, what
happens when held-out rows are measured against themselves rather than against
the training rows, and what the arithmetic no longer says when a column has no
spread.

Everything is deterministic. Scoring is leave-one-out and pooled, so no seed is
involved and the numbers below are exact rather than approximate.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

ORCHARD = "/concepts/the-standard-score/orchard"
COLUMN = "/concepts/the-standard-score/column"
NURSERY = "/concepts/the-standard-score/nursery"
NEIGHBOURS = "/concepts/the-standard-score/neighbours"
MODELS = "/concepts/the-standard-score/models"
HOLD_OUT = "/concepts/the-standard-score/hold-out"
LIMITS = "/concepts/the-standard-score/limits"

EVERY_ENDPOINT = (ORCHARD, NURSERY, MODELS, HOLD_OUT, LIMITS)


def get(path: str) -> dict:
    response = client.get(path)
    assert response.status_code == 200, response.text
    return response.json()


def post(path: str, body: dict) -> dict:
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    return response.json()


class TestTheOrchard:
    def test_holds_twenty_four_trees_in_every_column(self):
        orchard = get(ORCHARD)

        assert len(orchard["girth"]["values"]) == 24
        assert len(orchard["water"]["values"]) == 24
        assert len(orchard["yields"]) == 24

    def test_the_two_numbers_each_column_teaches(self):
        orchard = get(ORCHARD)

        assert orchard["girth"]["mean"] == 387.625
        assert orchard["girth"]["deviation"] == pytest.approx(113.65181201811082)
        assert orchard["water"]["mean"] == pytest.approx(0.6791666666666666)
        assert orchard["water"]["deviation"] == pytest.approx(0.17240738254365895)

    def test_one_spread_is_six_hundred_times_the_other(self):
        orchard = get(ORCHARD)

        assert orchard["spread_ratio"] == pytest.approx(659.2050197695602)
        assert orchard["squared_spread_ratio"] == pytest.approx(434551.2580893862)

    def test_the_scores_have_mean_zero_and_variance_one(self):
        orchard = get(ORCHARD)

        for column in ("girth", "water"):
            scores = orchard[column]["scores"]
            mean = sum(scores) / len(scores)
            variance = sum((score - mean) ** 2 for score in scores) / len(scores)
            assert mean == pytest.approx(0.0, abs=1e-12)
            assert variance == pytest.approx(1.0)

    def test_standardising_turns_the_covariance_into_the_correlation(self):
        orchard = get(ORCHARD)

        assert orchard["standardised_covariance"] == pytest.approx(
            orchard["correlation"], abs=1e-15
        )
        assert orchard["correlation"] == pytest.approx(-0.08262878962552436)


class TestTheNurseryRow:
    def test_the_sum_the_page_does_by_hand(self):
        nursery = get(NURSERY)

        assert nursery["girths"] == [210.0, 210.0, 260.0, 160.0, 310.0]
        assert nursery["mean"] == 230.0
        assert nursery["deviations_from_mean"] == [-20.0, -20.0, 30.0, -70.0, 80.0]
        assert nursery["squared_deviations"] == [400.0, 400.0, 900.0, 4900.0, 6400.0]
        assert nursery["variance"] == 2600.0
        assert nursery["deviation"] == pytest.approx(50.99019513592785)

    def test_every_transformed_value(self):
        nursery = get(NURSERY)

        assert nursery["scores"] == pytest.approx(
            [-0.392232, -0.392232, 0.588348, -1.372813, 1.568929], abs=1e-6
        )

    def test_a_shift_and_a_stretch_leave_the_scores_where_they_were(self):
        nursery = get(NURSERY)

        assert nursery["whole_numbers"] == [1.0, 1.0, 2.0, 0.0, 3.0]
        assert nursery["largest_gap"] < 1e-15

    def test_the_population_divisor_is_what_makes_the_variance_exactly_one(self):
        nursery = get(NURSERY)

        assert nursery["score_mean"] == pytest.approx(0.0, abs=1e-15)
        assert nursery["score_variance"] == pytest.approx(1.0)
        assert nursery["sample_score_variance"] == pytest.approx(1.25)


class TestOneColumn:
    def test_reports_what_the_column_taught(self):
        answer = post(COLUMN, {"values": [210.0, 210.0, 260.0, 160.0, 310.0]})

        assert answer["mean"] == 230.0
        assert answer["deviation"] == pytest.approx(50.99019513592785)

    def test_a_shift_is_invisible_and_a_stretch_nearly_so(self):
        answer = post(COLUMN, {"values": [210.0, 210.0, 260.0, 160.0, 310.0]})

        assert answer["shifted_largest_gap"] == 0.0
        assert answer["stretched_largest_gap"] < 1e-15

    def test_two_values_a_hair_apart_come_out_at_minus_one_and_one(self):
        answer = post(COLUMN, {"values": [300.0, 300.0000000000001]})

        assert answer["deviation"] == pytest.approx(5.684341886080802e-14)
        assert answer["scores"] == [-1.0, 1.0]

    def test_a_shift_that_destroys_the_spread_reports_no_gap(self):
        # Adding a thousand to values a few units in the last place apart lands
        # every one of them on the same float, so there is nothing to compare.
        answer = post(COLUMN, {"values": [300.0, 300.0000000000001]})

        assert answer["shifted_largest_gap"] is None
        assert answer["stretched_largest_gap"] == 0.0

    def test_a_column_with_no_spread_is_refused_in_plain_words(self):
        response = client.post(COLUMN, json={"values": [5.0, 5.0, 5.0, 5.0]})

        assert response.status_code == 422
        assert "no spread" in response.json()["detail"]

    def test_an_unknown_field_is_refused(self):
        response = client.post(
            COLUMN, json={"values": [1.0, 2.0, 3.0], "centre": "median"}
        )

        assert response.status_code == 422


class TestNeighbours:
    def test_the_recorded_columns_choose_by_girth_almost_alone(self):
        answer = post(NEIGHBOURS, {"tree": 16, "k": 3})
        recorded = answer["readings"][0]

        assert recorded["reading"] == "recorded"
        assert [row["tree"] for row in recorded["neighbours"]] == [10, 20, 15]
        assert min(row["girth_share"] for row in recorded["neighbours"]) > 0.99

    def test_the_wettest_tree_is_predicted_twenty_one_kilograms_short(self):
        answer = post(NEIGHBOURS, {"tree": 16, "k": 3})
        recorded, standardised = answer["readings"]

        assert answer["borne"] == 55.14
        assert recorded["predicted"] == pytest.approx(33.836666666666666)
        assert standardised["predicted"] == pytest.approx(52.72666666666667)
        assert recorded["error"] == pytest.approx(-21.303333333333335)
        assert standardised["error"] == pytest.approx(-2.413333333333334)

    def test_standardising_gives_the_wetter_column_a_say(self):
        answer = post(NEIGHBOURS, {"tree": 16, "k": 3})
        standardised = answer["readings"][1]

        assert [row["tree"] for row in standardised["neighbours"]] == [23, 18, 3]
        assert min(row["water"] for row in standardised["neighbours"]) >= 0.9

    def test_a_tree_outside_the_orchard_is_refused(self):
        assert client.post(NEIGHBOURS, json={"tree": 24, "k": 3}).status_code == 422


def scores_for(family: str) -> dict:
    for row in get(MODELS)["scores"]:
        if row["family"] == family:
            return row
    raise AssertionError(f"no row for {family}")


class TestWhichModelsNoticed:
    @pytest.mark.parametrize(
        ("family", "recorded", "standardised"),
        [
            ("nearest neighbours, three", -0.7175376570579866, 0.9126089093902285),
            ("nearest neighbours, five", -0.36244014782404266, 0.8816512599047158),
            ("ridge, penalty 1", 0.5451997481469768, 0.9767579754212307),
            ("ridge, penalty 10", -0.06650069976250772, 0.8712147247394506),
        ],
    )
    def test_the_families_the_unit_moved(self, family, recorded, standardised):
        row = scores_for(family)

        assert row["recorded"] == pytest.approx(recorded)
        assert row["inside"] == pytest.approx(standardised)
        assert row["moved"] is True

    @pytest.mark.parametrize(
        ("family", "score"),
        [
            ("a tree, depth 3", 0.8940073683828749),
            ("least squares", 0.9791054098363695),
        ],
    )
    def test_the_families_the_unit_left_alone(self, family, score):
        row = scores_for(family)

        assert row["recorded"] == pytest.approx(score)
        assert row["inside"] == pytest.approx(score)
        assert row["moved"] is False

    def test_the_worse_answer_is_below_predicting_the_average(self):
        assert scores_for("nearest neighbours, three")["recorded"] < 0.0

    def test_dropping_the_useless_column_beats_standardising_both(self):
        models = get(MODELS)

        assert models["water_alone_three"] == pytest.approx(0.9544465219946987)
        assert (
            models["water_alone_three"]
            > scores_for("nearest neighbours, three")["inside"]
        )

    def test_the_column_the_recorded_units_read_carries_nothing(self):
        assert get(MODELS)["girth_alone_three"] == pytest.approx(-0.7237518794185707)


class TestCoefficientsAndTheTree:
    def test_least_squares_scores_the_same_either_way(self):
        models = get(MODELS)

        assert models["least_squares_score_gap"] == 0.0
        assert models["least_squares"][0]["fitted_score"] == pytest.approx(
            0.9844555509555813
        )

    def test_a_standardised_weight_divided_by_its_spread_is_the_recorded_weight(self):
        models = get(MODELS)
        recorded = {
            row["name"]: row["value"]
            for row in models["least_squares"][0]["coefficients"]
        }
        translated = {row["name"]: row["value"] for row in models["translated_back"]}

        for name, value in recorded.items():
            assert translated[name] == pytest.approx(value)

    def test_the_recorded_weights_cannot_be_ranked_and_the_standardised_ones_can(self):
        models = get(MODELS)
        recorded = {
            row["name"]: row["value"]
            for row in models["least_squares"][0]["coefficients"]
        }
        standardised = {
            row["name"]: row["value"]
            for row in models["least_squares"][1]["coefficients"]
        }

        assert recorded["water"] / recorded["girth"] > 10000
        assert standardised["water"] / standardised["girth"] == pytest.approx(
            19.375, abs=0.01
        )

    def test_the_penalty_falls_on_the_wrong_column_in_the_recorded_units(self):
        models = get(MODELS)
        recorded, standardised = models["ridge_penalty_one"]

        assert recorded["fitted_score"] == pytest.approx(0.6475259317908681)
        assert standardised["fitted_score"] == pytest.approx(0.9828451880112357)

    def test_the_tree_asks_the_same_question_in_both_units(self):
        models = get(MODELS)
        recorded, standardised = models["trees"]

        assert recorded["root_column"] == standardised["root_column"] == "water"
        assert recorded["root_threshold"] == pytest.approx(0.665)
        assert models["tree_threshold_translated"] == pytest.approx(0.665)
        assert models["tree_largest_prediction_gap"] == 0.0


class TestHoldingRowsOut:
    def test_the_wettest_four_are_held_out(self):
        answer = get(HOLD_OUT)

        assert answer["held_out"] == [3, 16, 18, 23]
        assert min(answer["water"]) >= 0.9

    def test_the_training_numbers_place_them_above_everything_the_fit_saw(self):
        training = get(HOLD_OUT)["readings"][0]

        assert min(training["water_scores"]) > 1.8

    def test_measured_against_themselves_two_of_the_four_read_as_dry(self):
        own = get(HOLD_OUT)["readings"][1]

        assert sum(1 for score in own["water_scores"] if score < 0) == 2

    def test_the_error_the_refit_costs(self):
        training, own = get(HOLD_OUT)["readings"]

        assert training["largest_error"] == pytest.approx(1.8118969770736584)
        assert own["largest_error"] == pytest.approx(20.474907022123155)


class TestTheLeak:
    def test_learning_the_numbers_over_everything_barely_moved_anything(self):
        for row in get(MODELS)["scores"]:
            assert abs(row["leak"]) < 0.004

    def test_the_two_invariant_families_leak_exactly_nothing(self):
        for family in ("a tree, depth 3", "least squares"):
            assert scores_for(family)["leak"] == 0.0

    def test_the_penalised_fit_is_where_the_leak_shows_at_all(self):
        assert scores_for("ridge, penalty 10")["leak"] == pytest.approx(
            -0.0037236012267467533
        )


class TestWhereItStopsBeingDefined:
    def test_a_constant_column_has_no_standard_score(self):
        limits = get(LIMITS)

        assert limits["constant_defined"] is False
        assert limits["single_row_defined"] is False

    def test_a_spread_of_a_few_machine_epsilons_is_still_a_spread(self):
        limits = get(LIMITS)

        assert limits["tiny_deviation"] == pytest.approx(2.581988897471611e-16)
        assert min(limits["tiny_scores"]) == pytest.approx(-1.5491933384829668)
        assert max(limits["tiny_scores"]) == pytest.approx(1.5491933384829668)

    def test_one_stray_digit_moves_both_numbers(self):
        mistyped = get(LIMITS)["mistyped"]

        assert mistyped["before_mean"] == 387.625
        assert mistyped["after_mean"] == pytest.approx(551.875)
        assert mistyped["before_deviation"] == pytest.approx(113.65181201811082)
        assert mistyped["after_deviation"] == pytest.approx(806.201190383021)

    def test_the_ordinary_trees_are_squashed_into_a_seventh_of_the_ruler(self):
        mistyped = get(LIMITS)["mistyped"]

        assert mistyped["others_span_before"] == pytest.approx(3.677900005090902)
        assert mistyped["others_span_after"] == pytest.approx(0.5184810007554206)
        assert mistyped["widest_real_score_before"] == pytest.approx(2.044622042303824)
        assert mistyped["widest_real_score_after"] == pytest.approx(0.0845012396566101)

    def test_the_shape_survives_the_transformation_exactly(self):
        limits = get(LIMITS)

        for reading in (limits["skewed"], limits["girth_shape"]):
            assert reading["skew_after"] == pytest.approx(
                reading["skew_before"], rel=1e-12
            )
            assert reading["order_kept"] is True

    def test_a_standard_score_says_nothing_about_a_bell(self):
        limits = get(LIMITS)

        assert limits["skewed"]["skew_before"] == pytest.approx(2.4896725429570834)
        assert limits["skewed"]["share_within_one"] == pytest.approx(0.8666666666666667)
        assert limits["normal_within_one"] == pytest.approx(0.6826894921370859)
        assert limits["normal_within_two"] == pytest.approx(0.9544997361036416)
        assert limits["normal_within_three"] == pytest.approx(0.9973002039367398)

    def test_the_distribution_free_bound_holds_and_is_loose(self):
        limits = get(LIMITS)

        for share in limits["shares"]:
            assert share["within_two"] >= limits["chebyshev_within_two"]


IDENTIFIER = re.compile(
    r"\b(?:[A-Z][a-z]+[A-Z]\w*|oop_ml|feature_values|standardizer|"
    r"NotFittedError|AllSameValuesError)\b"
)
"""Anything a reader has no reason to meet, in a string a widget may render."""


def every_string(payload: object) -> list[str]:
    """Every string anywhere inside a response body."""
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, dict):
        return [text for value in payload.values() for text in every_string(value)]
    if isinstance(payload, list):
        return [text for value in payload for text in every_string(value)]
    return []


class TestNothingNamesTheCodeBehindThePage:
    @pytest.mark.parametrize("path", EVERY_ENDPOINT)
    def test_no_endpoint_returns_an_identifier(self, path):
        for text in every_string(get(path)):
            assert not IDENTIFIER.search(text), text

    def test_the_refusal_names_no_identifier(self):
        detail = client.post(COLUMN, json={"values": [5.0, 5.0]}).json()["detail"]

        assert not IDENTIFIER.search(detail)

    def test_the_neighbour_answer_names_no_identifier(self):
        for text in every_string(post(NEIGHBOURS, {"tree": 0, "k": 3})):
            assert not IDENTIFIER.search(text)
