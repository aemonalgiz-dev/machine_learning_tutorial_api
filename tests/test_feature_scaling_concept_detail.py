"""The feature-scaling page's further endpoints, pinned on the crowd.

The crowd is the eleven people the classification pages share, five children
and six adults by height in centimetres and weight in kilograms. Every number
the page quotes about them is asserted here: which unit decides a neighbour
vote at (150, 60), how many passes a gradient walk takes under each unit,
which of four models notice the unit at all, what standardizing held-out rows
with their own numbers does to a prediction, where the quartiles fall on the
sorted heights, and what the library says to every edge the page tabulates.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

NEIGHBOURS = "/concepts/feature-scaling/units/neighbours"
MODELS = "/concepts/feature-scaling/units/models"
HOLD_OUT = "/concepts/feature-scaling/hold-out"
QUARTILES = "/concepts/feature-scaling/quartiles"
SCALE = "/concepts/feature-scaling/scale"
CONTRACTS = "/concepts/feature-scaling/contracts"

CROWD = [
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
]
PLAIN = [{"x": person["x"], "y": person["y"]} for person in CROWD]
HEIGHTS = [person["x"] for person in CROWD]

UNITS = ["millimetres", "centimetres", "metres", "standardized"]


def votes_at(x: float, y: float, k: int = 3) -> dict[str, dict]:
    response = client.post(
        NEIGHBOURS, json={"points": CROWD, "query": {"x": x, "y": y}, "k": k}
    )
    assert response.status_code == 200
    return {vote["unit"]: vote for vote in response.json()["votes"]}


@pytest.fixture(scope="module")
def models() -> dict:
    response = client.post(MODELS, json={"points": CROWD})
    assert response.status_code == 200
    return response.json()


def by_unit(rows: list[dict]) -> dict[str, dict]:
    return {row["unit"]: row for row in rows}


class TestTheUnitDecidesTheVote:
    def test_the_four_readings_come_back_in_order(self) -> None:
        assert list(votes_at(150, 60)) == UNITS

    def test_at_150_and_60_millimetres_say_child_and_the_rest_say_adult(self) -> None:
        votes = votes_at(150, 60)

        assert votes["millimetres"]["prediction"] == 0
        assert votes["centimetres"]["prediction"] == 1
        assert votes["metres"]["prediction"] == 1
        assert votes["standardized"]["prediction"] == 1

    def test_each_reading_consults_different_people(self) -> None:
        votes = votes_at(150, 60)

        chosen = {
            unit: [neighbour["index"] for neighbour in votes[unit]["neighbours"]]
            for unit in UNITS
        }
        assert chosen["millimetres"] == [0, 2, 1]
        assert chosen["centimetres"] == [2, 1, 3]
        assert chosen["metres"] == [4, 2, 3]
        assert chosen["standardized"] == [2, 3, 1]

    def test_height_decides_in_millimetres_and_weight_decides_in_metres(self) -> None:
        votes = votes_at(150, 60)

        millimetre_shares = [
            n["height_share"] for n in votes["millimetres"]["neighbours"]
        ]
        metre_shares = [n["height_share"] for n in votes["metres"]["neighbours"]]
        assert min(millimetre_shares) > 0.71
        assert max(metre_shares) < 0.015

    def test_the_neighbours_page_query_keeps_its_pythagorean_distances(self) -> None:
        votes = votes_at(150, 45)

        distances = [n["distance"] for n in votes["centimetres"]["neighbours"]]
        assert distances == [5.0, 10.0, 13.0]
        assert votes["centimetres"]["neighbours"][0]["height_share"] == pytest.approx(
            0.36
        )

    def test_the_standardizing_numbers_are_reported(self) -> None:
        response = client.post(
            NEIGHBOURS, json={"points": CROWD, "query": {"x": 150, "y": 60}, "k": 3}
        ).json()

        assert response["height_mean"] == pytest.approx(151.8181818)
        assert response["height_deviation"] == pytest.approx(22.8426164)
        assert response["weight_mean"] == pytest.approx(53.3636364)
        assert response["weight_deviation"] == pytest.approx(20.7639229)

    def test_more_voters_than_people_is_refused(self) -> None:
        response = client.post(
            NEIGHBOURS, json={"points": CROWD[:2], "query": {"x": 150, "y": 60}, "k": 3}
        )

        assert response.status_code == 422


class TestTheGradientWalkByUnit:
    def test_the_condition_numbers(self, models: dict) -> None:
        descents = by_unit(models["descents"])

        assert descents["millimetres"]["condition_number"] == pytest.approx(
            52178.51, abs=0.01
        )
        assert descents["centimetres"]["condition_number"] == pytest.approx(
            521.785, abs=0.001
        )
        assert descents["metres"]["condition_number"] == pytest.approx(
            19.165, abs=0.001
        )
        assert descents["standardized"]["condition_number"] == pytest.approx(1.0)

    def test_every_walk_takes_half_its_own_threshold(self, models: dict) -> None:
        for descent in models["descents"]:
            assert descent["learning_rate"] == pytest.approx(
                descent["divergence_threshold"] / 2
            )

    def test_the_passes_run(self, models: dict) -> None:
        descents = by_unit(models["descents"])

        assert descents["standardized"]["epochs_run"] == 2
        assert descents["metres"]["epochs_run"] == 374
        assert descents["centimetres"]["epochs_run"] == 8415
        assert descents["millimetres"]["epochs_run"] == 20000
        assert descents["millimetres"]["converged"] is False
        assert all(descents[unit]["converged"] for unit in UNITS[1:])

    def test_millimetres_had_the_slope_and_not_the_level(self, models: dict) -> None:
        stuck = by_unit(models["descents"])["millimetres"]

        assert stuck["slope"] == pytest.approx(0.0890079, abs=1e-6)
        assert stuck["level"] == pytest.approx(16.99, abs=0.01)


class TestWhichModelsNotice:
    def test_least_squares_slopes_scale_exactly_and_r_squared_does_not_move(
        self, models: dict
    ) -> None:
        lines = by_unit(models["least_squares"])

        assert lines["centimetres"]["slope"] == pytest.approx(0.8900786, abs=1e-6)
        assert lines["metres"]["slope"] == pytest.approx(89.00786, abs=1e-4)
        assert lines["millimetres"]["slope"] == pytest.approx(0.08900786, abs=1e-7)
        scores = [line["r_squared"] for line in models["least_squares"]]
        assert max(scores) - min(scores) < 1e-9
        assert scores[1] == pytest.approx(0.9588034, abs=1e-6)

    def test_the_standardized_line_translates_back_to_the_raw_one(
        self, models: dict
    ) -> None:
        lines = by_unit(models["least_squares"])

        assert lines["standardized"]["slope"] == pytest.approx(20.33172, abs=1e-4)
        assert lines["standardized"]["intercept"] == pytest.approx(53.36364, abs=1e-4)
        assert lines["standardized"]["slope_in_centimetres"] == pytest.approx(
            lines["centimetres"]["slope"]
        )
        assert lines["standardized"]["intercept_in_centimetres"] == pytest.approx(
            lines["centimetres"]["intercept"]
        )

    def test_ridge_reads_the_unit_as_a_fact(self, models: dict) -> None:
        ridges = by_unit(models["ridges"])

        assert ridges["centimetres"]["coefficient"] == pytest.approx(
            0.8899235, abs=1e-6
        )
        assert ridges["centimetres"]["r_squared"] == pytest.approx(0.9588034, abs=1e-6)
        assert ridges["metres"]["coefficient"] == pytest.approx(32.4577, abs=1e-3)
        assert ridges["metres"]["r_squared"] == pytest.approx(0.5717774, abs=1e-6)
        assert ridges["millimetres"]["coefficient"] == pytest.approx(
            0.0890077, abs=1e-6
        )
        assert ridges["standardized"]["coefficient"] == pytest.approx(
            18.63741, abs=1e-4
        )
        assert ridges["standardized"]["r_squared"] == pytest.approx(0.9521450, abs=1e-6)

    def test_the_tree_is_the_same_tree_with_its_threshold_moved(
        self, models: dict
    ) -> None:
        trees = by_unit(models["trees"])

        assert models["trees_agree"] is True
        for tree in models["trees"]:
            assert tree["depth"] == 1
            assert tree["n_leaves"] == 2
            assert tree["root_feature"] == "height"
        assert trees["centimetres"]["root_threshold"] == 151.5
        assert trees["millimetres"]["root_threshold"] == 1515.0
        assert trees["metres"]["root_threshold"] == pytest.approx(1.515)

    def test_every_scaler_restores_the_heights(self, models: dict) -> None:
        gaps = {trip["method"]: trip["max_gap"] for trip in models["round_trips"]}

        assert gaps["standardize"] == 0.0
        assert gaps["min_max"] == 0.0
        assert gaps["robust"] == 0.0
        assert gaps["max_abs"] < 1e-13
        assert gaps["root_mean_square"] < 1e-13


class TestFitOnTrainingRowsOnly:
    def test_the_three_tallest_held_out(self) -> None:
        body = client.post(
            HOLD_OUT, json={"points": PLAIN, "held_out": [8, 9, 10]}
        ).json()

        assert body["training_mean"] == pytest.approx(141.125)
        assert body["training_deviation"] == pytest.approx(17.22235, abs=1e-4)
        assert body["by_training"] == pytest.approx([2.2572, 2.4314, 2.1411], abs=1e-4)
        assert body["mean_by_training"] == pytest.approx(2.2766, abs=1e-4)
        assert body["own_mean"] == pytest.approx(180.3333, abs=1e-4)
        assert body["own_deviation"] == pytest.approx(2.0548, abs=1e-4)
        assert body["by_own"] == pytest.approx([-0.1622, 1.2978, -1.1355], abs=1e-4)

    def test_the_line_predicts_sensibly_by_training_and_not_by_own(self) -> None:
        body = client.post(
            HOLD_OUT, json={"points": PLAIN, "held_out": [8, 9, 10]}
        ).json()

        assert body["slope"] == pytest.approx(13.9626, abs=1e-4)
        assert body["intercept"] == pytest.approx(43.25)
        assert body["predictions_by_training"] == pytest.approx(
            [74.767, 77.199, 73.145], abs=1e-3
        )
        assert body["predictions_by_own"] == pytest.approx(
            [40.985, 61.370, 27.395], abs=1e-3
        )
        assert body["actual_weights"] == [80.0, 83.0, 78.0]

    def test_a_mixed_hold_out_barely_moves(self) -> None:
        body = client.post(
            HOLD_OUT, json={"points": PLAIN, "held_out": [4, 7, 10]}
        ).json()

        assert body["training_mean"] == pytest.approx(151.5)
        assert body["mean_by_training"] == pytest.approx(0.0535, abs=1e-4)
        assert body["predictions_by_training"] == pytest.approx(
            [62.358, 23.144, 76.617], abs=1e-3
        )
        assert body["predictions_by_own"] == pytest.approx(
            [60.152, 26.436, 72.412], abs=1e-3
        )

    def test_a_repeated_or_out_of_range_index_is_refused(self) -> None:
        assert (
            client.post(
                HOLD_OUT, json={"points": PLAIN, "held_out": [8, 8]}
            ).status_code
            == 422
        )
        assert (
            client.post(
                HOLD_OUT, json={"points": PLAIN, "held_out": [8, 11]}
            ).status_code
            == 422
        )

    def test_too_few_training_rows_is_refused(self) -> None:
        response = client.post(
            HOLD_OUT, json={"points": PLAIN, "held_out": list(range(10))}
        )

        assert response.status_code == 422

    def test_two_identical_held_out_heights_are_refused_by_the_library(self) -> None:
        points = PLAIN + [{"x": 180, "y": 70}]
        response = client.post(HOLD_OUT, json={"points": points, "held_out": [8, 11]})

        assert response.status_code == 422
        assert response.json()["error"] == "AllSameValuesError"


class TestWhereTheQuartilesLand:
    def test_eleven_heights_read_at_two_and_a_half_five_and_seven_and_a_half(
        self,
    ) -> None:
        body = client.post(QUARTILES, json={"values": HEIGHTS}).json()

        assert body["sorted_values"][:3] == [118.0, 120.0, 122.0]
        assert body["first_quartile"] == {"position": 2.5, "value": 133.5}
        assert body["median"] == {"position": 5.0, "value": 156.0}
        assert body["third_quartile"] == {"position": 7.5, "value": 170.0}
        assert body["interquartile_range"] == 36.5

    def test_the_positions_are_what_the_robust_scaler_learned(self) -> None:
        body = client.post(QUARTILES, json={"values": HEIGHTS}).json()

        assert body["robust_centre"] == body["median"]["value"]
        assert body["robust_spread"] == body["interquartile_range"]

    def test_five_values_read_at_whole_positions(self) -> None:
        body = client.post(QUARTILES, json={"values": [9, 19, 24, 29, 39]}).json()

        assert body["first_quartile"] == {"position": 1.0, "value": 19.0}
        assert body["third_quartile"] == {"position": 3.0, "value": 29.0}


class TestTheCrowdsHeightsWithATypo:
    def test_the_five_readings_with_the_99_move_two_spreads_the_page_tabulates(
        self,
    ) -> None:
        scalings = client.post(SCALE, json={"values": [9, 19, 24, 29, 99]}).json()[
            "scalings"
        ]

        assert scalings["standardize"]["spread"] == pytest.approx(32.187, abs=1e-3)
        assert scalings["root_mean_square"]["spread"] == pytest.approx(48.291, abs=1e-3)

    def test_the_clean_heights(self) -> None:
        scalings = client.post(SCALE, json={"values": HEIGHTS}).json()["scalings"]

        assert scalings["standardize"]["centre"] == pytest.approx(151.8182, abs=1e-4)
        assert scalings["standardize"]["spread"] == pytest.approx(22.8426, abs=1e-4)
        assert (scalings["min_max"]["centre"], scalings["min_max"]["spread"]) == (
            118.0,
            65.0,
        )
        assert scalings["max_abs"]["spread"] == 183.0
        assert (scalings["robust"]["centre"], scalings["robust"]["spread"]) == (
            156.0,
            36.5,
        )
        assert scalings["root_mean_square"]["spread"] == pytest.approx(
            153.527, abs=1e-3
        )

    def test_the_tallest_typed_in_millimetres(self) -> None:
        typo = list(HEIGHTS)
        typo[9] = 1830
        scalings = client.post(SCALE, json={"values": typo}).json()["scalings"]

        assert scalings["standardize"]["centre"] == pytest.approx(301.545, abs=1e-3)
        assert scalings["standardize"]["spread"] == pytest.approx(483.779, abs=1e-3)
        assert scalings["min_max"]["spread"] == 1712.0
        assert scalings["max_abs"]["spread"] == 1830.0
        assert (scalings["robust"]["centre"], scalings["robust"]["spread"]) == (
            156.0,
            36.5,
        )
        assert scalings["robust"]["scaled"][9] == pytest.approx(45.863, abs=1e-3)
        assert scalings["min_max"]["scaled"][1] == pytest.approx(0.0222, abs=1e-4)
        assert scalings["root_mean_square"]["spread"] == pytest.approx(
            570.063, abs=1e-3
        )

    def test_a_middle_value_typed_in_millimetres_moves_the_median_one_rank(
        self,
    ) -> None:
        typo = list(HEIGHTS)
        typo[0] = 1470
        scalings = client.post(SCALE, json={"values": typo}).json()["scalings"]

        assert (scalings["robust"]["centre"], scalings["robust"]["spread"]) == (
            159.0,
            45.5,
        )


NO_SPREAD = (
    "a column carrying no variation has nothing to divide by",
    "must not be constant (zero variance)",
)
"""The two sentences a spread of zero comes back with.

The affine scalers and the standardizer guard the same thing in different
words, which is worth keeping rather than flattening, so a probe passes on
either.
"""


def says_no_spread(reason: str) -> bool:
    """Whether one reason is the zero-spread refusal, in either wording."""
    return any(wording in reason for wording in NO_SPREAD)


FINITE_ONLY = "a column\u2019s values must contain only finite values"
NOT_FITTED = "this scaler must be fitted before that can be read"


def refusal(probes: dict, edge: str, method: str) -> str:
    """One probe's reason, asserting it was refused rather than accepted."""
    probe = probes[(edge, method)]
    assert probe["outcome"] == "refused", f"{edge}, {method} was accepted"
    return probe["reason"]


@pytest.fixture(scope="module")
def probes() -> dict[tuple[str, str], dict]:
    response = client.get(CONTRACTS)
    assert response.status_code == 200
    return {(p["edge"], p["method"]): p for p in response.json()["probes"]}


class TestTheContracts:
    def test_nothing_and_empty_are_refused(self, probes: dict) -> None:
        assert refusal(probes, "no features at all", "standardize") == (
            "at least one feature is required"
        )
        assert refusal(probes, "an empty column", "standardize") == (
            "a column\u2019s values must not be empty"
        )

    def test_one_row_and_a_constant_column_split_the_family(self, probes: dict) -> None:
        for edge in ("one row", "a constant column"):
            for method in ("standardize", "min-max", "robust"):
                assert says_no_spread(refusal(probes, edge, method))
            for method in ("max-abs", "root mean square"):
                assert probes[(edge, method)]["outcome"] == "accepted"
                assert probes[(edge, method)]["result"] == "170.0"
        assert says_no_spread(refusal(probes, "a column of zeros", "max-abs"))

    def test_non_finite_values_are_refused(self, probes: dict) -> None:
        assert refusal(probes, "a missing value", "standardize") == FINITE_ONLY
        assert refusal(probes, "an infinite value", "min-max") == FINITE_ONLY

    def test_the_robust_scaler_turns_at_about_three_quarters(
        self, probes: dict
    ) -> None:
        assert probes[("six identical values of eight", "robust")]["result"] == "0.75"
        assert says_no_spread(
            refusal(probes, "seven identical values of eight", "robust")
        )
        assert probes[("nine identical values of twelve", "robust")]["result"] == "0.25"
        assert says_no_spread(
            refusal(probes, "ten identical values of twelve", "robust")
        )

    def test_a_spread_of_a_few_machine_epsilons_is_accepted(self, probes: dict) -> None:
        assert (
            probes[("nine values one machine epsilon apart", "standardize")]["outcome"]
            == "accepted"
        )
        assert float(
            probes[("nine values one machine epsilon apart", "standardize")]["result"]
        ) == pytest.approx(2.582e-16, rel=1e-3)
        assert (
            probes[("nine values one machine epsilon apart", "min-max")]["result"]
            == "8e-16"
        )

    def test_a_subset_is_accepted_and_an_unknown_name_is_refused(
        self, probes: dict
    ) -> None:
        assert (
            probes[("transform with a subset of the fitted features", "standardize")][
                "result"
            ]
            == "1"
        )
        assert (
            probes[("transform with a subset of the fitted features", "min-max")][
                "result"
            ]
            == "1"
        )
        for method in ("standardize", "min-max"):
            unknown = refusal(
                probes, "transform with a feature the fit never saw", method
            )
            assert "age" in unknown and "height" in unknown
        assert "age" in refusal(
            probes, "inverse transform with a feature the fit never saw", "min-max"
        )

    def test_use_before_fit_is_refused(self, probes: dict) -> None:
        assert refusal(probes, "transform before fit", "standardize") == NOT_FITTED
        assert refusal(probes, "transform before fit", "min-max") == NOT_FITTED
        assert (
            refusal(probes, "read the learned numbers before fit", "robust")
            == NOT_FITTED
        )

    def test_structural_faults_are_refused(self, probes: dict) -> None:
        assert "unique" in refusal(probes, "two features with one name", "standardize")
        assert "rows" in refusal(probes, "two features of different lengths", "min-max")

    def test_no_reason_names_anything_from_the_code(self, probes: dict) -> None:
        """The website shows these sentences, so no identifier may survive.

        The reasons are the refusals' own words with the parameter and class
        names taken out, and this is what stops one growing back. A message
        that starts naming a class again fails here rather than reaching a
        reader who has no idea what a class is.
        """
        forbidden = (
            "feature_values",
            "Standardizer",
            "MinMaxScaler",
            "MaxAbsScaler",
            "RobustScaler",
            "RootMeanSquareScaler",
            "Error",
            "oop_ml",
            "self.",
            "__",
        )
        for (edge, method), probe in probes.items():
            reason = probe["reason"]
            if reason is None:
                continue
            for identifier in forbidden:
                assert identifier not in reason, f"{edge}, {method}: {reason}"

    def test_min_max_promises_the_unit_interval_only_on_training_rows(
        self, probes: dict
    ) -> None:
        beyond = probes[("a value beyond the fitted range, 250 cm", "min-max")]
        assert beyond["outcome"] == "accepted"
        assert float(beyond["result"]) == pytest.approx(2.0308, abs=1e-4)
