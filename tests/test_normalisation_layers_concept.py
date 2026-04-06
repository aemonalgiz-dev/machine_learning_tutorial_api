"""The normalisation-layers endpoint, pinned to the page's hand arithmetic.

The page works a four-row block of two features, (1, 7), (1, 9), (7, 1) and
(23, 7), chosen so that every statistic the batch layer reads down a column is
a whole number, means of 8 and 6 and deviations of 9 and 3, and so that the
first row is whole across as well, a mean of 4, a deviation of 3 and a root
mean square of 5. The library keeps epsilon inside the root, so the API
reports the deviations as 9.0000006 and 3.0000017 rather than 9 and 3, and the
page quotes them that way. After one training pass and one step the running
figures sit a tenth of the way from mean 0, variance 1 toward the batch, so
the second column predicts by a mean of 0.6 and a deviation of 1.3416445. All
pinned.
"""

import math

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

NORMALISE = "/concepts/normalisation-layers/normalise"

EPSILON = 1e-5

WORKED_BLOCK = [[1, 7], [1, 9], [7, 1], [23, 7]]

THREE_FEATURES = [[5, 4, 3], [7, 4, 9], [7, 8, 11], [13, 8, 17]]


def normalised_by(layer: str, rows: list[list[float]]) -> dict:
    response = client.post(NORMALISE, json={"rows": rows, "layer": layer})
    assert response.status_code == 200
    return response.json()


class TestTheWorkedExampleUnderBatch:
    def test_the_column_means_are_eight_and_six(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        assert [entry["mean"] for entry in body["statistics"]] == [8.0, 6.0]

    def test_the_column_deviations_carry_the_epsilon(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        deviations = [entry["deviation"] for entry in body["statistics"]]
        assert deviations == [math.sqrt(81 + EPSILON), math.sqrt(9 + EPSILON)]
        assert [round(value, 7) for value in deviations] == [9.0000006, 3.0000017]

    def test_the_second_column_normalises_as_the_page_says(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        second = [round(row[1], 7) for row in body["normalised"]]
        assert second == [0.3333331, 0.9999994, -1.6666657, 0.3333331]

    def test_the_first_column_normalises_as_the_page_says(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        first = [round(row[0], 7) for row in body["normalised"]]
        assert first == [-0.7777777, -0.7777777, -0.1111111, 1.6666666]

    def test_every_value_is_its_deviation_over_the_column_deviation(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        for row, raw_row in zip(body["normalised"], WORKED_BLOCK, strict=True):
            for value, raw, statistic in zip(
                row, raw_row, body["statistics"], strict=True
            ):
                expected = (raw - statistic["mean"]) / statistic["deviation"]
                assert value == pytest.approx(expected, abs=1e-12)

    def test_one_statistic_per_feature(self) -> None:
        body = normalised_by("batch", THREE_FEATURES)

        assert len(body["statistics"]) == 3
        assert [entry["mean"] for entry in body["statistics"]] == [8.0, 6.0, 10.0]
        assert [round(entry["deviation"], 7) for entry in body["statistics"]] == [
            3.0000017,
            2.0000025,
            5.000001,
        ]


class TestPredictingAfterOneStep:
    def test_the_running_figures_moved_a_tenth_of_the_way(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        first, second = body["predicting_statistics"]
        assert first["mean"] == pytest.approx(0.8)
        assert second["mean"] == pytest.approx(0.6)
        assert first["deviation"] == pytest.approx(math.sqrt(0.9 + 0.1 * 81 + EPSILON))
        assert second["deviation"] == pytest.approx(math.sqrt(0.9 + 0.1 * 9 + EPSILON))
        assert round(first["deviation"], 7) == 3.0000017
        assert round(second["deviation"], 7) == 1.3416445

    def test_the_seven_predicts_to_four_point_seven_seven(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        assert round(body["as_predicting"][0][1], 7) == 4.7702651
        assert round(body["as_predicting"][3][0], 7) == 7.3999959

    def test_predicting_divides_by_the_running_figures(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        for row, raw_row in zip(body["as_predicting"], WORKED_BLOCK, strict=True):
            for value, raw, statistic in zip(
                row, raw_row, body["predicting_statistics"], strict=True
            ):
                expected = (raw - statistic["mean"]) / statistic["deviation"]
                assert value == pytest.approx(expected, abs=1e-12)

    def test_the_two_answers_differ(self) -> None:
        body = normalised_by("batch", WORKED_BLOCK)

        assert body["as_predicting"] != body["normalised"]

    def test_the_row_layers_have_no_predicting_block(self) -> None:
        for layer in ("layer", "rms"):
            body = normalised_by(layer, WORKED_BLOCK)

            assert body["as_predicting"] is None
            assert body["predicting_statistics"] is None


class TestTheWorkedExampleAcrossARow:
    def test_layer_normalisation_reads_the_first_row(self) -> None:
        body = normalised_by("layer", WORKED_BLOCK)

        first = body["statistics"][0]
        assert first["mean"] == 4.0
        assert first["deviation"] == math.sqrt(9 + EPSILON)
        assert round(first["deviation"], 7) == 3.0000017
        assert [round(value, 7) for value in body["normalised"][0]] == [
            -0.9999994,
            0.9999994,
        ]

    def test_rms_normalisation_reads_the_first_row(self) -> None:
        body = normalised_by("rms", WORKED_BLOCK)

        first = body["statistics"][0]
        assert first["mean"] is None
        assert first["deviation"] == math.sqrt(25 + EPSILON)
        assert round(first["deviation"], 7) == 5.000001
        assert [round(value, 7) for value in body["normalised"][0]] == [
            0.2,
            1.3999997,
        ]

    def test_one_statistic_per_row(self) -> None:
        for layer in ("layer", "rms"):
            body = normalised_by(layer, WORKED_BLOCK)

            assert len(body["statistics"]) == 4

    def test_every_value_is_its_deviation_over_the_row_deviation(self) -> None:
        for layer in ("layer", "rms"):
            body = normalised_by(layer, THREE_FEATURES)

            for row, raw_row, statistic in zip(
                body["normalised"], THREE_FEATURES, body["statistics"], strict=True
            ):
                centre = statistic["mean"] or 0.0
                for value, raw in zip(row, raw_row, strict=True):
                    expected = (raw - centre) / statistic["deviation"]
                    assert value == pytest.approx(expected, abs=1e-12)


class TestWhichWayTheMeanRuns:
    def test_batch_leaves_every_column_summing_to_zero(self) -> None:
        body = normalised_by("batch", THREE_FEATURES)

        for column in range(3):
            total = sum(row[column] for row in body["normalised"])
            assert total == pytest.approx(0.0, abs=1e-12)

    def test_layer_leaves_every_row_summing_to_zero(self) -> None:
        body = normalised_by("layer", THREE_FEATURES)

        for row in body["normalised"]:
            assert sum(row) == pytest.approx(0.0, abs=1e-12)

    def test_rms_does_not_centre(self) -> None:
        body = normalised_by("rms", THREE_FEATURES)

        for row in body["normalised"]:
            assert sum(row) > 0.0

    def test_a_normalised_column_has_variance_v_over_v_plus_epsilon(self) -> None:
        # The derivation's var(z) = v / (v + epsilon), 9 / 9.00001 on the
        # worked block's second column.
        body = normalised_by("batch", WORKED_BLOCK)

        second = [row[1] for row in body["normalised"]]
        mean = sum(second) / len(second)
        variance = sum((value - mean) ** 2 for value in second) / len(second)
        assert variance == pytest.approx(9 / (9 + EPSILON), abs=1e-12)
        assert variance < 1.0

    def test_a_row_answers_the_same_whatever_else_arrived(self) -> None:
        for layer in ("layer", "rms"):
            alone = normalised_by(layer, WORKED_BLOCK[:2])
            among = normalised_by(layer, WORKED_BLOCK)

            assert alone["normalised"][0] == among["normalised"][0]
            assert alone["statistics"][0] == among["statistics"][0]

    def test_a_batch_row_answers_differently_when_its_company_changes(self) -> None:
        fewer = normalised_by("batch", WORKED_BLOCK[:2])
        more = normalised_by("batch", WORKED_BLOCK)

        assert fewer["normalised"][0] != more["normalised"][0]

    def test_a_constant_feature_normalises_to_zero(self) -> None:
        body = normalised_by("batch", [[5, 1], [5, 2], [5, 6]])

        assert [row[0] for row in body["normalised"]] == [0.0, 0.0, 0.0]


class TestRefusals:
    def test_a_ragged_block_is_refused_in_words(self) -> None:
        response = client.post(
            NORMALISE, json={"rows": [[1, 7], [1, 9, 4], [7, 1]], "layer": "batch"}
        )

        assert response.status_code == 422
        assert "rectangle" in response.json()["detail"]
        assert "row 2" in response.json()["detail"]

    def test_an_unknown_layer_is_refused(self) -> None:
        response = client.post(NORMALISE, json={"rows": WORKED_BLOCK, "layer": "group"})

        assert response.status_code == 422

    def test_a_lone_row_is_refused(self) -> None:
        response = client.post(NORMALISE, json={"rows": [[1, 7]], "layer": "batch"})

        assert response.status_code == 422

    def test_a_row_of_one_value_is_refused(self) -> None:
        response = client.post(NORMALISE, json={"rows": [[1], [2]], "layer": "layer"})

        assert response.status_code == 422

    def test_too_many_features_are_refused(self) -> None:
        wide = [[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1]]
        response = client.post(NORMALISE, json={"rows": wide, "layer": "batch"})

        assert response.status_code == 422

    def test_too_many_rows_are_refused(self) -> None:
        tall = [[float(index), 1.0] for index in range(21)]
        response = client.post(NORMALISE, json={"rows": tall, "layer": "batch"})

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            NORMALISE, json={"rows": WORKED_BLOCK, "layer": "batch", "momentum": 0.5}
        )

        assert response.status_code == 422
