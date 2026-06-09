"""The rest of the judging-a-classifier page, pinned to its own arithmetic.

The first test module pins the four cells and the fold pooling on the twelve.
This one pins what the rebuilt page added: the majority-class baseline the
fit has to beat, the two curves the threshold sweep draws and their areas,
the three F-scores, the reliability reading, a crowd where adults are rare,
and the three-class table with the width stated or inferred.

Two areas are checked against their definitions rather than against a
number read off the response. The area under the ROC curve is the share of
adult-child pairs where the adult's chance is the higher, and on the twelve
that is a count, 33 of 36. The average precision is the step sum under the
precision-recall corners, and the test rebuilds it from the corners the
response carries.
"""

from itertools import pairwise

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

EVALUATE = "/concepts/judging-a-classifier/evaluate"
MULTICLASS = "/concepts/judging-a-classifier/multiclass"

# The twelve the first module works: six children, six adults, one of each
# on the wrong side of any boundary.
OVERLAPPING_CROWD = [
    {"x": 118, "y": 24, "label": 0},
    {"x": 120, "y": 25, "label": 0},
    {"x": 122, "y": 28, "label": 0},
    {"x": 125, "y": 31, "label": 0},
    {"x": 140, "y": 45, "label": 0},
    {"x": 168, "y": 66, "label": 0},
    {"x": 150, "y": 50, "label": 1},
    {"x": 159, "y": 57, "label": 1},
    {"x": 162, "y": 61, "label": 1},
    {"x": 178, "y": 78, "label": 1},
    {"x": 180, "y": 80, "label": 1},
    {"x": 183, "y": 83, "label": 1},
]

# Thirty people of whom four are adults, two of them child-sized, so the
# accuracy of calling everybody a child is respectable and finds nobody.
RARE_CROWD = [
    {"x": 110, "y": 20, "label": 0},
    {"x": 113, "y": 21, "label": 0},
    {"x": 115, "y": 22, "label": 0},
    {"x": 118, "y": 24, "label": 0},
    {"x": 120, "y": 25, "label": 0},
    {"x": 121, "y": 27, "label": 0},
    {"x": 122, "y": 28, "label": 0},
    {"x": 124, "y": 29, "label": 0},
    {"x": 125, "y": 31, "label": 0},
    {"x": 127, "y": 32, "label": 0},
    {"x": 128, "y": 34, "label": 0},
    {"x": 130, "y": 35, "label": 0},
    {"x": 131, "y": 37, "label": 0},
    {"x": 133, "y": 38, "label": 0},
    {"x": 134, "y": 40, "label": 0},
    {"x": 136, "y": 41, "label": 0},
    {"x": 137, "y": 42, "label": 0},
    {"x": 139, "y": 44, "label": 0},
    {"x": 140, "y": 45, "label": 0},
    {"x": 142, "y": 46, "label": 0},
    {"x": 144, "y": 47, "label": 0},
    {"x": 145, "y": 48, "label": 0},
    {"x": 147, "y": 49, "label": 0},
    {"x": 149, "y": 51, "label": 0},
    {"x": 152, "y": 52, "label": 0},
    {"x": 155, "y": 54, "label": 0},
    {"x": 150, "y": 50, "label": 1},
    {"x": 156, "y": 53, "label": 1},
    {"x": 178, "y": 78, "label": 1},
    {"x": 183, "y": 83, "label": 1},
]

# The multiclass page's thirteen with three people added who confuse the
# classes: a teenager built like an adult, an adult built like a teenager,
# and a tall child the model still places.
THREE_CLASS_CROWD = [
    {"x": 118, "y": 24, "label": 0},
    {"x": 120, "y": 25, "label": 0},
    {"x": 122, "y": 28, "label": 0},
    {"x": 125, "y": 31, "label": 0},
    {"x": 145, "y": 57, "label": 1},
    {"x": 147, "y": 41, "label": 1},
    {"x": 156, "y": 53, "label": 1},
    {"x": 159, "y": 57, "label": 1},
    {"x": 162, "y": 61, "label": 1},
    {"x": 178, "y": 78, "label": 2},
    {"x": 180, "y": 80, "label": 2},
    {"x": 183, "y": 83, "label": 2},
    {"x": 186, "y": 77, "label": 2},
    {"x": 172, "y": 70, "label": 1},
    {"x": 160, "y": 58, "label": 2},
    {"x": 141, "y": 40, "label": 0},
]

WITHOUT_ADULTS = [
    index for index, point in enumerate(THREE_CLASS_CROWD) if point["label"] != 2
]
WITHOUT_TEENAGERS = [
    index for index, point in enumerate(THREE_CLASS_CROWD) if point["label"] != 1
]


def evaluated(points: list[dict], threshold: float = 0.5) -> dict:
    response = client.post(EVALUATE, json={"points": points, "threshold": threshold})
    assert response.status_code == 200, response.text
    return response.json()


def judged(points: list[dict], **extra) -> dict:
    response = client.post(MULTICLASS, json={"points": points, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def reading_at(body: dict, threshold: float) -> dict:
    return next(
        reading
        for reading in body["curve"]
        if reading["threshold"] == pytest.approx(threshold)
    )


class TestTheBaseline:
    """Calling everybody the majority class, judged through the same table."""

    def test_the_twelve_are_balanced_so_the_baseline_is_a_coin(self) -> None:
        body = evaluated(OVERLAPPING_CROWD)

        assert body["prevalence"] == pytest.approx(0.5)
        assert body["baseline"]["majority_label"] == 0
        assert body["baseline"]["accuracy"] == pytest.approx(0.5)
        assert body["baseline"]["precision"] is None
        assert body["baseline"]["recall"] == 0.0

    def test_on_the_rare_crowd_the_baseline_looks_respectable(self) -> None:
        body = evaluated(RARE_CROWD)

        assert body["prevalence"] == pytest.approx(4 / 30)
        assert body["baseline"]["accuracy"] == pytest.approx(26 / 30)
        assert body["baseline"]["recall"] == 0.0

    def test_the_fit_beats_the_baseline_by_two_people_and_finds_three(self) -> None:
        body = evaluated(RARE_CROWD)
        counts = body["counts"]

        assert counts == {
            "true_positives": 3,
            "true_negatives": 25,
            "false_positives": 1,
            "false_negatives": 1,
        }
        assert body["rates"]["accuracy"] == pytest.approx(28 / 30)
        assert body["rates"]["recall"] == pytest.approx(0.75)
        assert body["rates"]["precision"] == pytest.approx(0.75)

    def test_two_thresholds_share_an_accuracy_and_differ_in_what_they_find(
        self,
    ) -> None:
        # At 0.5 three adults are found with one false alarm; at 0.7 two are
        # found with none. The accuracy cannot tell those apart.
        body = evaluated(RARE_CROWD)
        at_half = reading_at(body, 0.5)
        at_seven = reading_at(body, 0.7)

        assert at_half["accuracy"] == pytest.approx(at_seven["accuracy"])
        assert at_half["recall"] == pytest.approx(0.75)
        assert at_seven["recall"] == pytest.approx(0.5)
        assert at_seven["precision"] == pytest.approx(1.0)


class TestTheSweep:
    """Every rate at every hundredth, with the counts behind them."""

    def test_the_counts_at_every_threshold_total_the_crowd(self) -> None:
        for reading in evaluated(OVERLAPPING_CROWD)["curve"]:
            total = (
                reading["true_positives"]
                + reading["false_positives"]
                + reading["false_negatives"]
                + reading["true_negatives"]
            )
            assert total == len(OVERLAPPING_CROWD)

    def test_precision_is_not_monotone_where_recall_is(self) -> None:
        body = evaluated(OVERLAPPING_CROWD)
        precisions = [
            reading_at(body, threshold)["precision"]
            for threshold in (0.1, 0.2, 0.5, 0.8, 0.9)
        ]

        assert precisions == pytest.approx([0.75, 6 / 7, 5 / 6, 0.75, 1.0])

    def test_the_false_positive_rate_is_one_minus_specificity(self) -> None:
        body = evaluated(OVERLAPPING_CROWD)

        assert body["rates"]["false_positive_rate"] == pytest.approx(
            1.0 - body["rates"]["specificity"]
        )

    def test_above_the_highest_chance_precision_and_every_f_score_vanish(
        self,
    ) -> None:
        reading = reading_at(evaluated(OVERLAPPING_CROWD), 0.99)

        assert reading["true_positives"] + reading["false_positives"] == 0
        assert reading["precision"] is None
        assert reading["f_one"] is None
        assert reading["f_half"] is None
        assert reading["f_two"] is None


class TestTheFScores:
    """One harmonic mean, three weightings."""

    def test_f_one_is_the_harmonic_mean_and_not_the_arithmetic(self) -> None:
        rates = evaluated(OVERLAPPING_CROWD, 0.9)["rates"]

        assert rates["precision"] == pytest.approx(1.0)
        assert rates["recall"] == pytest.approx(0.5)
        assert rates["f_one"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)
        assert rates["f_one"] != pytest.approx(0.75)

    def test_the_other_betas_lean_the_mean(self) -> None:
        rates = evaluated(OVERLAPPING_CROWD, 0.9)["rates"]

        assert rates["f_half"] == pytest.approx(1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5))
        assert rates["f_two"] == pytest.approx(5.0 * 1.0 * 0.5 / (4.0 * 1.0 + 0.5))
        assert rates["f_half"] > rates["f_one"] > rates["f_two"]

    def test_when_precision_and_recall_agree_every_f_score_is_that_number(
        self,
    ) -> None:
        rates = evaluated(OVERLAPPING_CROWD, 0.5)["rates"]

        assert rates["f_half"] == pytest.approx(5 / 6)
        assert rates["f_one"] == pytest.approx(5 / 6)
        assert rates["f_two"] == pytest.approx(5 / 6)

    def test_the_peaks_are_read_off_the_sweep_at_the_lowest_tied_threshold(
        self,
    ) -> None:
        best = evaluated(OVERLAPPING_CROWD)["best"]

        for name in ("f_half", "f_one", "f_two"):
            assert best[name]["threshold"] == pytest.approx(0.18)
        assert best["f_one"]["value"] == pytest.approx(12 / 13)

    def test_on_the_rare_crowd_the_betas_peak_at_different_thresholds(self) -> None:
        best = evaluated(RARE_CROWD)["best"]

        assert best["f_half"]["threshold"] == pytest.approx(0.51)
        assert best["f_one"]["threshold"] == pytest.approx(0.51)
        assert best["f_two"]["threshold"] == pytest.approx(0.14)
        assert best["f_two"]["threshold"] < best["f_one"]["threshold"]


class TestTheTwoCurves:
    """The receiver operating characteristic and the precision-recall curve."""

    def test_the_roc_runs_from_the_origin_to_the_far_corner(self) -> None:
        roc = evaluated(OVERLAPPING_CROWD)["roc"]

        assert roc[0] == {"threshold": None, "false_positive_rate": 0.0, "recall": 0.0}
        assert roc[-1]["false_positive_rate"] == pytest.approx(1.0)
        assert roc[-1]["recall"] == pytest.approx(1.0)
        assert len(roc) == len(OVERLAPPING_CROWD) + 1

    def test_the_roc_only_ever_moves_up_or_right(self) -> None:
        roc = evaluated(OVERLAPPING_CROWD)["roc"]

        for earlier, later in pairwise(roc):
            assert later["false_positive_rate"] >= earlier["false_positive_rate"]
            assert later["recall"] >= earlier["recall"]

    def test_the_roc_area_is_the_share_of_correctly_ordered_pairs(self) -> None:
        # Six adults against six children is thirty-six pairs, and the tall
        # child outranks three of the adults.
        body = evaluated(OVERLAPPING_CROWD)
        chances = body["probabilities"]
        adults = [
            chance
            for chance, point in zip(chances, OVERLAPPING_CROWD)
            if point["label"] == 1
        ]
        children = [
            chance
            for chance, point in zip(chances, OVERLAPPING_CROWD)
            if point["label"] == 0
        ]
        ordered = sum(adult > child for adult in adults for child in children)

        assert ordered == 33
        assert body["roc_area"] == pytest.approx(33 / 36)

    def test_the_roc_area_on_the_rare_crowd(self) -> None:
        assert evaluated(RARE_CROWD)["roc_area"] == pytest.approx(102 / 104)

    def test_the_precision_recall_curve_has_a_corner_per_distinct_chance(self) -> None:
        body = evaluated(OVERLAPPING_CROWD)

        assert len(body["precision_recall"]) == len(set(body["probabilities"]))
        assert body["precision_recall"][0]["precision"] == pytest.approx(1.0)
        assert body["precision_recall"][0]["recall"] == pytest.approx(1 / 6)

    def test_the_curve_ends_at_the_prevalence(self) -> None:
        # Everybody called adult finds every adult, and the precision is then
        # the share of the crowd that is adult.
        for crowd in (OVERLAPPING_CROWD, RARE_CROWD):
            body = evaluated(crowd)
            last = body["precision_recall"][-1]

            assert last["recall"] == pytest.approx(1.0)
            assert last["precision"] == pytest.approx(body["prevalence"])

    def test_the_average_precision_is_the_step_sum_under_the_corners(self) -> None:
        body = evaluated(OVERLAPPING_CROWD)
        total = 0.0
        previous = 0.0
        for corner in body["precision_recall"]:
            total += (corner["recall"] - previous) * corner["precision"]
            previous = corner["recall"]

        assert body["average_precision"] == pytest.approx(total)
        assert body["average_precision"] == pytest.approx((3 + 0.8 + 5 / 6 + 6 / 7) / 6)


class TestReliability:
    """The chances binned into fifths against the share of adults in each."""

    def test_the_twelve_fall_into_the_bins_the_page_reads(self) -> None:
        bins = evaluated(OVERLAPPING_CROWD)["reliability"]["bins"]

        assert [each["count"] for each in bins] == [5, 0, 1, 2, 4]
        assert bins[1]["mean_probability"] is None
        assert bins[1]["observed_share"] is None
        assert bins[0]["observed_share"] == pytest.approx(0.0)
        assert bins[-1]["observed_share"] == pytest.approx(0.75)

    def test_the_mean_chance_matches_the_observed_rate_in_the_large(self) -> None:
        # The fit balances its chances against the outcomes overall, which
        # the bins then show it does not do locally.
        reliability = evaluated(OVERLAPPING_CROWD)["reliability"]

        assert reliability["mean_probability"] == pytest.approx(0.5, abs=1e-3)
        assert reliability["observed_rate"] == pytest.approx(0.5)
        assert reliability["expected_calibration_error"] > 0.1

    def test_the_calibration_error_is_the_count_weighted_gap(self) -> None:
        reliability = evaluated(OVERLAPPING_CROWD)["reliability"]
        weighted = sum(
            each["count"] * abs(each["mean_probability"] - each["observed_share"])
            for each in reliability["bins"]
            if each["count"]
        )

        assert reliability["expected_calibration_error"] == pytest.approx(
            weighted / len(OVERLAPPING_CROWD)
        )

    def test_the_brier_score_is_the_mean_squared_gap_per_person(self) -> None:
        body = evaluated(OVERLAPPING_CROWD)
        gaps = [
            (chance - point["label"]) ** 2
            for chance, point in zip(body["probabilities"], OVERLAPPING_CROWD)
        ]

        assert body["reliability"]["brier_score"] == pytest.approx(
            sum(gaps) / len(gaps)
        )
        assert body["reliability"]["brier_score"] < 0.25


class TestTheThreeClassTable:
    """Sixteen people, three classes, two of them confused with each other."""

    def test_the_table_and_its_diagonal(self) -> None:
        body = judged(THREE_CLASS_CROWD)

        assert body["n_classes"] == 3
        assert body["width_was_stated"] is False
        assert body["counts"] == [[5, 0, 0], [0, 5, 1], [0, 1, 4]]
        assert body["accuracy"] == pytest.approx(14 / 16)

    def test_the_per_class_rates_are_read_off_the_rows_and_columns(self) -> None:
        classes = judged(THREE_CLASS_CROWD)["classes"]
        teenager = classes[1]
        adult = classes[2]

        assert teenager["actually_are"] == 6
        assert teenager["predicted_as"] == 6
        assert teenager["precision"] == pytest.approx(5 / 6)
        assert teenager["recall"] == pytest.approx(5 / 6)
        assert adult["precision"] == pytest.approx(4 / 5)
        assert adult["recall"] == pytest.approx(4 / 5)
        assert classes[0]["f_one"] == pytest.approx(1.0)

    def test_macro_averages_the_classes_and_micro_equals_the_accuracy(self) -> None:
        body = judged(THREE_CLASS_CROWD)

        assert body["macro_precision"] == pytest.approx((1.0 + 5 / 6 + 4 / 5) / 3)
        assert body["macro_recall"] == pytest.approx(body["macro_precision"])
        assert body["micro_precision"] == pytest.approx(body["accuracy"])
        assert body["micro_recall"] == pytest.approx(body["accuracy"])
        assert body["macro_precision"] != pytest.approx(body["accuracy"])

    def test_a_judged_set_without_adults_is_refused_when_the_width_is_inferred(
        self,
    ) -> None:
        # One teenager is called adult, and an inferred table has no column
        # for a class the judged rows never belong to.
        response = client.post(
            MULTICLASS, json={"points": THREE_CLASS_CROWD, "judged": WITHOUT_ADULTS}
        )

        assert response.status_code == 422
        assert response.json()["error"] == "InvalidValuesError"
        assert "class 2" in response.json()["detail"]

    def test_the_same_rows_are_accepted_once_the_width_is_stated(self) -> None:
        body = judged(THREE_CLASS_CROWD, judged=WITHOUT_ADULTS, stated_width=3)
        adult = body["classes"][2]

        assert body["n_judged"] == 11
        assert body["counts"] == [[5, 0, 0], [0, 5, 1], [0, 0, 0]]
        assert body["accuracy"] == pytest.approx(10 / 11)
        assert adult["predicted_as"] == 1
        assert adult["precision"] == pytest.approx(0.0)
        assert adult["recall"] is None
        assert body["macro_recall"] is None
        assert body["macro_precision"] == pytest.approx((1.0 + 1.0 + 0.0) / 3)

    def test_a_judged_set_leaving_a_gap_is_refused_by_name(self) -> None:
        response = client.post(
            MULTICLASS, json={"points": THREE_CLASS_CROWD, "judged": WITHOUT_TEENAGERS}
        )

        assert response.status_code == 422
        assert response.json()["error"] == "SingleClassError"
        assert "gap" in response.json()["detail"]

        stated = judged(THREE_CLASS_CROWD, judged=WITHOUT_TEENAGERS, stated_width=3)
        assert stated["classes"][1]["recall"] is None

    def test_a_stated_width_narrower_than_the_rows_is_refused(self) -> None:
        response = client.post(
            MULTICLASS, json={"points": THREE_CLASS_CROWD, "stated_width": 2}
        )

        assert response.status_code == 422
        assert response.json()["error"] == "InvalidValuesError"

    def test_a_stated_width_wider_than_the_rows_adds_an_empty_class(self) -> None:
        body = judged(THREE_CLASS_CROWD, stated_width=4)

        assert body["n_classes"] == 4
        assert body["class_names"][3] == "class 3"
        assert body["classes"][3]["actually_are"] == 0
        assert body["classes"][3]["recall"] is None
        assert body["macro_recall"] is None
        assert body["accuracy"] == pytest.approx(14 / 16)

    def test_a_crowd_missing_a_class_is_refused_in_words(self) -> None:
        response = client.post(
            MULTICLASS,
            json={"points": [p for p in THREE_CLASS_CROWD if p["label"] != 2]},
        )

        assert response.status_code == 422
        assert "no adult" in response.json()["detail"]

    def test_a_judged_row_outside_the_crowd_is_refused(self) -> None:
        response = client.post(
            MULTICLASS, json={"points": THREE_CLASS_CROWD, "judged": [16]}
        )

        assert response.status_code == 422
        assert "outside a crowd of 16" in response.json()["detail"]
