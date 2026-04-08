"""The judging-a-classifier endpoints, pinned to the page's hand arithmetic.

The page works a crowd of twelve with two people who sit on the wrong side of
any boundary, so the confusion matrix has all four cells filled and every rate
is five sixths. The threshold sweep is where precision and recall trade: at
three tenths nobody positive is missed and one negative is wrongly called, at
nine tenths nothing is wrongly called and half the positives are missed.

The folding half pins the pooling argument. Dealt seven ways the folds are
uneven, so averaging their accuracies gives 0.8571 where adding their tables
first and dividing once gives 0.8333. Every deal of the same twelve people
gives the identical pooled figure, because every row is held out exactly once
whatever the deal, and that is the whole case for pooling.
"""

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

EVALUATE = "/concepts/judging-a-classifier/evaluate"
FOLD = "/concepts/judging-a-classifier/fold"

# Twelve people, ten of whom a boundary can place and two of whom it cannot:
# one adult-sized child at 168 centimetres and one child-sized adult at 150.
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

FIVE_SIXTHS = 5 / 6


def evaluated(threshold: float = 0.5) -> dict:
    response = client.post(
        EVALUATE, json={"points": OVERLAPPING_CROWD, "threshold": threshold}
    )
    assert response.status_code == 200, response.text
    return response.json()


def folded(n_folds: int, stratified: bool) -> dict:
    response = client.post(
        FOLD,
        json={
            "points": OVERLAPPING_CROWD,
            "n_folds": n_folds,
            "stratified": stratified,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestTheWorkedMatrix:
    """Twelve people at the halfway threshold, the page's hand-counted table."""

    def test_all_four_cells_are_the_page_figures(self) -> None:
        counts = evaluated()["counts"]

        assert counts["true_positives"] == 5
        assert counts["false_positives"] == 1
        assert counts["false_negatives"] == 1
        assert counts["true_negatives"] == 6 - 1

    def test_the_cells_total_the_crowd(self) -> None:
        counts = evaluated()["counts"]

        assert sum(counts.values()) == len(OVERLAPPING_CROWD)

    def test_every_rate_is_five_sixths(self) -> None:
        # One miss in each direction out of six of each class, so all four
        # ratios land on the same number, which is what makes the table worth
        # reading rather than the accuracy alone.
        rates = evaluated()["rates"]

        assert rates["accuracy"] == pytest.approx(10 / 12)
        assert rates["precision"] == pytest.approx(FIVE_SIXTHS)
        assert rates["recall"] == pytest.approx(FIVE_SIXTHS)
        assert rates["specificity"] == pytest.approx(FIVE_SIXTHS)

    def test_each_rate_is_the_ratio_of_its_own_cells(self) -> None:
        body = evaluated()
        counts, rates = body["counts"], body["rates"]
        true_positives = counts["true_positives"]

        assert rates["precision"] == pytest.approx(
            true_positives / (true_positives + counts["false_positives"])
        )
        assert rates["recall"] == pytest.approx(
            true_positives / (true_positives + counts["false_negatives"])
        )
        assert rates["specificity"] == pytest.approx(
            counts["true_negatives"]
            / (counts["true_negatives"] + counts["false_positives"])
        )

    def test_one_answer_and_one_chance_per_person_in_order(self) -> None:
        body = evaluated()

        assert len(body["probabilities"]) == len(OVERLAPPING_CROWD)
        assert len(body["predictions"]) == len(OVERLAPPING_CROWD)
        assert all(0.0 <= chance <= 1.0 for chance in body["probabilities"])

    def test_a_prediction_is_the_chance_against_the_threshold(self) -> None:
        body = evaluated(0.5)

        for chance, prediction in zip(
            body["probabilities"], body["predictions"], strict=True
        ):
            assert prediction == (1 if chance >= 0.5 else 0)

    def test_the_two_the_boundary_cannot_place_are_the_two_misses(self) -> None:
        # The child at 168 is called adult and the adult at 150 is called
        # child, which is the false positive and the false negative.
        body = evaluated()
        predictions = body["predictions"]

        assert predictions[5] == 1  # the tall child
        assert predictions[6] == 0  # the short adult


class TestTheThresholdTrade:
    """Moving the threshold buys one rate with the other."""

    def test_a_low_threshold_misses_no_positive(self) -> None:
        body = evaluated(0.3)

        assert body["counts"]["false_negatives"] == 0
        assert body["rates"]["recall"] == 1.0
        assert body["rates"]["precision"] == pytest.approx(6 / 7)

    def test_a_high_threshold_calls_nothing_wrongly(self) -> None:
        body = evaluated(0.9)

        assert body["counts"]["false_positives"] == 0
        assert body["rates"]["precision"] == 1.0
        assert body["rates"]["recall"] == pytest.approx(0.5)

    def test_recall_never_rises_as_the_threshold_rises(self) -> None:
        curve = evaluated()["curve"]
        recalls = [reading["recall"] for reading in curve]

        assert recalls == sorted(recalls, reverse=True)

    def test_the_curve_spans_the_open_interval(self) -> None:
        curve = evaluated()["curve"]

        assert len(curve) == 99
        assert curve[0]["threshold"] == pytest.approx(0.01)
        assert curve[-1]["threshold"] == pytest.approx(0.99)

    def test_the_curve_agrees_with_the_chosen_threshold(self) -> None:
        body = evaluated(0.5)
        at_half = next(
            reading
            for reading in body["curve"]
            if reading["threshold"] == pytest.approx(0.5)
        )

        assert at_half["recall"] == pytest.approx(body["rates"]["recall"])
        assert at_half["precision"] == pytest.approx(body["rates"]["precision"])


class TestPoolingAgainstAveraging:
    """The argument the library settles by declining to average."""

    def test_seven_uneven_folds_make_the_two_routes_disagree(self) -> None:
        body = folded(7, stratified=False)

        assert body["mean_fold_accuracy"] == pytest.approx(0.8571, abs=5e-5)
        assert body["pooled_accuracy"] == pytest.approx(FIVE_SIXTHS)
        assert body["mean_fold_accuracy"] != pytest.approx(body["pooled_accuracy"])

    def test_the_pooled_figure_is_the_same_however_the_folds_are_dealt(
        self,
    ) -> None:
        # Every row is held out exactly once whatever the deal, so the pooled
        # table is the same table and the pooled figures cannot move. This is
        # the whole case for pooling.
        deals = [
            folded(count, stratified)
            for count in (5, 7)
            for stratified in (False, True)
        ]

        for body in deals:
            assert body["pooled_accuracy"] == pytest.approx(FIVE_SIXTHS)
            assert body["pooled_recall"] == pytest.approx(FIVE_SIXTHS)
            assert body["pooled_macro_precision"] == pytest.approx(FIVE_SIXTHS)

    def test_the_averaged_figure_moves_with_the_deal(self) -> None:
        averaged = {
            folded(count, stratified)["mean_fold_accuracy"]
            for count in (5, 7)
            for stratified in (False, True)
        }

        assert len(averaged) > 1

    def test_a_fold_can_hold_no_positive_at_all(self) -> None:
        # Recall on such a fold is zero over zero, which averaging has to
        # invent a convention for and pooling simply adds nothing to.
        body = folded(5, stratified=False)

        assert any(fold["recall"] is None for fold in body["folds"])
        assert any(fold["positives_held_out"] == 0 for fold in body["folds"])

    def test_every_row_is_held_out_exactly_once(self) -> None:
        body = folded(7, stratified=False)

        assert sum(fold["held_out"] for fold in body["folds"]) == len(OVERLAPPING_CROWD)
        assert sum(fold["positives_held_out"] for fold in body["folds"]) == 6


class TestStratifying:
    """It makes a missing class rare, and nothing makes it impossible."""

    def test_stratifying_removes_the_empty_fold_at_five(self) -> None:
        assert folded(5, stratified=False)["classes_missing_from_a_fold"] == 1
        assert folded(5, stratified=True)["classes_missing_from_a_fold"] == 0

    def test_stratifying_cannot_remove_it_at_seven(self) -> None:
        # Twelve people in seven folds leaves folds of one person, and one
        # person cannot carry both classes however carefully the deal is made.
        plain = folded(7, stratified=False)["classes_missing_from_a_fold"]
        stratified = folded(7, stratified=True)["classes_missing_from_a_fold"]

        assert stratified < plain
        assert stratified > 0

    def test_the_spread_is_reported_alongside_the_pooled_figure(self) -> None:
        body = folded(5, stratified=False)

        assert body["accuracy_spread"] == pytest.approx(0.5)


class TestRefusals:
    def test_a_crowd_of_one_class_is_refused(self) -> None:
        response = client.post(
            EVALUATE,
            json={
                "points": [
                    {"x": 118, "y": 24, "label": 0},
                    {"x": 120, "y": 25, "label": 0},
                ],
                "threshold": 0.5,
            },
        )

        assert response.status_code == 422

    def test_a_threshold_at_the_ends_is_refused(self) -> None:
        for threshold in (0.0, 1.0):
            response = client.post(
                EVALUATE,
                json={"points": OVERLAPPING_CROWD, "threshold": threshold},
            )
            assert response.status_code == 422

    def test_one_fold_is_refused(self) -> None:
        response = client.post(FOLD, json={"points": OVERLAPPING_CROWD, "n_folds": 1})

        assert response.status_code == 422

    def test_a_field_the_request_does_not_have_is_refused(self) -> None:
        response = client.post(
            EVALUATE,
            json={
                "points": OVERLAPPING_CROWD,
                "threshold": 0.5,
                "cutoff": 0.5,
            },
        )

        assert response.status_code == 422

    def test_too_few_people_are_refused(self) -> None:
        response = client.post(
            EVALUATE, json={"points": [{"x": 118, "y": 24, "label": 0}]}
        )

        assert response.status_code == 422
