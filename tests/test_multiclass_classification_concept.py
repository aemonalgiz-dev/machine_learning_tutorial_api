"""The multiclass endpoint, pinned to the page's hand arithmetic.

The page works three people, one per class, standing on a line at (120, 25),
(150, 50) and (180, 75). The middle person sits at the mean of both columns,
so after standardising their numbers are zero and their softmax scores are
the intercepts alone, 0, 3.96 and -0.79, which exponentiate to 1, 52.47 and
0.45, total 53.92, and divide to 0.02, 0.97 and 0.01, summing to one. Under
one-vs-rest the middle class's own fit settles at exactly one third for
everyone, because no line can pick the middle person out from the two on
either side and the best a line can do is answer a third for all three, so
the middle person's three chances total 0.35 and each end person's total
1.33. On the crowd the teenagers' totals run from 0.24 to 0.59 and everyone
else's from 1.31 to 1.50 under one-vs-rest, and every total is one under
softmax. All pinned here.
"""

import numpy as np
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

CLASSIFY = "/concepts/multiclass-classification/classify"

# Three people on a line, one per class, the page's worked example.
THREE_PEOPLE = [
    {"x": 120, "y": 25, "label": 0},
    {"x": 150, "y": 50, "label": 1},
    {"x": 180, "y": 75, "label": 2},
]

# The classification pages' crowd with its middle relabelled as teenagers and
# one more child and one more adult, the widget's opening crowd.
CROWD = [
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
]

MIDDLE = 1
"""The index of the worked example's middle person."""


def classify(points: list[dict], route: str) -> dict:
    response = client.post(CLASSIFY, json={"points": points, "route": route})
    assert response.status_code == 200, response.json()
    return response.json()


class TestWorkedExampleUnderSoftmax:
    def test_the_middle_person_scores_are_the_intercepts(self) -> None:
        # Standardised, the middle person is zero in both columns, so the
        # scores before the squash are the intercepts alone, class 0's at zero.
        body = classify(THREE_PEOPLE, "softmax")

        linear = body["linear_scores"][MIDDLE]
        assert linear[0] == 0.0
        assert abs(linear[1] - 3.9602) < 1e-4
        assert abs(linear[2] - (-0.7918)) < 1e-4

    def test_the_exponentials_total_the_page_figure(self) -> None:
        body = classify(THREE_PEOPLE, "softmax")

        exponentials = np.exp(body["linear_scores"][MIDDLE])
        assert abs(exponentials[1] - 52.47) < 0.01
        assert abs(exponentials[2] - 0.45) < 0.01
        assert abs(exponentials.sum() - 53.92) < 0.01

    def test_the_middle_person_probabilities_are_the_page_figures(self) -> None:
        body = classify(THREE_PEOPLE, "softmax")

        scores = body["scores"][MIDDLE]
        assert abs(scores[0] - 0.0185) < 1e-4
        assert abs(scores[1] - 0.9731) < 1e-4
        assert abs(scores[2] - 0.0084) < 1e-4

    def test_every_row_sums_to_one(self) -> None:
        body = classify(THREE_PEOPLE, "softmax")

        assert all(abs(total - 1.0) < 1e-9 for total in body["row_sums"])

    def test_the_scores_are_the_softmax_of_the_linear_scores(self) -> None:
        # The endpoint assembles the linear scores from the library's
        # intercepts and weights; pushing them through a softmax has to land
        # on the library's own probabilities, or the assembly is wrong.
        body = classify(THREE_PEOPLE, "softmax")

        for linear, scores in zip(body["linear_scores"], body["scores"], strict=True):
            exponentials = np.exp(np.asarray(linear))
            assert np.allclose(exponentials / exponentials.sum(), scores, atol=1e-9)

    def test_the_walk_ran_out_of_passes_on_separable_people(self) -> None:
        body = classify(THREE_PEOPLE, "softmax")

        assert body["epochs_run"] == 500
        assert body["converged"] is False
        assert body["class_fits"] == []

    def test_everyone_is_called_their_own_class(self) -> None:
        body = classify(THREE_PEOPLE, "softmax")

        assert body["predictions"] == [0, 1, 2]
        assert body["accuracy"] == 1.0


class TestWorkedExampleUnderOneVsRest:
    def test_the_middle_fit_settles_at_one_third_for_everyone(self) -> None:
        # No line separates the middle person from the two on either side,
        # and the likelihood p(1-p)^2 peaks at p = 1/3.
        body = classify(THREE_PEOPLE, "one_vs_rest")

        middle_column = [row[MIDDLE] for row in body["scores"]]
        assert all(abs(chance - 1 / 3) < 1e-4 for chance in middle_column)

    def test_the_middle_fit_intercept_is_log_one_half(self) -> None:
        # A third is the sigmoid of log(1/2), and the slope is zero.
        body = classify(THREE_PEOPLE, "one_vs_rest")

        assert abs(body["linear_scores"][MIDDLE][MIDDLE] - np.log(0.5)) < 1e-4

    def test_the_middle_person_totals_the_page_figure(self) -> None:
        body = classify(THREE_PEOPLE, "one_vs_rest")

        scores = body["scores"][MIDDLE]
        assert abs(scores[0] - 0.0101) < 1e-4
        assert abs(scores[2] - 0.0101) < 1e-4
        assert abs(body["row_sums"][MIDDLE] - 0.3535) < 1e-4

    def test_each_end_person_totals_the_page_figure(self) -> None:
        body = classify(THREE_PEOPLE, "one_vs_rest")

        assert abs(body["row_sums"][0] - 1.3293) < 1e-4
        assert abs(body["row_sums"][2] - 1.3293) < 1e-4
        assert abs(body["scores"][0][0] - 0.996) < 1e-3

    def test_the_scores_are_the_sigmoid_of_the_linear_scores(self) -> None:
        body = classify(THREE_PEOPLE, "one_vs_rest")

        for linear, scores in zip(body["linear_scores"], body["scores"], strict=True):
            chances = 1.0 / (1.0 + np.exp(-np.asarray(linear)))
            assert np.allclose(chances, scores, atol=1e-9)

    def test_only_the_middle_fit_converged(self) -> None:
        body = classify(THREE_PEOPLE, "one_vs_rest")

        fits = body["class_fits"]
        assert [fit["class_index"] for fit in fits] == [0, 1, 2]
        assert [fit["positive_rows"] for fit in fits] == [1, 1, 1]
        assert [fit["converged"] for fit in fits] == [False, True, False]
        assert fits[MIDDLE]["epochs_run"] == 49
        assert fits[0]["epochs_run"] == 500 and fits[2]["epochs_run"] == 500
        assert body["epochs_run"] is None
        assert body["converged"] is None

    def test_everyone_is_still_called_their_own_class(self) -> None:
        body = classify(THREE_PEOPLE, "one_vs_rest")

        assert body["predictions"] == [0, 1, 2]
        assert body["accuracy"] == 1.0


class TestCrowd:
    def test_both_routes_call_every_person_correctly(self) -> None:
        expected = [point["label"] for point in CROWD]

        for route in ("softmax", "one_vs_rest"):
            body = classify(CROWD, route)
            assert body["predictions"] == expected
            assert body["accuracy"] == 1.0

    def test_softmax_totals_are_one_for_everyone(self) -> None:
        body = classify(CROWD, "softmax")

        assert all(abs(total - 1.0) < 1e-9 for total in body["row_sums"])
        assert body["epochs_run"] == 500

    def test_one_vs_rest_totals_are_the_page_ranges(self) -> None:
        body = classify(CROWD, "one_vs_rest")

        teenagers = [
            total
            for total, point in zip(body["row_sums"], CROWD, strict=True)
            if point["label"] == 1
        ]
        others = [
            total
            for total, point in zip(body["row_sums"], CROWD, strict=True)
            if point["label"] != 1
        ]
        assert round(min(teenagers), 2) == 0.24
        assert round(max(teenagers), 2) == 0.59
        assert round(min(others), 2) == 1.31
        assert round(max(others), 2) == 1.50

    def test_the_three_binary_fits_see_the_crowd_unbalanced(self) -> None:
        body = classify(CROWD, "one_vs_rest")

        assert [fit["positive_rows"] for fit in body["class_fits"]] == [4, 5, 4]

    def test_the_regions_show_all_three_classes(self) -> None:
        for route in ("softmax", "one_vs_rest"):
            regions = classify(CROWD, route)["regions"]
            assert regions["cells"] == 26
            assert len(regions["labels"]) == 26
            seen = {label for row in regions["labels"] for label in row}
            assert seen == {0, 1, 2}


class TestRefusals:
    def test_a_crowd_missing_a_class_is_refused_in_words(self) -> None:
        response = client.post(CLASSIFY, json={"points": CROWD[:9], "route": "softmax"})

        assert response.status_code == 422
        assert "no adult" in response.json()["detail"]

    def test_a_constant_column_is_refused_by_the_library(self) -> None:
        same_height = [
            {"x": 150, "y": point["y"], "label": point["label"]} for point in CROWD
        ]
        response = client.post(
            CLASSIFY, json={"points": same_height, "route": "one_vs_rest"}
        )

        assert response.status_code == 422
        assert response.json()["error"] == "AllSameValuesError"

    def test_a_fourth_class_is_refused_at_the_door(self) -> None:
        response = client.post(
            CLASSIFY,
            json={
                "points": [*THREE_PEOPLE, {"x": 200, "y": 90, "label": 3}],
                "route": "softmax",
            },
        )

        assert response.status_code == 422

    def test_an_unknown_route_is_refused(self) -> None:
        response = client.post(
            CLASSIFY, json={"points": THREE_PEOPLE, "route": "argmax"}
        )

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            CLASSIFY,
            json={"points": THREE_PEOPLE, "route": "softmax", "epochs": 5},
        )

        assert response.status_code == 422
