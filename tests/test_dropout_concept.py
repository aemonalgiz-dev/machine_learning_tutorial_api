"""The dropout endpoint, pinned to the page's hand arithmetic.

The page works four units holding 1, 2, 3 and 4 at a drop probability of one
half under seed 10. The first draw keeps the first and third units and
silences the other two, so with the survivors doubled the row reads 2, 0, 6,
0, while the same row passed while predicting reads 1, 2, 3, 4 untouched.
Over four hundred draws the units survive 200, 192, 190 and 204 times, which
makes their mean outputs 1.00, 1.92, 2.85 and 4.08, none further than 0.15
from the predicting answer, with the running means passing through 1.0, 2.4,
2.4, 4.8 after ten draws and 1.08, 2.04, 2.58, 3.76 after a hundred. The
page also puts a zero into a surviving unit to show the mask still says
kept. All of it is pinned here, along with the refusals.
"""

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

APPLY = "/concepts/dropout/apply"

WORKED_EXAMPLE = {
    "values": [1.0, 2.0, 3.0, 4.0],
    "drop_probability": 0.5,
    "random_seed": 10,
    "n_draws": 400,
}


def worked_example(**changes: object) -> dict:
    response = client.post(APPLY, json={**WORKED_EXAMPLE, **changes})
    assert response.status_code == 200
    return response.json()


class TestTheOneDraw:
    def test_the_scale_is_two_at_one_half(self) -> None:
        assert worked_example()["scale"] == 2.0

    def test_the_first_draw_keeps_the_first_and_third_units(self) -> None:
        body = worked_example()

        assert body["kept"] == [True, False, True, False]
        assert body["mask"] == [2.0, 0.0, 2.0, 0.0]

    def test_the_survivors_are_doubled_and_the_rest_are_zero(self) -> None:
        assert worked_example()["training_output"] == [2.0, 0.0, 6.0, 0.0]

    def test_predicting_passes_the_row_through_untouched(self) -> None:
        assert worked_example()["predicting_output"] == [1.0, 2.0, 3.0, 4.0]

    def test_the_same_seed_draws_the_same_sequence(self) -> None:
        assert worked_example() == worked_example()

    def test_the_seed_and_draw_count_default_to_the_worked_example(self) -> None:
        response = client.post(
            APPLY, json={"values": [1.0, 2.0, 3.0, 4.0], "drop_probability": 0.5}
        )

        assert response.status_code == 200
        assert response.json() == worked_example()

    def test_a_different_seed_is_a_different_draw(self) -> None:
        # Seed 3 keeps the third and fourth units instead.
        body = worked_example(random_seed=3)

        assert body["kept"] == [False, False, True, True]
        assert body["training_output"] == [0.0, 0.0, 6.0, 8.0]

    def test_a_quarter_scales_the_survivors_by_four_thirds(self) -> None:
        body = worked_example(drop_probability=0.25)

        assert abs(body["scale"] - 4.0 / 3.0) < 1e-12
        assert body["kept"] == [True, False, True, False]
        assert abs(body["training_output"][0] - 4.0 / 3.0) < 1e-12
        assert body["training_output"][2] == 4.0


class TestTheAverage:
    def test_the_units_survive_a_pinned_number_of_times(self) -> None:
        body = worked_example()

        assert body["n_draws"] == 400
        assert body["times_kept"] == [200, 192, 190, 204]

    def test_the_mean_over_four_hundred_draws(self) -> None:
        body = worked_example()

        assert [round(value, 6) for value in body["mean_output"]] == [
            1.0,
            1.92,
            2.85,
            4.08,
        ]

    def test_the_mean_is_the_survival_share_times_the_doubled_value(self) -> None:
        # The derivation's expectation with a count in place of a probability.
        # Each unit's mean is how often it survived, over the draws, times its
        # doubled value.
        body = worked_example()

        for unit_index, (count, value) in enumerate(
            zip(body["times_kept"], WORKED_EXAMPLE["values"], strict=True)
        ):
            expected = count / 400 * value * 2.0
            assert abs(body["mean_output"][unit_index] - expected) < 1e-12

    def test_the_largest_gap_is_fifteen_hundredths(self) -> None:
        assert round(worked_example()["largest_gap"], 6) == 0.15

    def test_the_running_mean_runs_from_the_first_draw_to_the_mean(self) -> None:
        body = worked_example()

        assert len(body["running_means"]) == 400
        assert body["running_means"][0] == body["training_output"]
        assert body["running_means"][-1] == body["mean_output"]

    def test_the_running_mean_after_ten_and_after_a_hundred_draws(self) -> None:
        body = worked_example()

        after_ten = [round(value, 6) for value in body["running_means"][9]]
        after_a_hundred = [round(value, 6) for value in body["running_means"][99]]
        assert after_ten == [1.0, 2.4, 2.4, 4.8]
        assert after_a_hundred == [1.08, 2.04, 2.58, 3.76]

    def test_a_shorter_run_is_a_prefix_of_a_longer_one(self) -> None:
        # One seeded layer whose generator advances between passes, so the
        # first ten draws of a long run are the ten draws of a short one.
        short = worked_example(n_draws=10)
        long = worked_example()

        assert short["running_means"] == long["running_means"][:10]
        assert short["training_output"] == long["training_output"]
        assert short["mask"] == long["mask"]


class TestTheMaskIsADraw:
    def test_a_zero_that_survived_is_reported_as_kept(self) -> None:
        # The first unit now holds zero. Its output is zero either way, and the
        # mask is the only thing that can say it survived, and it does.
        body = worked_example(values=[0.0, 2.0, 3.0, 4.0])

        assert body["training_output"] == [0.0, 0.0, 6.0, 0.0]
        assert body["kept"] == [True, False, True, False]
        assert body["mask"] == [2.0, 0.0, 2.0, 0.0]

    def test_the_survival_count_is_read_from_the_mask_not_the_output(self) -> None:
        body = worked_example(values=[0.0, 2.0, 3.0, 4.0])

        assert body["times_kept"] == [200, 192, 190, 204]
        assert body["mean_output"][0] == 0.0

    def test_no_dropping_at_all_is_the_identity(self) -> None:
        body = worked_example(drop_probability=0.0, n_draws=5)

        assert body["scale"] == 1.0
        assert body["kept"] == [True, True, True, True]
        assert body["training_output"] == body["predicting_output"]
        assert body["times_kept"] == [5, 5, 5, 5]
        assert body["largest_gap"] == 0.0


class TestRefusals:
    def test_a_probability_of_one_is_refused_in_the_library_words(self) -> None:
        response = client.post(APPLY, json={**WORKED_EXAMPLE, "drop_probability": 1.0})

        assert response.status_code == 422
        assert "silences the whole layer" in response.json()["detail"]

    def test_an_empty_row_is_refused(self) -> None:
        response = client.post(APPLY, json={**WORKED_EXAMPLE, "values": []})

        assert response.status_code == 422

    def test_a_row_wider_than_the_widget_allows_is_refused(self) -> None:
        response = client.post(APPLY, json={**WORKED_EXAMPLE, "values": [1.0] * 13})

        assert response.status_code == 422

    def test_too_many_draws_are_refused(self) -> None:
        response = client.post(APPLY, json={**WORKED_EXAMPLE, "n_draws": 1001})

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(APPLY, json={**WORKED_EXAMPLE, "keep_probability": 0.5})

        assert response.status_code == 422
