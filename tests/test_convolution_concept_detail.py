"""The convolution page's later parts, pinned to the numbers it quotes.

The first test module covers the sweep itself and the parameter counts. This
one covers what the page adds around them: that a dense layer cannot tell the
picture from a shuffle of it, that the sweep's answer moves with the picture
and a dense layer's does not, the arrangement worked out in integers before
any data arrives, the backward pass checked against a central difference of
the loss, what the sweep costs, and every refusal the last part quotes.

Two of those are pinned as orderings rather than as figures. The cost endpoint
times whichever machine ran it, so what is asserted is which of the three
passes is quicker at which size, that the loops and the sweep agree, and that
the arrangement is right, not the milliseconds. And
the gradient check is asserted against a tolerance rather than a value,
because a central difference agrees with an exact slope to about the square of
its step and not to a fixed number of digits.
"""

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

ARRANGEMENT = "/concepts/convolution/arrangement"
SHIFT = "/concepts/convolution/shift"
BANK = "/concepts/convolution/bank"
GEOMETRY = "/concepts/convolution/geometry"
BACKWARD = "/concepts/convolution/backward"
COST = "/concepts/convolution/cost"
EDGE_CASES = "/concepts/convolution/edge-cases"

VERTICAL_EDGE = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
HORIZONTAL_EDGE = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
BLUR = [[1 / 9] * 3] * 3

SQUARE = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]


def get(path: str) -> dict:
    response = client.get(path)
    assert response.status_code == 200
    return response.json()


def post(path: str, body: dict) -> dict:
    response = client.post(path, json=body)
    assert response.status_code == 200
    return response.json()


class TestADenseLayerCannotSeeTheArrangement:
    def test_the_shuffle_moves_eighteen_of_the_sixty_four_cells(self) -> None:
        body = get(ARRANGEMENT)

        assert body["n_picture_cells"] == 64
        assert body["picture_cells_moved"] == 18

    def test_the_rewired_dense_layer_answers_the_same_to_one_rounding_step(
        self,
    ) -> None:
        body = get(ARRANGEMENT)

        assert body["dense_largest_gap"] == pytest.approx(0.0, abs=1e-15)
        assert body["dense_largest_gap"] > 0.0

    def test_and_not_bit_for_bit_because_the_products_are_summed_in_another_order(
        self,
    ) -> None:
        body = get(ARRANGEMENT)

        assert body["dense_identical"] is False

    def test_the_sweep_answers_something_else_entirely(self) -> None:
        body = get(ARRANGEMENT)

        assert body["swept_largest_gap"] == 6.0
        assert body["swept_cells_changed"] == 27
        assert body["n_answer_cells"] == 36

    def test_the_two_readers_hold_ten_parameters_and_two_thousand_three_hundred_and_forty(
        self,
    ) -> None:
        body = get(ARRANGEMENT)

        assert body["convolution_parameters"] == 10
        assert body["dense_parameters"] == 2340


class TestTheAnswerMovesWithThePicture:
    def test_every_shift_gives_the_resting_answer_moved_by_the_same_amount(
        self,
    ) -> None:
        body = post(SHIFT, {"kernel": VERTICAL_EDGE, "shifts": [0, 1, 2, 3]})

        for entry in body["shifts"]:
            assert entry["output_shift"] == entry["shift"]
            assert entry["largest_gap"] == 0.0
            assert entry["matches_moved_answer"] is True

    @pytest.mark.parametrize("kernel", [VERTICAL_EDGE, HORIZONTAL_EDGE, BLUR])
    def test_it_holds_for_whatever_kernel_the_layer_carries(
        self, kernel: list[list[float]]
    ) -> None:
        body = post(SHIFT, {"kernel": kernel, "shifts": [1, 2]})

        assert all(entry["largest_gap"] == 0.0 for entry in body["shifts"])

    def test_the_picture_really_did_change_under_it(self) -> None:
        body = post(SHIFT, {"kernel": VERTICAL_EDGE, "shifts": [1, 2]})

        assert [entry["picture_cells_changed"] for entry in body["shifts"]] == [8, 16]
        assert [entry["swept_cells_changed"] for entry in body["shifts"]] == [18, 36]


class TestADenseLayerOfEqualWidthDoesNot:
    def test_the_single_detector_answers_four_at_rest_and_nothing_after(self) -> None:
        body = post(SHIFT, {"kernel": VERTICAL_EDGE, "shifts": [0, 1, 2, 3]})

        assert body["detector_at_rest"] == 4.0
        assert [entry["detector_answer"] for entry in body["shifts"]] == [
            4.0,
            0.0,
            0.0,
            0.0,
        ]

    def test_and_the_sweep_answers_four_at_the_moved_position_every_time(self) -> None:
        body = post(SHIFT, {"kernel": VERTICAL_EDGE, "shifts": [0, 1, 2, 3]})

        for entry in body["shifts"]:
            assert entry["swept"][2][entry["shift"]] == 4.0

    def test_an_ordinary_draw_of_equal_width_moves_by_more_than_it_answers(
        self,
    ) -> None:
        body = post(SHIFT, {"kernel": VERTICAL_EDGE, "shifts": [1, 2]})

        for entry in body["shifts"]:
            assert entry["dense_largest_gap"] > entry["dense_largest_answer"]
            assert entry["dense_cells_changed"] == 36


class TestAStrideBreaksTheShiftsItCannotDivide:
    def test_a_shift_of_one_at_a_stride_of_two_has_nowhere_to_move_to(self) -> None:
        body = post(SHIFT, {"kernel": VERTICAL_EDGE, "stride": 2, "shifts": [1]})
        entry = body["shifts"][0]

        assert entry["output_shift"] is None
        assert entry["largest_gap"] is None
        assert entry["matches_moved_answer"] is False

    def test_a_shift_of_two_moves_the_answer_one_position(self) -> None:
        body = post(SHIFT, {"kernel": VERTICAL_EDGE, "stride": 2, "shifts": [2]})
        entry = body["shifts"][0]

        assert body["answers"] == [1, 3, 3]
        assert entry["output_shift"] == 1
        assert entry["largest_gap"] == 0.0


class TestABankOfKernels:
    def test_three_filters_answer_with_three_maps_of_the_same_size(self) -> None:
        body = post(
            BANK,
            {"picture": SQUARE, "kernels": [VERTICAL_EDGE, HORIZONTAL_EDGE, BLUR]},
        )

        assert body["reads"] == [1, 8, 8]
        assert body["answers"] == [3, 6, 6]
        assert body["kernel_shape"] == [3, 1, 3, 3]
        assert len(body["maps"]) == 3

    def test_each_filter_answers_most_strongly_somewhere_of_its_own(self) -> None:
        body = post(
            BANK,
            {"picture": SQUARE, "kernels": [VERTICAL_EDGE, HORIZONTAL_EDGE, BLUR]},
        )
        strongest = [
            (entry["strongest_row"], entry["strongest_column"])
            for entry in body["maps"]
        ]

        assert strongest == [(2, 0), (0, 2), (2, 2)]

    def test_the_bank_costs_ten_parameters_a_filter(self) -> None:
        body = post(
            BANK,
            {"picture": SQUARE, "kernels": [VERTICAL_EDGE, HORIZONTAL_EDGE, BLUR]},
        )

        assert body["parameters"]["convolution"] == 30
        assert body["parameters"]["n_outputs"] == 108
        assert body["parameters"]["dense"] == 7020

    def test_one_filter_matches_what_the_single_kernel_endpoint_answers(self) -> None:
        alone = post(BANK, {"picture": SQUARE, "kernels": [VERTICAL_EDGE]})
        swept = client.post(
            "/concepts/convolution/apply",
            json={"picture": SQUARE, "kernel": VERTICAL_EDGE},
        ).json()

        assert alone["maps"][0]["output"] == swept["output"]


class TestTheArrangementBeforeAnyData:
    def test_the_plain_sweep_shrinks_the_side_by_two(self) -> None:
        body = post(
            GEOMETRY,
            {"side": 8, "kernel_sizes": [3], "strides": [1], "paddings": [0]},
        )
        row = body["rows"][0]

        assert row["answers"] == [1, 6, 6]
        assert row["keeps_the_side"] is False

    def test_one_row_of_zeros_keeps_the_side_and_lifts_the_corner_from_one_to_four(
        self,
    ) -> None:
        body = post(
            GEOMETRY,
            {"side": 8, "kernel_sizes": [3], "strides": [1], "paddings": [0, 1]},
        )
        plain, padded = body["rows"]

        assert plain["corner_windows"] == 1
        assert plain["centre_windows"] == 9
        assert padded["corner_windows"] == 4
        assert padded["centre_windows"] == 9
        assert padded["answers"] == [1, 8, 8]
        assert padded["keeps_the_side"] is True

    def test_a_stride_of_two_leaves_fifteen_cells_no_window_reaches(self) -> None:
        body = post(
            GEOMETRY,
            {"side": 8, "kernel_sizes": [3], "strides": [2], "paddings": [0]},
        )
        row = body["rows"][0]

        assert row["answers"] == [1, 3, 3]
        assert row["unread_cells"] == 15

    def test_the_count_of_parameters_does_not_depend_on_the_picture(self) -> None:
        body = post(
            GEOMETRY,
            {
                "side": 8,
                "n_filters": 2,
                "kernel_sizes": [3],
                "strides": [1, 2],
                "paddings": [0, 1],
            },
        )

        assert {row["parameters"] for row in body["rows"]} == {20}

    def test_a_kernel_that_does_not_fit_is_a_refused_row_beside_the_others(
        self,
    ) -> None:
        body = post(
            GEOMETRY,
            {"side": 4, "kernel_sizes": [3, 7], "strides": [1], "paddings": [0]},
        )
        fits, does_not = body["rows"]

        assert fits["refused"] is False
        assert does_not["refused"] is True
        assert "(4 - 7 + 2 * 0) // 1 + 1 = -2" in does_not["detail"]
        assert does_not["answers"] is None

    def test_a_wider_window_reads_the_middle_more_and_the_corner_no_more(
        self,
    ) -> None:
        # The page's coverage table quotes all four of these rows, and the
        # claim they carry is that widening the window lifts the middle's
        # coverage while the corner stays at one window until padding arrives.
        body = post(
            GEOMETRY,
            {"side": 8, "kernel_sizes": [5], "strides": [1], "paddings": [0, 1]},
        )
        plain, padded = body["rows"]

        assert plain["answers"] == [1, 4, 4]
        assert plain["corner_windows"] == 1
        assert plain["centre_windows"] == 16
        assert padded["answers"] == [1, 6, 6]
        assert padded["corner_windows"] == 4
        assert padded["centre_windows"] == 25
        assert plain["parameters"] == padded["parameters"] == 26


class TestTheBackwardPassWorkedByHand:
    def test_every_kernel_entry_wants_the_number_of_lit_cells(self) -> None:
        worked = get(BACKWARD)["worked"]

        assert worked["n_lit_cells"] == 16
        assert worked["kernel_slopes"] == [[16.0] * 3] * 3

    def test_the_library_and_the_hand_count_agree(self) -> None:
        worked = get(BACKWARD)["worked"]

        assert worked["kernel_slopes_by_hand"] == worked["kernel_slopes"]
        assert worked["slopes_agree"] is True

    def test_the_bias_wants_one_for_every_position_it_was_added_at(self) -> None:
        worked = get(BACKWARD)["worked"]

        assert worked["n_output_positions"] == 36
        assert worked["bias_slope"] == 36.0

    def test_a_corner_cell_collects_one_kernel_entry_and_an_interior_cell_all_nine(
        self,
    ) -> None:
        worked = get(BACKWARD)["worked"]

        assert worked["corner_blame"] == -1.0
        assert worked["kernel_total"] == 0.0
        assert worked["centre_blame"] == 0.0

    def test_thirty_two_of_the_sixty_four_cells_receive_nothing_at_all(self) -> None:
        worked = get(BACKWARD)["worked"]

        assert worked["n_cells_receiving_nothing"] == 32


class TestTheBackwardPassAgainstFiniteDifferences:
    def test_all_three_fixtures_are_checked(self) -> None:
        body = get(BACKWARD)

        assert [check["name"] for check in body["checks"]] == [
            "a plain sweep",
            "padded by one",
            "two channels at a stride of two",
        ]

    def test_every_kernel_bias_and_input_slope_agrees(self) -> None:
        body = get(BACKWARD)

        for check in body["checks"]:
            assert check["largest_kernel_gap"] < 1e-6
            assert check["largest_bias_gap"] < 1e-6
            assert check["largest_input_gap"] < 1e-6

    def test_the_agreement_is_far_below_the_slopes_being_reported(self) -> None:
        body = get(BACKWARD)

        for check in body["checks"]:
            assert check["largest_true_slope"] > 1.0
            assert body["largest_gap_anywhere"] < 1e-6

    def test_the_strided_fixture_checks_the_most(self) -> None:
        body = get(BACKWARD)

        assert [check["n_checked"] for check in body["checks"]] == [70, 70, 182]

    def test_the_worst_gap_and_the_largest_slope_the_page_quotes(self) -> None:
        # The page says the worst disagreement anywhere was 7.8e-09, on the
        # padded fixture's inputs, against reported slopes as large as 13.9.
        # Both are deterministic, since every fixture is drawn under a fixed
        # seed; the gap moved from 5.3e-09 when the sweep became a contraction,
        # because the losses are summed in another order and the difference
        # divides that rounding by twice the step.
        body = get(BACKWARD)
        padded = body["checks"][1]

        assert body["largest_gap_anywhere"] == pytest.approx(7.8e-09, rel=0.1)
        assert padded["largest_input_gap"] == body["largest_gap_anywhere"]
        assert padded["largest_true_slope"] == pytest.approx(13.87, abs=0.01)


class TestWhatTheSweepCosts:
    def test_three_sizes_are_timed_in_order(self) -> None:
        body = get(COST)

        assert [row["side"] for row in body["rows"]] == [8, 14, 28]
        assert [row["answers"] for row in body["rows"]] == [
            [1, 6, 6],
            [4, 12, 12],
            [8, 26, 26],
        ]

    def test_each_pass_adds_up_one_product_per_kernel_entry_per_answer(self) -> None:
        body = get(COST)

        assert [row["n_terms"] for row in body["rows"]] == [324, 5184, 48672]

    def test_the_loops_and_the_sweep_give_the_same_answer_at_every_size(
        self,
    ) -> None:
        # Deterministic, since every draw is seeded. The page quotes the
        # worst of the three, which is a few rounding steps and no more.
        body = get(COST)

        assert [row["largest_disagreement"] for row in body["rows"]] == [
            pytest.approx(2.2e-16, rel=0.1),
            pytest.approx(8.9e-16, rel=0.1),
            pytest.approx(1.8e-15, rel=0.1),
        ]
        assert body["largest_disagreement_anywhere"] < 1e-12

    def test_the_loops_fall_further_behind_the_sweep_as_the_picture_grows(
        self,
    ) -> None:
        # Timings, so orderings rather than figures. On the page's own
        # picture the loops and the sweep are within a few times of each
        # other, since there are only 324 products and the sweep's fixed
        # overhead is most of its time; at 28 by 28 through eight filters the
        # loops took over a hundred times as long on every run behind the
        # page, so ten is a floor no ordinary run comes near.
        rows = get(COST)["rows"]

        assert rows[1]["loop_milliseconds"] > rows[1]["sweep_milliseconds"]
        assert rows[2]["loop_over_sweep"] > 10.0
        assert rows[2]["loop_over_sweep"] > rows[0]["loop_over_sweep"]

    def test_the_sweep_is_slower_than_the_dense_layer_small_and_quicker_large(
        self,
    ) -> None:
        # On the page's picture a dense layer of 2,340 parameters is one small
        # matrix multiply and the sweep's fixed overhead dominates, so the
        # sweep is several times slower. At 28 by 28 through eight filters the
        # dense layer multiplies 4,245,280 weights and the sweep a view of
        # 48,672 products, and the order reverses. The ratio falls between
        # the middle size and the largest on every run behind the page.
        rows = get(COST)["rows"]

        assert rows[0]["ratio"] > 1.0
        assert rows[2]["ratio"] < 1.0
        assert rows[1]["ratio"] > rows[2]["ratio"]

    def test_and_holds_far_fewer_parameters_at_every_size(self) -> None:
        body = get(COST)

        for row in body["rows"]:
            assert row["convolution_parameters"] < row["dense_parameters"]
        assert [row["convolution_parameters"] for row in body["rows"]] == [10, 40, 80]
        assert body["rows"][-1]["dense_parameters"] == 4_245_280


class TestTheEdges:
    def test_sixteen_are_refused_and_three_accepted(self) -> None:
        body = get(EDGE_CASES)

        assert body["n_refused"] == 16
        assert body["n_accepted"] == 3
        assert len(body["cases"]) == 19

    def test_every_refusal_is_in_the_library_own_words(self) -> None:
        body = get(EDGE_CASES)

        assert body["n_in_the_library_words"] == body["n_refused"]

    def test_a_bare_integer_where_extents_are_wanted_is_one_of_them(self) -> None:
        body = get(EDGE_CASES)
        case = next(
            entry
            for entry in body["cases"]
            if entry["name"] == "a bare integer where extents are wanted"
        )

        assert case["error"] == "ShapeMismatchError"
        assert case["library_error"] is True
        assert "not a sequence of extents" in case["detail"]

    def test_the_three_accepted_are_named(self) -> None:
        body = get(EDGE_CASES)
        accepted = [entry["name"] for entry in body["cases"] if not entry["raised"]]

        assert accepted == [
            "a kernel the size of the whole picture",
            "a stride larger than the kernel",
            "padding wider than the kernel",
        ]

    def test_a_window_the_size_of_the_picture_is_a_dense_layer_in_disguise(
        self,
    ) -> None:
        body = get(EDGE_CASES)
        case = next(
            entry
            for entry in body["cases"]
            if entry["name"] == "a kernel the size of the whole picture"
        )

        assert "answering (1, 1, 1) from one window" in case["outcome"]
        assert "65 parameters reading 64" in case["outcome"]

    def test_a_stride_larger_than_the_kernel_leaves_cells_no_window_reaches(
        self,
    ) -> None:
        body = get(EDGE_CASES)
        case = next(
            entry
            for entry in body["cases"]
            if entry["name"] == "a stride larger than the kernel"
        )

        assert "leaving 28 of 64 cells outside every window" in case["outcome"]

    def test_padding_wider_than_the_kernel_strands_whole_windows_on_the_border(
        self,
    ) -> None:
        body = get(EDGE_CASES)
        case = next(
            entry
            for entry in body["cases"]
            if entry["name"] == "padding wider than the kernel"
        )

        assert "96 of 196 positions read nothing but border" in case["outcome"]


class TestTheJoinAStackRefuses:
    def test_a_layer_that_reads_a_row_cannot_sit_straight_on_a_convolution(
        self,
    ) -> None:
        joins = get(EDGE_CASES)["joins"]

        assert joins[0]["refused"] is True
        assert joins[0]["error"] == "ShapeMismatchError"
        assert joins[0]["detail"] == (
            "layer 0 answers with (1, 6, 6) and layer 1 reads (36,); both hold "
            "36 numbers, so it is the arrangement that disagrees and not the width"
        )

    def test_the_refusal_admits_that_the_counts_agree(self) -> None:
        joins = get(EDGE_CASES)["joins"]

        assert joins[0]["counts_agree"] is True
        assert joins[0]["both_hold"] == 36

    def test_a_flattening_layer_between_them_is_accepted(self) -> None:
        joins = get(EDGE_CASES)["joins"]

        assert joins[1]["refused"] is False
        assert joins[1]["n_layers"] == 3
        assert joins[1]["stack_reads"] == [1, 8, 8]
        assert joins[1]["stack_answers"] == [3]

    def test_the_library_own_example_is_refused_the_same_way(self) -> None:
        joins = get(EDGE_CASES)["joins"]

        assert joins[2]["detail"] == (
            "layer 0 answers with (8, 26, 26) and layer 1 reads (5408,); both "
            "hold 5408 numbers, so it is the arrangement that disagrees and not "
            "the width"
        )


class TestRefusals:
    def test_a_bank_of_more_than_six_kernels_is_refused(self) -> None:
        response = client.post(
            BANK, json={"picture": SQUARE, "kernels": [VERTICAL_EDGE] * 7}
        )

        assert response.status_code == 422

    def test_a_shift_past_the_picture_is_refused(self) -> None:
        response = client.post(SHIFT, json={"kernel": VERTICAL_EDGE, "shifts": [9]})

        assert response.status_code == 422

    def test_an_unknown_setting_on_a_shift_is_refused(self) -> None:
        response = client.post(
            SHIFT, json={"kernel": VERTICAL_EDGE, "shifts": [1], "dilation": 2}
        )

        assert response.status_code == 422

    def test_a_geometry_side_below_the_bound_is_refused(self) -> None:
        response = client.post(
            GEOMETRY,
            json={
                "side": 2,
                "kernel_sizes": [3],
                "strides": [1],
                "paddings": [0],
            },
        )

        assert response.status_code == 422
