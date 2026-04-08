"""The convolution endpoints, pinned to the page's hand arithmetic.

The page paints a four by four square in the middle of an eight by eight
picture and sweeps the vertical edge kernel across it. The cell at answer row
two, column zero reads the patch whose right column is the square's left edge,
and its nine products sum to 1 + 2 + 1 = 4. The whole of answer row two is
4, 4, 0, 0, -4, -4, positive where the square begins and negative where it
ends. One filter of three by three costs ten parameters, a layer that kept
the window but shared nothing would cost 360, and the dense layer of equal
width costs 2,340, which is 234 times the convolution. At the library's own
28 by 28 example through eight filters the ratio is 53,066. All pinned here,
together with the shift theorem, the other presets' answers the page
describes, and the library's refusal of a window that does not fit.
"""

import pytest
from fastapi.testclient import TestClient
from oop_ml import DenseLayer, Identity, Neuron

from oop_ml_api.main import create_app

client = TestClient(create_app())

APPLY = "/concepts/convolution/apply"
PARAMETER_COUNT = "/concepts/convolution/parameter-count"

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

SQUARE_SHIFTED_RIGHT = [row[-1:] + row[:-1] for row in SQUARE]

DIAGONAL = [[1 if row == column else 0 for column in range(8)] for row in range(8)]

VERTICAL_EDGE = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
HORIZONTAL_EDGE = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
BLUR = [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]

SQUARE_UNDER_VERTICAL_EDGE = [
    [1, 1, 0, 0, -1, -1],
    [3, 3, 0, 0, -3, -3],
    [4, 4, 0, 0, -4, -4],
    [4, 4, 0, 0, -4, -4],
    [3, 3, 0, 0, -3, -3],
    [1, 1, 0, 0, -1, -1],
]


def apply(
    picture: list[list[int]],
    kernel: list[list[float]] = VERTICAL_EDGE,
    **settings: int,
) -> dict:
    response = client.post(
        APPLY, json={"picture": picture, "kernel": kernel, **settings}
    )
    assert response.status_code == 200
    return response.json()


class TestTheWorkedCell:
    def test_answer_row_two_column_zero_is_four(self) -> None:
        body = apply(SQUARE)

        assert body["output"][2][0] == 4.0

    def test_its_window_sits_on_the_square_left_edge(self) -> None:
        window = apply(SQUARE)["windows"][2][0]

        assert window["top"] == 2
        assert window["left"] == 0
        assert window["patch"] == [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

    def test_its_nine_products_are_one_two_one_down_the_right_column(self) -> None:
        window = apply(SQUARE)["windows"][2][0]

        assert window["products"] == [[0, 0, 1], [0, 0, 2], [0, 0, 1]]

    def test_one_cell_to_the_right_still_answers_four(self) -> None:
        body = apply(SQUARE)

        assert body["output"][2][1] == 4.0

    def test_inside_the_square_the_two_columns_cancel_to_zero(self) -> None:
        body = apply(SQUARE)

        assert body["output"][2][2] == 0.0
        assert body["output"][2][3] == 0.0

    def test_the_square_right_edge_answers_minus_four(self) -> None:
        body = apply(SQUARE)

        assert body["output"][2][4] == -4.0

    def test_the_whole_of_row_two(self) -> None:
        body = apply(SQUARE)

        assert body["output"][2] == [4, 4, 0, 0, -4, -4]

    def test_the_whole_map_is_the_hand_worked_one(self) -> None:
        body = apply(SQUARE)

        assert body["output"] == SQUARE_UNDER_VERTICAL_EDGE
        assert body["output_height"] == 6
        assert body["output_width"] == 6


class TestTheOtherPresets:
    def test_the_horizontal_edge_kernel_finds_the_top_and_bottom_instead(self) -> None:
        output = apply(SQUARE, HORIZONTAL_EDGE)["output"]

        assert output[0] == [1, 3, 4, 4, 3, 1]
        assert output[2] == [0] * 6
        assert output[4] == [-1, -3, -4, -4, -3, -1]

    def test_the_blur_kernel_answers_one_inside_and_a_ninth_at_a_corner(self) -> None:
        output = apply(SQUARE, BLUR)["output"]

        assert output[2][2] == pytest.approx(1.0)
        assert output[0][0] == pytest.approx(1 / 9)

    def test_a_diagonal_is_found_by_both_edge_kernels_at_half_strength(self) -> None:
        vertical = apply(DIAGONAL, VERTICAL_EDGE)["output"]
        horizontal = apply(DIAGONAL, HORIZONTAL_EDGE)["output"]

        assert max(abs(value) for row in vertical for value in row) == 2.0
        assert max(abs(value) for row in horizontal for value in row) == 2.0


class TestEveryWindowAgreesWithTheLibrary:
    @pytest.mark.parametrize(("stride", "padding"), [(1, 0), (1, 1), (2, 0), (2, 1)])
    def test_the_products_sum_to_the_answer_cell(
        self, stride: int, padding: int
    ) -> None:
        body = apply(SQUARE, stride=stride, padding=padding)

        for output_row, row_of_windows in enumerate(body["windows"]):
            for output_column, window in enumerate(row_of_windows):
                summed = sum(sum(row) for row in window["products"])
                assert summed == pytest.approx(
                    body["output"][output_row][output_column], abs=1e-12
                )

    def test_the_windows_are_laid_out_like_the_answer(self) -> None:
        body = apply(SQUARE)

        assert len(body["windows"]) == body["output_height"]
        assert all(len(row) == body["output_width"] for row in body["windows"])


class TestStrideAndPadding:
    def test_padding_keeps_the_answer_the_size_of_the_picture(self) -> None:
        body = apply(SQUARE, padding=1)

        assert body["output_height"] == 8
        assert body["output_width"] == 8
        assert body["output"][0] == [0] * 8
        assert body["output"][3] == [0, 4, 4, 0, 0, -4, -4, 0]

    def test_a_padded_corner_window_begins_on_the_border(self) -> None:
        window = apply(SQUARE, padding=1)["windows"][0][0]

        assert window["top"] == -1
        assert window["left"] == -1
        assert window["patch"] == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    def test_a_stride_of_two_answers_three_by_three(self) -> None:
        body = apply(SQUARE, stride=2)

        assert body["output_height"] == 3
        assert body["output_width"] == 3
        assert body["output"] == [[1, 0, -1], [4, 0, -4], [3, 0, -3]]

    def test_a_stride_of_two_reads_every_other_window(self) -> None:
        windows = apply(SQUARE, stride=2)["windows"]

        assert [window["left"] for window in windows[0]] == [0, 2, 4]
        assert [row[0]["top"] for row in windows] == [0, 2, 4]

    def test_stride_two_with_padding_answers_four_by_four(self) -> None:
        body = apply(SQUARE, stride=2, padding=1)

        assert body["output"] == [
            [0, 0, 0, 0],
            [0, 3, 0, -3],
            [0, 4, 0, -4],
            [0, 1, 0, -1],
        ]


class TestShiftingThePictureShiftsTheAnswer:
    def test_the_map_moves_one_column_with_the_square(self) -> None:
        original = apply(SQUARE)["output"]
        shifted = apply(SQUARE_SHIFTED_RIGHT)["output"]

        for original_row, shifted_row in zip(original, shifted, strict=True):
            assert shifted_row[1:] == original_row[:-1]

    def test_the_worked_cell_moves_with_it(self) -> None:
        shifted = apply(SQUARE_SHIFTED_RIGHT)["output"]

        assert shifted[2][1] == 4.0
        assert shifted[2][0] == 0.0


class TestTheParameterCount:
    def test_one_filter_over_the_page_picture_costs_ten(self) -> None:
        parameters = apply(SQUARE)["parameters"]

        assert parameters["reads"] == [1, 8, 8]
        assert parameters["answers"] == [1, 6, 6]
        assert parameters["n_inputs"] == 64
        assert parameters["n_outputs"] == 36
        assert parameters["convolution"] == 10

    def test_the_window_without_the_sharing_costs_three_hundred_and_sixty(
        self,
    ) -> None:
        parameters = apply(SQUARE)["parameters"]

        assert parameters["unshared"] == 360

    def test_the_dense_layer_of_equal_width_costs_two_thousand_three_hundred_and_forty(
        self,
    ) -> None:
        parameters = apply(SQUARE)["parameters"]

        assert parameters["dense"] == 2340
        assert parameters["ratio"] == 234.0

    def test_the_dense_count_is_what_a_real_dense_layer_holds(self) -> None:
        # Thirty-six neurons, one per answer cell, each reading all sixty-four
        # pixels, built through the library rather than multiplied out.
        dense = DenseLayer(
            [
                Neuron(weights=[1.0] * 64, bias=0.0, activation=Identity())
                for output_cell in range(36)
            ]
        )
        held = dense.weight_matrix.size + dense.bias_vector.size

        assert held == apply(SQUARE)["parameters"]["dense"]

    def test_padding_widens_the_dense_layer_but_not_the_convolution(self) -> None:
        parameters = apply(SQUARE, padding=1)["parameters"]

        assert parameters["convolution"] == 10
        assert parameters["dense"] == 4160
        assert parameters["ratio"] == 416.0

    def test_the_library_twenty_eight_by_twenty_eight_example_is_fifty_three_thousand_and_sixty_six_times(
        self,
    ) -> None:
        response = client.post(
            PARAMETER_COUNT,
            json={"channels": 1, "side": 28, "n_filters": 8, "kernel_size": 3},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["reads"] == [1, 28, 28]
        assert body["answers"] == [8, 26, 26]
        assert body["n_inputs"] == 784
        assert body["n_outputs"] == 5408
        assert body["convolution"] == 80
        assert body["unshared"] == 54080
        assert body["dense"] == 4_245_280
        assert body["ratio"] == 53066.0

    def test_a_colour_picture_costs_each_filter_twenty_seven_weights(self) -> None:
        response = client.post(
            PARAMETER_COUNT,
            json={"channels": 3, "side": 8, "n_filters": 1, "kernel_size": 3},
        )

        assert response.json()["convolution"] == 28

    def test_the_page_picture_costed_through_the_counting_endpoint_agrees(
        self,
    ) -> None:
        response = client.post(
            PARAMETER_COUNT, json={"side": 8, "n_filters": 1, "kernel_size": 3}
        )

        assert response.json() == apply(SQUARE)["parameters"]


class TestRefusals:
    def test_a_window_wider_than_the_picture_is_refused_in_the_library_words(
        self,
    ) -> None:
        response = client.post(
            PARAMETER_COUNT, json={"side": 4, "n_filters": 1, "kernel_size": 7}
        )

        assert response.status_code == 422
        assert response.json()["error"] == "ShapeMismatchError"
        assert "(4 - 7 + 2 * 0) // 1 + 1 = -2" in response.json()["detail"]
        assert "does not fit" in response.json()["detail"]

    def test_a_pixel_outside_the_unit_range_is_refused(self) -> None:
        too_bright = [row[:] for row in SQUARE]
        too_bright[0][0] = 2

        response = client.post(
            APPLY, json={"picture": too_bright, "kernel": VERTICAL_EDGE}
        )

        assert response.status_code == 422

    def test_a_kernel_weight_past_ten_is_refused(self) -> None:
        heavy = [[0, 0, 0], [0, 11, 0], [0, 0, 0]]

        response = client.post(APPLY, json={"picture": SQUARE, "kernel": heavy})

        assert response.status_code == 422

    def test_a_picture_of_the_wrong_size_is_refused(self) -> None:
        response = client.post(
            APPLY, json={"picture": SQUARE[:7], "kernel": VERTICAL_EDGE}
        )

        assert response.status_code == 422

    def test_a_stride_of_three_is_refused(self) -> None:
        response = client.post(
            APPLY, json={"picture": SQUARE, "kernel": VERTICAL_EDGE, "stride": 3}
        )

        assert response.status_code == 422

    def test_an_unknown_setting_is_refused(self) -> None:
        response = client.post(
            APPLY,
            json={"picture": SQUARE, "kernel": VERTICAL_EDGE, "dilation": 2},
        )

        assert response.status_code == 422
