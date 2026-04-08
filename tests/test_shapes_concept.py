"""The stack check endpoint, pinned to the shape page's hand arithmetic.

The page works a picture of (1, 8, 8) through a convolution of four kernels
of three by three to (4, 6, 6), a pooling layer to (4, 3, 3), a flatten to
(36,) and a dense layer to (10,), every extent by hand, and then removes the
flatten and quotes the library's refusal. Its page-specific section quotes the
(8, 26, 26) against (5408,) case in every form the widget offers, the counts
agreeing, the counts disagreeing, and two rearrangements of the same 5408
numbers. Every extent, every count and every message on the page is pinned
here.
"""

from typing import ClassVar

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

CHECK = "/concepts/shapes/check"

CONV_ON_EIGHT = {"kind": "conv", "reads": [1, 8, 8], "n_filters": 4, "kernel_size": 3}
POOL_TWO = {"kind": "pool", "window": 2, "stride": 2}
FLATTEN = {"kind": "flatten"}
DENSE_TEN = {"kind": "dense", "n_neurons": 10}

WORKED_STACK = [CONV_ON_EIGHT, POOL_TWO, FLATTEN, DENSE_TEN]

CONV_ON_TWENTY_EIGHT = {
    "kind": "conv",
    "reads": [1, 28, 28],
    "n_filters": 8,
    "kernel_size": 3,
}

ARRANGEMENT_MESSAGE = (
    "layer 1 answers with (4, 3, 3) and layer 2 reads (36,); both hold 36 "
    "numbers, so it is the arrangement that disagrees and not the width"
)

COUNTS_AGREE_MESSAGE = (
    "layer 0 answers with (8, 26, 26) and layer 1 reads (5408,); both hold 5408 "
    "numbers, so it is the arrangement that disagrees and not the width"
)

COUNTS_DISAGREE_MESSAGE = "layer 0 answers with (8, 26, 26) and layer 1 reads (784,)"

CHANNELS_MOVED_MESSAGE = (
    "layer 0 answers with (8, 26, 26) and layer 1 reads (26, 8, 26); both hold "
    "5408 numbers, so it is the arrangement that disagrees and not the width"
)

HALVED_AND_DOUBLED_MESSAGE = (
    "layer 0 answers with (8, 26, 26) and layer 1 reads (4, 52, 26); both hold "
    "5408 numbers, so it is the arrangement that disagrees and not the width"
)


def check(layers: list[dict]) -> dict:
    response = client.post(CHECK, json={"layers": layers})
    assert response.status_code == 200, response.json()
    return response.json()


class TestTheWorkedStack:
    """(1, 8, 8) to (10,) in four layers, every extent by hand."""

    def test_every_layer_reads_and_answers_the_hand_worked_extents(self) -> None:
        body = check(WORKED_STACK)

        arrangements = [(layer["reads"], layer["answers"]) for layer in body["layers"]]
        assert arrangements == [
            ([1, 8, 8], [4, 6, 6]),
            ([4, 6, 6], [4, 3, 3]),
            ([4, 3, 3], [36]),
            ([36], [10]),
        ]

    def test_every_layer_counts_the_hand_worked_numbers(self) -> None:
        body = check(WORKED_STACK)

        counts = [(layer["n_reads"], layer["n_answers"]) for layer in body["layers"]]
        assert counts == [(64, 144), (144, 36), (36, 36), (36, 10)]

    def test_every_join_holds_and_carries_no_message(self) -> None:
        body = check(WORKED_STACK)

        assert [join["verdict"] for join in body["joins"]] == ["holds"] * 3
        assert [join["position"] for join in body["joins"]] == [0, 1, 2]
        assert all(join["message"] is None for join in body["joins"])

    def test_the_stack_exists_and_reports_its_two_ends_as_arrangements(self) -> None:
        body = check(WORKED_STACK)

        assert body["holds"] is True
        assert body["reads"] == [1, 8, 8]
        assert body["answers"] == [10]
        assert body["refusal"] is None
        assert all(layer["refusal"] is None for layer in body["layers"])

    def test_the_layers_reordered_still_stack_and_the_picture_shrinks_first(
        self,
    ) -> None:
        body = check(
            [
                {"kind": "pool", "reads": [1, 8, 8], "window": 2, "stride": 2},
                {"kind": "conv", "n_filters": 4, "kernel_size": 3},
                FLATTEN,
                DENSE_TEN,
            ]
        )

        answers = [layer["answers"] for layer in body["layers"]]
        assert answers == [[1, 4, 4], [4, 2, 2], [16], [10]]
        assert body["holds"] is True


class TestTheMissingFlatten:
    """Remove the bridge and the join turns red with the arrangement message."""

    WITHOUT_FLATTEN: ClassVar[list[dict]] = [CONV_ON_EIGHT, POOL_TWO, DENSE_TEN]

    def test_the_dense_layer_reads_as_many_numbers_as_arrive(self) -> None:
        body = check(self.WITHOUT_FLATTEN)

        dense = body["layers"][2]
        assert dense["reads"] == [36]
        assert dense["answers"] == [10]
        assert dense["n_reads"] == 36
        assert dense["refusal"] is None

    def test_the_join_fails_with_the_library_arrangement_message(self) -> None:
        body = check(self.WITHOUT_FLATTEN)

        assert [join["verdict"] for join in body["joins"]] == ["holds", "fails"]
        assert body["joins"][0]["message"] is None
        assert body["joins"][1]["message"] == ARRANGEMENT_MESSAGE

    def test_the_stack_is_refused_in_the_same_words(self) -> None:
        body = check(self.WITHOUT_FLATTEN)

        assert body["holds"] is False
        assert body["reads"] is None
        assert body["answers"] is None
        assert body["refusal"] == ARRANGEMENT_MESSAGE


class TestCountsAgreeAndArrangementsDoNot:
    """The (8, 26, 26) against (5408,) case, in every form the page shows."""

    def test_the_convolution_answers_five_thousand_four_hundred_and_eight(self) -> None:
        body = check([CONV_ON_TWENTY_EIGHT, DENSE_TEN])

        convolution = body["layers"][0]
        assert convolution["reads"] == [1, 28, 28]
        assert convolution["answers"] == [8, 26, 26]
        assert convolution["n_reads"] == 784
        assert convolution["n_answers"] == 5408

    def test_a_dense_layer_told_nothing_reads_the_count_and_is_refused(self) -> None:
        body = check([CONV_ON_TWENTY_EIGHT, DENSE_TEN])

        assert body["layers"][1]["reads"] == [5408]
        assert body["layers"][1]["n_reads"] == 5408
        assert body["joins"][0]["verdict"] == "fails"
        assert body["joins"][0]["message"] == COUNTS_AGREE_MESSAGE
        assert body["refusal"] == COUNTS_AGREE_MESSAGE

    def test_a_dense_layer_reading_the_picture_width_gets_the_shorter_message(
        self,
    ) -> None:
        body = check(
            [CONV_ON_TWENTY_EIGHT, {"kind": "dense", "reads": [784], "n_neurons": 10}]
        )

        assert body["layers"][1]["reads"] == [784]
        assert body["joins"][0]["verdict"] == "fails"
        assert body["joins"][0]["message"] == COUNTS_DISAGREE_MESSAGE
        assert "both hold" not in body["refusal"]

    def test_the_same_count_with_the_channels_moved_is_refused(self) -> None:
        body = check([CONV_ON_TWENTY_EIGHT, {"kind": "flatten", "reads": [26, 8, 26]}])

        assert body["layers"][1]["n_reads"] == 5408
        assert body["joins"][0]["verdict"] == "fails"
        assert body["joins"][0]["message"] == CHANNELS_MOVED_MESSAGE

    def test_the_same_count_halved_and_doubled_is_refused(self) -> None:
        body = check([CONV_ON_TWENTY_EIGHT, {"kind": "flatten", "reads": [4, 52, 26]}])

        assert body["layers"][1]["n_reads"] == 5408
        assert body["joins"][0]["verdict"] == "fails"
        assert body["joins"][0]["message"] == HALVED_AND_DOUBLED_MESSAGE

    def test_a_flatten_reading_the_arrangement_holds(self) -> None:
        body = check([CONV_ON_TWENTY_EIGHT, {"kind": "flatten", "reads": [8, 26, 26]}])

        assert body["layers"][1]["answers"] == [5408]
        assert body["joins"][0]["verdict"] == "holds"
        assert body["holds"] is True
        assert body["answers"] == [5408]

    def test_a_flatten_between_them_makes_every_join_hold(self) -> None:
        body = check([CONV_ON_TWENTY_EIGHT, FLATTEN, DENSE_TEN])

        assert body["layers"][1]["reads"] == [8, 26, 26]
        assert body["layers"][1]["answers"] == [5408]
        assert body["layers"][2]["reads"] == [5408]
        assert [join["verdict"] for join in body["joins"]] == ["holds", "holds"]
        assert body["holds"] is True
        assert body["reads"] == [1, 28, 28]
        assert body["answers"] == [10]


class TestTheDerivation:
    """Each layer kind's output extents as the integer arithmetic the page states."""

    def test_padding_and_stride_enter_the_convolution_sum(self) -> None:
        # (8 - 3 + 2 * 1) // 2 + 1 = 4 along both axes.
        body = check(
            [
                {
                    "kind": "conv",
                    "reads": [1, 8, 8],
                    "n_filters": 4,
                    "kernel_size": 3,
                    "stride": 2,
                    "padding": 1,
                }
            ]
        )

        assert body["layers"][0]["answers"] == [4, 4, 4]
        assert body["layers"][0]["n_answers"] == 64

    def test_a_pooling_window_of_three_at_stride_two_reaches_twice(self) -> None:
        # (6 - 3) // 2 + 1 = 2, and the average kind pools by the same sum.
        body = check(
            [
                {
                    "kind": "pool",
                    "reads": [4, 6, 6],
                    "summary": "average",
                    "window": 3,
                    "stride": 2,
                }
            ]
        )

        assert body["layers"][0]["answers"] == [4, 2, 2]
        assert body["layers"][0]["n_answers"] == 16

    def test_a_flatten_answers_the_product_of_its_extents(self) -> None:
        body = check([{"kind": "flatten", "reads": [4, 3, 3]}])

        assert body["layers"][0]["answers"] == [36]
        assert body["layers"][0]["n_reads"] == 36
        assert body["layers"][0]["n_answers"] == 36

    def test_a_flatten_over_a_row_is_the_identity(self) -> None:
        body = check([{"kind": "flatten", "reads": [36]}, DENSE_TEN])

        assert body["layers"][0]["answers"] == [36]
        assert body["joins"][0]["verdict"] == "holds"
        assert body["reads"] == [36]

    def test_a_dense_layer_answers_its_neuron_count_whatever_it_reads(self) -> None:
        body = check(
            [
                {"kind": "dense", "reads": [36], "n_neurons": 10},
                {"kind": "dense", "n_neurons": 3},
            ]
        )

        arrangements = [(layer["reads"], layer["answers"]) for layer in body["layers"]]
        assert arrangements == [([36], [10]), ([10], [3])]
        assert body["holds"] is True


class TestLayersTheLibraryWillNotBuild:
    """A refusal from a constructor, and what it leaves unsettled above it."""

    TOO_WIDE_A_WINDOW: ClassVar[list[dict]] = [
        CONV_ON_EIGHT,
        POOL_TWO,
        {"kind": "pool", "window": 4, "stride": 2},
        FLATTEN,
        DENSE_TEN,
    ]

    def test_a_window_wider_than_the_picture_is_refused_in_the_library_words(
        self,
    ) -> None:
        body = check(self.TOO_WIDE_A_WINDOW)

        refused = body["layers"][2]
        assert refused["reads"] == [4, 3, 3]
        assert refused["answers"] is None
        assert refused["refusal"] == "a window of 4 does not fit in a picture 3 by 3"
        assert body["refusal"] == refused["refusal"]
        assert body["holds"] is False

    def test_the_layers_above_a_refused_one_are_left_unsettled(self) -> None:
        body = check(self.TOO_WIDE_A_WINDOW)

        for layer in body["layers"][3:]:
            assert layer["reads"] is None
            assert layer["answers"] is None
            assert layer["refusal"] is None
        assert [join["verdict"] for join in body["joins"]] == [
            "holds",
            "unsettled",
            "unsettled",
            "unsettled",
        ]

    def test_a_convolution_handed_a_row_is_refused_in_the_library_words(self) -> None:
        body = check(
            [
                {"kind": "dense", "reads": [36], "n_neurons": 10},
                {"kind": "conv", "n_filters": 4, "kernel_size": 3},
            ]
        )

        assert body["layers"][1]["reads"] == [10]
        assert (
            body["layers"][1]["refusal"]
            == "a convolution reads (channels, height, width), got 1 extents"
        )
        assert body["joins"][0]["verdict"] == "unsettled"

    def test_a_pooling_layer_handed_a_row_is_refused_in_the_library_words(
        self,
    ) -> None:
        body = check([{"kind": "flatten", "reads": [36]}, {"kind": "pool"}])

        assert (
            body["layers"][1]["refusal"]
            == "a pooling layer reads (channels, height, width), got 1 extents"
        )
        assert body["holds"] is False

    def test_a_kernel_wider_than_the_picture_spells_out_the_sum(self) -> None:
        body = check(
            [{"kind": "conv", "reads": [1, 8, 8], "n_filters": 4, "kernel_size": 9}]
        )

        assert body["layers"][0]["refusal"] == (
            "this convolution's answer would have height (8 - 9 + 2 * 0) // 1 + 1 "
            "= 0, so the window does not fit over what it reads"
        )
        assert body["joins"] == []


class TestOnlyTheFirstFailingJoinCarriesAMessage:
    def test_the_library_stops_at_the_first_bad_join(self) -> None:
        body = check(
            [
                CONV_ON_EIGHT,
                POOL_TWO,
                DENSE_TEN,
                {"kind": "dense", "reads": [12], "n_neurons": 3},
            ]
        )

        verdicts = [
            (join["verdict"], join["message"] is not None) for join in body["joins"]
        ]
        assert verdicts == [("holds", False), ("fails", True), ("fails", False)]
        assert body["refusal"] == ARRANGEMENT_MESSAGE

    def test_a_refused_layer_beneath_a_bad_join_is_what_gets_reported(self) -> None:
        body = check(
            [
                {"kind": "conv", "reads": [1, 8, 8], "n_filters": 4, "kernel_size": 9},
                {"kind": "dense", "reads": [12], "n_neurons": 3},
                {"kind": "dense", "reads": [5], "n_neurons": 3},
            ]
        )

        assert body["layers"][0]["refusal"] is not None
        assert body["refusal"] == body["layers"][0]["refusal"]
        assert [join["verdict"] for join in body["joins"]] == ["unsettled", "fails"]
        assert body["joins"][1]["message"] is None


class TestRefusals:
    def test_a_first_layer_that_states_nothing_is_refused_in_words(self) -> None:
        response = client.post(
            CHECK, json={"layers": [{"kind": "conv", "n_filters": 4, "kernel_size": 3}]}
        )

        assert response.status_code == 422
        assert "nothing beneath it" in response.json()["detail"]

    def test_a_dense_layer_told_a_picture_is_refused_in_words(self) -> None:
        response = client.post(
            CHECK,
            json={"layers": [{"kind": "dense", "reads": [4, 3, 3], "n_neurons": 10}]},
        )

        assert response.status_code == 422
        assert "one extent, not (4, 3, 3)" in response.json()["detail"]

    def test_a_dense_layer_handed_a_row_over_the_bound_is_refused_in_words(
        self,
    ) -> None:
        response = client.post(
            CHECK,
            json={"layers": [{"kind": "flatten", "reads": [16, 32, 33]}, DENSE_TEN]},
        )

        assert response.status_code == 422
        assert "at most 16384 numbers, and 16896 arrive" in response.json()["detail"]

    def test_a_dense_layer_handed_a_row_at_the_bound_is_built(self) -> None:
        body = check([{"kind": "flatten", "reads": [16, 32, 32]}, DENSE_TEN])

        assert body["layers"][1]["reads"] == [16384]
        assert body["holds"] is True

    def test_a_layer_kind_outside_the_closed_set_is_refused(self) -> None:
        response = client.post(CHECK, json={"layers": [{"kind": "dropout"}]})

        assert response.status_code == 422

    def test_a_field_the_layer_does_not_have_is_refused(self) -> None:
        response = client.post(
            CHECK, json={"layers": [{"kind": "flatten", "reads": [36], "rate": 0.5}]}
        )

        assert response.status_code == 422

    def test_an_empty_stack_is_refused(self) -> None:
        response = client.post(CHECK, json={"layers": []})

        assert response.status_code == 422

    def test_too_many_layers_are_refused(self) -> None:
        response = client.post(CHECK, json={"layers": [FLATTEN] * 9})

        assert response.status_code == 422

    def test_a_picture_extent_over_the_bound_is_refused(self) -> None:
        response = client.post(
            CHECK,
            json={
                "layers": [
                    {
                        "kind": "conv",
                        "reads": [1, 64, 64],
                        "n_filters": 4,
                        "kernel_size": 3,
                    }
                ]
            },
        )

        assert response.status_code == 422

    def test_more_kernels_than_the_bound_are_refused(self) -> None:
        response = client.post(
            CHECK,
            json={
                "layers": [
                    {
                        "kind": "conv",
                        "reads": [1, 8, 8],
                        "n_filters": 17,
                        "kernel_size": 3,
                    }
                ]
            },
        )

        assert response.status_code == 422
