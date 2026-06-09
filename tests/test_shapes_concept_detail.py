"""The four narrower shape endpoints, and the digit chain the page reads across.

``test_shapes_concept.py`` pins the stack check over the small hand-worked
chain. This file pins what the page's other widgets quote: the deep chain over
a twenty-eight by twenty-eight picture, one bare shape with no layer around it,
one seam asked on its own, and the two weaknesses the library records rather
than defends. Every number the page prints in prose is asserted here.
"""

from typing import ClassVar

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

CHECK = "/concepts/shapes/check"
EXTENTS = "/concepts/shapes/extents"
JOIN = "/concepts/shapes/join"
TIED = "/concepts/shapes/tied-weights"
WHOLE = "/concepts/shapes/whole-numbers"

DIGIT_PICTURE = [1, 28, 28]

DEEP_CHAIN: list[dict] = [
    {"kind": "conv", "reads": DIGIT_PICTURE, "n_filters": 8, "kernel_size": 3},
    {"kind": "pool", "window": 2, "stride": 2},
    {"kind": "conv", "n_filters": 16, "kernel_size": 3},
    {"kind": "pool", "window": 2, "stride": 2},
    {"kind": "flatten"},
    {"kind": "dense", "n_neurons": 32},
    {"kind": "dense", "n_neurons": 10},
]

DEEP_CHAIN_WITHOUT_THE_BRIDGE: list[dict] = [
    layer for layer in DEEP_CHAIN if layer["kind"] != "flatten"
]

DEEP_BRIDGE_MESSAGE = (
    "layer 3 answers with (16, 5, 5) and layer 4 reads (400,); both hold 400 "
    "numbers, so it is the arrangement that disagrees and not the width"
)

SHARED_LAYER: dict = {
    "weights": [[1.0, 0.5], [-0.5, 2.0]],
    "biases": [0.0, 0.0],
    "rows": [[1.0, 2.0], [3.0, -1.0]],
    "targets": [[0.0, 0.0], [1.0, 1.0]],
}


def check(layers: list[dict]) -> dict:
    response = client.post(CHECK, json={"layers": layers})
    assert response.status_code == 200, response.json()
    return response.json()


def shape(reads: list[int], answers: list[int]) -> dict:
    response = client.post(EXTENTS, json={"reads": reads, "answers": answers})
    assert response.status_code == 200, response.json()
    return response.json()


def seam(beneath: list[int], above: list[int]) -> dict:
    response = client.post(JOIN, json={"beneath": beneath, "above": above})
    assert response.status_code == 200, response.json()
    return response.json()


class TestTheDigitChain:
    """Two convolutions, two pooling layers, a bridge and two dense layers."""

    def test_every_step_answers_the_extents_the_page_prints(self) -> None:
        body = check(DEEP_CHAIN)

        assert [layer["answers"] for layer in body["layers"]] == [
            [8, 26, 26],
            [8, 13, 13],
            [16, 11, 11],
            [16, 5, 5],
            [400],
            [32],
            [10],
        ]

    def test_every_step_counts_the_numbers_the_page_prints(self) -> None:
        body = check(DEEP_CHAIN)

        assert body["layers"][0]["n_reads"] == 784
        assert [layer["n_answers"] for layer in body["layers"]] == [
            5408,
            1352,
            1936,
            400,
            400,
            32,
            10,
        ]

    def test_the_bridge_is_the_one_step_that_changes_no_count(self) -> None:
        body = check(DEEP_CHAIN)

        unchanged = [
            layer["kind"]
            for layer in body["layers"]
            if layer["n_reads"] == layer["n_answers"]
        ]
        assert unchanged == ["flatten"]

    def test_the_chain_reads_the_picture_and_answers_ten_numbers(self) -> None:
        body = check(DEEP_CHAIN)

        assert body["holds"] is True
        assert body["reads"] == DIGIT_PICTURE
        assert body["answers"] == [10]
        assert [join["verdict"] for join in body["joins"]] == ["holds"] * 6

    def test_the_chain_without_the_bridge_fails_at_the_fourth_seam(self) -> None:
        body = check(DEEP_CHAIN_WITHOUT_THE_BRIDGE)

        assert body["holds"] is False
        assert [join["verdict"] for join in body["joins"]] == [
            "holds",
            "holds",
            "holds",
            "fails",
            "holds",
        ]
        assert body["joins"][3]["message"] == DEEP_BRIDGE_MESSAGE
        assert body["refusal"] == DEEP_BRIDGE_MESSAGE

    def test_the_dense_layer_reading_the_arriving_count_was_built_happily(self) -> None:
        body = check(DEEP_CHAIN_WITHOUT_THE_BRIDGE)

        dense = body["layers"][4]
        assert dense["reads"] == [400]
        assert dense["answers"] == [32]
        assert dense["refusal"] is None


class TestAShapeOnItsOwn:
    """The one route that reaches the object owning the extent rule."""

    def test_a_picture_reports_its_arrangement_and_its_count_apart(self) -> None:
        body = shape([1, 28, 28], [8, 26, 26])

        assert body["reads"] == [1, 28, 28]
        assert body["n_reads"] == 784
        assert body["answers"] == [8, 26, 26]
        assert body["n_answers"] == 5408
        assert body["refusal"] is None

    def test_eight_numbers_arranged_are_not_a_row_of_eight(self) -> None:
        arranged = shape([2, 2, 2], [8])

        assert arranged["reads"] == [2, 2, 2]
        assert arranged["n_reads"] == 8
        assert arranged["n_answers"] == 8

    def test_a_height_of_zero_is_refused_in_the_library_words(self) -> None:
        body = shape([1, 0, 28], [10])

        assert body["reads"] is None
        assert body["n_reads"] is None
        assert body["refusal"] == "n_inputs extents must be at least 1, got 0"

    def test_a_negative_width_is_refused_in_the_library_words(self) -> None:
        body = shape([1, 28, -1], [10])

        assert body["refusal"] == "n_inputs extents must be at least 1, got -1"

    def test_a_reading_side_with_no_extents_is_refused(self) -> None:
        body = shape([], [10])

        assert body["refusal"] == "n_inputs needs at least one extent"

    def test_an_answering_side_with_no_extents_is_refused_by_name(self) -> None:
        body = shape([8], [])

        assert body["refusal"] == "n_outputs needs at least one extent"

    def test_a_side_wider_than_the_bound_is_refused_by_the_request(self) -> None:
        response = client.post(EXTENTS, json={"reads": [1, 2, 3, 4, 5], "answers": [1]})

        assert response.status_code == 422

    def test_a_field_the_request_does_not_have_is_refused(self) -> None:
        response = client.post(
            EXTENTS, json={"reads": [8], "answers": [4], "n_reads": 8}
        )

        assert response.status_code == 422


class TestOneSeamOnItsOwn:
    """The counts and the verdict reported side by side, which is the subject."""

    def test_the_recorded_case_agrees_on_the_count_and_is_still_refused(self) -> None:
        body = seam([8, 26, 26], [5408])

        assert body["n_beneath"] == 5408
        assert body["n_above"] == 5408
        assert body["counts_agree"] is True
        assert body["holds"] is False
        assert body["refusal"] == (
            "layer 0 answers with (8, 26, 26) and layer 1 reads (5408,); both "
            "hold 5408 numbers, so it is the arrangement that disagrees and not "
            "the width"
        )

    def test_the_bridge_reading_the_arrangement_holds(self) -> None:
        body = seam([8, 26, 26], [8, 26, 26])

        assert body["counts_agree"] is True
        assert body["holds"] is True
        assert body["refusal"] is None

    def test_the_same_count_with_the_channels_moved_is_refused(self) -> None:
        body = seam([8, 26, 26], [26, 8, 26])

        assert body["counts_agree"] is True
        assert body["holds"] is False
        assert "(26, 8, 26)" in body["refusal"]

    def test_the_same_count_halved_and_doubled_is_refused(self) -> None:
        body = seam([8, 26, 26], [4, 52, 26])

        assert body["counts_agree"] is True
        assert body["holds"] is False
        assert "(4, 52, 26)" in body["refusal"]

    def test_two_sides_agreeing_on_neither_get_the_shorter_message(self) -> None:
        body = seam([36], [10])

        assert body["counts_agree"] is False
        assert body["holds"] is False
        assert body["refusal"] == ("layer 0 answers with (36,) and layer 1 reads (10,)")
        assert "both hold" not in body["refusal"]

    def test_two_rows_of_one_width_hold(self) -> None:
        body = seam([36], [36])

        assert body["holds"] is True
        assert body["refusal"] is None

    def test_a_picture_against_the_row_it_flattens_to_is_refused(self) -> None:
        body = seam([4, 3, 3], [36])

        assert body["counts_agree"] is True
        assert body["holds"] is False

    def test_a_seam_of_two_extents_is_refused_in_words(self) -> None:
        response = client.post(JOIN, json={"beneath": [4, 6], "above": [24]})

        assert response.status_code == 422
        assert "no layer answers with two extents" in response.json()["detail"]

    def test_more_channels_than_the_bound_are_refused_in_words(self) -> None:
        response = client.post(JOIN, json={"beneath": [17, 4, 4], "above": [272]})

        assert response.status_code == 422
        assert "at most 16 channels" in response.json()["detail"]

    def test_a_picture_larger_than_the_bound_is_refused_in_words(self) -> None:
        response = client.post(JOIN, json={"beneath": [1, 64, 64], "above": [4096]})

        assert response.status_code == 422
        assert "at most 32 by 32" in response.json()["detail"]

    def test_an_empty_side_is_refused_by_the_request(self) -> None:
        response = client.post(JOIN, json={"beneath": [], "above": [8]})

        assert response.status_code == 422


class TestAChainReportsArrangementsRatherThanCounts:
    """The fixture that can tell the two facts apart, and the one that cannot."""

    ARRANGED: ClassVar[list[dict]] = [
        {"kind": "flatten", "reads": [2, 2, 2]},
        {"kind": "dense", "n_neurons": 3},
    ]

    AS_A_ROW: ClassVar[list[dict]] = [
        {"kind": "flatten", "reads": [8]},
        {"kind": "dense", "n_neurons": 3},
    ]

    ROWS_ONLY: ClassVar[list[dict]] = [
        {"kind": "dense", "reads": [8], "n_neurons": 4},
        {"kind": "dense", "n_neurons": 3},
    ]

    def test_a_chain_reading_an_arrangement_says_so_rather_than_its_count(
        self,
    ) -> None:
        body = check(self.ARRANGED)

        assert body["holds"] is True
        assert body["reads"] == [2, 2, 2]
        assert body["layers"][0]["n_reads"] == 8

    def test_the_same_eight_numbers_as_a_row_are_a_different_report(self) -> None:
        arranged = check(self.ARRANGED)
        as_a_row = check(self.AS_A_ROW)

        assert arranged["layers"][0]["n_reads"] == as_a_row["layers"][0]["n_reads"]
        assert arranged["reads"] != as_a_row["reads"]

    def test_a_chain_of_rows_alone_cannot_tell_the_two_facts_apart(self) -> None:
        body = check(self.ROWS_ONLY)

        assert body["reads"] == [8]
        assert body["layers"][0]["n_reads"] == 8

    def test_a_stack_of_one_layer_holds_and_has_no_seams(self) -> None:
        body = check([{"kind": "dense", "reads": [36], "n_neurons": 10}])

        assert body["holds"] is True
        assert body["joins"] == []
        assert body["reads"] == [36]
        assert body["answers"] == [10]

    def test_a_bridge_over_a_row_is_the_identity_and_still_holds(self) -> None:
        body = check(
            [{"kind": "flatten", "reads": [36]}, {"kind": "dense", "n_neurons": 4}]
        )

        assert body["layers"][0]["reads"] == [36]
        assert body["layers"][0]["answers"] == [36]
        assert body["joins"][0]["verdict"] == "holds"


class TestOneLayerAtTwoPositions:
    """The weakness the library records: a step unties what was one object."""

    def test_the_chain_really_does_hold_one_object_twice(self) -> None:
        response = client.post(TIED, json={**SHARED_LAYER, "learning_rate": 0.1})

        assert response.status_code == 200, response.json()
        assert response.json()["same_object_before"] is True

    def test_one_step_leaves_two_layers_where_there_was_one(self) -> None:
        body = client.post(TIED, json={**SHARED_LAYER, "learning_rate": 0.1}).json()

        assert body["same_object_after"] is False
        assert body["weights_before"] == [[1.0, 0.5], [-0.5, 2.0]]
        assert body["first_after"] != body["second_after"]

    def test_the_gap_and_the_loss_are_the_numbers_the_page_quotes(self) -> None:
        body = client.post(TIED, json={**SHARED_LAYER, "learning_rate": 0.1}).json()

        assert body["loss"] == 33.921875
        assert body["largest_gap"] == 1.5437500000000002

    def test_a_smaller_step_narrows_the_gap_without_closing_it(self) -> None:
        small = client.post(TIED, json={**SHARED_LAYER, "learning_rate": 0.001}).json()

        assert small["same_object_after"] is False
        assert small["largest_gap"] == 0.015437499999999993

    def test_nothing_was_raised_by_a_chain_with_a_layer_used_twice(self) -> None:
        response = client.post(TIED, json={**SHARED_LAYER, "learning_rate": 0.1})

        assert response.status_code == 200

    def test_a_layer_that_does_not_read_what_it_answers_is_refused_in_words(
        self,
    ) -> None:
        response = client.post(
            TIED,
            json={
                "weights": [[1.0, 0.5, 0.25], [-0.5, 2.0, 1.0]],
                "biases": [0.0, 0.0],
                "rows": [[1.0, 2.0]],
                "targets": [[0.0, 0.0]],
                "learning_rate": 0.1,
            },
        )

        assert response.status_code == 422
        assert "its weights are 2 by 2" in response.json()["detail"]

    def test_a_bias_per_neuron_is_required_in_words(self) -> None:
        response = client.post(
            TIED,
            json={
                "weights": [[1.0, 0.5], [-0.5, 2.0]],
                "biases": [0.0],
                "rows": [[1.0, 2.0]],
                "targets": [[0.0, 0.0]],
                "learning_rate": 0.1,
            },
        )

        assert response.status_code == 422
        assert "carries 2 biases, got 1" in response.json()["detail"]

    def test_one_target_per_row_is_required_in_words(self) -> None:
        response = client.post(
            TIED,
            json={
                **SHARED_LAYER,
                "targets": [[0.0, 0.0]],
                "learning_rate": 0.1,
            },
        )

        assert response.status_code == 422
        assert "one target per row" in response.json()["detail"]

    def test_a_rate_of_zero_is_refused_by_the_request(self) -> None:
        response = client.post(TIED, json={**SHARED_LAYER, "learning_rate": 0.0})

        assert response.status_code == 422


class TestWholeNumbersHandedToALayer:
    """The other recorded weakness: everything is coerced to floating point."""

    ASKED: ClassVar[list[str]] = [
        "0",
        "3",
        "-1",
        "9007199254740992",
        "9007199254740993",
    ]

    NOT_WHOLE: ClassVar[list[float]] = [2.9999999999999996, -1.5]

    def probe(self) -> dict:
        response = client.post(
            WHOLE,
            json={"whole_numbers": self.ASKED, "fractional_values": self.NOT_WHOLE},
        )
        assert response.status_code == 200, response.json()
        return response.json()

    def test_a_negative_position_goes_straight_through_unremarked(self) -> None:
        negative = self.probe()["whole_numbers"][2]

        assert negative["asked"] == "-1"
        assert negative["answered"] == "-1"
        assert negative["unchanged"] is True

    def test_a_value_that_is_not_whole_goes_through_unremarked(self) -> None:
        reports = self.probe()["fractional_values"]

        assert [report["answered"] for report in reports] == [
            "2.9999999999999996",
            "-1.5",
        ]
        assert all(report["whole"] is False for report in reports)
        assert all(report["unchanged"] is True for report in reports)

    def test_two_consecutive_whole_numbers_arrive_as_one(self) -> None:
        body = self.probe()

        assert body["whole_numbers"][3]["answered"] == "9007199254740992"
        assert body["whole_numbers"][4]["asked"] == "9007199254740993"
        assert body["whole_numbers"][4]["answered"] == "9007199254740992"
        assert body["whole_numbers"][4]["unchanged"] is False

    def test_five_distinct_numbers_come_back_as_four(self) -> None:
        body = self.probe()

        assert body["n_whole_asked"] == 5
        assert body["n_distinct_after"] == 4

    def test_nothing_objected_to_any_of_them(self) -> None:
        body = self.probe()

        assert body["refusal"] is None

    def test_the_smaller_whole_numbers_all_survive(self) -> None:
        body = self.probe()

        assert [report["unchanged"] for report in body["whole_numbers"][:4]] == [
            True,
            True,
            True,
            True,
        ]

    def test_a_number_past_the_bound_is_refused_by_the_request(self) -> None:
        response = client.post(WHOLE, json={"whole_numbers": [2**61]})

        assert response.status_code == 422

    def test_no_numbers_at_all_are_refused_by_the_request(self) -> None:
        response = client.post(WHOLE, json={"whole_numbers": []})

        assert response.status_code == 422

    def test_a_field_the_request_does_not_have_is_refused(self) -> None:
        response = client.post(WHOLE, json={"whole_numbers": [1], "indices": [1, 2, 3]})

        assert response.status_code == 422


class TestTheWindowArithmetic:
    """The sliding-window sum, every row of the page's table."""

    @staticmethod
    def convolution(kernel: int, stride: int, padding: int) -> dict:
        return check(
            [
                {
                    "kind": "conv",
                    "reads": [1, 8, 8],
                    "n_filters": 4,
                    "kernel_size": kernel,
                    "stride": stride,
                    "padding": padding,
                }
            ]
        )["layers"][0]

    def test_every_row_of_the_settings_table_answers_the_side_the_page_prints(
        self,
    ) -> None:
        settings = [
            (3, 1, 0, 6),
            (3, 1, 1, 8),
            (3, 2, 0, 3),
            (3, 2, 1, 4),
            (5, 1, 0, 4),
            (5, 1, 2, 8),
            (1, 1, 0, 8),
            (8, 1, 0, 1),
        ]

        for kernel, stride, padding, side in settings:
            layer = self.convolution(kernel, stride, padding)
            assert layer["answers"] == [4, side, side], (kernel, stride, padding)

    def test_the_stride_two_case_the_page_works_by_hand(self) -> None:
        layer = self.convolution(3, 2, 0)

        assert layer["answers"] == [4, 3, 3]
        assert layer["n_answers"] == 36

    def test_a_window_as_wide_as_the_side_is_built_and_answers_one(self) -> None:
        assert self.convolution(8, 1, 0)["answers"] == [4, 1, 1]

    def test_a_window_one_wider_is_refused_with_the_sum_written_out(self) -> None:
        assert self.convolution(9, 1, 0)["refusal"] == (
            "this convolution's answer would have height (8 - 9 + 2 * 0) // 1 + 1 "
            "= 0, so the window does not fit over what it reads"
        )

    def test_a_pooling_window_wider_than_the_picture_names_both_numbers(self) -> None:
        body = check([{"kind": "pool", "reads": [4, 6, 6], "window": 7, "stride": 1}])

        assert body["layers"][0]["refusal"] == (
            "a window of 7 does not fit in a picture 6 by 6"
        )

    def test_two_different_pooling_layers_carry_the_identical_shape(self) -> None:
        wide = check([{"kind": "pool", "reads": [1, 4, 4], "window": 3, "stride": 1}])
        tiled = check([{"kind": "pool", "reads": [1, 4, 4], "window": 2, "stride": 2}])

        assert wide["layers"][0]["answers"] == [1, 2, 2]
        assert tiled["layers"][0]["answers"] == [1, 2, 2]
        assert wide["reads"] == tiled["reads"]
        assert wide["answers"] == tiled["answers"]
