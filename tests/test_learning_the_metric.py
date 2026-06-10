"""The learning-the-metric-itself page's endpoints, pinned at the numbers it quotes.

Every network here is trained from a fixed weight seed on fixed batches, so a
retrain reproduces these values exactly and the tolerances only absorb the
page's rounding to four places. The networks are shared through the module's
own caches, so each is trained once however many tests read it. The last class
asserts that no identifier reaches a reader through any string an endpoint
returns.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/learning-the-metric-itself"
FOUR = 5e-5
KINDS = ["cross", "square", "disc", "bar", "ring"]


def _get(path: str) -> dict:
    response = client.get(BASE + path)
    assert response.status_code == 200, response.text
    return response.json()


def _post(path: str, body: dict) -> dict:
    response = client.post(BASE + path, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _space(body: dict, name: str) -> dict:
    return next(space for space in body["spaces"] if space["space"] == name)


def _counts(space: dict) -> list[int]:
    return [kind["nearest_count"] for kind in space["kinds"]]


def _seen(body: dict, name: str) -> list[int]:
    space = next(space for space in body["seen_kinds_only"] if space["space"] == name)
    return [kind["nearest_count"] for kind in space["kinds"]]


def _ratio(space: dict, kind: str) -> float:
    return next(row for row in space["kinds"] if row["kind"] == kind)[
        "within_over_across"
    ]


class TestPairLoss:
    def test_a_same_kind_pair_is_pulled_by_its_gap(self) -> None:
        body = _post(
            "/pair-loss",
            {"first": [0, 0], "second": [0.36, 0.48], "same_kind": True, "margin": 1},
        )
        assert body["distance"] == pytest.approx(0.6)
        assert body["loss"] == pytest.approx(0.18)
        assert body["loss_if_different"] == pytest.approx(0.08)
        assert body["slope_on_first"] == pytest.approx([-0.36, -0.48])
        assert body["slope_on_second"] == pytest.approx([0.36, 0.48])

    def test_a_different_kind_pair_inside_the_margin_is_pushed(self) -> None:
        body = _post(
            "/pair-loss",
            {"first": [0, 0], "second": [0.36, 0.48], "same_kind": False, "margin": 1},
        )
        assert body["loss"] == pytest.approx(0.08)
        assert body["slope_on_first"] == pytest.approx([0.24, 0.32])
        assert body["beyond_margin"] is False

    def test_a_different_kind_pair_beyond_the_margin_is_left_alone(self) -> None:
        body = _post(
            "/pair-loss",
            {"first": [0, 0], "second": [1.2, 1.6], "same_kind": False, "margin": 1},
        )
        assert body["distance"] == pytest.approx(2.0)
        assert body["loss"] == 0.0
        assert body["slope_on_first"] == [0.0, 0.0]
        assert body["loss_if_same"] == pytest.approx(2.0)

    def test_a_push_at_distance_zero_has_a_size_and_no_direction(self) -> None:
        body = _post(
            "/pair-loss",
            {"first": [0, 0], "second": [0, 0], "same_kind": False, "margin": 1},
        )
        assert body["loss"] == pytest.approx(0.5)
        assert body["direction_defined"] is False
        assert body["slope_on_first"] == [0.0, 0.0]


class TestWorkedPairs:
    def test_two_crosses_and_a_cross_with_a_square(self) -> None:
        body = _get("/worked-pair")
        crosses, mixed = body["pairs"]
        assert body["width"] == 8
        assert crosses["same_kind"] is True
        assert crosses["distance"] == pytest.approx(0.1536, abs=FOUR)
        assert crosses["loss"] == pytest.approx(0.0118, abs=FOUR)
        assert crosses["loss_if_different"] == pytest.approx(0.3582, abs=FOUR)
        assert crosses["pixel_distance"] == pytest.approx(3.7191, abs=FOUR)
        assert mixed["same_kind"] is False
        assert mixed["distance"] == pytest.approx(1.5737, abs=FOUR)
        assert mixed["loss"] == 0.0
        assert mixed["loss_if_same"] == pytest.approx(1.2382, abs=FOUR)
        assert mixed["pixel_distance"] == pytest.approx(4.3732, abs=FOUR)


class TestSharedWeights:
    def test_the_sum_of_the_two_towers_is_the_measured_slope(self) -> None:
        body = _get("/shared-weights")
        assert body["mean_loss"] == pytest.approx(0.2056, abs=FOUR)
        assert body["largest_gap_to_both"] < 1e-9
        for row in body["rows"]:
            assert row["both"] == pytest.approx(
                row["first_tower"] + row["second_tower"], abs=1e-12
            )

    def test_one_tower_alone_is_not_the_slope(self) -> None:
        body = _get("/shared-weights")
        assert min(row["gap_to_first_tower"] for row in body["rows"]) > 0.01

    def test_the_position_biases_cancel_exactly(self) -> None:
        body = _get("/shared-weights")
        assert body["position_biases_cancel_exactly"] is True
        assert body["largest_position_bias_slope_one_tower"] == pytest.approx(
            0.1462, abs=FOUR
        )


class TestComparison:
    def test_the_main_study_among_all_300(self) -> None:
        body = _get("/comparison")
        assert body["training_kinds"] == ["cross", "square", "disc"]
        assert body["n_training_pictures"] == 180
        assert body["n_evaluation_pictures"] == 300
        assert body["classifier_accuracy"] == pytest.approx(0.9778, abs=FOUR)
        assert body["pair_loss_first_epoch"] == pytest.approx(0.1351, abs=FOUR)
        assert body["pair_loss_last_epoch"] == pytest.approx(0.0122, abs=FOUR)
        assert _counts(_space(body, "pixels")) == [45, 38, 50, 54, 44]
        assert _counts(_space(body, "classifier_all_four")) == [56, 40, 58, 59, 43]
        assert _counts(_space(body, "classifier_three")) == [54, 51, 57, 52, 46]
        assert _counts(_space(body, "pairs")) == [39, 45, 57, 33, 33]

    def test_the_main_study_among_the_seen_kinds_only(self) -> None:
        body = _get("/comparison")
        assert _seen(body, "pixels") == [47, 46, 51]
        assert _seen(body, "classifier_all_four") == [56, 60, 58]
        assert _seen(body, "classifier_three") == [60, 59, 60]
        assert _seen(body, "pairs") == [56, 56, 59]

    @pytest.mark.parametrize(
        ("path", "accuracy", "pairs", "classifier", "seen_pairs", "seen_classifier"),
        [
            (
                "/comparison/seed-1",
                0.9611,
                [32, 43, 49, 27, 33],
                [45, 47, 57, 51, 51],
                [47, 55, 55],
                [57, 59, 58],
            ),
            (
                "/comparison/seed-2",
                0.9833,
                [38, 60, 45, 34, 39],
                [56, 50, 59, 53, 43],
                [59, 60, 59],
                [60, 57, 59],
            ),
        ],
    )
    def test_the_other_two_seeds(
        self,
        path: str,
        accuracy: float,
        pairs: list[int],
        classifier: list[int],
        seen_pairs: list[int],
        seen_classifier: list[int],
    ) -> None:
        body = _get(path)
        assert body["classifier_accuracy"] == pytest.approx(accuracy, abs=FOUR)
        assert _counts(_space(body, "pairs")) == pairs
        assert _counts(_space(body, "classifier_three")) == classifier
        assert _seen(body, "pairs") == seen_pairs
        assert _seen(body, "classifier_three") == seen_classifier

    @pytest.mark.parametrize(
        ("path", "held_back", "pairs_held", "classifier_held", "pairs_ring", "ring"),
        [
            ("/comparison/without-cross", "cross", 21, 44, 38, 43),
            ("/comparison/without-disc", "disc", 29, 47, 31, 47),
            ("/comparison/without-square", "square", 30, 46, 35, 44),
        ],
    )
    def test_whichever_kind_is_held_back(
        self,
        path: str,
        held_back: str,
        pairs_held: int,
        classifier_held: int,
        pairs_ring: int,
        ring: int,
    ) -> None:
        body = _get(path)
        assert body["held_back"] == held_back
        position = KINDS.index(held_back)
        assert _counts(_space(body, "pairs"))[position] == pairs_held
        assert _counts(_space(body, "classifier_three"))[position] == classifier_held
        assert _counts(_space(body, "pairs"))[4] == pairs_ring
        assert _counts(_space(body, "classifier_three"))[4] == ring

    @pytest.mark.parametrize(
        "path",
        [
            "/comparison",
            "/comparison/seed-1",
            "/comparison/seed-2",
            "/comparison/without-cross",
            "/comparison/without-disc",
            "/comparison/without-square",
        ],
    )
    def test_the_whole_cloud_reading_puts_the_rings_tighter_under_pairs(
        self, path: str
    ) -> None:
        body = _get(path)
        assert _ratio(_space(body, "pairs"), "ring") < _ratio(
            _space(body, "classifier_three"), "ring"
        )

    def test_the_wider_tower_does_no_better_on_the_bar(self) -> None:
        body = _get("/wider-positions")
        assert body["wide_width"] == 16
        assert [kind["nearest_count"] for kind in body["narrow"]] == [
            39,
            45,
            57,
            33,
            33,
        ]
        assert [kind["nearest_count"] for kind in body["wide"]] == [
            37,
            47,
            53,
            31,
            43,
        ]


class TestSphere:
    def test_two_spaces_and_the_pictures(self) -> None:
        body = _get("/sphere")
        assert [space["space"] for space in body["spaces"]] == [
            "pairs",
            "classifier_three",
        ]
        assert len(body["levels"]) == 300
        pairs, classifier = body["spaces"]
        assert pairs["kept_share"] == pytest.approx(0.9405, abs=FOUR)
        assert classifier["kept_share"] == pytest.approx(0.8547, abs=FOUR)
        assert all(len(point["neighbours"]) == 5 for point in pairs["points"])


class TestMargin:
    def test_at_a_fixed_rate_margin_two_freezes_on_one_point(self) -> None:
        body = _get("/margin?margin=2&rate_rule=fixed")
        different_share = 192 / 276
        assert body["spreads"][-1] < 1e-3
        assert body["final_loss"] == pytest.approx(different_share * 2.0**2 / 2)
        assert body["final_loss"] == pytest.approx(1.3913, abs=FOUR)

    def test_at_a_fixed_rate_margin_four_freezes_at_once(self) -> None:
        body = _get("/margin?margin=4&rate_rule=fixed")
        assert max(body["spreads"]) < 1e-3
        assert body["final_loss"] == pytest.approx(5.5652, abs=FOUR)

    @pytest.mark.parametrize(
        ("margin", "rings", "within", "across"),
        [
            (0.5, 32, 0.2267, 0.6474),
            (2, 44, 0.5982, 2.8644),
            (4, 49, 1.1180, 5.7965),
        ],
    )
    def test_with_the_rate_scaled(
        self, margin: float, rings: int, within: float, across: float
    ) -> None:
        body = _get(f"/margin?margin={margin}&rate_rule=scaled")
        assert body["rate"] == pytest.approx(1.0 / margin**2)
        assert body["kinds"][4]["nearest_count"] == rings
        assert body["mean_within"] == pytest.approx(within, abs=FOUR)
        assert body["mean_across"] == pytest.approx(across, abs=FOUR)

    def test_a_rate_of_four_freezes_margin_one_half_on_seed_one(self) -> None:
        body = _get("/margin?margin=0.5&rate_rule=scaled&seed=1")
        assert body["rate"] == pytest.approx(4.0)
        assert body["spreads"][-1] < 1e-3
        assert body["final_loss"] == pytest.approx(192 / 276 * 0.5**2 / 2)

    def test_the_default_margin_is_the_main_study(self) -> None:
        body = _get("/margin?margin=1&rate_rule=fixed")
        assert body["rate_rule"] == "scaled"
        assert body["final_loss"] == pytest.approx(0.0122, abs=FOUR)
        assert body["beyond_margin"] == pytest.approx(0.6620, abs=FOUR)


class TestWithoutDifferentKinds:
    def test_the_positions_fall_onto_one_point(self) -> None:
        body = _get("/without-different-kinds")
        assert body["start_spread"] == pytest.approx(0.7325, abs=FOUR)
        assert body["spreads_without"][0] == pytest.approx(0.0393, abs=FOUR)
        assert body["spreads_without"][-1] == pytest.approx(0.0018, abs=FOUR)
        assert body["spreads_with"][-1] == pytest.approx(0.8284, abs=FOUR)
        assert body["losses_without"][-1] < 1e-5
        assert body["losses_with"][-1] == pytest.approx(0.0122, abs=FOUR)
        assert [kind["ten_nearest_same_kind"] for kind in body["kinds_without"]] == (
            pytest.approx([0.2917, 0.3, 0.2633, 0.12, 0.085], abs=FOUR)
        )


class TestCost:
    def test_the_pairs_and_how_many_still_taught(self) -> None:
        body = _get("/cost")
        assert body["all_pairs"] == 16110
        assert body["same_kind_pairs"] == 5310
        assert body["different_kind_pairs"] == 10800
        assert body["pairs_per_batch"] == 276
        assert body["same_kind_per_batch"] == 84
        assert body["steps_per_epoch"] == 7
        assert body["pair_evaluations"] == 77280
        assert body["distinct_pairs_met"] == 16015
        assert body["beyond_margin_training"] == pytest.approx(0.7909, abs=FOUR)
        assert body["beyond_margin_held_out"] == pytest.approx(0.6620, abs=FOUR)
        assert body["growth"][-1] == {"pictures": 1_000_000, "pairs": 499_999_500_000}


IDENTIFIER = re.compile(r"[a-z]+_[a-z_]+|[A-Z][a-z]+[A-Z]\w*|oop_ml|librar|\bAPI\b")


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    return []


class TestRefusalsAndWords:
    @pytest.mark.parametrize(
        "path",
        [
            "/margin?margin=3",
            "/margin?margin=1&rate_rule=halved",
            "/margin?margin=1&seed=7",
        ],
    )
    def test_the_margin_sweep_refuses_in_plain_words(self, path: str) -> None:
        response = client.get(BASE + path)
        assert response.status_code == 422
        assert not IDENTIFIER.search(response.json()["detail"])

    def test_two_widths_are_refused_in_plain_words(self) -> None:
        response = client.post(
            BASE + "/pair-loss",
            json={"first": [0, 0], "second": [1], "same_kind": False},
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "same number of coordinates" in detail
        assert not IDENTIFIER.search(detail)

    @pytest.mark.parametrize(
        "path",
        [
            "/comparison",
            "/worked-pair",
            "/shared-weights",
            "/without-different-kinds",
            "/cost",
            "/wider-positions",
        ],
    )
    def test_no_identifier_reaches_a_reader(self, path: str) -> None:
        body = _get(path)
        texts = [
            text
            for text in _strings(body)
            if text
            not in (
                "pixels",
                "classifier_all_four",
                "classifier_three",
                "pairs",
                "scaled",
                "fixed",
            )
        ]
        for text in texts:
            assert not IDENTIFIER.search(text), text
