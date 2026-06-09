"""The maximum matching page's endpoints, on the five texts the page carries.

Every number the page quotes in prose is pinned here, so a change underneath
that moved one of them would fail a test rather than quietly rewrite the page.
Three other things are held. The walk is a second implementation of the greedy
loop, so its takings are checked against what the segmenter itself answered.
The exhaustive floor is checked against a small independent enumeration written
from the definition rather than from the dynamic program that computes it. And
the last class asserts that nothing identifier-shaped survives into any string
a widget renders, including into the one refusal the page's own controls can
reach.
"""

from __future__ import annotations

import re
from itertools import product

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/maximum-matching"
SETUP = f"{BASE}/setup"
SCENARIOS = f"{BASE}/scenarios"
MISSES = f"{BASE}/misses"
SEGMENT = f"{BASE}/segment"

SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
SPACELESS = "Dr.Alvarezdidn'texpectthelow-costre-analysis."
TABLE = "thetabledownthere"
PARK = "我们在野生动物园玩"
ORIGINS = "研究生命起源"
NAME = "阿尔瓦雷斯研究生命起源"

PARK_WORDS = ["我们", "在", "在野", "野生", "生动", "动物园", "物", "园", "玩"]
ORIGINS_WORDS = ["研究", "研究生", "生命", "命", "起源"]


def _scenario(key: str) -> dict:
    body = client.get(SCENARIOS).json()
    return next(item for item in body["scenarios"] if item["key"] == key)


def _reading(analysis: dict, scan: str) -> dict:
    return next(item for item in analysis["readings"] if item["scan"] == scan)


def _texts(reading: dict) -> list[str]:
    return [piece["text"] for piece in reading["pieces"]]


class TestSetup:
    def test_the_sentence_loses_six_spaces_and_keeps_forty_five_characters(
        self,
    ) -> None:
        body = client.get(SETUP).json()

        assert body["sentence"] == SENTENCE
        assert body["n_sentence_characters"] == 51
        assert body["n_spaces"] == 6
        assert body["n_words_with_spaces"] == 7
        assert body["spaceless_sentence"] == SPACELESS
        assert body["n_spaceless_characters"] == 45

    def test_forty_five_characters_admit_two_to_the_forty_fourth_cuts(self) -> None:
        body = client.get(SETUP).json()

        assert body["n_cuts"] == "17592186044416"
        assert body["n_cuts"] == str(2**44)
        assert body["n_park_cuts"] == "256"
        assert body["n_park_characters"] == 9

    def test_no_earlier_rule_finds_a_word_in_the_spaceless_sentence(self) -> None:
        body = client.get(SETUP).json()
        rules = body["earlier_rules_on_the_spaceless_sentence"]

        assert [rule["n_pieces"] for rule in rules] == [1, 3, 2, 3, 10]
        assert rules[0]["pieces"] == [SPACELESS]

    def test_the_earlier_rules_answer_one_or_nine_for_the_park_sentence(self) -> None:
        body = client.get(SETUP).json()
        rules = body["earlier_rules_on_the_park_sentence"]

        assert [rule["n_pieces"] for rule in rules] == [1, 9, 1, 1, 1]
        assert rules[1]["pieces"] == list(PARK)
        assert body["n_words_a_reader_finds_in_the_park_sentence"] == 5

    def test_the_english_word_list_holds_eight_hundred_and_ten_words(self) -> None:
        assert client.get(SETUP).json()["n_english_words"] == 810


class TestTheSpacelessSentence:
    def test_it_comes_back_in_twenty_four_pieces_against_seven_words(self) -> None:
        scenario = _scenario("spaceless")
        reading = _reading(scenario["analysis"], "left to right")

        assert reading["n_pieces"] == 24
        assert scenario["reader_reading"] == [
            "Dr.",
            "Alvarez",
            "didn't",
            "expect",
            "the",
            "low-cost",
            "re-analysis.",
        ]
        assert len(scenario["reader_reading"]) == 7

    def test_eighteen_of_the_twenty_four_are_lone_characters(self) -> None:
        reading = _reading(_scenario("spaceless")["analysis"], "left to right")

        assert reading["n_single_characters"] == 18
        assert reading["n_entries"] == 7
        assert _texts(reading)[:11] == [
            "D",
            "r",
            ".",
            "A",
            "l",
            "v",
            "a",
            "r",
            "e",
            "z",
            "did",
        ]

    def test_the_two_scans_agree_here_and_the_answer_is_the_shortest_possible(
        self,
    ) -> None:
        analysis = _scenario("spaceless")["analysis"]

        assert _texts(_reading(analysis, "left to right")) == _texts(
            _reading(analysis, "right to left")
        )
        assert analysis["shortest"]["n_pieces"] == 24
        assert analysis["shortest"]["n_readings"] == "1"
        assert analysis["greedy_reaches_shortest"] is True

    def test_the_scan_asks_about_one_hundred_and_ninety_one_candidates(self) -> None:
        analysis = _scenario("spaceless")["analysis"]

        assert _reading(analysis, "left to right")["n_lookups"] == 191
        assert _reading(analysis, "right to left")["n_lookups"] == 156
        assert analysis["longest_entry"] == 10

    def test_the_reading_a_person_gives_is_not_available_from_this_list(self) -> None:
        assert _scenario("spaceless")["reader_reading_is_all_entries"] is False


class TestTheDirectionsDisagree:
    def test_the_two_scans_give_different_english_readings(self) -> None:
        analysis = _scenario("table")["analysis"]

        assert _texts(_reading(analysis, "left to right")) == [
            "theta",
            "bled",
            "own",
            "there",
        ]
        assert _texts(_reading(analysis, "right to left")) == [
            "the",
            "table",
            "down",
            "there",
        ]

    def test_both_are_four_pieces_of_four_entries_with_no_lone_characters(self) -> None:
        analysis = _scenario("table")["analysis"]

        for scan in ("left to right", "right to left"):
            reading = _reading(analysis, scan)
            assert reading["n_pieces"] == 4
            assert reading["n_entries"] == 4
            assert reading["n_single_characters"] == 0

    def test_counting_pieces_cannot_separate_them(self) -> None:
        analysis = _scenario("table")["analysis"]

        assert analysis["shortest"]["n_pieces"] == 4
        assert analysis["shortest"]["n_readings"] == "2"
        assert analysis["shortest"]["readings"] == [
            ["the", "table", "down", "there"],
            ["theta", "bled", "own", "there"],
        ]

    def test_the_reader_reading_is_the_right_to_left_one(self) -> None:
        scenario = _scenario("table")

        assert scenario["reader_reading"] == ["the", "table", "down", "there"]
        assert scenario["reader_reading_is_all_entries"] is True

    def test_the_convention_alone_decides_which_is_kept(self) -> None:
        both = _reading(_scenario("table")["analysis"], "both")

        assert both["chosen_scan"] == "right to left"
        assert "standing preference" in both["decided_by"]


class TestThePark:
    def test_the_left_to_right_scan_takes_two_words_the_sentence_does_not_hold(
        self,
    ) -> None:
        analysis = _scenario("park")["analysis"]

        assert _texts(_reading(analysis, "left to right")) == [
            "我们",
            "在野",
            "生动",
            "物",
            "园",
            "玩",
        ]
        assert _reading(analysis, "left to right")["n_pieces"] == 6

    def test_the_right_to_left_scan_meets_the_park_whole(self) -> None:
        analysis = _scenario("park")["analysis"]

        assert _texts(_reading(analysis, "right to left")) == [
            "我们",
            "在",
            "野生",
            "动物园",
            "玩",
        ]
        assert _reading(analysis, "right to left")["n_pieces"] == 5

    def test_the_greedy_answer_is_not_even_the_shortest_available(self) -> None:
        analysis = _scenario("park")["analysis"]

        assert analysis["shortest"]["n_pieces"] == 5
        assert analysis["shortest"]["n_readings"] == "2"
        assert analysis["greedy_reaches_shortest"] is False
        assert analysis["shortest"]["readings"][0] == [
            "我们",
            "在",
            "野生",
            "动物园",
            "玩",
        ]

    def test_fewer_pieces_is_what_settles_the_both_ways_answer(self) -> None:
        both = _reading(_scenario("park")["analysis"], "both")

        assert both["chosen_scan"] == "right to left"
        assert both["decided_by"] == "it used fewer pieces"

    def test_nine_characters_admit_two_hundred_and_fifty_six_cuts_and_nine_lookups(
        self,
    ) -> None:
        analysis = _scenario("park")["analysis"]

        assert analysis["n_cuts"] == "256"
        assert _reading(analysis, "left to right")["n_lookups"] == 9
        assert _reading(analysis, "right to left")["n_lookups"] == 8


class TestTheTiedSentence:
    def test_both_scans_use_three_pieces(self) -> None:
        analysis = _scenario("origins")["analysis"]

        assert _texts(_reading(analysis, "left to right")) == ["研究生", "命", "起源"]
        assert _texts(_reading(analysis, "right to left")) == ["研究", "生命", "起源"]
        assert analysis["shortest"]["n_pieces"] == 3
        assert analysis["shortest"]["n_readings"] == "2"

    def test_the_count_of_lone_characters_breaks_the_tie(self) -> None:
        analysis = _scenario("origins")["analysis"]
        both = _reading(analysis, "both")

        assert _reading(analysis, "left to right")["n_single_characters"] == 1
        assert _reading(analysis, "right to left")["n_single_characters"] == 0
        assert both["chosen_scan"] == "right to left"
        assert "fewer" in both["decided_by"]

    def test_the_page_claims_no_reading_of_this_sentence(self) -> None:
        """Both readings are of real words, which is the whole point of the fixture."""
        assert _scenario("origins")["reader_reading"] is None


class TestWhichRuleSettlesTheTie:
    """The page counts how often each of the three conventions decides, so pin it."""

    def test_the_third_rule_decides_two_of_the_five_texts(self) -> None:
        reasons = [
            _reading(scenario["analysis"], "both")["decided_by"]
            for scenario in client.get(SCENARIOS).json()["scenarios"]
        ]

        assert sum(1 for reason in reasons if reason == "it used fewer pieces") == 1
        assert sum(1 for reason in reasons if "lone characters" in reason) == 2
        assert sum(1 for reason in reasons if "standing preference" in reason) == 2
        assert len(reasons) == 5


class TestOutsideTheWordList:
    def test_a_surname_becomes_five_lone_characters(self) -> None:
        analysis = _scenario("name")["analysis"]
        reading = _reading(analysis, "right to left")

        assert _texts(reading)[:5] == ["阿", "尔", "瓦", "雷", "斯"]
        assert reading["n_pieces"] == 8
        assert reading["n_entries"] == 3
        assert reading["n_single_characters"] == 5

    def test_a_reader_answers_four_words_where_the_method_answers_eight(self) -> None:
        scenario = _scenario("name")

        assert scenario["reader_reading"] == ["阿尔瓦雷斯", "研究", "生命", "起源"]
        assert len(scenario["reader_reading"]) == 4
        assert scenario["reader_reading_is_all_entries"] is False

    def test_the_lone_characters_are_the_only_fallback_there_is(self) -> None:
        analysis = _scenario("name")["analysis"]

        assert analysis["shortest"]["n_pieces"] == 8
        assert analysis["greedy_reaches_shortest"] is True


class TestAMissingWordLeavesNoTrace:
    def test_taking_the_park_out_makes_both_scans_agree_on_the_wrong_answer(
        self,
    ) -> None:
        miss = client.get(MISSES).json()["misses"][0]

        assert miss["removed"] == "动物园"
        assert [_texts(reading) for reading in miss["without_the_word"]] == [
            ["我们", "在野", "生动", "物", "园", "玩"],
            ["我们", "在野", "生动", "物", "园", "玩"],
        ]

    def test_and_every_piece_of_that_answer_is_an_entry(self) -> None:
        miss = client.get(MISSES).json()["misses"][0]

        for reading in miss["without_the_word"]:
            assert reading["n_pieces"] == 6
            assert reading["n_entries"] == 6
            assert reading["n_single_characters"] == 3

    def test_the_right_to_left_answer_was_five_pieces_before_the_word_went(
        self,
    ) -> None:
        miss = client.get(MISSES).json()["misses"][0]
        before = miss["with_the_word"][1]

        assert before["n_pieces"] == 5
        assert _texts(before) == miss["reader_reading"]

    def test_in_english_one_scan_notices_and_the_other_does_not(self) -> None:
        miss = client.get(MISSES).json()["misses"][1]
        left, right = miss["without_the_word"]

        assert miss["removed"] == "table"
        assert _texts(left) == ["theta", "bled", "own", "there"]
        assert left["n_pieces"] == 4
        assert left["n_entries"] == 4
        assert _texts(right) == ["theta", "b", "l", "e", "down", "there"]
        assert right["n_entries"] == 3

    def test_the_left_to_right_answer_is_unchanged_by_losing_the_word(self) -> None:
        miss = client.get(MISSES).json()["misses"][1]

        assert _texts(miss["with_the_word"][0]) == _texts(miss["without_the_word"][0])


class TestTheWalkAgreesWithTheAnswer:
    """The walk is a second implementation of the loop, so it is checked against one."""

    def test_every_walk_rebuilds_the_text_it_walked(self) -> None:
        for scenario in client.get(SCENARIOS).json()["scenarios"]:
            analysis = scenario["analysis"]
            for reading in analysis["readings"]:
                if reading["walk"] is None:
                    continue
                takings = [step["taken"] for step in reading["walk"]]
                if reading["scan"] == "right to left":
                    takings.reverse()
                assert "".join(takings) == analysis["text"].replace(" ", "")

    def test_every_walk_takes_exactly_the_pieces_the_answer_holds(self) -> None:
        for scenario in client.get(SCENARIOS).json()["scenarios"]:
            for reading in scenario["analysis"]["readings"]:
                if reading["walk"] is None:
                    continue
                takings = [step["taken"] for step in reading["walk"]]
                if reading["scan"] == "right to left":
                    takings.reverse()
                assert takings == _texts(reading)

    def test_only_the_last_candidate_tried_at_a_position_was_an_entry(self) -> None:
        for scenario in client.get(SCENARIOS).json()["scenarios"]:
            for reading in scenario["analysis"]["readings"]:
                for step in reading["walk"] or []:
                    for attempt in step["attempts"][:-1]:
                        assert attempt["is_entry"] is False

    def test_the_lookups_are_the_candidates_the_walk_asked_about(self) -> None:
        for scenario in client.get(SCENARIOS).json()["scenarios"]:
            for reading in scenario["analysis"]["readings"]:
                if reading["walk"] is None:
                    continue
                assert reading["n_lookups"] == sum(
                    len(step["attempts"]) for step in reading["walk"]
                )

    def test_the_walk_over_the_park_sentence_takes_six_steps(self) -> None:
        walk = _reading(_scenario("park")["analysis"], "left to right")["walk"]

        assert [step["at"] for step in walk] == [0, 2, 4, 6, 7, 8]
        assert [step["taken"] for step in walk] == [
            "我们",
            "在野",
            "生动",
            "物",
            "园",
            "玩",
        ]
        assert walk[1]["attempts"] == [
            {"text": "在野生", "is_entry": False},
            {"text": "在野", "is_entry": True},
        ]


class TestTheFloorIsTheRealFloor:
    """The dynamic program is checked against an enumeration written from the definition."""

    def _by_hand(self, text: str, words: list[str]) -> int:
        held = set(words)
        best = len(text)
        for cuts in product([False, True], repeat=len(text) - 1):
            pieces: list[str] = [text[0]]
            for character, cut in zip(text[1:], cuts):
                if cut:
                    pieces.append(character)
                else:
                    pieces[-1] += character
            if all(len(piece) == 1 or piece in held for piece in pieces):
                best = min(best, len(pieces))
        return best

    def test_the_park_floor_agrees_with_every_cut_tried(self) -> None:
        assert self._by_hand(PARK, PARK_WORDS) == 5

    def test_the_tied_sentence_floor_agrees_with_every_cut_tried(self) -> None:
        assert self._by_hand(ORIGINS, ORIGINS_WORDS) == 3

    def test_the_surname_floor_agrees_with_every_cut_tried(self) -> None:
        assert self._by_hand(NAME, ORIGINS_WORDS) == 8


class TestSegmenting:
    def test_a_stretch_outside_the_list_becomes_one_piece_per_character(self) -> None:
        body = client.post(SEGMENT, json={"text": "thecatsatxyz"}).json()
        reading = _reading(body["analysis"], "left to right")

        assert _texts(reading) == ["the", "cat", "s", "at", "x", "y", "z"]

    def test_a_match_never_crosses_a_space(self) -> None:
        joined = client.post(
            SEGMENT, json={"text": "thecat", "word_list": "english"}
        ).json()
        apart = client.post(
            SEGMENT, json={"text": "the cat", "word_list": "english"}
        ).json()

        assert _texts(_reading(joined["analysis"], "left to right")) == ["the", "cat"]
        assert _texts(_reading(apart["analysis"], "left to right")) == ["the", "cat"]
        assert [
            piece["start"]
            for piece in _reading(apart["analysis"], "left to right")["pieces"]
        ] == [0, 4]

    def test_striking_out_the_park_changes_the_right_to_left_answer(self) -> None:
        body = client.post(
            SEGMENT,
            json={"text": PARK, "word_list": "park", "removed": ["动物园"]},
        ).json()

        assert body["removed"] == ["动物园"]
        assert _texts(_reading(body["analysis"], "right to left")) == [
            "我们",
            "在野",
            "生动",
            "物",
            "园",
            "玩",
        ]

    def test_the_words_of_a_small_list_come_back_and_the_long_one_does_not(
        self,
    ) -> None:
        small = client.post(SEGMENT, json={"text": PARK, "word_list": "park"}).json()
        large = client.post(SEGMENT, json={"text": TABLE}).json()

        assert small["words_in_list"] == PARK_WORDS
        assert large["words_in_list"] == []
        assert large["analysis"]["n_words_in_list"] == 810

    def test_a_word_not_in_the_chosen_list_is_ignored_rather_than_refused(self) -> None:
        body = client.post(
            SEGMENT,
            json={"text": PARK, "word_list": "park", "removed": ["table"]},
        ).json()

        assert body["removed"] == []
        assert _reading(body["analysis"], "left to right")["n_pieces"] == 6

    def test_a_blank_text_has_no_pieces(self) -> None:
        body = client.post(SEGMENT, json={"text": "   "}).json()

        assert _reading(body["analysis"], "left to right")["n_pieces"] == 0
        assert body["analysis"]["shortest"]["n_pieces"] == 0


class TestRefusals:
    def test_striking_out_every_word_is_refused_in_the_pages_own_words(self) -> None:
        response = client.post(
            SEGMENT,
            json={"text": PARK, "word_list": "park", "removed": PARK_WORDS},
        )

        assert response.status_code == 422
        assert response.json()["detail"] == "a word list cannot be empty"

    def test_a_text_longer_than_the_bound_is_refused(self) -> None:
        response = client.post(SEGMENT, json={"text": "x" * 61})

        assert response.status_code == 422

    def test_an_unknown_word_list_is_refused(self) -> None:
        response = client.post(SEGMENT, json={"text": "abc", "word_list": "german"})

        assert response.status_code == 422

    def test_an_unknown_keyword_is_refused(self) -> None:
        response = client.post(SEGMENT, json={"text": "abc", "direction": "left"})

        assert response.status_code == 422


IDENTIFIER = re.compile(r"[a-z]+_[a-z]+|[A-Z][a-z]+[A-Z]")
"""Anything shaped like a parameter name or a class name."""

ALLOWED = frozenset(
    {
        "Dr",
        "Alvarez",
    }
)
"""Words of the running sentence that look like nothing else."""


def _strings(value: object) -> list[str]:
    """Every string anywhere inside a response body."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            found
            for key, item in value.items()
            for found in _strings(item)
            if key not in {"text", "taken", "sentence", "spaceless_sentence"}
        ]
    if isinstance(value, list):
        return [found for item in value for found in _strings(item)]
    return []


class TestNothingNamesTheCodeBehindThePage:
    def test_no_endpoint_returns_an_identifier(self) -> None:
        for path in (SETUP, SCENARIOS, MISSES):
            for sentence in _strings(client.get(path).json()):
                assert not IDENTIFIER.search(sentence), (path, sentence)

    def test_the_refusal_names_no_identifier(self) -> None:
        response = client.post(
            SEGMENT,
            json={"text": PARK, "word_list": "park", "removed": PARK_WORDS},
        )
        detail = response.json()["detail"]

        assert not IDENTIFIER.search(detail)
        for forbidden in ("dictionary", "Word", "Error", "segmenter"):
            assert forbidden not in detail

    def test_the_labels_a_widget_shows_name_no_identifier(self) -> None:
        for scenario in client.get(SCENARIOS).json()["scenarios"]:
            assert not IDENTIFIER.search(scenario["label"])
            assert not IDENTIFIER.search(scenario["analysis"]["word_list_label"])
            for reading in scenario["analysis"]["readings"]:
                assert not IDENTIFIER.search(reading["decided_by"] or "")

    def test_the_allowed_words_are_the_running_sentences_own(self) -> None:
        assert ALLOWED == {"Dr", "Alvarez"}
        assert all(word in SENTENCE for word in ALLOWED)
