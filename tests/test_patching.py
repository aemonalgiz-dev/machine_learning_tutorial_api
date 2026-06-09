"""The patching endpoints, on the texts and the corpora the page is built from.

Every number the page quotes is pinned here, so a change in the library or in
this module fails a test rather than quietly changing a sentence on the page.
Three classes carry more than their weight. One checks that repeating a corpus
and lowering the smoothing are the same operation, which is the caveat the page
attaches to its own demonstration; one checks the four hand-derived entropies of
the four-byte example against Shannon's definition written out independently;
and the last checks that nothing these endpoints return carries an identifier,
because a widget renders these strings.
"""

from __future__ import annotations

import math
import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.patching import (
    ACCENTED_TEXT,
    CORPUS,
    FIXTURE_SENTENCE,
    RUNNING_SENTENCE,
    SEEN_SENTENCE,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

PATCH = "/concepts/patching-without-a-vocabulary/patch"
CUT_LANDING = "/concepts/patching-without-a-vocabulary/cut-landing"
CORPUS_CONDITION = "/concepts/patching-without-a-vocabulary/corpus-condition"
LIMITS = "/concepts/patching-without-a-vocabulary/limits"


def patched(text: str, **settings: object) -> dict:
    """One text cut both ways, at the page's own settings unless told otherwise."""
    response = client.post(PATCH, json={"text": text, **settings})
    assert response.status_code == 200, response.text
    return response.json()


def pieces_of(body: dict, method: str) -> list[str]:
    """What one of the two rules cut the text into, as text."""
    return [piece["text"] for piece in body[method]["pieces"]]


class TestCuttingAtAFixedSize:
    def test_the_running_sentence_at_four_bytes(self) -> None:
        body = patched(RUNNING_SENTENCE)

        assert body["n_bytes"] == 51
        assert body["fixed"]["n_patches"] == 13
        assert pieces_of(body, "fixed") == [
            "Dr. ",
            "Alva",
            "rez ",
            "didn",
            "'t e",
            "xpec",
            "t th",
            "e lo",
            "w-co",
            "st r",
            "e-an",
            "alys",
            "is.",
        ]

    def test_nine_of_its_twelve_cuts_fall_inside_a_word(self) -> None:
        view = patched(RUNNING_SENTENCE)["fixed"]

        assert view["inside_a_word"] == 9
        assert view["at_a_word_start"] == 2
        assert view["elsewhere"] == 1

    def test_the_last_block_is_the_short_one(self) -> None:
        view = patched(RUNNING_SENTENCE)["fixed"]

        assert view["longest"] == 4
        assert view["shortest"] == 3

    def test_a_block_size_of_one_is_the_bytes_themselves(self) -> None:
        body = patched(RUNNING_SENTENCE, patch_size=1)

        assert body["fixed"]["n_patches"] == body["n_bytes"]

    def test_a_block_can_stop_part_way_through_a_character(self) -> None:
        body = client.get(LIMITS).json()["broken"]

        assert body["n_characters"] == 38
        assert body["n_bytes"] == 40
        assert body["n_broken_characters"] == 2
        assert body["pieces"][2].endswith("�")
        assert body["pieces"][3].startswith("�")


class TestCuttingWhereTheNextByteIsHardToGuess:
    def test_a_sentence_the_model_read_comes_out_as_its_words(self) -> None:
        body = patched(SEEN_SENTENCE)

        assert pieces_of(body, "entropy") == [
            "The ",
            "re",
            "port was ",
            "expected ",
            "on ",
            "Monday.",
        ]

    def test_four_of_its_five_cuts_fall_at_a_word_start(self) -> None:
        view = patched(SEEN_SENTENCE)["entropy"]

        assert view["at_a_word_start"] == 4
        assert view["inside_a_word"] == 1
        assert view["elsewhere"] == 0

    def test_the_same_sentence_cut_every_four_bytes_is_worse_on_both_counts(
        self,
    ) -> None:
        body = patched(SEEN_SENTENCE)

        assert body["fixed"]["n_patches"] == 9
        assert body["fixed"]["inside_a_word"] == 6
        assert body["entropy"]["n_patches"] == 6

    def test_the_first_byte_of_a_word_is_the_uncertain_one(self) -> None:
        positions = patched(SEEN_SENTENCE)["positions"]
        starting = [row["entropy"] for row in positions if row["begins_a_patch"]]
        inside = [row["entropy"] for row in positions if not row["begins_a_patch"]]

        assert min(starting) > 1.0
        assert max(inside) < 1.0

    def test_the_settings_come_back_with_the_answer(self) -> None:
        body = patched(SEEN_SENTENCE)

        assert body["order"] == 4
        assert body["threshold"] == 1.0
        assert body["smoothing"] == 0.001
        assert body["n_contexts"] == 402
        assert body["n_texts_counted"] == 18 * 20

    def test_the_six_word_sentence_comes_out_as_its_words_too(self) -> None:
        body = patched(
            FIXTURE_SENTENCE,
            source="one sentence",
            order=2,
            threshold=0.5,
            patch_size=5,
        )

        assert pieces_of(body, "entropy") == [
            "the ",
            "cat ",
            "sat ",
            "on the ",
            "mat",
        ]
        assert pieces_of(body, "fixed") == ["the c", "at sa", "t on ", "the m", "at"]

    def test_a_word_only_ever_followed_by_one_other_is_not_cut_from_it(self) -> None:
        body = patched(FIXTURE_SENTENCE, source="one sentence", order=2, threshold=0.5)

        assert 15 not in body["entropy"]["boundaries"]
        assert body["entropy"]["boundaries"] == [0, 4, 8, 12, 19]


class TestASentenceTheModelNeverRead:
    def test_most_of_it_arrives_at_the_very_top_of_the_scale(self) -> None:
        body = client.get(LIMITS).json()

        assert body["sentence_bytes_at_the_top"] == 51
        assert body["unseen_bytes_at_the_top"] == 26

    def test_so_the_blocks_are_mostly_single_bytes(self) -> None:
        body = patched(RUNNING_SENTENCE)

        assert body["entropy"]["n_patches"] == 30
        assert body["entropy"]["shortest"] == 1
        assert pieces_of(body, "entropy")[:4] == ["Dr. A", "l", "v", "a"]

    def test_and_the_blocks_are_worse_than_the_fixed_ones_on_the_same_text(
        self,
    ) -> None:
        body = patched(RUNNING_SENTENCE)

        assert body["entropy"]["n_patches"] > body["fixed"]["n_patches"]


class TestWhereEveryCutLands:
    def test_the_share_of_all_positions_that_fall_inside_a_word(self) -> None:
        body = client.get(CUT_LANDING).json()

        assert body["corpus_bytes"] == 763
        assert body["n_texts"] == 18
        assert body["positions_inside_a_word"] == 489
        assert body["positions_between_bytes"] == 745
        assert body["word_starts"] == 115
        assert round(body["share_of_positions_inside_a_word"], 4) == 0.6564

    def test_no_fixed_size_does_better_than_that_share(self) -> None:
        """A rule that never reads the text cannot beat cutting anywhere."""
        rows = client.get(CUT_LANDING).json()["fixed"]

        assert min(row["share_inside_a_word"] for row in rows) > 0.55
        assert max(row["share_inside_a_word"] for row in rows) < 0.76

    def test_the_fixed_sweep_is_pinned_at_the_page_block_size(self) -> None:
        rows = client.get(CUT_LANDING).json()["fixed"]
        at_four = next(row for row in rows if row["setting"] == 4.0)

        assert at_four["n_patches"] == 197
        assert at_four["inside_a_word"] == 102
        assert at_four["at_a_word_start"] == 38
        assert round(at_four["share_inside_a_word"], 3) == 0.570

    def test_the_uncertainty_rule_halves_it_at_the_page_threshold(self) -> None:
        rows = client.get(CUT_LANDING).json()["thresholds"]
        at_one = next(row for row in rows if row["setting"] == 1.0)

        assert at_one["n_patches"] == 129
        assert at_one["inside_a_word"] == 32
        assert at_one["at_a_word_start"] == 69
        assert round(at_one["share_inside_a_word"], 3) == 0.288

    def test_at_a_high_enough_threshold_no_cut_falls_inside_a_word_at_all(
        self,
    ) -> None:
        rows = client.get(CUT_LANDING).json()["thresholds"]
        at_two = next(row for row in rows if row["setting"] == 2.0)

        assert at_two["inside_a_word"] == 0
        assert at_two["n_patches"] == 57

    def test_a_higher_threshold_never_makes_more_blocks(self) -> None:
        rows = client.get(CUT_LANDING).json()["thresholds"]
        counts = [row["n_patches"] for row in rows]

        assert counts == sorted(counts, reverse=True)

    def test_looking_further_back_cuts_less_often_and_lands_better(self) -> None:
        rows = client.get(CUT_LANDING).json()["orders"]

        assert [row["n_patches"] for row in rows] == [693, 338, 191, 129, 110]
        assert [round(row["share_inside_a_word"], 3) for row in rows] == [
            0.630,
            0.522,
            0.393,
            0.288,
            0.272,
        ]

    def test_one_byte_of_context_is_no_better_than_the_uninformed_rate(self) -> None:
        body = client.get(CUT_LANDING).json()
        first = body["orders"][0]

        assert (
            first["share_inside_a_word"]
            < body["share_of_positions_inside_a_word"] + 0.01
        )


class TestWhatTheCorpusHasToDoFirst:
    def test_add_one_smoothing_leaves_every_position_near_the_top(self) -> None:
        rows = client.get(CORPUS_CONDITION).json()["rows"]
        add_one = [row for row in rows if row["smoothing"] == 1.0]

        for row in add_one:
            assert row["lowest"] > 7.4
            assert row["n_patches"] == 22
            assert not row["words_recovered"]

    def test_twenty_copies_at_add_one_still_separate_nothing(self) -> None:
        rows = client.get(CORPUS_CONDITION).json()["rows"]
        row = next(
            item for item in rows if item["smoothing"] == 1.0 and item["n_copies"] == 20
        )

        assert round(row["inside_a_word"], 4) == 7.4674
        assert round(row["at_a_word_start"], 4) == 7.5862
        assert round(row["gap"], 4) == 0.1189

    def test_eight_copies_recover_the_words_and_seven_do_not(self) -> None:
        rows = client.get(CORPUS_CONDITION).json()["rows"]
        small = next(
            item
            for item in rows
            if item["smoothing"] == 0.001 and item["n_copies"] == 7
        )
        enough = next(
            item
            for item in rows
            if item["smoothing"] == 0.001 and item["n_copies"] == 8
        )

        assert not small["words_recovered"]
        assert small["n_patches"] == 16
        assert enough["words_recovered"]
        assert enough["n_patches"] == 5
        assert enough["patches"] == ["the ", "cat ", "sat ", "on the ", "mat"]

    def test_the_docstring_numbers_at_twenty_copies(self) -> None:
        rows = client.get(CORPUS_CONDITION).json()["rows"]
        row = next(
            item
            for item in rows
            if item["smoothing"] == 0.001 and item["n_copies"] == 20
        )

        assert round(row["inside_a_word"], 3) == 0.106
        assert round(row["at_a_word_start"], 3) == 1.099

    def test_one_repeated_phrase_makes_every_position_equally_certain(self) -> None:
        phrase = client.get(CORPUS_CONDITION).json()["one_phrase"]

        assert phrase["every_position_agrees"]
        assert round(phrase["entropies"][0], 6) == 0.198144
        assert phrase["patches"] == ["the cat sat"]

    def test_repeating_a_corpus_is_lowering_the_smoothing_and_nothing_else(
        self,
    ) -> None:
        """Twenty copies at one smoothing against one copy at a twentieth of it."""
        same = client.get(CORPUS_CONDITION).json()["equivalence"]

        assert same["single_copy_smoothing"] == 0.001 / 20
        assert same["largest_gap"] < 1e-12
        assert same["patches_agree"]


class TestTheFourByteExample:
    def test_the_entropies_match_shannon_written_out_independently(self) -> None:
        """The corpus ``aab`` at a context of one byte and add-one smoothing.

        The start was followed by ``a`` once, so ``a`` has probability 2/257 and
        each of the other 255 values 1/257. The context ``a`` was followed by
        ``a`` once and ``b`` once, so those two have 2/258 and the other 254 have
        1/258. The context ``b`` was never seen, so all 256 have 1/256.
        """
        after_start = -(2 / 257) * math.log2(2 / 257) - 255 * (1 / 257) * math.log2(
            1 / 257
        )
        after_a = -2 * (2 / 258) * math.log2(2 / 258) - 254 * (1 / 258) * math.log2(
            1 / 258
        )

        example = client.get(LIMITS).json()["small_example"]

        assert example["entropies"] == pytest.approx(
            [after_start, after_a, after_a, 8.0], abs=1e-12
        )

    def test_they_are_the_numbers_the_page_prints(self) -> None:
        example = client.get(LIMITS).json()["small_example"]

        assert [round(value, 5) for value in example["entropies"]] == [
            7.99784,
            7.99572,
            7.99572,
            8.0,
        ]
        assert example["n_contexts"] == 2

    def test_a_threshold_between_them_cuts_once(self) -> None:
        example = client.get(LIMITS).json()["small_example"]

        assert example["threshold"] == 7.996
        assert example["patches_at_a_high_threshold"] == ["aab", "b"]


class TestTheLengths:
    def test_the_four_readings_of_the_sentence_and_the_corpus(self) -> None:
        rows = client.get(LIMITS).json()["lengths"]

        assert [(row["sentence"], row["corpus"]) for row in rows] == [
            (51, 763),
            (13, 197),
            (30, 129),
            (25, 263),
        ]

    def test_the_two_readings_with_a_table_report_its_size(self) -> None:
        rows = client.get(LIMITS).json()["lengths"]

        assert [row["table_entries"] for row in rows] == [256, None, None, 137]

    def test_on_text_the_model_read_the_blocks_beat_the_learned_pieces(self) -> None:
        rows = client.get(LIMITS).json()["lengths"]

        assert rows[2]["corpus"] < rows[3]["corpus"]

    def test_on_text_it_did_not_the_learned_pieces_win(self) -> None:
        rows = client.get(LIMITS).json()["lengths"]

        assert rows[2]["sentence"] > rows[3]["sentence"]


class TestThePiecesBelongToTheModel:
    def test_four_models_cut_one_sentence_four_ways(self) -> None:
        models = client.get(LIMITS).json()["models"]

        assert [row["n_patches"] for row in models] == [15, 7, 6, 6]
        assert [row["n_contexts"] for row in models] == [181, 319, 402, 272]

    def test_the_last_word_is_split_differently_by_each_of_the_first_three(
        self,
    ) -> None:
        models = client.get(LIMITS).json()["models"]

        assert models[0]["patches"][-2:] == ["Mon", "day."]
        assert models[1]["patches"][-2:] == ["Mond", "ay."]
        assert models[2]["patches"][-1] == "Monday."

    def test_dropping_half_the_corpus_changes_the_pieces_without_changing_the_count(
        self,
    ) -> None:
        models = client.get(LIMITS).json()["models"]

        assert models[2]["n_patches"] == models[3]["n_patches"]
        assert models[2]["patches"] != models[3]["patches"]

    def test_a_block_has_no_identity_two_occurrences_could_share(self) -> None:
        body = client.get(LIMITS).json()

        assert body["total_patches"] == 129
        assert body["distinct_patch_spellings"] == 77
        assert body["repeated_pieces"][0] == {"text": "The ", "n_occurrences": 12}


class TestTheTwoRules:
    def test_they_agree_about_the_opening_and_part_company_after_it(self) -> None:
        rules = client.get(LIMITS).json()["rules"]

        assert rules[0]["seen"][:3] == rules[1]["seen"][:3]
        assert rules[0]["seen"] != rules[1]["seen"]

    def test_the_rising_rule_makes_fewer_blocks_across_the_corpus(self) -> None:
        rules = client.get(LIMITS).json()["rules"]

        assert rules[0]["corpus_patches"] == 129
        assert rules[1]["corpus_patches"] == 79

    def test_on_a_flat_stretch_the_two_rules_are_opposites(self) -> None:
        """Nothing the model read is 8 bits everywhere, so nothing rises."""
        rules = client.get(LIMITS).json()["rules"]

        assert len(rules[0]["unseen"]) == 30
        assert len(rules[1]["unseen"]) == 7

    def test_the_rising_rule_is_reachable_through_the_request(self) -> None:
        body = patched(SEEN_SENTENCE, rule="relative_increase")

        assert pieces_of(body, "entropy") == [
            "The ",
            "re",
            "port was ",
            "expected on Monday.",
        ]


class TestTheRequestBounds:
    def test_an_unknown_setting_is_refused(self) -> None:
        assert client.post(PATCH, json={"text": "abc", "stride": 2}).status_code == 422

    def test_a_threshold_of_zero_is_refused(self) -> None:
        assert (
            client.post(PATCH, json={"text": "abc", "threshold": 0.0}).status_code
            == 422
        )

    def test_a_threshold_above_eight_bits_is_refused(self) -> None:
        assert (
            client.post(PATCH, json={"text": "abc", "threshold": 8.5}).status_code
            == 422
        )

    def test_a_block_size_of_zero_is_refused(self) -> None:
        assert (
            client.post(PATCH, json={"text": "abc", "patch_size": 0}).status_code == 422
        )

    def test_an_empty_text_is_refused(self) -> None:
        assert client.post(PATCH, json={"text": ""}).status_code == 422

    def test_a_text_past_the_bound_is_refused(self) -> None:
        assert client.post(PATCH, json={"text": "a" * 201}).status_code == 422

    def test_a_corpus_that_is_not_one_of_the_two_is_refused(self) -> None:
        assert (
            client.post(PATCH, json={"text": "abc", "source": "greek"}).status_code
            == 422
        )

    def test_the_ordinary_smoothing_is_reachable_and_says_nothing(self) -> None:
        body = patched(RUNNING_SENTENCE, smoothing=1.0)

        assert body["entropy"]["n_patches"] == body["n_bytes"]


CAMEL_CASE = re.compile(r"[a-z][A-Z]")
"""Two words run together, which is what an identifier looks like in prose."""

FORBIDDEN = (
    "oop_ml",
    "the library",
    "the API",
    "patch_size",
    "byte_ids",
    "n_copies",
    "Patcher",
    "Tokenizer",
    "Vocabulary",
    "Error",
)
"""Words and identifiers no string a reader can see may carry."""

SOURCE_TEXTS = (
    RUNNING_SENTENCE,
    SEEN_SENTENCE,
    FIXTURE_SENTENCE,
    ACCENTED_TEXT,
    "the cat sat",
    "aabb",
    "aab",
    *CORPUS,
)
"""The texts themselves, which are quoted back and are not the page's own words."""


def every_string(value: object) -> list[str]:
    """Every string anywhere inside a decoded response body."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in every_string(item)]
    if isinstance(value, list):
        return [text for item in value for text in every_string(item)]
    return []


class TestNothingNamesTheCodeBehindThePage:
    def test_no_endpoint_returns_an_identifier(self) -> None:
        bodies = [
            client.get(CUT_LANDING).json(),
            client.get(CORPUS_CONDITION).json(),
            client.get(LIMITS).json(),
            client.post(PATCH, json={"text": RUNNING_SENTENCE}).json(),
            client.post(PATCH, json={"text": ACCENTED_TEXT, "patch_size": 3}).json(),
            client.post(
                PATCH,
                json={
                    "text": FIXTURE_SENTENCE,
                    "source": "one sentence",
                    "rule": "relative_increase",
                },
            ).json(),
        ]

        for body in bodies:
            for text in every_string(body):
                if any(text in source for source in SOURCE_TEXTS):
                    continue
                assert CAMEL_CASE.search(text) is None, text
                for forbidden in FORBIDDEN:
                    assert forbidden not in text, text

    def test_the_labels_a_widget_prints_name_behaviour_rather_than_settings(
        self,
    ) -> None:
        body = client.post(PATCH, json={"text": RUNNING_SENTENCE}).json()

        assert body["fixed"]["label"] == "every 4 bytes"
        assert body["entropy"]["label"] == (
            "wherever the uncertainty is above the threshold"
        )
        assert body["source_label"] == "eighteen sentences about reports and costs"
