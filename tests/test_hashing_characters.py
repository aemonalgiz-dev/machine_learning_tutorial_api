"""The hashing-characters endpoints, on the texts the page is built from.

Every number the page quotes is pinned here, so a change in the library or in
this module fails a test rather than quietly changing a sentence on the page.
Two classes carry more than their weight. One checks the residue shortcut the
separation count rests on against the sweep over all 1,114,112 codepoints it
replaces, and the last one checks that nothing these endpoints return carries an
identifier, because a widget renders these strings.
"""

from __future__ import annotations

import re

import numpy as np
from fastapi.testclient import TestClient
from oop_ml.core.natural_language_processing.tokenization.hashing.hashed_characters import (
    HASH_PRIMES,
)

from oop_ml_api.concepts.hashing_characters import (
    CORPUS,
    GREEK_TEXT,
    LANGUAGES,
    RUNNING_SENTENCE,
    UNICODE_CODEPOINTS,
    _distinct_bucket_sets,
    _sharing_step,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

HASH_TEXT = "/concepts/hashing-characters/hash-text"
COLLISIONS = "/concepts/hashing-characters/collisions"
SEPARATION = "/concepts/hashing-characters/separation"
FINGERPRINTS = "/concepts/hashing-characters/fingerprints"
LIMITS = "/concepts/hashing-characters/limits"


def hashed(text: str, **settings: int) -> dict:
    """One text through the hash, at the published settings unless told otherwise."""
    response = client.post(HASH_TEXT, json={"text": text, **settings})
    assert response.status_code == 200, response.text
    return response.json()


class TestTheArithmeticOfOneCharacter:
    def test_the_multipliers_are_the_published_ones(self) -> None:
        body = hashed(RUNNING_SENTENCE)

        assert body["multipliers"] == [31, 43, 59, 61, 73, 97, 103, 113]

    def test_the_letter_a_is_ninety_eight_times_each_multiplier(self) -> None:
        body = hashed("a")
        buckets = body["characters"][0]["bucket_ids"]

        assert buckets == [98 * multiplier for multiplier in HASH_PRIMES[:8]]
        assert buckets == [3038, 4214, 5782, 5978, 7154, 9506, 10094, 11074]

    def test_the_first_characters_of_the_sentence(self) -> None:
        characters = hashed(RUNNING_SENTENCE)["characters"]

        assert characters[0]["character"] == "D"
        assert characters[0]["codepoint"] == 68
        assert characters[0]["bucket_ids"] == [
            2139,
            2967,
            4071,
            4209,
            5037,
            6693,
            7107,
            7797,
        ]
        assert characters[3]["display"] == "␣"
        assert characters[3]["codepoint"] == 32

    def test_a_reduction_happens_once_the_product_passes_the_width(self) -> None:
        # 98 * 113 is 11,074 and stays put; at 4,096 buckets it wraps twice.
        narrow = hashed("a", n_buckets=4096)["characters"][0]["bucket_ids"]

        assert narrow == [(98 * multiplier) % 4096 for multiplier in HASH_PRIMES[:8]]
        assert narrow[-1] == 11074 % 4096


class TestTheRunningSentence:
    def test_one_position_per_character_and_eight_numbers_each(self) -> None:
        body = hashed(RUNNING_SENTENCE)

        assert body["n_characters"] == 51
        assert body["n_bytes"] == 51
        assert body["n_ids"] == 408

    def test_the_table_is_the_bucket_count_whatever_the_text(self) -> None:
        body = hashed(RUNNING_SENTENCE)

        assert body["table_rows"] == 16384
        assert body["table_numbers"] == 12582912
        assert body["slice_width"] == 96

    def test_nothing_in_it_collides_at_the_published_width(self) -> None:
        body = hashed(RUNNING_SENTENCE)

        assert body["clashes"] == []
        assert body["n_distinct_characters"] == 24
        assert body["n_distinct_bucket_sets"] == 24

    def test_the_characters_are_kept_so_the_text_returns(self) -> None:
        body = hashed(RUNNING_SENTENCE)

        assert body["rebuilt"] == RUNNING_SENTENCE
        assert body["rebuilt_exactly"] is True

    def test_a_narrow_width_makes_two_of_its_characters_agree(self) -> None:
        body = hashed(RUNNING_SENTENCE, n_buckets=64)

        assert body["n_distinct_bucket_sets"] == 23
        assert [group["characters"] for group in body["clashes"]] == [[".", "n"]]

    def test_a_narrower_width_still_makes_three_pairs_agree(self) -> None:
        body = hashed(RUNNING_SENTENCE, n_buckets=32)

        assert body["n_distinct_bucket_sets"] == 21
        assert [group["characters"] for group in body["clashes"]] == [
            ["D", "d"],
            [".", "n"],
            ["A", "a"],
        ]

    def test_a_script_the_english_corpus_never_met_costs_nothing_extra(self) -> None:
        body = hashed(GREEK_TEXT)

        assert body["n_characters"] == 21
        assert body["n_bytes"] == 38
        assert body["n_ids"] == 168
        assert body["rebuilt_exactly"] is True
        assert body["clashes"] == []


class TestWhoElseHasTheseBuckets:
    def test_a_capital_a_and_one_ideograph_agree_on_all_eight(self) -> None:
        body = client.post(COLLISIONS, json={"character": "A"}).json()

        assert body["codepoint"] == 65
        assert body["codepoint_label"] == "U+0041"
        assert body["bucket_ids"] == [2046, 2838, 3894, 4026, 4818, 6402, 6798, 7458]
        assert body["named"][0]["codepoint"] == 16449
        assert body["named"][0]["codepoint_label"] == "U+4041"
        assert body["named"][0]["unicode_name"] == "CJK UNIFIED IDEOGRAPH-4041"

    def test_every_character_shares_its_set_with_sixty_seven_others(self) -> None:
        for character in ("A", "a", "z", ".", "Η"):
            body = client.post(COLLISIONS, json={"character": character}).json()

            assert body["sharing_step"] == 16384
            assert body["n_sharers"] == 67
            assert body["every_sharer_agrees"] is True

    def test_the_crowd_shrinks_as_the_width_grows(self) -> None:
        counts = {}
        for n_buckets in (256, 4096, 16384, 65536):
            body = client.post(
                COLLISIONS, json={"character": "A", "n_buckets": n_buckets}
            ).json()
            counts[n_buckets] = body["n_sharers"]

        assert counts == {256: 4351, 4096: 271, 16384: 67, 65536: 16}

    def test_more_than_one_character_is_refused(self) -> None:
        response = client.post(COLLISIONS, json={"character": "ab"})

        assert response.status_code == 422


class TestHowManyKindsCanBeToldApart:
    def test_eight_hashes_separate_no_more_than_one_does(self) -> None:
        rows = {row["n_buckets"]: row for row in client.get(SEPARATION).json()["rows"]}
        published = rows[16384]

        assert published["ceiling"] == 16384
        assert published["shares_a_factor"] is False
        assert [count["distinct_sets"] for count in published["counts"]] == [
            16384,
            16384,
            16384,
            16384,
        ]

    def test_a_width_the_first_multiplier_divides_is_where_a_second_helps(self) -> None:
        rows = {row["n_buckets"]: row for row in client.get(SEPARATION).json()["rows"]}
        narrow = rows[93]

        assert narrow["shares_a_factor"] is True
        assert [count["distinct_sets"] for count in narrow["counts"]] == [3, 93, 93, 93]

    def test_the_bucket_count_is_a_ceiling_nothing_passes(self) -> None:
        for row in client.get(SEPARATION).json()["rows"]:
            for count in row["counts"]:
                assert count["distinct_sets"] <= row["ceiling"]
            assert row["counts"][-1]["reaches_the_ceiling"] is True

    def test_a_second_hash_is_short_at_no_width_at_all(self) -> None:
        body = client.get(SEPARATION).json()

        assert body["widths_tried"] == 1999
        assert body["widths_where_one_falls_short"] == 64
        assert body["widths_where_two_fall_short"] == 0
        assert body["first_ten_short_widths"][:3] == [31, 62, 93]
        assert body["every_short_width_is_a_multiple"] is True
        assert body["first_multiplier"] == 31
        assert body["multiples_of_the_first_multiplier"] == 64


class TestTheResidueShortcutAgreesWithTheSweep:
    """The counting identity, checked against the thing it stands in for."""

    @staticmethod
    def sweep(n_buckets: int, n_hash_functions: int) -> int:
        """Count distinct bucket sets over every codepoint, the slow way."""
        shifted = np.arange(UNICODE_CODEPOINTS, dtype=np.int64) + 1
        multipliers = np.array(HASH_PRIMES[:n_hash_functions], dtype=np.int64)
        sets = (shifted[:, None] * multipliers[None, :]) % n_buckets
        return len(np.unique(sets, axis=0))

    def test_the_residue_count_matches_the_whole_codepoint_space(self) -> None:
        for n_buckets, n_hashes in ((93, 1), (93, 2), (1333, 1), (16384, 8)):
            assert _distinct_bucket_sets(n_buckets, n_hashes) == self.sweep(
                n_buckets, n_hashes
            )

    def test_the_step_and_the_count_are_the_same_number(self) -> None:
        for n_buckets in (93, 256, 961, 1333, 2048, 3999, 16384):
            for n_hashes in (1, 2, 4, 8):
                assert _sharing_step(n_buckets, n_hashes) == _distinct_bucket_sets(
                    n_buckets, n_hashes
                )


class TestAWordAsASetOfBuckets:
    def test_cat_and_cats_share_two_buckets_of_the_three_and_four(self) -> None:
        body = client.post(
            FINGERPRINTS, json={"words": ["cat", "cats"], "n_buckets": 8192}
        ).json()

        assert body["words"][0]["trigrams"] == ["_ca", "cat", "at_"]
        assert body["words"][0]["bucket_ids"] == [2002, 5665, 6052]
        assert body["words"][1]["trigrams"] == ["_ca", "cat", "ats", "ts_"]
        assert body["words"][1]["bucket_ids"] == [1676, 2002, 5665, 6072]
        assert body["pairs"][0]["shared"] == 2
        assert body["pairs"][0]["identical"] is False

    def test_two_unrelated_words_share_nothing(self) -> None:
        body = client.post(
            FINGERPRINTS, json={"words": ["cat", "dog"], "n_buckets": 8192}
        ).json()

        assert body["pairs"][0]["shared"] == 0

    def test_a_word_and_its_prefixed_form_share_most_of_it(self) -> None:
        body = client.post(
            FINGERPRINTS,
            json={
                "words": ["analysis", "reanalysis", "cost", "costs"],
                "n_buckets": 8192,
            },
        ).json()
        shares = {
            (pair["first"], pair["second"]): pair["shared"] for pair in body["pairs"]
        }

        assert shares[("analysis", "reanalysis")] == 7
        assert shares[("cost", "costs")] == 3
        assert shares[("analysis", "cost")] == 0

    def test_a_word_of_n_characters_has_n_pieces(self) -> None:
        body = client.post(
            FINGERPRINTS, json={"words": ["a", "cat", "analysis"], "n_buckets": 8192}
        ).json()

        assert [word["n_trigrams"] for word in body["words"]] == [1, 3, 8]
        assert body["words"][0]["trigrams"] == ["_a_"]

    def test_a_repeat_leaves_the_fingerprint_where_it_was(self) -> None:
        body = client.post(
            FINGERPRINTS, json={"words": ["aaa", "aaaa"], "n_buckets": 262144}
        ).json()

        assert body["words"][0]["n_trigrams"] == 3
        assert body["words"][1]["n_trigrams"] == 4
        assert body["words"][1]["n_active"] == 3
        assert body["pairs"][0]["identical"] is True

    def test_too_many_words_are_refused(self) -> None:
        response = client.post(FINGERPRINTS, json={"words": ["a"] * 9})

        assert response.status_code == 422


class TestWhatTheClosingPartQuotes:
    def test_the_table_a_row_per_character_would_need(self) -> None:
        tables = {row["label"]: row for row in client.get(LIMITS).json()["tables"]}

        assert tables["one row per character Unicode has room for"]["rows"] == 1114112
        assert (
            tables["one row per character Unicode has room for"]["numbers"] == 855638016
        )
        assert (
            tables["one row per bucket, at the published width"]["numbers"] == 12582912
        )
        assert tables["one row per byte value"]["numbers"] == 196608

    def test_the_two_ratios_the_page_leans_on(self) -> None:
        body = client.get(LIMITS).json()

        assert body["codepoint_table_over_hashed"] == 68.0
        assert body["hashed_over_byte_table"] == 64.0

    def test_one_position_per_character_whatever_the_writing_system(self) -> None:
        rows = {row["language"]: row for row in client.get(LIMITS).json()["lengths"]}

        assert rows["Chinese"]["hashed_positions"] == 6
        assert rows["Chinese"]["n_bytes"] == 18
        assert rows["Chinese"]["character_stand_ins"] == 6
        assert rows["Greek"]["hashed_positions"] == 21
        assert rows["Greek"]["n_bytes"] == 38
        assert rows["English"]["hashed_positions"] == 24
        assert rows["English"]["n_bytes"] == 24

    def test_the_crowd_and_the_cost_move_together(self) -> None:
        trades = {
            row["n_buckets"]: row for row in client.get(LIMITS).json()["width_trades"]
        }

        assert trades[256]["characters_per_bucket_set"] == 4352.0
        assert trades[16384]["characters_per_bucket_set"] == 68.0
        assert trades[262144]["characters_per_bucket_set"] == 4.25
        assert trades[262144]["numbers"] == 201326592

    def test_the_three_character_pieces_spread_but_not_evenly(self) -> None:
        spread = {row["n_buckets"]: row for row in client.get(LIMITS).json()["spread"]}

        assert client.get(LIMITS).json()["trigrams_counted"] == 19683
        assert spread[8192]["buckets_used"] == 8122
        assert spread[8192]["heaviest_bucket"] == 5
        assert spread[8192]["mean_load"] == 2.423
        assert spread[32768]["buckets_used"] == 13635

    def test_a_piece_of_the_sentence_shares_a_bucket_with_two_others(self) -> None:
        clash = client.get(LIMITS).json()["trigram_clash"]

        assert clash["bucket_id"] == 3245
        assert clash["trigrams"] == ["aaa", "nal", "yme"]
        assert "nal" in RUNNING_SENTENCE

    def test_whole_words_collide_once_the_width_is_narrow(self) -> None:
        body = client.get(LIMITS).json()

        assert body["narrow_buckets"] == 512
        assert body["narrow_words_tried"] == 17576
        assert body["narrow_distinct_fingerprints"] == 17570
        assert body["narrow_clashes"] == 6
        assert body["narrow_examples"][0] == {"first": "fek", "second": "hab"}

    def test_a_repeated_piece_is_invisible_and_a_new_one_is_not(self) -> None:
        cases = client.get(LIMITS).json()["repeated_letters"]

        assert cases[0]["first_trigrams"] == ["_aa", "aaa", "aa_"]
        assert cases[0]["second_trigrams"] == ["_aa", "aaa", "aaa", "aa_"]
        assert cases[0]["identical"] is True
        assert cases[1]["second"] == "aaaaa"
        assert cases[1]["identical"] is True
        # The boundary: one more a after b makes a piece the shorter word
        # did not have, so the two fingerprints do differ.
        assert cases[2]["second_trigrams"] == ["_ba", "baa", "aaa", "aa_"]
        assert cases[2]["identical"] is False

    def test_the_sentence_becomes_seven_positions_with_the_stops_attached(self) -> None:
        body = client.get(LIMITS).json()

        assert body["sentence_words"] == [
            "Dr.",
            "Alvarez",
            "didn't",
            "expect",
            "the",
            "low-cost",
            "re-analysis.",
        ]
        assert body["sentence_positions"] == 7
        assert body["sentence_active_buckets"] == 45
        assert body["sentence_merged_tokens"] == 25
        assert body["sentence_characters"] == 51


class TestTheBoundsOnARequest:
    def test_a_text_longer_than_the_limit_is_refused(self) -> None:
        response = client.post(HASH_TEXT, json={"text": "a" * 201})

        assert response.status_code == 422

    def test_an_empty_text_is_refused(self) -> None:
        response = client.post(HASH_TEXT, json={"text": ""})

        assert response.status_code == 422

    def test_one_bucket_is_refused(self) -> None:
        response = client.post(HASH_TEXT, json={"text": "a", "n_buckets": 1})

        assert response.status_code == 422

    def test_more_hashes_than_there_are_multipliers_is_refused(self) -> None:
        response = client.post(HASH_TEXT, json={"text": "a", "n_hash_functions": 33})

        assert response.status_code == 422

    def test_an_unexpected_field_is_refused(self) -> None:
        response = client.post(HASH_TEXT, json={"text": "a", "scheme": "word"})

        assert response.status_code == 422

    def test_a_run_of_codepoints_that_is_not_text_is_refused_at_the_door(self) -> None:
        response = client.post(
            HASH_TEXT,
            content=b'{"text": "\\ud800"}',
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


CAMEL_CASE = re.compile(r"[a-z][A-Z]")
"""Two words run together, which is what an identifier looks like in prose."""

FORBIDDEN = (
    "oop_ml",
    "the library",
    "the API",
    "n_buckets",
    "n_hash_functions",
    "codepoint",
    "Tokenizer",
    "Vocabulary",
    "Error",
)
"""Words and identifiers no string a reader can see may carry."""

SOURCE_TEXTS = frozenset(
    {RUNNING_SENTENCE, GREEK_TEXT}
    | set(CORPUS)
    | {sentence for _, sentence in LANGUAGES}
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
            client.get(SEPARATION).json(),
            client.get(LIMITS).json(),
            client.post(HASH_TEXT, json={"text": RUNNING_SENTENCE}).json(),
            client.post(HASH_TEXT, json={"text": GREEK_TEXT}).json(),
            client.post(COLLISIONS, json={"character": "A"}).json(),
            client.post(
                FINGERPRINTS, json={"words": ["cat", "cats"], "n_buckets": 8192}
            ).json(),
        ]

        for body in bodies:
            for text in every_string(body):
                if text in SOURCE_TEXTS or any(
                    source.startswith(text) for source in SOURCE_TEXTS
                ):
                    continue
                assert CAMEL_CASE.search(text) is None, text
                for forbidden in FORBIDDEN:
                    assert forbidden not in text, text

    def test_a_refusal_a_caller_can_reach_carries_no_identifier(self) -> None:
        sentence = client.post(COLLISIONS, json={"character": "ab"}).json()["detail"]

        assert CAMEL_CASE.search(sentence) is None
        for forbidden in FORBIDDEN:
            assert forbidden not in sentence
