"""The moving-a-vocabulary page's endpoints, on the corpora the page carries.

Every number the page quotes in prose is pinned here, so a change to the library
that moved one of them would fail a test rather than quietly rewrite the page.
The refusal tests are the other half: the page may not show a parameter name,
and the last class asserts that nothing identifier-shaped survives the
rewriting.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/moving-a-vocabulary"
WORKED = f"{BASE}/worked"
DOMAINS = f"{BASE}/domains"
TRANSFER = f"{BASE}/transfer"


def _row(rows: list[dict], token: str) -> dict:
    for candidate in rows:
        if candidate["target_token"] == token:
            return candidate
    raise AssertionError(f"no mapping for {token!r}")


class TestWorked:
    def test_a_smaller_vocabulary_over_the_same_words_needs_no_rebuilding(self) -> None:
        """Every token of the three-merge fit is a token of the ten-merge fit."""
        body = client.get(WORKED).json()
        summary = body["larger_to_smaller_summary"]

        assert summary["n_target_tokens"] == 15
        assert summary["n_rebuilt"] == 0
        assert summary["n_copied"] == 14
        assert summary["n_stand_in_route"] == 1

    def test_growing_the_vocabulary_copies_fourteen_and_rebuilds_seven(self) -> None:
        body = client.get(WORKED).json()
        summary = body["smaller_to_larger_summary"]

        assert summary["n_target_tokens"] == 22
        assert summary["n_copied"] == 14
        assert summary["n_rebuilt"] == 7
        assert summary["n_nothing"] == 0

    def test_a_whole_word_is_rebuilt_from_the_pieces_that_spell_it(self) -> None:
        body = client.get(WORKED).json()

        assert _row(body["smaller_to_larger"], "low</w>")["source_tokens"] == [
            "lo",
            "w</w>",
        ]
        assert _row(body["smaller_to_larger"], "ewest</w>")["source_tokens"] == [
            "e",
            "w",
            "est</w>",
        ]

    def test_an_inner_piece_comes_back_claiming_a_word_ended(self) -> None:
        """``ew`` never ends a word; the text ``ew`` is read as one that does."""
        body = client.get(WORKED).json()
        row = _row(body["smaller_to_larger"], "ew")

        assert row["decoded"] == "ew"
        assert row["source_tokens"] == ["e", "w</w>"]
        assert row["gained_a_marker"] is True
        assert body["smaller_to_larger_summary"]["n_gained_a_marker"] == 1

    def test_a_shared_inner_piece_is_copied_rather_than_read_again(self) -> None:
        """``lo`` is in both, so the copy rule takes it before decoding can."""
        body = client.get(WORKED).json()
        row = _row(body["smaller_to_larger"], "lo")

        assert row["route"] == "copied"
        assert row["source_tokens"] == ["lo"]

    def test_a_piece_the_source_cannot_spell_reaches_the_stand_in(self) -> None:
        """No word of the four ends in ``o``, so the text ``lo`` is half unreadable."""
        body = client.get(WORKED).json()
        row = _row(body["narrow"], "lo")

        assert row["route"] == "rebuilt"
        assert row["source_tokens"] == ["l", "[UNK]"]
        assert row["uses_stand_in"] is True
        assert body["narrow_summary"]["n_using_stand_in"] == 1

    def test_across_conventions_a_continuation_lands_and_five_letters_do_not(
        self,
    ) -> None:
        body = client.get(WORKED).json()
        rows = body["word_piece"]["rows"]

        assert _row(rows, "##est")["source_tokens"] == ["est</w>"]
        assert _row(rows, "low")["source_tokens"] == ["low</w>"]
        assert body["word_piece"]["summary"]["n_copied"] == 4
        assert body["word_piece"]["summary"]["n_rebuilt"] == 17
        assert body["word_piece"]["summary"]["n_stand_in_only"] == 5
        for token in ("##d", "##e", "##i", "##o", "##s"):
            assert _row(rows, token)["source_tokens"] == ["[UNK]"]

    def test_ten_tokens_of_a_byte_level_target_name_no_text_at_all(self) -> None:
        body = client.get(WORKED).json()

        assert body["byte_level_n_tokens"] == 266
        assert body["byte_level_n_nothing"] == 10
        assert body["byte_level_space_symbol"] in body["byte_level_nothing"]
        assert body["byte_level_mapped_example"]["n_source_ids"] > 0


class TestDomains:
    def test_the_two_fits_run_out_at_different_sizes(self) -> None:
        body = client.get(DOMAINS).json()

        assert body["old_tokens"] == 137
        assert body["new_tokens"] == 125
        assert body["old_merges"] == 85
        assert body["new_merges"] == 79

    def test_fifty_seven_tokens_survive_by_spelling_and_sixty_seven_are_rebuilt(
        self,
    ) -> None:
        body = client.get(DOMAINS).json()
        summary = body["summary"]

        assert summary["n_target_tokens"] == 125
        assert summary["n_copied"] == 57
        assert summary["n_copied_multi_character"] == 14
        assert summary["n_rebuilt"] == 67
        assert summary["n_stand_in_route"] == 1
        assert summary["n_nothing"] == 0
        assert round(summary["mean_pieces_when_rebuilt"], 2) == 2.63

    def test_twelve_rebuilt_tokens_reach_the_stand_in_and_two_reach_nothing_else(
        self,
    ) -> None:
        body = client.get(DOMAINS).json()

        assert body["summary"]["n_using_stand_in"] == 12
        assert body["summary"]["n_stand_in_only"] == 2
        alone = [
            row["target_token"]
            for row in body["stand_in_rows"]
            if row["n_source_ids"] == 1
        ]
        assert alone == ["k</w>", "x</w>"]

    def test_twenty_five_inner_pieces_come_back_as_whole_words(self) -> None:
        body = client.get(DOMAINS).json()

        assert body["summary"]["n_inner_rebuilt"] == 35
        assert body["summary"]["n_gained_a_marker"] == 25
        assert _row(body["marker_rows"], "ea")["source_tokens"] == ["e", "a</w>"]
        assert _row(body["marker_rows"], "he")["source_tokens"] == ["he</w>"]

    def test_the_new_corpus_shortens_and_the_old_one_lengthens(self) -> None:
        body = client.get(DOMAINS).json()
        lengths = {row["label"]: row for row in body["lengths"]}

        assert lengths["the new corpus"]["old_pieces"] == 428
        assert lengths["the new corpus"]["new_pieces"] == 276
        assert lengths["the old corpus"]["old_pieces"] == 263
        assert lengths["the old corpus"]["new_pieces"] == 505

    def test_one_sentence_of_each_domain_moves_the_opposite_way(self) -> None:
        body = client.get(DOMAINS).json()
        lengths = {row["label"]: row for row in body["lengths"]}

        assert lengths["one new sentence"]["old_pieces"] == 34
        assert lengths["one new sentence"]["new_pieces"] == 19
        assert lengths["one old sentence"]["old_pieces"] == 25
        assert lengths["one old sentence"]["new_pieces"] == 37

    def test_the_old_sentence_loses_four_more_characters_after_the_move(self) -> None:
        """The apostrophe, both hyphens and the x, none of which the new
        corpus uses in that position."""
        body = client.get(DOMAINS).json()
        stand_ins = [
            sum(1 for piece in reading["pieces"] if piece == "[UNK]")
            for reading in body["old_sentence_readings"]
        ]

        assert stand_ins == [1, 5]

    def test_the_table_sizes_are_rows_times_the_stated_width(self) -> None:
        body = client.get(DOMAINS).json()
        tables = {table["label"]: table for table in body["tables"]}

        assert tables["the old table"]["n_numbers"] == 35_072
        assert tables["the new table"]["n_numbers"] == 32_000
        assert tables["a small language model"]["n_numbers"] == 24_576_000
        assert body["seeded_numbers"] == 32_000
        assert body["stand_in_only_numbers"] == 512

    def test_the_sweep_copies_a_plateau_while_the_rebuilding_grows(self) -> None:
        body = client.get(DOMAINS).json()
        rows = {row["asked"]: row for row in body["sweep"]}

        assert rows[52]["n_copied"] == 47
        assert rows[52]["n_rebuilt"] == 4
        assert rows[52]["new_corpus_pieces"] == 472
        assert rows[125]["n_copied"] == 57
        assert rows[125]["n_rebuilt"] == 67
        assert rows[125]["new_corpus_pieces"] == 276

    def test_a_size_neither_corpus_can_fill_learns_the_same_vocabulary(self) -> None:
        body = client.get(DOMAINS).json()
        rows = {row["asked"]: row for row in body["sweep"]}

        assert rows[137]["learned"] == 125
        assert rows[137]["new_corpus_pieces"] == rows[125]["new_corpus_pieces"]

    def test_a_surviving_piece_sits_inside_different_words_on_each_side(self) -> None:
        body = client.get(DOMAINS).json()
        pieces = {piece["piece"]: piece for piece in body["shared_pieces"]}

        assert pieces["lo"]["old_words"] == ["low"]
        assert pieces["lo"]["new_words"] == ["hollow."]
        assert pieces["lo"]["shared_words"] == []
        assert pieces["an</w>"]["old_words"] == ["reran", "than"]
        assert pieces["an</w>"]["new_words"] == ["an"]

    def test_one_position_holds_two_different_tokens(self) -> None:
        body = client.get(DOMAINS).json()
        ids = {row["token_id"]: row for row in body["colliding_ids"]}

        assert ids[0]["old_token"] == ids[0]["new_token"] == "[UNK]"
        assert ids[60]["old_token"] == "as</w>"
        assert ids[60]["new_token"] == "dou"


class TestTransfer:
    def test_the_default_request_is_the_domain_move(self) -> None:
        body = client.post(TRANSFER, json={}).json()

        assert body["source_learned"] == 137
        assert body["target_learned"] == 125
        assert body["summary"]["n_copied"] == 57
        assert body["source_corpus_pieces"] == 428
        assert body["target_corpus_pieces"] == 276

    def test_it_reads_one_text_with_each_vocabulary(self) -> None:
        body = client.post(TRANSFER, json={"text": "The warm dough."}).json()

        assert body["source_reading"]["n_pieces"] > body["target_reading"]["n_pieces"]
        assert "".join(body["target_reading"]["pieces"]).replace(
            "</w>", " "
        ).strip() == ("The warm dough.")

    def test_moving_a_vocabulary_into_itself_copies_everything(self) -> None:
        body = client.post(
            TRANSFER, json={"source_corpus": "kitchen", "target_corpus": "kitchen"}
        ).json()

        assert body["summary"]["n_rebuilt"] == 0
        assert body["summary"]["n_copied"] == body["summary"]["n_target_tokens"] - 1

    def test_a_size_below_the_alphabet_is_refused_in_the_page_s_own_words(self) -> None:
        response = client.post(TRANSFER, json={"source_size": 20})

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "51 symbols" in detail
        assert "smallest workable size is 52" in detail

    def test_a_size_beyond_the_bound_is_refused_before_anything_is_fitted(self) -> None:
        assert client.post(TRANSFER, json={"target_size": 4000}).status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        assert client.post(TRANSFER, json={"vocab_size": 60}).status_code == 422

    def test_an_unknown_corpus_is_refused(self) -> None:
        assert client.post(TRANSFER, json={"source_corpus": "poems"}).status_code == 422


IDENTIFIER = re.compile(r"[a-z]+_[a-z_]+|[A-Z][a-z]+[A-Z][A-Za-z]*")


class TestNothingIdentifierShapedReachesThePage:
    """A refusal is page text, so it may not carry a parameter or a class name."""

    def test_the_refusals_the_playground_can_provoke_are_in_plain_words(self) -> None:
        for request in (
            {"source_size": 20},
            {"target_size": 20},
            {"source_corpus": "four_words", "source_size": 12, "target_size": 20},
        ):
            response = client.post(TRANSFER, json=request)
            assert response.status_code == 422
            detail = response.json()["detail"]
            if isinstance(detail, str):
                assert not IDENTIFIER.search(detail), detail

    def test_every_label_the_page_prints_is_plain(self) -> None:
        body = client.get(DOMAINS).json()
        labels = [row["label"] for row in body["lengths"]]
        labels += [table["label"] for table in body["tables"]]
        labels += [reading["label"] for reading in body["old_sentence_readings"]]
        labels.append(client.get(WORKED).json()["word_piece"]["label"])

        for label in labels:
            assert not IDENTIFIER.search(label), label
