"""The FastText page's endpoints, on the two-list corpus the page carries.

Every number the page prints is pinned here. The expensive endpoints are cached
in the module under test, so the first test that reaches one pays for it and the
rest are free.

Two claims are asserted exactly rather than to a tolerance. A fit whose pieces
cannot exist agrees with a fit that has no pieces at all, and the gap is
literally zero rather than nearly zero, so ``== 0.0`` is the assertion. And a
word none of whose pieces reached a trained row has the zero vector, whose
length is exactly zero and which has no direction at all.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.fasttext import (
    MONEY_WORDS,
    TWO_LIST_CORPUS,
    VERB_WORDS,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

PIECES = "/concepts/fasttext/pieces"
FIT = "/concepts/fasttext/fit"
UNSEEN = "/concepts/fasttext/unseen"
REDUCTION = "/concepts/fasttext/reduction"
BUCKETS = "/concepts/fasttext/buckets"
COMPARISON = "/concepts/fasttext/comparison"
HELD_OUT = "/concepts/fasttext/held-out"


@pytest.fixture(scope="module")
def unseen() -> dict:
    return client.get(UNSEEN).json()


@pytest.fixture(scope="module")
def buckets() -> dict:
    return client.get(BUCKETS).json()


@pytest.fixture(scope="module")
def comparison() -> dict:
    return client.get(COMPARISON).json()


def probe(body: dict, word: str) -> dict:
    return next(entry for entry in body["probes"] if entry["word"] == word)


class TestTheCorpus:
    def test_it_is_the_corpus_the_specs_fit(self) -> None:
        """Reproduced rather than imported, so a test holds the two together.

        ``test`` is not part of the installed package, so this is skipped
        wherever the library's own suite is not on the path.
        """
        corpora = pytest.importorskip(
            "test.core.natural_language_processing.embeddings.prediction.corpora"
        )

        assert TWO_LIST_CORPUS == corpora.TWO_TOPIC_CORPUS
        assert VERB_WORDS == corpora.VERB_WORDS
        assert MONEY_WORDS == corpora.FINANCE_WORDS
        assert corpora.UNSEEN_VERB not in {
            word for sentence in TWO_LIST_CORPUS for word in sentence.split()
        }

    def test_two_hundred_sentences_of_thirty_one_words(self) -> None:
        body = client.post(FIT, json={"epochs": 1}).json()

        assert body["n_sentences"] == 200
        assert body["n_occurrences"] == 1311
        assert body["n_words"] == 31


class TestPieces:
    def test_the_published_example_at_one_length(self) -> None:
        """Five pieces of the wrapped word, which is the paper's own example."""
        body = client.post(
            PIECES,
            json={"word": "where", "minimum_length": 3, "maximum_length": 3},
        ).json()

        assert [entry["piece"] for entry in body["pieces"]] == [
            "<wh",
            "whe",
            "her",
            "ere",
            "re>",
        ]

    def test_the_wrapped_word_is_never_one_of_its_own_pieces(self) -> None:
        body = client.post(PIECES, json={"word": "at"}).json()

        assert body["wrapped"] == "<at>"
        assert [entry["piece"] for entry in body["pieces"]] == ["<at", "at>"]

    def test_a_word_shorter_than_the_shortest_piece_has_none(self) -> None:
        body = client.post(PIECES, json={"word": "a"}).json()

        assert body["pieces"] == []
        assert body["n_pieces"] == 0
        assert body["shared_buckets"] == 0

    def test_the_missing_form_has_twenty_two_pieces_and_sixteen_were_seen(self) -> None:
        body = client.post(PIECES, json={"word": "playing"}).json()

        assert body["seen"] is False
        assert body["n_pieces"] == 22
        assert body["n_pieces_owned"] == 16
        assert body["shared_buckets"] == 17

    def test_a_piece_names_the_words_that_own_it(self) -> None:
        body = client.post(PIECES, json={"word": "playing"}).json()
        by_piece = {entry["piece"]: entry for entry in body["pieces"]}

        assert by_piece["<pl"]["bucket"] == 1127
        assert set(by_piece["<pl"]["owners"]) == {"play", "played", "plays", "player"}
        assert set(by_piece["ing"]["owners"]) == {
            "staying",
            "saying",
            "walking",
            "talking",
        }
        assert by_piece["laying"]["owners"] == []

    def test_the_marker_tells_a_prefix_from_the_same_letters_inside(self) -> None:
        """``her`` on its own and ``her`` inside ``where`` are the same piece;
        the pieces at the word's two ends are not."""
        alone = client.post(
            PIECES, json={"word": "her", "minimum_length": 3, "maximum_length": 3}
        ).json()
        inside = client.post(
            PIECES, json={"word": "where", "minimum_length": 3, "maximum_length": 3}
        ).json()
        alone_buckets = {entry["piece"]: entry["bucket"] for entry in alone["pieces"]}
        inside_buckets = {entry["piece"]: entry["bucket"] for entry in inside["pieces"]}

        assert alone_buckets["her"] == inside_buckets["her"] == 1420
        assert alone_buckets["<he"] != alone_buckets["her"]

    def test_a_smaller_table_puts_the_pieces_in_fewer_rows(self) -> None:
        body = client.post(PIECES, json={"word": "playing", "n_buckets": 8}).json()

        assert all(0 <= entry["bucket"] < 8 for entry in body["pieces"])


class TestFit:
    def test_the_reference_fit_separates_the_two_lists(self) -> None:
        body = client.post(FIT, json={}).json()

        assert body["within_topic"] == pytest.approx(0.9954, abs=5e-4)
        assert body["across_topic"] == pytest.approx(0.1231, abs=5e-4)
        assert body["total_pairs"] == 19519

    def test_the_missing_form_is_answered_from_its_spelling(self) -> None:
        body = client.post(FIT, json={"word": "playing"}).json()

        assert body["word_seen"] is False
        assert body["word_refusal"] is None
        assert body["mean_to_verbs"] == pytest.approx(0.9970, abs=5e-4)
        assert body["mean_to_money"] == pytest.approx(0.1003, abs=5e-4)
        assert [entry["word"] for entry in body["neighbours"]][:4] == [
            "play",
            "plays",
            "player",
            "played",
        ]
        assert all(entry["topic"] == "verb" for entry in body["neighbours"])

    def test_the_corpus_owns_more_pieces_than_it_has_words(self) -> None:
        body = client.post(FIT, json={"epochs": 1}).json()

        assert body["n_distinct_pieces"] == 279
        assert body["n_distinct_rows"] == 260
        assert body["n_shared_pieces"] == 19
        assert body["word_row_numbers"] == 372
        assert body["bucket_row_numbers"] == 24000
        assert body["rows_that_moved"] == 260

    def test_the_objective_falls_and_the_rate_falls_with_it(self) -> None:
        body = client.post(FIT, json={}).json()

        losses = [entry["mean_loss"] for entry in body["epochs"]]
        rates = [entry["learning_rate"] for entry in body["epochs"]]
        assert losses[0] == pytest.approx(2.5560, abs=5e-4)
        assert losses[-1] == pytest.approx(2.0840, abs=5e-4)
        assert rates == sorted(rates, reverse=True)

    def test_a_word_sharing_no_row_is_refused_rather_than_placed(self) -> None:
        body = client.post(FIT, json={"epochs": 1, "word": "zap"}).json()

        assert body["shared_buckets"] == 0
        assert body["neighbours"] == []
        assert body["word_refusal"] is not None
        assert "direction" in body["word_refusal"]

    def test_a_pieceless_fit_is_accepted_and_answers_nothing_unseen(self) -> None:
        body = client.post(
            FIT,
            json={"epochs": 1, "minimum_length": 9, "maximum_length": 10},
        ).json()

        assert body["n_distinct_pieces"] == 0
        assert body["rows_that_moved"] == 0
        assert body["word_refusal"] is not None


class TestRefusals:
    def test_a_longest_below_the_shortest_is_refused(self) -> None:
        response = client.post(
            PIECES, json={"word": "playing", "minimum_length": 5, "maximum_length": 3}
        )

        assert response.status_code == 422
        assert "shorter" in response.json()["detail"]

    def test_an_empty_word_is_refused(self) -> None:
        response = client.post(PIECES, json={"word": ""})

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(FIT, json={"buckets": 10})

        assert response.status_code == 422

    def test_more_passes_than_the_limit_are_refused(self) -> None:
        response = client.post(FIT, json={"epochs": 99})

        assert response.status_code == 422
        assert "6" in response.json()["detail"]

    def test_a_table_larger_than_the_limit_is_refused(self) -> None:
        response = client.post(FIT, json={"n_buckets": 500_000})

        assert response.status_code == 422


class TestUnseen:
    def test_a_fit_with_one_row_per_word_has_no_answer_at_all(
        self, unseen: dict
    ) -> None:
        assert "playing" in unseen["plain_refusal"]
        assert "31" in unseen["plain_refusal"]

    def test_the_two_ranges_do_not_overlap(self, unseen: dict) -> None:
        assert unseen["mean_to_verbs"] == pytest.approx(0.9970, abs=5e-4)
        assert unseen["smallest_to_verbs"] == pytest.approx(0.9939, abs=5e-4)
        assert unseen["mean_to_money"] == pytest.approx(0.1003, abs=5e-4)
        assert unseen["largest_to_money"] == pytest.approx(0.1106, abs=5e-4)
        assert unseen["smallest_to_verbs"] > unseen["largest_to_money"]

    def test_eight_corpus_words_share_a_piece_with_the_missing_form(
        self, unseen: dict
    ) -> None:
        sharers = {entry["word"]: entry["pieces"] for entry in unseen["sharers"]}

        assert len(sharers) == 8
        assert set(sharers) == {
            "saying",
            "staying",
            "play",
            "played",
            "player",
            "plays",
            "talking",
            "walking",
        }
        assert len(sharers["saying"]) == 10
        assert sharers["talking"] == ["ing", "ing>", "ng>"]

    def test_a_seen_word_is_nearly_all_spelling_here(self, unseen: dict) -> None:
        shares = {entry["word"]: entry for entry in unseen["spelling_shares"]}

        assert shares["played"]["cosine"] == pytest.approx(0.9997, abs=5e-4)
        assert shares["played"]["length_share"] == pytest.approx(0.9827, abs=5e-4)
        assert shares["bond"]["length_share"] == pytest.approx(0.9043, abs=5e-4)

    def test_a_word_spelled_like_a_money_word_lands_among_them(
        self, unseen: dict
    ) -> None:
        banking = probe(unseen, "banking")

        assert banking["mean_to_money"] == pytest.approx(0.9632, abs=5e-4)
        assert banking["mean_to_verbs"] == pytest.approx(0.3718, abs=5e-4)
        assert [entry["word"] for entry in banking["nearest"]] == [
            "trade",
            "yield",
            "debt",
        ]

    def test_a_word_placed_by_one_shared_row(self, unseen: dict) -> None:
        """Fourteen pieces, one of which happened to hash where a piece of
        ``market`` hashed, and that one row is the whole answer."""
        queue = probe(unseen, "queue")

        assert queue["n_pieces"] == 14
        assert queue["shared_buckets"] == 1
        assert queue["shared_rows"][0]["piece"] == "queue>"
        assert queue["shared_rows"][0]["corpus_pieces"] == ["arket"]
        assert queue["shared_rows"][0]["owners"] == ["market"]
        assert queue["mean_to_money"] == pytest.approx(0.9919, abs=5e-4)
        assert queue["length"] == pytest.approx(0.1893, abs=5e-4)

    def test_a_word_reaching_no_trained_row_has_no_direction(
        self, unseen: dict
    ) -> None:
        shareless = probe(unseen, "zap")

        assert shareless["shared_rows"] == []
        assert shareless["length"] == 0.0
        assert shareless["nearest"] == []
        assert shareless["refusal"] is not None


class TestReduction:
    def test_no_word_is_long_enough_to_have_a_piece(
        self,
    ) -> None:
        body = client.get(REDUCTION).json()

        assert body["minimum_length"] == 50
        assert body["longest_wrapped_word"] == 9

    def test_every_pairing_agrees_to_a_gap_of_exactly_zero(self) -> None:
        """Not nearly zero. The rows the pieces would write to start at zero and
        consume nothing from the draw, so the two walks are the same walk."""
        body = client.get(REDUCTION).json()

        assert len(body["rows"]) == 4
        for row in body["rows"]:
            assert row["table_gap"] == 0.0
            assert row["output_gap"] == 0.0
            assert row["loss_gap"] == 0.0
            assert row["largest_bucket_value"] == 0.0

    def test_the_two_columns_of_losses_are_the_same_numbers(self) -> None:
        body = client.get(REDUCTION).json()

        assert body["losses_with_pieces"] == body["losses_without"]
        assert len(body["losses_with_pieces"]) == 2


class TestBuckets:
    def test_two_hundred_and_seventy_nine_pieces_for_thirty_one_words(
        self, buckets: dict
    ) -> None:
        assert buckets["n_words"] == 31
        assert buckets["n_distinct_pieces"] == 279
        assert buckets["pieces_per_word"] == pytest.approx(13.6129, abs=5e-4)
        assert buckets["fewest_pieces"] == 5
        assert buckets["most_pieces"] == 22

    def test_the_sharing_falls_as_the_table_grows(self, buckets: dict) -> None:
        rows = {entry["n_buckets"]: entry for entry in buckets["sharing"]}

        assert rows[2000]["n_rows_used"] == 260
        assert rows[2000]["n_shared"] == 19
        assert rows[20_000]["n_shared"] == 2
        assert rows[2_000_000]["n_shared"] == 0
        assert rows[50]["n_rows_used"] == 50
        assert [entry["n_shared"] for entry in buckets["sharing"]] == sorted(
            (entry["n_shared"] for entry in buckets["sharing"]), reverse=True
        )

    def test_nineteen_rows_carry_two_pieces_each(self, buckets: dict) -> None:
        assert len(buckets["colliding"]) == 19
        by_bucket = {entry["bucket"]: entry for entry in buckets["colliding"]}

        assert by_bucket[1713]["pieces"] == ["<play", "alk"]
        assert "play" in by_bucket[1713]["owners"][0]
        assert by_bucket[897]["pieces"] == ["<sto", "lay>"]

    def test_a_table_far_too_small_costs_the_separation(self, buckets: dict) -> None:
        fits = {entry["n_buckets"]: entry for entry in buckets["fits"]}

        assert fits[10]["n_rows_used"] == 10
        assert fits[10]["gap"] == pytest.approx(0.4417, abs=5e-4)
        assert fits[10]["missing_to_money"] == pytest.approx(0.5497, abs=5e-4)
        assert fits[2000]["gap"] == pytest.approx(0.8723, abs=5e-4)
        assert fits[2000]["missing_to_money"] == pytest.approx(0.1003, abs=5e-4)
        assert fits[10]["gap"] < fits[2000]["gap"]

    def test_how_short_a_piece_may_be_changes_what_is_shared(
        self, buckets: dict
    ) -> None:
        rows = {
            (entry["minimum_length"], entry["maximum_length"]): entry
            for entry in buckets["lengths"]
        }

        assert rows[(3, 3)]["n_distinct_pieces"] == 84
        assert rows[(3, 3)]["pieces_per_word"] == pytest.approx(5.0, abs=1e-9)
        assert rows[(3, 6)]["n_distinct_pieces"] == 279
        assert rows[(4, 6)]["n_distinct_pieces"] == 195
        assert rows[(3, 3)]["missing_to_money"] == pytest.approx(0.1427, abs=5e-4)
        assert rows[(3, 6)]["missing_to_money"] == pytest.approx(0.1003, abs=5e-4)
        assert rows[(4, 6)]["missing_to_money"] == pytest.approx(0.2302, abs=5e-4)
        assert rows[(4, 6)]["missing_to_verbs"] == pytest.approx(0.9947, abs=5e-4)


class TestComparison:
    def test_the_pieces_cost_sixty_five_times_the_numbers(
        self, comparison: dict
    ) -> None:
        assert comparison["numbers_without_pieces"] == 372
        assert comparison["numbers_with_pieces"] == 24372
        assert comparison["rows_that_moved"] == 260
        assert comparison["composed_table_numbers"] == 372

    def test_a_correction_reaches_every_part_of_the_word(
        self, comparison: dict
    ) -> None:
        assert comparison["parts_per_word"] == pytest.approx(14.6129, abs=5e-4)
        assert comparison["fewest_parts"] == 6
        assert comparison["most_parts"] == 23

    def test_the_pieces_separate_the_lists_better_at_both_steps(
        self, comparison: dict
    ) -> None:
        rows = {
            (entry["method"], entry["learning_rate"]): entry
            for entry in comparison["separation"]
        }

        assert rows[("pieces", 0.025)]["gap"] == pytest.approx(0.8723, abs=5e-4)
        assert rows[("words", 0.025)]["gap"] == pytest.approx(0.7033, abs=5e-4)
        assert rows[("pieces", 0.05)]["gap"] == pytest.approx(0.8814, abs=5e-4)
        assert rows[("words", 0.05)]["gap"] == pytest.approx(0.8266, abs=5e-4)

    def test_the_first_pass_already_costs_less_with_the_pieces(
        self, comparison: dict
    ) -> None:
        assert comparison["losses_with_pieces"][0] == pytest.approx(2.5560, abs=5e-4)
        assert comparison["losses_without"][0] == pytest.approx(3.7370, abs=5e-4)

    def test_the_two_fits_name_different_neighbours_for_the_same_word(
        self, comparison: dict
    ) -> None:
        walked = next(
            entry for entry in comparison["words"] if entry["word"] == "walked"
        )

        assert [entry["word"] for entry in walked["with_pieces"]][:2] == [
            "talked",
            "walk",
        ]
        assert walked["shared"] == ["walking"]
        assert all(entry["topic"] == "verb" for entry in walked["with_pieces"])


class TestHeldOut:
    def test_this_corpus_has_no_rare_words_of_its_own(self) -> None:
        body = client.get(HELD_OUT).json()

        assert body["fewest_occurrences"] == 17
        assert body["most_occurrences"] == 73
        assert body["words_seen_once"] == 0

    def test_dropping_every_sentence_holding_one_word(self) -> None:
        body = client.get(HELD_OUT).json()

        assert body["n_sentences_dropped"] == 30
        assert body["n_occurrences_dropped"] == 34
        assert body["n_words_left"] == 30
        assert "walking" in body["plain_refusal"]
        assert "30" in body["plain_refusal"]

    def test_the_removed_word_is_still_placed_among_its_own_list(self) -> None:
        body = client.get(HELD_OUT).json()

        assert body["mean_to_verbs"] == pytest.approx(0.9953, abs=5e-4)
        assert body["mean_to_money"] == pytest.approx(0.0497, abs=5e-4)
        assert body["nearest"][0]["word"] == "walked"
        assert all(entry["topic"] == "verb" for entry in body["nearest"])
        assert body["shared_buckets"] == 20
        assert body["n_pieces"] == 22


IDENTIFIER = re.compile(r"\b(?:[a-z]+_[a-z_]+|[A-Z][a-z]+[A-Z][A-Za-z]*)\b")
"""A snake-cased or camel-cased name, which no string a reader sees may hold."""

FORBIDDEN = ("oop_ml", "the library", "the API", "Error")
"""Words that would credit the code behind the page rather than describe it."""


def _strings(value: object) -> list[str]:
    """Every string anywhere inside a decoded response body."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for entry in value.values() for text in _strings(entry)]
    if isinstance(value, list):
        return [text for entry in value for text in _strings(entry)]
    return []


class TestNothingAReaderSeesNamesTheCode:
    """Every string a widget renders is page text, so it is checked like text.

    The strings that reach a reader here are words of the corpus, pieces of
    their spellings, the page's own words for the two choices, and the
    refusals, so this asserts exactly that and fails if a new field starts
    carrying prose.
    """

    def test_the_words_that_cross_the_wire_hold_no_identifier(self) -> None:
        bodies = [
            client.post(PIECES, json={"word": "playing"}).json(),
            client.post(FIT, json={"epochs": 1}).json(),
            client.get(UNSEEN).json(),
            client.get(REDUCTION).json(),
            client.get(BUCKETS).json(),
            client.get(COMPARISON).json(),
            client.get(HELD_OUT).json(),
        ]
        for body in bodies:
            for text in _strings(body):
                assert not IDENTIFIER.search(text), text
                for forbidden in FORBIDDEN:
                    assert forbidden not in text

    def test_every_refusal_a_widget_shows_is_written_in_plain_words(
        self, unseen: dict
    ) -> None:
        """The interactive fit reaches one more of them, for any word a reader
        types whose pieces all land in rows nothing wrote to."""
        typed = client.post(FIT, json={"epochs": 1, "word": "zap"}).json()
        shown = [unseen["plain_refusal"], typed["word_refusal"]] + [
            entry["refusal"] for entry in unseen["probes"] if entry["refusal"]
        ]

        assert len(shown) == 3
        for sentence in shown:
            assert not IDENTIFIER.search(sentence), sentence
            assert "token" not in sentence
            for forbidden in FORBIDDEN:
                assert forbidden not in sentence
