"""The pooling page's endpoints, on the two collections it carries.

Every number the page prints is pinned here. The smaller collection is the
pooling spec's own hand-built table, so its four separation margins can be
asserted against the numbers that spec pins rather than against this service;
the larger is the twenty-four documents, where the claim about what lies along
the shared direction lives.

The load-bearing tests of the module are
``TestTheSharedDirection::test_the_three_words_carrying_no_subject_lead_it``,
which is the page's headline probe and is asserted as a gap between two
populations rather than as three separate numbers;
``TestContributions::test_the_rebuilt_shares_add_up_to_the_vector``, which is
what makes a share of a pooled vector mean anything at all; and
``TestNothingIdentifierShapedReachesAReader``, which holds every string a widget
can render to the rule that a reader never meets a class, a parameter or a
module.
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.pooling_a_text import (
    ALL_METHODS,
    COOKING_DOCUMENTS,
    IDENTIFIER_SHAPED,
    MIXED_TEXT,
    REVERSED_PAIR,
    SAILING_DOCUMENTS,
    SKETCH_DOCUMENTS,
    SKETCH_TABLE,
    SKETCH_WORDS,
    THREE_TEXTS,
    TWENTY_FOUR_DOCUMENTS,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/pooling-a-text"
POOL = f"{BASE}/pool"
SEPARATION = f"{BASE}/separation"
DIRECTION = f"{BASE}/direction"
ORDER = f"{BASE}/order"
COMPANY = f"{BASE}/company"
EDGES = f"{BASE}/edges"


def rows_of(corpus: str) -> dict[str, dict[str, float]]:
    body = client.get(SEPARATION).json()
    return {row["method"]: row for row in body["rows"] if row["corpus"] == corpus}


class TestTheCollections:
    def test_they_are_the_collections_the_specs_fit(self) -> None:
        """Reproduced rather than imported, so a test holds the two together.

        ``test`` is not part of the installed package, so this is skipped
        wherever the library's own suite is not on the path.
        """
        counts = pytest.importorskip(
            "test.core.natural_language_processing.embeddings.counts.fixtures"
        )
        documents = pytest.importorskip(
            "test.core.natural_language_processing.embeddings.documents.fixtures"
        )

        assert list(COOKING_DOCUMENTS) == counts.COOKING_DOCUMENTS
        assert list(SAILING_DOCUMENTS) == counts.SAILING_DOCUMENTS
        assert TWENTY_FOUR_DOCUMENTS == counts.TWO_TOPIC_CORPUS
        assert list(SKETCH_WORDS) == documents.TOPIC_WORDS
        assert list(SKETCH_DOCUMENTS) == documents.TOPIC_CORPUS
        assert [list(row) for row in SKETCH_TABLE] == documents.TOPIC_TABLE.tolist()

    def test_the_larger_one_is_twenty_three_words_over_twenty_four_documents(
        self,
    ) -> None:
        body = client.get(SEPARATION).json()

        assert body["n_documents"] == 24
        assert body["n_words"] == 23
        assert body["counted_width"] == 23
        assert body["position_width"] == 4

    def test_the_three_probe_texts_share_only_the_words_carrying_no_subject(
        self,
    ) -> None:
        body = client.get(SEPARATION).json()

        assert body["texts"] == list(THREE_TEXTS)
        assert body["shared_between_texts"] == ["and", "the", "we"]


class TestCounting:
    """What a method that never reads a position can and cannot see."""

    def test_the_three_texts_are_all_exactly_three_quarters_alike(self) -> None:
        """Five shared tokens out of twelve squared units, three times over."""
        body = client.post(POOL, json={"method": "counts"}).json()
        similarities = [pair["similarity"] for pair in body["pairs"]]

        assert similarities == pytest.approx([0.75, 0.75, 0.75], abs=1e-15)
        assert max(similarities) - min(similarities) < 1e-15

    def test_the_counted_vector_is_as_wide_as_the_vocabulary(self) -> None:
        body = client.post(POOL, json={"method": "counts"}).json()

        assert body["dimension"] == 23
        assert body["reads_positions"] is False

    def test_two_documents_of_one_half_can_share_no_word_at_all(self) -> None:
        assert rows_of("documents")["counts"]["within_smallest"] == pytest.approx(
            0.0, abs=1e-12
        )

    def test_so_its_margin_over_the_larger_collection_is_negative(self) -> None:
        counted = rows_of("documents")

        assert counted["counts"]["margin"] == pytest.approx(-0.1667, abs=5e-5)
        assert counted["weighted-counts"]["margin"] == pytest.approx(-0.1483, abs=5e-5)


class TestAveragingPositions:
    def test_the_average_separates_where_counting_could_not(self) -> None:
        body = client.get(SEPARATION).json()
        three = {row["method"]: row for row in body["three_texts"]}

        assert three["counts"]["margin"] == pytest.approx(0.0, abs=1e-12)
        assert three["average"]["margin"] == pytest.approx(0.2106, abs=5e-5)
        assert three["average"]["sailing_to_sailing"] == pytest.approx(0.8737, abs=5e-5)

    def test_the_pooled_vector_is_four_wide_whatever_the_vocabulary(self) -> None:
        for method in ("average", "weighted-average", "smooth-inverse-frequency"):
            body = client.post(POOL, json={"method": method}).json()

            assert body["dimension"] == 4
            assert body["reads_positions"] is True

    def test_the_margin_rises_with_every_repair_in_turn(self) -> None:
        documents = rows_of("documents")

        assert documents["average"]["margin"] == pytest.approx(0.0622, abs=5e-5)
        assert documents["weighted-average"]["margin"] == pytest.approx(
            0.0819, abs=5e-5
        )
        assert documents["smooth-weights"]["margin"] == pytest.approx(0.1096, abs=5e-5)
        assert documents["smooth-inverse-frequency"]["margin"] == pytest.approx(
            0.2752, abs=5e-5
        )

    def test_the_hand_built_table_reproduces_the_four_margins_the_spec_pins(
        self,
    ) -> None:
        sketch = rows_of("sketch")

        assert sketch["average"]["margin"] == pytest.approx(0.1640, abs=5e-5)
        assert sketch["weighted-average"]["margin"] == pytest.approx(0.3871, abs=5e-5)
        assert sketch["smooth-weights"]["margin"] == pytest.approx(0.7886, abs=5e-5)
        assert sketch["smooth-inverse-frequency"]["margin"] == pytest.approx(
            1.9755, abs=5e-5
        )


class TestContributions:
    def test_the_rebuilt_shares_add_up_to_the_vector(self) -> None:
        """The one claim that makes a share mean anything, on every method."""
        for corpus, texts in (
            ("documents", list(THREE_TEXTS)),
            ("sketch", list(SKETCH_DOCUMENTS)[:3]),
        ):
            for method in ALL_METHODS:
                body = client.post(
                    POOL, json={"corpus": corpus, "method": method, "texts": texts}
                ).json()

                assert body["contribution_gap"] < 1e-15

    def test_the_shares_of_one_text_sum_to_one(self) -> None:
        body = client.post(POOL, json={"method": "average"}).json()

        for text in body["texts"]:
            total = sum(share["share"] for share in text["shares"])
            assert total == pytest.approx(1.0, abs=1e-12)

    def test_one_word_in_two_supplies_four_fifths_of_the_hand_built_average(
        self,
    ) -> None:
        body = client.post(
            POOL,
            json={
                "corpus": "sketch",
                "method": "average",
                "texts": ["the cat the dog"],
            },
        ).json()
        shares = {
            entry["word"]: entry["share"] for entry in body["texts"][0]["by_word"]
        }

        assert shares["the"] == pytest.approx(0.8000, abs=5e-5)
        assert shares["cat"] == pytest.approx(0.1000, abs=5e-5)

    def test_weighting_by_rarity_takes_that_word_down_to_about_a_half(self) -> None:
        body = client.post(
            POOL,
            json={
                "corpus": "sketch",
                "method": "weighted-average",
                "texts": ["the cat the dog"],
            },
        ).json()
        shares = {
            entry["word"]: entry["share"] for entry in body["texts"][0]["by_word"]
        }

        assert shares["the"] == pytest.approx(0.5396, abs=5e-5)
        assert shares["cat"] == pytest.approx(0.2302, abs=5e-5)

    def test_on_the_larger_collection_the_shared_words_lead_nothing(self) -> None:
        """The textbook failure does not reproduce there, and the page says so."""
        body = client.post(POOL, json={"method": "average"}).json()
        shares = {
            entry["word"]: entry["share"] for entry in body["texts"][0]["by_word"]
        }

        assert shares["crew"] == pytest.approx(0.2069, abs=5e-5)
        assert shares["and"] == pytest.approx(0.1736, abs=5e-5)
        assert shares["the"] == pytest.approx(0.1532, abs=5e-5)
        assert shares["crew"] > shares["and"] > shares["the"]


class TestTheSharedDirection:
    def test_the_three_words_carrying_no_subject_lead_it(self) -> None:
        """The page's headline probe, as a gap between two populations."""
        body = client.post(DIRECTION, json={"corpus": "documents"}).json()

        assert body["shared_words"] == ["and", "we", "the"]
        assert body["smallest_shared_cosine"] == pytest.approx(0.9032, abs=5e-5)
        assert body["largest_content_cosine"] == pytest.approx(0.7087, abs=5e-5)
        assert body["smallest_shared_cosine"] > body["largest_content_cosine"]

    def test_the_first_three_ranked_words_are_those_three(self) -> None:
        body = client.post(DIRECTION, json={"corpus": "documents"}).json()

        assert [entry["word"] for entry in body["words"][:3]] == ["and", "we", "the"]
        assert body["words"][0]["cosine"] == pytest.approx(0.9897, abs=5e-5)
        assert body["words"][1]["cosine"] == pytest.approx(0.9848, abs=5e-5)

    def test_two_thirds_of_a_plain_average_lies_along_it(self) -> None:
        body = client.post(DIRECTION, json={"corpus": "documents"}).json()

        assert body["along_mean"] == pytest.approx(0.6797, abs=5e-5)
        assert body["along_smallest"] == pytest.approx(0.5820, abs=5e-5)

    def test_they_supply_less_of_an_average_than_their_share_of_the_tokens(
        self,
    ) -> None:
        """The textbook failure, measured on the larger collection and absent."""
        body = client.post(DIRECTION, json={"corpus": "documents"}).json()

        assert body["shared_share_equal"] == pytest.approx(0.1667, abs=5e-5)
        assert body["shared_share_mean"] == pytest.approx(0.0649, abs=5e-5)
        assert body["shared_share_mean"] < body["shared_share_equal"]

    def test_and_on_the_hand_built_table_they_supply_far_more(self) -> None:
        body = client.post(DIRECTION, json={"corpus": "sketch"}).json()

        assert body["shared_share_equal"] == pytest.approx(0.5000, abs=5e-5)
        assert body["shared_share_mean"] == pytest.approx(0.8101, abs=5e-5)

    def test_those_three_words_are_the_three_shortest_positions(self) -> None:
        """Which is why the textbook account of the failure does not hold here."""
        body = client.post(DIRECTION, json={"corpus": "documents"}).json()
        shortest = sorted(body["words"], key=lambda entry: entry["length"])[:3]

        assert {entry["word"] for entry in shortest} == {"and", "the", "we"}

    def test_the_smooth_weights_barely_separate_the_words_here(self) -> None:
        body = client.post(DIRECTION, json={"corpus": "documents"}).json()

        assert body["weight_ratio"] == pytest.approx(1.5832, abs=5e-5)
        assert body["illustrated_ratio"] == pytest.approx(500.4995, abs=5e-4)
        assert body["weight_of_a_common_word"] == pytest.approx(0.001996, abs=5e-7)

    def test_the_hand_built_table_puts_something_else_first(self) -> None:
        """A collection with one shared word and two symmetric halves does not."""
        body = client.post(DIRECTION, json={"corpus": "sketch"}).json()

        assert body["shared_words"] == ["the"]
        assert body["smallest_shared_cosine"] == pytest.approx(0.4299, abs=5e-5)
        assert body["largest_content_cosine"] == pytest.approx(0.7050, abs=5e-5)
        assert body["smallest_shared_cosine"] < body["largest_content_cosine"]


class TestOrder:
    def test_every_method_reads_the_two_orderings_as_one_text(self) -> None:
        body = client.get(ORDER).json()

        assert body["first"] == REVERSED_PAIR[0]
        assert body["second"] == REVERSED_PAIR[1]
        for row in body["rows"]:
            assert row["similarity"] == pytest.approx(1.0, abs=1e-12)

    def test_counting_answers_the_identical_vector_and_averaging_all_but(
        self,
    ) -> None:
        """The only trace order leaves anywhere is the order the additions ran in."""
        body = client.get(ORDER).json()
        rows = {row["method"]: row for row in body["rows"]}

        assert rows["counts"]["identical"] is True
        assert rows["weighted-counts"]["identical"] is True
        for method in ("average", "weighted-average", "smooth-inverse-frequency"):
            assert rows[method]["largest_gap"] < 1e-16

    def test_four_words_can_be_written_in_twenty_four_orders(self) -> None:
        body = client.get(ORDER).json()

        assert body["n_orderings"] == math.factorial(4)


class TestCompany:
    def test_one_text_scored_in_two_collections_points_the_other_way(self) -> None:
        body = client.get(COMPANY).json()

        assert body["similarity_between_readings"] == pytest.approx(-0.6124, abs=5e-5)
        assert body["n_whole"] == 24
        assert body["n_half"] == 12

    def test_the_two_leading_directions_are_not_the_same_direction(self) -> None:
        body = client.get(COMPANY).json()

        assert body["component_similarity"] == pytest.approx(0.7812, abs=5e-5)

    def test_a_word_leads_one_collections_direction_and_not_the_others(self) -> None:
        body = client.get(COMPANY).json()
        words = {entry["word"]: entry for entry in body["words"]}

        assert words["and"]["in_the_whole"] == pytest.approx(0.9897, abs=5e-5)
        assert words["and"]["in_the_half"] == pytest.approx(0.7715, abs=5e-5)
        assert words["sail"]["in_the_half"] == pytest.approx(0.9031, abs=5e-5)
        assert words["sail"]["in_the_half"] > words["and"]["in_the_half"]


class TestTheEdges:
    def test_a_text_about_both_halves_is_placed_in_one_of_them(self) -> None:
        body = client.get(EDGES).json()
        mixed = {row["method"]: row for row in body["mixed"]}

        assert body["mixed_text"] == MIXED_TEXT
        assert mixed["smooth-inverse-frequency"]["to_sailing"] == pytest.approx(
            -0.1187, abs=5e-5
        )
        assert mixed["smooth-inverse-frequency"]["to_cooking"] == pytest.approx(
            0.5057, abs=5e-5
        )
        assert body["mixed_nearest_groups"] == ["cooking"] * 4

    def test_the_plain_average_puts_it_near_both_instead(self) -> None:
        body = client.get(EDGES).json()
        mixed = {row["method"]: row for row in body["mixed"]}

        assert mixed["average"]["to_sailing"] == pytest.approx(0.8862, abs=5e-5)
        assert mixed["average"]["to_cooking"] == pytest.approx(0.9202, abs=5e-5)
        assert mixed["average"]["to_sailing"] > mixed["average"]["sailing_to_cooking"]

    def test_a_text_of_unknown_words_has_no_direction_to_compare(self) -> None:
        body = client.get(EDGES).json()

        assert body["unknown_vector_length"] == 0.0
        assert "no direction" in body["unknown_refusal"]

    def test_a_collection_of_one_leaves_its_own_document_at_the_origin(self) -> None:
        body = client.get(EDGES).json()

        assert body["single_residue_length"] < 1e-16

    def test_the_sharpest_repair_costs_cohesion_inside_a_half(self) -> None:
        """Where it is worse than the plain average, reported rather than hidden."""
        body = client.get(EDGES).json()

        assert body["average_smallest_within"] == pytest.approx(0.3202, abs=5e-5)
        assert body["smooth_smallest_within"] == pytest.approx(-0.0767, abs=5e-5)
        assert body["smooth_smallest_within"] < body["average_smallest_within"]

    def test_a_pooled_vector_gives_back_four_of_a_documents_six_words(self) -> None:
        documents = rows_of("documents")

        assert documents["average"]["words_given_back"] == pytest.approx(
            4.3333, abs=5e-5
        )
        assert documents["smooth-inverse-frequency"][
            "words_given_back"
        ] == pytest.approx(4.1667, abs=5e-5)
        assert documents["counts"]["words_given_back"] is None


class TestRefusals:
    def test_a_blank_text_is_refused_in_words_a_reader_can_read(self) -> None:
        response = client.post(POOL, json={"texts": [" "]})

        assert response.status_code == 422
        assert "blank" in response.json()["detail"]

    def test_a_very_long_text_is_refused_before_anything_is_fitted(self) -> None:
        response = client.post(POOL, json={"texts": ["word " * 60]})

        assert response.status_code == 422
        assert "120 characters" in response.json()["detail"]

    def test_a_setting_this_page_does_not_offer_is_refused(self) -> None:
        assert client.post(POOL, json={"corpus": "elsewhere"}).status_code == 422
        assert client.post(POOL, json={"method": "elsewhere"}).status_code == 422
        assert client.post(POOL, json={"texts": ["a"], "depth": 2}).status_code == 422

    def test_a_pair_with_a_text_of_unknown_words_carries_its_reason(self) -> None:
        body = client.post(
            POOL, json={"texts": ["zebra kettle", "the crew and the boat"]}
        ).json()

        assert body["pairs"][0]["similarity"] is None
        assert "no direction" in body["pairs"][0]["undefined_because"]
        assert body["texts"][0]["known_words"] == 0


class TestNothingIdentifierShapedReachesAReader:
    """Every string a widget can render, held to the rule."""

    def test_no_response_carries_a_name_spelled_the_way_code_is(self) -> None:
        bodies = [
            client.get(path).json() for path in (SEPARATION, ORDER, COMPANY, EDGES)
        ]
        bodies.append(client.post(DIRECTION, json={"corpus": "documents"}).json())
        bodies.append(client.post(POOL, json={"method": "counts"}).json())
        bodies.append(
            client.post(
                POOL, json={"texts": ["zebra kettle", "the crew and the boat"]}
            ).json()
        )
        bodies.append(client.post(POOL, json={"texts": [" "]}).json())

        for body in bodies:
            for sentence in _strings(body):
                assert not IDENTIFIER_SHAPED.search(sentence), sentence

    def test_no_response_names_the_code_behind_the_page(self) -> None:
        forbidden = ("library", "the API", "oop_ml", "module", "package", "backend")
        bodies = [
            client.get(path).json() for path in (SEPARATION, ORDER, COMPANY, EDGES)
        ]

        for body in bodies:
            for sentence in _strings(body):
                for word in forbidden:
                    assert word not in sentence, sentence


def _strings(value: object) -> list[str]:
    """Every string a widget could render, which is the values and not the keys.

    A field's name is read by the code that draws a widget and never by a
    reader, so it is deliberately not held to the rule the values are.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [found for item in value.values() for found in _strings(item)]
    if isinstance(value, list):
        return [found for item in value for found in _strings(item)]
    return []
