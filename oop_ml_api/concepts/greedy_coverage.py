"""The greedy coverage page's endpoints.

Five of them, one per family of widgets rather than one per number.
``/choosing`` walks the four-word corpus one chosen piece at a time, carrying
which positions each pick claimed and what its rivals were worth, so the page
can show the objective working. ``/units`` scores every candidate under both
units of covering on two corpora, since the unit is the page's central
measurement and the library counts in one of them by definition. ``/matched``
fits choosing-by-coverage and merging at the same vocabulary sizes on the
eighteen-sentence corpus and reports what the corpus and the held-out sentence
cost under each. ``/limits`` carries the closing part: the falling gains, an
exhaustive search for the best pair of pieces, the rows spent on words seen
once, and what capping the piece length does. ``/fit`` is the playground.

Two calculations here are this module's own, and both are stated rather than
hidden. Counting a covering in characters is not something the library does,
because it counts positions of the spelled word; the scorer here reuses the
library's own occurrence finder and weights each claimed position by the
characters underneath it. And the exhaustive search over pairs of pieces is
this module's, because the library chooses greedily by definition and has no
reason to enumerate sets.

One fit per corpus is done at the ceiling and every smaller vocabulary is read
off it as a prefix, because a greedy run of ``k`` steps is the first ``k`` steps
of a longer run. That is worth several seconds a request, and the test module
pins the prefix reading against genuine fits so the shortcut cannot drift.

Every refusal is rewritten by ``_in_plain_words`` before it leaves here, since
the page must never show a parameter name, and the test module asserts that no
identifier survives the rewriting.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from enum import StrEnum
from functools import cache

from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.greedy_coverage import (
    ChosenPiece,
    ChosenPieces,
    GreedyCoverageTokenizer,
    cover,
    newly_covered_positions,
)
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    SpelledWord,
    alphabet_of,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/greedy-coverage", tags=["greedy-coverage"])


END_OF_WORD = "</w>"
"""The marker every fit on this page uses, written out once."""

UNKNOWN = "[UNK]"
"""What stands in for a symbol the corpus never used."""

MAX_VOCABULARY_SIZE = 300
"""The largest vocabulary a request may ask for.

Both page corpora run out of coverable positions well below this, so the bound
never truncates anything the page shows; it exists so a crafted request cannot
turn a teaching endpoint into a long fit.
"""

MAX_TEXT_LENGTH = 200
"""The longest text the playground may ask to read."""

MAX_PIECE_LENGTH = 16
"""The longest candidate, in symbols, everywhere except the length sweep."""


# --------------------------------------------------------------------------
# The corpora the page works with.
# --------------------------------------------------------------------------

FOUR_WORD_COUNTS: dict[str, int] = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
"""Four words with their counts, small enough that every choice is checkable.

The corpus the original subword paper worked its own example on, and the one
the from-scratch trainer's spec is pinned against, so every number the page
quotes from it is a number a test already holds.
"""

FOUR_WORD_CORPUS: list[str] = [
    " ".join(word for word, count in FOUR_WORD_COUNTS.items() for _ in range(count))
]

SENTENCE_CORPUS: list[str] = [
    "The report was expected on Monday.",
    "The costs were lower than the first estimate.",
    "We reviewed the results and rewrote the summary.",
    "Dr. Bell asked for the analysis of the samples.",
    "A second analysis agreed with the first analysis.",
    "The low readings weren't expected.",
    "The team rechecked the costing and the totals.",
    "Every report carries the date and the analyst's name.",
    "The revised estimate was lower again.",
    "Nobody expected the samples to arrive early.",
    "The cost of the analysis was the reason.",
    "The analysts reran the tests on Tuesday.",
    "The lowest cost was the reason the report was late.",
    "The reviewers expected a lower estimate.",
    "The high-cost option was dropped.",
    "The size of the August batch was fixed.",
    "Dr. Bell rewrote the costing and the report.",
    "The analysis was expected to cost less.",
]
"""Eighteen sentences of ordinary English, none of them the running sentence.

The same corpus the merging pages carry, so a vocabulary chosen by coverage and
one grown by merging can be compared on identical text.
"""

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page in this section carries."""

CONTROL_SENTENCE = "The report was expected on Monday."
"""One sentence of the corpus, for the round trip that is exact."""

UNIT_COUNTS: dict[str, int] = {
    "test": 3,
    "rest": 3,
    "best": 3,
    "stem": 5,
    "stop": 5,
    "stub": 5,
}
"""Six words on which the two units of covering choose different first pieces.

Three words ending in the same three letters against three words opening with
the same two, weighted so that the shorter piece wins on positions and the
longer one wins once the end-of-word mark is counted as a character of its own.
"""

TRAP_COUNTS: dict[str, int] = {"costing": 3, "costly": 3}
"""Two words on which taking the best piece first forfeits the best pair."""

SWEEP_SIZES = (52, 55, 60, 65, 70, 75, 80, 90, 100, 110, 121)
"""The vocabulary sizes both methods are fitted at."""

COMPARED_AT = 100
"""The size at which the page prints the two vocabularies side by side."""

LENGTH_CAPS = (3, 4, 5, 6, 8, 16)
"""The longest piece allowed, swept."""

ONCE_SIZES = (80, 100, 121)
"""The sizes at which rows spent on words seen once are counted."""

TRAP_BUDGETS = (1, 2, 3)
"""The set sizes the exhaustive search is run at."""


class CorpusChoice(StrEnum):
    """Which of the page's two corpora a request wants fitted."""

    FOUR_WORDS = "four_words"
    SENTENCES = "sentences"


CORPORA: dict[CorpusChoice, list[str]] = {
    CorpusChoice.FOUR_WORDS: FOUR_WORD_CORPUS,
    CorpusChoice.SENTENCES: SENTENCE_CORPUS,
}


# --------------------------------------------------------------------------
# Turning a refusal into the page's own words.
# --------------------------------------------------------------------------

PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("vocabulary_size=", "a vocabulary of "),
    ("max_piece_length", "the longest piece allowed"),
    ("n_pieces", "the number of pieces asked for"),
    ("pre_tokenizer", "the rule that finds the words"),
    ("the unknown token", "one token for anything unspellable"),
    ("GreedyCoverageTokenizer", "this scheme"),
)
"""The substitutions that take an identifier out of a message a reader sees."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence with the identifiers taken out of it.

    The messages are written to be read and most of them already are, so this
    substitutes the phrases that name a parameter and leaves the rest alone. A
    test asserts that nothing identifier-shaped survives, so a message that
    grows one later fails there rather than reaching the page.
    """
    sentence = str(refusal)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence


# --------------------------------------------------------------------------
# Small helpers shared by the endpoints.
# --------------------------------------------------------------------------


def _symbols_of(word: str) -> tuple[str, ...]:
    """A word as characters, the last one carrying the end-of-word marker."""
    return (*word[:-1], word[-1] + END_OF_WORD)


def _word_counts(corpus: Sequence[str]) -> dict[str, int]:
    counts = Corpus.of(list(corpus)).word_counts(WhitespacePreTokenizer())
    return {word_count.word: word_count.count for word_count in counts}


def _spelled(counts: dict[str, int]) -> list[SpelledWord]:
    return [SpelledWord(_symbols_of(word), count) for word, count in counts.items()]


def _candidates(words: Sequence[SpelledWord]) -> dict[str, tuple[str, ...]]:
    """Every run of two or more symbols the corpus contains, spelling to span."""
    found: dict[str, tuple[str, ...]] = {}
    for word in words:
        symbols = word.symbols
        for start in range(len(symbols)):
            longest = min(MAX_PIECE_LENGTH, len(symbols) - start)
            for length in range(2, longest + 1):
                span = symbols[start : start + length]
                found["".join(span)] = span
    return found


def _characters_in(symbol: str) -> int:
    """How many characters one position stands for, counting the mark as one."""
    if symbol.endswith(END_OF_WORD):
        return len(symbol) - len(END_OF_WORD) + 1
    return len(symbol)


def _weight(symbol: str, in_characters: bool) -> int:
    return _characters_in(symbol) if in_characters else 1


def _bare_coverage(
    words: Sequence[SpelledWord], span: Sequence[str], in_characters: bool
) -> int:
    """What one candidate covers on an untouched corpus, weighted by count."""
    total = 0
    for word in words:
        claimed = newly_covered_positions(
            word.symbols, [False] * len(word.symbols), span
        )
        total += word.count * sum(
            _weight(word.symbols[position], in_characters) for position in claimed
        )
    return total


def _occurrences(words: Sequence[SpelledWord], span: Sequence[str]) -> int:
    """How many times a candidate occurs, weighted by how often its word does."""
    total = 0
    for word in words:
        claimed = newly_covered_positions(
            word.symbols, [False] * len(word.symbols), span
        )
        total += word.count * (len(claimed) // len(span))
    return total


def _greedy_choices(
    words: Sequence[SpelledWord], budget: int, in_characters: bool
) -> list[tuple[str, int]]:
    """The greedy run under either unit, as pieces paired with what they won.

    Under positions this reproduces the library's own trainer, which the test
    module asserts; under characters it is the same rule with the other weight,
    which is the comparison the page exists to make.
    """
    candidates = _candidates(words)
    covered = [[False] * len(word.symbols) for word in words]
    chosen: list[tuple[str, int]] = []
    remaining = sorted(candidates)
    while len(chosen) < budget and remaining:
        best_piece: str | None = None
        best_score = 0
        for piece in remaining:
            score = 0
            for word, flags in zip(words, covered, strict=True):
                for position in newly_covered_positions(
                    word.symbols, flags, candidates[piece]
                ):
                    score += word.count * _weight(word.symbols[position], in_characters)
            if score > best_score:
                best_piece, best_score = piece, score
        if best_piece is None:
            break
        for word, flags in zip(words, covered, strict=True):
            for position in newly_covered_positions(
                word.symbols, flags, candidates[best_piece]
            ):
                flags[position] = True
        chosen.append((best_piece, best_score))
        remaining.remove(best_piece)
    return chosen


def _set_coverage(
    words: Sequence[SpelledWord],
    candidates: dict[str, tuple[str, ...]],
    pieces: Sequence[str],
) -> int:
    """The most positions a set of pieces can cover, best over every order."""
    best = 0
    for order in itertools.permutations(pieces):
        total = 0
        for word in words:
            flags = [False] * len(word.symbols)
            for piece in order:
                for position in newly_covered_positions(
                    word.symbols, flags, candidates[piece]
                ):
                    flags[position] = True
            total += word.count * sum(flags)
        best = max(best, total)
    return best


def _as_chosen(
    words: Sequence[SpelledWord],
    candidates: dict[str, tuple[str, ...]],
    pieces: Sequence[str],
) -> ChosenPieces:
    """A set of pieces in the shape the cutter takes, ranked as given."""
    return ChosenPieces(
        [
            ChosenPiece(
                candidates[piece],
                max(1, _bare_coverage(words, candidates[piece], False)),
            )
            for piece in pieces
        ]
    )


class _PrefixFit:
    """A fitted vocabulary read off a longer one, and how it cuts a text.

    A greedy run of ``k`` steps is the first ``k`` steps of any longer run, so
    one fit at the ceiling answers every smaller size. This holds the alphabet
    and a prefix of the chosen pieces, builds the vocabulary the trainer would
    have built, and cuts a text the way the trainer's own encoder does.
    """

    def __init__(self, alphabet: Sequence[str], chosen: ChosenPieces) -> None:
        self.alphabet = list(alphabet)
        self.chosen = chosen
        self.vocabulary = Vocabulary(
            [UNKNOWN, *alphabet, *chosen.pieces], unknown_token=UNKNOWN
        )

    @property
    def n_tokens(self) -> int:
        return self.vocabulary.n_tokens

    def pieces_of(self, text: str) -> list[str]:
        """The pieces a text becomes, with anything unspellable stood in for."""
        pieces: list[str] = []
        for word in WhitespacePreTokenizer().split(text).texts:
            for piece in cover(_symbols_of(word), self.chosen).pieces:
                pieces.append(piece if piece in self.vocabulary else UNKNOWN)
        return pieces

    def decoded(self, pieces: Sequence[str]) -> str:
        """The pieces glued back, with every marker turned into a space."""
        return "".join(pieces).replace(END_OF_WORD, " ").rstrip(" ")

    def corpus_pieces(self, counts: dict[str, int]) -> int:
        return sum(
            len(cover(_symbols_of(word), self.chosen).pieces) * count
            for word, count in counts.items()
        )

    def covered_positions(self, counts: dict[str, int]) -> int:
        return sum(
            cover(_symbols_of(word), self.chosen).n_covered_symbols * count
            for word, count in counts.items()
        )


@cache
def _full_fit(choice: CorpusChoice) -> GreedyCoverageTokenizer:
    """One fit per corpus, at a size no corpus here can fill."""
    return GreedyCoverageTokenizer(vocabulary_size=MAX_VOCABULARY_SIZE).fit(
        CORPORA[choice]
    )


def _prefix_at(choice: CorpusChoice, vocabulary_size: int) -> _PrefixFit:
    """The vocabulary a fit of this size would have chosen."""
    fitted = _full_fit(choice)
    alphabet = alphabet_of(_spelled(_word_counts(CORPORA[choice])))
    wanted = max(0, vocabulary_size - len(alphabet) - 1)
    return _PrefixFit(alphabet, ChosenPieces(list(fitted.chosen_pieces)[:wanted]))


def _merged_pieces(model: BytePairEncoding) -> list[str]:
    """The rows a merging fit learned, which is one per merge it made."""
    return [merge.merged for merge in model.merges]


def _plain(piece: str) -> str:
    return piece.removesuffix(END_OF_WORD)


def _is_bare(piece: str, alphabet: Sequence[str]) -> bool:
    """Whether a piece is one symbol of the alphabet rather than a chosen row."""
    return piece in set(alphabet)


class PairCount(BaseModel):
    """One adjacent pair of symbols and how often it occurs, weighted.

    This is what a merging scheme ranks by, carried here so the page can put
    the two orderings beside each other without quoting a figure of its own.
    """

    pair: str
    count: int


def _top_pairs(words: Sequence[SpelledWord], many: int) -> list[PairCount]:
    counted: dict[tuple[str, str], int] = {}
    for word in words:
        for left, right in zip(word.symbols, word.symbols[1:], strict=False):
            counted[(left, right)] = counted.get((left, right), 0) + word.count
    ordered = sorted(counted.items(), key=lambda entry: (-entry[1], entry[0]))
    return [
        PairCount(pair=f"{left} + {right}", count=count)
        for (left, right), count in ordered[:many]
    ]


# --------------------------------------------------------------------------
# /choosing: the four-word corpus, one piece at a time.
# --------------------------------------------------------------------------


class SpelledEntry(BaseModel):
    """One distinct word, how often it occurs, and the positions it holds."""

    word: str
    count: int
    symbols: list[str]


class Rival(BaseModel):
    """A candidate that was scored and, in every case but one, not taken."""

    piece: str
    n_symbols: int
    occurrences: int
    coverage: int


class ChoiceStep(BaseModel):
    """One turn of the rule: what was still bare, and what was taken.

    ``covered`` carries one list of flags per word, in the order of
    ``word_counts``, describing the corpus *after* this piece was claimed.
    ``rivals`` is what the leading candidates were worth *before* it.
    """

    step: int
    piece: str | None
    coverage: int | None
    rivals: list[Rival]
    covered: list[list[bool]]
    covered_positions: int
    corpus_pieces: int


class WordReading(BaseModel):
    """One word cut by the finished vocabulary."""

    word: str
    seen_in_training: bool
    pieces: list[str]
    n_covered_symbols: int


class ChoosingResponse(BaseModel):
    """The four-word corpus walked from bare symbols to a finished vocabulary."""

    corpus: list[str]
    word_counts: list[SpelledEntry]
    alphabet: list[str]
    total_occurrences: int
    total_positions: int
    steps: list[ChoiceStep]
    tokens: list[str]
    stops_at: int
    readings: list[WordReading]
    top_pairs: list[PairCount]


@cache
def _choosing() -> ChoosingResponse:
    counts = FOUR_WORD_COUNTS
    words = _spelled(counts)
    candidates = _candidates(words)
    fitted = _full_fit(CorpusChoice.FOUR_WORDS)
    every = list(fitted.chosen_pieces)

    steps: list[ChoiceStep] = []
    for step in range(len(every) + 1):
        so_far = ChosenPieces(every[:step])
        covered: list[list[bool]] = []
        for word in words:
            flags = [False] * len(word.symbols)
            for chosen in so_far:
                for position in newly_covered_positions(
                    word.symbols, flags, chosen.symbols
                ):
                    flags[position] = True
            covered.append(flags)

        rivals: list[Rival] = []
        if step < len(every):
            taken = {chosen.piece for chosen in so_far}
            scored: list[tuple[int, str]] = []
            for piece, span in candidates.items():
                if piece in taken:
                    continue
                value = sum(
                    word.count * len(newly_covered_positions(word.symbols, flags, span))
                    for word, flags in zip(words, covered, strict=True)
                )
                if value > 0:
                    scored.append((value, piece))
            scored.sort(key=lambda pair: (-pair[0], pair[1]))
            rivals = [
                Rival(
                    piece=piece,
                    n_symbols=len(candidates[piece]),
                    occurrences=_occurrences(words, candidates[piece]),
                    coverage=value,
                )
                for value, piece in scored[:6]
            ]

        steps.append(
            ChoiceStep(
                step=step,
                piece=every[step].piece if step < len(every) else None,
                coverage=every[step].coverage if step < len(every) else None,
                rivals=rivals,
                covered=covered,
                covered_positions=sum(
                    word.count * sum(flags)
                    for word, flags in zip(words, covered, strict=True)
                ),
                corpus_pieces=sum(
                    len(cover(word.symbols, so_far).pieces) * word.count
                    for word in words
                ),
            )
        )

    return ChoosingResponse(
        corpus=FOUR_WORD_CORPUS,
        word_counts=[
            SpelledEntry(word=word, count=count, symbols=list(_symbols_of(word)))
            for word, count in counts.items()
        ],
        alphabet=alphabet_of(words),
        total_occurrences=sum(counts.values()),
        total_positions=sum(word.count * len(word.symbols) for word in words),
        steps=steps,
        tokens=list(fitted.vocabulary),
        stops_at=fitted.vocabulary.n_tokens,
        top_pairs=_top_pairs(words, 4),
        readings=[
            WordReading(
                word=word,
                seen_in_training=word in counts,
                pieces=list(fitted.best_segmentation(word).pieces),
                n_covered_symbols=fitted.best_segmentation(word).n_covered_symbols,
            )
            for word in (
                "newest",
                "widest",
                "low",
                "lower",
                "lowest",
                "wide",
                "unwidest",
            )
        ],
    )


@router.get("/choosing", response_model=ChoosingResponse)
def choosing() -> ChoosingResponse:
    """The four-word corpus, one chosen piece at a time."""
    return _choosing()


# --------------------------------------------------------------------------
# /units: the same corpus scored two ways.
# --------------------------------------------------------------------------


class UnitScore(BaseModel):
    """One candidate under both units of covering."""

    piece: str
    n_symbols: int
    occurrences: int
    in_positions: int
    in_characters: int


class UnitCorpus(BaseModel):
    """One corpus, its leading candidates, and what each unit chooses."""

    key: str
    label: str
    word_counts: list[SpelledEntry]
    total_positions: int
    total_characters: int
    candidates: list[UnitScore]
    winner_by_positions: str
    winner_positions_score: int
    winner_by_characters: str
    winner_characters_score: int
    same_winner: bool
    run_by_positions: list[UnitScore]
    run_by_characters: list[UnitScore]


def _unit_corpus(
    key: str, label: str, counts: dict[str, int], budget: int
) -> UnitCorpus:
    words = _spelled(counts)
    candidates = _candidates(words)
    scored = {
        piece: (
            _bare_coverage(words, span, False),
            _bare_coverage(words, span, True),
        )
        for piece, span in candidates.items()
    }
    by_positions = sorted(scored, key=lambda piece: (-scored[piece][0], piece))
    by_characters = sorted(scored, key=lambda piece: (-scored[piece][1], piece))
    shown = list(dict.fromkeys(by_positions[:6] + by_characters[:6]))
    shown.sort(key=lambda piece: (-scored[piece][0], piece))

    def entry(piece: str, positions: int, characters: int) -> UnitScore:
        return UnitScore(
            piece=piece,
            n_symbols=len(candidates[piece]),
            occurrences=_occurrences(words, candidates[piece]),
            in_positions=positions,
            in_characters=characters,
        )

    return UnitCorpus(
        key=key,
        label=label,
        word_counts=[
            SpelledEntry(word=word, count=count, symbols=list(_symbols_of(word)))
            for word, count in counts.items()
        ],
        total_positions=sum(word.count * len(word.symbols) for word in words),
        total_characters=sum(
            word.count * sum(_characters_in(symbol) for symbol in word.symbols)
            for word in words
        ),
        candidates=[entry(piece, *scored[piece]) for piece in shown],
        winner_by_positions=by_positions[0],
        winner_positions_score=scored[by_positions[0]][0],
        winner_by_characters=by_characters[0],
        winner_characters_score=scored[by_characters[0]][1],
        same_winner=by_positions[0] == by_characters[0],
        run_by_positions=[
            entry(piece, score, scored[piece][1])
            for piece, score in _greedy_choices(words, budget, False)
        ],
        run_by_characters=[
            entry(piece, scored[piece][0], score)
            for piece, score in _greedy_choices(words, budget, True)
        ],
    )


class UnitsResponse(BaseModel):
    """Both corpora, each scored under both units."""

    corpora: list[UnitCorpus]


@cache
def _units() -> UnitsResponse:
    return UnitsResponse(
        corpora=[
            _unit_corpus("four_words", "Four words, repeated", FOUR_WORD_COUNTS, 4),
            _unit_corpus("six_words", "Three endings, three openings", UNIT_COUNTS, 3),
        ]
    )


@router.get("/units", response_model=UnitsResponse)
def units() -> UnitsResponse:
    """Every candidate scored in positions and in characters, on two corpora."""
    return _units()


# --------------------------------------------------------------------------
# /matched: coverage against merging at the same sizes.
# --------------------------------------------------------------------------


class MatchedRow(BaseModel):
    """One vocabulary size, read by both methods."""

    asked: int
    coverage_learned: int
    merging_learned: int
    coverage_corpus: int
    merging_corpus: int
    coverage_sentence: int
    merging_sentence: int


class Reading(BaseModel):
    """One text cut by both vocabularies at the compared size."""

    label: str
    text: str
    coverage: list[str]
    merging: list[str]


class VocabularyPair(BaseModel):
    """The two chosen vocabularies at one size, and where they agree."""

    size: int
    coverage_pieces: list[str]
    merging_pieces: list[str]
    shared: list[str]
    coverage_whole_words: int
    merging_whole_words: int
    coverage_mean_length: float
    merging_mean_length: float


class MatchedResponse(BaseModel):
    """The two methods on one corpus, at eleven matched sizes."""

    corpus: list[str]
    sentence: str
    control: str
    n_distinct_words: int
    n_word_occurrences: int
    alphabet_size: int
    total_positions: int
    n_candidates: int
    top_pairs: list[PairCount]
    sentence_bare_by_coverage: int
    sentence_bare_by_merging: int
    coverage_ceiling: int
    merging_ceiling: int
    rows: list[MatchedRow]
    crossover: int | None
    compared: VocabularyPair
    readings: list[Reading]
    sentence_decoded: str
    sentence_round_trip_exact: bool
    control_pieces: int
    control_round_trip_exact: bool


@cache
def _matched() -> MatchedResponse:
    counts = _word_counts(SENTENCE_CORPUS)
    words = _spelled(counts)
    alphabet = alphabet_of(words)

    rows: list[MatchedRow] = []
    for size in SWEEP_SIZES:
        coverage = _prefix_at(CorpusChoice.SENTENCES, size)
        merging = BytePairEncoding(byte_fallback=False, vocabulary_size=size).fit(
            SENTENCE_CORPUS
        )
        rows.append(
            MatchedRow(
                asked=size,
                coverage_learned=coverage.n_tokens,
                merging_learned=merging.vocabulary.n_tokens,
                coverage_corpus=coverage.corpus_pieces(counts),
                merging_corpus=sum(
                    merging.encode(word).n_tokens * count
                    for word, count in counts.items()
                ),
                coverage_sentence=len(coverage.pieces_of(RUNNING_SENTENCE)),
                merging_sentence=merging.encode(RUNNING_SENTENCE).n_tokens,
            )
        )

    crossover = next(
        (row.asked for row in rows if row.merging_sentence < row.coverage_sentence),
        None,
    )

    coverage = _prefix_at(CorpusChoice.SENTENCES, COMPARED_AT)
    merging = BytePairEncoding(byte_fallback=False, vocabulary_size=COMPARED_AT).fit(
        SENTENCE_CORPUS
    )
    coverage_pieces = list(coverage.chosen.pieces)
    merging_pieces = _merged_pieces(merging)
    whole = {word + END_OF_WORD for word in counts}

    sentence_pieces = coverage.pieces_of(RUNNING_SENTENCE)
    control_pieces = coverage.pieces_of(CONTROL_SENTENCE)

    return MatchedResponse(
        corpus=SENTENCE_CORPUS,
        sentence=RUNNING_SENTENCE,
        control=CONTROL_SENTENCE,
        n_distinct_words=len(counts),
        n_word_occurrences=sum(counts.values()),
        alphabet_size=len(alphabet),
        total_positions=sum(word.count * len(word.symbols) for word in words),
        n_candidates=len(_candidates(words)),
        top_pairs=_top_pairs(words, 4),
        sentence_bare_by_coverage=sum(
            1 for piece in sentence_pieces if _is_bare(piece, alphabet)
        ),
        sentence_bare_by_merging=sum(
            1
            for piece in merging.encode(RUNNING_SENTENCE).texts
            if _is_bare(piece, alphabet)
        ),
        coverage_ceiling=_full_fit(CorpusChoice.SENTENCES).vocabulary.n_tokens,
        merging_ceiling=BytePairEncoding(
            byte_fallback=False, vocabulary_size=MAX_VOCABULARY_SIZE
        )
        .fit(SENTENCE_CORPUS)
        .vocabulary.n_tokens,
        rows=rows,
        crossover=crossover,
        compared=VocabularyPair(
            size=COMPARED_AT,
            coverage_pieces=coverage_pieces,
            merging_pieces=merging_pieces,
            shared=sorted(set(coverage_pieces) & set(merging_pieces)),
            coverage_whole_words=sum(1 for piece in coverage_pieces if piece in whole),
            merging_whole_words=sum(1 for piece in merging_pieces if piece in whole),
            coverage_mean_length=sum(len(_plain(piece)) for piece in coverage_pieces)
            / len(coverage_pieces),
            merging_mean_length=sum(len(_plain(piece)) for piece in merging_pieces)
            / len(merging_pieces),
        ),
        readings=[
            Reading(
                label=label,
                text=text,
                coverage=coverage.pieces_of(text),
                merging=list(merging.encode(text).texts),
            )
            for label, text in (
                ("the held-out sentence", RUNNING_SENTENCE),
                ("expected, which the corpus holds", "expected"),
                ("expect, which it does not", "expect"),
                ("analysis", "analysis"),
                ("costing", "costing"),
            )
        ],
        sentence_decoded=coverage.decoded(sentence_pieces),
        sentence_round_trip_exact=coverage.decoded(sentence_pieces) == RUNNING_SENTENCE,
        control_pieces=len(control_pieces),
        control_round_trip_exact=coverage.decoded(control_pieces) == CONTROL_SENTENCE,
    )


@router.get("/matched", response_model=MatchedResponse)
def matched() -> MatchedResponse:
    """Both methods fitted at the same eleven sizes on one corpus."""
    return _matched()


# --------------------------------------------------------------------------
# /limits: the falling gains, the best pair, and the two dials.
# --------------------------------------------------------------------------


class GainRow(BaseModel):
    """One chosen piece, how long it is, how often it occurs, and what it won."""

    rank: int
    piece: str
    n_symbols: int
    occurrences: int
    coverage: int


class BestSetRow(BaseModel):
    """What the rule reaches against what the best set of that size covers."""

    budget: int
    greedy_pieces: list[str]
    greedy_coverage: int
    best_pieces: list[str]
    best_coverage: int
    ratio: float


class OnceRow(BaseModel):
    """How many rows each method spends on a word the corpus saw once."""

    size: int
    coverage_rows: int
    coverage_once: int
    merging_rows: int
    merging_once: int
    examples: list[str]


class LimitsResponse(BaseModel):
    """Everything the closing part rests on, except the length sweep."""

    gains: list[GainRow]
    total_positions: int
    covered_positions: int
    trap_words: list[SpelledEntry]
    trap_positions: int
    trap_rows: list[BestSetRow]
    trap_greedy_corpus: int
    trap_best_corpus: int
    guarantee: float
    longest_tie_run: int
    longest_tie_value: int
    n_distinct_words: int
    n_words_seen_once: int
    once_rows: list[OnceRow]


@cache
def _limits() -> LimitsResponse:
    counts = _word_counts(SENTENCE_CORPUS)
    fitted = _full_fit(CorpusChoice.SENTENCES)

    trap_words = _spelled(TRAP_COUNTS)
    trap_candidates = _candidates(trap_words)
    trap_rows: list[BestSetRow] = []
    for budget in TRAP_BUDGETS:
        picks = _greedy_choices(trap_words, budget, False)
        best_value = 0
        best_set: tuple[str, ...] = ()
        for combination in itertools.combinations(sorted(trap_candidates), budget):
            value = _set_coverage(trap_words, trap_candidates, combination)
            if value > best_value:
                best_value, best_set = value, combination
        greedy_total = sum(score for _, score in picks)
        trap_rows.append(
            BestSetRow(
                budget=budget,
                greedy_pieces=[piece for piece, _ in picks],
                greedy_coverage=greedy_total,
                best_pieces=list(best_set),
                best_coverage=best_value,
                ratio=greedy_total / best_value,
            )
        )

    at_two = next(row for row in trap_rows if row.budget == 2)

    def trap_corpus(pieces: Sequence[str]) -> int:
        chosen = _as_chosen(trap_words, trap_candidates, pieces)
        return sum(
            len(cover(_symbols_of(word), chosen).pieces) * count
            for word, count in TRAP_COUNTS.items()
        )

    once = {word for word, count in counts.items() if count == 1}
    once_rows: list[OnceRow] = []
    for size in ONCE_SIZES:
        coverage = _prefix_at(CorpusChoice.SENTENCES, size)
        merging = BytePairEncoding(byte_fallback=False, vocabulary_size=size).fit(
            SENTENCE_CORPUS
        )
        coverage_pieces = list(coverage.chosen.pieces)
        merging_pieces = _merged_pieces(merging)
        spent = [
            piece
            for piece in coverage_pieces
            if piece.endswith(END_OF_WORD) and _plain(piece) in once
        ]
        once_rows.append(
            OnceRow(
                size=size,
                coverage_rows=len(coverage_pieces),
                coverage_once=len(spent),
                merging_rows=len(merging_pieces),
                merging_once=sum(
                    1
                    for piece in merging_pieces
                    if piece.endswith(END_OF_WORD) and _plain(piece) in once
                ),
                examples=spent[:6],
            )
        )

    every_gain = [chosen.coverage for chosen in fitted.chosen_pieces]
    longest_run, longest_value, run, previous = 0, 0, 0, None
    for gain in every_gain:
        run = run + 1 if gain == previous else 1
        previous = gain
        if run > longest_run:
            longest_run, longest_value = run, gain

    return LimitsResponse(
        gains=[
            GainRow(
                rank=rank,
                piece=chosen.piece,
                n_symbols=chosen.n_symbols,
                occurrences=_occurrences(_spelled(counts), chosen.symbols),
                coverage=chosen.coverage,
            )
            for rank, chosen in enumerate(fitted.chosen_pieces)
        ],
        total_positions=sum(
            len(_symbols_of(word)) * count for word, count in counts.items()
        ),
        covered_positions=fitted.chosen_pieces.total_coverage,
        trap_words=[
            SpelledEntry(word=word, count=count, symbols=list(_symbols_of(word)))
            for word, count in TRAP_COUNTS.items()
        ],
        trap_positions=sum(word.count * len(word.symbols) for word in trap_words),
        trap_rows=trap_rows,
        trap_greedy_corpus=trap_corpus(at_two.greedy_pieces),
        trap_best_corpus=trap_corpus(at_two.best_pieces),
        guarantee=1.0 - 1.0 / math.e,
        longest_tie_run=longest_run,
        longest_tie_value=longest_value,
        n_distinct_words=len(counts),
        n_words_seen_once=len(once),
        once_rows=once_rows,
    )


@router.get("/limits", response_model=LimitsResponse)
def limits() -> LimitsResponse:
    """The falling gains, the best pair, and the rows spent on rare words."""
    return _limits()


# --------------------------------------------------------------------------
# /capping: the longest piece allowed, swept.
# --------------------------------------------------------------------------


class CapRow(BaseModel):
    """One setting of the longest piece allowed."""

    cap: int
    learned: int
    corpus_pieces: int
    sentence_pieces: int
    once_seen_rows: int
    first_pieces: list[str]


class CappingResponse(BaseModel):
    """The same size asked for, under six ceilings on how long a piece may be.

    This is the page's one slow endpoint, since a different ceiling admits a
    different set of candidates and so needs a fit of its own rather than a
    prefix of one already done.
    """

    size: int
    sentence: str
    rows: list[CapRow]


@cache
def _capping() -> CappingResponse:
    counts = _word_counts(SENTENCE_CORPUS)
    once = {word for word, count in counts.items() if count == 1}
    rows: list[CapRow] = []
    for cap in LENGTH_CAPS:
        model = GreedyCoverageTokenizer(
            vocabulary_size=COMPARED_AT, max_piece_length=cap
        ).fit(SENTENCE_CORPUS)
        pieces = [chosen.piece for chosen in model.chosen_pieces]
        rows.append(
            CapRow(
                cap=cap,
                learned=model.vocabulary.n_tokens,
                corpus_pieces=sum(
                    model.encode(word).n_tokens * count
                    for word, count in counts.items()
                ),
                sentence_pieces=model.encode(RUNNING_SENTENCE).n_tokens,
                once_seen_rows=sum(
                    1
                    for piece in pieces
                    if piece.endswith(END_OF_WORD) and _plain(piece) in once
                ),
                first_pieces=pieces[:4],
            )
        )
    return CappingResponse(size=COMPARED_AT, sentence=RUNNING_SENTENCE, rows=rows)


@router.get("/capping", response_model=CappingResponse)
def capping() -> CappingResponse:
    """One vocabulary size under six ceilings on how long a piece may be."""
    return _capping()


# --------------------------------------------------------------------------
# /fit: the playground.
# --------------------------------------------------------------------------


class FitRequest(BaseModel):
    """A corpus to choose from, a size to stop at, and a text to read back."""

    corpus: CorpusChoice = CorpusChoice.SENTENCES
    vocabulary_size: int = Field(ge=2, le=MAX_VOCABULARY_SIZE)
    text: str = Field(default=RUNNING_SENTENCE, max_length=MAX_TEXT_LENGTH)

    model_config = {"extra": "forbid"}


class ChosenRow(BaseModel):
    """One chosen piece, with the positions it newly covered."""

    rank: int
    piece: str
    n_symbols: int
    coverage: int


class FitResponse(BaseModel):
    """Everything the playground shows for one fit."""

    corpus: list[str]
    alphabet: list[str]
    smallest_size: int
    asked: int
    learned: int
    n_pieces: int
    pieces: list[ChosenRow]
    tokens: list[str]
    total_positions: int
    covered_positions: int
    corpus_pieces: int
    corpus_bare_pieces: int
    text: str
    text_pieces: list[str]
    n_text_pieces: int
    unknown_pieces: int
    decoded: str
    round_trip_exact: bool


@router.post("/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Choose a vocabulary of the requested size and read one text with it.

    A size below the alphabet is refused by the trainer itself rather than
    here, so the sentence a reader sees is the one the trainer writes.
    """
    corpus = CORPORA[request.corpus]
    counts = _word_counts(corpus)
    alphabet = alphabet_of(_spelled(counts))
    try:
        if request.vocabulary_size < len(alphabet) + 1:
            GreedyCoverageTokenizer(vocabulary_size=request.vocabulary_size).fit(corpus)
        fitted = _prefix_at(request.corpus, request.vocabulary_size)
        pieces = fitted.pieces_of(request.text)
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    decoded = fitted.decoded(pieces)
    bare = sum(len(_symbols_of(word)) * count for word, count in counts.items())
    return FitResponse(
        corpus=corpus,
        alphabet=alphabet,
        smallest_size=len(alphabet) + 1,
        asked=request.vocabulary_size,
        learned=fitted.n_tokens,
        n_pieces=len(fitted.chosen),
        pieces=[
            ChosenRow(
                rank=rank,
                piece=chosen.piece,
                n_symbols=chosen.n_symbols,
                coverage=chosen.coverage,
            )
            for rank, chosen in enumerate(fitted.chosen)
        ],
        tokens=list(fitted.vocabulary),
        total_positions=bare,
        covered_positions=fitted.covered_positions(counts),
        corpus_pieces=fitted.corpus_pieces(counts),
        corpus_bare_pieces=bare,
        text=request.text,
        text_pieces=pieces,
        n_text_pieces=len(pieces),
        unknown_pieces=sum(1 for piece in pieces if piece == UNKNOWN),
        decoded=decoded,
        round_trip_exact=decoded == request.text,
    )
