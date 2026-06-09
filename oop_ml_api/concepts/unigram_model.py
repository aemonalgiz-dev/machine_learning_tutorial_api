"""The unigram language model page's endpoints.

Four of them, one per family of widgets rather than one per number.
``/spellings`` is everything about a piece carrying a probability: the four-word
corpus, the seed table it produces, every spelling of a handful of words with
the probability of each, and the lattice one of those words is spelled over.
``/pruning`` is everything about what a piece is worth: the one-word corpus the
expectation-maximisation step is worked on by hand, the four rounds of it on the
four words, the measured loss of every removable piece on both corpora, and the
sizes each pruning round leaves behind. ``/sizes`` sweeps the vocabulary size
over the eighteen sentences with a merge-grown vocabulary of the same size
beside it, and cuts the running sentence at three of those sizes. ``/fit`` is
the playground.

Two things here are the page's own rather than the library's. The first is the
enumeration of *every* spelling of a word, which the library has no reason to
build, since both of its questions are dynamic programmes that never write the
paths down; showing that a cut was chosen rather than forced means listing what
it was chosen from. The second is the count of how many learned pieces of each
vocabulary are whole words of the corpus, which is a question about the two
methods rather than about either one. Everything else is a fit.

Every refusal is rewritten by ``_in_plain_words`` before it leaves here, since
the page must never show a parameter name, and the test module asserts that no
identifier survives the rewriting.
"""

from __future__ import annotations

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
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    SpelledWord,
)
from oop_ml.core.natural_language_processing.tokenization.subword.shortest_path import (
    shortest_path,
)
from oop_ml.core.natural_language_processing.tokenization.subword.unigram import (
    PieceTable,
    UnigramLanguageModel,
    seed_pieces,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/concepts/the-unigram-language-model", tags=["the-unigram-language-model"]
)


MAX_VOCABULARY_SIZE = 600
"""The largest vocabulary a request may ask for.

The eighteen sentences hold 866 candidate pieces in all, so this never truncates
anything the page shows; it exists so a crafted request cannot turn a teaching
endpoint into a long fit.
"""

MAX_TEXT_LENGTH = 200
"""The longest text the playground may ask to read."""

END_OF_WORD = "</w>"
"""The mark every fit on this page writes on a word's last character."""

UNKNOWN = "[UNK]"
"""What a fit answers when it cannot spell something."""

SEED_SIZE = 1000
"""How many candidate substrings the seed keeps, on top of the alphabet.

Larger than either corpus can supply, so the seed on both pages' corpora is
every substring up to the length bound and the number is a fact about the
corpus rather than about this setting.
"""

MAX_PIECE_LENGTH = 16
"""The longest candidate piece, in symbols."""

SHRINKING_FACTOR = 0.75
"""The share of the table each pruning round keeps."""

ESTIMATION_ROUNDS = 2
"""How many rounds of re-estimation precede each size check."""

MAX_SPELLINGS_SHOWN = 8
"""How many of a word's spellings the page lists, highest probability first."""


# --------------------------------------------------------------------------
# The corpora the page works with.
# --------------------------------------------------------------------------

FOUR_WORD_COUNTS: dict[str, int] = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
"""Four words with their counts, small enough that every number is checkable.

The corpus the original subword paper worked its example on, and the corpus the
from-scratch trainer's own spec is pinned against, so every number the page
quotes from it is a number a test already holds.
"""

FOUR_WORD_CORPUS: list[str] = [
    " ".join(word for word, count in FOUR_WORD_COUNTS.items() for _ in range(count))
]

UNSEEN_WORD = "lowest"
"""A word the four-word corpus never contains, spellable from what it taught."""

LADDER_WORDS = ("lowest", "newest", "lower", "low")
"""The words whose spellings the page lists, one of them held out."""

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

Written so that every character of the running sentence appears somewhere,
including the apostrophe, the hyphen and both capitals, and so that the words
around it are plausible neighbours of the ones it uses. The same eighteen
sentences the sibling pages on merging carry, kept here rather than imported so
the three pages cannot move each other's numbers.
"""

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page in this section carries."""

CONTROL_TEXT = "The revised cost of the low analysis was expected."
"""A sentence spelled entirely in symbols the corpus taught, for the round trip."""

SWEEP_SIZES = (52, 56, 60, 66, 72, 80, 90, 100, 110, 120, 130, 137, 160, 200)
"""The vocabulary sizes both methods are fitted at."""

CUT_SIZES = (52, 80, 137)
"""The three sizes the running sentence is printed at."""

COMPOSITION_SIZES = (80, 100, 137)
"""The three sizes the two vocabularies are taken apart at."""

LENGTH_PROBE_WORDS = (
    "lowest",
    "lower",
    "newest",
    "widest",
    "expect",
    "expects",
    "expecting",
    "analyse",
    "reanalysis",
    "reporting",
    "estimates",
    "Alvarez",
    "didn't",
    "low-cost",
    "re-analysis.",
    "unexpected",
)
"""Words the page asks both questions of, whether the likeliest spelling is also
the shortest one the same pieces allow."""


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
    ("seed_size", "how many candidates the seed keeps"),
    ("max_piece_length", "the longest a candidate may be"),
    ("shrinking_factor", "the share each round keeps"),
    ("n_expectation_maximisation_rounds", "how many rounds of re-estimation run"),
    ("sampling_temperature", "how peaked the draw is"),
    ("pre_tokenizer", "the rule that finds the words"),
    ("the unknown token", "one token for anything unspellable"),
    ("UnigramLanguageModel", "this scheme"),
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


def _alphabet(corpus: Sequence[str]) -> list[str]:
    return sorted(
        {symbol for word in _spelled(_word_counts(corpus)) for symbol in word}
    )


def _character_pieces(counts: dict[str, int]) -> int:
    """What the corpus costs when every symbol is its own piece."""
    return sum(len(_symbols_of(word)) * count for word, count in counts.items())


def _seed_table(spelled: Sequence[SpelledWord]) -> PieceTable:
    """The starting model: every candidate, weighted by how often it occurs."""
    return PieceTable.seeded_from(
        sorted(spelled, key=lambda word: word.symbols), SEED_SIZE, MAX_PIECE_LENGTH
    )


def _every_spelling(
    table: PieceTable, symbols: Sequence[str]
) -> list[tuple[tuple[str, ...], float]]:
    """Every path through the lattice, highest score first.

    Written here rather than in the library because neither question the model
    asks needs the paths written down: one is a maximum over them and the other
    a total, and both are dynamic programmes that keep one number per position.
    Showing that a cut was chosen means listing what it was chosen from, which
    is the page's question rather than the model's.
    """
    answers: list[tuple[tuple[str, ...], float]] = []

    def walk(position: int, chosen: list[str], score: float) -> None:
        if position == len(symbols):
            answers.append((tuple(chosen), score))
            return
        for end in range(position + 1, len(symbols) + 1):
            piece = "".join(symbols[position:end])
            if piece in table:
                walk(end, [*chosen, piece], score + table.log_probability_of(piece))
            elif end == position + 1:
                walk(end, [*chosen, piece], score + table.unknown_symbol_score)

    walk(0, [], 0.0)
    return sorted(answers, key=lambda pair: (-pair[1], pair[0]))


def _unigram_corpus_pieces(model: UnigramLanguageModel, counts: dict[str, int]) -> int:
    return sum(model.encode(word).n_tokens * count for word, count in counts.items())


def _merged_corpus_pieces(model: BytePairEncoding, counts: dict[str, int]) -> int:
    return sum(model.encode(word).n_tokens * count for word, count in counts.items())


# --------------------------------------------------------------------------
# /spellings: a probability on a piece, and what it does to one word.
# --------------------------------------------------------------------------


class SpelledEntry(BaseModel):
    """One distinct word, how often it occurs, and its spelling in symbols."""

    word: str
    count: int
    symbols: list[str]


class SeedCandidate(BaseModel):
    """One candidate piece, how often the corpus contains it, and its rank score."""

    piece: str
    frequency: int
    n_symbols: int
    score: float


class PieceProbability(BaseModel):
    """One piece and how likely the model thinks it is."""

    piece: str
    log_probability: float
    probability: float


class SpellingRow(BaseModel):
    """One spelling of one word, and what the model gives it.

    ``share`` is the spelling's probability divided by the word's own, so the
    shares of every spelling of a word add to one.
    """

    pieces: list[str]
    n_pieces: int
    log_probability: float
    probability: float
    share: float
    is_best: bool
    is_merged_cut: bool


class WordSpellings(BaseModel):
    """Every spelling of one word under the seed model, ranked.

    ``merged_rank`` is where the cut a merge-grown vocabulary of the same
    corpus produces comes in that ranking, counting from one, and is null when
    that cut uses a piece the seed does not hold.
    """

    word: str
    symbols: list[str]
    in_corpus: bool
    n_spellings: int
    marginal_log_probability: float
    marginal_probability: float
    best_pieces: list[str]
    best_log_probability: float
    merged_pieces: list[str]
    merged_rank: int | None
    merged_share: float | None
    rows: list[SpellingRow]


class LatticeArc(BaseModel):
    """One piece the model holds, spanning positions ``start`` to ``end``."""

    start: int
    end: int
    piece: str
    log_probability: float
    probability: float
    on_best_path: bool


class LatticeView(BaseModel):
    """One word as positions and arcs, which is what both questions read."""

    word: str
    symbols: list[str]
    n_positions: int
    arcs: list[LatticeArc]
    n_paths: int
    best_pieces: list[str]
    best_log_probability: float


class SpellingsResponse(BaseModel):
    """The four-word corpus, its seed model, and the spellings that follow."""

    corpus: list[str]
    word_counts: list[SpelledEntry]
    alphabet: list[str]
    total_occurrences: int
    character_pieces: int
    n_candidates: int
    candidates: list[SeedCandidate]
    seed_table: list[PieceProbability]
    words: list[WordSpellings]
    lattice: LatticeView


def _merged_cut(model: BytePairEncoding, word: str) -> list[str]:
    return list(model.encode(word).texts)


@cache
def _spellings() -> SpellingsResponse:
    counts = FOUR_WORD_COUNTS
    spelled = _spelled(counts)
    ordered = sorted(spelled, key=lambda word: word.symbols)
    seed = seed_pieces(ordered, SEED_SIZE, MAX_PIECE_LENGTH)
    table = _seed_table(spelled)
    merged = BytePairEncoding(byte_fallback=False, vocabulary_size=22).fit(
        FOUR_WORD_CORPUS
    )

    words: list[WordSpellings] = []
    for word in LADDER_WORDS:
        symbols = _symbols_of(word)
        spellings = _every_spelling(table, symbols)
        marginal = table.marginal_log_likelihood(symbols)
        best = table.best_segmentation(symbols)
        merged_pieces = _merged_cut(merged, word)
        merged_rank: int | None = None
        merged_share: float | None = None
        for position, (pieces, score) in enumerate(spellings):
            if list(pieces) == merged_pieces:
                merged_rank = position + 1
                merged_share = math.exp(score - marginal)
        words.append(
            WordSpellings(
                word=word,
                symbols=list(symbols),
                in_corpus=word in counts,
                n_spellings=len(spellings),
                marginal_log_probability=marginal,
                marginal_probability=math.exp(marginal),
                best_pieces=list(best.pieces),
                best_log_probability=best.log_probability,
                merged_pieces=merged_pieces,
                merged_rank=merged_rank,
                merged_share=merged_share,
                rows=[
                    SpellingRow(
                        pieces=list(pieces),
                        n_pieces=len(pieces),
                        log_probability=score,
                        probability=math.exp(score),
                        share=math.exp(score - marginal),
                        is_best=list(pieces) == list(best.pieces),
                        is_merged_cut=list(pieces) == merged_pieces,
                    )
                    for pieces, score in spellings[:MAX_SPELLINGS_SHOWN]
                ],
            )
        )

    lattice_symbols = _symbols_of(UNSEEN_WORD)
    lattice_best = table.best_segmentation(lattice_symbols)
    on_best: set[tuple[int, int]] = set()
    position = 0
    for piece in lattice_best.pieces:
        span = len(_split_symbols(piece))
        on_best.add((position, position + span))
        position += span

    arcs: list[LatticeArc] = []
    for start in range(len(lattice_symbols)):
        for end in range(start + 1, len(lattice_symbols) + 1):
            piece = "".join(lattice_symbols[start:end])
            if piece not in table:
                continue
            log_probability = table.log_probability_of(piece)
            arcs.append(
                LatticeArc(
                    start=start,
                    end=end,
                    piece=piece,
                    log_probability=log_probability,
                    probability=math.exp(log_probability),
                    on_best_path=(start, end) in on_best,
                )
            )

    return SpellingsResponse(
        corpus=FOUR_WORD_CORPUS,
        word_counts=[
            SpelledEntry(word=word, count=count, symbols=list(_symbols_of(word)))
            for word, count in counts.items()
        ],
        alphabet=_alphabet(FOUR_WORD_CORPUS),
        total_occurrences=sum(counts.values()),
        character_pieces=_character_pieces(counts),
        n_candidates=seed.n_pieces,
        candidates=[
            SeedCandidate(
                piece=piece,
                frequency=int(seed.score_of(piece)),
                n_symbols=len(_split_symbols(piece)),
                score=seed.score_of(piece) * len(_split_symbols(piece)),
            )
            for piece in list(seed)[:12]
        ],
        seed_table=[
            PieceProbability(
                piece=piece,
                log_probability=table.log_probability_of(piece),
                probability=math.exp(table.log_probability_of(piece)),
            )
            for piece in table.pieces_by_probability[:12]
        ],
        words=words,
        lattice=LatticeView(
            word=UNSEEN_WORD,
            symbols=list(lattice_symbols),
            n_positions=len(lattice_symbols) + 1,
            arcs=arcs,
            n_paths=len(_every_spelling(table, lattice_symbols)),
            best_pieces=list(lattice_best.pieces),
            best_log_probability=lattice_best.log_probability,
        ),
    )


def _split_symbols(piece: str) -> list[str]:
    """A piece back into the symbols it spans, the marker riding on the last."""
    if piece.endswith(END_OF_WORD):
        body = piece[: -len(END_OF_WORD)]
        return [*body[:-1], body[-1] + END_OF_WORD]
    return list(piece)


@router.get("/spellings", response_model=SpellingsResponse)
def spellings() -> SpellingsResponse:
    """The four-word corpus, its starting model, and every cut of four words."""
    return _spellings()


# --------------------------------------------------------------------------
# /pruning: what a piece is worth, and how the table is shrunk.
# --------------------------------------------------------------------------


class ExpectedCount(BaseModel):
    """One piece and how often the model expects it was used."""

    piece: str
    count: float


class EstimationRound(BaseModel):
    """One round of re-estimation, and what it left the probabilities at."""

    round_number: int
    probabilities: list[PieceProbability]


class WorkedEstimation(BaseModel):
    """The smallest corpus with a choice in it, taken through two rounds.

    One word, ``ab``, seen once, spelled in two symbols, with three pieces at
    equal probability. Everything here is small enough to check by hand.
    """

    word: str
    symbols: list[str]
    start: list[PieceProbability]
    spellings: list[SpellingRow]
    marginal_log_probability: float
    marginal_probability: float
    expected: list[ExpectedCount]
    expected_total: float
    rounds: list[EstimationRound]


class ConcentrationRow(BaseModel):
    """One round on the four words: what three pieces are worth, and one cut."""

    round_number: int
    probabilities: list[PieceProbability]
    unseen_pieces: list[str]
    unseen_log_probability: float


class LossRow(BaseModel):
    """One piece, and the corpus log probability lost by dropping it."""

    piece: str
    loss: float


class LossView(BaseModel):
    """Every removable piece of one table ranked by what its removal costs."""

    label: str
    n_pieces: int
    n_removable: int
    n_costing_nothing: int
    most_costly: list[LossRow]
    least_costly: list[LossRow]


class PruningTrace(BaseModel):
    """How large the table was after each round, ending at the target."""

    label: str
    target: int
    sizes: list[int]
    n_rounds: int


class PruningResponse(BaseModel):
    """The worked estimation, the concentration it causes, and the losses."""

    worked: WorkedEstimation
    concentration: list[ConcentrationRow]
    losses: list[LossView]
    traces: list[PruningTrace]
    shrinking_factor: float
    estimation_rounds: int
    tie_broken_survivors: list[str]
    tie_broken_size: int


def _loss_view(
    label: str, table: PieceTable, spelled: Sequence[SpelledWord]
) -> LossView:
    ordered = sorted(spelled, key=lambda word: word.symbols)
    losses = table.losses_if_removed(ordered)
    ranked = [LossRow(piece=piece, loss=losses.score_of(piece)) for piece in losses]
    return LossView(
        label=label,
        n_pieces=table.n_pieces,
        n_removable=losses.n_pieces,
        n_costing_nothing=sum(1 for row in ranked if row.loss == 0.0),
        most_costly=ranked[:10],
        least_costly=ranked[-5:],
    )


def _trace(label: str, spelled: Sequence[SpelledWord], target: int) -> PruningTrace:
    ordered = sorted(spelled, key=lambda word: word.symbols)
    table = _seed_table(ordered)
    sizes = [table.n_pieces]
    while True:
        for _ in range(ESTIMATION_ROUNDS):
            table = table.expectation_maximisation_round(ordered)
        if table.n_pieces <= target:
            break
        table = table.pruned_to(
            max(target, int(SHRINKING_FACTOR * table.n_pieces)), ordered
        )
        sizes.append(table.n_pieces)
    return PruningTrace(
        label=label, target=target, sizes=sizes, n_rounds=len(sizes) - 1
    )


@cache
def _pruning() -> PruningResponse:
    third = math.log(1 / 3)
    start = PieceTable(
        {"a": third, "b" + END_OF_WORD: third, "ab" + END_OF_WORD: third}
    )
    ab_symbols = ("a", "b" + END_OF_WORD)
    ab_word = [SpelledWord(ab_symbols, 1)]
    ab_marginal = start.marginal_log_likelihood(ab_symbols)
    ab_expected = start.expected_counts(ab_word)

    rounds: list[EstimationRound] = []
    table = start
    for round_number in (1, 2):
        table = table.expectation_maximisation_round(ab_word)
        rounds.append(
            EstimationRound(
                round_number=round_number,
                probabilities=[
                    PieceProbability(
                        piece=piece,
                        log_probability=table.log_probability_of(piece),
                        probability=math.exp(table.log_probability_of(piece)),
                    )
                    for piece in table.pieces_by_probability
                ],
            )
        )

    four = sorted(_spelled(FOUR_WORD_COUNTS), key=lambda word: word.symbols)
    watched = ("low" + END_OF_WORD, "lo", "l")
    unseen_symbols = _symbols_of(UNSEEN_WORD)
    concentration: list[ConcentrationRow] = []
    four_table = _seed_table(four)
    for round_number in range(5):
        if round_number > 0:
            four_table = four_table.expectation_maximisation_round(four)
        best = four_table.best_segmentation(unseen_symbols)
        concentration.append(
            ConcentrationRow(
                round_number=round_number,
                probabilities=[
                    PieceProbability(
                        piece=piece,
                        log_probability=four_table.log_probability_of(piece),
                        probability=math.exp(four_table.log_probability_of(piece)),
                    )
                    for piece in watched
                ],
                unseen_pieces=list(best.pieces),
                unseen_log_probability=best.log_probability,
            )
        )

    sentences = _spelled(_word_counts(SENTENCE_CORPUS))
    four_estimated = _seed_table(four)
    sentence_estimated = _seed_table(sentences)
    for _ in range(ESTIMATION_ROUNDS):
        four_estimated = four_estimated.expectation_maximisation_round(four)
        sentence_estimated = sentence_estimated.expectation_maximisation_round(
            sorted(sentences, key=lambda word: word.symbols)
        )

    survivors = UnigramLanguageModel(vocabulary_size=22).fit(FOUR_WORD_CORPUS)
    alphabet = set(_alphabet(FOUR_WORD_CORPUS))
    four_word_pieces = {
        f"{word[:-1]}{word[-1]}{END_OF_WORD}" for word in FOUR_WORD_COUNTS
    }

    return PruningResponse(
        worked=WorkedEstimation(
            word="ab",
            symbols=list(ab_symbols),
            start=[
                PieceProbability(
                    piece=piece,
                    log_probability=start.log_probability_of(piece),
                    probability=math.exp(start.log_probability_of(piece)),
                )
                for piece in start.pieces_by_probability
            ],
            spellings=[
                SpellingRow(
                    pieces=list(pieces),
                    n_pieces=len(pieces),
                    log_probability=score,
                    probability=math.exp(score),
                    share=math.exp(score - ab_marginal),
                    is_best=len(pieces) == 1,
                    is_merged_cut=False,
                )
                for pieces, score in _every_spelling(start, ab_symbols)
            ],
            marginal_log_probability=ab_marginal,
            marginal_probability=math.exp(ab_marginal),
            expected=[
                ExpectedCount(piece=piece, count=ab_expected.score_of(piece))
                for piece in ab_expected
            ],
            expected_total=ab_expected.total,
            rounds=rounds,
        ),
        concentration=concentration,
        losses=[
            _loss_view("four words", four_estimated, four),
            _loss_view("eighteen sentences", sentence_estimated, sentences),
        ],
        traces=[
            _trace("four words", four, 15),
            _trace("eighteen sentences", sentences, 136),
        ],
        shrinking_factor=SHRINKING_FACTOR,
        estimation_rounds=ESTIMATION_ROUNDS,
        tie_broken_survivors=[
            piece
            for piece in survivors.piece_table.pieces
            if piece not in alphabet and piece not in four_word_pieces
        ],
        tie_broken_size=survivors.vocabulary.n_tokens,
    )


@router.get("/pruning", response_model=PruningResponse)
def pruning() -> PruningResponse:
    """The estimation worked by hand, and what each piece turns out to be worth."""
    return _pruning()


# --------------------------------------------------------------------------
# /sizes: the sweep, against a vocabulary of the same size grown by merging.
# --------------------------------------------------------------------------


class SizeRow(BaseModel):
    """One vocabulary size, and what each method costs the corpus and the sentence."""

    asked: int
    unigram_learned: int
    unigram_corpus: int
    unigram_sentence: int
    merged_learned: int
    merged_corpus: int
    merged_sentence: int


class CutPair(BaseModel):
    """The running sentence cut by both methods at one size."""

    vocabulary_size: int
    unigram_pieces: list[str]
    unigram_n: int
    unigram_decoded: str
    unigram_round_trip: bool
    merged_pieces: list[str]
    merged_n: int


class Composition(BaseModel):
    """What each vocabulary of one size turned out to be made of.

    ``learned`` counts the pieces that are not single symbols, and
    ``whole_words`` how many of those are a whole word of the corpus with its
    end-of-word mark.
    """

    asked: int
    unigram_learned: int
    unigram_whole_words: int
    unigram_other: list[str]
    merged_learned: int
    merged_whole_words: int


class LengthComparison(BaseModel):
    """One word whose likeliest spelling is not the shortest available."""

    word: str
    likeliest_pieces: list[str]
    likeliest_n: int
    likeliest_log_probability: float
    shortest_pieces: list[str]
    shortest_n: int
    shortest_log_probability: float


class SizesResponse(BaseModel):
    """The sweep, the cuts, the composition, and the round trip."""

    corpus: list[str]
    sentence: str
    control_text: str
    n_distinct_words: int
    n_word_occurrences: int
    alphabet_size: int
    character_pieces: int
    smallest_size: int
    n_candidates: int
    rows: list[SizeRow]
    cuts: list[CutPair]
    compositions: list[Composition]
    control_pieces: int
    control_round_trip: bool
    unspellable_symbol: str
    length_comparisons: list[LengthComparison]
    length_comparison_size: int
    n_words_examined: int


@cache
def _sizes() -> SizesResponse:
    counts = _word_counts(SENTENCE_CORPUS)
    alphabet = _alphabet(SENTENCE_CORPUS)
    alphabet_set = set(alphabet)
    spelled = _spelled(counts)
    whole_words = {f"{word[:-1]}{word[-1]}{END_OF_WORD}" for word in counts}

    rows: list[SizeRow] = []
    for asked in SWEEP_SIZES:
        unigram = UnigramLanguageModel(vocabulary_size=asked).fit(SENTENCE_CORPUS)
        merged = BytePairEncoding(byte_fallback=False, vocabulary_size=asked).fit(
            SENTENCE_CORPUS
        )
        rows.append(
            SizeRow(
                asked=asked,
                unigram_learned=unigram.vocabulary.n_tokens,
                unigram_corpus=_unigram_corpus_pieces(unigram, counts),
                unigram_sentence=unigram.encode(RUNNING_SENTENCE).n_tokens,
                merged_learned=merged.vocabulary.n_tokens,
                merged_corpus=_merged_corpus_pieces(merged, counts),
                merged_sentence=merged.encode(RUNNING_SENTENCE).n_tokens,
            )
        )

    cuts: list[CutPair] = []
    for asked in CUT_SIZES:
        unigram = UnigramLanguageModel(vocabulary_size=asked).fit(SENTENCE_CORPUS)
        merged = BytePairEncoding(byte_fallback=False, vocabulary_size=asked).fit(
            SENTENCE_CORPUS
        )
        encoding = unigram.encode(RUNNING_SENTENCE)
        decoded = unigram.decode(encoding.ids)
        cuts.append(
            CutPair(
                vocabulary_size=asked,
                unigram_pieces=list(encoding.texts),
                unigram_n=encoding.n_tokens,
                unigram_decoded=decoded,
                unigram_round_trip=decoded == RUNNING_SENTENCE,
                merged_pieces=list(merged.encode(RUNNING_SENTENCE).texts),
                merged_n=merged.encode(RUNNING_SENTENCE).n_tokens,
            )
        )

    compositions: list[Composition] = []
    for asked in COMPOSITION_SIZES:
        unigram = UnigramLanguageModel(vocabulary_size=asked).fit(SENTENCE_CORPUS)
        merged = BytePairEncoding(byte_fallback=False, vocabulary_size=asked).fit(
            SENTENCE_CORPUS
        )
        learned = [
            piece
            for piece in unigram.piece_table.pieces_by_probability
            if piece not in alphabet_set
        ]
        merged_learned = [
            token
            for token in merged.vocabulary
            if token not in alphabet_set and token != UNKNOWN
        ]
        compositions.append(
            Composition(
                asked=asked,
                unigram_learned=len(learned),
                unigram_whole_words=sum(1 for p in learned if p in whole_words),
                unigram_other=[p for p in learned if p not in whole_words],
                merged_learned=len(merged_learned),
                merged_whole_words=sum(1 for p in merged_learned if p in whole_words),
            )
        )

    control = UnigramLanguageModel(vocabulary_size=137).fit(SENTENCE_CORPUS)
    control_encoding = control.encode(CONTROL_TEXT)

    probe_size = 200
    probed = UnigramLanguageModel(vocabulary_size=probe_size).fit(SENTENCE_CORPUS)
    probe_table = probed.piece_table
    probe_pieces = set(probe_table.pieces)
    examined = [*LENGTH_PROBE_WORDS, *counts]
    comparisons: list[LengthComparison] = []
    for word in examined:
        symbols = _symbols_of(word)
        likeliest = probe_table.best_segmentation(symbols)
        shortest = shortest_path(symbols, probe_pieces)
        if likeliest.n_tokens == shortest.n_tokens:
            continue
        comparisons.append(
            LengthComparison(
                word=word,
                likeliest_pieces=list(likeliest.pieces),
                likeliest_n=likeliest.n_tokens,
                likeliest_log_probability=likeliest.log_probability,
                shortest_pieces=list(shortest.pieces),
                shortest_n=shortest.n_tokens,
                shortest_log_probability=sum(
                    probe_table.log_probability_of(piece)
                    if piece in probe_table
                    else probe_table.unknown_symbol_score
                    for piece in shortest.pieces
                ),
            )
        )

    return SizesResponse(
        corpus=SENTENCE_CORPUS,
        sentence=RUNNING_SENTENCE,
        control_text=CONTROL_TEXT,
        n_distinct_words=len(counts),
        n_word_occurrences=sum(counts.values()),
        alphabet_size=len(alphabet),
        character_pieces=_character_pieces(counts),
        smallest_size=len(alphabet) + 1,
        n_candidates=seed_pieces(
            sorted(spelled, key=lambda word: word.symbols), SEED_SIZE, MAX_PIECE_LENGTH
        ).n_pieces,
        rows=rows,
        cuts=cuts,
        compositions=compositions,
        control_pieces=control_encoding.n_tokens,
        control_round_trip=control.decode(control_encoding.ids) == CONTROL_TEXT,
        unspellable_symbol="z" + END_OF_WORD,
        length_comparisons=comparisons,
        length_comparison_size=probe_size,
        n_words_examined=len(examined),
    )


@router.get("/sizes", response_model=SizesResponse)
def sizes() -> SizesResponse:
    """The vocabulary-size sweep against a merge-grown vocabulary of each size."""
    return _sizes()


# --------------------------------------------------------------------------
# /fit: the playground.
# --------------------------------------------------------------------------


class FitRequest(BaseModel):
    """A corpus to learn from, a size to stop at, and a text to read back."""

    corpus: CorpusChoice = CorpusChoice.SENTENCES
    vocabulary_size: int = Field(ge=2, le=MAX_VOCABULARY_SIZE)
    text: str = Field(default=RUNNING_SENTENCE, max_length=MAX_TEXT_LENGTH)

    model_config = {"extra": "forbid"}


class FitResponse(BaseModel):
    """Everything the playground shows for one fit, both methods at one size."""

    corpus: list[str]
    alphabet: list[str]
    smallest_size: int
    n_candidates: int
    asked: int
    learned: int
    n_pruning_rounds: int
    pieces: list[PieceProbability]
    corpus_pieces: int
    corpus_character_pieces: int
    text: str
    cut: list[str]
    n_pieces: int
    unknown_pieces: int
    decoded: str
    round_trip_exact: bool
    merged_learned: int
    merged_corpus_pieces: int
    merged_cut: list[str]
    merged_n_pieces: int


@router.post("/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Learn a vocabulary of the requested size and read one text with it."""
    corpus = CORPORA[request.corpus]
    counts = _word_counts(corpus)
    alphabet = _alphabet(corpus)
    try:
        model = UnigramLanguageModel(vocabulary_size=request.vocabulary_size).fit(
            corpus
        )
        merged = BytePairEncoding(
            byte_fallback=False, vocabulary_size=request.vocabulary_size
        ).fit(corpus)
        encoding = model.encode(request.text)
        merged_encoding = merged.encode(request.text)
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    table = model.piece_table
    decoded = model.decode(encoding.ids)
    return FitResponse(
        corpus=corpus,
        alphabet=alphabet,
        smallest_size=len(alphabet) + 1,
        n_candidates=seed_pieces(
            sorted(_spelled(counts), key=lambda word: word.symbols),
            SEED_SIZE,
            MAX_PIECE_LENGTH,
        ).n_pieces,
        asked=request.vocabulary_size,
        learned=model.vocabulary.n_tokens,
        n_pruning_rounds=model.n_pruning_rounds,
        pieces=[
            PieceProbability(
                piece=piece,
                log_probability=table.log_probability_of(piece),
                probability=math.exp(table.log_probability_of(piece)),
            )
            for piece in table.pieces_by_probability[:24]
        ],
        corpus_pieces=_unigram_corpus_pieces(model, counts),
        corpus_character_pieces=_character_pieces(counts),
        text=request.text,
        cut=list(encoding.texts),
        n_pieces=encoding.n_tokens,
        unknown_pieces=sum(1 for piece in encoding.texts if piece == UNKNOWN),
        decoded=decoded,
        round_trip_exact=decoded == request.text,
        merged_learned=merged.vocabulary.n_tokens,
        merged_corpus_pieces=_merged_corpus_pieces(merged, counts),
        merged_cut=list(merged_encoding.texts),
        merged_n_pieces=merged_encoding.n_tokens,
    )
