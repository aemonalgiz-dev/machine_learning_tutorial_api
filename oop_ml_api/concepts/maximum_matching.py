"""Maximum Matching.

Every endpoint here is registered under ``/concepts/maximum-matching``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about MaximumMatchingSegmenter.

What this module adds to the library
------------------------------------
Three things the segmenter has no reason to carry, and one it does.

The segmenter itself produces every piece and every span on this page; nothing
here re-implements the greedy rule as a way of answering. What is assembled
around it is the bookkeeping the page quotes. First, the arithmetic of the
problem: how many characters a text has once its spaces are gone and how many
ways that many characters could be cut, which is what makes the greedy rule
worth having. Second, the exhaustive optimum, a dynamic program over every
reading whose pieces are dictionary words or lone characters, which is the only
honest way to say that a greedy answer was or was not the shortest available.
Third, the walk: the candidates tried at each position, longest first, and which
one was taken.

The walk *is* a second implementation of the loop, and it is treated as one. Its
takings are asserted in the tests to concatenate to the text and to agree piece
for piece with what the segmenter answered, so a walk that drifted from the rule
would fail there rather than reach the page as a plausible-looking trace.

The five earlier word-finding rules are imported for one contrast, in the setup
endpoint only: what each of them answers for a text with no spaces in it, which
is the failure this method exists for.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings and the page's readers have no reason to
meet an identifier. The only free text returned is the labels written here, the
words of the dictionaries, and the reader's own text; a test asserts that no
identifier survives into any of them, including into a refusal.
"""

from __future__ import annotations

from enum import StrEnum
from functools import cache
from typing import Annotated

from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.tokenization.segmentation.dictionary import (
    WordDictionary,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.maximum_matching import (
    MatchingDirection,
    MaximumMatchingSegmenter,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.word_level.moses import (
    MosesPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.pattern import (
    GPT2_PATTERN,
    PatternPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.penn_treebank import (
    PennTreebankPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.unicode_words import (
    UnicodeWordPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    NON_WHITESPACE_RUN,
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/maximum-matching", tags=["maximum-matching"])


# --------------------------------------------------------------------------
# The texts and the word lists the page carries.
# --------------------------------------------------------------------------

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page of this section carries, spaces and all."""

SPACELESS_SENTENCE = RUNNING_SENTENCE.replace(" ", "")
"""The same sentence with its spaces taken out, which is the page's opening puzzle."""

TABLE_TEXT = "thetabledownthere"
"""The English demonstration that the two scan directions disagree.

Four words of ordinary English run together. Read from the left the longest
entry at the start is a five-letter one and the reading falls apart behind it;
read from the right every take is the intended word.
"""

PARK_SENTENCE = "我们在野生动物园玩"
"""We play at the wildlife park, the sentence this method is classically shown on."""

PARK_WORDS: tuple[str, ...] = (
    "我们",
    "在",
    "在野",
    "野生",
    "生动",
    "动物园",
    "物",
    "园",
    "玩",
)
"""Nine words, six of which the sentence contains and three of which it does not."""

ORIGINS_SENTENCE = "研究生命起源"
"""Research into the origin of life, or a graduate student's fate and origin."""

ORIGINS_WORDS: tuple[str, ...] = ("研究", "研究生", "生命", "命", "起源")
"""Five words, out of which two complete readings of the sentence can be built."""

NAME_SENTENCE = "阿尔瓦雷斯研究生命起源"
"""The same sentence with a transliterated surname in front of it.

The surname is the one the running English sentence carries, written the way
Chinese writes a foreign name, and no word list built from Chinese text holds
it.
"""

NAME_READING: tuple[str, ...] = ("阿尔瓦雷斯", "研究", "生命", "起源")
"""What a reader answers for that sentence, which is four words rather than eight."""

ORDINARY_ENGLISH_RAW = """
a about above across act add after again against age ago agree air all allow
almost alone along already also although always among an analysis and animal
another answer any appear apply area arm around arrive art as ask at away
back bad bag ball bank base be bear beat beautiful because become bed been
before begin behind believe below beside best better between beyond big bird
bit black bled block blood blue board boat body book born both box boy break
bring broad brother build burn business but buy by call can car care carry
case cat catch cause cell centre century certain chance change character
charge check child choose church city claim class clean clear close cold
collect colour come common company compare complete consider contain
continue control cook cool copy corner cost could count country couple course
cover create cross cry cup cut dark data date day dead deal dear death decide
deep degree describe design detail develop did die difference different
difficult dinner direct discover discuss do doctor does dog door double doubt
down draw dream dress drink drive drop dry during each early earth ease east
easy eat edge effect effort egg eight either else end enough enter equal even
evening event ever every exact example except exist expect experience
explain eye face fact fail fall family far farm fast father fear feed feel
few field fight figure fill film final find fine finger finish fire first
fish fit five fix flat floor flow flower fly follow food foot for force
forget form former forward four free fresh friend from front full fun
further future game garden gas gather general get gift girl give glass go
gold good govern grand grass great green ground group grow guess guide
hair half hand hang happen happy hard has hat hate have he head health hear
heart heat heavy help her here hide high hill him his history hit hold hole
home hope horse hospital hot hour house how however human hundred hurt
i ice idea if imagine important improve in include increase indeed industry
inside instead into iron is island it its job join joy judge jump just keep
key kill kind king know lack lady land language large last late laugh law
lay lead learn leave left leg length less let letter level lie life light
like limit line list listen little live local long look lose lot loud love
low machine main major make man many map mark market marry match matter may
me mean measure meet member memory men mention method middle might mile milk
mind mine minute miss mix model modern moment money month moon more morning
most mother mountain move much music must my name nation nature near need
never new news next nice night nine no none nor north nose not note nothing
notice now number object observe of off offer office often oil old on once
one only open or order other our out over own page pain paint pair paper
part pass past pay peace people perhaps period person picture piece place
plan plant play please point police poor position possible power practice
prepare present press pretty price print probably problem produce program
prove provide public pull purpose push put quality question quick quiet quite
race radio rain raise rate rather reach read ready real reason receive record
red reduce refer region remain remember repeat report require rest result
return rich ride right ring rise river road rock roll room round row rule run
safe same save say school science sea season seat second see seem sell send
sense separate serve set seven several shape share she ship shoe shop short
should show side sight sign silence simple since sing single sister sit six
size skin sky sleep slow small smile snow so social soft soil soldier some
son song soon sort sound south space speak special speed spend spring stand
star start state station stay step stick still stone stop store story
straight strange street strong student study subject such sudden suffer
suggest summer sun supply support sure surface surprise system table take
talk teach team tell ten term test than thank that the theatre their them
then theory there these theta they thick thin thing think third this those
though thought three through throw thus time to today together too top total
touch toward town trade train travel tree trouble true try turn twenty two
type under understand unit until up upon us use usual value various very view
village visit voice wait walk wall want war warm wash watch water way we wear
week weight well west what wheel when where whether which while white who
whole why wide wife will win wind window wine winter wish with within without
woman wonder wood word work world would write wrong year yes yesterday yet
you young your
"""
"""An ordinary list of common English words, written out so the page is reproducible.

It holds no proper names, no abbreviations and no punctuation, which is what an
ordinary word list is like and is most of what the page's English measurements
turn on. Nothing in it was added for the sake of a demonstration; theta, bled
and own are as ordinary as table, down and there.
"""

ORDINARY_ENGLISH: tuple[str, ...] = tuple(sorted(set(ORDINARY_ENGLISH_RAW.split())))


class WordListKey(StrEnum):
    """Which word list a request is matched against.

    Attributes
    ----------
    ENGLISH:
        The list of common English words written out above.
    PARK:
        The nine words of the wildlife park sentence.
    ORIGINS:
        The five words of the sentence about the origin of life.
    """

    ENGLISH = "english"
    PARK = "park"
    ORIGINS = "origins"


WORD_LISTS: dict[WordListKey, tuple[str, ...]] = {
    WordListKey.ENGLISH: ORDINARY_ENGLISH,
    WordListKey.PARK: PARK_WORDS,
    WordListKey.ORIGINS: ORIGINS_WORDS,
}

WORD_LIST_LABELS: dict[WordListKey, str] = {
    WordListKey.ENGLISH: "common English words",
    WordListKey.PARK: "the wildlife park words",
    WordListKey.ORIGINS: "the origin of life words",
}


class Scan(StrEnum):
    """Which end of a run the greedy match starts from.

    The page's own word for each, so a reader meets a direction rather than a
    setting.
    """

    LEFT_TO_RIGHT = "left to right"
    RIGHT_TO_LEFT = "right to left"
    BOTH = "both"


SCANS: dict[Scan, MatchingDirection] = {
    Scan.LEFT_TO_RIGHT: MatchingDirection.FORWARD,
    Scan.RIGHT_TO_LEFT: MatchingDirection.BACKWARD,
    Scan.BOTH: MatchingDirection.BIDIRECTIONAL,
}


EARLIER_RULES: tuple[tuple[str, PreTokenizer], ...] = (
    ("a run of characters with a space either side", WhitespacePreTokenizer()),
    ("the standard word boundary rule", UnicodeWordPreTokenizer()),
    ("the annotation rules for English", PennTreebankPreTokenizer()),
    ("the rules written for translation", MosesPreTokenizer()),
    (
        "the pattern language models use",
        PatternPreTokenizer(pattern=GPT2_PATTERN),
    ),
)
"""The five rules from earlier in this section, and the page's name for each.

Imported for one contrast only: what each answers for a text with nothing
between its words.
"""


MAX_TEXT_CHARACTERS = 60
"""How long a text the playground may send, a little longer than the running sentence.

The bound is what keeps the exhaustive count honest as well as fast: the number
of ways to cut a text doubles with every character, so a text of sixty already
has more readings than there are grains of sand anybody has counted, and it is
returned as digits rather than as a number for that reason.
"""

MAX_REMOVED_WORDS = 40
"""How many entries one request may strike out of the list it chose."""

MAX_LISTED_READINGS = 12
"""How many shortest readings are listed, however many there turn out to be."""

BoundedText = Annotated[str, Field(max_length=MAX_TEXT_CHARACTERS)]
BoundedWord = Annotated[str, Field(max_length=MAX_TEXT_CHARACTERS)]


# --------------------------------------------------------------------------
# Turning a refusal into the page's own words.
# --------------------------------------------------------------------------

PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("a word dictionary needs at least one entry", "a word list cannot be empty"),
    ("dictionary word", "word of the list"),
    ("dictionary", "word list"),
)
"""The substitutions that take an identifier out of a message a reader sees."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence with the identifiers taken out of it.

    A test asserts that nothing identifier-shaped survives, so a message that
    grows one later fails there rather than reaching the page.
    """
    sentence = str(refusal)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence


# --------------------------------------------------------------------------
# The shapes every endpoint answers in.
# --------------------------------------------------------------------------


class Piece(BaseModel):
    """One piece a text was cut into, and where it came from.

    ``is_entry`` is false exactly when the piece is a lone character the word
    list does not hold, which is the only thing the rule can fall back on.
    """

    text: str
    start: int
    end: int
    is_entry: bool


class Attempt(BaseModel):
    """One candidate substring tried at one position, and whether the list held it."""

    text: str
    is_entry: bool


class WalkStep(BaseModel):
    """What happened at one position of the scan.

    ``at`` is the offset the step started from, counting from the left for a
    left-to-right scan and from the right for the other one. ``attempts`` are
    the candidates in the order they were tried, longest first, and the walk
    stops at the first one the list holds.
    """

    at: int
    attempts: list[Attempt]
    taken: str
    taken_is_entry: bool


class Reading(BaseModel):
    """One whole answer for one text.

    ``n_lookups`` counts the candidate substrings the scan asked about, which is
    the whole of what the method costs. ``chosen_scan`` and ``decided_by`` are
    filled in only when both scans were run and one of them was kept.
    """

    scan: Scan
    pieces: list[Piece]
    n_pieces: int
    n_single_characters: int
    n_entries: int
    n_lookups: int
    walk: list[WalkStep] | None = None
    chosen_scan: Scan | None = None
    decided_by: str | None = None


class Shortest(BaseModel):
    """The fewest pieces any reading of a text could have used.

    Found by searching every reading whose pieces are entries or lone
    characters, so it is the true floor rather than another greedy answer.
    ``n_readings`` is how many readings reach it and is digits rather than a
    number because it can be enormous; ``readings`` lists at most the first few.
    """

    n_pieces: int
    n_readings: str
    readings: list[list[str]]


class Analysis(BaseModel):
    """A text, the list it was matched against, and every answer the page quotes.

    ``n_cuts`` is how many ways the characters could be cut in principle, which
    is two to the power of one less than the length, written as digits.
    ``greedy_reaches_shortest`` says whether the left-to-right answer used as
    few pieces as any reading could.
    """

    text: str
    n_characters: int
    n_cuts: str
    word_list_label: str
    n_words_in_list: int
    longest_entry: int
    readings: list[Reading]
    shortest: Shortest
    greedy_reaches_shortest: bool


class RuleOnText(BaseModel):
    """What one of the earlier word-finding rules answers for one text."""

    rule: str
    n_pieces: int
    pieces: list[str]


class SetupResponse(BaseModel):
    """The opening arithmetic, and what the rules from earlier in the section answer.

    ``n_cuts`` for a text of forty-five characters is the reason a search over
    every reading is not on the table and a rule that commits as it goes is.
    """

    sentence: str
    n_sentence_characters: int
    n_spaces: int
    n_words_with_spaces: int
    spaceless_sentence: str
    n_spaceless_characters: int
    n_cuts: str
    earlier_rules_on_the_spaceless_sentence: list[RuleOnText]
    earlier_rules_on_the_park_sentence: list[RuleOnText]
    park_sentence: str
    n_park_characters: int
    n_park_cuts: str
    n_words_a_reader_finds_in_the_park_sentence: int
    n_english_words: int


class ScenarioView(BaseModel):
    """One of the page's fixed texts, fully worked.

    ``reader_reading`` is what a person answers for the text where that is not
    itself in dispute, and is absent for the sentence whose whole point is that
    two readings are defensible.
    """

    key: str
    label: str
    script: str
    analysis: Analysis
    words_in_list: list[str]
    reader_reading: list[str] | None = None
    reader_reading_is_all_entries: bool | None = None


class ScenariosResponse(BaseModel):
    """Every fixed text of the page, in the order the page meets them."""

    scenarios: list[ScenarioView]


class MissView(BaseModel):
    """One word taken out of a list, and the answers either side of taking it.

    ``n_entries`` inside each reading is the point of the probe: an answer whose
    pieces are all entries looks exactly as finished as one that is right.
    """

    label: str
    text: str
    removed: str
    with_the_word: list[Reading]
    without_the_word: list[Reading]
    reader_reading: list[str]


class MissesResponse(BaseModel):
    """The two probes in which a missing word leaves no trace in the answer."""

    misses: list[MissView]


class SegmentRequest(BaseModel):
    """A text, a word list, entries struck out of it, and which way to scan."""

    text: BoundedText
    word_list: WordListKey = WordListKey.ENGLISH
    removed: list[BoundedWord] = Field(
        default_factory=list, max_length=MAX_REMOVED_WORDS
    )

    model_config = {"extra": "forbid"}


class SegmentResponse(BaseModel):
    """Every scan of the requested text, and the floor a reading could reach.

    ``words_in_list`` is the whole of a short list in the order it was written,
    struck entries included, so a caller can show what was taken out beside what
    was left; ``removed`` says which of them were struck. The long English list
    is not sent, since eight hundred words is not something to show.
    """

    analysis: Analysis
    words_in_list: list[str]
    removed: list[str]


# --------------------------------------------------------------------------
# The work.
# --------------------------------------------------------------------------


def _dictionary(words: tuple[str, ...] | list[str]) -> WordDictionary:
    """The word list as something a scan can be run against."""
    return WordDictionary.from_words(list(words))


def _pieces(
    text: str, dictionary: WordDictionary, direction: MatchingDirection
) -> list[Piece]:
    """The pieces the segmenter answers with, each told whether the list held it."""
    words = MaximumMatchingSegmenter(dictionary=dictionary, direction=direction).split(
        text
    )
    return [
        Piece(
            text=word.text,
            start=word.start,
            end=word.end,
            is_entry=word.text in dictionary,
        )
        for word in words
    ]


class Run:
    """One stretch of a text that whitespace does not interrupt, and where it starts.

    Whitespace ends a run before any matching happens, so a match can never
    reach across a space; the runs are found by the same rule every segmenter in
    this family uses, so the offsets reported here are the offsets the pieces
    carry.
    """

    __slots__ = ("start", "text")

    def __init__(self, text: str, start: int) -> None:
        self.text = text
        self.start = start


def _runs(text: str) -> list[Run]:
    """The stretches of the text that whitespace does not interrupt."""
    return [
        Run(match.group(), match.start()) for match in NON_WHITESPACE_RUN.finditer(text)
    ]


def _walk_left_to_right(run: Run, dictionary: WordDictionary) -> list[WalkStep]:
    """Every candidate tried from the left, and what each position took."""
    steps: list[WalkStep] = []
    position = 0
    while position < len(run.text):
        longest = min(dictionary.longest_word_length, len(run.text) - position)
        taken = run.text[position]
        attempts: list[Attempt] = []
        for length in range(longest, 1, -1):
            candidate = run.text[position : position + length]
            held = candidate in dictionary
            attempts.append(Attempt(text=candidate, is_entry=held))
            if held:
                taken = candidate
                break
        steps.append(
            WalkStep(
                at=run.start + position,
                attempts=attempts,
                taken=taken,
                taken_is_entry=taken in dictionary,
            )
        )
        position += len(taken)
    return steps


def _walk_right_to_left(run: Run, dictionary: WordDictionary) -> list[WalkStep]:
    """The same, from the other end, reported in the order the scan visits."""
    steps: list[WalkStep] = []
    end = len(run.text)
    while end > 0:
        longest = min(dictionary.longest_word_length, end)
        taken = run.text[end - 1]
        attempts: list[Attempt] = []
        for length in range(longest, 1, -1):
            candidate = run.text[end - length : end]
            held = candidate in dictionary
            attempts.append(Attempt(text=candidate, is_entry=held))
            if held:
                taken = candidate
                break
        steps.append(
            WalkStep(
                at=run.start + end,
                attempts=attempts,
                taken=taken,
                taken_is_entry=taken in dictionary,
            )
        )
        end -= len(taken)
    return steps


def _walk(text: str, dictionary: WordDictionary, scan: Scan) -> list[WalkStep]:
    """The walk over every run of the text, one after another."""
    steps: list[WalkStep] = []
    for run in _runs(text):
        if scan is Scan.LEFT_TO_RIGHT:
            steps.extend(_walk_left_to_right(run, dictionary))
        else:
            steps.extend(_walk_right_to_left(run, dictionary))
    return steps


def _decision(text: str, dictionary: WordDictionary) -> tuple[Scan, str]:
    """Which scan the both-ways answer kept, and which of the three rules kept it.

    The rules are fewer pieces, then fewer lone characters, then a standing
    preference for the right-to-left scan, and none of the three is a theorem.
    """
    left = _pieces(text, dictionary, MatchingDirection.FORWARD)
    right = _pieces(text, dictionary, MatchingDirection.BACKWARD)

    if len(left) != len(right):
        winner = Scan.LEFT_TO_RIGHT if len(left) < len(right) else Scan.RIGHT_TO_LEFT
        return winner, "it used fewer pieces"

    left_singles = sum(1 for piece in left if len(piece.text) == 1)
    right_singles = sum(1 for piece in right if len(piece.text) == 1)
    if left_singles != right_singles:
        winner = (
            Scan.LEFT_TO_RIGHT if left_singles < right_singles else Scan.RIGHT_TO_LEFT
        )
        return winner, (
            "the two used the same number of pieces, and it left fewer lone characters"
        )

    return Scan.RIGHT_TO_LEFT, (
        "the two answers were level on both counts, so the standing "
        "preference for the right-to-left scan decided it"
    )


def _reading(text: str, dictionary: WordDictionary, scan: Scan) -> Reading:
    """One scan of one text, with its walk and what the walk cost."""
    pieces = _pieces(text, dictionary, SCANS[scan])
    walk = None if scan is Scan.BOTH else _walk(text, dictionary, scan)

    if scan is Scan.BOTH:
        chosen, decided_by = _decision(text, dictionary)
        n_lookups = sum(
            len(step.attempts)
            for side in (Scan.LEFT_TO_RIGHT, Scan.RIGHT_TO_LEFT)
            for step in _walk(text, dictionary, side)
        )
    else:
        chosen, decided_by = None, None
        n_lookups = sum(len(step.attempts) for step in walk or [])

    return Reading(
        scan=scan,
        pieces=pieces,
        n_pieces=len(pieces),
        n_single_characters=sum(1 for piece in pieces if len(piece.text) == 1),
        n_entries=sum(1 for piece in pieces if piece.is_entry),
        n_lookups=n_lookups,
        walk=walk,
        chosen_scan=chosen,
        decided_by=decided_by,
    )


def _shortest_of_run(run: str, dictionary: WordDictionary) -> tuple[int, int]:
    """The fewest pieces one run admits, and how many readings reach that.

    Every reading is considered, not only the ones a greedy scan could reach,
    which is what makes the floor a fact about the text rather than about the
    rule.
    """
    length = len(run)
    fewest = [0] + [-1] * length
    ways = [1] + [0] * length

    for position in range(1, length + 1):
        best = position
        for size in range(1, min(dictionary.longest_word_length, position) + 1):
            candidate = run[position - size : position]
            if size == 1 or candidate in dictionary:
                best = min(best, fewest[position - size] + 1)
        fewest[position] = best
        for size in range(1, min(dictionary.longest_word_length, position) + 1):
            candidate = run[position - size : position]
            if (size == 1 or candidate in dictionary) and fewest[
                position - size
            ] + 1 == best:
                ways[position] += ways[position - size]

    return fewest[length], ways[length]


def _shortest_readings(
    run: str, dictionary: WordDictionary, target: int, wanted: int
) -> list[list[str]]:
    """At most ``wanted`` of the readings that use exactly ``target`` pieces."""
    found: list[list[str]] = []

    def walk(position: int, so_far: list[str]) -> None:
        if len(found) >= wanted or len(so_far) > target:
            return
        if position == len(run):
            if len(so_far) == target:
                found.append(list(so_far))
            return
        for size in range(
            1, min(dictionary.longest_word_length, len(run) - position) + 1
        ):
            candidate = run[position : position + size]
            if size == 1 or candidate in dictionary:
                walk(position + size, so_far + [candidate])

    walk(0, [])
    return found


def _shortest(text: str, dictionary: WordDictionary) -> Shortest:
    """The floor over the whole text, which is the sum of the floors of its runs."""
    runs = _runs(text)
    if not runs:
        return Shortest(n_pieces=0, n_readings="1", readings=[])

    total_pieces = 0
    total_ways = 1
    for run in runs:
        pieces, ways = _shortest_of_run(run.text, dictionary)
        total_pieces += pieces
        total_ways *= ways

    listed: list[list[str]] = []
    if len(runs) == 1:
        listed = _shortest_readings(
            runs[0].text, dictionary, total_pieces, MAX_LISTED_READINGS
        )

    return Shortest(n_pieces=total_pieces, n_readings=str(total_ways), readings=listed)


def _n_cuts(text: str) -> str:
    """How many ways the characters of a text could be cut, as digits.

    A text of ``n`` characters has ``n - 1`` places a boundary could go and each
    is free, so the count is two to that power. It counts every cut and not only
    the readings a word list allows, which is exactly why it is so large.
    """
    if len(text) < 2:
        return "1"
    return str(2 ** (len(text) - 1))


def _analysis(text: str, words: tuple[str, ...] | list[str], label: str) -> Analysis:
    """One text against one word list, scanned every way, with the floor beside it."""
    dictionary = _dictionary(words)
    readings = [_reading(text, dictionary, scan) for scan in Scan]
    shortest = _shortest(text, dictionary)
    left_to_right = next(
        reading for reading in readings if reading.scan is Scan.LEFT_TO_RIGHT
    )

    return Analysis(
        text=text,
        n_characters=len(text),
        n_cuts=_n_cuts(text),
        word_list_label=label,
        n_words_in_list=dictionary.n_words,
        longest_entry=dictionary.longest_word_length,
        readings=readings,
        shortest=shortest,
        greedy_reaches_shortest=left_to_right.n_pieces == shortest.n_pieces,
    )


def _rule_on(text: str) -> list[RuleOnText]:
    """What each of the earlier word-finding rules answers for one text."""
    return [
        RuleOnText(
            rule=label,
            n_pieces=rule.split(text).n_words,
            pieces=list(rule.split(text).texts),
        )
        for label, rule in EARLIER_RULES
    ]


@cache
def _setup() -> SetupResponse:
    """The opening arithmetic, computed once because it takes no input."""
    return SetupResponse(
        sentence=RUNNING_SENTENCE,
        n_sentence_characters=len(RUNNING_SENTENCE),
        n_spaces=RUNNING_SENTENCE.count(" "),
        n_words_with_spaces=WhitespacePreTokenizer().split(RUNNING_SENTENCE).n_words,
        spaceless_sentence=SPACELESS_SENTENCE,
        n_spaceless_characters=len(SPACELESS_SENTENCE),
        n_cuts=_n_cuts(SPACELESS_SENTENCE),
        earlier_rules_on_the_spaceless_sentence=_rule_on(SPACELESS_SENTENCE),
        earlier_rules_on_the_park_sentence=_rule_on(PARK_SENTENCE),
        park_sentence=PARK_SENTENCE,
        n_park_characters=len(PARK_SENTENCE),
        n_park_cuts=_n_cuts(PARK_SENTENCE),
        n_words_a_reader_finds_in_the_park_sentence=5,
        n_english_words=len(ORDINARY_ENGLISH),
    )


@router.get("/setup", response_model=SetupResponse)
def setup() -> SetupResponse:
    """The sentence with its spaces gone, and what the earlier rules make of it."""
    return _setup()


@cache
def _scenarios() -> ScenariosResponse:
    """The five fixed texts, each worked every way, computed once."""
    english_label = WORD_LIST_LABELS[WordListKey.ENGLISH]
    park_label = WORD_LIST_LABELS[WordListKey.PARK]
    origins_label = WORD_LIST_LABELS[WordListKey.ORIGINS]

    park_dictionary = _dictionary(PARK_WORDS)
    english_dictionary = _dictionary(ORDINARY_ENGLISH)
    origins_dictionary = _dictionary(ORIGINS_WORDS)

    park_reading = ["我们", "在", "野生", "动物园", "玩"]
    table_reading = ["the", "table", "down", "there"]
    sentence_reading = list(WhitespacePreTokenizer().split(RUNNING_SENTENCE).texts)

    return ScenariosResponse(
        scenarios=[
            ScenarioView(
                key="spaceless",
                label="our sentence with its spaces taken out",
                script="English",
                analysis=_analysis(SPACELESS_SENTENCE, ORDINARY_ENGLISH, english_label),
                words_in_list=[],
                reader_reading=sentence_reading,
                reader_reading_is_all_entries=all(
                    word in english_dictionary for word in sentence_reading
                ),
            ),
            ScenarioView(
                key="table",
                label="four English words run together",
                script="English",
                analysis=_analysis(TABLE_TEXT, ORDINARY_ENGLISH, english_label),
                words_in_list=[],
                reader_reading=table_reading,
                reader_reading_is_all_entries=all(
                    word in english_dictionary for word in table_reading
                ),
            ),
            ScenarioView(
                key="park",
                label="we play at the wildlife park",
                script="Chinese",
                analysis=_analysis(PARK_SENTENCE, PARK_WORDS, park_label),
                words_in_list=list(PARK_WORDS),
                reader_reading=park_reading,
                reader_reading_is_all_entries=all(
                    word in park_dictionary for word in park_reading
                ),
            ),
            ScenarioView(
                key="origins",
                label="the sentence with two defensible readings",
                script="Chinese",
                analysis=_analysis(ORIGINS_SENTENCE, ORIGINS_WORDS, origins_label),
                words_in_list=list(ORIGINS_WORDS),
                reader_reading=None,
                reader_reading_is_all_entries=None,
            ),
            ScenarioView(
                key="name",
                label="the same sentence behind a surname",
                script="Chinese",
                analysis=_analysis(NAME_SENTENCE, ORIGINS_WORDS, origins_label),
                words_in_list=list(ORIGINS_WORDS),
                reader_reading=list(NAME_READING),
                reader_reading_is_all_entries=all(
                    word in origins_dictionary for word in NAME_READING
                ),
            ),
        ]
    )


@router.get("/scenarios", response_model=ScenariosResponse)
def scenarios() -> ScenariosResponse:
    """The page's five texts, each scanned both ways and measured against the floor."""
    return _scenarios()


@cache
def _misses() -> MissesResponse:
    """The two probes in which taking a word out leaves the answer looking finished."""
    park_without = tuple(word for word in PARK_WORDS if word != "动物园")
    english_without = tuple(word for word in ORDINARY_ENGLISH if word != "table")

    def both_scans(text: str, words: tuple[str, ...]) -> list[Reading]:
        dictionary = _dictionary(words)
        return [
            _reading(text, dictionary, scan)
            for scan in (Scan.LEFT_TO_RIGHT, Scan.RIGHT_TO_LEFT)
        ]

    return MissesResponse(
        misses=[
            MissView(
                label="the wildlife park sentence, without the word for the park",
                text=PARK_SENTENCE,
                removed="动物园",
                with_the_word=both_scans(PARK_SENTENCE, PARK_WORDS),
                without_the_word=both_scans(PARK_SENTENCE, park_without),
                reader_reading=["我们", "在", "野生", "动物园", "玩"],
            ),
            MissView(
                label="four English words run together, without the noun in the middle",
                text=TABLE_TEXT,
                removed="table",
                with_the_word=both_scans(TABLE_TEXT, ORDINARY_ENGLISH),
                without_the_word=both_scans(TABLE_TEXT, english_without),
                reader_reading=["the", "table", "down", "there"],
            ),
        ]
    )


@router.get("/misses", response_model=MissesResponse)
def misses() -> MissesResponse:
    """What an answer looks like when the word list is missing a word the text uses."""
    return _misses()


@router.post("/segment", response_model=SegmentResponse)
def segment(request: SegmentRequest) -> SegmentResponse:
    """Scan the requested text every way, against the chosen list less what was struck out."""
    struck = set(request.removed)
    words = tuple(word for word in WORD_LISTS[request.word_list] if word not in struck)

    try:
        analysis = _analysis(request.text, words, WORD_LIST_LABELS[request.word_list])
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    whole = WORD_LISTS[request.word_list]
    listed = [] if request.word_list is WordListKey.ENGLISH else list(whole)

    return SegmentResponse(
        analysis=analysis,
        words_in_list=listed,
        removed=[word for word in whole if word in struck],
    )
