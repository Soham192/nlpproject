"""Sensitive-span detection: Presidio (regex recognizers + spaCy NER) on the full joined transcript.

Detection runs on the whole transcript, not per segment, so an SSN spoken
across two Whisper segments still matches. Every character of the joined text
maps back to a word index (Transcript.word_char_spans).

Spoken numbers are normalised to digits first ("four five three two" -> "4532"),
otherwise Presidio's card/SSN regexes find nothing.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from .alignment import Word
from .config import DetectionConfig

log = logging.getLogger("redact.detect")


@dataclass
class Detection:
    entity_type: str
    score: float
    char_start: int
    char_end: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Transcript:
    """Joined transcript text plus, for every word, its [start, end) char range in `text`."""
    words: list[Word]
    text: str
    word_char_spans: list[tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Spoken-number normalisation
# ---------------------------------------------------------------------------

_UNITS = {"zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 2, "thirty": 3, "forty": 4, "fourty": 4, "fifty": 5, "sixty": 6, "seventy": 7,
         "eighty": 8, "ninety": 9}
_REPEAT = {"double": 2, "triple": 3}
_AMBIGUOUS_ZERO = {"oh", "o"}  # only a digit when next to other number words
_GROUP_PUNCT = set(",;:-/")      # pause inside a number -> group separator
_END_PUNCT = set(".?!")          # sentence end -> number run ends


def _split(raw: str) -> tuple[str, str, str]:
    """'(four,' -> ('(', 'four', ',')"""
    m = re.match(r"^(\W*)(.*?)(\W*)$", raw)
    return m.group(1), m.group(2), m.group(3)


def _kind(core: str) -> str | None:
    c = core.lower()
    if re.fullmatch(r"\d+", c):
        return "literal"
    if re.fullmatch(r"\d+(-\d+)+", c):
        return "literal_grouped"
    if c in _UNITS:
        return "unit"
    if c in _TEENS:
        return "teen"
    if c in _TENS:
        return "tens"
    if c in _REPEAT:
        return "repeat"
    if c == "hundred":
        return "hundred"
    if "-" in c and all(p in _TENS or p in _UNITS for p in c.split("-")) and c.split("-")[0] in _TENS:
        return "compound"  # forty-five
    return None


@dataclass
class _Piece:
    digits: str
    open: int = 0          # trailing zeros a following word may fill ("40" -> 1, "400" -> 2)
    literal: bool = False
    words: list[int] = field(default_factory=list)
    sep_before: str = ""


def _parse_run(items: list[tuple[int, str, str]]) -> list[_Piece]:
    """items: [(word_idx, core_lower, trailing_punct)] for a run of number words."""
    pieces: list[_Piece] = []
    repeat: tuple[int, int] | None = None  # (count, word idx)
    pending_sep = ""

    def new(p: _Piece) -> None:
        nonlocal pending_sep
        if pieces and not pending_sep and p.literal and pieces[-1].literal:
            pending_sep = "-"  # "555 123 4567": Whisper's own grouping
        p.sep_before = pending_sep if pieces else ""
        pending_sep = ""
        pieces.append(p)

    for idx, core, trail in items:
        k = _kind(core)
        last = pieces[-1] if pieces and not pending_sep else None
        if k == "literal_grouped" and all(len(g) == 1 for g in core.split("-")):
            # Whisper writes digit-by-digit speech as "4-1-1-1": one spoken group.
            new(_Piece(core.replace("-", ""), literal=True, words=[idx]))
        elif k in ("literal", "literal_grouped"):
            new(_Piece(core, literal=True, words=[idx]))
        elif k == "repeat":
            repeat = (_REPEAT[core], idx)
        elif k == "unit":
            d = str(_UNITS[core])
            if repeat:
                new(_Piece(d * repeat[0], words=[repeat[1], idx]))
                repeat = None
            elif last and last.open >= 1:
                last.digits = last.digits[:-1] + d
                last.open = 0
                last.words.append(idx)
            else:
                new(_Piece(d, words=[idx]))
        elif k == "teen":
            v = str(_TEENS[core])
            if last and last.open == 2:
                last.digits = last.digits[:-2] + v
                last.open = 0
                last.words.append(idx)
            else:
                new(_Piece(v, words=[idx]))
        elif k == "tens":
            t = str(_TENS[core])
            if last and last.open == 2:
                last.digits = last.digits[:-2] + t + "0"
                last.open = 1
                last.words.append(idx)
            else:
                new(_Piece(t + "0", open=1, words=[idx]))
        elif k == "compound":
            a, b = core.split("-")
            v = str(_TENS[a]) + str(_UNITS[b])
            if last and last.open == 2:
                last.digits = last.digits[:-2] + v
                last.open = 0
                last.words.append(idx)
            else:
                new(_Piece(v, words=[idx]))
        elif k == "hundred":
            if last and len(last.digits) == 1 and not last.literal:
                last.digits += "00"
                last.open = 2
                last.words.append(idx)
            else:
                new(_Piece("100", open=2, words=[idx]))
        if trail and set(trail) & _GROUP_PUNCT:
            pending_sep = "-"
    return pieces


def build_transcript(words: list[Word], normalize_numbers: bool = True) -> Transcript:
    """Join words into one text with per-word char spans, normalising spoken numbers if asked."""
    n = len(words)
    parts = [_split(w.text.strip()) for w in words]
    kinds = [_kind(core) for _, core, _ in parts] if normalize_numbers else [None] * n

    # "oh"/"o" only count as zero when adjacent to another number word.
    for i in range(n):
        if kinds[i] == "unit" and parts[i][1].lower() in _AMBIGUOUS_ZERO:
            nb = [kinds[j] for j in (i - 1, i + 1) if 0 <= j < n and kinds[j] and
                  parts[j][1].lower() not in _AMBIGUOUS_ZERO]
            if not nb:
                kinds[i] = None
    # "double"/"triple" only when followed by a unit.
    for i in range(n):
        if kinds[i] == "repeat" and not (i + 1 < n and kinds[i + 1] == "unit"):
            kinds[i] = None

    text_parts: list[str] = []
    spans: list[tuple[int, int]] = [(0, 0)] * n
    pos = 0

    def emit(s: str) -> None:
        nonlocal pos
        text_parts.append(s)
        pos += len(s)

    i = 0
    while i < n:
        if kinds[i] is None:
            if i:
                emit(" ")
            start = pos
            emit(words[i].text.strip())
            spans[i] = (start, pos)
            i += 1
            continue
        # Collect a run of number words; a sentence-ending mark closes it.
        j = i
        while j < n and kinds[j] is not None:
            lead, _, trail = parts[j]
            if j > i and lead:
                break
            j += 1
            if set(trail) & _END_PUNCT:
                break
        items = [(k, parts[k][1].lower(), parts[k][2]) for k in range(i, j)]
        pieces = _parse_run(items)
        n_digits = sum(len(re.sub(r"\D", "", p.digits)) for p in pieces)
        spoken_only = not any(p.literal for p in pieces)
        if not pieces or (spoken_only and n_digits < 3):
            # Short spoken numbers ("one moment", "two weeks") stay as words.
            for k in range(i, j):
                if k:
                    emit(" ")
                start = pos
                emit(words[k].text.strip())
                spans[k] = (start, pos)
            i = j
            continue
        if i:
            emit(" ")
        emit(parts[i][0])
        for p in pieces:
            emit(p.sep_before)
            start = pos
            emit(p.digits)
            for k in p.words:
                spans[k] = (start, pos)
        # Words that contributed nothing (shouldn't happen) still get a position.
        for k in range(i, j):
            if spans[k] == (0, 0) and not (k == 0 and pos == 0):
                spans[k] = (pos, pos)
        emit(parts[j - 1][2])
        i = j
    return Transcript(words=words, text="".join(text_parts), word_char_spans=spans)


# ---------------------------------------------------------------------------
# Presidio
# ---------------------------------------------------------------------------

def _custom_recognizers():
    from presidio_analyzer import Pattern, PatternRecognizer

    return [
        PatternRecognizer(
            supported_entity="ACCOUNT_NUMBER",
            patterns=[Pattern("account_number", r"\b\d{6,12}\b", 0.3),
                      Pattern("account_number_grouped", r"\b\d{2,6}(?:-\d{2,6}){1,3}\b", 0.2)],
            context=["account", "acct", "acc", "routing", "member", "customer"],
        ),
        PatternRecognizer(
            supported_entity="MEDICAL_RECORD_NUMBER",
            patterns=[Pattern("mrn", r"\b(?:MRN[-\s]?)?\d{6,10}\b", 0.2)],
            context=["mrn", "medical", "record", "chart", "patient"],
        ),
        PatternRecognizer(
            supported_entity="POLICY_NUMBER",
            patterns=[Pattern("policy", r"\b[A-Z]{1,3}-?\d{6,10}\b", 0.3),
                      Pattern("policy_digits", r"\b\d{8,12}\b", 0.1)],
            context=["policy", "insurance", "claim", "plan"],
        ),
        # Presidio's US_SSN validator rejects some test-range numbers spoken in
        # demos; this catches the canonical dashed form with context.
        PatternRecognizer(
            supported_entity="US_SSN",
            patterns=[Pattern("ssn_dashed", r"\b\d{3}-\d{2}-\d{4}\b", 0.5)],
            context=["social", "security", "ssn", "ss"],
            name="SpokenSsnRecognizer",
        ),
    ]


@lru_cache(maxsize=2)
def get_analyzer(spacy_model: str):
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": spacy_model}],
    }).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    for r in _custom_recognizers():
        analyzer.registry.add_recognizer(r)
    return analyzer


def detect(transcript: Transcript, cfg: DetectionConfig) -> list[Detection]:
    """Run Presidio over the full joined transcript. Returns all detections (any score), sorted."""
    # Presidio treats an empty entity list as "all entities"; an empty selection must mean none.
    if not transcript.text.strip() or not cfg.entities:
        return []
    analyzer = get_analyzer(cfg.spacy_model)
    results = analyzer.analyze(text=transcript.text, language="en", entities=list(cfg.entities))
    dets = [Detection(r.entity_type, round(float(r.score), 3), r.start, r.end, transcript.text[r.start:r.end])
            for r in results]
    return sorted(dets, key=lambda d: (d.char_start, -d.score))


def split_by_threshold(dets: list[Detection], threshold: float) -> tuple[list[Detection], list[Detection]]:
    """(to_mask, not_masked). Low-score hits overlapping a masked hit are dropped as redundant."""
    keep = [d for d in dets if d.score >= threshold]
    low = [d for d in dets if d.score < threshold
           and not any(k.char_start < d.char_end and d.char_start < k.char_end for k in keep)]
    return keep, low


def detect_words(words: list[Word], cfg: DetectionConfig) -> tuple[Transcript, list[Detection]]:
    t = build_transcript(words, cfg.normalize_spoken_numbers)
    return t, detect(t, cfg)
