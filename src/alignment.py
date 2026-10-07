"""Word-level timestamps via WhisperX (wav2vec2 CTC forced alignment).

Whisper's own segment timestamps are too coarse for masking; this stage exists
so mask boundaries land on word edges.

The English wav2vec2 aligner's vocabulary has no digits, so a token like "4532"
would come back without a timestamp. We therefore spell digits out ("four five
three two") before aligning and fold the timings back onto the original token.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from .config import AlignmentConfig, resolve_device

log = logging.getLogger("redact.alignment")

_DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


@dataclass
class Word:
    text: str
    start: float
    end: float
    score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def spell_digits(token: str) -> list[str]:
    """'4532,' -> ['four', 'five', 'three', 'two,']; tokens without digits are returned as-is."""
    if not re.search(r"\d", token):
        return [token]
    spelled = [_DIGIT_WORDS[int(c)] for c in token if c.isdigit()]
    trail = re.search(r"[^\w]+$", token)
    if trail:
        spelled[-1] += trail.group(0)
    return spelled


def _expand_segments(segments: list[dict]) -> tuple[list[dict], list[tuple[str, int]]]:
    """Spell out digits in segment text. Returns new segments + [(orig_token, n_expanded)]."""
    out, plan = [], []
    for seg in segments:
        words = []
        for tok in seg["text"].split():
            exp = spell_digits(tok)
            plan.append((tok, len(exp)))
            words.extend(exp)
        out.append({**seg, "text": " ".join(words)})
    return out, plan


def _fold(aligned: list[dict], plan: list[tuple[str, int]]) -> list[Word]:
    words, i = [], 0
    for tok, n in plan:
        chunk = aligned[i:i + n]
        i += n
        starts = [w["start"] for w in chunk if w.get("start") is not None]
        ends = [w["end"] for w in chunk if w.get("end") is not None]
        scores = [w["score"] for w in chunk if w.get("score") is not None]
        words.append(Word(
            text=tok,
            start=min(starts) if starts else float("nan"),
            end=max(ends) if ends else float("nan"),
            score=float(np.mean(scores)) if scores else None,
        ))
    return words


def fill_missing_times(words: list[Word], segments: list[dict]) -> list[Word]:
    """Words the aligner couldn't place get times interpolated between neighbours."""
    n = len(words)
    lo = segments[0]["start"] if segments else 0.0
    hi = segments[-1]["end"] if segments else 0.0
    i = 0
    while i < n:
        if not np.isnan(words[i].start) and not np.isnan(words[i].end):
            i += 1
            continue
        j = i
        while j < n and (np.isnan(words[j].start) or np.isnan(words[j].end)):
            j += 1
        left = words[i - 1].end if i > 0 else lo
        right = words[j].start if j < n else hi
        if right < left:
            right = left
        step = (right - left) / (j - i)
        for k in range(i, j):
            words[k].start = left + step * (k - i)
            words[k].end = left + step * (k - i + 1)
        log.warning("interpolated timestamps for %d unaligned word(s) starting at %r", j - i, words[i].text)
        i = j
    return words


@lru_cache(maxsize=2)
def _load(language: str, model_name: str, device: str):
    import whisperx

    log.info("loading alignment model %s on %s", model_name, device)
    return whisperx.load_align_model(language_code=language, device=device, model_name=model_name)


def align(segments: list[dict], audio: np.ndarray, cfg: AlignmentConfig, language: str = "en") -> list[Word]:
    """segments: from asr.transcribe; audio: float32 mono 16 kHz. Returns words in order."""
    if cfg.use_mfa:
        raise NotImplementedError("Montreal Forced Aligner fallback is not wired up; set alignment.use_mfa: false")
    if not segments:
        return []
    import whisperx

    device = resolve_device(cfg.device)
    model, metadata = _load(language, cfg.model, device)
    expanded, plan = _expand_segments(segments)
    result = whisperx.align(expanded, model, metadata, audio, device, return_char_alignments=False)
    aligned = result["word_segments"]
    if len(aligned) != sum(n for _, n in plan):
        log.warning("aligner returned %d words, expected %d; re-tokenising from segment text",
                    len(aligned), sum(n for _, n in plan))
        aligned = [w for seg in result["segments"] for w in seg.get("words", [])]
        if len(aligned) != sum(n for _, n in plan):
            raise RuntimeError("could not reconcile aligned words with transcript tokens")
    return fill_missing_times(_fold(aligned, plan), segments)
