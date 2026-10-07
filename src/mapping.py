"""char offset -> word index -> audio sample range (with padding and merging)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .alignment import Word
from .detect import Detection, Transcript


@dataclass
class SampleSpan:
    """A merged masking range plus the detections it covers."""
    sample_start: int
    sample_end: int
    detections: list[Detection] = field(default_factory=list)
    word_start_idx: int = 0
    word_end_idx: int = 0  # exclusive

    @property
    def entity_type(self) -> str:
        return "+".join(dict.fromkeys(d.entity_type for d in self.detections))

    @property
    def score(self) -> float:
        return max(d.score for d in self.detections)

    @property
    def char_start(self) -> int:
        return min(d.char_start for d in self.detections)

    @property
    def char_end(self) -> int:
        return max(d.char_end for d in self.detections)


def chars_to_words(t: Transcript, char_start: int, char_end: int) -> tuple[int, int]:
    """Word index range [first, last+1) touched by [char_start, char_end)."""
    idx = [i for i, (ws, we) in enumerate(t.word_char_spans) if ws < char_end and we > char_start]
    if not idx:
        raise ValueError(f"char range [{char_start},{char_end}) matches no word")
    return idx[0], idx[-1] + 1


def words_to_seconds(words: list[Word], w0: int, w1: int) -> tuple[float, float]:
    ws = words[w0:w1]
    return min(w.start for w in ws), max(w.end for w in ws)


def seconds_to_samples(start: float, end: float, sample_rate: int, padding_ms: int, total: int) -> tuple[int, int]:
    pad = int(round(padding_ms * sample_rate / 1000))
    s = max(0, math.floor(start * sample_rate) - pad)
    e = min(total, math.ceil(end * sample_rate) + pad)
    return s, e


def merge_ranges(spans: list[SampleSpan]) -> list[SampleSpan]:
    """Sort and merge overlapping or touching spans so no sample is encrypted twice."""
    out: list[SampleSpan] = []
    for s in sorted(spans, key=lambda x: (x.sample_start, x.sample_end)):
        if out and s.sample_start <= out[-1].sample_end:
            last = out[-1]
            last.sample_end = max(last.sample_end, s.sample_end)
            last.detections.extend(s.detections)
            last.word_start_idx = min(last.word_start_idx, s.word_start_idx)
            last.word_end_idx = max(last.word_end_idx, s.word_end_idx)
        else:
            out.append(SampleSpan(s.sample_start, s.sample_end, list(s.detections), s.word_start_idx, s.word_end_idx))
    return out


def map_detections(
    detections: list[Detection],
    transcript: Transcript,
    sample_rate: int,
    total_samples: int,
    padding_ms: int,
) -> list[SampleSpan]:
    spans = []
    for d in detections:
        w0, w1 = chars_to_words(transcript, d.char_start, d.char_end)
        t0, t1 = words_to_seconds(transcript.words, w0, w1)
        s, e = seconds_to_samples(t0, t1, sample_rate, padding_ms, total_samples)
        if e > s:
            spans.append(SampleSpan(s, e, [d], w0, w1))
    return merge_ranges(spans)
