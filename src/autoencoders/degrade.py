"""Synthetic test material for layers 1 and 5 (generated, nothing downloaded).

Layer 1 positives ("not usable speech"): pure tone, digital silence, heavy clipping, DC offset,
white noise, heavy mu-law. Layer 5: masked spans whose fill (silence / tone / low-level noise)
does or does not contain a real speech fragment at one edge.
"""
from __future__ import annotations

import numpy as np
import torch

from ..mask import replacement_signal
from .data import HOP, SR, mulaw_roundtrip

INGEST_KINDS = ("tone", "silence", "clipping", "dc_offset", "white_noise", "mulaw_heavy")


def to_pcm_grid(x: np.ndarray) -> np.ndarray:
    """Quantise to the 16-bit grid, as if written to and read back from a PCM WAV."""
    return (np.clip(np.round(x * 32767), -32768, 32767) / 32768.0).astype(np.float32)


def dbfs_to_rms(dbfs: float) -> float:
    return 10 ** (dbfs / 20)


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-10))


# --------------------------------------------------------------------------- layer 1

def make_unusable(kind: str, speech: np.ndarray, rng: np.random.Generator, c: dict) -> np.ndarray:
    """A 'not usable speech' file of the same length as `speech`. `c` = config autoencoder.eval.ingest."""
    n = len(speech)
    if kind == "tone":
        f = rng.uniform(*c["tone_hz"])
        x = 0.3 * np.sin(2 * np.pi * f * np.arange(n) / SR + rng.uniform(0, 2 * np.pi))
    elif kind == "silence":
        x = np.zeros(n)
    elif kind == "clipping":
        x = np.clip(speech * c["clip_gain"], -1, 1)
    elif kind == "dc_offset":
        x = np.clip(speech + c["dc_offset"], -1, 1)
    elif kind == "white_noise":
        x = dbfs_to_rms(c["noise_dbfs"]) * rng.standard_normal(n)
    elif kind == "mulaw_heavy":
        x = mulaw_roundtrip(torch.from_numpy(speech.astype(np.float32)), c["mulaw_heavy_channels"]).numpy()
    else:
        raise ValueError(f"unknown kind {kind!r}; expected one of {INGEST_KINDS}")
    return to_pcm_grid(x)


# --------------------------------------------------------------------------- layer 5

def active_frames(audio: np.ndarray, rel_db: float) -> np.ndarray:
    """Energy VAD on 10 ms frames: active where frame RMS > (loudest frame + rel_db)."""
    n = len(audio) // HOP
    fr = audio[:n * HOP].reshape(n, HOP).astype(np.float64)
    db = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-10)
    return db > db.max() + rel_db


def place_span(audio: np.ndarray, rng: np.random.Generator, span_ms: tuple[float, float], frag_max_ms: float,
               rel_db: float) -> tuple[int, int, str] | None:
    """Pick [s, e) and an edge so the edge's first/last `frag_max_ms` lies entirely in active speech.

    Using the largest fragment length for placement means every fragment length shares the same span.
    Returns None if the utterance has no suitable position.
    """
    act = active_frames(audio, rel_db)
    k = int(np.ceil(frag_max_ms / 10))
    length = int(rng.uniform(*span_ms) / 1000 * SR)
    edge = "leading" if rng.random() < 0.5 else "trailing"
    ok = np.convolve(act.astype(int), np.ones(k, int), mode="valid") == k  # ok[i]: frames i..i+k-1 active
    if edge == "leading":
        cands = [i * HOP for i in np.flatnonzero(ok) if i * HOP + length <= len(audio)]
        if not cands:
            return None
        s = int(rng.choice(cands))
        return s, s + length, edge
    cands = [(i + k) * HOP for i in np.flatnonzero(ok) if (i + k) * HOP - length >= 0]
    if not cands:
        return None
    e = int(rng.choice(cands))
    return e - length, e, edge


def fill_signal(kind: str, n: int, rng: np.random.Generator, c: dict) -> np.ndarray:
    """What a masked span contains. tone reuses src.mask.replacement_signal. `c` = autoencoder.eval.leak."""
    if kind == "silence":
        return np.zeros(n, np.float32)
    if kind == "tone":
        return replacement_signal(n, 1, SR, "tone", c["tone_hz"])[:, 0].astype(np.float32) / 32768.0
    if kind == "noise":
        return to_pcm_grid(dbfs_to_rms(c["noise_dbfs"]) * rng.standard_normal(n))
    raise ValueError(f"unknown fill {kind!r} (silence|tone|noise)")


def build_span(audio: np.ndarray, s: int, e: int, edge: str, frag_ms: float, fill: np.ndarray) -> np.ndarray:
    """The span's samples after masking: fill everywhere except a `frag_ms` speech fragment at `edge`.

    frag_ms == 0 is a clean (negative) span. Models a masked range that misses the entity's first or
    last few ms because the padding was too small.
    """
    n = e - s
    f = int(round(frag_ms / 1000 * SR))
    out = fill[:n].copy()
    if f:
        if edge == "leading":
            out[:f] = audio[s:s + f]
        else:
            out[n - f:] = audio[e - f:e]
    return to_pcm_grid(out)


def max_frame_rms_db(x: np.ndarray) -> float:
    """Energy baseline for layer 5: loudest 10 ms frame in the span."""
    n = len(x) // HOP
    fr = x[:n * HOP].reshape(n, HOP).astype(np.float64)
    return float(np.max(20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-10)))
