from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.alignment import Word
from src.audio import AudioParams, write_wav
from src.config import load_config
from src.detect import Detection

SR = 16000


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    return dataclasses.replace(c, output=dataclasses.replace(c.output, dir=str(tmp_path / "outputs")))


@pytest.fixture
def key() -> bytes:
    return bytes(range(32))


def make_wav(path: Path, seconds: float = 4.0, seed: int = 0, extra_chunk: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    sig = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(n)
    samples = (sig * 32767).clip(-32768, 32767).astype(np.int16)[:, None]
    write_wav(path, samples, AudioParams(SR, 1, 2, n))
    if extra_chunk:  # emulate ffmpeg's LIST/INFO chunk before 'data'
        raw = path.read_bytes()
        info = b"LIST" + (len(b"INFO" + b"ISFT" + (6).to_bytes(4, "little") + b"Lavf6\x00")).to_bytes(4, "little") \
            + b"INFO" + b"ISFT" + (6).to_bytes(4, "little") + b"Lavf6\x00"
        body = raw[12:36] + info + raw[36:]
        path.write_bytes(b"RIFF" + (4 + len(body)).to_bytes(4, "little") + b"WAVE" + body)
    return path


@pytest.fixture
def wav(tmp_path) -> Path:
    return make_wav(tmp_path / "in.wav")


# "my card is 4532 0151 1283 0366 thanks" — 4 s of audio, words with timings.
WORDS = [
    Word("my", 0.10, 0.30), Word("card", 0.30, 0.60), Word("is", 0.60, 0.75),
    Word("four", 0.80, 1.00), Word("five", 1.00, 1.20), Word("three", 1.20, 1.40), Word("two", 1.40, 1.60),
    Word("zero", 1.70, 1.90), Word("one", 1.90, 2.10), Word("five", 2.10, 2.30), Word("one", 2.30, 2.50),
    Word("thanks.", 3.20, 3.60),
]


def stub_pipeline(monkeypatch, words: list[Word], detections_fn):
    """Replace ASR/alignment/Presidio with fixtures; everything else runs for real."""
    from src import detect as det_mod
    from src.pipeline import Pipeline

    monkeypatch.setattr(Pipeline, "run_asr", lambda self, samples: {"language": "en", "segments": []})
    monkeypatch.setattr(Pipeline, "run_alignment", lambda self, res, samples: words)
    monkeypatch.setattr(det_mod, "detect", lambda transcript, cfg: detections_fn(transcript))


def det(t, entity: str, needle: str, score: float = 0.9) -> Detection:
    i = t.text.index(needle)
    return Detection(entity, score, i, i + len(needle), needle)
