"""WAV/PCM I/O. PCM-only by design: lossy codecs break sample-exact reconstruction."""
from __future__ import annotations

import hashlib
import os
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AudioConfig


class AudioFormatError(ValueError):
    pass


@dataclass(frozen=True)
class AudioParams:
    sample_rate: int
    channels: int
    sample_width_bytes: int
    total_samples: int  # frames (samples per channel)

    @property
    def duration_sec(self) -> float:
        return self.total_samples / self.sample_rate


def load_wav(path: str | Path, expected: AudioConfig | None = None) -> tuple[np.ndarray, AudioParams]:
    """Load a PCM WAV as int16 array of shape (frames, channels).

    Rejects non-WAV input and, if `expected` is given, any format mismatch.
    Never transcodes.
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as w:
            params = AudioParams(
                sample_rate=w.getframerate(),
                channels=w.getnchannels(),
                sample_width_bytes=w.getsampwidth(),
                total_samples=w.getnframes(),
            )
            raw = w.readframes(params.total_samples)
    except (wave.Error, EOFError) as e:
        raise AudioFormatError(
            f"{path}: not a PCM WAV file ({e}). Convert to 16 kHz mono 16-bit PCM WAV first, e.g.\n"
            f"  ffmpeg -i {path.name} -ac 1 -ar 16000 -c:a pcm_s16le out.wav"
        ) from e

    if expected is not None:
        want = (expected.sample_rate, expected.channels, expected.sample_width_bytes)
        got = (params.sample_rate, params.channels, params.sample_width_bytes)
        if want != got:
            raise AudioFormatError(
                f"{path}: format (rate, channels, width)={got}, expected {want}. "
                "Refusing to transcode silently; convert on ingest with ffmpeg."
            )
    if params.sample_width_bytes != 2:
        raise AudioFormatError(f"{path}: only 16-bit PCM is supported, got {8 * params.sample_width_bytes}-bit")

    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, params.channels).copy()
    return samples, params


def write_wav(path: str | Path, samples: np.ndarray, params: AudioParams) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(samples, dtype="<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(params.channels)
        w.setsampwidth(params.sample_width_bytes)
        w.setframerate(params.sample_rate)
        w.writeframes(data.tobytes())


def to_float_mono(samples: np.ndarray) -> np.ndarray:
    """float32 mono in [-1, 1], the format Whisper/WhisperX expect."""
    return (samples.astype(np.float32).mean(axis=1) / 32768.0).astype(np.float32)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_bytes_atomic(path: str | Path, data: bytes) -> None:
    """Write via a temp file + os.replace so a crash never leaves a partial file at `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def find_data_chunk(raw: bytes) -> tuple[int, int]:
    """Return (offset, length) of the PCM payload of a RIFF/WAVE file's `data` chunk."""
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise AudioFormatError("not a RIFF/WAVE file")
    pos = 12
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = int.from_bytes(raw[pos + 4:pos + 8], "little")
        if cid == b"data":
            return pos + 8, min(size, len(raw) - pos - 8)
        pos += 8 + size + (size & 1)  # chunks are word-aligned
    raise AudioFormatError("WAV has no data chunk")


def write_wav_like(path: str | Path, samples: np.ndarray, template: str | Path | bytes) -> None:
    """Write `samples` into a copy of `template`, replacing only the PCM payload.

    Every header/metadata byte of the template is preserved, so a file whose PCM
    is restored exactly is byte-identical (same sha256) to the original — even
    when the original carries extra chunks (LIST/INFO etc.) that `wave` drops.
    """
    raw = template if isinstance(template, bytes) else Path(template).read_bytes()
    off, length = find_data_chunk(raw)
    payload = np.ascontiguousarray(samples, dtype="<i2").tobytes()
    if len(payload) != length:
        raise AudioFormatError(f"payload size {len(payload)} != template data size {length}")
    write_bytes_atomic(path, raw[:off] + payload + raw[off + length:])
