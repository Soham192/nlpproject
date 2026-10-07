"""AES-GCM encrypt/decrypt of PCM sample ranges.

The ciphertext goes into the manifest; the audio gets silence (or a tone) in
its place. Writing ciphertext into the waveform itself is deliberately avoided.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .manifest import b64d, b64e

NONCE_BYTES = 12
TAG_BYTES = 16


@dataclass(frozen=True)
class EncryptedRange:
    sample_start: int
    sample_end: int
    nonce: bytes
    ciphertext: bytes
    tag: bytes


def load_key(path: str | Path, key_length_bytes: int = 32, create: bool = False) -> bytes:
    """Read a raw key file. With create=True, generate one (0600) if absent."""
    path = Path(path)
    if not path.exists():
        if not create:
            raise FileNotFoundError(f"key file not found: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(AESGCM.generate_key(bit_length=8 * key_length_bytes))
    key = path.read_bytes()
    if len(key) != key_length_bytes:
        raise ValueError(f"{path}: key is {len(key)} bytes, expected {key_length_bytes}")
    return key


def _aad(sample_start: int, sample_end: int) -> bytes:
    # Bind ciphertext to its position so spans can't be swapped/moved undetected.
    return f"{sample_start}:{sample_end}".encode()


def encrypt_range(key: bytes, samples: np.ndarray, start: int, end: int) -> EncryptedRange:
    plaintext = np.ascontiguousarray(samples[start:end], dtype="<i2").tobytes()
    nonce = os.urandom(NONCE_BYTES)  # fresh per span, never derived from the index
    ct_and_tag = AESGCM(key).encrypt(nonce, plaintext, _aad(start, end))
    return EncryptedRange(start, end, nonce, ct_and_tag[:-TAG_BYTES], ct_and_tag[-TAG_BYTES:])


def decrypt_range(key: bytes, nonce: bytes, ciphertext: bytes, tag: bytes, start: int, end: int) -> bytes:
    return AESGCM(key).decrypt(nonce, ciphertext + tag, _aad(start, end))


def replacement_signal(n: int, channels: int, sample_rate: int, kind: str, tone_hz: int) -> np.ndarray:
    """What the listener hears in place of a masked span."""
    if kind == "silence":
        return np.zeros((n, channels), dtype=np.int16)
    t = np.arange(n) / sample_rate
    wave = np.sin(2 * np.pi * tone_hz * t)
    if kind == "tone":
        amp = 0.1
    elif kind == "beep":
        amp = 0.3
        wave = wave * (np.floor(t * 4) % 2 == 0)  # 125 ms on / off
    else:
        raise ValueError(f"unknown replacement {kind!r} (silence|tone|beep)")
    # Short fades so the edges don't click.
    fade = min(n // 2, int(0.005 * sample_rate))
    if fade:
        ramp = np.linspace(0, 1, fade)
        wave[:fade] *= ramp
        wave[-fade:] *= ramp[::-1]
    mono = (amp * 32767 * wave).astype(np.int16)
    return np.repeat(mono[:, None], channels, axis=1)


def mask_samples(
    samples: np.ndarray,
    ranges: list[tuple[int, int]],
    key: bytes,
    sample_rate: int,
    replacement: str = "silence",
    tone_hz: int = 1000,
) -> tuple[np.ndarray, list[EncryptedRange]]:
    """Encrypt each [start, end) range and overwrite it in a copy of `samples`.

    `ranges` must be sorted and non-overlapping (mapping.merge_ranges guarantees this).
    Samples outside the ranges are untouched.
    """
    out = samples.copy()
    enc: list[EncryptedRange] = []
    prev_end = 0
    for start, end in ranges:
        if not (0 <= start < end <= len(samples)) or start < prev_end:
            raise ValueError(f"invalid/overlapping range [{start},{end})")
        enc.append(encrypt_range(key, samples, start, end))
        out[start:end] = replacement_signal(end - start, samples.shape[1], sample_rate, replacement, tone_hz)
        prev_end = end
    return out, enc


def unmask_samples(masked: np.ndarray, spans: list, key: bytes) -> np.ndarray:
    """Restore original PCM from manifest spans (objects with sample_start/end, nonce, ciphertext, tag)."""
    out = masked.copy()
    channels = masked.shape[1]
    for s in spans:
        try:
            pt = decrypt_range(key, b64d(s.nonce), b64d(s.ciphertext), b64d(s.tag), s.sample_start, s.sample_end)
        except InvalidTag as e:
            raise ValueError(f"{s.id}: authentication failed — wrong key or tampered manifest") from e
        out[s.sample_start:s.sample_end] = np.frombuffer(pt, dtype="<i2").reshape(-1, channels)
    return out


def encode(e: EncryptedRange) -> dict[str, str]:
    return {"nonce": b64e(e.nonce), "ciphertext": b64e(e.ciphertext), "tag": b64e(e.tag)}
