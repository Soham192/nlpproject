"""Phase 0 gate: audio I/O is byte-exact."""
import numpy as np
import pytest

from src.audio import AudioFormatError, load_wav, sha256_file, write_wav, write_wav_like
from tests.conftest import make_wav


def test_load_write_byte_identical(wav, tmp_path):
    samples, params = load_wav(wav)
    out = tmp_path / "copy.wav"
    write_wav(out, samples, params)
    assert sha256_file(out) == sha256_file(wav)


def test_write_like_preserves_extra_chunks(tmp_path):
    src = make_wav(tmp_path / "lavf.wav", extra_chunk=True)
    samples, _ = load_wav(src)
    out = tmp_path / "copy.wav"
    write_wav_like(out, samples, src)
    assert sha256_file(out) == sha256_file(src)


def test_rejects_non_wav(tmp_path):
    p = tmp_path / "x.mp3"
    p.write_bytes(b"ID3\x03\x00" + b"\x00" * 100)
    with pytest.raises(AudioFormatError, match="not a PCM WAV"):
        load_wav(p)


def test_rejects_wrong_format(tmp_path, cfg):
    from src.audio import AudioParams
    p = tmp_path / "44k.wav"
    write_wav(p, np.zeros((441, 1), np.int16), AudioParams(44100, 1, 2, 441))
    with pytest.raises(AudioFormatError, match="Refusing to transcode"):
        load_wav(p, cfg.audio)


def test_crash_during_write_leaves_no_partial_file(wav, tmp_path, monkeypatch):
    """A process killed mid-write must never leave a truncated file at the destination."""
    import os

    from src import audio

    samples, _ = load_wav(wav)
    dest = tmp_path / "masked.wav"
    write_wav_like(dest, samples, wav)
    before = dest.read_bytes()

    def boom(*a, **k):
        raise KeyboardInterrupt("killed before rename")

    monkeypatch.setattr(audio.os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        write_wav_like(dest, np.zeros_like(samples), wav)
    assert dest.read_bytes() == before            # old file intact, never half-overwritten
    monkeypatch.setattr(audio.os, "replace", os.replace)
    new = tmp_path / "fresh.wav"
    monkeypatch.setattr(audio.os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        write_wav_like(new, samples, wav)
    assert not new.exists()                       # nothing at the destination, only *.tmp
