"""Test priorities 1, 2, 6: round-trip exactness, outside-span immutability, boundary cases."""
import numpy as np
import pytest

from src.audio import load_wav, sha256_file
from src.manifest import read_manifest
from src.mask import mask_samples, unmask_samples
from src.alignment import Word
from src.pipeline import Pipeline, unmask
from tests.conftest import SR, WORDS, det, make_wav, stub_pipeline


def _run(monkeypatch, cfg, key, src, tmp_path, words, dets_fn):
    stub_pipeline(monkeypatch, words, dets_fn)
    out = tmp_path / "out.wav"
    m = Pipeline(cfg, "test").mask(src, out, key)
    return out, m


def test_mask_unmask_sha256_identical(monkeypatch, cfg, key, wav, tmp_path):
    out, m = _run(monkeypatch, cfg, key, wav, tmp_path, WORDS,
                  lambda t: [det(t, "CREDIT_CARD", "45320151")])
    assert m.spans, "nothing was masked"
    assert sha256_file(out) != sha256_file(wav)
    restored = tmp_path / "restored.wav"
    m2 = read_manifest(tmp_path / "out.manifest.json")
    assert unmask(out, m2, key, restored, cfg)
    assert sha256_file(restored) == m.source.sha256 == sha256_file(wav)


def test_roundtrip_with_extra_wav_chunks(monkeypatch, cfg, key, tmp_path):
    src = make_wav(tmp_path / "lavf.wav", extra_chunk=True)
    out, m = _run(monkeypatch, cfg, key, src, tmp_path, WORDS, lambda t: [det(t, "CREDIT_CARD", "45320151")])
    restored = tmp_path / "r.wav"
    assert unmask(out, m, key, restored, cfg)
    assert sha256_file(restored) == sha256_file(src)


def test_outside_span_identical_and_inside_silent(monkeypatch, cfg, key, wav, tmp_path):
    out, m = _run(monkeypatch, cfg, key, wav, tmp_path, WORDS, lambda t: [det(t, "CREDIT_CARD", "45320151")])
    orig, _ = load_wav(wav)
    masked, _ = load_wav(out)
    keep = np.ones(len(orig), bool)
    for s in m.spans:
        keep[s.sample_start:s.sample_end] = False
        assert not masked[s.sample_start:s.sample_end].any()
    assert np.array_equal(orig[keep], masked[keep])
    # padding: 100 ms before "four" (0.80 s) and after "one" (2.50 s)
    s = m.spans[0]
    assert s.sample_start == int(0.70 * SR) and s.sample_end == int(2.60 * SR)


def test_wrong_key_fails_loudly(monkeypatch, cfg, key, wav, tmp_path):
    out, m = _run(monkeypatch, cfg, key, wav, tmp_path, WORDS, lambda t: [det(t, "CREDIT_CARD", "45320151")])
    with pytest.raises(ValueError, match="wrong key"):
        unmask(out, m, bytes(32), tmp_path / "r.wav", cfg)


@pytest.mark.parametrize("words,needle", [
    ([Word("4532015112830366", 0.0, 1.0), Word("ok", 1.5, 2.0)], "4532015112830366"),        # at sample 0
    ([Word("ok", 0.5, 1.0), Word("4532015112830366", 3.0, 4.0)], "4532015112830366"),        # at file end
])
def test_boundaries(monkeypatch, cfg, key, wav, tmp_path, words, needle):
    out, m = _run(monkeypatch, cfg, key, wav, tmp_path, words, lambda t: [det(t, "CREDIT_CARD", needle)])
    s = m.spans[0]
    assert s.sample_start >= 0 and s.sample_end <= m.source.total_samples
    assert s.sample_start == 0 or s.sample_end == m.source.total_samples
    restored = tmp_path / "r.wav"
    assert unmask(out, m, key, restored, cfg)


@pytest.mark.parametrize("replacement", ["silence", "tone", "beep"])
def test_replacements_roundtrip(replacement, key):
    rng = np.random.default_rng(1)
    x = rng.integers(-3000, 3000, size=(SR, 1), dtype=np.int16)
    masked, enc = mask_samples(x, [(100, 4000), (8000, 9000)], key, SR, replacement)
    from src.mask import encode
    from types import SimpleNamespace
    spans = [SimpleNamespace(sample_start=e.sample_start, sample_end=e.sample_end, **encode(e)) for e in enc]
    assert np.array_equal(unmask_samples(masked, spans, key), x)
    assert len({e.nonce for e in enc}) == 2
