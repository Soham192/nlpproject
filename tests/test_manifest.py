"""Test priority 3: each MANIFEST_SCHEMA.md validation rule has a failing case."""
import copy

import pytest

from src.manifest import (CipherInfo, Manifest, ManifestError, MaskedInfo, SourceInfo, Span, b64e,
                          text_preview, validate)


def _span(i, start, end, nonce=None):
    return Span(id=f"span_{i:03d}", entity_type="CREDIT_CARD", score=0.9, text_preview="x", char_start=0,
                char_end=1, word_start_idx=0, word_end_idx=1, start_sec=start / 16000, end_sec=end / 16000,
                sample_start=start, sample_end=end, padding_applied_ms=100,
                nonce=nonce or b64e(bytes([i]) * 12), ciphertext=b64e(b"\x01" * (end - start) * 2),
                tag=b64e(b"\x02" * 16))


def _manifest(spans):
    return Manifest(run_id="r", source=SourceInfo("a.wav", "h", 16000, 1, 2, 16000, 1.0),
                    masked=MaskedInfo("b.wav", "h2"), cipher=CipherInfo("AES-GCM", 32, "k"), config={},
                    spans=spans)


def test_valid_manifest_passes():
    validate(_manifest([_span(1, 0, 100), _span(2, 200, 300)]))


@pytest.mark.parametrize("field", ["ciphertext", "nonce", "tag"])
def test_rule1_empty_fields(field):
    s = _span(1, 0, 100)
    setattr(s, field, "")
    with pytest.raises(ManifestError, match=f"empty {field}"):
        validate(_manifest([s]))


def test_rule2_nonce_reuse():
    a, b = _span(1, 0, 100), _span(2, 200, 300)
    b.nonce = a.nonce
    with pytest.raises(ManifestError, match="nonce reused"):
        validate(_manifest([a, b]))


def test_rule3_unsorted():
    with pytest.raises(ManifestError, match="not sorted"):
        validate(_manifest([_span(1, 200, 300), _span(2, 0, 100)]))


def test_rule3_overlap():
    with pytest.raises(ManifestError, match="overlaps"):
        validate(_manifest([_span(1, 0, 150), _span(2, 100, 300)]))


def test_rule4_empty_range():
    s = _span(1, 100, 101)
    s.sample_end = 100
    with pytest.raises(ManifestError, match="sample_end"):
        validate(_manifest([s]))


def test_rule5_past_end():
    with pytest.raises(ManifestError, match="outside"):
        validate(_manifest([_span(1, 15900, 16100)]))


def test_rule6_ciphertext_length():
    s = _span(1, 0, 100)
    s.ciphertext = b64e(b"\x01" * 199)
    with pytest.raises(ManifestError, match="expected 200"):
        validate(_manifest([s]))


def test_rule7_key_field():
    m = _manifest([_span(1, 0, 100)])
    m.config["key"] = "abc"
    with pytest.raises(ManifestError, match="key material"):
        validate(m)


def test_rule7_key_bytes():
    key = bytes(range(32))
    m = _manifest([_span(1, 0, 100)])
    m.cipher.key_id = key.hex()
    with pytest.raises(ManifestError, match="key material"):
        validate(m, key)


def test_roundtrip_dict():
    m = _manifest([_span(1, 0, 100)])
    assert Manifest.from_dict(copy.deepcopy(m.to_dict())) == m
    assert list(m.to_dict())[0] == "version"


def test_text_preview():
    assert text_preview("4532-0151-1283-0366") == "4532-****-****-0366"
    assert text_preview("Bob") == "***"
