"""Manifest schema, read/write and validation (see MANIFEST_SCHEMA.md).

The manifest is the access-controlled artifact. It holds ciphertext, nonces,
offsets and metadata — never key material.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "1.0"


class ManifestError(ValueError):
    pass


@dataclass
class SourceInfo:
    filename: str
    sha256: str
    sample_rate: int
    channels: int
    sample_width_bytes: int
    total_samples: int
    duration_sec: float


@dataclass
class MaskedInfo:
    filename: str
    sha256: str


@dataclass
class CipherInfo:
    algorithm: str
    key_length_bytes: int
    key_id: str


@dataclass
class Span:
    id: str
    entity_type: str
    score: float
    text_preview: str
    char_start: int
    char_end: int
    word_start_idx: int
    word_end_idx: int
    start_sec: float
    end_sec: float
    sample_start: int
    sample_end: int
    padding_applied_ms: int
    nonce: str
    ciphertext: str
    tag: str


@dataclass
class Manifest:
    run_id: str
    source: SourceInfo
    masked: MaskedInfo
    cipher: CipherInfo
    config: dict[str, Any]
    spans: list[Span]
    detected_but_not_masked: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    timings_sec: dict[str, float] = field(default_factory=dict)
    version: str = MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Keep the documented key order: version first.
        return {"version": d.pop("version"), **d}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Manifest":
        try:
            return cls(
                version=d["version"],
                run_id=d["run_id"],
                source=SourceInfo(**d["source"]),
                masked=MaskedInfo(**d["masked"]),
                cipher=CipherInfo(**d["cipher"]),
                config=d["config"],
                spans=[Span(**s) for s in d["spans"]],
                detected_but_not_masked=d.get("detected_but_not_masked", []),
                stats=d.get("stats", {}),
                timings_sec=d.get("timings_sec", {}),
            )
        except (KeyError, TypeError) as e:
            raise ManifestError(f"malformed manifest: {e}") from e


def text_preview(text: str) -> str:
    """Keep first 4 and last 4 characters, star out the middle (separators kept)."""
    if len(text) <= 8:
        return "*" * len(text)
    mid = re.sub(r"[^\s\-]", "*", text[4:-4])
    return text[:4] + mid + text[-4:]


def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"), validate=True)


# Field names that would indicate key material leaked into the manifest.
_KEY_FIELD = re.compile(r"^(key|secret|password|passphrase|private_key|key_bytes|key_material|raw_key)$", re.I)


def _find_key_fields(obj: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if _KEY_FIELD.match(str(k)):
                hits.append(p)
            hits.extend(_find_key_fields(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(_find_key_fields(v, f"{path}[{i}]"))
    return hits


def validate(m: Manifest, key: bytes | None = None) -> None:
    """Enforce the 7 rules in MANIFEST_SCHEMA.md. Raises ManifestError on the first violation.

    If `key` is given, also checks that its bytes (raw or base64/hex) appear nowhere in the file.
    """
    src = m.source
    frame_bytes = src.sample_width_bytes * src.channels
    nonces: set[str] = set()
    prev_end = -1
    prev_start = -1
    for s in m.spans:
        # Rule 1
        for name in ("ciphertext", "nonce", "tag"):
            if not getattr(s, name):
                raise ManifestError(f"{s.id}: empty {name}")
        # Rule 2
        if s.nonce in nonces:
            raise ManifestError(f"{s.id}: nonce reused — breaks AES-GCM security")
        nonces.add(s.nonce)
        # Rule 4
        if s.sample_end <= s.sample_start:
            raise ManifestError(f"{s.id}: sample_end ({s.sample_end}) <= sample_start ({s.sample_start})")
        # Rule 3
        if s.sample_start < prev_start:
            raise ManifestError(f"{s.id}: spans not sorted by sample_start")
        if s.sample_start < prev_end:
            raise ManifestError(f"{s.id}: overlaps previous span (starts {s.sample_start} < {prev_end})")
        prev_start, prev_end = s.sample_start, s.sample_end
        # Rule 5
        if s.sample_start < 0 or s.sample_end > src.total_samples:
            raise ManifestError(f"{s.id}: range [{s.sample_start},{s.sample_end}) outside [0,{src.total_samples}]")
        # Rule 6
        try:
            ct = b64d(s.ciphertext)
            nonce = b64d(s.nonce)
            b64d(s.tag)
        except Exception as e:  # binascii.Error
            raise ManifestError(f"{s.id}: invalid base64 ({e})") from e
        expected = (s.sample_end - s.sample_start) * frame_bytes
        if len(ct) != expected:
            raise ManifestError(f"{s.id}: ciphertext is {len(ct)} bytes, expected {expected}")
        if len(nonce) != 12:
            raise ManifestError(f"{s.id}: nonce must be 12 bytes, got {len(nonce)}")
    # Rule 7
    d = m.to_dict()
    hits = _find_key_fields(d)
    if hits:
        raise ManifestError(f"key material field(s) present: {hits}")
    if key is not None:
        blob = json.dumps(d)
        for enc in (b64e(key), key.hex()):
            if enc in blob:
                raise ManifestError("key material found in manifest content")


def write_manifest(m: Manifest, path: str | Path, key: bytes | None = None) -> None:
    from .audio import write_bytes_atomic

    validate(m, key)
    write_bytes_atomic(path, json.dumps(m.to_dict(), indent=2).encode("utf-8"))


def read_manifest(path: str | Path) -> Manifest:
    m = Manifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    validate(m)
    return m


SECRET_SPAN_FIELDS = ("ciphertext", "nonce", "tag")


def public_dict(m: Manifest) -> dict[str, Any]:
    """Manifest without per-span ciphertext/nonce/tag — what a viewer (e.g. the web UI) needs."""
    d = m.to_dict()
    d["spans"] = [{k: v for k, v in s.items() if k not in SECRET_SPAN_FIELDS} for s in d["spans"]]
    return d
