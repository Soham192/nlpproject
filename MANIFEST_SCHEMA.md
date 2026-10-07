# Manifest schema

The manifest is the sidecar file written next to every masked WAV. It is the
artifact that makes masking **reversible** — the masked audio alone cannot be
restored, and the manifest alone contains no playable audio.

**Access-control this file, not the WAV.** That split is the whole point of the
design: the WAV goes to QA, analytics, and vendors freely; the manifest goes
only to whoever is authorized to recover the sensitive spans.

**The key is never in this file.** Only ciphertext, nonces, and offsets.

---

## Schema

```json
{
  "version": "1.0",
  "run_id": "2026-10-06T10-51-03_a3f2c1",
  "source": {
    "filename": "call_0042.wav",
    "sha256": "<hash of the ORIGINAL audio, pre-masking>",
    "sample_rate": 16000,
    "channels": 1,
    "sample_width_bytes": 2,
    "total_samples": 960000,
    "duration_sec": 60.0
  },
  "masked": {
    "filename": "call_0042.masked.wav",
    "sha256": "<hash of the masked audio as written>"
  },
  "cipher": {
    "algorithm": "AES-GCM",
    "key_length_bytes": 32,
    "key_id": "local-key-01"
  },
  "config": {
    "padding_ms": 100,
    "replacement": "silence",
    "asr_model": "openai/whisper-small",
    "alignment_model": "WAV2VEC2_ASR_BASE_960H",
    "score_threshold": 0.5
  },
  "spans": [
    {
      "id": "span_001",
      "entity_type": "CREDIT_CARD",
      "score": 0.99,
      "text_preview": "4532-****-****-1567",
      "char_start": 112,
      "char_end": 131,
      "word_start_idx": 24,
      "word_end_idx": 28,
      "start_sec": 12.14,
      "end_sec": 16.02,
      "sample_start": 194240,
      "sample_end": 256320,
      "padding_applied_ms": 100,
      "nonce": "<base64, 12 bytes, unique per span>",
      "ciphertext": "<base64 of the original PCM bytes for this range>",
      "tag": "<base64 GCM auth tag>"
    }
  ],
  "detected_but_not_masked": [
    {
      "entity_type": "PERSON",
      "score": 0.42,
      "start_sec": 31.8,
      "end_sec": 32.4,
      "reason": "below score_threshold"
    }
  ],
  "stats": {
    "spans_masked": 2,
    "samples_masked": 94080,
    "duration_masked_sec": 5.88,
    "fraction_masked": 0.098
  },
  "timings_sec": {
    "asr": 8.2,
    "alignment": 3.1,
    "detection": 0.4,
    "masking": 0.2,
    "total": 11.9
  }
}
```

---

## Field notes

**`source.sha256`** — hash of the original, pre-masking audio. After an unmask,
recompute and compare. If it matches, reconstruction was exact. This is what
the round-trip test asserts against.

**`sample_start` / `sample_end`** — the authoritative masking boundaries, in PCM
sample indices, **padding already applied**. The `start_sec`/`end_sec` fields are
human-readable duplicates for the demo table; code should use the sample
indices so there is no float rounding at the boundary.

**`text_preview`** — partially redacted on purpose. The manifest should be
readable for debugging without itself leaking the full card number in plaintext.
Mask all but the first four and last four characters.

**`ciphertext`** — base64 of the raw PCM bytes that originally occupied
`[sample_start, sample_end)`. Per-span encryption (rather than one blob) means a
future version could grant access to one span without unlocking the others.

**`nonce`** — must be unique per span. Reusing a nonce with the same key breaks
AES-GCM's security entirely. Generate fresh per span, never derive it from the
span index.

**`detected_but_not_masked`** — entities found but left alone, with the reason.
Needed for the evaluation step (recall accounting) and useful in the defense:
it shows the system is selective by design, not just missing things.

**`stats.fraction_masked`** — the headline demo number. "We masked 9.8% of the
call and the other 90.2% is still fully usable."

**`timings_sec`** — per-stage, because "how long does this take per minute of
audio" is a predictable defense question.

---

## Validation rules

Enforce these in `manifest.py` — a manifest that violates any of them should
fail loudly rather than produce a silently-unrecoverable file:

1. Every span has non-empty `ciphertext`, `nonce`, and `tag`.
2. All nonces in a manifest are distinct.
3. Spans are sorted by `sample_start` and do not overlap. Merge overlapping
   spans at mask time (two adjacent entities with padding can collide).
4. `sample_end > sample_start` for every span.
5. `sample_end <= source.total_samples`.
6. `len(ciphertext_bytes) == (sample_end - sample_start) * sample_width_bytes * channels`.
7. No key material anywhere in the file.
