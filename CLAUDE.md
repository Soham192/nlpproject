# CLAUDE.md

Project context for Claude Code. Read this before making changes.

---

## What this project is

A pipeline that takes a voice recording, finds the spans where sensitive information is spoken (card numbers, SSNs, names, medical terms), **encrypts only those spans** so the audio is safe to share, and then **proves with metrics** that the rest of the recording is still intelligible and usable.

The target user is an enterprise compliance/engineering team at a call center, telehealth platform, or financial support line. They are legally required (GDPR Art. 5(1)(c) data minimization, HIPAA minimum-necessary) to limit *access* to sensitive data, not merely encrypt files at rest.

**The problem with existing options:**
- Encrypting the whole file is all-or-nothing — anyone who needs any part of the call must decrypt all of it.
- Deleting/bleeping the sensitive span is irreversible — useless when a fraud investigation or subpoena later needs the original.
- Commercial tools (Azure, AssemblyAI, Presidio pipelines) redact the **transcript** and leave the **audio file** untouched — but the audio file is the artifact that actually gets stored, backed up, and sent to vendors.

**Our two contributions (these are the novelty — do not compromise them):**
1. **Reversible, key-based masking.** Sensitive spans are AES-encrypted and recoverable with a key, not destroyed.
2. **Formal content-preservation verification.** We re-transcribe the masked audio and publish WER + BERTScore over the *non-redacted* spans, proving masking did not damage the rest. No surveyed production system publishes this.

---

## Pipeline

```
input.wav
   |
[1] ASR            Whisper -> transcript + segment timestamps
   |
[2] ALIGNMENT      WhisperX (wav2vec2 CTC) -> word-level timestamps
   |
[3] DETECTION      Presidio + spaCy NER -> sensitive char spans in transcript
   |
[4] MAPPING        char offsets -> word indices -> audio sample ranges
   |
[5] MASKING        AES-GCM encrypt those sample ranges;
                   write silence into output.wav;
                   store ciphertext in output.manifest.json
   |
output.wav  (safe to share, no key needed to play)
output.manifest.json  (encrypted spans + metadata)
   |
[6] EVALUATION     re-run Whisper on output.wav
                   -> WER + BERTScore over non-redacted spans only
                   -> entity detection precision/recall
```

Unmasking is the reverse: `output.wav + manifest + key -> original.wav`.

---

## Critical design decision: how masking actually works

Do **not** try to write AES ciphertext bytes directly into the waveform. It produces loud noise, breaks any player, and is destroyed by the first transcode.

Instead:
1. Extract the raw PCM samples for the sensitive span.
2. AES-GCM encrypt those bytes with a per-span nonce.
3. Write the ciphertext + nonce + sample offsets into a **sidecar manifest** (`*.manifest.json`).
4. Replace the span in the output WAV with **silence** (or a soft tone, configurable).

This gives both properties at once: the shipped audio is clean and playable by anyone, and the original is exactly recoverable by whoever holds the key. The manifest is the thing you access-control, not the audio.

**Consequence:** the pipeline operates on **WAV/PCM only**. Lossy codecs (mp3, opus) destroy sample-exact reconstruction. Convert to 16 kHz mono PCM WAV on ingest and keep it that way. Reject non-WAV input with a clear error rather than silently transcoding.

---

## Invariants (violating these breaks the project's claims)

- **Masking must be reversible.** Every masked span has recoverable ciphertext in the manifest. If a change makes a span unrecoverable, it is a bug, not a tradeoff.
- **Only flagged spans are touched.** Never apply a global filter, resample, renormalize, or re-encode the output. Sample-for-sample identical outside the masked ranges. The evaluation step depends on this.
- **Span boundaries get a padding margin** (default 100 ms each side, configurable). Alignment is imperfect; clipping the first phoneme of a card number leaks information.
- **Evaluation is computed on non-redacted spans only.** Including the masked regions in the WER reference would make the metric meaningless (of course silence doesn't transcribe).
- **The key never goes in the manifest.** Key handling stays outside the artifact; manifest holds ciphertext, nonce, offsets, entity type, and timestamps only.

---

## Repo layout

```
src/
  pipeline.py      orchestrator - wires stages together, the one entry point
  asr.py           Whisper transcription
  alignment.py     WhisperX word-level timestamps
  detect.py        Presidio analyzer + custom recognizers
  mapping.py       char offset -> word index -> sample range (incl. padding)
  mask.py          AES-GCM encrypt/decrypt of sample ranges
  manifest.py      manifest schema, read/write, validation
  evaluate.py      WER, BERTScore, entity precision/recall
  cli.py           CLI entry points
tests/
data/
  samples/         input audio
  outputs/         generated wav + manifest + metrics
config.yaml
```

Keep each stage independently callable with plain data in/out (paths, dicts, dataclasses). The defense demo needs to show each stage's output separately, so don't fuse stages into one opaque function.

---

## CLI surface

```
redact mask    input.wav  -o output.wav  --key-file key.bin
redact unmask  output.wav --manifest output.manifest.json --key-file key.bin -o restored.wav
redact eval    --original input.wav --masked output.wav --manifest output.manifest.json
redact detect  input.wav   # dry run: print detected spans, change nothing
```

`detect` as a dry-run mode matters — it's the demo slide where you show the detection table before anything is modified.

---

## Entity types to support

Prioritize compliance-driven categories over open-ended NER:

- Pattern-based (Presidio regex recognizers, high precision): `CREDIT_CARD`, `US_SSN`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `IBAN_CODE`, `DATE_TIME`
- NER-based (spaCy, context-dependent, lower precision): `PERSON`, `LOCATION`, `NRP`
- Add custom recognizers for anything domain-specific (account numbers, policy numbers, MRN).

Make the active entity set **configurable in config.yaml**, not hardcoded. Different compliance regimes need different sets, and the demo is clearer when you can show toggling one on/off.

---

## Known gotchas

- **Spoken numbers transcribe as words.** Whisper often emits "four five three two" or "forty-five thirty-two" rather than "4532". Presidio's credit-card regex will miss these. Normalize number words to digits before detection, or add a spoken-number recognizer. **This is the single most likely cause of a failed demo** — test it early with real spoken digits.
- **Chunk boundaries split entities.** An SSN spoken across two Whisper segments won't match a regex applied per-segment. Run detection on the full joined transcript, keeping a char-offset -> word-index map.
- **Whisper's native timestamps are coarse.** That's why the WhisperX alignment stage exists. Don't skip it and use Whisper's segment times for masking.
- **WER needs matched text normalization.** Lowercase, strip punctuation, expand contractions consistently on both reference and hypothesis, or the metric is noise.
- **BERTScore downloads a model on first run.** Pin the model name in config; don't let it silently change between runs you're comparing.

---

## Conventions

- Python 3.10+, type hints on public functions.
- Config in `config.yaml`, loaded once; no magic numbers scattered through modules.
- Each stage writes an intermediate artifact to `data/outputs/<run_id>/` so the demo can show them.
- Log timings per stage — "how long does this take per minute of audio" is a likely defense question.
- Tests: a round-trip test (`mask -> unmask -> byte-identical to original`) is the single most important one. Write it first.

---

## Scope

**In scope:** the three layers above (detect -> mask -> verify), batch processing of pre-recorded WAV files, CLI.

**Explicitly out of scope** (mention as future work, don't build):
- Speaker anonymization / voice conversion — different problem (data *sharing*, not access control), different literature (VoicePrivacy).
- Real-time / streaming redaction — Whisper isn't natively streaming.
- A personal-privacy consumer variant — would need on-device inference and a different architecture.
- Multi-tenant key management / KMS integration — keep key handling to a local key file.
