# Selective Audio Redaction

Find the sensitive spans in a voice recording (card numbers, SSNs, names, account numbers), **encrypt only those spans**, and **prove with metrics** that the rest of the recording is untouched and still intelligible.

Built for call-center, telehealth, and financial-support recordings, where GDPR data minimization and HIPAA minimum-necessary require limiting *access* to sensitive data, not just encrypting files at rest.

## Why

| Existing option | Problem |
|---|---|
| Encrypt the whole file | All-or-nothing: anyone who needs any part of the call must decrypt all of it |
| Bleep / delete the span | Irreversible: useless when a fraud investigation or subpoena needs the original |
| Commercial PII redaction (Azure, AssemblyAI, Presidio pipelines) | Redacts the **transcript**; the **audio file**, the artifact that actually gets stored and shared, is left untouched |

This project contributes:

1. **Reversible, key-based masking.** Sensitive spans are AES-256-GCM encrypted into a sidecar manifest and silenced in the audio. With the key, the original is restored **byte-identical**.
2. **Content-preservation verification.** The masked audio is re-transcribed, and WER + BERTScore are reported over the *non-redacted* spans, along with a check that every sample outside the masked ranges is unchanged.

## How it works

```
input.wav
  [1] ASR          Whisper -> transcript + segment timestamps
  [2] Alignment    WhisperX (wav2vec2 CTC) -> word-level timestamps
  [3] Detection    Presidio + spaCy NER on the full transcript (spoken numbers normalized to digits first)
  [4] Mapping      char offsets -> words -> sample ranges (+100 ms padding, overlaps merged)
  [5] Masking      AES-GCM encrypt each range's PCM bytes -> manifest; write silence in its place
output.wav              playable by anyone, sensitive spans silenced
output.manifest.json    ciphertext + nonce + offsets + entity type (never the key)
  [6] Evaluation   re-run ASR on output.wav -> WER / BERTScore on non-redacted text, entity P/R
```

Unmasking is the reverse: `output.wav + manifest + key -> original.wav`. The manifest is what you access-control; the audio is safe to share.

The ciphertext lives in the manifest and is never written into the waveform. Writing ciphertext into the waveform would produce loud noise and wouldn't survive a transcode. For the same reason the pipeline accepts **16 kHz mono 16-bit PCM WAV only**. Other inputs are rejected with an `ffmpeg` command to convert them, never silently transcoded. See [`MANIFEST_SCHEMA.md`](MANIFEST_SCHEMA.md) for the manifest format.

## Results (sample call)

`data/samples/meridian_call.wav`: a 2:10 synthetic bank-support call with planted fake PII, run on CPU.

| Metric | Value |
|---|---|
| Round trip (mask -> unmask) | byte-identical (sha256 match) |
| Samples outside masked spans | identical |
| WER on non-redacted text | 3.65% (219 ref words, 0 substitutions) |
| BERTScore F1 (roberta-large) | 0.987 |
| Entity detection precision / recall | 0.75 / 0.80 |
| Audio masked | 16 spans, 28.7 s (22%) |

Detection is the weakest stage. Some spoken phone numbers, email addresses, and short policy numbers are still missed, so recall is the number to improve.

## Setup

Requires Python 3.10+. A CUDA GPU is optional; CPU works and is roughly 5-10x slower.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api,dev]"
python -m spacy download en_core_web_lg
```

Models download on first run: Whisper-small, wav2vec2, and roberta-large for BERTScore, about 2 GB in total.

## CLI

```bash
# Dry run: print the detection table, change nothing
redact detect data/samples/meridian_call.wav

# Mask (generates key.bin with 0600 permissions if it doesn't exist)
redact mask data/samples/meridian_call.wav -o out/masked.wav --key-file key.bin

# Restore the original and verify its sha256
redact unmask out/masked.wav --manifest out/masked.manifest.json --key-file key.bin -o out/restored.wav

# Evaluate content preservation (+ entity P/R if ground truth is given)
redact eval --original data/samples/meridian_call.wav --masked out/masked.wav \
            --manifest out/masked.manifest.json \
            --ground-truth data/samples/meridian_call.ground_truth.json

# Demo assets: waveform figure, before/after clips, slide tables
redact demo --original data/samples/meridian_call.wav --masked out/masked.wav \
            --manifest out/masked.manifest.json
```

Each stage writes its intermediate artifact (transcript, words, detections, spans, manifest, metrics) to `data/outputs/<run_id>/`, and per-stage timings are logged.

> **Keep `key.bin` out of version control.** It is gitignored. Without it, masked spans cannot be recovered.

## Web UI

A React + Vite interface for showing the stages visually: detection table, waveform with masked regions, original/masked playback, the round-trip check, and metrics.

```bash
uvicorn api.main:app --port 8000      # FastAPI backend (also serves web/dist if built)
cd web && npm install && npm run dev  # UI on http://localhost:5173, proxies /api to :8000
```

The backend wraps the same `src/` functions the CLI uses. The key is read and used server-side only. `api.allow_original_playback` in `config.yaml` lets the UI play the unredacted original. That setting is for demos only; turn it off (or set `ALLOW_ORIGINAL_PLAYBACK=0`) anywhere real data is involved.

## Configuration

Everything lives in [`config.yaml`](config.yaml):

- **`detection.entities`**: the active entity set. Toggle per compliance regime. The default set:
  - pattern-based: `CREDIT_CARD`, `US_SSN`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `IBAN_CODE`, `DATE_TIME`
  - NER-based: `PERSON`, `LOCATION`
  - custom: `ACCOUNT_NUMBER`, `MEDICAL_RECORD_NUMBER`, `POLICY_NUMBER`
- **`masking.padding_ms`**: margin around each span (default 100 ms), so that alignment error doesn't leak the first phoneme.
- **`masking.replacement`**: `silence`, `tone`, or `beep`.
- **`asr.model`**, **`evaluation.bertscore_model`**: pinned so runs stay comparable.

## Tests

```bash
pytest
```

The most important test is `tests/test_roundtrip.py`: mask -> unmask -> byte-identical to the original, with every sample outside the masked spans unchanged.

To generate new synthetic calls with exact ground truth (requires internet for gTTS, plus ffmpeg):

```bash
python scripts/make_sample.py --call meridian_call
```

## Optional: autoencoder study

`src/autoencoders/` trains four autoencoders (dense, sparse, denoising, VAE) on Whisper log-mel windows, and scores each one at three pipeline layers: an ingest quality gate, an ASR front-end that cleans a *copy* of Whisper's mel, and a leak check on masked spans. Each layer is compared against a non-learned baseline. Training and GPU evaluation run on Colab only, through [`notebooks/train_autoencoders_colab.ipynb`](notebooks/train_autoencoders_colab.ipynb). Results are collected by `python -m src.autoencoders.compare` into `results/autoencoder_comparison.md`.

All of it is off by default (`autoencoder.integrations.*.enabled: false`). With the flags off, the pipeline never imports it. Detection stays Presidio + spaCy, and the output audio is never touched.

## Repo layout

```
src/        pipeline stages: asr, alignment, detect, mapping, mask, manifest, evaluate, demo, cli
src/autoencoders/   optional AE study: models, data, train, eval_{ingest,asr,leak}, bench_cpu, compare
notebooks/  thin Colab runner for the AE study
api/        FastAPI backend for the web UI (in-process job queue)
web/        React + Vite frontend
tests/      round-trip, mapping, detection, manifest, evaluation, API tests
scripts/    synthetic sample generation
data/samples/   sample calls + ground truth
```

## Out of scope / future work

- Speaker anonymization and voice conversion (the VoicePrivacy problem; a different goal from access control)
- Real-time or streaming redaction
- KMS or multi-tenant key management (keys are a local file)
- Hosted deployment: the backend needs torch, WhisperX, and spaCy (several GB), so it runs locally
