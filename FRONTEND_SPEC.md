# FRONTEND_SPEC.md

Brief for adding a web interface to the audio redaction pipeline.

Read `CLAUDE.md` first — the invariants there still hold, and two of them
constrain this work directly.

---

## What this is for

The pipeline currently runs as a CLI. The interface exists so the three stages
can be *seen*: which spans were detected, where they sit in the audio, what the
masked file sounds like, and that the original comes back byte-identical.

It is a demonstration and operator surface, not a product. Build the smallest
thing that makes the mechanics visible.

---

## Architecture

```
React + Vite  ──HTTP──>  FastAPI  ──imports──>  existing src/ modules
  (browser)                (local)               asr, detect, mask, …
```

**The backend wraps the existing modules. It does not reimplement any of them.**
`api/` imports from `src/` and calls the same functions the CLI calls. If a
pipeline change is needed to support the UI, change `src/` and let both callers
benefit — never fork the logic into the API layer.

Suggested layout:

```
api/
  main.py          FastAPI app, CORS for the Vite dev server
  jobs.py          in-process job store (dict is fine; no database)
  routes.py        the endpoints below
web/
  src/
    App.jsx
    components/
      Dropzone.jsx
      DetectionTable.jsx
      WaveformView.jsx
      MetricsPanel.jsx
      RoundTripBadge.jsx
      EntityConfig.jsx
    api.js         fetch wrappers
```

---

## Hard constraints

These are not style preferences. Breaking any of them breaks a claim the
project rests on.

**The key never reaches the browser.** `key.bin` is read server-side and used
server-side. It is never returned by an endpoint, never embedded in a response,
never put in localStorage. There is no "paste your key" field. Unmasking is
triggered by the browser but performed entirely on the server.

**Audio is served as raw WAV bytes, unmodified.** Do not transcode, normalise,
resample or re-encode anything on the way to the browser, and do not let the
frontend process audio and send it back. Sample-exact preservation is the whole
basis of reversibility. `Content-Type: audio/wav`, streamed from disk, nothing
in between.

**The original recording is gated.** The unredacted audio contains the PII the
system exists to protect. Serve it only from an endpoint that is explicitly
marked demo-only and disabled by a config flag (`ALLOW_ORIGINAL_PLAYBACK`,
default on for the demo, documented as something a real deployment turns off).
Say so in the UI next to the player.

**Uploads are bounded.** Reject files over a configured size, reject anything
that is not RIFF/WAVE PCM, and reject it with the same clear error the CLI
gives rather than silently transcoding.

---

## Endpoints

The pipeline takes roughly 26 seconds per minute of audio on CPU, so masking
cannot be a blocking request. Use a job + poll pattern; server-sent events are
fine if they're not more trouble than they're worth.

```
POST   /api/upload              multipart WAV  →  { file_id, duration, sample_rate }
POST   /api/detect              { file_id, entity_types[] }  →  { job_id }
POST   /api/mask                { file_id, entity_types[] }  →  { job_id }
POST   /api/unmask              { file_id }  →  { job_id }
POST   /api/evaluate            { file_id }  →  { job_id }
GET    /api/jobs/{job_id}       →  { status, stage, progress, result?, error? }
GET    /api/files/{file_id}/masked.wav      streamed WAV
GET    /api/files/{file_id}/original.wav    streamed WAV, demo-gated
GET    /api/files/{file_id}/manifest        manifest JSON, ciphertext fields stripped
GET    /api/config/entities     →  available entity types + defaults from config.yaml
```

`stage` in the job status should name the real pipeline stage — `asr`,
`alignment`, `detection`, `mapping`, `masking` — so the UI can show what is
happening rather than a meaningless spinner. The per-stage timings already
logged by the pipeline feed this directly.

The manifest endpoint strips `ciphertext`, `nonce` and `tag` from each span.
The UI needs offsets, entity types, scores and timestamps; it has no use for
the ciphertext, and sending it to a browser serves no purpose.

---

## Screens

One page, four sections revealed in sequence. No routing, no sidebar, no login.

**1 · Upload**
Dropzone for a WAV file. On success show duration, sample rate, channels. If
the file is rejected, show the reason plainly — "needs uncompressed WAV; MP3
cannot be restored sample-for-sample" is more useful than "invalid file".

**2 · Detect (dry run)**
A button that runs detection without modifying anything — this mirrors
`redact detect` and is the most important screen for a demo, because it shows
the system's reasoning before any audio changes.

Render a table: entity type, redacted text preview, confidence, start and end
time, and whether it will be masked. Rows below the confidence threshold appear
greyed with the reason, matching the manifest's `detected_but_not_masked`.

Beside it, an entity-type config panel with checkboxes, defaulting to
`config.yaml`. Being able to toggle `DATE_TIME` off live and watch the two
false positives disappear is worth building — it demonstrates the
precision/recall trade-off interactively instead of describing it.

**3 · Mask**
Runs the masking job with a stage-by-stage progress indicator.

Then the core visual: a waveform with the masked spans marked. Use
**wavesurfer.js** with the regions plugin — it handles the waveform rendering
and region overlays, and `peaks` can be computed server-side if client-side
decoding of a long file is slow. Hovering a region shows its entity type and
time range. Clicking a table row scrolls to and highlights its region.

Below the waveform, two audio players side by side, clearly labelled:
*Original (contains PII — demo only)* and *Redacted (safe to share)*. Playing
one pauses the other.

**4 · Verify**
Three things, in this order of prominence:

- **Round-trip badge.** Run unmask server-side, compare SHA-256 against the
  digest in the manifest, and show pass/fail large. This is the project's
  strongest single result and should be the most visually prominent element on
  the page. Show both digests, truncated, so it is evidently a comparison and
  not a hardcoded tick.
- **Preservation metrics.** WER and BERTScore over non-masked spans, each with
  a one-line explanation of what it measures. Label clearly that they are
  computed on non-masked spans only.
- **Detection metrics.** Precision, recall, F1, and the masked-fraction figure
  ("9.8% of this call was masked; 90.2% remains usable").

---

## Visual design

Match the project deck so screenshots sit consistently alongside the slides:

```
ink / headings   #2D1B4E
primary          #6B2FA0
accent           #C2185B
surface          #EFEAF6
muted text       #5F5873
background       #FFFFFF
```

Purple for pipeline and safe states, crimson for masked spans and sensitive
content, green only for the round-trip pass badge. Headings and body in a
system sans; use a monospace face for file names, digests and sample offsets.

Keep it plain. A clean table and a legible waveform communicate more here than
any amount of ornament.

---

## Out of scope

Authentication, multi-user support, persistence beyond the current session, key
management UI, batch upload, streaming/live redaction, mobile layout. None of
these serve the demonstration, and each is a place to lose time.

---

## Done when

- A WAV can be dropped in, detected, masked, played back redacted, and verified
  as exactly restorable, without touching a terminal.
- Toggling an entity type off changes the detection table and the resulting
  precision.
- The key appears nowhere in any network response — check this in the browser's
  network tab before calling it finished.
- Stopping the API mid-job leaves no corrupted output file.
- The CLI still works unchanged, and `pytest` still passes.
