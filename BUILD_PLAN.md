# Build plan

Ordered so that something demonstrable exists early and the riskiest unknown is
hit first, not last.

---

## Phase 0 — Skeleton

- `config.yaml` loader, `data/` dirs, logging with per-stage timings.
- Audio I/O: load WAV to numpy PCM, write WAV back. Assert round-trip is
  byte-identical before building anything on top of it.

**Done when:** `load(path) -> write(path2)` produces a byte-identical file.

---

## Phase 1 — Detection, dry run (riskiest part, do it first)

- `asr.py`: Whisper transcription with segment timestamps.
- `detect.py`: Presidio analyzer on the **full joined transcript**, not
  per-segment.
- `redact detect input.wav` prints a table: entity type, text preview, score,
  char offsets.

**Hit the spoken-number problem immediately.** Record yourself saying a card
number out loud. Whisper will likely emit "four five three two" and Presidio's
regex will find nothing. Build `normalize_spoken_numbers` now, while it's the
only thing that can be wrong, rather than discovering it during a demo.

**Done when:** a WAV of someone reading a card number and an SSN aloud produces
correct detections with correct char offsets.

---

## Phase 2 — Alignment and mapping

- `alignment.py`: WhisperX word-level timestamps.
- `mapping.py`: char offset -> word index -> sample range, with padding, and
  merging of overlapping spans.

**Done when:** for each detected entity, you can slice exactly those samples out
of the original audio and hear only that entity spoken, with a little margin.
Listen to the slices — this is the fastest way to catch an off-by-one in the
offset chain.

---

## Phase 3 — Mask and unmask (the core claim)

- `mask.py`: AES-GCM encrypt the sample ranges, write silence in their place.
- `manifest.py`: schema, write, and the validation rules in MANIFEST_SCHEMA.md.
- `redact mask` and `redact unmask`.

**Write the round-trip test before the code:**
`mask -> unmask -> sha256 == source.sha256`. That single assertion is the whole
reversibility claim. If it ever fails, stop and fix it before anything else.

Also assert: the masked WAV is sample-identical to the original **outside** the
masked ranges. Phase 4's metrics are meaningless without this.

**Done when:** round-trip is exact, and the masked audio plays normally in any
player with silence where the sensitive spans were.

---

## Phase 4 — Evaluation

- `evaluate.py`: re-transcribe the masked WAV, compute WER and BERTScore over
  **non-redacted spans only**, plus entity precision/recall against a hand-
  labeled ground truth on the sample set.
- `redact eval` writes a metrics JSON + a printable table.

Build the ground-truth labels by hand for a handful of clips. It's tedious and
there's no way around it — without labels there is no precision/recall number,
and that's a number the panel will ask for.

**Done when:** you can produce the results table from the PPT with real numbers.

---

## Phase 5 — Demo polish

- Waveform figure with masked spans highlighted (matplotlib, saved to PNG).
- Before/after audio clips exported for the presentation.
- The detection table formatted for a slide.
- Timing numbers per minute of audio.

---

## Test priorities

In order of how much they protect the project's claims:

1. **Round-trip exactness.** mask -> unmask -> identical bytes.
2. **Outside-span immutability.** Masked file identical to original outside
   masked ranges.
3. **Manifest validation.** Each rule in MANIFEST_SCHEMA.md gets a failing-case
   test.
4. **Overlapping span merge.** Two entities 50 ms apart with 100 ms padding must
   merge into one range, not produce overlapping or double-encrypted spans.
5. **Spoken-number normalization.** "four five three two eight eight nine one"
   is detected as a card number.
6. **Boundary cases.** Entity at sample 0; entity at the very end of the file;
   entity spanning a Whisper segment boundary.

---

## Demo dataset

You need audio where sensitive data is actually spoken aloud. Options, in order
of effort:

1. **Record it yourself.** A 60-second mock support call with 3-5 planted
   entities (card number, SSN, name, date, phone). Fastest path, full control
   over ground truth, and no licensing question in the defense. Use fake
   numbers that still pass format validation (Luhn-valid test card numbers).
2. Public call-center corpora, if licensing permits redistribution in a report.

Either way: **hand-label the ground truth** (entity type + start/end time) for
every clip. That's what precision/recall is computed against.

---

## Likely defense questions, and where the answer lives

- *Why not just encrypt the whole file?* — CLAUDE.md, "What this project is".
  All-or-nothing at point of use; fails data minimization.
- *Why not just delete the sensitive part?* — Irreversible; no recovery for a
  fraud investigation or subpoena.
- *How do you know you didn't break the rest of the audio?* — Phase 4 metrics,
  computed on non-redacted spans only.
- *What happens if detection misses something?* — `detected_but_not_masked` plus
  the recall number. Be honest about the number; NER on names is the weak spot.
- *How long does it take?* — `timings_sec` in the manifest, per minute of audio.
- *What's actually new here?* — Reversible masking + published preservation
  verification. The components are all off-the-shelf; say so plainly rather
  than overclaiming.
