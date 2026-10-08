"""Orchestrator: wires the stages together. Each stage stays independently callable.

    input.wav -> [asr] -> [alignment] -> [detection] -> [mapping] -> [masking] -> output.wav + manifest
"""
from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from . import alignment, asr, detect, mapping, mask
from .audio import AudioParams, load_wav, sha256_file, to_float_mono, write_wav_like
from .config import Config, ae_integration
from .manifest import CipherInfo, Manifest, MaskedInfo, SourceInfo, Span, text_preview, write_manifest
from .timing import StageTimer

log = logging.getLogger("redact.pipeline")


def new_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + "_" + secrets.token_hex(3)


@dataclass
class Analysis:
    """Output of stages 1-4 (no audio modified)."""
    samples: np.ndarray
    params: AudioParams
    asr: dict
    transcript: detect.Transcript
    detections: list[detect.Detection]
    to_mask: list[detect.Detection]
    not_masked: list[detect.Detection]
    spans: list[mapping.SampleSpan]

    def rows(self) -> list[dict]:
        """One row per detection, for the `detect` table (CLI and web UI share this).

        `span_id` names the merged masking range a row falls into (matches manifest span ids).
        """
        span_of = {id(d): f"span_{i:03d}" for i, s in enumerate(self.spans, 1) for d in s.detections}
        masked = {id(d) for d in self.to_mask}
        low = {id(d) for d in self.not_masked}
        out = []
        for d in self.detections:
            w0, w1 = mapping.chars_to_words(self.transcript, d.char_start, d.char_end)
            t0, t1 = mapping.words_to_seconds(self.transcript.words, w0, w1)
            if id(d) in masked:
                reason = None
            elif id(d) in low:
                reason = "below score_threshold"
            else:
                reason = "below score_threshold; overlaps a masked span"
            out.append({
                "entity_type": d.entity_type, "score": d.score, "text": d.text, "text_preview": text_preview(d.text),
                "char_start": d.char_start, "char_end": d.char_end,
                "start_sec": round(t0, 3), "end_sec": round(t1, 3),
                "will_mask": id(d) in masked, "reason": reason, "span_id": span_of.get(id(d)),
            })
        return out


class Pipeline:
    def __init__(self, cfg: Config, run_id: str | None = None, out_dir: str | Path | None = None,
                 on_stage: Callable[[str], None] | None = None) -> None:
        self.cfg = cfg
        self.run_id = run_id or new_run_id()
        self.timer = StageTimer(cfg.logging.log_stage_timings, on_stage)
        self.out_dir = Path(out_dir) if out_dir else Path(cfg.output.dir) / self.run_id

    # -- intermediates ---------------------------------------------------------
    def save(self, name: str, obj) -> None:
        if not self.cfg.output.save_intermediates:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")

    # -- stages ----------------------------------------------------------------
    def run_ingest_gate(self, samples: np.ndarray) -> dict | None:
        """Optional AE quality gate (off by default). Reads the audio, never modifies it."""
        ae = ae_integration(self.cfg.raw, "ingest_gate")
        if ae is None:
            return None
        from .autoencoders.frontend import ingest_score

        with self.timer.stage("ingest_gate"):
            res = ingest_score(to_float_mono(samples), ae)
        self.save("ingest_gate.json", res)
        if res["flagged"]:
            msg = f"ingest gate: input does not look like usable speech (score {res['score']:.4f} >= {res['threshold']})"
            if res["action"] == "reject":
                raise ValueError(msg)
            log.warning(msg)
        return res

    def run_asr(self, samples: np.ndarray) -> dict:
        with self.timer.stage("asr"):
            res = asr.transcribe(to_float_mono(samples), self.cfg.asr,
                                 ae_frontend=ae_integration(self.cfg.raw, "asr_frontend"))
        self.save("transcript.json", res)
        return res

    def run_alignment(self, asr_result: dict, samples: np.ndarray) -> list[alignment.Word]:
        with self.timer.stage("alignment"):
            words = alignment.align(asr_result["segments"], to_float_mono(samples), self.cfg.alignment,
                                    asr_result.get("language", "en"))
        self.save("words.json", [w.to_dict() for w in words])
        return words

    def run_detection(self, words: list[alignment.Word]):
        with self.timer.stage("detection"):
            transcript, dets = detect.detect_words(words, self.cfg.detection)
            to_mask, low = detect.split_by_threshold(dets, self.cfg.detection.score_threshold)
        self.save("detections.json", {
            "transcript": transcript.text,
            "detections": [d.to_dict() for d in dets],
            "to_mask": [d.to_dict() for d in to_mask],
            "below_threshold": [d.to_dict() for d in low],
        })
        return transcript, dets, to_mask, low

    def run_mapping(self, to_mask, transcript, params: AudioParams) -> list[mapping.SampleSpan]:
        with self.timer.stage("mapping"):
            spans = mapping.map_detections(to_mask, transcript, params.sample_rate, params.total_samples,
                                           self.cfg.masking.padding_ms)
        self.save("spans.json", [
            {"sample_start": s.sample_start, "sample_end": s.sample_end, "entity_type": s.entity_type,
             "start_sec": s.sample_start / params.sample_rate, "end_sec": s.sample_end / params.sample_rate,
             "words": [w.text for w in transcript.words[s.word_start_idx:s.word_end_idx]]}
            for s in spans])
        return spans

    def analyze(self, input_path: str | Path, words: list[alignment.Word] | None = None) -> Analysis:
        """Stages 1-4: everything up to (not including) modifying audio. Used by `detect`.

        `words`: aligned words from an earlier run on the same file. ASR + alignment are
        deterministic for a given file and config, so re-detecting with a different entity
        set (the UI's checkboxes) only needs to redo detection and mapping.
        """
        samples, params = load_wav(input_path, self.cfg.audio)
        self.run_ingest_gate(samples)
        if words is None:
            res = self.run_asr(samples)
            words = self.run_alignment(res, samples)
        else:
            res = {"language": self.cfg.asr.language, "segments": [], "reused_words": True}
            self.save("words.json", [w.to_dict() for w in words])
        transcript, dets, to_mask, low = self.run_detection(words)
        spans = self.run_mapping(to_mask, transcript, params)
        return Analysis(samples, params, res, transcript, dets, to_mask, low, spans)

    def mask(self, input_path: str | Path, output_path: str | Path, key: bytes, key_id: str = "local-key-01",
             manifest_path: str | Path | None = None, words: list[alignment.Word] | None = None) -> Manifest:
        input_path, output_path = Path(input_path), Path(output_path)
        manifest_path = Path(manifest_path) if manifest_path else default_manifest_path(output_path, self.cfg)
        a = self.analyze(input_path, words)
        m = self.cfg.masking
        with self.timer.stage("masking"):
            masked, enc = mask.mask_samples(a.samples, [(s.sample_start, s.sample_end) for s in a.spans], key,
                                            a.params.sample_rate, m.replacement, m.tone_hz)
            write_wav_like(output_path, masked, input_path)
        manifest = build_manifest(self, a, enc, input_path, output_path, key_id)
        write_manifest(manifest, manifest_path, key)
        if self.cfg.output.save_intermediates:
            self.save("manifest.json", manifest.to_dict())
        log.info("masked %d span(s), %.1f%% of audio -> %s", len(manifest.spans),
                 100 * manifest.stats["fraction_masked"], output_path)
        return manifest


def default_manifest_path(output_wav: Path, cfg: Config) -> Path:
    return output_wav.with_suffix("").with_name(output_wav.stem + cfg.output.manifest_suffix)


def build_manifest(p: Pipeline, a: Analysis, enc: list[mask.EncryptedRange], input_path: Path,
                   output_path: Path, key_id: str) -> Manifest:
    cfg, params = p.cfg, a.params
    sr = params.sample_rate
    spans = []
    for i, (s, e) in enumerate(zip(a.spans, enc), 1):
        text = a.transcript.text[s.char_start:s.char_end]
        spans.append(Span(
            id=f"span_{i:03d}", entity_type=s.entity_type, score=s.score, text_preview=text_preview(text),
            char_start=s.char_start, char_end=s.char_end,
            word_start_idx=s.word_start_idx, word_end_idx=s.word_end_idx,
            start_sec=round(s.sample_start / sr, 3), end_sec=round(s.sample_end / sr, 3),
            sample_start=s.sample_start, sample_end=s.sample_end,
            padding_applied_ms=cfg.masking.padding_ms, **mask.encode(e)))
    not_masked = []
    for d in a.not_masked:
        w0, w1 = mapping.chars_to_words(a.transcript, d.char_start, d.char_end)
        t0, t1 = mapping.words_to_seconds(a.transcript.words, w0, w1)
        not_masked.append({"entity_type": d.entity_type, "score": d.score, "start_sec": round(t0, 3),
                           "end_sec": round(t1, 3), "reason": "below score_threshold"})
    n_masked = sum(s.sample_end - s.sample_start for s in spans)
    return Manifest(
        run_id=p.run_id,
        source=SourceInfo(filename=input_path.name, sha256=sha256_file(input_path), sample_rate=sr,
                          channels=params.channels, sample_width_bytes=params.sample_width_bytes,
                          total_samples=params.total_samples, duration_sec=round(params.duration_sec, 3)),
        masked=MaskedInfo(filename=output_path.name, sha256=sha256_file(output_path)),
        cipher=CipherInfo(algorithm=cfg.masking.cipher, key_length_bytes=cfg.masking.key_length_bytes,
                          key_id=key_id),
        config={"padding_ms": cfg.masking.padding_ms, "replacement": cfg.masking.replacement,
                "asr_model": cfg.asr.model, "alignment_model": cfg.alignment.model,
                "score_threshold": cfg.detection.score_threshold},
        spans=spans,
        detected_but_not_masked=not_masked,
        stats={"spans_masked": len(spans), "samples_masked": n_masked,
               "duration_masked_sec": round(n_masked / sr, 3),
               "fraction_masked": round(n_masked / params.total_samples, 4) if params.total_samples else 0.0},
        timings_sec=p.timer.as_dict(),
    )


def unmask(masked_path: str | Path, manifest: Manifest, key: bytes, output_path: str | Path,
           cfg: Config | None = None) -> bool:
    """Restore the original audio. Returns True iff the result's sha256 matches source.sha256."""
    if sha256_file(masked_path) != manifest.masked.sha256:
        raise ValueError(f"{masked_path} does not match manifest.masked.sha256 (wrong or modified file)")
    samples, _ = load_wav(masked_path, cfg.audio if cfg else None)
    restored = mask.unmask_samples(samples, manifest.spans, key)
    write_wav_like(output_path, restored, masked_path)
    ok = sha256_file(output_path) == manifest.source.sha256
    log.info("unmasked -> %s (sha256 %s)", output_path, "MATCHES source" if ok else "DOES NOT MATCH source")
    return ok
