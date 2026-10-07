"""Content-preservation verification + detection accuracy.

WER and BERTScore are computed over NON-redacted spans only: words whose
timestamps overlap any masked range are dropped from both reference (original
audio transcript) and hypothesis (masked audio transcript). Including masked
regions would make WER dominated by deletions by construction.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import alignment, asr
from .audio import load_wav, to_float_mono
from .config import Config, EvaluationConfig, NormalizeConfig, resolve_device
from .manifest import Manifest
from .timing import StageTimer

log = logging.getLogger("redact.evaluate")

_CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not", "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'ve": " have", "'m": " am",
}
_FILLERS = {"um", "uh", "erm", "ah", "hmm", "mm", "uhm"}


def normalize_text(text: str, n: NormalizeConfig) -> str:
    """Applied identically to reference and hypothesis."""
    t = text.replace("’", "'")
    if n.lowercase:
        t = t.lower()
    if n.expand_contractions:
        for k in ("won't", "can't"):
            t = re.sub(rf"\b{k}\b", _CONTRACTIONS[k], t, flags=re.I)
        for suf in ("n't", "'re", "'s", "'d", "'ll", "'ve", "'m"):
            t = re.sub(rf"(\w){re.escape(suf)}\b", rf"\1{_CONTRACTIONS[suf]}", t, flags=re.I)
    if n.strip_punctuation:
        t = re.sub(r"[^\w\s']", " ", t)
        t = re.sub(r"(?<!\w)'|'(?!\w)", " ", t)
    words = t.split()
    if n.remove_filler_words:
        words = [w for w in words if w.lower() not in _FILLERS]
    return " ".join(words)


def _overlaps(t0: float, t1: float, ranges_sec: list[tuple[float, float]]) -> bool:
    return any(t0 < e and s < t1 for s, e in ranges_sec)


def unredacted_text(words: list[alignment.Word], ranges_sec: list[tuple[float, float]]) -> str:
    return " ".join(w.text for w in words if not _overlaps(w.start, w.end, ranges_sec))


def wer(ref: str, hyp: str) -> dict:
    import jiwer

    if not ref.strip():
        return {"wer": None, "substitutions": 0, "deletions": 0, "insertions": 0, "ref_words": 0}
    out = jiwer.process_words(ref, hyp)
    return {"wer": round(out.wer, 4), "substitutions": out.substitutions, "deletions": out.deletions,
            "insertions": out.insertions, "ref_words": len(ref.split())}


def bertscore(ref: str, hyp: str, cfg: EvaluationConfig) -> dict:
    from bert_score import score

    if not ref.strip() or not hyp.strip():
        return {"precision": None, "recall": None, "f1": None, "model": cfg.bertscore_model}
    P, R, F = score([hyp], [ref], model_type=cfg.bertscore_model, lang=cfg.bertscore_lang,
                    device=resolve_device("auto"), verbose=False)
    return {"precision": round(float(P[0]), 4), "recall": round(float(R[0]), 4), "f1": round(float(F[0]), 4),
            "model": cfg.bertscore_model}


def entity_metrics(manifest: Manifest, ground_truth: list[dict], min_coverage: float = 0.5) -> dict:
    """Precision/recall of the manifest's masked spans against hand-labelled ground truth."""
    return span_metrics([(s.start_sec, s.end_sec, s.entity_type) for s in manifest.spans], ground_truth,
                        min_coverage)


def span_metrics(masked: list[tuple[float, float, str]], ground_truth: list[dict],
                 min_coverage: float = 0.5) -> dict:
    """Precision/recall of masked (start_sec, end_sec, type) ranges against ground truth (type + start/end sec).

    - A ground-truth entity is *recalled* if >= min_coverage of its duration lies inside masked ranges.
    - A masked span is a *true positive* if it overlaps any ground-truth entity.
    """

    def coverage(g0: float, g1: float) -> float:
        if g1 <= g0:
            return 0.0
        cov = sum(max(0.0, min(g1, e) - max(g0, s)) for s, e, _ in masked)
        return min(1.0, cov / (g1 - g0))

    per_gt = []
    for g in ground_truth:
        c = coverage(g["start_sec"], g["end_sec"])
        per_gt.append({**g, "coverage": round(c, 3), "recalled": c >= min_coverage})
    tp_spans = sum(1 for s, e, _ in masked if any(s < g["end_sec"] and g["start_sec"] < e for g in ground_truth))
    n_rec = sum(1 for g in per_gt if g["recalled"])
    by_type: dict[str, dict] = {}
    for g in per_gt:
        b = by_type.setdefault(g["entity_type"], {"total": 0, "recalled": 0})
        b["total"] += 1
        b["recalled"] += int(g["recalled"])
    for b in by_type.values():
        b["recall"] = round(b["recalled"] / b["total"], 4)
    precision = tp_spans / len(masked) if masked else None
    recall = n_rec / len(per_gt) if per_gt else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    return {
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1": None if f1 is None else round(f1, 4),
        "masked_spans": len(masked), "true_positive_spans": tp_spans,
        "ground_truth_entities": len(per_gt), "recalled_entities": n_rec,
        "min_coverage": min_coverage, "by_type": by_type, "per_entity": per_gt,
    }


def outside_span_identical(original: np.ndarray, masked: np.ndarray, manifest: Manifest) -> bool:
    """Invariant: masked audio is sample-identical to the original outside masked ranges."""
    if original.shape != masked.shape:
        return False
    keep = np.ones(len(original), dtype=bool)
    for s in manifest.spans:
        keep[s.sample_start:s.sample_end] = False
    return bool(np.array_equal(original[keep], masked[keep]))


@dataclass
class EvalResult:
    metrics: dict

    def table(self) -> str:
        m = self.metrics
        w, b, st = m["wer"], m["bertscore"], m["stats"]
        rows = [
            ("Fraction of audio masked", f"{100 * st['fraction_masked']:.1f}%"),
            ("Spans masked", str(st["spans_masked"])),
            ("Outside-span sample identity", "PASS" if m["outside_span_identical"] else "FAIL"),
            ("WER (non-redacted)", "n/a" if w["wer"] is None else f"{100 * w['wer']:.2f}%"),
            ("  S / D / I  (ref words)", f"{w['substitutions']} / {w['deletions']} / {w['insertions']}  ({w['ref_words']})"),
            ("BERTScore F1 (non-redacted)", "n/a" if b["f1"] is None else f"{b['f1']:.4f}"),
        ]
        e = m.get("entities")
        if e:
            fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
            rows += [("Entity precision", fmt(e["precision"])), ("Entity recall", fmt(e["recall"])),
                     ("Entity F1", fmt(e["f1"]))]
            for t, bt in sorted(e["by_type"].items()):
                rows.append((f"  recall {t}", f"{bt['recalled']}/{bt['total']}"))
        width = max(len(r[0]) for r in rows)
        line = "-" * (width + 22)
        return "\n".join([line, *(f"{k:<{width}}  {v}" for k, v in rows), line])


def evaluate(original_path: str | Path, masked_path: str | Path, manifest: Manifest, cfg: Config,
             ground_truth: list[dict] | None = None, out_dir: str | Path | None = None,
             ref_words: list[alignment.Word] | None = None, on_stage=None) -> EvalResult:
    """`ref_words`: the original's aligned words if already computed (e.g. words.json from the mask run)."""
    timer = StageTimer(cfg.logging.log_stage_timings, on_stage)
    orig, _ = load_wav(original_path, cfg.audio)
    masked, _ = load_wav(masked_path, cfg.audio)
    ranges = [(s.sample_start / manifest.source.sample_rate, s.sample_end / manifest.source.sample_rate)
              for s in manifest.spans] if cfg.evaluation.exclude_masked_spans else []

    def words_for(samples: np.ndarray, tag: str) -> list[alignment.Word]:
        audio = to_float_mono(samples)
        with timer.stage(f"asr_{tag}"):
            res = asr.transcribe(audio, cfg.asr)
        with timer.stage(f"align_{tag}"):
            return alignment.align(res["segments"], audio, cfg.alignment, res.get("language", "en"))

    if ref_words is None:
        ref_words = words_for(orig, "original")
    hyp_words = words_for(masked, "masked")
    n = cfg.evaluation.normalize
    ref = normalize_text(unredacted_text(ref_words, ranges), n)
    hyp = normalize_text(unredacted_text(hyp_words, ranges), n)
    with timer.stage("metrics"):
        w = wer(ref, hyp)
        b = bertscore(ref, hyp, cfg.evaluation)
    metrics = {
        "run_id": manifest.run_id,
        "original": str(original_path), "masked": str(masked_path),
        "outside_span_identical": outside_span_identical(orig, masked, manifest),
        "wer": w, "bertscore": b, "stats": manifest.stats,
        "reference_text": ref, "hypothesis_text": hyp,
        "timings_sec": timer.as_dict(),
    }
    if ground_truth is not None:
        metrics["entities"] = entity_metrics(manifest, ground_truth)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return EvalResult(metrics)


def load_ground_truth(path: str | Path) -> list[dict]:
    """JSON: list of {"entity_type", "start_sec", "end_sec"} or {"entities": [...]}."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["entities"] if isinstance(d, dict) else d
