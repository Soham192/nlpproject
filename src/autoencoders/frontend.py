"""Apply a trained AE to Whisper log-mels, plus the three optional pipeline integrations.

Nothing here writes audio. The ASR front-end cleans a COPY of the mel on its way into Whisper;
the masked WAV stays sample-identical outside masked spans (CLAUDE.md invariant).
"""
from __future__ import annotations

import contextlib
import importlib
import logging
from functools import lru_cache

import numpy as np
import torch

from .config import AEConfig, from_raw
from .data import HOP, frame_windows, log_mel, overlap_add
from .models import AutoEncoder

log = logging.getLogger("redact.ae.frontend")


# --------------------------------------------------------------------------- core ops

@torch.no_grad()
def score_mel(model: AutoEncoder, mel: torch.Tensor, window: int, hop: int, batch: int = 8192) -> torch.Tensor:
    """Per-window score (N,) over a (T, M) mel. Low = looks like the speech the AE was trained on."""
    wins, _, _ = frame_windows(mel.float(), window, hop)
    return torch.cat([model.score(wins[i:i + batch]) for i in range(0, len(wins), batch)])


@torch.no_grad()
def clean_mel(model: AutoEncoder, mel: torch.Tensor, window: int, hop: int, batch: int = 8192) -> torch.Tensor:
    """Reconstruct a (T, M) mel through the AE with overlapping windows + overlap-add. Returns a new tensor."""
    wins, starts, padded = frame_windows(mel.float(), window, hop)
    rec = torch.cat([model.reconstruct(wins[i:i + batch]) for i in range(0, len(wins), batch)])
    return overlap_add(rec, starts, padded, window, mel.shape[0]).to(mel.dtype)


def clean_whisper_mel(model: AutoEncoder, mel: torch.Tensor, n_content: int, window: int, hop: int) -> torch.Tensor:
    """Whisper-layout (M, T) mel: clean the first `n_content` frames, leave the padding frames alone."""
    out = mel.clone()
    n = min(n_content, mel.shape[-1])
    if n > 0:
        out[:, :n] = clean_mel(model, mel[:, :n].T, window, hop).T
    return out


# --------------------------------------------------------------------------- integrations (pipeline / evaluate)

_MODELS: dict[tuple[str, str], AutoEncoder] = {}


def _model_for(ae_raw: dict, name: str) -> tuple[AutoEncoder, AEConfig, dict]:
    from .common import load_model_from_checkpoint

    cfg = from_raw(ae_raw)
    icfg = cfg.integrations[name]
    if not icfg.get("checkpoint"):
        raise ValueError(f"autoencoder.integrations.{name} is enabled but has no `checkpoint` in config.yaml")
    key = (icfg["checkpoint"], "cuda" if torch.cuda.is_available() else "cpu")
    if key not in _MODELS:
        _MODELS[key] = load_model_from_checkpoint(key[0], cfg, key[1])
    return _MODELS[key], cfg, icfg


def ingest_score(audio: np.ndarray, ae_raw: dict) -> dict:
    """Layer 1: mean per-window reconstruction error of a whole file; above threshold = not usable speech."""
    model, cfg, icfg = _model_for(ae_raw, "ingest_gate")
    device = next(model.parameters()).device
    f = cfg.features
    s = float(score_mel(model, log_mel(audio, f.n_mels, device), f.window_frames, f.ola_hop_frames).mean())
    thr = icfg.get("threshold")
    return {"score": s, "threshold": thr, "flagged": thr is not None and s >= thr,
            "checkpoint": icfg["checkpoint"], "action": icfg.get("action", "warn")}


@contextlib.contextmanager
def _patched_whisper_mel(model: AutoEncoder, window: int, hop: int):
    """Inside the block, whisper.transcribe sees AE-cleaned mels. Scoped; restored on exit."""
    tmod = importlib.import_module("whisper.transcribe")
    orig = tmod.log_mel_spectrogram

    def cleaned(audio, n_mels=80, padding=0, device=None):
        mel = orig(audio, n_mels, padding=padding, device=device)
        n_content = mel.shape[-1] - padding // HOP
        return clean_whisper_mel(model.to(mel.device), mel, n_content, window, hop)

    tmod.log_mel_spectrogram = cleaned
    try:
        yield
    finally:
        tmod.log_mel_spectrogram = orig


@lru_cache(maxsize=1)
def _whisper(name: str, device: str):
    import whisper

    return whisper.load_model(name, device=device)


def transcribe_with_frontend(audio: np.ndarray, asr_cfg, ae_raw: dict) -> dict:
    """Layer 2 integration: openai-whisper transcribe() with the AE cleaning a copy of the mel.

    Same return shape as src.asr.transcribe, so WhisperX alignment downstream is unchanged.
    """
    model, cfg, _ = _model_for(ae_raw, "asr_frontend")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wmodel = _whisper(asr_cfg.model.split("/")[-1].removeprefix("whisper-"), device)
    f = cfg.features
    with _patched_whisper_mel(model, f.window_frames, f.ola_hop_frames):
        res = wmodel.transcribe(audio.astype(np.float32), language=asr_cfg.language, fp16=device == "cuda")
    segs = [{"text": s["text"].strip(), "start": float(s["start"]), "end": float(s["end"])} for s in res["segments"]]
    return {"language": res.get("language", asr_cfg.language), "segments": segs, "ae_frontend": True}


def span_scores(model: AutoEncoder, span: np.ndarray, window: int, device) -> np.ndarray:
    """Per-window AE scores (hop 1) over a span's own samples; mel computed on the span alone."""
    return score_mel(model, log_mel(span, model_n_mels(model, window), device), window, 1).cpu().numpy()


def model_n_mels(model: AutoEncoder, window: int) -> int:
    return model.input_dim // window


def leak_scores(errs: np.ndarray, mu: float, sigma: float) -> dict[str, float]:
    """Span-level scores from per-window errors. calibrated: max |err - mu_fill| / sigma_fill; literal: max(-err)."""
    return {"calibrated": float(np.max(np.abs(errs - mu) / sigma)), "literal": float(np.max(-errs))}


def leak_check(masked: np.ndarray, spans: list, sample_rate: int, replacement: str, ae_raw: dict) -> dict:
    """Layer 5 integration: per manifest span, is there speech-like audio inside the masked range?

    Uses the dev-calibrated (mu, sigma, threshold) for the active fill type from eval_leak's metrics.json.
    Note: the pipeline overwrites the whole padded range with fill, so on pipeline output this only fires
    when something other than the fill ended up inside the range; see the research eval for the
    true-extent vs masked-range setup.
    """
    from .common import read_json

    model, cfg, icfg = _model_for(ae_raw, "leak_check")
    calib = read_json(icfg["metrics"])["chosen"]["calibration"].get(replacement)
    if calib is None:
        return {"skipped": f"no calibration for fill {replacement!r}"}
    device = next(model.parameters()).device
    w = cfg.features.window_frames
    out = []
    for s in spans:
        seg = masked[s.sample_start:s.sample_end]
        if len(seg) < (w + 1) * HOP:
            out.append({"id": s.id, "skipped": "span shorter than one AE window"})
            continue
        sc = leak_scores(span_scores(model, seg, w, device), calib["mu"], calib["sigma"])["calibrated"]
        out.append({"id": s.id, "score": round(sc, 4), "flagged": sc >= calib["threshold"]})
    return {"fill": replacement, "threshold": calib["threshold"], "spans": out,
            "flagged": sum(1 for o in out if o.get("flagged"))}

