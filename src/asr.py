"""Whisper transcription (via WhisperX's faster-whisper backend) with segment timestamps."""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from .config import ASRConfig, resolve_compute_type, resolve_device

log = logging.getLogger("redact.asr")


def _model_name(name: str) -> str:
    # config uses HF-style ids ("openai/whisper-small"); faster-whisper wants "small".
    return name.split("/")[-1].removeprefix("whisper-")


@lru_cache(maxsize=2)
def _load(name: str, device: str, compute_type: str, language: str):
    import whisperx

    log.info("loading ASR model %s on %s (%s)", name, device, compute_type)
    return whisperx.load_model(name, device, compute_type=compute_type, language=language)


def transcribe(audio: np.ndarray, cfg: ASRConfig, batch_size: int = 8) -> dict:
    """audio: float32 mono 16 kHz. Returns {"language", "segments": [{"text","start","end"}]}."""
    device = resolve_device(cfg.device)
    model = _load(_model_name(cfg.model), device, resolve_compute_type(cfg.compute_type, device), cfg.language)
    result = model.transcribe(audio, batch_size=batch_size, language=cfg.language)
    segments = [
        {"text": s["text"].strip(), "start": float(s["start"]), "end": float(s["end"])}
        for s in result["segments"]
    ]
    return {"language": result.get("language", cfg.language), "segments": segments}
