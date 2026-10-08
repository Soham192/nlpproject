"""Load config.yaml once into typed, frozen dataclasses."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    channels: int
    sample_width_bytes: int
    reject_non_wav: bool


@dataclass(frozen=True)
class ASRConfig:
    model: str
    language: str
    device: str
    compute_type: str


@dataclass(frozen=True)
class AlignmentConfig:
    model: str
    device: str
    use_mfa: bool


@dataclass(frozen=True)
class DetectionConfig:
    entities: tuple[str, ...]
    spacy_model: str
    score_threshold: float
    normalize_spoken_numbers: bool


@dataclass(frozen=True)
class MaskingConfig:
    padding_ms: int
    replacement: str
    tone_hz: int
    cipher: str
    key_length_bytes: int
    key_file: str


@dataclass(frozen=True)
class NormalizeConfig:
    lowercase: bool
    strip_punctuation: bool
    expand_contractions: bool
    remove_filler_words: bool


@dataclass(frozen=True)
class EvaluationConfig:
    exclude_masked_spans: bool
    bertscore_model: str
    bertscore_lang: str
    normalize: NormalizeConfig


@dataclass(frozen=True)
class OutputConfig:
    dir: str
    save_intermediates: bool
    manifest_suffix: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    log_stage_timings: bool


@dataclass(frozen=True)
class ApiConfig:
    allow_original_playback: bool = True
    max_upload_mb: int = 100
    workdir: str = "data/outputs/api"
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)


@dataclass(frozen=True)
class Config:
    audio: AudioConfig
    asr: ASRConfig
    alignment: AlignmentConfig
    detection: DetectionConfig
    masking: MaskingConfig
    evaluation: EvaluationConfig
    output: OutputConfig
    logging: LoggingConfig
    api: ApiConfig = field(default_factory=ApiConfig)
    raw: dict = field(repr=False, compare=False, default_factory=dict)


def _build(raw: dict) -> Config:
    ev = dict(raw["evaluation"])
    ev["normalize"] = NormalizeConfig(**ev["normalize"])
    det = dict(raw["detection"])
    det["entities"] = tuple(det["entities"])
    api = dict(raw.get("api") or {})
    if "cors_origins" in api:
        api["cors_origins"] = tuple(api["cors_origins"])
    env = os.environ.get("ALLOW_ORIGINAL_PLAYBACK")
    if env is not None:
        api["allow_original_playback"] = env.strip().lower() in ("1", "true", "yes", "on")
    return Config(
        audio=AudioConfig(**raw["audio"]),
        asr=ASRConfig(**raw["asr"]),
        alignment=AlignmentConfig(**raw["alignment"]),
        detection=DetectionConfig(**det),
        masking=MaskingConfig(**raw["masking"]),
        evaluation=EvaluationConfig(**ev),
        output=OutputConfig(**raw["output"]),
        logging=LoggingConfig(**raw["logging"]),
        api=ApiConfig(**api),
        raw=raw,
    )


@lru_cache(maxsize=None)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and cache the config. Called once per process per path."""
    with open(path, "r", encoding="utf-8") as f:
        return _build(yaml.safe_load(f))


def ae_integration(raw: dict, name: str) -> dict | None:
    """The `autoencoder:` section if `autoencoder.integrations.<name>.enabled`, else None.

    Lets callers check the flag without importing src.autoencoders (or torch models) when it is off.
    """
    ae = raw.get("autoencoder") or {}
    return ae if ((ae.get("integrations") or {}).get(name) or {}).get("enabled") else None


def resolve_device(device: str) -> str:
    """'auto' -> 'cuda' when available, else 'cpu'."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_compute_type(compute_type: str, device: str) -> str:
    """float16 is unsupported on CPU; fall back to int8 there."""
    if device == "cpu" and compute_type == "float16":
        return "int8"
    return compute_type
