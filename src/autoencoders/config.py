"""Typed view of config.yaml's `autoencoder:` section (src/config.py stays unaware of it)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import DEFAULT_CONFIG_PATH, load_config

VARIANTS = ("dense", "sparse", "dae", "vae")
LAYERS = ("ingest", "asr", "leak")


@dataclass(frozen=True)
class FeatureConfig:
    whisper_model: str
    n_mels: int
    window_frames: int
    ola_hop_frames: int

    @property
    def input_dim(self) -> int:
        return self.window_frames * self.n_mels


@dataclass(frozen=True)
class AEConfig:
    seed: int
    variants: tuple[str, ...]
    drive_root: Path
    local_cache: Path
    features: FeatureConfig
    data: dict
    model: dict
    train: dict
    variant_hparams: dict
    corruption: dict
    eval: dict
    bench: dict
    integrations: dict
    raw: dict

    def hparam_grid(self, variant: str) -> tuple[str, list[float]] | None:
        """(name, grid) for variants with a dev-tuned knob, else None."""
        h = self.variant_hparams.get(variant)
        return (h["name"], list(h["grid"])) if h else None


def from_raw(ae: dict) -> AEConfig:
    for v in ae["variants"]:
        if v not in VARIANTS:
            raise ValueError(f"unknown autoencoder variant {v!r}; expected one of {VARIANTS}")
    return AEConfig(
        seed=int(ae["seed"]), variants=tuple(ae["variants"]),
        drive_root=Path(ae["drive_root"]), local_cache=Path(ae["local_cache"]),
        features=FeatureConfig(**ae["features"]),
        data=ae["data"], model=ae["model"], train=ae["train"], variant_hparams=ae.get("variant_hparams") or {},
        corruption=ae["corruption"], eval=ae["eval"], bench=ae["bench"],
        integrations=ae.get("integrations") or {}, raw=ae,
    )


def load_ae_config(path: str | Path = DEFAULT_CONFIG_PATH, drive_root: str | Path | None = None) -> AEConfig:
    raw = load_config(path).raw.get("autoencoder")
    if not raw:
        raise KeyError(f"{path} has no `autoencoder:` section")
    cfg = from_raw(raw)
    if drive_root is not None:
        from dataclasses import replace
        cfg = replace(cfg, drive_root=Path(drive_root))
    return cfg
