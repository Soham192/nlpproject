"""Shared plumbing: seeding, GPU check, run paths, checkpoints, metrics files, ROC/F1 helpers."""
from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .config import AEConfig
from .models import AutoEncoder, build


# --------------------------------------------------------------------------- environment

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_gpu() -> str:
    """Print the attached GPU and return "cuda"; fail clearly if there is none."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "No GPU attached. Training and GPU evaluation run on Colab only: "
            "Runtime -> Change runtime type -> T4 GPU, then rerun this cell.")
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name}", flush=True)
    return "cuda"


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(cfg: AEConfig) -> dict:
    """Recorded in every metrics.json / run_info.json."""
    return {"seed": cfg.seed, "dataset": cfg.data["dataset"], "dataset_config": cfg.data["config"],
            "dataset_revision": cfg.data["revision"], "whisper_model": cfg.features.whisper_model,
            "git_sha": git_sha(), "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


# --------------------------------------------------------------------------- paths

def hparam_tag(name: str, value: float) -> str:
    return f"{name}={value:g}"


def run_dir(cfg: AEConfig, variant: str, hparam: float | None = None) -> Path:
    """<drive>/<variant>/ for untuned variants, <drive>/<variant>/<name>=<value>/ for tuned ones."""
    grid = cfg.hparam_grid(variant)
    base = cfg.drive_root / variant
    return base if grid is None else base / hparam_tag(grid[0], hparam)


def layer_dir(cfg: AEConfig, variant: str, layer: str) -> Path:
    return cfg.drive_root / variant / layer


def baseline_dir(cfg: AEConfig, layer: str) -> Path:
    return cfg.drive_root / "baselines" / layer


def candidate_runs(cfg: AEConfig, variant: str) -> list[tuple[float | None, Path]]:
    """Every trained run of `variant` (one per grid value for tuned variants) that has a best.pt."""
    grid = cfg.hparam_grid(variant)
    values = [None] if grid is None else grid[1]
    runs = [(v, run_dir(cfg, variant, v)) for v in values]
    found = [(v, d) for v, d in runs if (d / "best.pt").exists()]
    if not found:
        raise FileNotFoundError(f"no trained {variant} checkpoint under {cfg.drive_root / variant}; "
                                f"run `python -m src.autoencoders.train --variant {variant}` first")
    missing = [str(d) for v, d in runs if (v, d) not in found]
    if missing:
        print(f"warning: {variant} grid runs not trained yet, excluded from dev selection: {missing}")
    return found


def record_choice(cfg: AEConfig, variant: str, layer: str, hparam: float | None) -> None:
    """Write the dev-selected hyperparameter for `layer` into run_info.json: the variant-level file
    (<drive>/<variant>/run_info.json -> chosen_hparam[layer]) and each grid run's own run_info.json
    (selected_for_layers). No-op for untuned variants."""
    grid = cfg.hparam_grid(variant)
    if grid is None:
        return
    vpath = cfg.drive_root / variant / "run_info.json"
    v = read_json(vpath) if vpath.exists() else {}
    v.update({"variant": variant, "hparam_name": grid[0], "grid": grid[1],
              "selection": "per layer, on dev-clean only"})
    v.setdefault("chosen_hparam", {})[layer] = hparam
    write_json(vpath, v)
    for value in grid[1]:
        rpath = run_dir(cfg, variant, value) / "run_info.json"
        if not rpath.exists():
            continue
        r = read_json(rpath)
        layers = set(r.get("selected_for_layers", [])) - {layer}
        if value == hparam:
            layers.add(layer)
        r["selected_for_layers"] = sorted(layers)
        write_json(rpath, r)


# --------------------------------------------------------------------------- checkpoints

def save_atomic(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_model(cfg: AEConfig, variant: str, hparam: float | None, ckpt: Path, device: str) -> AutoEncoder:
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = build(variant, cfg, hparam)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def load_model_from_checkpoint(path: str | Path, cfg: AEConfig, device: str) -> AutoEncoder:
    """Rebuild from a checkpoint's own recorded variant/hparam (used by the pipeline integrations)."""
    state = torch.load(path, map_location=device, weights_only=False)
    model = build(state["variant"], cfg, state.get("hparam"))
    model.load_state_dict(state["model"])
    return model.to(device).eval()


# --------------------------------------------------------------------------- metrics files

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_jsonable))
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """One JSON object per line (per-item scores: ROC curves, bootstrap CIs, breakdowns without re-running)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_jsonable) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- classification metrics

def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(score_pos > score_neg), ties count half (Mann-Whitney U). Higher score = more positive."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(neg) * len(pos)))


def f1_at(neg: np.ndarray, pos: np.ndarray, threshold: float) -> dict:
    """Predict positive when score >= threshold."""
    tp = int(np.sum(np.asarray(pos) >= threshold))
    fp = int(np.sum(np.asarray(neg) >= threshold))
    fn = len(pos) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"threshold": float(threshold), "f1": round(f1, 4), "precision": round(p, 4), "recall": round(r, 4),
            "tp": tp, "fp": fp, "fn": fn}


def best_f1_threshold(neg: np.ndarray, pos: np.ndarray) -> dict:
    """Threshold maximising F1 (call on dev only)."""
    cands = np.unique(np.concatenate([np.asarray(neg, float), np.asarray(pos, float)]))
    best = max((f1_at(neg, pos, t) for t in cands), key=lambda m: (m["f1"], -m["threshold"]))
    return best
