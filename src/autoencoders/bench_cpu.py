"""CPU inference time per minute of audio, for every variant. Run in a Colab CPU runtime.

    python -m src.autoencoders.bench_cpu

Refuses to run if a GPU is visible, so the number is a genuine CPU number. Latency depends only on the
architecture, not the weights, so trained checkpoints are not needed (randomly initialised models are timed;
the VAE's extra logvar head is included). Writes <drive>/<variant>/latency/metrics.json.
"""
from __future__ import annotations

import argparse
import platform
import statistics
import time

import numpy as np
import torch

from .common import layer_dir, provenance, seed_everything, write_json
from .config import AEConfig, load_ae_config
from .data import SR, log_mel
from .frontend import clean_mel, score_mel
from .models import build, param_count


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _time(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


@torch.no_grad()
def bench(cfg: AEConfig) -> dict[str, dict]:
    if torch.cuda.is_available():
        raise SystemExit("A GPU is attached. Switch to a CPU runtime (Runtime -> Change runtime type -> CPU) "
                         "so this measures CPU latency.")
    seed_everything(cfg.seed)
    b, f = cfg.bench, cfg.features
    audio = (0.1 * np.random.default_rng(cfg.seed).standard_normal(int(b["audio_sec"] * SR))).astype(np.float32)
    per_min = 60.0 / b["audio_sec"]
    mel_sec = _time(lambda: log_mel(audio, f.n_mels, "cpu"), b["warmup"], b["repeats"])
    mel = log_mel(audio, f.n_mels, "cpu")
    env = {"cpu_model": cpu_model(), "torch_threads": torch.get_num_threads(), "audio_sec": b["audio_sec"]}
    out = {}
    for variant in cfg.variants:
        grid = cfg.hparam_grid(variant)
        model = build(variant, cfg, grid[1][0] if grid else None).eval()
        clean = _time(lambda: clean_mel(model, mel, f.window_frames, f.ola_hop_frames), b["warmup"], b["repeats"])
        score = _time(lambda: score_mel(model, mel, f.window_frames, 1), b["warmup"], b["repeats"])
        m = {"variant": variant, "param_count": param_count(model),
             "ms_per_min_mel": round(1000 * mel_sec * per_min, 2),
             "ms_per_min_ae_clean_ola": round(1000 * clean * per_min, 2),       # layer 2 (and layer 1 scoring)
             "ms_per_min_ae_score_hop1": round(1000 * score * per_min, 2),      # layer 5
             "ms_per_min_mel_plus_ae_clean": round(1000 * (mel_sec + clean) * per_min, 2),
             "ola_hop_frames": f.ola_hop_frames, **env, **provenance(cfg)}
        write_json(layer_dir(cfg, variant, "latency") / "metrics.json", m)
        out[variant] = m
        print(f"[cpu] {variant:6s} mel={m['ms_per_min_mel']:.1f} ms/min  AE clean={m['ms_per_min_ae_clean_ola']:.1f} "
              f"ms/min  AE score hop1={m['ms_per_min_ae_score_hop1']:.1f} ms/min  ({env['cpu_model']}, "
              f"{env['torch_threads']} threads)", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-root", default=None)
    args = ap.parse_args(argv)
    bench(load_ae_config(drive_root=args.drive_root))


if __name__ == "__main__":
    main()
