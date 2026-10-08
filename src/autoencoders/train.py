"""Train one AE variant on the cached LibriSpeech features (Colab GPU only).

    python -m src.autoencoders.train --variant dae --resume      # trains every min_snr_db in the grid
    python -m src.autoencoders.train --variant sparse            # trains every l1_weight in the grid
    python -m src.autoencoders.train --variant vae --hparam 0.5  # one grid value

Writes to <drive_root>/<variant>[/<hparam>=<v>]/:
  latest.pt (every epoch), best.pt (lowest val loss), ckpt_epochNN.pt (last few),
  train_log.csv, curves.png, run_info.json
All variants share layer sizes, optimiser, lr, batch size and epoch count (config.yaml `autoencoder:`).
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .common import load_model, provenance, require_gpu, run_dir, save_atomic, seed_everything, write_json
from .config import VARIANTS, AEConfig, load_ae_config
from .data import TrainData, ensure_local_cache, load_split, log_mel, telephone_chain
from .frontend import clean_mel
from .models import AutoEncoder, build, param_count

log = logging.getLogger("redact.ae.train")

VAL_SEED_OFFSET = 1_000_003  # fixed corruption of the DAE's validation set, distinct from any epoch seed


def epoch_generator(seed: int, epoch: int, device: str) -> torch.Generator:
    """Per-epoch RNG, so a resumed run samples exactly the windows/corruptions it would have."""
    return torch.Generator(device=device).manual_seed(seed * 1000 + epoch)


def train_step(model: AutoEncoder, opt: torch.optim.Optimizer, x_in: torch.Tensor, target: torch.Tensor,
               grad_clip: float) -> dict[str, float]:
    model.train()
    loss, parts = model.loss(x_in, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    return {"loss": float(loss.detach()), **parts}


@torch.no_grad()
def validate(model: AutoEncoder, val: TrainData, starts: torch.Tensor, batch: int) -> dict[str, float]:
    """Val loss on the variant's own objective, plus clean-input reconstruction MSE (comparable across variants)."""
    model.eval()
    tot_loss = tot_mse = 0.0
    n = 0
    for i in range(0, len(starts), batch):
        s = starts[i:i + batch]
        x_in, target = val.batch(s, corrupted=model.corrupts_input)
        loss, _ = model.loss(x_in, target)
        clean = target
        mse = torch.mean((model.reconstruct(clean) - clean) ** 2)
        tot_loss += float(loss) * len(s)
        tot_mse += float(mse) * len(s)
        n += len(s)
    return {"val_loss": tot_loss / n, "val_mse_clean": tot_mse / n}


RECON_UTTS = 3          # first N test_asr utterances: the same ones for every variant
RECON_MAX_FRAMES = 500  # 5 s
RECON_SNR_DB = 10.0


@torch.no_grad()
def save_recon_sample(cfg: AEConfig, variant: str, hparam: float | None, rd: Path, root: Path, device: str) -> Path:
    """recon_sample.npz in the run dir, for recon_examples.png: log-mels of a few fixed test utterances
    (clean and at RECON_SNR_DB through the telephone chain) and best.pt's reconstruction of each.
    Visualisation only; no metric is computed from test data here."""
    model = load_model(cfg, variant, hparam, rd / "best.pt", device)
    split = load_split(root, "test_asr")
    f = cfg.features
    out: dict[str, np.ndarray] = {}
    ids = []
    for i in range(min(RECON_UTTS, len(split))):
        audio = torch.from_numpy(split.utt_audio(i))
        g = torch.Generator().manual_seed(cfg.seed + i)
        noisy = torch.round(telephone_chain(audio, RECON_SNR_DB, cfg.corruption, g) * 32767) / 32768.0
        for tag, x in (("clean", audio), ("noisy", noisy)):
            mel = log_mel(x, f.n_mels, device)[:RECON_MAX_FRAMES]
            out[f"{i}_{tag}_input"] = mel.cpu().numpy()
            out[f"{i}_{tag}_recon"] = clean_mel(model, mel, f.window_frames, f.ola_hop_frames).cpu().numpy()
        ids.append(split.index[i]["id"])
    path = rd / "recon_sample.npz"
    np.savez_compressed(path, ids=np.array(ids), variant=variant, hparam=np.nan if hparam is None else hparam,
                        snr_db=RECON_SNR_DB, ola_hop_frames=f.ola_hop_frames, **out)
    return path


def plot_curves(history: list[dict], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=120)
    ax.plot(ep, [h["train_loss"] for h in history], label="train loss")
    ax.plot(ep, [h["val_loss"] for h in history], label="val loss (dev-clean)")
    ax.plot(ep, [h["val_mse_clean"] for h in history], "--", label="val MSE, clean input")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _write_log(history: list[dict], path: Path) -> None:
    keys = list(dict.fromkeys(k for h in history for k in h))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)


def train_run(cfg: AEConfig, variant: str, hparam: float | None, resume: bool, device: str) -> Path:
    rd = run_dir(cfg, variant, hparam)
    latest = rd / "latest.pt"
    if latest.exists() and not resume:
        raise SystemExit(f"{latest} exists. Pass --resume to continue it, or --fresh to discard and restart.")
    t = cfg.train
    seed_everything(cfg.seed)
    root = ensure_local_cache(cfg)
    w, m = cfg.features.window_frames, cfg.features.n_mels
    train = TrainData(load_split(root, "train"), w, m, device)
    val = TrainData(load_split(root, "dev_val"), w, m, device)
    model = build(variant, cfg, hparam).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    start, history, best = 0, [], math.inf
    if resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start, history, best = state["epoch"], state["history"], state["best_val"]
        print(f"resuming {rd.name} from epoch {start}/{t['epochs']}", flush=True)

    val_starts = val.grid_starts(t["val_hop_frames"])
    corruption = model.corruption_cfg(cfg.corruption) if model.corrupts_input else None
    if model.corrupts_input:
        val.recorrupt(corruption, torch.Generator(device=device).manual_seed(cfg.seed + VAL_SEED_OFFSET))
    info = {"variant": variant, "hparam": hparam, "hparams": model.hparams(), "param_count": param_count(model),
            "train_utts": len(train.index), "train_windows_available": len(train.valid_starts),
            "val_utts": len(val.index), "val_windows": len(val_starts),
            "model": cfg.model, "train": t, "features": cfg.features.__dict__,
            "corruption": corruption, **provenance(cfg)}
    rd.mkdir(parents=True, exist_ok=True)
    write_json(rd / "run_info.json", info)
    print(f"{variant} {model.hparams()} params={info['param_count']:,}", flush=True)

    steps = t["windows_per_epoch"] // t["batch_size"]
    for epoch in range(start, t["epochs"]):
        t0 = time.time()
        g = epoch_generator(cfg.seed, epoch, device)
        if model.corrupts_input:
            train.recorrupt(corruption, g)
        t_corrupt = time.time() - t0
        sums: dict[str, float] = {}
        for _ in range(steps):
            x_in, target = train.batch(train.sample_starts(t["batch_size"], g), corrupted=model.corrupts_input)
            for k, v in train_step(model, opt, x_in, target, t["grad_clip"]).items():
                sums[k] = sums.get(k, 0.0) + v
        row = {"epoch": epoch + 1, "train_loss": sums.pop("loss") / steps,
               **{f"train_{k}": v / steps for k, v in sums.items()},
               **validate(model, val, val_starts, cfg.eval["batch_windows"]),
               "corrupt_sec": round(t_corrupt, 2), "epoch_sec": round(time.time() - t0, 2)}
        history.append(row)
        improved = row["val_loss"] < best
        best = min(best, row["val_loss"])
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch + 1,
                 "history": history, "best_val": best, "variant": variant, "hparam": hparam,
                 "seed": cfg.seed, "dataset_revision": cfg.data["revision"]}
        ckpt = rd / f"ckpt_epoch{epoch + 1:02d}.pt"
        save_atomic(state, ckpt)
        shutil.copyfile(ckpt, rd / "latest.pt.tmp")
        (rd / "latest.pt.tmp").replace(latest)
        if improved:
            shutil.copyfile(ckpt, rd / "best.pt.tmp")
            (rd / "best.pt.tmp").replace(rd / "best.pt")
        for old in sorted(rd.glob("ckpt_epoch*.pt"))[:-t["keep_epoch_checkpoints"]]:
            old.unlink()
        _write_log(history, rd / "train_log.csv")
        plot_curves(history, rd / "curves.png", f"{variant} {model.hparams() or ''}".strip())
        print(f"[{variant}] epoch {epoch + 1}/{t['epochs']} train={row['train_loss']:.5f} "
              f"val={row['val_loss']:.5f} val_mse_clean={row['val_mse_clean']:.5f} "
              f"({row['epoch_sec']:.1f}s){' *' if improved else ''}", flush=True)

    info.update({"epochs_done": len(history), "best_val_loss": best,
                 "final_train_loss": history[-1]["train_loss"] if history else None,
                 "final_val_loss": history[-1]["val_loss"] if history else None,
                 "train_sec_total": round(sum(h["epoch_sec"] for h in history), 1),
                 "best_checkpoint": str(rd / "best.pt"),
                 "recon_sample": str(save_recon_sample(cfg, variant, hparam, rd, root, device))})
    write_json(rd / "run_info.json", info)
    return rd


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--hparam", type=float, default=None, help="one grid value (sparse: l1_weight, vae: beta, dae: min_snr_db)")
    ap.add_argument("--resume", action="store_true", help="continue from latest.pt if present")
    ap.add_argument("--fresh", action="store_true", help="delete existing checkpoints for this run first")
    ap.add_argument("--drive-root", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    device = require_gpu()
    cfg = load_ae_config(drive_root=args.drive_root)
    grid = cfg.hparam_grid(args.variant)
    if grid is None:
        if args.hparam is not None:
            raise SystemExit(f"{args.variant} has no tuned hyperparameter")
        values = [None]
    else:
        if args.hparam is not None and args.hparam not in grid[1]:
            raise SystemExit(f"{grid[0]}={args.hparam} is not in the config grid {grid[1]}")
        values = [args.hparam] if args.hparam is not None else grid[1]
    for v in values:
        if args.fresh:
            for p in run_dir(cfg, args.variant, v).glob("*.pt"):
                p.unlink()
        print(train_run(cfg, args.variant, v, args.resume or args.fresh, device), flush=True)


if __name__ == "__main__":
    main()
