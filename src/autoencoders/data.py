"""Features, windowing, corruption and the LibriSpeech subset/cache.

Features are exactly what Whisper consumes: whisper.log_mel_spectrogram (log10, clamp to max-8,
(x+4)/4), computed per utterance. Model input is `window_frames` consecutive frames, flattened.

    python -m src.autoencoders.data --prepare     # one-time, writes the cache to Drive (idempotent)
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from .config import AEConfig, load_ae_config

log = logging.getLogger("redact.ae.data")

SR = 16000
HOP = 160  # Whisper mel hop (10 ms)

# Eval subsets carved from dev-clean / test-clean, in this order, by a seeded permutation.
DEV_SUBSETS = ("dev_val", "dev_tune")
TEST_SUBSETS = ("test_asr", "test_ingest", "test_leak")
CACHE_SPLITS = ("train",) + DEV_SUBSETS + TEST_SUBSETS
CORPUS_OF = {"train": "train", **{s: "dev" for s in DEV_SUBSETS}, **{s: "test" for s in TEST_SUBSETS}}


# --------------------------------------------------------------------------- features

def log_mel(audio: torch.Tensor | np.ndarray, n_mels: int = 80, device: str | torch.device | None = None) -> torch.Tensor:
    """(n_frames, n_mels) Whisper log-mel of float32 mono 16 kHz audio. n_frames = len(audio) // 160."""
    import whisper

    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    return whisper.log_mel_spectrogram(audio.float(), n_mels=n_mels, device=device).T.contiguous()


# --------------------------------------------------------------------------- windowing / overlap-add

def window_starts(n_frames: int, window: int, hop: int) -> list[int]:
    """Start frames covering [0, max(n_frames, window)); the last window is pinned to the end."""
    if n_frames <= window:
        return [0]
    starts = list(range(0, n_frames - window + 1, hop))
    if starts[-1] != n_frames - window:
        starts.append(n_frames - window)
    return starts


def frame_windows(mel: torch.Tensor, window: int, hop: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(T, M) -> (N, window*M) flattened windows, their start frames, and the padded length.

    Inputs shorter than `window` are edge-padded (last frame repeated) up to `window`.
    """
    t, m = mel.shape
    if t < window:
        mel = torch.cat([mel, mel[-1:].expand(window - t, m)], dim=0)
    starts = torch.tensor(window_starts(mel.shape[0], window, hop), device=mel.device)
    idx = starts[:, None] + torch.arange(window, device=mel.device)
    return mel[idx].reshape(len(starts), window * m), starts, mel.shape[0]


def overlap_add(windows: torch.Tensor, starts: torch.Tensor, padded_len: int, window: int, n_frames: int) -> torch.Tensor:
    """Inverse of frame_windows: average every window's contribution to each frame, crop to n_frames."""
    n = windows.shape[0]
    m = windows.shape[1] // window
    idx = (starts[:, None] + torch.arange(window, device=windows.device)).reshape(-1)
    acc = torch.zeros(padded_len, m, dtype=windows.dtype, device=windows.device)
    cnt = torch.zeros(padded_len, 1, dtype=windows.dtype, device=windows.device)
    acc.index_add_(0, idx, windows.reshape(n * window, m))
    cnt.index_add_(0, idx, torch.ones(n * window, 1, dtype=windows.dtype, device=windows.device))
    return (acc / cnt)[:n_frames]


# --------------------------------------------------------------------------- corruption (generated, no noise corpora)

def _rand(g: torch.Generator) -> float:
    return float(torch.rand(1, generator=g, device=g.device))


def add_noise_snr(x: torch.Tensor, snr_db: float, g: torch.Generator, color: str = "white") -> torch.Tensor:
    """x + noise scaled so 10*log10(P_x / P_noise) == snr_db. Silent input is returned unchanged."""
    noise = torch.randn(x.shape, generator=g, device=g.device, dtype=torch.float32).to(x.device)
    if color == "pink":
        spec = torch.fft.rfft(noise)
        f = torch.arange(spec.shape[-1], device=x.device, dtype=torch.float32)
        f[0] = 1.0
        noise = torch.fft.irfft(spec / torch.sqrt(f), n=x.shape[-1])
    elif color != "white":
        raise ValueError(f"unknown noise color {color!r}")
    px = torch.mean(x.float() ** 2)
    if px <= 0:
        return x.clone()
    pn = torch.mean(noise ** 2)
    scale = torch.sqrt(px / (pn * 10 ** (snr_db / 10)))
    return (x.float() + scale * noise).to(x.dtype)


def telephone_bandlimit(x: torch.Tensor, lo_hz: float = 300.0, hi_hz: float = 3400.0, sr: int = SR) -> torch.Tensor:
    """Zero every FFT bin outside [lo_hz, hi_hz]."""
    spec = torch.fft.rfft(x.float())
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / sr).to(x.device)
    spec = spec * ((freqs >= lo_hz) & (freqs <= hi_hz))
    return torch.fft.irfft(spec, n=x.shape[-1]).to(x.dtype)


def mulaw_roundtrip(x: torch.Tensor, channels: int = 256) -> torch.Tensor:
    """mu-law encode to `channels` levels and decode back (G.711 uses 256)."""
    mu = channels - 1
    y = torch.sign(x) * torch.log1p(mu * torch.abs(x.clamp(-1, 1))) / math.log1p(mu)
    q = torch.round((y + 1) / 2 * mu)
    y = q / mu * 2 - 1
    return (torch.sign(y) * ((1 + mu) ** torch.abs(y) - 1) / mu).to(x.dtype)


def corrupt(x: torch.Tensor, c: dict, g: torch.Generator) -> torch.Tensor:
    """Training corruption: noise at SNR ~ U(snr_db) (white/pink), then maybe band-limit, then maybe mu-law."""
    lo, hi = c["snr_db"]
    snr = lo + (hi - lo) * _rand(g)
    y = add_noise_snr(x, snr, g, "pink" if _rand(g) < c["pink_noise_prob"] else "white")
    if _rand(g) < c["bandlimit_prob"]:
        y = telephone_bandlimit(y, *c["bandlimit_hz"])
    if _rand(g) < c["mulaw_prob"]:
        y = mulaw_roundtrip(y, c["mulaw_channels"])
    return y.clamp(-1, 1)


def telephone_chain(x: torch.Tensor, snr_db: float, c: dict, g: torch.Generator) -> torch.Tensor:
    """Eval corruption at a fixed SNR: noise + band-limit + mu-law, all applied (the full training chain)."""
    y = add_noise_snr(x, snr_db, g, "pink" if _rand(g) < c["pink_noise_prob"] else "white")
    y = telephone_bandlimit(y, *c["bandlimit_hz"])
    return mulaw_roundtrip(y, c["mulaw_channels"]).clamp(-1, 1)


# --------------------------------------------------------------------------- subset selection (pure, unit-tested)

def select_eval_subsets(rows: list[dict], sizes: dict[str, int], seed: int) -> dict[str, list[dict]]:
    """Disjoint subsets of `rows` (dicts with "id"), sized per `sizes`, in dict order, by a seeded permutation.

    Rows are sorted by id first, so the result does not depend on stream order.
    """
    rows = sorted(rows, key=lambda r: r["id"])
    need = sum(sizes.values())
    if need > len(rows):
        raise ValueError(f"need {need} utterances, split only has {len(rows)}")
    perm = np.random.default_rng(seed).permutation(len(rows))
    out, i = {}, 0
    for name, n in sizes.items():
        out[name] = [rows[j] for j in perm[i:i + n]]
        i += n
    return out


def select_train_balanced(rows: Iterable[dict], duration_sec: Callable[[dict], float], total_sec: float,
                          n_speakers: int, max_utt_sec: float) -> list[dict]:
    """Speaker-balanced subset: each speaker contributes utterances (in stream order) until it reaches
    total_sec / n_speakers. Rows of speakers already at quota are skipped without decoding."""
    quota = total_sec / n_speakers
    have: dict[int, float] = defaultdict(float)
    full: set[int] = set()
    picked = []
    for r in rows:
        spk = r["speaker_id"]
        if spk in full:
            continue
        d = duration_sec(r)
        if d > max_utt_sec:
            continue
        picked.append({**r, "duration_sec": d})
        have[spk] += d
        if have[spk] >= quota:
            full.add(spk)
            if len(full) >= n_speakers:
                break
    return picked


def assert_disjoint(splits: dict[str, list[dict]], corpus_of: dict[str, str] = CORPUS_OF) -> None:
    """No utterance id appears in two splits; splits from different corpora share no speaker."""
    seen: dict[str, str] = {}
    for name, rows in splits.items():
        for r in rows:
            if r["id"] in seen:
                raise AssertionError(f"utterance {r['id']} in both {seen[r['id']]} and {name}")
            seen[r["id"]] = name
    spk = {name: {r["speaker_id"] for r in rows} for name, rows in splits.items()}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if corpus_of[a] != corpus_of[b] and spk[a] & spk[b]:
                raise AssertionError(f"speakers shared by {a} and {b}: {sorted(spk[a] & spk[b])[:5]}")


# --------------------------------------------------------------------------- cache

@dataclass
class SplitCache:
    """One cached split: concatenated int16 audio, fp16 mels (T, M), and per-utterance offsets."""
    audio: np.ndarray
    mel: np.ndarray
    index: list[dict]

    def utt_audio(self, i: int) -> np.ndarray:
        r = self.index[i]
        return self.audio[r["sample_offset"]:r["sample_offset"] + r["n_samples"]].astype(np.float32) / 32768.0

    def utt_mel(self, i: int) -> np.ndarray:
        r = self.index[i]
        return self.mel[r["frame_offset"]:r["frame_offset"] + r["n_frames"]].astype(np.float32)

    def __len__(self) -> int:
        return len(self.index)


def cache_dir(root: Path) -> Path:
    return Path(root) / "cache"


def load_split(root: Path, name: str, mmap: bool = True) -> SplitCache:
    d = cache_dir(root) / name
    mode = "r" if mmap else None
    return SplitCache(np.load(d / "audio.npy", mmap_mode=mode), np.load(d / "mel.npy", mmap_mode=mode),
                      json.loads((d / "index.json").read_text()))


def ensure_local_cache(cfg: AEConfig) -> Path:
    """Copy the Drive cache to local disk once per session (Drive reads are slow). Returns the root to read from."""
    src, dst = cache_dir(cfg.drive_root), cache_dir(cfg.local_cache)
    if not (src / "prep_info.json").exists():
        raise FileNotFoundError(f"no feature cache at {src}; run `python -m src.autoencoders.data --prepare` first")
    if not (dst / "prep_info.json").exists() or (dst / "prep_info.json").read_text() != (src / "prep_info.json").read_text():
        log.info("copying feature cache %s -> %s", src, dst)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    return cfg.local_cache


def _write_split(d: Path, rows: list[dict], decode: Callable[[dict], np.ndarray], n_mels: int, device: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    audios, mels, index = [], [], []
    so = fo = 0
    for r in rows:
        a = decode(r)
        pcm = np.clip(np.round(a * 32767), -32768, 32767).astype(np.int16)
        # Features from the int16-quantised audio, i.e. exactly what a 16-bit WAV would give.
        m = log_mel(pcm.astype(np.float32) / 32768.0, n_mels, device).cpu().numpy().astype(np.float16)
        audios.append(pcm)
        mels.append(m)
        index.append({"id": r["id"], "speaker_id": int(r["speaker_id"]), "chapter_id": int(r["chapter_id"]),
                      "text": r["text"], "sample_offset": so, "n_samples": len(pcm),
                      "frame_offset": fo, "n_frames": len(m)})
        so += len(pcm)
        fo += len(m)
    np.save(d / "audio.npy", np.concatenate(audios))
    np.save(d / "mel.npy", np.concatenate(mels))
    (d / "index.json").write_text(json.dumps(index))


def _decode_flac(r: dict) -> np.ndarray:
    import soundfile as sf

    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    if sr != SR:
        raise ValueError(f"{r['id']}: sample rate {sr}, expected {SR}")
    return a if a.ndim == 1 else a.mean(axis=1)


def _flac_seconds(r: dict) -> float:
    import soundfile as sf

    info = sf.info(io.BytesIO(r["audio"]["bytes"]))
    return info.frames / info.samplerate


def _stream(cfg: AEConfig, split: str):
    from datasets import Audio, load_dataset

    d = cfg.data
    ds = load_dataset(d["dataset"], d["config"], split=split, revision=d["revision"], streaming=True)
    return ds.cast_column("audio", Audio(decode=False))  # skipped rows are never decoded


def prepare(cfg: AEConfig, force: bool = False) -> Path:
    """Select the subsets, decode, compute Whisper log-mels and write the cache to Drive. Idempotent."""
    out = cache_dir(cfg.drive_root)
    d = cfg.data
    info = {"dataset": d["dataset"], "config": d["config"], "revision": d["revision"], "seed": cfg.seed,
            "train_hours": d["train_hours"], "whisper_model": cfg.features.whisper_model,
            "n_mels": cfg.features.n_mels,
            "sizes": {k: d[f"{k}_utts"] for k in DEV_SUBSETS + TEST_SUBSETS}}
    marker = out / "prep_info.json"
    if marker.exists() and not force:
        old = json.loads(marker.read_text())
        if {k: old.get(k) for k in info} == info:
            log.info("feature cache already prepared at %s", out)
            return out
        raise RuntimeError(f"{marker} was built with different settings; rerun with --force to rebuild")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    splits: dict[str, list[dict]] = {}
    for split, names in ((d["dev_split"], DEV_SUBSETS), (d["test_split"], TEST_SUBSETS)):
        rows = [{k: r[k] for k in ("id", "speaker_id", "chapter_id", "text", "audio")} for r in _stream(cfg, split)]
        n_all = len(rows)
        rows = [r for r in rows if _flac_seconds(r) <= d["max_utt_sec"]]  # Whisper would truncate longer ones
        log.info("%s: dropped %d utterances longer than %ss", split, n_all - len(rows), d["max_utt_sec"])
        splits.update(select_eval_subsets(rows, {n: d[f"{n}_utts"] for n in names}, cfg.seed))
        log.info("%s: %d utterances, selected %s", split, len(rows), {n: len(splits[n]) for n in names})

    decoded: dict[str, np.ndarray] = {}

    def duration(r: dict) -> float:
        decoded[r["id"]] = _decode_flac(r)
        return len(decoded[r["id"]]) / SR

    train_rows = ({k: r[k] for k in ("id", "speaker_id", "chapter_id", "text", "audio")}
                  for r in _stream(cfg, d["train_split"]))
    splits["train"] = select_train_balanced(train_rows, duration, d["train_hours"] * 3600, d["train_speakers"],
                                            d["max_utt_sec"])
    hours = sum(r["duration_sec"] for r in splits["train"]) / 3600
    log.info("train: %d utterances, %.2f h, %d speakers", len(splits["train"]), hours,
             len({r["speaker_id"] for r in splits["train"]}))
    assert_disjoint(splits)

    for name in CACHE_SPLITS:
        rows = splits[name]
        dec = (lambda r: decoded.pop(r["id"])) if name == "train" else _decode_flac
        _write_split(out / name, rows, dec, cfg.features.n_mels, device)
        log.info("wrote %s (%d utts)", name, len(rows))

    (cfg.drive_root / "splits.json").write_text(json.dumps(
        {n: [{"id": r["id"], "speaker_id": int(r["speaker_id"])} for r in splits[n]] for n in CACHE_SPLITS}, indent=1))
    info.update({"train_hours_actual": round(hours, 3), "prep_sec": round(time.time() - t0, 1)})
    marker.write_text(json.dumps(info, indent=2))
    return out


# --------------------------------------------------------------------------- GPU training data

class TrainData:
    """A cached split held on the training device, with random-window sampling and on-the-fly corruption."""

    def __init__(self, split: SplitCache, window: int, n_mels: int, device: str) -> None:
        self.window, self.n_mels, self.device = window, n_mels, device
        self.index = split.index
        self.mel = torch.from_numpy(np.asarray(split.mel)).to(device)          # (T, M) fp16
        self._audio_np = split.audio                                           # int16; to device only for the DAE
        self.audio: torch.Tensor | None = None
        starts = [np.arange(r["frame_offset"], r["frame_offset"] + r["n_frames"] - window + 1)
                  for r in self.index if r["n_frames"] >= window]
        self.valid_starts = torch.from_numpy(np.concatenate(starts)).to(device)
        self.offsets = torch.arange(window, device=device)
        self.corrupted: torch.Tensor | None = None

    def recorrupt(self, c: dict, g: torch.Generator) -> None:
        """Rebuild the corrupted-mel buffer: corrupt each utterance's waveform, recompute its log-mel per
        utterance (so Whisper's max-relative clamp matches the clean features)."""
        if self.audio is None:
            self.audio = torch.from_numpy(np.asarray(self._audio_np)).to(self.device)
        buf = torch.empty_like(self.mel)
        for r in self.index:
            x = self.audio[r["sample_offset"]:r["sample_offset"] + r["n_samples"]].float() / 32768.0
            y = corrupt(x, c, g)
            y = torch.round(y * 32767) / 32768.0  # stays a valid 16-bit signal
            m = log_mel(y, self.n_mels, self.device)
            buf[r["frame_offset"]:r["frame_offset"] + r["n_frames"]] = m[:r["n_frames"]].to(buf.dtype)
        self.corrupted = buf

    def batch(self, starts: torch.Tensor, corrupted: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """(input, target) flattened windows, float32."""
        idx = starts[:, None] + self.offsets
        target = self.mel[idx].reshape(len(starts), -1).float()
        if not corrupted:
            return target, target
        return self.corrupted[idx].reshape(len(starts), -1).float(), target

    def sample_starts(self, n: int, g: torch.Generator) -> torch.Tensor:
        pick = torch.randint(len(self.valid_starts), (n,), generator=g, device=g.device).to(self.device)
        return self.valid_starts[pick]

    def grid_starts(self, hop: int) -> torch.Tensor:
        """Deterministic windows at `hop` (validation)."""
        return torch.cat([torch.arange(r["frame_offset"], r["frame_offset"] + r["n_frames"] - self.window + 1, hop,
                                       device=self.device) for r in self.index if r["n_frames"] >= self.window])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true", help="build the LibriSpeech feature cache on Drive")
    ap.add_argument("--force", action="store_true", help="rebuild even if a cache exists")
    ap.add_argument("--drive-root", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_ae_config(drive_root=args.drive_root)
    if args.prepare:
        print(prepare(cfg, force=args.force))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
