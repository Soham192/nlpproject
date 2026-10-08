"""Layer 2, ASR front-end: does cleaning a COPY of Whisper's log-mel with the AE lower WER?

    python -m src.autoencoders.eval_asr --baseline           # Whisper-small on raw mels (once, cached)
    python -m src.autoencoders.eval_asr --variant dae

The cleaned mel exists only in memory on its way into whisper.decode. No audio is written or modified.
200 test-clean utterances, clean and at each SNR in config (noise + telephone band-limit + mu-law, the
same corruption types as training, fixed seed per utterance so every system hears identical audio).
WER uses src.evaluate.normalize_text and src.evaluate.wer, i.e. exactly the pipeline's normalisation.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..config import load_config
from ..evaluate import normalize_text, wer
from .common import (baseline_dir, candidate_runs, layer_dir, load_model, provenance, require_gpu,
                     seed_everything, write_json, write_jsonl)
from .config import VARIANTS, AEConfig, load_ae_config
from .data import HOP, SplitCache, ensure_local_cache, load_split, telephone_chain
from .degrade import to_pcm_grid
from .frontend import clean_whisper_mel
from .models import param_count

TEST_SEED, DEV_SEED = 21, 22


def conditions(cfg: AEConfig) -> list[str]:
    return ["clean"] + [f"snr{s}" for s in cfg.eval["asr"]["snrs_db"]]


def condition_audio(split: SplitCache, cond: str, c: dict, seed: int, n: int | None = None) -> list[np.ndarray]:
    """Deterministic per (utterance, condition): baseline and every variant decode identical audio."""
    out = []
    for i in range(len(split) if n is None else n):
        x = split.utt_audio(i)
        if cond != "clean":
            snr = float(cond.removeprefix("snr"))
            g = torch.Generator().manual_seed(seed * 1_000_003 + i * 1009 + int(snr * 10))
            x = to_pcm_grid(telephone_chain(torch.from_numpy(x), snr, c, g).numpy())
        out.append(x)
    return out


def whisper_mel(audio: np.ndarray, n_mels: int, device: str) -> tuple[torch.Tensor, int]:
    """(M, 3000) mel exactly as whisper's own decode path builds it, plus the count of real-audio frames."""
    import whisper

    padded = whisper.pad_or_trim(torch.from_numpy(audio))
    return whisper.log_mel_spectrogram(padded, n_mels=n_mels, device=device), min(len(audio), whisper.audio.N_SAMPLES) // HOP


def transcribe(wmodel, audios: list[np.ndarray], cfg: AEConfig, ae=None) -> list[str]:
    import whisper

    f = cfg.features
    device = str(next(wmodel.parameters()).device)
    opts = whisper.DecodingOptions(language="en", task="transcribe", temperature=0.0, without_timestamps=True,
                                   fp16=device.startswith("cuda"))
    texts: list[str] = []
    bs = cfg.eval["asr"]["batch_size"]
    for i in range(0, len(audios), bs):
        mels = []
        for a in audios[i:i + bs]:
            mel, n = whisper_mel(a, f.n_mels, device)
            if ae is not None:
                mel = clean_whisper_mel(ae, mel, n, f.window_frames, f.ola_hop_frames)
            mels.append(mel)
        texts += [r.text for r in whisper.decode(wmodel, torch.stack(mels), opts)]
    return texts


def item_rows(variant: str, split: SplitCache, hyps: dict[str, list[str]]) -> list[dict]:
    """scores.jsonl rows, one per (utterance, condition). `label` is the condition; `score` is that utterance's
    WER; `errors` / `ref_words` let corpus WER (and bootstrap CIs over utterances) be recomputed exactly."""
    n = load_config().evaluation.normalize
    rows = []
    for cond, hs in hyps.items():
        for r, h in zip(split.index, hs):
            ref, hyp = normalize_text(r["text"], n), normalize_text(h, n)
            w = wer(ref, hyp)
            errs = w["substitutions"] + w["deletions"] + w["insertions"]
            rows.append({"id": r["id"], "label": cond, "score": errs / w["ref_words"], "errors": errs,
                         "ref_words": w["ref_words"], "substitutions": w["substitutions"],
                         "deletions": w["deletions"], "insertions": w["insertions"],
                         "hyp": h, "variant": variant, "split": "test"})
    return rows


def corpus_wer(refs: list[str], hyps: list[str]) -> dict:
    """Sum of S+D+I over sum of reference words, after the pipeline's own normalisation."""
    n = load_config().evaluation.normalize
    tot = {"substitutions": 0, "deletions": 0, "insertions": 0, "ref_words": 0}
    for r, h in zip(refs, hyps):
        w = wer(normalize_text(r, n), normalize_text(h, n))
        for k in tot:
            tot[k] += w[k]
    errs = tot["substitutions"] + tot["deletions"] + tot["insertions"]
    return {"wer": round(errs / tot["ref_words"], 5), **tot, "utterances": len(refs)}


def eligible(split: SplitCache) -> SplitCache:
    """Drop utterances longer than Whisper's 30 s window (they would be truncated, inflating WER)."""
    keep = [r for r in split.index if r["n_samples"] <= 30 * 16000]
    return SplitCache(split.audio, split.mel, keep)


def evaluate_conditions(wmodel, split: SplitCache, conds: list[str], cfg: AEConfig, seed: int, ae=None,
                        n: int | None = None, log_tag: str = "") -> tuple[dict, dict]:
    refs = [r["text"] for r in split.index[:n]]
    res, hyps = {}, {}
    for cond in conds:
        t0 = time.time()
        hyps[cond] = transcribe(wmodel, condition_audio(split, cond, cfg.corruption, seed, n), cfg, ae)
        res[cond] = {**corpus_wer(refs, hyps[cond]), "sec": round(time.time() - t0, 1)}
        print(f"[asr{log_tag}] {cond}: WER={res[cond]['wer']:.4f} ({res[cond]['sec']}s)", flush=True)
    return res, hyps


def _summary(test: dict, cfg: AEConfig) -> dict:
    snr = [test[f"snr{s}"]["wer"] for s in cfg.eval["asr"]["snrs_db"]]
    return {"wer_clean": test["clean"]["wer"], "wer_noisy_mean": round(float(np.mean(snr)), 5)}


def _dump_hyps(path, split: SplitCache, hyps: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, r in enumerate(split.index[:len(next(iter(hyps.values())))]):
            f.write(json.dumps({"id": r["id"], "ref": r["text"], **{c: h[i] for c, h in hyps.items()}}) + "\n")


def _setup(cfg: AEConfig):
    import whisper

    device = require_gpu()
    seed_everything(cfg.seed)
    root = ensure_local_cache(cfg)
    wmodel = whisper.load_model(cfg.features.whisper_model, device=device)
    return eligible(load_split(root, "test_asr")), eligible(load_split(root, "dev_tune")), wmodel


def _dev_conds(cfg: AEConfig) -> list[str]:
    return ["clean", f"snr{cfg.eval['asr']['dev_select_snr_db']}"]


def run_baseline(cfg: AEConfig) -> dict:
    test, dev, wmodel = _setup(cfg)
    a = cfg.eval["asr"]
    dev_res, _ = evaluate_conditions(wmodel, dev, _dev_conds(cfg), cfg, cfg.seed + DEV_SEED, None,
                                     a["dev_select_utts"], "/baseline/dev")
    test_res, hyps = evaluate_conditions(wmodel, test, conditions(cfg), cfg, cfg.seed + TEST_SEED, None, None,
                                         "/baseline")
    _dump_hyps(baseline_dir(cfg, "asr") / "hyps.jsonl", test, hyps)
    write_jsonl(baseline_dir(cfg, "asr") / "scores.jsonl", item_rows("baseline_raw", test, hyps))
    metrics = {"layer": "asr", "variant": "baseline_raw", "main_metric": "wer_noisy_mean",
               **_summary(test_res, cfg), "test": test_res, "dev": dev_res, **provenance(cfg)}
    write_json(baseline_dir(cfg, "asr") / "metrics.json", metrics)
    return metrics


def run_variant(cfg: AEConfig, variant: str) -> dict:
    test, dev, wmodel = _setup(cfg)
    a = cfg.eval["asr"]
    cands = []
    runs = candidate_runs(cfg, variant)
    for hp, rd in runs:
        ae = load_model(cfg, variant, hp, rd / "best.pt", str(next(wmodel.parameters()).device))
        if len(runs) > 1:  # dev selection only matters when there is a choice
            dev_res, _ = evaluate_conditions(wmodel, dev, _dev_conds(cfg), cfg, cfg.seed + DEV_SEED, ae,
                                             a["dev_select_utts"], f"/{variant}/dev hparam={hp}")
            score = float(np.mean([r["wer"] for r in dev_res.values()]))
        else:
            dev_res, score = None, None
        cands.append({"hparam": hp, "run_dir": str(rd), "dev": dev_res, "dev_wer_mean": score, "_model": ae})
    best = min(cands, key=lambda d: d["dev_wer_mean"] if d["dev_wer_mean"] is not None else 0.0)
    test_res, hyps = evaluate_conditions(wmodel, test, conditions(cfg), cfg, cfg.seed + TEST_SEED, best["_model"],
                                         None, f"/{variant}")
    _dump_hyps(layer_dir(cfg, variant, "asr") / "hyps.jsonl", test, hyps)
    write_jsonl(layer_dir(cfg, variant, "asr") / "scores.jsonl", item_rows(variant, test, hyps))
    metrics = {"layer": "asr", "variant": variant, "main_metric": "wer_noisy_mean", **_summary(test_res, cfg),
               "test": test_res,
               "chosen": {"hparam": best["hparam"], "checkpoint": f"{best['run_dir']}/best.pt"},
               "candidates": [{k: v for k, v in d.items() if not k.startswith("_")} for d in cands],
               "dev_selection": {"utterances": a["dev_select_utts"], "conditions": _dev_conds(cfg)},
               "param_count": param_count(best["_model"]), **provenance(cfg)}
    write_json(layer_dir(cfg, variant, "asr") / "metrics.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--variant", choices=VARIANTS)
    g.add_argument("--baseline", action="store_true")
    ap.add_argument("--drive-root", default=None)
    args = ap.parse_args(argv)
    cfg = load_ae_config(drive_root=args.drive_root)
    with torch.no_grad():
        m = run_baseline(cfg) if args.baseline else run_variant(cfg, args.variant)
    print(f"[asr/{m['variant']}] clean WER={m['wer_clean']:.4f}  noisy mean WER={m['wer_noisy_mean']:.4f}")


if __name__ == "__main__":
    main()
