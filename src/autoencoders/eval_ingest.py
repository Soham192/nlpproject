"""Layer 1, ingest quality gate: does high reconstruction error flag input that is not usable speech?

    python -m src.autoencoders.eval_ingest --variant dense
    python -m src.autoencoders.eval_ingest --baseline        # RMS energy check (no GPU needed)

Negatives: clean speech. Positives: tone, digital silence, heavy clipping, DC offset, white noise,
heavy mu-law (degrade.py). Threshold (and lambda/beta for tuned variants) chosen on dev-clean only;
reported on test-clean. Writes <drive>/<variant>/ingest/metrics.json or <drive>/baselines/ingest/metrics.json.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .common import (auroc, baseline_dir, best_f1_threshold, candidate_runs, f1_at, layer_dir, load_model,
                     provenance, record_choice, require_gpu, seed_everything, write_json, write_jsonl)
from .config import VARIANTS, AEConfig, load_ae_config
from .data import SplitCache, ensure_local_cache, load_split, log_mel
from .degrade import INGEST_KINDS, make_unusable, rms_db
from .frontend import score_mel
from .models import param_count

DEV_SEED, TEST_SEED = 11, 12  # offsets from cfg.seed


def item_rows(variant: str, split_name: str, split: SplitCache, neg_scores, pos_scores, kinds: list[str]) -> list[dict]:
    """scores.jsonl rows: label 1 = not usable speech (positive), 0 = clean speech."""
    ids = [r["id"] for r in split.index]
    rows = [{"id": f"{u}#clean", "label": 0, "score": float(s), "kind": "clean_speech", "variant": variant,
             "split": split_name} for u, s in zip(ids, neg_scores)]
    rows += [{"id": f"{u}#{k}", "label": 1, "score": float(s), "kind": k, "variant": variant, "split": split_name}
             for u, s, k in zip(ids, pos_scores, kinds)]
    return rows


def build_examples(split: SplitCache, c: dict, seed: int) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """One negative per utterance, and one positive derived from it; positive kinds cycle evenly."""
    rng = np.random.default_rng(seed)
    neg = [split.utt_audio(i) for i in range(len(split))]
    kinds = [INGEST_KINDS[i % len(INGEST_KINDS)] for i in range(len(split))]
    pos = [make_unusable(k, a, rng, c) for k, a in zip(kinds, neg)]
    return neg, pos, kinds


def ae_file_scores(model, audios: list[np.ndarray], cfg: AEConfig, device: str) -> np.ndarray:
    f = cfg.features
    return np.array([float(score_mel(model, log_mel(a, f.n_mels, device), f.window_frames, f.ola_hop_frames,
                                     cfg.eval["batch_windows"]).mean()) for a in audios])


def summarize(dev_neg, dev_pos, test_neg, test_pos, kinds: list[str]) -> dict:
    thr = best_f1_threshold(dev_neg, dev_pos)
    test_pos, kinds = np.asarray(test_pos), np.asarray(kinds)
    return {
        "dev": {"auroc": round(auroc(dev_neg, dev_pos), 4), "f1_at_threshold": thr},
        "test": {"auroc": round(auroc(test_neg, test_pos), 4),
                 "f1_at_dev_threshold": f1_at(test_neg, test_pos, thr["threshold"]),
                 "auroc_by_kind": {k: round(auroc(test_neg, test_pos[kinds == k]), 4) for k in INGEST_KINDS},
                 "n_neg": len(test_neg), "n_pos": int(len(test_pos))},
        "threshold": thr["threshold"],
    }


def run_variant(cfg: AEConfig, variant: str) -> dict:
    device = require_gpu()
    seed_everything(cfg.seed)
    root = ensure_local_cache(cfg)
    c = cfg.eval["ingest"]
    dsplit, tsplit = load_split(root, "dev_tune"), load_split(root, "test_ingest")
    dneg, dpos, dkinds = build_examples(dsplit, c, cfg.seed + DEV_SEED)
    tneg, tpos, kinds = build_examples(tsplit, c, cfg.seed + TEST_SEED)

    cands = []
    for hp, rd in candidate_runs(cfg, variant):
        model = load_model(cfg, variant, hp, rd / "best.pt", device)
        sn, sp = ae_file_scores(model, dneg, cfg, device), ae_file_scores(model, dpos, cfg, device)
        cands.append({"hparam": hp, "run_dir": str(rd), "dev_auroc": round(auroc(sn, sp), 4),
                      "_scores": (sn, sp), "_model": model})
        print(f"[ingest/{variant}] hparam={hp} dev AUROC={cands[-1]['dev_auroc']}", flush=True)
    best = max(cands, key=lambda d: d["dev_auroc"])
    model = best["_model"]
    tsn, tsp = ae_file_scores(model, tneg, cfg, device), ae_file_scores(model, tpos, cfg, device)
    res = summarize(*best["_scores"], tsn, tsp, kinds)
    write_jsonl(layer_dir(cfg, variant, "ingest") / "scores.jsonl",
                item_rows(variant, "test", tsplit, tsn, tsp, kinds)
                + item_rows(variant, "dev", dsplit, *best["_scores"], dkinds))
    metrics = {"layer": "ingest", "variant": variant, "main_metric": "test.auroc", **res,
               "score": "negative ELBO per window, file mean" if variant == "vae"
               else "reconstruction MSE per window, file mean",
               "chosen": {"hparam": best["hparam"], "checkpoint": f"{best['run_dir']}/best.pt",
                          "threshold": res["threshold"]},
               "candidates": [{k: v for k, v in d.items() if not k.startswith("_")} for d in cands],
               "param_count": param_count(model), **provenance(cfg)}
    write_json(layer_dir(cfg, variant, "ingest") / "metrics.json", metrics)
    record_choice(cfg, variant, "ingest", best["hparam"])
    return metrics


def run_baseline(cfg: AEConfig) -> dict:
    root = ensure_local_cache(cfg)
    c = cfg.eval["ingest"]
    dsplit, tsplit = load_split(root, "dev_tune"), load_split(root, "test_ingest")
    dneg, dpos, dkinds = build_examples(dsplit, c, cfg.seed + DEV_SEED)
    tneg, tpos, kinds = build_examples(tsplit, c, cfg.seed + TEST_SEED)
    ref = float(np.median([rms_db(a) for a in dneg]))  # typical speech level, from dev only

    def score(audios):
        return np.array([abs(rms_db(a) - ref) for a in audios])

    dsn, dsp, tsn, tsp = score(dneg), score(dpos), score(tneg), score(tpos)
    res = summarize(dsn, dsp, tsn, tsp, kinds)
    write_jsonl(baseline_dir(cfg, "ingest") / "scores.jsonl",
                item_rows("baseline_rms", "test", tsplit, tsn, tsp, kinds)
                + item_rows("baseline_rms", "dev", dsplit, dsn, dsp, dkinds))
    metrics = {"layer": "ingest", "variant": "baseline_rms", "main_metric": "test.auroc", **res,
               "score": f"|RMS dBFS - dev median ({ref:.2f} dBFS)|", "dev_median_rms_db": ref,
               **provenance(cfg)}
    write_json(baseline_dir(cfg, "ingest") / "metrics.json", metrics)
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
    print(f"[ingest/{m['variant']}] test AUROC={m['test']['auroc']} "
          f"F1@dev-thr={m['test']['f1_at_dev_threshold']['f1']}  by kind={m['test']['auroc_by_kind']}")


if __name__ == "__main__":
    main()
