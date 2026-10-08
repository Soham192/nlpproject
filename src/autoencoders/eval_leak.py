"""Layer 5, verification leak check: does a masked span still hold a speech fragment at its edge?

    python -m src.autoencoders.eval_leak --baseline          # max-frame RMS energy (no GPU needed)
    python -m src.autoencoders.eval_leak --variant dense

Spans come from real test-clean speech (degrade.place_span): the span's edge sits in active speech, and
the masked version keeps 0 ms (negative), 25 ms or 50 ms (positives) of that speech at one edge, the rest
being the fill (silence / tone / low-level noise). The mel is computed on the span's samples alone.

AE scores, per 16-frame window at hop 1:
  calibrated (main)  max |err - mu_fill| / sigma_fill, mu/sigma from pure-fill dev spans
  literal            max (-err)  ("low error = looks like speech", as originally specified)
Results are reported per fill type and per fragment length. Calibration, thresholds and lambda/beta
selection use dev-clean only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from .common import (auroc, baseline_dir, best_f1_threshold, candidate_runs, layer_dir, load_model, provenance,
                     require_gpu, seed_everything, write_json, write_jsonl)
from .config import VARIANTS, AEConfig, load_ae_config
from .data import SplitCache, ensure_local_cache, load_split
from .degrade import build_span, fill_signal, max_frame_rms_db, place_span
from .frontend import leak_scores, span_scores
from .models import param_count

TEST_SEED, DEV_SEED = 51, 52
MAIN_FILLS = ("tone", "noise")  # silence is reported separately: the energy baseline wins it trivially


@dataclass
class Example:
    utt: str
    fill: str
    frag_ms: int
    edge: str
    audio: np.ndarray


def build_examples(split: SplitCache, c: dict, seed: int, n: int | None = None) -> tuple[list[Example], int]:
    """Paired design: per utterance one span position; per fill one fill signal; frag 0 / 25 / 50 ms."""
    rng = np.random.default_rng(seed)
    frags = [0] + list(c["fragment_ms"])
    out, skipped = [], 0
    for i in range(len(split) if n is None else min(n, len(split))):
        audio = split.utt_audio(i)
        place = place_span(audio, rng, tuple(c["span_ms"]), max(c["fragment_ms"]), c["vad_rel_db"])
        if place is None:
            skipped += 1
            continue
        s, e, edge = place
        for fill in c["fills"]:
            sig = fill_signal(fill, e - s, rng, c)
            for f in frags:
                out.append(Example(split.index[i]["id"], fill, f, edge, build_span(audio, s, e, edge, f, sig)))
    return out, skipped


def item_rows(variant: str, split_name: str, ex: list[Example], scores: dict[str, np.ndarray]) -> list[dict]:
    """scores.jsonl rows: label 1 = speech fragment left in the span (frag_ms > 0), 0 = clean fill.
    `score` is the main score (calibrated for AEs, energy for the baseline); other scores ride along."""
    main = "calibrated" if "calibrated" in scores else "energy"
    return [{"id": f"{e.utt}#{e.fill}#{e.frag_ms}ms", "label": int(e.frag_ms > 0), "score": float(scores[main][i]),
             **{f"score_{k}": float(v[i]) for k, v in scores.items() if k != main},
             "fill": e.fill, "frag_ms": e.frag_ms, "edge": e.edge, "variant": variant, "split": split_name}
            for i, e in enumerate(ex)]


def auroc_table(ex: list[Example], scores: np.ndarray, c: dict) -> dict:
    """{fill: {"25": auroc, "50": auroc, "all": auroc}} — negatives are frag 0 of the same fill."""
    fills = np.array([e.fill for e in ex])
    frags = np.array([e.frag_ms for e in ex])
    tab = {}
    for fill in c["fills"]:
        neg = scores[(fills == fill) & (frags == 0)]
        row = {str(f): round(auroc(neg, scores[(fills == fill) & (frags == f)]), 4) for f in c["fragment_ms"]}
        row["all"] = round(auroc(neg, scores[(fills == fill) & (frags > 0)]), 4)
        tab[fill] = row
    return tab


def main_metric(tab: dict) -> float:
    return round(float(np.mean([tab[f][k] for f in MAIN_FILLS if f in tab for k in tab[f] if k != "all"])), 4)


def thresholds(ex: list[Example], scores: np.ndarray, c: dict) -> dict:
    fills = np.array([e.fill for e in ex])
    frags = np.array([e.frag_ms for e in ex])
    return {fill: best_f1_threshold(scores[(fills == fill) & (frags == 0)], scores[(fills == fill) & (frags > 0)])
            for fill in c["fills"]}


def window_errors(model, ex: list[Example], cfg: AEConfig, device) -> list[np.ndarray]:
    return [span_scores(model, e.audio, cfg.features.window_frames, device) for e in ex]


def calibrate(ex: list[Example], errs: list[np.ndarray], c: dict) -> dict:
    """Per fill: mean/std of window errors over pure-fill (frag 0) dev spans.

    Digital silence gives (near-)identical windows, so std ~ 0; sigma is floored at 1% of mu to keep
    z-scores readable. The floor is monotone, so AUROC is unaffected.
    """
    out = {}
    for fill in c["fills"]:
        w = np.concatenate([er for e, er in zip(ex, errs) if e.fill == fill and e.frag_ms == 0])
        sigma = max(float(w.std()), 0.01 * abs(float(w.mean())), 1e-8)
        out[fill] = {"mu": float(w.mean()), "sigma": sigma, "std_raw": float(w.std()), "n_windows": int(len(w))}
    return out


def span_level(ex: list[Example], errs: list[np.ndarray], calib: dict) -> dict[str, np.ndarray]:
    s = [leak_scores(er, calib[e.fill]["mu"], calib[e.fill]["sigma"]) for e, er in zip(ex, errs)]
    return {k: np.array([d[k] for d in s]) for k in ("calibrated", "literal")}


def _load_sets(cfg: AEConfig):
    root = ensure_local_cache(cfg)
    c = cfg.eval["leak"]
    dev, dskip = build_examples(load_split(root, "dev_tune"), c, cfg.seed + DEV_SEED, c["dev_spans"])
    test, tskip = build_examples(load_split(root, "test_leak"), c, cfg.seed + TEST_SEED)
    counts = {"dev_spans": len(dev) // (len(c["fills"]) * (1 + len(c["fragment_ms"]))), "dev_skipped_utts": dskip,
              "test_spans": len(test) // (len(c["fills"]) * (1 + len(c["fragment_ms"]))), "test_skipped_utts": tskip}
    return c, dev, test, counts


def run_variant(cfg: AEConfig, variant: str) -> dict:
    device = require_gpu()
    seed_everything(cfg.seed)
    c, dev, test, counts = _load_sets(cfg)
    cands = []
    for hp, rd in candidate_runs(cfg, variant):
        model = load_model(cfg, variant, hp, rd / "best.pt", device)
        derr = window_errors(model, dev, cfg, device)
        calib = calibrate(dev, derr, c)
        dsc = span_level(dev, derr, calib)
        dtab = auroc_table(dev, dsc["calibrated"], c)
        thr = thresholds(dev, dsc["calibrated"], c)
        for fill in calib:
            calib[fill]["threshold"] = thr[fill]["threshold"]
        cands.append({"hparam": hp, "run_dir": str(rd), "dev_main": main_metric(dtab), "dev_auroc": dtab,
                      "_model": model, "_calib": calib, "_dev_scores": dsc})
        print(f"[leak/{variant}] hparam={hp} dev mean calibrated AUROC (tone+noise)={cands[-1]['dev_main']}",
              flush=True)
    best = max(cands, key=lambda d: d["dev_main"])
    terr = window_errors(best["_model"], test, cfg, device)
    tsc = span_level(test, terr, best["_calib"])
    cal_tab, lit_tab = auroc_table(test, tsc["calibrated"], c), auroc_table(test, tsc["literal"], c)
    write_jsonl(layer_dir(cfg, variant, "leak") / "scores.jsonl",
                item_rows(variant, "test", test, tsc) + item_rows(variant, "dev", dev, best["_dev_scores"]))
    metrics = {"layer": "leak", "variant": variant, "main_metric": "main_auroc",
               "main_auroc": main_metric(cal_tab), "main_auroc_literal": main_metric(lit_tab),
               "silence_auroc": cal_tab.get("silence", {}).get("all"),
               "test": {"calibrated": cal_tab, "literal": lit_tab},
               "chosen": {"hparam": best["hparam"], "checkpoint": f"{best['run_dir']}/best.pt",
                          "calibration": best["_calib"]},
               "candidates": [{k: v for k, v in d.items() if not k.startswith("_")} for d in cands],
               **counts, "param_count": param_count(best["_model"]), **provenance(cfg)}
    write_json(layer_dir(cfg, variant, "leak") / "metrics.json", metrics)
    return metrics


def run_baseline(cfg: AEConfig) -> dict:
    c, dev, test, counts = _load_sets(cfg)
    dsc = np.array([max_frame_rms_db(e.audio) for e in dev])
    tsc = np.array([max_frame_rms_db(e.audio) for e in test])
    tab = auroc_table(test, tsc, c)
    write_jsonl(baseline_dir(cfg, "leak") / "scores.jsonl",
                item_rows("baseline_rms", "test", test, {"energy": tsc})
                + item_rows("baseline_rms", "dev", dev, {"energy": dsc}))
    metrics = {"layer": "leak", "variant": "baseline_rms", "main_metric": "main_auroc",
               "main_auroc": main_metric(tab), "silence_auroc": tab.get("silence", {}).get("all"),
               "test": {"calibrated": tab}, "score": "max 10 ms-frame RMS dBFS inside the span",
               "dev_thresholds": {f: t["threshold"] for f, t in thresholds(dev, dsc, c).items()},
               **counts, **provenance(cfg)}
    write_json(baseline_dir(cfg, "leak") / "metrics.json", metrics)
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
    print(f"[leak/{m['variant']}] mean AUROC tone+noise={m['main_auroc']}  silence={m['silence_auroc']}")
    for fill, row in m["test"]["calibrated"].items():
        print(f"   {fill:8s} {row}")


if __name__ == "__main__":
    main()
