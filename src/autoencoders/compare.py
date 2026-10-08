"""Collect every metrics.json from Drive into results/autoencoder_comparison.md.

    python -m src.autoencoders.compare [--out results/autoencoder_comparison.md]

Missing results show as "—", so this can be run after any subset of sessions.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .common import read_json
from .config import LAYERS, AEConfig, load_ae_config

ROOT = Path(__file__).resolve().parents[2]
DASH = "—"


def _load(path: Path) -> dict | None:
    return read_json(path) if path.exists() else None


def collect(cfg: AEConfig) -> dict:
    r = cfg.drive_root
    res = {v: {layer: _load(r / v / layer / "metrics.json") for layer in (*LAYERS, "latency")} for v in cfg.variants}
    res["baseline"] = {layer: _load(r / "baselines" / layer / "metrics.json") for layer in LAYERS}
    return res


def _f(x, nd=4, pct=False) -> str:
    if x is None:
        return DASH
    return f"{100 * x:.2f}%" if pct else f"{x:.{nd}f}"


def _get(d: dict | None, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def winners(res: dict, cfg: AEConfig) -> dict[str, str | None]:
    """Best variant per layer (baseline excluded; whether it beats the baseline is reported alongside).
    Layer 2: a variant that raises clean WER beyond the tolerance cannot win."""
    vs = cfg.variants
    tol = cfg.eval["asr"]["clean_wer_tolerance_pp"] / 100
    base_clean = _get(res["baseline"]["asr"], "wer_clean")
    out = {}
    l1 = {v: _get(res[v]["ingest"], "test", "auroc") for v in vs}
    l1 = {v: x for v, x in l1.items() if x is not None}
    out["ingest"] = max(l1, key=l1.get) if l1 else None
    l2 = {v: _get(res[v]["asr"], "wer_noisy_mean") for v in vs
          if _get(res[v]["asr"], "wer_noisy_mean") is not None and not clean_fail(res[v]["asr"], base_clean, tol)}
    out["asr"] = min(l2, key=l2.get) if l2 else None
    l5 = {v: _get(res[v]["leak"], "main_auroc") for v in vs}
    l5 = {v: x for v, x in l5.items() if x is not None}
    out["leak"] = max(l5, key=l5.get) if l5 else None
    return out


def clean_fail(asr: dict | None, base_clean: float | None, tol: float) -> bool:
    return bool(asr and base_clean is not None and asr["wer_clean"] - base_clean > tol)


def render(res: dict, cfg: AEConfig, curves: dict[str, list[str]]) -> str:
    vs = cfg.variants
    win = winners(res, cfg)
    tol = cfg.eval["asr"]["clean_wer_tolerance_pp"] / 100
    base = res["baseline"]
    base_clean = _get(base["asr"], "wer_clean")

    def mark(v, layer, text):
        return f"**{text}** 🏆" if win.get(layer) == v and text != DASH else text

    L = ["# Autoencoder comparison: 4 variants × 3 pipeline layers", "",
         f"Seed {cfg.seed} · {cfg.data['dataset']} ({cfg.data['config']}) @ `{cfg.data['revision'][:12]}` · "
         f"Whisper `{cfg.features.whisper_model}` · window {cfg.features.window_frames} frames × "
         f"{cfg.features.n_mels} mels · all tuning on dev-clean, all numbers below on test-clean.", "",
         "## Main table", "",
         "| Variant | L1 ingest gate: AUROC ↑ | L2 ASR front-end: mean WER at SNR "
         f"{'/'.join(map(str, cfg.eval['asr']['snrs_db']))} dB ↓ (clean WER) | L5 leak check: AUROC tone+noise ↑ "
         "(silence) |", "|---|---|---|---|"]
    for v in vs:
        r = res[v]
        a = r["asr"]
        l2 = DASH if a is None else (f"{mark(v, 'asr', _f(a['wer_noisy_mean'], pct=True))} "
                                     f"({_f(a['wer_clean'], pct=True)}"
                                     f"{' ⚠ clean FAIL' if clean_fail(a, base_clean, tol) else ''})")
        lk = r["leak"]
        l5 = DASH if lk is None else f"{mark(v, 'leak', _f(lk['main_auroc']))} ({_f(lk.get('silence_auroc'))})"
        L.append(f"| {v} | {mark(v, 'ingest', _f(_get(r['ingest'], 'test', 'auroc')))} | {l2} | {l5} |")
    b2 = DASH if base["asr"] is None else f"{_f(base['asr']['wer_noisy_mean'], pct=True)} ({_f(base_clean, pct=True)})"
    b5 = DASH if base["leak"] is None else f"{_f(base['leak']['main_auroc'])} ({_f(base['leak'].get('silence_auroc'))})"
    L.append(f"| *baseline* (L1 RMS check · L2 raw Whisper · L5 RMS energy) | "
             f"{_f(_get(base['ingest'], 'test', 'auroc'))} | {b2} | {b5} |")
    L += ["", "**Winner per layer** (best variant; compared with the non-learned baseline):", ""]
    for layer, name, key, higher in (("ingest", "L1 ingest gate", ("test", "auroc"), True),
                                     ("asr", "L2 ASR front-end", ("wer_noisy_mean",), False),
                                     ("leak", "L5 leak check", ("main_auroc",), True)):
        w = win.get(layer)
        if w is None:
            L.append(f"- {name}: {DASH} (no results yet)")
            continue
        wv, bv = _get(res[w][layer], *key), _get(base[layer], *key)
        if bv is None:
            verdict = "baseline not run yet"
        else:
            beats = wv > bv if higher else wv < bv
            verdict = "beats the baseline" if beats else "**does not beat the baseline**"
        L.append(f"- {name}: **{w}** ({_f(wv, pct=layer == 'asr')} vs baseline {_f(bv, pct=layer == 'asr')}; {verdict})")
    L += ["", f"L2 rule: a variant whose clean WER is more than {cfg.eval['asr']['clean_wer_tolerance_pp']} pp above "
          "the raw-Whisper clean WER is marked ⚠ and cannot win, however much it helps on noisy audio.",
          "L5 main metric averages tone and noise fills over both fragment lengths. Silence is shown separately "
          "because an energy threshold solves it trivially.", ""]

    # L1 details
    L += ["## Layer 1: ingest gate details", "",
          "| System | AUROC | F1 @ dev threshold | " + " | ".join(
              _get(res[vs[0]]["ingest"], "test", "auroc_by_kind") or _get(base["ingest"], "test", "auroc_by_kind")
              or {}) + " |"]
    kinds = list((_get(res[vs[0]]["ingest"], "test", "auroc_by_kind") or _get(base["ingest"], "test", "auroc_by_kind")
                  or {}).keys())
    L.append("|---|---|---|" + "---|" * len(kinds))
    for name, m in [*((v, res[v]["ingest"]) for v in vs), ("baseline (RMS)", base["ingest"])]:
        if m is None:
            L.append(f"| {name} | {DASH} | {DASH} |" + f" {DASH} |" * len(kinds))
            continue
        bk = m["test"]["auroc_by_kind"]
        L.append(f"| {name} | {_f(m['test']['auroc'])} | {_f(m['test']['f1_at_dev_threshold']['f1'])} | "
                 + " | ".join(_f(bk.get(k)) for k in kinds) + " |")
    L.append("")

    # L2 per SNR
    conds = ["clean"] + [f"snr{s}" for s in cfg.eval["asr"]["snrs_db"]]
    L += ["## Layer 2: WER per condition", "",
          "Noisy conditions = noise at the given SNR + telephone band-limit (300–3400 Hz) + μ-law round trip. "
          "WER uses `src/evaluate.py`'s normalisation (no numeral/spelling normalisation), so absolute values sit "
          "above Whisper's published LibriSpeech numbers; the same applies to every row.", "",
          "| System | " + " | ".join("clean" if c == "clean" else f"{c[3:]} dB" for c in conds) + " |",
          "|---|" + "---|" * len(conds)]
    for name, m in [("baseline (raw Whisper)", base["asr"]), *((v, res[v]["asr"]) for v in vs)]:
        cells = [DASH if m is None else _f(_get(m, "test", c, "wer"), pct=True) for c in conds]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")

    # L5 per fill
    lc = cfg.eval["leak"]
    cols = [(fill, str(f)) for fill in lc["fills"] for f in lc["fragment_ms"]]
    L += ["## Layer 5: AUROC per fill type and fragment length", "",
          "Negatives: fully filled spans. Positives: the same span with 25 or 50 ms of real speech kept at one edge.",
          "", "| System | score | " + " | ".join(f"{fill} {f} ms" for fill, f in cols) + " |",
          "|---|---|" + "---|" * len(cols)]
    rows = [("baseline (max RMS)", "energy", base["leak"], "calibrated")]
    for v in vs:
        rows += [(v, "calibrated", res[v]["leak"], "calibrated"), (v, "literal", res[v]["leak"], "literal")]
    for name, label, m, key in rows:
        tab = _get(m, "test", key)
        L.append(f"| {name} | {label} | " + " | ".join(DASH if tab is None else _f(_get(tab, fill, f))
                                                       for fill, f in cols) + " |")
    L.append("")

    # Cost / provenance
    L += ["## Model size, CPU latency and tuned values", "",
          "| Variant | Parameters | CPU ms per min of audio: AE clean (OLA) | AE score (hop 1) | log-mel | "
          "λ/β chosen for L1 / L2 / L5 |", "|---|---|---|---|---|---|"]
    for v in vs:
        lat = res[v]["latency"]
        pc = _get(lat, "param_count") or _get(res[v]["ingest"], "param_count")
        ch = " / ".join(DASH if _get(res[v][layer], "chosen", "hparam") is None
                        else f"{_get(res[v][layer], 'chosen', 'hparam'):g}" for layer in LAYERS)
        L.append(f"| {v} | {DASH if pc is None else f'{pc:,}'} | {_f(_get(lat, 'ms_per_min_ae_clean_ola'), 1)} | "
                 f"{_f(_get(lat, 'ms_per_min_ae_score_hop1'), 1)} | {_f(_get(lat, 'ms_per_min_mel'), 1)} | {ch} |")
    cpu = next((_get(res[v]["latency"], "cpu_model") for v in vs if res[v]["latency"]), None)
    L += ["", f"CPU: {cpu or DASH}. The VAE's extra parameters are its logvar head; every other layer is identical.",
          ""]
    if curves:
        L += ["## Training curves", ""]
        for v, paths in curves.items():
            for p in paths:
                L.append(f"![{v} {Path(p).stem}]({p})")
        L.append("")
    return "\n".join(L)


def copy_curves(cfg: AEConfig, out_dir: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for v in cfg.variants:
        for png in sorted((cfg.drive_root / v).glob("**/curves.png")):
            rel = png.parent.relative_to(cfg.drive_root / v)
            name = v if str(rel) == "." else f"{v}_{str(rel).replace('=', '-')}"
            dst = out_dir / "curves" / f"{name}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(png, dst)
            found.setdefault(v, []).append(f"curves/{name}.png")
    return found


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-root", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "autoencoder_comparison.md"))
    args = ap.parse_args(argv)
    cfg = load_ae_config(drive_root=args.drive_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = render(collect(cfg), cfg, copy_curves(cfg, out.parent))
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
