"""CLI entry points.

    redact mask    input.wav  -o output.wav  --key-file key.bin
    redact unmask  output.wav --manifest output.manifest.json --key-file key.bin -o restored.wav
    redact eval    --original input.wav --masked output.wav --manifest output.manifest.json
    redact detect  input.wav   # dry run: print detected spans, change nothing
    redact demo    --original input.wav --masked output.wav --manifest output.manifest.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, load_config
from .timing import setup_logging


def _cmd_detect(args, cfg) -> int:
    from .pipeline import Pipeline

    p = Pipeline(cfg, args.run_id)
    a = p.analyze(args.input)
    print(f"\nTranscript (normalised):\n  {a.transcript.text}\n")
    hdr = f"{'entity':<22} {'score':>5}  {'chars':>11}  {'time (s)':>13}  {'mask':<4}  text"
    print(hdr)
    print("-" * len(hdr))
    for r in a.rows():
        flag = "yes" if r["will_mask"] else "no"
        print(f"{r['entity_type']:<22} {r['score']:>5.2f}  {r['char_start']:>5}-{r['char_end']:<5}  "
              f"{r['start_sec']:>6.2f}-{r['end_sec']:<6.2f}  {flag:<4}  {r['text']}")
    print(f"\n{len(a.spans)} merged range(s) would be masked "
          f"({sum(s.sample_end - s.sample_start for s in a.spans) / a.params.sample_rate:.2f}s of "
          f"{a.params.duration_sec:.2f}s). Nothing was modified.")
    print(f"Intermediates: {p.out_dir}")
    return 0


def _cmd_mask(args, cfg) -> int:
    from .mask import load_key
    from .pipeline import Pipeline

    key = load_key(args.key_file, cfg.masking.key_length_bytes, create=True)
    p = Pipeline(cfg, args.run_id)
    m = p.mask(args.input, args.output, key, key_id=args.key_id, manifest_path=args.manifest)
    from .pipeline import default_manifest_path
    mp = args.manifest or default_manifest_path(Path(args.output), cfg)
    print(f"masked {m.stats['spans_masked']} span(s), {100 * m.stats['fraction_masked']:.1f}% of audio")
    print(f"  audio:    {args.output}")
    print(f"  manifest: {mp}")
    print(f"  timings:  {m.timings_sec}")
    return 0


def _cmd_unmask(args, cfg) -> int:
    from .manifest import read_manifest
    from .mask import load_key
    from .pipeline import unmask

    key = load_key(args.key_file, cfg.masking.key_length_bytes)
    ok = unmask(args.input, read_manifest(args.manifest), key, args.output, cfg)
    print(f"restored -> {args.output}: sha256 {'matches' if ok else 'DOES NOT match'} the original")
    return 0 if ok else 1


def _cmd_eval(args, cfg) -> int:
    from .evaluate import evaluate, load_ground_truth
    from .manifest import read_manifest

    m = read_manifest(args.manifest)
    gt = load_ground_truth(args.ground_truth) if args.ground_truth else None
    out = args.output or Path(cfg.output.dir) / m.run_id
    r = evaluate(args.original, args.masked, m, cfg, gt, out)
    print(r.table())
    print(f"metrics -> {Path(out) / 'metrics.json'}")
    return 0 if r.metrics["outside_span_identical"] else 1


def _cmd_demo(args, cfg) -> int:
    from .demo import build_demo
    from .manifest import read_manifest

    m = read_manifest(args.manifest)
    out = args.output or Path(cfg.output.dir) / m.run_id / "demo"
    res = build_demo(args.original, args.masked, m, out)
    print(f"figure:  {res['figure']}\nclips:   {len(res['clips'])} in {Path(out) / 'clips'}\n"
          f"table:   {res['table']}\ntimings: {res['timings']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="redact", description="Reversible selective audio redaction")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="dry run: print detected spans, change nothing")
    d.add_argument("input")
    d.add_argument("--run-id")
    d.set_defaults(fn=_cmd_detect)

    m = sub.add_parser("mask", help="encrypt sensitive spans into a manifest, silence them in the audio")
    m.add_argument("input")
    m.add_argument("-o", "--output", required=True)
    m.add_argument("--key-file", required=True, help="raw AES key; generated (0600) if missing")
    m.add_argument("--key-id", default="local-key-01")
    m.add_argument("--manifest", help="manifest path (default: <output stem>.manifest.json)")
    m.add_argument("--run-id")
    m.set_defaults(fn=_cmd_mask)

    u = sub.add_parser("unmask", help="restore the original audio with manifest + key")
    u.add_argument("input")
    u.add_argument("--manifest", required=True)
    u.add_argument("--key-file", required=True)
    u.add_argument("-o", "--output", required=True)
    u.set_defaults(fn=_cmd_unmask)

    e = sub.add_parser("eval", help="WER/BERTScore over non-redacted spans + entity P/R")
    e.add_argument("--original", required=True)
    e.add_argument("--masked", required=True)
    e.add_argument("--manifest", required=True)
    e.add_argument("--ground-truth", help="JSON list of {entity_type, start_sec, end_sec}")
    e.add_argument("-o", "--output", help="dir for metrics.json (default: data/outputs/<run_id>)")
    e.set_defaults(fn=_cmd_eval)

    g = sub.add_parser("demo", help="waveform figure, before/after clips, slide tables")
    g.add_argument("--original", required=True)
    g.add_argument("--masked", required=True)
    g.add_argument("--manifest", required=True)
    g.add_argument("-o", "--output")
    g.set_defaults(fn=_cmd_demo)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)
    try:
        return args.fn(args, cfg)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
