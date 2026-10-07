"""Demo assets: waveform figure, before/after clips, slide-ready detection table, timings."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import AudioParams, load_wav, write_wav
from .manifest import Manifest


def waveform_figure(original: np.ndarray, masked: np.ndarray, manifest: Manifest, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sr = manifest.source.sample_rate
    t = np.arange(len(original)) / sr
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True, constrained_layout=True)
    for ax, data, title in ((axes[0], original, "Original"), (axes[1], masked, "Masked (shareable)")):
        ax.plot(t, data[:, 0] / 32768.0, linewidth=0.4, color="#3b5b92")
        for s in manifest.spans:
            ax.axvspan(s.sample_start / sr, s.sample_end / sr, color="#d9534f", alpha=0.25, linewidth=0)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("amplitude")
        ax.set_title(title, loc="left", fontsize=10)
    for s in manifest.spans:
        axes[0].text((s.sample_start + s.sample_end) / 2 / sr, 0.85, s.entity_type, ha="center",
                     fontsize=7, color="#a02622")
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"{manifest.source.filename}: {manifest.stats['spans_masked']} spans masked, "
                 f"{100 * manifest.stats['fraction_masked']:.1f}% of audio", fontsize=11)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def export_clips(original: np.ndarray, masked: np.ndarray, manifest: Manifest, out_dir: str | Path,
                 context_sec: float = 1.5) -> list[Path]:
    """Before/after clip per span with some surrounding context."""
    src = manifest.source
    sr = src.sample_rate
    out_dir = Path(out_dir)
    paths = []
    for s in manifest.spans:
        a = max(0, s.sample_start - int(context_sec * sr))
        b = min(src.total_samples, s.sample_end + int(context_sec * sr))
        for tag, data in (("before", original), ("after", masked)):
            p = out_dir / f"{s.id}_{s.entity_type.replace('+', '_')}_{tag}.wav"
            write_wav(p, data[a:b], AudioParams(sr, src.channels, src.sample_width_bytes, b - a))
            paths.append(p)
    return paths


def detection_table(manifest: Manifest) -> str:
    """Markdown table for a slide."""
    lines = ["| # | Entity | Score | Preview | Time (s) | Masked |", "|---|---|---|---|---|---|"]
    for s in manifest.spans:
        lines.append(f"| {s.id} | {s.entity_type} | {s.score:.2f} | `{s.text_preview}` | "
                     f"{s.start_sec:.2f}–{s.end_sec:.2f} | yes |")
    for i, d in enumerate(manifest.detected_but_not_masked, 1):
        lines.append(f"| low_{i:03d} | {d['entity_type']} | {d['score']:.2f} | — | "
                     f"{d['start_sec']:.2f}–{d['end_sec']:.2f} | no ({d['reason']}) |")
    return "\n".join(lines)


def timing_table(manifest: Manifest) -> str:
    minutes = manifest.source.duration_sec / 60 or 1
    lines = ["| Stage | Seconds | Sec per minute of audio |", "|---|---|---|"]
    for k, v in manifest.timings_sec.items():
        lines.append(f"| {k} | {v:.2f} | {v / minutes:.2f} |")
    return "\n".join(lines)


def build_demo(original_path: str | Path, masked_path: str | Path, manifest: Manifest, out_dir: str | Path) -> dict:
    original, _ = load_wav(original_path)
    masked, _ = load_wav(masked_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = out_dir / "waveform.png"
    waveform_figure(original, masked, manifest, fig)
    clips = export_clips(original, masked, manifest, out_dir / "clips")
    (out_dir / "detection_table.md").write_text(detection_table(manifest) + "\n", encoding="utf-8")
    (out_dir / "timings.md").write_text(timing_table(manifest) + "\n", encoding="utf-8")
    return {"figure": fig, "clips": clips, "table": out_dir / "detection_table.md", "timings": out_dir / "timings.md"}
