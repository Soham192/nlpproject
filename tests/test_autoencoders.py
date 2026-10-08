"""Autoencoder add-on: local, CPU-only checks. No training, no dataset download."""
import builtins
import importlib.util

import numpy as np
import pytest
import torch

from src.autoencoders import data, degrade
from src.autoencoders.common import auroc, best_f1_threshold
from src.autoencoders.config import VARIANTS, load_ae_config
from src.autoencoders.models import build, param_count

HAS_WHISPER = importlib.util.find_spec("whisper") is not None
needs_whisper = pytest.mark.skipif(not HAS_WHISPER, reason="openai-whisper not installed (pip install -e .[ae])")


@pytest.fixture(scope="module")
def ae_cfg():
    return load_ae_config()


def _model(variant, cfg):
    grid = cfg.hparam_grid(variant)
    return build(variant, cfg, grid[1][0] if grid else None)


# --------------------------------------------------------------------------- models

@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_output_shapes(variant, ae_cfg):
    m = _model(variant, ae_cfg).train()
    x = torch.randn(7, ae_cfg.features.input_dim)
    out = m(x)
    assert out.recon.shape == x.shape
    assert out.z.shape == (7, ae_cfg.model["latent"])
    if variant == "vae":
        assert out.mu.shape == out.logvar.shape == (7, ae_cfg.model["latent"])
    loss, parts = m.loss(x, x)
    assert loss.ndim == 0 and torch.isfinite(loss) and "mse" in parts
    assert m.eval().score(x).shape == (7,)


def test_param_counts_match_plan(ae_cfg):
    counts = {v: param_count(_model(v, ae_cfg)) for v in VARIANTS}
    assert counts["dense"] == counts["sparse"] == counts["dae"] == 1_608_512
    assert counts["vae"] == 1_608_512 + 256 * 64 + 64  # only the logvar head differs


def test_one_train_step_reduces_loss(ae_cfg):
    from src.autoencoders.train import train_step

    torch.manual_seed(0)
    m = _model("dense", ae_cfg)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(64, ae_cfg.features.input_dim)
    first = train_step(m, opt, x, x, 1.0)["loss"]
    for _ in range(20):
        last = train_step(m, opt, x, x, 1.0)["loss"]
    assert last < first


def test_untuned_variant_rejects_missing_hparam(ae_cfg):
    with pytest.raises(ValueError):
        build("vae", ae_cfg)


# --------------------------------------------------------------------------- corruption

def _speechlike(n=16000, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / 16000
    return (0.3 * torch.sin(2 * np.pi * 220 * t) * torch.sin(2 * np.pi * 3 * t)
            + 0.02 * torch.randn(n, generator=g)).float()


@pytest.mark.parametrize("fn", [
    lambda x, g: data.add_noise_snr(x, 10.0, g),
    lambda x, g: data.add_noise_snr(x, 5.0, g, "pink"),
    lambda x, g: data.telephone_bandlimit(x),
    lambda x, g: data.mulaw_roundtrip(x, 256),
    lambda x, g: data.corrupt(x, load_ae_config().corruption, g),
    lambda x, g: data.telephone_chain(x, 0.0, load_ae_config().corruption, g),
])
def test_corruption_changes_input_keeps_shape(fn):
    x = _speechlike()
    y = fn(x, torch.Generator().manual_seed(1))
    assert y.shape == x.shape and y.dtype == x.dtype
    assert not torch.allclose(x, y)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("snr", [0.0, 7.5, 20.0])
def test_noise_hits_target_snr(snr):
    x = _speechlike(48000)
    y = data.add_noise_snr(x, snr, torch.Generator().manual_seed(3))
    got = 10 * torch.log10(torch.mean(x ** 2) / torch.mean((y - x) ** 2))
    assert abs(float(got) - snr) < 0.1


def test_bandlimit_removes_out_of_band_energy():
    t = torch.arange(16000) / 16000
    x = (torch.sin(2 * np.pi * 100 * t) + torch.sin(2 * np.pi * 1000 * t) + torch.sin(2 * np.pi * 6000 * t)).float()
    spec = torch.fft.rfft(data.telephone_bandlimit(x)).abs()
    assert spec[100] < 1e-3 and spec[6000] < 1e-3 and spec[1000] > 1000


def test_ingest_degradations_change_input_keep_shape(ae_cfg):
    speech = _speechlike().numpy()
    rng = np.random.default_rng(0)
    for kind in degrade.INGEST_KINDS:
        y = degrade.make_unusable(kind, speech, rng, ae_cfg.eval["ingest"])
        assert y.shape == speech.shape and not np.allclose(y, speech), kind


# --------------------------------------------------------------------------- windowing / overlap-add

@pytest.mark.parametrize("n_frames", [5, 16, 17, 61, 300])
@pytest.mark.parametrize("hop", [1, 4, 16])
def test_windowing_overlap_add_roundtrip(n_frames, hop):
    mel = torch.randn(n_frames, 80, dtype=torch.float64)
    wins, starts, padded = data.frame_windows(mel, 16, hop)
    assert wins.shape == (len(starts), 16 * 80)
    rec = data.overlap_add(wins, starts, padded, 16, n_frames)
    assert rec.shape == mel.shape
    assert torch.allclose(rec, mel, rtol=0, atol=1e-12)


def test_window_starts_cover_every_frame():
    for n in range(1, 80):
        s = data.window_starts(n, 16, 4)
        covered = set()
        for a in s:
            covered.update(range(a, a + 16))
        assert set(range(n)) <= covered


def test_identity_clean_mel_is_exact(ae_cfg):
    from src.autoencoders.frontend import clean_mel

    class Identity(torch.nn.Module):
        def reconstruct(self, x):
            return x

    mel = torch.randn(123, 80)  # the model path runs in float32
    assert torch.allclose(clean_mel(Identity(), mel, 16, 4), mel, rtol=0, atol=1e-6)


# --------------------------------------------------------------------------- splits

def _rows(prefix, speakers, per=10):
    return [{"id": f"{s}-{prefix}-{i}", "speaker_id": s, "chapter_id": 1} for s in speakers for i in range(per)]


def test_eval_subsets_disjoint_and_seeded():
    dev = _rows("d", range(100, 140))
    a = data.select_eval_subsets(dev, {"dev_val": 150, "dev_tune": 100}, seed=1)
    b = data.select_eval_subsets(list(reversed(dev)), {"dev_val": 150, "dev_tune": 100}, seed=1)
    assert [r["id"] for r in a["dev_val"]] == [r["id"] for r in b["dev_val"]]  # stream order irrelevant
    assert not {r["id"] for r in a["dev_val"]} & {r["id"] for r in a["dev_tune"]}
    with pytest.raises(ValueError):
        data.select_eval_subsets(dev, {"dev_val": 500}, seed=1)


def test_dev_test_train_disjoint_check():
    splits = {"train": _rows("t", range(0, 5)), "dev_val": _rows("d", range(100, 105)),
              "test_asr": _rows("x", range(200, 205))}
    data.assert_disjoint(splits)
    splits["test_asr"].append({"id": "leak", "speaker_id": 100, "chapter_id": 1})
    with pytest.raises(AssertionError, match="speakers shared"):
        data.assert_disjoint(splits)
    splits["test_asr"].pop()
    splits["test_ingest"] = [splits["test_asr"][0]]
    with pytest.raises(AssertionError, match="in both"):
        data.assert_disjoint(splits)


def test_train_selection_speaker_balanced():
    rows = [{"id": f"{s}-{i}", "speaker_id": s, "chapter_id": 1} for i in range(50) for s in range(4)]
    picked = data.select_train_balanced(iter(rows), lambda r: 10.0, total_sec=200, n_speakers=4, max_utt_sec=30)
    per = {s: sum(r["duration_sec"] for r in picked if r["speaker_id"] == s) for s in range(4)}
    assert all(v == 50 for v in per.values())


# --------------------------------------------------------------------------- leak spans

def test_leak_spans_built_from_speech(ae_cfg):
    c = ae_cfg.eval["leak"]
    rng = np.random.default_rng(0)
    sr = 16000
    audio = np.zeros(3 * sr, np.float32)
    audio[sr:2 * sr] = _speechlike(sr).numpy()  # 1 s of "speech" between silences
    s, e, edge = degrade.place_span(audio, rng, (400, 600), 50, c["vad_rel_db"])
    frag = slice(s, s + 800) if edge == "leading" else slice(e - 800, e)
    assert sr <= frag.start and frag.stop <= 2 * sr  # the 50 ms fragment lies in the active region
    fill = degrade.fill_signal("tone", e - s, rng, c)
    neg = degrade.build_span(audio, s, e, edge, 0, fill)
    pos = degrade.build_span(audio, s, e, edge, 25, fill)
    assert len(neg) == len(pos) == e - s
    np.testing.assert_allclose(neg, degrade.to_pcm_grid(fill))
    k = 400  # 25 ms
    diff = np.flatnonzero(~np.isclose(neg, pos))
    assert diff.size and (diff.max() < k if edge == "leading" else diff.min() >= len(pos) - k)


# --------------------------------------------------------------------------- metrics

def test_auroc_and_threshold():
    assert auroc([0, 1, 2], [3, 4, 5]) == 1.0
    assert auroc([3, 4, 5], [0, 1, 2]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    t = best_f1_threshold(np.array([0.1, 0.2, 0.3]), np.array([0.8, 0.9]))
    assert t["f1"] == 1.0 and 0.3 < t["threshold"] <= 0.8


# --------------------------------------------------------------------------- whisper features

@needs_whisper
def test_log_mel_matches_whisper_exactly():
    import whisper

    x = _speechlike(32000)
    ours = data.log_mel(x, 80)
    theirs = whisper.log_mel_spectrogram(x, n_mels=80)
    assert ours.shape == (200, 80)
    assert torch.equal(ours, theirs.T)


@needs_whisper
def test_window_scoring_on_real_mel(ae_cfg):
    from src.autoencoders.frontend import clean_whisper_mel, score_mel

    m = _model("dense", ae_cfg).eval()
    mel = data.log_mel(_speechlike(), 80)
    assert score_mel(m, mel, 16, 4).ndim == 1
    wmel = torch.full((80, 3000), -0.5)
    wmel[:, :100] = mel.T
    out = clean_whisper_mel(m, wmel, 100, 16, 4)
    assert torch.equal(out[:, 100:], wmel[:, 100:])  # padding frames untouched
    assert torch.equal(wmel[:, :100], mel.T)         # input copy not modified


# --------------------------------------------------------------------------- integrations stay off

def test_disabled_integrations_never_import_autoencoders(monkeypatch, cfg, key, wav, tmp_path):
    """With every `enabled: false`, masking runs without touching src.autoencoders."""
    from src.config import ae_integration
    from src.pipeline import Pipeline
    from tests.conftest import WORDS, det, stub_pipeline

    for name in ("ingest_gate", "asr_frontend", "leak_check"):
        assert ae_integration(cfg.raw, name) is None
    real_import = builtins.__import__

    def guard(name, globals=None, locals=None, fromlist=(), level=0):
        if "autoencoders" in name or any("autoencoders" in f for f in (fromlist or ())):
            raise AssertionError(f"pipeline imported {name} {fromlist} with integrations disabled")
        return real_import(name, globals, locals, fromlist, level)

    stub_pipeline(monkeypatch, WORDS, lambda t: [det(t, "CREDIT_CARD", "45320151")])
    monkeypatch.setattr(builtins, "__import__", guard)
    m = Pipeline(cfg, "test").mask(wav, tmp_path / "out.wav", key)
    assert m.spans


# --------------------------------------------------------------------------- per-item scores

def test_scores_jsonl_rows(tmp_path):
    from src.autoencoders import eval_ingest, eval_leak
    from src.autoencoders.common import read_jsonl, write_jsonl

    split = data.SplitCache(np.zeros(1), np.zeros((1, 80)), [{"id": "a"}, {"id": "b"}])
    rows = eval_ingest.item_rows("dense", "test", split, [0.1, 0.2], [0.9, 0.8], ["tone", "silence"])
    assert [(r["id"], r["label"]) for r in rows] == [("a#clean", 0), ("b#clean", 0), ("a#tone", 1), ("b#silence", 1)]
    ex = [eval_leak.Example("u1", "tone", 0, "leading", np.zeros(1), "test_leak"),
          eval_leak.Example("u1", "tone", 25, "leading", np.zeros(1), "test_leak")]
    lrows = eval_leak.item_rows("vae", "test", ex, {"calibrated": np.array([1.0, 5.0]), "literal": np.array([-1.0, -0.5])})
    assert [(r["label"], r["score"], r["score_literal"], r["frag_ms"]) for r in lrows] == [(0, 1.0, -1.0, 0), (1, 5.0, -0.5, 25)]
    write_jsonl(tmp_path / "scores.jsonl", rows + lrows)
    back = read_jsonl(tmp_path / "scores.jsonl")
    assert back == rows + lrows and all({"id", "label", "score", "variant"} <= set(r) for r in back)


# --------------------------------------------------------------------------- dev-only fitting (layer 5)

def _leak_examples(split_name, utts=("u1", "u2")):
    from src.autoencoders.eval_leak import Example

    return [Example(u, fill, f, "leading", np.zeros(1), split_name)
            for u in utts for fill in ("silence", "tone", "noise") for f in (0, 25, 50)]


def test_leak_calibration_rejects_test_items(ae_cfg):
    from src.autoencoders.eval_leak import calibrate, thresholds

    c = ae_cfg.eval["leak"]
    dev = _leak_examples("dev_tune")
    errs = [np.random.default_rng(i).random(5) for i in range(len(dev))]
    calibrate(dev, errs, c)  # dev only: fine
    thresholds(dev, np.arange(len(dev), dtype=float), c)
    mixed = dev + _leak_examples("test_leak", ("t1",))
    merrs = errs + [np.zeros(5)] * (len(mixed) - len(dev))
    with pytest.raises(ValueError, match="dev items only"):
        calibrate(mixed, merrs, c)
    with pytest.raises(ValueError, match="dev items only"):
        thresholds(mixed, np.arange(len(mixed), dtype=float), c)


def test_leak_eval_fits_calibration_on_dev_only(monkeypatch, tmp_path, ae_cfg):
    """Run eval_leak.run_variant end to end on a synthetic cache and record which split's items reach
    calibrate() and thresholds(). Fails if any test-split item is used for fitting."""
    import dataclasses
    from src.autoencoders import eval_leak

    rng = np.random.default_rng(0)

    def split(prefix, n):
        audios, index, so = [], [], 0
        for i in range(n):
            t = np.arange(32000) / 16000
            a = (0.3 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 2 * t) > 0)
                 + 0.01 * rng.standard_normal(len(t)))
            pcm = (a * 32767).astype(np.int16)
            index.append({"id": f"{prefix}{i}", "speaker_id": i, "sample_offset": so, "n_samples": len(pcm),
                          "frame_offset": 0, "n_frames": 0})
            audios.append(pcm)
            so += len(pcm)
        return data.SplitCache(np.concatenate(audios), np.zeros((1, 80)), index)

    splits = {"dev_tune": split("d", 4), "test_leak": split("t", 4)}
    cfg = dataclasses.replace(ae_cfg, drive_root=tmp_path, eval={**ae_cfg.eval, "leak": {**ae_cfg.eval["leak"], "dev_spans": 4}})
    seen = {"calibrate": set(), "thresholds": set()}
    real_cal, real_thr = eval_leak.calibrate, eval_leak.thresholds

    def spy_cal(ex, errs, c):
        seen["calibrate"] |= {e.split for e in ex}
        return real_cal(ex, errs, c)

    def spy_thr(ex, scores, c):
        seen["thresholds"] |= {e.split for e in ex}
        return real_thr(ex, scores, c)

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(eval_leak, "calibrate", spy_cal)
    monkeypatch.setattr(eval_leak, "thresholds", spy_thr)
    monkeypatch.setattr(eval_leak, "require_gpu", lambda: "cpu")
    monkeypatch.setattr(eval_leak, "ensure_local_cache", lambda cfg: tmp_path)
    monkeypatch.setattr(eval_leak, "load_split", lambda root, name: splits[name])
    monkeypatch.setattr(eval_leak, "candidate_runs", lambda cfg, v: [(None, tmp_path / "run")])
    monkeypatch.setattr(eval_leak, "load_model", lambda *a, **k: Stub())
    monkeypatch.setattr(eval_leak, "param_count", lambda m: 0)
    monkeypatch.setattr(eval_leak, "window_errors",
                        lambda model, ex, cfg, device: [np.abs(e.audio[:20]) + 0.1 for e in ex])
    m = eval_leak.run_variant(cfg, "dense")
    assert seen["calibrate"] == {"dev_tune"} and seen["thresholds"] == {"dev_tune"}
    assert set(m["chosen"]["calibration"]) == set(ae_cfg.eval["leak"]["fills"])


# --------------------------------------------------------------------------- DAE sweep

def test_dae_sweeps_corruption_strength(ae_cfg):
    grid = ae_cfg.hparam_grid("dae")
    assert grid is not None and len(grid[1]) == 3
    assert all(len(ae_cfg.hparam_grid(v)[1]) == 3 for v in ("sparse", "dae", "vae"))  # same tuning budget
    for v in grid[1]:
        m = build("dae", ae_cfg, v)
        c = m.corruption_cfg(ae_cfg.corruption)
        assert c["snr_db"] == [v, ae_cfg.corruption["snr_db"][1]]
        assert param_count(m) == 1_608_512  # network unchanged by the knob


def test_record_choice_writes_run_info(tmp_path, ae_cfg):
    import dataclasses
    from src.autoencoders.common import read_json, record_choice, run_dir, write_json

    cfg = dataclasses.replace(ae_cfg, drive_root=tmp_path)
    for v in cfg.hparam_grid("dae")[1]:
        write_json(run_dir(cfg, "dae", v) / "run_info.json", {"hparam": v})
    record_choice(cfg, "dae", "ingest", 5.0)
    record_choice(cfg, "dae", "asr", 0.0)
    record_choice(cfg, "dae", "ingest", 10.0)  # re-selection replaces the old choice
    top = read_json(tmp_path / "dae" / "run_info.json")
    assert top["chosen_hparam"] == {"ingest": 10.0, "asr": 0.0} and top["hparam_name"] == "min_snr_db"
    assert read_json(run_dir(cfg, "dae", 10.0) / "run_info.json")["selected_for_layers"] == ["ingest"]
    assert read_json(run_dir(cfg, "dae", 0.0) / "run_info.json")["selected_for_layers"] == ["asr"]
    assert read_json(run_dir(cfg, "dae", 5.0) / "run_info.json")["selected_for_layers"] == []
    record_choice(cfg, "dense", "ingest", None)  # untuned: no-op
    assert not (tmp_path / "dense").exists()
