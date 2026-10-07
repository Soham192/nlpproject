"""API constraints from FRONTEND_SPEC.md. ASR/alignment/Presidio are stubbed (see conftest.stub_pipeline)."""
from __future__ import annotations

import base64
import dataclasses
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.audio import AudioParams, write_wav  # noqa: E402
from src.detect import Detection  # noqa: E402
from tests.conftest import WORDS, det, make_wav, stub_pipeline  # noqa: E402


def _dets(t):
    out = [det(t, "CREDIT_CARD", "45320151")]
    if "thanks." in t.text:
        out.append(Detection("DATE_TIME", 0.3, t.text.index("thanks."), t.text.index("thanks.") + 7, "thanks."))
    return out


@pytest.fixture
def setup(tmp_path, monkeypatch, cfg):
    stub_pipeline(monkeypatch, WORDS, _dets)
    from api.jobs import JobStore
    from api.main import create_app

    key_file = tmp_path / "key.bin"
    key_file.write_bytes(bytes(range(32)))
    c = dataclasses.replace(
        cfg,
        masking=dataclasses.replace(cfg.masking, key_file=str(key_file)),
        api=dataclasses.replace(cfg.api, workdir=str(tmp_path / "api"), max_upload_mb=1),
        output=dataclasses.replace(cfg.output, save_intermediates=False),
    )
    jobs = JobStore()
    client = TestClient(create_app(c, jobs))
    return client, c, key_file.read_bytes(), tmp_path


def _upload(client, path, name="call.wav"):
    with open(path, "rb") as f:
        return client.post("/api/upload", files={"file": (name, f, "audio/wav")})


def _wait(client, job_id, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise TimeoutError(job_id)


def _all_responses_flow(client, tmp_path):
    """Run the whole UI flow; return (file_id, every response body)."""
    bodies = []
    up = _upload(client, make_wav(tmp_path / "in.wav"))
    bodies.append(up.content)
    fid = up.json()["file_id"]
    for ep, payload in (("detect", {"file_id": fid, "entity_types": ["CREDIT_CARD", "DATE_TIME"]}),
                        ("mask", {"file_id": fid, "entity_types": ["CREDIT_CARD", "DATE_TIME"]}),
                        ("unmask", {"file_id": fid})):
        r = client.post(f"/api/{ep}", json=payload)
        bodies.append(r.content)
        j = _wait(client, r.json()["job_id"])
        assert j["status"] == "done", j
        bodies.append(client.get(f"/api/jobs/{r.json()['job_id']}").content)
    for path in ("manifest", "masked.wav", "original.wav"):
        bodies.append(client.get(f"/api/files/{fid}/{path}").content)
    bodies.append(client.get("/api/config/entities").content)
    return fid, bodies


def test_full_flow_and_round_trip(setup):
    client, cfg, key, tmp_path = setup
    up = _upload(client, make_wav(tmp_path / "in.wav"))
    assert up.status_code == 200 and up.json()["sample_rate"] == 16000
    fid = up.json()["file_id"]

    d = _wait(client, client.post("/api/detect", json={"file_id": fid}).json()["job_id"])
    assert d["status"] == "done" and d["stages"] == ["asr", "alignment", "detection", "mapping"]
    rows = d["result"]["rows"]
    assert [r["will_mask"] for r in rows] == [True, False]
    assert rows[1]["reason"] == "below score_threshold"
    assert all("text" not in r for r in rows)  # preview only

    m = _wait(client, client.post("/api/mask", json={"file_id": fid}).json()["job_id"])
    assert m["status"] == "done" and m["stage"] == "masking"
    u = _wait(client, client.post("/api/unmask", json={"file_id": fid}).json()["job_id"])
    res = u["result"]
    assert res["match"] is True and res["source_sha256"] == res["restored_sha256"]


def test_key_never_in_any_response(setup):
    client, cfg, key, tmp_path = setup
    _, bodies = _all_responses_flow(client, tmp_path)
    needles = [key, key.hex().encode(), base64.b64encode(key)]
    for b in bodies:
        for n in needles:
            assert n not in b


def test_manifest_endpoint_strips_ciphertext(setup):
    client, cfg, key, tmp_path = setup
    fid, _ = _all_responses_flow(client, tmp_path)
    man = client.get(f"/api/files/{fid}/manifest").json()
    assert man["spans"]
    for s in man["spans"]:
        assert not {"ciphertext", "nonce", "tag"} & set(s)
        assert {"sample_start", "sample_end", "entity_type", "score"} <= set(s)


def test_wav_served_byte_exact(setup):
    client, cfg, key, tmp_path = setup
    src = make_wav(tmp_path / "in.wav")
    fid, _ = _all_responses_flow(client, tmp_path)
    r = client.get(f"/api/files/{fid}/original.wav")
    assert r.headers["content-type"] == "audio/wav"
    assert r.content == src.read_bytes()
    masked = client.get(f"/api/files/{fid}/masked.wav")
    assert masked.content == (tmp_path / "api" / fid / "masked.wav").read_bytes()


def test_original_gated(tmp_path, monkeypatch, cfg):
    stub_pipeline(monkeypatch, WORDS, _dets)
    from api.jobs import JobStore
    from api.main import create_app

    c = dataclasses.replace(cfg, api=dataclasses.replace(cfg.api, workdir=str(tmp_path / "api"),
                                                         allow_original_playback=False))
    client = TestClient(create_app(c, JobStore()))
    fid = _upload(client, make_wav(tmp_path / "in.wav")).json()["file_id"]
    assert client.get(f"/api/files/{fid}/original.wav").status_code == 403
    assert client.get("/api/config/entities").json()["allow_original_playback"] is False


def test_upload_rejects_mp3_with_cli_message(setup):
    client, cfg, key, tmp_path = setup
    p = tmp_path / "call.mp3"
    p.write_bytes(b"ID3\x03\x00" + b"\x00" * 1000)
    r = _upload(client, p, "call.mp3")
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "not a PCM WAV" in detail["message"] and "call.mp3" in detail["message"]
    assert "sample-for-sample" in detail["hint"]


def test_upload_rejects_wrong_format(setup):
    client, cfg, key, tmp_path = setup
    p = tmp_path / "44k.wav"
    write_wav(p, np.zeros((4410, 1), np.int16), AudioParams(44100, 1, 2, 4410))
    r = _upload(client, p, "44k.wav")
    assert r.status_code == 400 and "Refusing to transcode" in r.json()["detail"]["message"]


def test_upload_rejects_oversize(setup):
    client, cfg, key, tmp_path = setup
    big = make_wav(tmp_path / "big.wav", seconds=40)  # ~1.3 MB > 1 MB limit
    assert _upload(client, big).status_code == 413
    assert not any((tmp_path / "api").iterdir())  # rejected upload leaves nothing behind


def test_unknown_entity_type_rejected(setup):
    client, cfg, key, tmp_path = setup
    fid = _upload(client, make_wav(tmp_path / "in.wav")).json()["file_id"]
    assert client.post("/api/detect", json={"file_id": fid, "entity_types": ["NOPE"]}).status_code == 400
