"""HTTP endpoints. Every pipeline operation is a call into src/ — the same functions the CLI uses.

Hard constraints (FRONTEND_SPEC.md):
  - the key is read and used server-side only; no response ever contains it;
  - WAVs are streamed from disk as-is (audio/wav), never transcoded;
  - the original recording is served only when api.allow_original_playback is on;
  - uploads are size-bounded and must be RIFF/WAVE PCM in the configured format.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.alignment import Word
from src.audio import AudioFormatError, load_wav, sha256_file
from src.config import Config
from src.evaluate import evaluate as run_evaluate
from src.evaluate import load_ground_truth, span_metrics
from src.manifest import public_dict, read_manifest
from src.mask import load_key
from src.pipeline import Pipeline, unmask as run_unmask

from .jobs import JobStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"
CHUNK = 1 << 20

NOT_WAV_HINT = "Needs uncompressed WAV — MP3/AAC/Opus cannot be restored sample-for-sample."
FORMAT_HINT = "Needs 16 kHz, mono, 16-bit PCM WAV. Convert with: ffmpeg -i in.wav -ac 1 -ar 16000 -c:a pcm_s16le out.wav"


@dataclass
class FileRecord:
    id: str
    dir: Path
    filename: str
    duration: float
    sample_rate: int
    channels: int
    total_samples: int
    ground_truth: list[dict] | None = None
    masked_entities: list[str] | None = None  # entity set used by the latest completed mask
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def original(self) -> Path:
        return self.dir / "original.wav"

    @property
    def masked(self) -> Path:
        return self.dir / "masked.wav"

    @property
    def manifest(self) -> Path:
        return self.dir / "masked.manifest.json"

    @property
    def restored(self) -> Path:
        return self.dir / "restored.wav"

    def info(self) -> dict:
        return {"file_id": self.id, "filename": self.filename, "duration": round(self.duration, 3),
                "sample_rate": self.sample_rate, "channels": self.channels,
                "total_samples": self.total_samples, "has_ground_truth": self.ground_truth is not None,
                "ground_truth_entities": len(self.ground_truth or [])}


class EntityRequest(BaseModel):
    file_id: str
    entity_types: list[str] | None = None


class FileRequest(BaseModel):
    file_id: str


def build_router(cfg: Config, jobs: JobStore) -> APIRouter:
    r = APIRouter(prefix="/api")
    files: dict[str, FileRecord] = {}
    workdir = Path(cfg.api.workdir)
    if not workdir.is_absolute():
        workdir = ROOT / workdir
    workdir.mkdir(parents=True, exist_ok=True)
    # A job killed mid-write leaves only *.tmp files (all writes are temp + os.replace); sweep them.
    for tmp in workdir.rglob("*.tmp"):
        tmp.unlink(missing_ok=True)
    key_path = Path(cfg.masking.key_file)
    if not key_path.is_absolute():
        key_path = ROOT / key_path

    def get_file(file_id: str) -> FileRecord:
        rec = files.get(file_id)
        if rec is None:
            raise HTTPException(404, f"unknown file_id {file_id!r} (uploads live only for this server session)")
        return rec

    def cfg_for(entity_types: list[str] | None) -> Config:
        available = list(cfg.detection.entities)
        if entity_types is None:
            return cfg
        unknown = sorted(set(entity_types) - set(available))
        if unknown:
            raise HTTPException(400, f"unknown entity type(s): {unknown}; available: {available}")
        ordered = tuple(e for e in available if e in set(entity_types))
        return dataclasses.replace(cfg, detection=dataclasses.replace(cfg.detection, entities=ordered))

    # -- config -------------------------------------------------------------------
    @r.get("/config/entities")
    def config_entities() -> dict:
        return {"available": list(cfg.detection.entities), "defaults": list(cfg.detection.entities),
                "score_threshold": cfg.detection.score_threshold, "padding_ms": cfg.masking.padding_ms,
                "allow_original_playback": cfg.api.allow_original_playback,
                "max_upload_mb": cfg.api.max_upload_mb}

    # -- upload -------------------------------------------------------------------
    @r.post("/upload")
    async def upload(request: Request, file: UploadFile = File(...),
                     ground_truth: UploadFile | None = File(None)) -> dict:
        limit = cfg.api.max_upload_mb * 1024 * 1024
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > limit + 64 * 1024:
            raise HTTPException(413, f"File is larger than the {cfg.api.max_upload_mb} MB upload limit.")
        file_id = uuid.uuid4().hex[:12]
        d = workdir / file_id
        d.mkdir(parents=True)
        dest = d / "original.wav"
        size = 0
        try:
            with open(dest, "wb") as out:
                while chunk := await file.read(CHUNK):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"File is larger than the {cfg.api.max_upload_mb} MB upload limit.")
                    out.write(chunk)
            name = Path(file.filename or "upload.wav").name
            try:
                _, params = load_wav(dest, cfg.audio)  # same validation (and message) as the CLI
            except AudioFormatError as e:
                msg = str(e).replace(str(dest), name).replace(f"-i {dest.name}", f"-i {name}")
                hint = NOT_WAV_HINT if "not a PCM WAV" in msg else FORMAT_HINT
                raise HTTPException(400, {"message": msg, "hint": hint})
            gt = None
            if ground_truth is not None and ground_truth.filename:
                try:
                    gt = _parse_ground_truth(json.loads(await ground_truth.read()))
                except (ValueError, KeyError, TypeError) as e:
                    raise HTTPException(400, {"message": f"ground truth JSON is invalid: {e}",
                                              "hint": 'Expected [{"entity_type", "start_sec", "end_sec"}, ...]'})
            else:
                auto = SAMPLES_DIR / f"{Path(name).stem}.ground_truth.json"
                if auto.exists():
                    gt = _parse_ground_truth(load_ground_truth(auto))
        except HTTPException:
            shutil.rmtree(d, ignore_errors=True)
            raise
        rec = FileRecord(id=file_id, dir=d, filename=name, duration=params.duration_sec,
                         sample_rate=params.sample_rate, channels=params.channels,
                         total_samples=params.total_samples, ground_truth=gt)
        files[file_id] = rec
        return rec.info()

    # -- detect (dry run) -----------------------------------------------------------
    @r.post("/detect")
    def detect(req: EntityRequest) -> dict:
        rec = get_file(req.file_id)
        run_cfg = cfg_for(req.entity_types)

        def work(on_stage):
            p = Pipeline(run_cfg, run_id=f"{rec.id}-detect", out_dir=rec.dir / "detect", on_stage=on_stage)
            a = p.analyze(rec.original, rec.extra.get("words"))
            rec.extra["words"] = a.transcript.words
            sr = a.params.sample_rate
            spans = [{"id": f"span_{i:03d}", "entity_type": s.entity_type, "score": s.score,
                      "start_sec": round(s.sample_start / sr, 3), "end_sec": round(s.sample_end / sr, 3),
                      "sample_start": s.sample_start, "sample_end": s.sample_end}
                     for i, s in enumerate(a.spans, 1)]
            n = sum(s.sample_end - s.sample_start for s in a.spans)
            rows = [{k: v for k, v in row.items() if k != "text"} for row in a.rows()]  # preview only
            return {
                "entity_types": list(run_cfg.detection.entities),
                "score_threshold": run_cfg.detection.score_threshold,
                "rows": rows, "spans": spans,
                "fraction_masked": round(n / a.params.total_samples, 4) if a.params.total_samples else 0.0,
                "entities": _metrics([(s["start_sec"], s["end_sec"], s["entity_type"]) for s in spans],
                                     rec.ground_truth),
                "timings_sec": p.timer.as_dict(),
                "asr_reused": bool(a.asr.get("reused_words")),
            }

        return {"job_id": jobs.submit("detect", rec.id, work).id}

    # -- mask -----------------------------------------------------------------------
    @r.post("/mask")
    def mask(req: EntityRequest) -> dict:
        rec = get_file(req.file_id)
        run_cfg = cfg_for(req.entity_types)

        def work(on_stage):
            key = load_key(key_path, cfg.masking.key_length_bytes, create=True)
            p = Pipeline(run_cfg, run_id=f"{rec.id}-mask", out_dir=rec.dir / "mask", on_stage=on_stage)
            m = p.mask(rec.original, rec.masked, key, manifest_path=rec.manifest, words=rec.extra.get("words"))
            rec.extra["words"] = json_words(rec.dir / "mask" / "words.json") or rec.extra.get("words")
            del key
            rec.masked_entities = list(run_cfg.detection.entities)
            rec.restored.unlink(missing_ok=True)  # any earlier round-trip result is now stale
            return {"manifest": public_dict(m),
                    "entities": _metrics([(s.start_sec, s.end_sec, s.entity_type) for s in m.spans],
                                         rec.ground_truth)}

        return {"job_id": jobs.submit("mask", rec.id, work).id}

    # -- unmask (server-side; the browser only triggers it) --------------------------
    @r.post("/unmask")
    def unmask(req: FileRequest) -> dict:
        rec = get_file(req.file_id)
        if not rec.manifest.exists():
            raise HTTPException(409, "mask this file first")

        def work(on_stage):
            on_stage("unmask")
            m = read_manifest(rec.manifest)
            key = load_key(key_path, cfg.masking.key_length_bytes)
            match = run_unmask(rec.masked, m, key, rec.restored, cfg)
            del key
            return {"match": match, "source_sha256": m.source.sha256,
                    "restored_sha256": sha256_file(rec.restored), "masked_sha256": m.masked.sha256}

        return {"job_id": jobs.submit("unmask", rec.id, work).id}

    # -- evaluate -------------------------------------------------------------------
    @r.post("/evaluate")
    def evaluate(req: FileRequest) -> dict:
        rec = get_file(req.file_id)
        if not rec.manifest.exists():
            raise HTTPException(409, "mask this file first")

        def work(on_stage):
            m = read_manifest(rec.manifest)
            ref_words = rec.extra.get("words")  # original's aligned words, from detect/mask
            res = run_evaluate(rec.original, rec.masked, m, cfg, rec.ground_truth,
                               out_dir=rec.dir, ref_words=ref_words, on_stage=on_stage)
            out = {k: v for k, v in res.metrics.items()
                   if k not in ("reference_text", "hypothesis_text", "original", "masked")}
            if "entities" in out:  # per-entity rows carry ground-truth text; keep counts only
                out["entities"] = {k: v for k, v in out["entities"].items() if k != "per_entity"}
            out["entity_types"] = rec.masked_entities
            return out

        return {"job_id": jobs.submit("evaluate", rec.id, work).id}

    # -- jobs -----------------------------------------------------------------------
    @r.get("/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job.public()

    # -- files: raw bytes from disk, nothing in between -------------------------------
    no_store = {"Cache-Control": "no-store"}

    @r.get("/files/{file_id}/masked.wav")
    def masked_wav(file_id: str):
        rec = get_file(file_id)
        if not rec.masked.exists() or not rec.manifest.exists():
            raise HTTPException(404, "not masked yet")
        return FileResponse(rec.masked, media_type="audio/wav", headers=no_store)

    @r.get("/files/{file_id}/original.wav")
    def original_wav(file_id: str):
        """DEMO ONLY: the unredacted recording contains the PII this system protects.
        Disabled when api.allow_original_playback is false (real deployments)."""
        if not cfg.api.allow_original_playback:
            raise HTTPException(403, "Original playback is disabled (api.allow_original_playback = false).")
        rec = get_file(file_id)
        return FileResponse(rec.original, media_type="audio/wav", headers=no_store)

    @r.get("/files/{file_id}/manifest")
    def manifest(file_id: str) -> dict:
        rec = get_file(file_id)
        if not rec.manifest.exists():
            raise HTTPException(404, "not masked yet")
        return public_dict(read_manifest(rec.manifest))  # ciphertext/nonce/tag stripped

    return r


def json_words(path: Path) -> list[Word] | None:
    return [Word(**w) for w in json.loads(path.read_text())] if path.exists() else None


def _parse_ground_truth(d: Any) -> list[dict]:
    items = d["entities"] if isinstance(d, dict) else d
    if not isinstance(items, list):
        raise ValueError("expected a list of entities")
    out = []
    for g in items:
        out.append({"entity_type": str(g["entity_type"]), "start_sec": float(g["start_sec"]),
                    "end_sec": float(g["end_sec"])})
    return out


def _metrics(spans: list[tuple[float, float, str]], gt: list[dict] | None) -> dict | None:
    if gt is None:
        return None
    m = span_metrics(spans, gt)
    m.pop("per_entity", None)
    return m

