"""In-process job store. Pipeline runs take ~26 s per minute of audio on CPU, so the
API submits work here and the browser polls GET /api/jobs/{id}.

One worker thread: the ASR/alignment models are large and not worth running in parallel.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("redact.api.jobs")

# Real pipeline stage names, in order, per job kind. Names match StageTimer stages.
STAGES: dict[str, list[str]] = {
    "detect": ["asr", "alignment", "detection", "mapping"],
    "mask": ["asr", "alignment", "detection", "mapping", "masking"],
    "unmask": ["unmask"],
    "evaluate": ["asr_masked", "align_masked", "metrics"],
}


@dataclass
class Job:
    id: str
    kind: str
    file_id: str
    status: str = "queued"          # queued | running | done | error
    stage: str | None = None
    stages: list[str] = field(default_factory=list)
    progress: float = 0.0
    result: Any = None
    error: str | None = None
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        d = {"job_id": self.id, "kind": self.kind, "file_id": self.file_id, "status": self.status,
             "stage": self.stage, "stages": self.stages, "progress": round(self.progress, 3)}
        if self.status == "done":
            d["result"] = self.result
        if self.status == "error":
            d["error"] = self.error
        return d


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, kind: str, file_id: str, fn: Callable[[Callable[[str], None]], Any]) -> Job:
        """`fn(on_stage)` does the work and returns a JSON-able result."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, file_id=file_id, stages=list(STAGES.get(kind, [])))
        with self._lock:
            self._jobs[job.id] = job

        def on_stage(name: str) -> None:
            job.stage = name
            if name in job.stages:
                job.progress = job.stages.index(name) / max(1, len(job.stages))

        def run() -> None:
            job.status = "running"
            try:
                job.result = fn(on_stage)
                job.progress = 1.0
                job.status = "done"
            except Exception as e:  # surfaced to the UI, not swallowed
                log.error("job %s (%s) failed:\n%s", job.id, kind, traceback.format_exc())
                job.error = str(e) or e.__class__.__name__
                job.status = "error"

        self._pool.submit(run)
        return job

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
