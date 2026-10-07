"""FastAPI app for the redaction demo UI.

    uvicorn api.main:app --port 8000          # API (and the built UI from web/dist, if present)
    cd web && npm run dev                     # UI dev server on :5173, proxies /api to :8000
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import DEFAULT_CONFIG_PATH, Config, load_config
from src.timing import setup_logging

from .jobs import JobStore
from .routes import build_router

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


def create_app(cfg: Config | None = None, jobs: JobStore | None = None) -> FastAPI:
    cfg = cfg or load_config(os.environ.get("REDACT_CONFIG", DEFAULT_CONFIG_PATH))
    setup_logging(cfg.logging.level)
    jobs = jobs or JobStore()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        jobs.shutdown()

    app = FastAPI(title="Selective audio redaction", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(cfg.api.cors_origins),
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    app.include_router(build_router(cfg, jobs))
    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    return app


app = create_app()
