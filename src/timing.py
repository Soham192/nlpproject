"""Per-stage timing. 'How long per minute of audio' is a likely defense question."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Callable, Iterator

log = logging.getLogger("redact")


class StageTimer:
    def __init__(self, enabled: bool = True, on_stage: Callable[[str], None] | None = None) -> None:
        self.enabled = enabled
        self.on_stage = on_stage  # called with the stage name on entry (drives UI progress)
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self.on_stage:
            self.on_stage(name)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.timings[name] = round(self.timings.get(name, 0.0) + dt, 3)
            if self.enabled:
                log.info("stage %-10s %.2fs", name, dt)

    def as_dict(self) -> dict[str, float]:
        out = dict(self.timings)
        out["total"] = round(sum(self.timings.values()), 3)
        return out


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
