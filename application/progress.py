# subtitle_translator/application/progress.py
from __future__ import annotations

import logging
from typing import Protocol


class ProgressReporter(Protocol):
    def update(
        self,
        *,
        done_windows: int,
        total_windows: int,
        done_lines: int,
        total_lines: int,
        elapsed_s: float,
        pred_tokens: int,
        label: str,
    ) -> None: ...


def format_eta(seconds: float) -> str:
    s = int(max(0, round(seconds)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


class LoggingProgressReporter:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or logging.getLogger(__name__)

    def update(
        self,
        *,
        done_windows: int,
        total_windows: int,
        done_lines: int,
        total_lines: int,
        elapsed_s: float,
        pred_tokens: int,
        label: str,
    ) -> None:
        lps = done_lines / max(1e-6, elapsed_s)
        lines_remaining = max(0, total_lines - done_lines)
        eta_s = (lines_remaining / lps) if lps > 0 else 0.0
        gen_tps = (pred_tokens / max(1e-6, elapsed_s)) if pred_tokens > 0 else 0.0

        self._log.info(
            "%s: windows %d/%d (%.1f%%) | lines %d/%d (%.1f%%) | ~%.1f tok/s | ETA ~%s",
            label,
            done_windows,
            total_windows,
            100.0 * done_windows / max(1, total_windows),
            done_lines,
            total_lines,
            100.0 * done_lines / max(1, total_lines),
            gen_tps,
            format_eta(eta_s),
        )
