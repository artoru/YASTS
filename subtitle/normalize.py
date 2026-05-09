# subtitle_translator/subtitle/normalize.py
from __future__ import annotations

from typing import Any

from domain.models import SubtitleLine
from .srt_io import Cue


def _parse_srt_timecode_to_seconds(s: str) -> float:
    s = (s or "").strip().replace(".", ",")
    hh, mm, rest = s.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + (int(ms) / 1000.0)


def _to_seconds(t: Any) -> float:
    if isinstance(t, (int, float)):
        return float(t)
    if hasattr(t, "total_seconds"):
        return float(t.total_seconds())
    if isinstance(t, str):
        return _parse_srt_timecode_to_seconds(t)
    raise TypeError(f"Unsupported time type: {type(t)}")


def cues_to_items(cues: list[Cue]) -> list[SubtitleLine]:
    """
    Flatten SRT cues into SubtitleLine list.

    Each SubtitleLine carries its position, text, timing, and back-reference
    (cue_index, line_index) so translated text can be placed back into cues.
    """
    items: list[SubtitleLine] = []
    pos = 1
    for ci, cue in enumerate(cues):
        lines = cue.lines if cue.lines else [""]
        try:
            dur = max(0.01, _to_seconds(cue.end) - _to_seconds(cue.start))
        except Exception:
            dur = 1.0
        for li, ln in enumerate(lines):
            items.append(SubtitleLine(
                position=pos,
                text=ln.strip(),
                duration_s=dur,
                cue_index=ci,
                line_index=li,
            ))
            pos += 1
    return items


def _wrap_to_n_lines(text: str, *, max_chars: int, n_lines: int) -> list[str]:
    n_lines = max(1, int(n_lines))
    max_chars = int(max_chars)
    s = " ".join((text or "").split()).strip()
    if not s:
        return [""] * n_lines
    if s == "♪":
        return ["♪"] + ([""] * (n_lines - 1))
    if max_chars <= 0:
        return [s] + ([""] * (n_lines - 1))

    words = s.split()
    lines: list[str] = []
    wi = 0
    for li in range(n_lines):
        if wi >= len(words):
            lines.append("")
            continue
        if li == n_lines - 1:
            lines.append(" ".join(words[wi:]))
            wi = len(words)
            continue
        cur = ""
        while wi < len(words):
            w = words[wi]
            cand = w if not cur else f"{cur} {w}"
            if len(cand) <= max_chars:
                cur = cand
                wi += 1
                continue
            if cur:
                break
            cur = w
            wi += 1
            break
        lines.append(cur)
    return lines


def _is_probably_lyric_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if s in {"*", "♪", "♫", "♬", "♩"}:
        return True
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        return True
    return s[0] in "*♪♫♬♩" or s[-1] in "*♪♫♬♩"


def items_to_cues(
    cues: list[Cue],
    translated_by_pos: dict[int, str],
    items: list[SubtitleLine],
    *,
    max_line_chars: int = 42,
    reflow: bool = True,
) -> list[Cue]:
    """
    Apply translated lines back onto cues using position data from items.

    - Preserves original timestamps.
    - By default performs cue-level reflow to stay within original line count.
    - Falls back to original text for any position missing from translated_by_pos.
    """
    out: list[Cue] = []
    for c in cues:
        out.append(Cue(idx=c.idx, start=c.start, end=c.end, lines=list(c.lines) if c.lines else [""]))

    for item in items:
        ci, li = item.cue_index, item.line_index
        if ci < 0 or ci >= len(out):
            continue
        if li < 0 or li >= len(out[ci].lines):
            continue
        if item.position in translated_by_pos:
            out[ci].lines[li] = (translated_by_pos[item.position] or "").strip()

    if reflow:
        for cue in out:
            n_lines = len(cue.lines) if cue.lines else 1
            if any(_is_probably_lyric_line(ln) for ln in cue.lines):
                continue
            combined = " ".join(
                (ln or "").strip() for ln in (cue.lines or []) if (ln or "").strip()
            ).strip()
            if not combined:
                cue.lines = [""] * n_lines
                continue
            cue.lines = _wrap_to_n_lines(combined, max_chars=max_line_chars, n_lines=n_lines)

    return out
