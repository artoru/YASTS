# subtitle_translator/normalize.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .srt_io import Cue


@dataclass(frozen=True)
class PosRef:
    cue_index: int      # index into the cues list (0-based)
    line_index: int     # index into cue.lines (0-based)


def _parse_srt_timecode_to_seconds(s: str) -> float:
    """
    Parse SRT timecode 'HH:MM:SS,mmm' into seconds.
    Only used as a fallback if Cue.start/end are strings.
    """
    s = (s or "").strip()
    # tolerate '.' as ms separator too
    s = s.replace(".", ",")
    hh, mm, rest = s.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + (int(ms) / 1000.0)


def _to_seconds(t: Any) -> float:
    """
    Convert Cue start/end to seconds.
    Supports float/int seconds, datetime.timedelta-like, or SRT timecode strings.
    """
    if isinstance(t, (int, float)):
        return float(t)
    if hasattr(t, "total_seconds"):
        return float(t.total_seconds())  # type: ignore[attr-defined]
    if isinstance(t, str):
        return _parse_srt_timecode_to_seconds(t)
    raise TypeError(f"Unsupported time type: {type(t)}")


def cues_to_items(cues: List[Cue]) -> Tuple[List[Dict[str, Any]], Dict[int, PosRef], Dict[int, float]]:
    """
    Flatten SRT cues into the internal "items" format used by the translator:

      items = [{"Position": int, "Line": str}, ...]

    We keep mappings so we can restore translated lines back into the original cues.

    Returns:
      items: list of dicts in Lingarr-like schema
      pos_map: Position -> PosRef(cue_index, line_index)
      duration_by_pos: Position -> duration_seconds (per-cue duration copied to each line position)
    """
    items: List[Dict[str, Any]] = []
    pos_map: Dict[int, PosRef] = {}
    duration_by_pos: Dict[int, float] = {}

    pos = 1
    for ci, cue in enumerate(cues):
        # Ensure at least one line exists so every cue is representable.
        lines = cue.lines if cue.lines else [""]

        # Duration for this cue (seconds). Clamp to a small epsilon so weights behave.
        try:
            dur = max(0.01, _to_seconds(cue.end) - _to_seconds(cue.start))
        except Exception:
            dur = 1.0

        for li, ln in enumerate(lines):
            text = ln.strip()
            items.append({"Position": pos, "Line": text})
            pos_map[pos] = PosRef(cue_index=ci, line_index=li)
            duration_by_pos[pos] = dur
            pos += 1

    return items, pos_map, duration_by_pos


def _wrap_to_n_lines(text: str, *, max_chars: int, n_lines: int) -> List[str]:
    """
    Wrap text into exactly n_lines (or as close as possible), keeping each line <= max_chars
    when feasible. The last line receives remaining words (may exceed max_chars only if unavoidable).
    """
    n_lines = max(1, int(n_lines))
    max_chars = int(max_chars)

    s = " ".join((text or "").split()).strip()
    if not s:
        return [""] * n_lines

    # Special-case music
    if s == "♪":
        return ["♪"] + ([""] * (n_lines - 1))

    if max_chars <= 0:
        return [s] + ([""] * (n_lines - 1))

    words = s.split()
    lines: List[str] = []
    wi = 0

    for li in range(n_lines):
        if wi >= len(words):
            lines.append("")
            continue

        # Last line: dump everything remaining
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

            # single long word
            cur = w
            wi += 1
            break

        lines.append(cur)

    if len(lines) < n_lines:
        lines.extend([""] * (n_lines - len(lines)))
    elif len(lines) > n_lines:
        lines = lines[:n_lines]

    return lines


def items_to_cues(
    cues: List[Cue],
    translated_by_pos: Dict[int, str],
    pos_map: Dict[int, PosRef],
    *,
    max_line_chars: int = 42,
    reflow: bool = True,
) -> List[Cue]:
    """
    Apply translated lines back onto cues using pos_map.

    - Preserves original timestamps.
    - By default, performs cue-level reflow so each cue uses <= original line count.
    - If a position is missing in translated_by_pos, falls back to original text.
    """
    out: List[Cue] = []
    for c in cues:
        out.append(Cue(idx=c.idx, start=c.start, end=c.end, lines=list(c.lines) if c.lines else [""]))

    # 1) Apply position-level translations (no wrapping here)
    for pos, ref in pos_map.items():
        ci, li = ref.cue_index, ref.line_index
        if ci < 0 or ci >= len(out):
            continue
        if li < 0 or li >= len(out[ci].lines):
            continue

        if pos in translated_by_pos:
            out[ci].lines[li] = (translated_by_pos[pos] or "").strip()

    # 2) Cue-level reflow to keep within original line count
    if reflow:
        for cue in out:
            n_lines = len(cue.lines) if cue.lines else 1
            combined = " ".join((ln or "").strip() for ln in (cue.lines or []) if (ln or "").strip()).strip()

            if not combined:
                cue.lines = [""] * n_lines
                continue

            cue.lines = _wrap_to_n_lines(combined, max_chars=max_line_chars, n_lines=n_lines)

    return out
