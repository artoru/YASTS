# subtitle_translator/grouping.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .config import Settings

# Same regex shapes as your proxy
PUNCT_END_RE = re.compile(r"[.!?…]+[\"')\]]*$")
MULTI_SPEAKER_RE = re.compile(r"\s-\s")  # detects " - " inside a line


def _is_music_only(line: str) -> bool:
    return line.strip() == "♪"


def _starts_dash(line: str) -> bool:
    return line.lstrip().startswith("-")


def _contains_multi_speaker(line: str) -> bool:
    """
    True for lines like:
      "- Objection... - Overruled..."
    These are dangerous to merge with neighbors unless the next line clearly continues.
    """
    s = line.strip()
    if s.startswith("-"):
        s = s[1:]
    return bool(MULTI_SPEAKER_RE.search(s))


def _ends_phrase(line: str) -> bool:
    return bool(PUNCT_END_RE.search(line.strip()))


def _merge_tiny_groups(groups: List[Dict[str, Any]], settings: Settings) -> List[Dict[str, Any]]:
    """
    Merge tiny fragment groups into the previous group, but NEVER:
    - merge music
    - merge groups that start with dash
    - merge multi-speaker lines
    - merge across a phrase boundary (prev ends with punctuation)
    """
    merged: List[Dict[str, Any]] = []

    for g in groups:
        text = (g.get("text") or "").strip()

        # Keep music as-is
        if text == "♪":
            merged.append(g)
            continue

        # Keep dialogue starts and multi-speaker lines as-is
        if _starts_dash(text) or _contains_multi_speaker(text):
            merged.append(g)
            continue

        words = [w for w in text.split() if w]
        is_tiny = (len(text) < settings.min_group_text_chars) or (
            len(words) < settings.min_group_words
        )

        # If there's nothing to merge into, just append
        if not merged:
            merged.append(g)
            continue

        # Do NOT merge across sentence/phrase boundary
        prev_text = (merged[-1].get("text") or "").strip()
        if _ends_phrase(prev_text):
            merged.append(g)
            continue

        # Merge only if tiny
        if is_tiny:
            prev = merged[-1]
            prev["positions"].extend(g["positions"])
            prev["text"] = (prev["text"].rstrip() + " " + text).strip()
        else:
            merged.append(g)

    # Re-number group_id sequentially
    for i, gg in enumerate(merged, start=1):
        gg["group_id"] = i

    return merged


def group_subtitles(items: List[Dict[str, Any]], *, settings: Settings) -> List[Dict[str, Any]]:
    """
    Group subtitle lines into sentence-ish translation units.

    Key behaviors:
    - Keep dash-dialogue lines WITH their immediate non-dash continuation lines.
    - Isolate multi-speaker-in-one-line subtitles (e.g. "- A... - B...") as singletons,
      BUT only if line is self-contained OR next line looks like a new turn.
    - Always isolate music '♪'.
    - Flush on phrase-ending punctuation.
    - Cap groups by max_group_lines / max_group_chars.
    - Merge tiny fragments into previous group when safe.

    Input:
      items = [{"Position": int, "Line": str}, ...]
    Output groups:
      [{"group_id": int, "positions":[...], "text": "..."}]
    """
    groups: List[Dict[str, Any]] = []
    cur_positions: List[int] = []
    cur_parts: List[str] = []
    cur_chars = 0
    gid = 1
    prev_line: Optional[str] = None

    def flush() -> None:
        nonlocal gid, cur_positions, cur_parts, cur_chars
        if not cur_positions:
            return
        groups.append(
            {
                "group_id": gid,
                "positions": cur_positions[:],
                "text": " ".join(p.strip() for p in cur_parts).strip(),
            }
        )
        gid += 1
        cur_positions = []
        cur_parts = []
        cur_chars = 0

    # Indexed loop so we can peek at next line
    for idx, it in enumerate(items):
        pos = int(it["Position"])
        line = (it.get("Line") or "").strip()

        next_line = ""
        if idx + 1 < len(items):
            next_line = (items[idx + 1].get("Line") or "").strip()

        # 1) Always isolate music
        if _is_music_only(line):
            flush()
            groups.append({"group_id": gid, "positions": [pos], "text": "♪"})
            gid += 1
            prev_line = line
            continue

        # 2) Multi-speaker-in-one-line isolation (conditional, language-agnostic)
        if _contains_multi_speaker(line):
            next_is_new_turn = _starts_dash(next_line) or _is_music_only(next_line) or (next_line == "")
            self_contained = _ends_phrase(line)
            if self_contained or next_is_new_turn:
                flush()
                groups.append({"group_id": gid, "positions": [pos], "text": line})
                gid += 1
                prev_line = line
                continue
            # else: fall through and allow grouping (line likely continues)

        # 3) Decide whether to flush BEFORE adding this line
        if cur_positions:
            # Start a new group when a NEW dash-dialogue begins (new speaker turn),
            # but allow dash + continuation lines to remain together.
            if _starts_dash(line) and not _starts_dash(prev_line or ""):
                flush()

            # Cap group size
            if (len(cur_positions) >= settings.max_group_lines) or (
                (cur_chars + len(line) + 1) > settings.max_group_chars
            ):
                flush()

        # 4) Add line to current group
        cur_positions.append(pos)
        cur_parts.append(line)
        cur_chars += len(line) + 1

        # 5) Flush on phrase-ending punctuation
        if _ends_phrase(line):
            flush()

        prev_line = line

    flush()

    # 6) Merge tiny fragments when safe
    groups = _merge_tiny_groups(groups, settings=settings)

    return groups
