# subtitle_translator/subtitle/grouping.py
from __future__ import annotations

import re

from config import Settings
from domain.models import SubtitleGroup, SubtitleLine

PUNCT_END_RE = re.compile(r"[.!?…]+[\"')\]]*$")
MULTI_SPEAKER_RE = re.compile(r"\s-\s")

MUSIC_MARKERS = frozenset({"*", "♪", "♫", "♬", "♩"})


def _strip_edge_music_markers(line: str) -> str:
    return (line or "").strip().strip("".join(MUSIC_MARKERS)).strip()


def _is_bare_music_marker(line: str) -> bool:
    return (line or "").strip() in MUSIC_MARKERS


def _starts_music_marker(line: str) -> bool:
    s = (line or "").lstrip()
    return bool(s) and s[0] in MUSIC_MARKERS


def _ends_music_marker(line: str) -> bool:
    s = (line or "").rstrip()
    return bool(s) and s[-1] in MUSIC_MARKERS


def _is_probably_lyric_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if _is_bare_music_marker(s):
        return True
    return _starts_music_marker(s) or _ends_music_marker(s)


def _is_open_lyric_group(parts: list[str]) -> bool:
    if not parts:
        return False
    return _starts_music_marker(parts[0]) or _is_bare_music_marker(parts[0])


def _starts_dash(line: str) -> bool:
    return line.lstrip().startswith("-")


def _contains_multi_speaker(line: str) -> bool:
    s = line.strip()
    if s.startswith("-"):
        s = s[1:]
    return bool(MULTI_SPEAKER_RE.search(s))


def _is_hi_subtitle(line: str) -> bool:
    s = (line or "").strip()
    return len(s) >= 2 and (
        (s[0] == "(" and s[-1] == ")") or
        (s[0] == "[" and s[-1] == "]")
    )


def _ends_phrase(line: str) -> bool:
    return bool(PUNCT_END_RE.search(line.strip()))


def _merge_tiny_groups(groups: list[SubtitleGroup], settings: Settings) -> list[SubtitleGroup]:
    merged: list[SubtitleGroup] = []

    for g in groups:
        text = g.text.strip()

        if _is_probably_lyric_line(text):
            merged.append(g)
            continue

        if _is_hi_subtitle(text):
            merged.append(g)
            continue

        if _starts_dash(text) or _contains_multi_speaker(text):
            merged.append(g)
            continue

        words = [w for w in text.split() if w]
        is_tiny = (len(text) < settings.min_group_text_chars) or (
            len(words) < settings.min_group_words
        )

        if not merged:
            merged.append(g)
            continue

        prev_text = merged[-1].text.strip()
        if _is_probably_lyric_line(prev_text) or _is_hi_subtitle(prev_text):
            merged.append(g)
            continue

        if _ends_phrase(prev_text):
            merged.append(g)
            continue

        if is_tiny:
            prev = merged[-1]
            merged[-1] = SubtitleGroup(
                id=prev.id,
                positions=prev.positions + g.positions,
                text=(prev.text.rstrip() + " " + text).strip(),
            )
        else:
            merged.append(g)

    # Re-number sequentially
    return [SubtitleGroup(id=i + 1, positions=g.positions, text=g.text) for i, g in enumerate(merged)]


def group_subtitles(items: list[SubtitleLine], *, settings: Settings) -> list[SubtitleGroup]:
    """
    Group subtitle lines into sentence-ish translation units.

    - Isolates bare music markers, hearing-impaired cues, and multi-speaker lines.
    - Attaches lyric continuation lines to open lyric blocks.
    - Flushes on phrase-ending punctuation.
    - Caps groups by max_group_lines / max_group_chars.
    - Merges tiny fragments when safe.
    """
    groups: list[SubtitleGroup] = []
    cur_positions: list[int] = []
    cur_parts: list[str] = []
    cur_chars = 0
    gid = 1
    prev_line: str | None = None

    def flush() -> None:
        nonlocal gid, cur_positions, cur_parts, cur_chars
        if not cur_positions:
            return
        groups.append(SubtitleGroup(
            id=gid,
            positions=tuple(cur_positions),
            text=" ".join(p.strip() for p in cur_parts).strip(),
        ))
        gid += 1
        cur_positions = []
        cur_parts = []
        cur_chars = 0

    def add(pos: int, line: str) -> None:
        nonlocal cur_chars
        cur_positions.append(pos)
        cur_parts.append(line)
        cur_chars += len(line) + 1

    for idx, it in enumerate(items):
        pos = it.position
        line = it.text

        next_line = ""
        if idx + 1 < len(items):
            next_line = items[idx + 1].text

        # 1) Bare music markers
        if _is_bare_music_marker(line):
            flush()
            groups.append(SubtitleGroup(id=gid, positions=(pos,), text=line))
            gid += 1
            prev_line = line
            continue

        # 2) Hearing-impaired cues
        if _is_hi_subtitle(line):
            flush()
            groups.append(SubtitleGroup(id=gid, positions=(pos,), text=line))
            gid += 1
            prev_line = line
            continue

        # 3) Lyric/music block start
        if _starts_music_marker(line):
            flush()
            add(pos, line)
            if _ends_music_marker(line) and _strip_edge_music_markers(line):
                flush()
            prev_line = line
            continue

        # 4) Lyric continuation
        if cur_positions and _is_open_lyric_group(cur_parts):
            if (len(cur_positions) >= settings.max_group_lines) or (
                (cur_chars + len(line) + 1) > settings.max_group_chars
            ):
                flush()
            add(pos, line)
            next_starts_lyric = _starts_music_marker(next_line) or _is_bare_music_marker(next_line)
            if _ends_music_marker(line) or next_starts_lyric or next_line == "":
                flush()
            prev_line = line
            continue

        # 5) Multi-speaker isolation
        if _contains_multi_speaker(line):
            next_is_new_turn = (
                _starts_dash(next_line)
                or _is_probably_lyric_line(next_line)
                or (next_line == "")
            )
            if _ends_phrase(line) or next_is_new_turn:
                flush()
                groups.append(SubtitleGroup(id=gid, positions=(pos,), text=line))
                gid += 1
                prev_line = line
                continue

        # 6) Flush before adding if needed
        if cur_positions:
            if _starts_dash(line) and not _starts_dash(prev_line or ""):
                flush()
            if (len(cur_positions) >= settings.max_group_lines) or (
                (cur_chars + len(line) + 1) > settings.max_group_chars
            ):
                flush()

        add(pos, line)

        if _ends_phrase(line):
            flush()

        prev_line = line

    flush()
    return _merge_tiny_groups(groups, settings=settings)


def build_1to1_groups(items: list[SubtitleLine]) -> list[SubtitleGroup]:
    return [
        SubtitleGroup(id=i + 1, positions=(it.position,), text=it.text)
        for i, it in enumerate(items)
    ]
