# subtitle_translator/srt_io.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

_TS_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
    r"(?:\s+.*)?$"  # allow trailing cue settings (e.g., "X1:..", "align:..")
)


@dataclass
class Cue:
    idx: int
    start: str
    end: str
    lines: List[str]


def _split_blocks(text: str) -> List[str]:
    # SRT blocks are separated by one or more blank lines.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip("\n")
    if not text:
        return []
    return re.split(r"\n\s*\n+", text)


def parse_srt(path: str) -> List[Cue]:
    """
    Parse an .srt file into cues.

    Tolerant behavior:
    - Accepts missing/incorrect numeric indices (we'll renumber on write).
    - Keeps timestamps as strings.
    - Keeps cue text lines as-is (trim only line endings).
    - Allows trailing timestamp settings after end time.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    cues: List[Cue] = []
    blocks = _split_blocks(raw)

    for block in blocks:
        lines = [ln.rstrip("\n") for ln in block.split("\n")]
        # Remove leading/trailing empty lines inside block just in case
        while lines and lines[0].strip() == "":
            lines.pop(0)
        while lines and lines[-1].strip() == "":
            lines.pop()
        if not lines:
            continue

        # Try to detect index line
        i = 0
        idx: Optional[int] = None
        if lines[0].strip().isdigit():
            idx = int(lines[0].strip())
            i = 1

        if i >= len(lines):
            continue

        m = _TS_RE.match(lines[i].strip())
        if not m:
            # Not a valid cue block; skip (tolerant)
            continue

        start = m.group("start")
        end = m.group("end")
        i += 1

        text_lines = lines[i:] if i < len(lines) else [""]
        if idx is None:
            idx = len(cues) + 1

        cues.append(Cue(idx=idx, start=start, end=end, lines=text_lines))

    return cues


def write_srt(path: str, cues: List[Cue]) -> None:
    """
    Write cues to .srt.

    - Renumbers cues sequentially (1..N).
    - Writes timestamps exactly as stored.
    - Preserves cue line breaks.
    """
    out: List[str] = []
    for n, c in enumerate(cues, start=1):
        out.append(str(n))
        out.append(f"{c.start} --> {c.end}")
        if c.lines:
            out.extend(c.lines)
        else:
            out.append("")
        out.append("")  # blank line between cues

    text = "\n".join(out).rstrip() + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
