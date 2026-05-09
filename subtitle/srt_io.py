# subtitle_translator/subtitle/srt_io.py
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

_VENDOR_PATH = str(Path(__file__).parent.parent / "vendor")
if _VENDOR_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_PATH)

from charset_normalizer import from_bytes

_TS_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
    r"(?:\s+.*)?$"
)


@dataclass
class Cue:
    idx: int
    start: str
    end: str
    lines: list[str]


def _split_blocks(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip("\n")
    if not text:
        return []
    return re.split(r"\n\s*\n+", text)


def _to_text(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, bytearray)):
        return bytes(x).decode("utf-8", errors="replace")
    return str(x)


def _read_text(path: str) -> str:
    data = Path(path).read_bytes()
    match = from_bytes(data).best()
    if match is None:
        return data.decode("utf-8", errors="replace")
    return _to_text(match.output())


def parse_srt(path: str) -> list[Cue]:
    raw = _read_text(path)
    cues: list[Cue] = []
    blocks = _split_blocks(raw)

    for block in blocks:
        lines = [ln.rstrip("\n") for ln in block.split("\n")]
        while lines and lines[0].strip() == "":
            lines.pop(0)
        while lines and lines[-1].strip() == "":
            lines.pop()
        if not lines:
            continue

        i = 0
        idx: int | None = None
        if lines[0].strip().isdigit():
            idx = int(lines[0].strip())
            i = 1

        if i >= len(lines):
            continue

        m = _TS_RE.match(lines[i].strip())
        if not m:
            continue

        start = m.group("start")
        end = m.group("end")
        i += 1

        text_lines = lines[i:] if i < len(lines) else [""]
        if idx is None:
            idx = len(cues) + 1

        cues.append(Cue(idx=idx, start=start, end=end, lines=text_lines))

    return cues


def write_srt(path: str, cues: list[Cue]) -> None:
    out: list[str] = []
    for n, c in enumerate(cues, start=1):
        out.append(str(n))
        out.append(f"{c.start} --> {c.end}")
        if c.lines:
            out.extend(c.lines)
        else:
            out.append("")
        out.append("")

    text = "\n".join(out).rstrip() + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
