# subtitle_translator/domain/models.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubtitleLine:
    position: int
    text: str
    duration_s: float
    cue_index: int
    line_index: int


@dataclass(frozen=True)
class SubtitleGroup:
    id: int
    positions: tuple[int, ...]
    text: str


@dataclass(frozen=True)
class ModelResponseStats:
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
