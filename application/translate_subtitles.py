# subtitle_translator/application/translate_subtitles.py
from __future__ import annotations

import logging
from pathlib import Path

from config import Settings
from domain.errors import TranslationError
from domain.models import SubtitleGroup
from llm.client import LlmClient
from services.translator import Translator
from subtitle.grouping import build_1to1_groups, group_subtitles
from subtitle.normalize import cues_to_items, items_to_cues
from subtitle.srt_io import parse_srt, write_srt
from .progress import LoggingProgressReporter
from .window_runner import run_window_loop, run_windows_concurrent
from .windowing import Window, make_windows

logger = logging.getLogger(__name__)


def _translate_window(
    window: Window,
    *,
    translator: Translator,
    duration_by_pos: dict[int, float],
    settings: Settings,
):
    """
    Returns a coroutine that translates window.focus, adjusting ctx_after
    dynamically as focus shrinks during retries.
    """
    ctx_before = window.context_before
    ctx_after_base = window.context_after

    def make_chunk_processor(focus_remaining: list[SubtitleGroup]):
        async def process_chunk(focus: list[SubtitleGroup]) -> tuple[dict[int, str], int, int]:
            remainder = focus_remaining[len(focus):]
            ctx_after = [] if len(focus) == 1 else (remainder + ctx_after_base)
            return await translator.translate_chunk(
                focus, ctx_before, ctx_after,
                duration_by_pos=duration_by_pos,
            )
        return process_chunk

    return run_window_loop(
        window.focus,
        make_chunk_processor,
        max_retries=settings.max_retries_per_window,
        shrink_on_retry=settings.shrink_focus_on_retry,
        action="translate",
    )


class TranslateSubtitles:
    """
    Top-level use case: read SRT → group → translate → write SRT.
    """

    def __init__(self, client: LlmClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def run(
        self,
        input_path: Path,
        output_path: Path,
        *,
        src_lang: str,
        tgt_lang: str,
    ) -> None:
        cues = parse_srt(str(input_path))
        if not cues:
            raise TranslationError(f"No cues parsed from {input_path}")

        items = cues_to_items(cues)
        duration_by_pos = {it.position: it.duration_s for it in items}

        if self._settings.use_phrase_grouping:
            groups = group_subtitles(items, settings=self._settings)
        else:
            groups = build_1to1_groups(items)

        translator = Translator(self._client, self._settings, src_lang, tgt_lang)
        translator.set_items(items)

        windows = make_windows(
            groups,
            self._settings.max_window_chars,
            self._settings.context_pre_groups,
            self._settings.context_post_groups,
        )

        progress = LoggingProgressReporter(logger)

        async def process_window(idx: int, w: Window) -> tuple[dict[int, str], int, int, int]:
            return await _translate_window(
                w,
                translator=translator,
                duration_by_pos=duration_by_pos,
                settings=self._settings,
            )

        out_by_pos = await run_windows_concurrent(
            windows,
            process_window,
            concurrency=self._settings.concurrency,
            total_lines=len(items),
            progress=progress,
            label="Progress",
        )

        out_cues = items_to_cues(
            cues,
            out_by_pos,
            items,
            max_line_chars=self._settings.split_max_line_chars,
            reflow=True,
        )
        write_srt(str(output_path), out_cues)
