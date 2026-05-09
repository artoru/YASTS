# subtitle_translator/application/window_runner.py
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, TypeVar

from domain.models import SubtitleGroup
from .windowing import Window
from .progress import ProgressReporter

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Type alias: a chunk processor returns (translations_by_pos, prompt_tok, pred_tok)
ChunkProcessor = Callable[[list[SubtitleGroup]], Awaitable[tuple[dict[int, str], int, int]]]


def _shrink_list(xs: list[Any], factor: float = 0.5) -> list[Any]:
    if len(xs) <= 1:
        return xs
    return xs[:max(1, int(len(xs) * factor))]


async def run_with_shrinking_focus(
    initial_focus: list[SubtitleGroup],
    process_chunk: ChunkProcessor,
    *,
    max_retries: int,
    shrink_on_retry: bool,
    action: str = "chunk",
) -> tuple[dict[int, str], int, int, int]:
    """
    Try process_chunk with initial_focus. On failure, retry with a shrinking focus.
    If retries exceed max_retries and focus > 1, force-shrink to 1 group and retry from there.
    Raises only when a single-group chunk fails after all retries.

    Returns (translations_by_pos, groups_processed, total_prompt_tokens, total_pred_tokens).
    groups_processed may be less than len(initial_focus) if shrinking occurred.
    """
    cur_focus = list(initial_focus)
    attempt = 0
    total_p = 0
    total_r = 0

    while True:
        try:
            chunk_out, p_tok, r_tok = await process_chunk(cur_focus)
            total_p += p_tok
            total_r += r_tok
            return chunk_out, len(cur_focus), total_p, total_r

        except Exception as e:
            attempt += 1
            cur_ids = [g.id for g in cur_focus]

            if attempt > max_retries:
                if len(cur_focus) > 1:
                    logger.warning(
                        "%s failed after retries; forcing shrink to 1 group. ids=%s..%s err=%r",
                        action,
                        cur_ids[0] if cur_ids else None,
                        cur_ids[-1] if cur_ids else None,
                        e,
                    )
                    cur_focus = cur_focus[:1]
                    attempt = 0
                    continue

                logger.error(
                    "%s single-group FAILED. group_id=%s err=%r",
                    action,
                    cur_ids[0] if cur_ids else None,
                    e,
                )
                raise

            logger.warning(
                "%s failed (attempt %d/%d) ids=%s..%s err=%r",
                action,
                attempt,
                max_retries,
                cur_ids[0] if cur_ids else None,
                cur_ids[-1] if cur_ids else None,
                e,
            )
            if shrink_on_retry and len(cur_focus) > 1:
                cur_focus = _shrink_list(cur_focus, factor=0.5)


async def run_window_loop(
    focus: list[SubtitleGroup],
    make_chunk_processor: Callable[[list[SubtitleGroup]], ChunkProcessor],
    *,
    max_retries: int,
    shrink_on_retry: bool,
    action: str = "chunk",
) -> tuple[dict[int, str], int, int, int]:
    """
    Process all groups in focus in sequential sub-batches using run_with_shrinking_focus.

    make_chunk_processor receives the current focus_remaining list so the caller
    can adjust context (e.g. ctx_after for translation) based on what remains.

    Returns (translations_by_pos, groups_done, total_prompt_tokens, total_pred_tokens).
    """
    result: dict[int, str] = {}
    focus_remaining = list(focus)
    groups_done = 0
    total_p = 0
    total_r = 0

    while focus_remaining:
        cur_focus = list(focus_remaining)
        process_chunk = make_chunk_processor(focus_remaining)

        chunk_out, n_done, p_tok, r_tok = await run_with_shrinking_focus(
            cur_focus,
            process_chunk,
            max_retries=max_retries,
            shrink_on_retry=shrink_on_retry,
            action=action,
        )
        result.update(chunk_out)
        focus_remaining = focus_remaining[n_done:]
        groups_done += n_done
        total_p += p_tok
        total_r += r_tok

    return result, groups_done, total_p, total_r


async def run_windows_concurrent(
    windows: list[Window],
    process_window: Callable[[int, Window], Awaitable[tuple[dict[int, str], int, int, int]]],
    *,
    concurrency: int,
    total_lines: int,
    progress: ProgressReporter,
    label: str = "Progress",
) -> dict[int, str]:
    """
    Execute process_window for each window with bounded concurrency.
    Collects results, updates progress after each window completes.

    process_window(idx, window) -> (translations_by_pos, groups_done, prompt_tok, pred_tok)
    """
    out_by_pos: dict[int, str] = {}
    pos_seen: set[int] = set()
    done_windows = 0
    total_prompt_tok = 0
    total_pred_tok = 0
    start_ts = time.time()

    sem = asyncio.Semaphore(max(1, concurrency))
    total_windows = len(windows)

    async def run_one(idx: int, w: Window):
        async with sem:
            return idx, await process_window(idx, w)

    tasks = [asyncio.create_task(run_one(i, w)) for i, w in enumerate(windows)]

    try:
        for fut in asyncio.as_completed(tasks):
            idx, (res_by_pos, groups_done, win_p, win_r) = await fut

            out_by_pos.update(res_by_pos)
            for p in res_by_pos.keys():
                pos_seen.add(int(p))

            done_windows += 1
            total_prompt_tok += int(win_p)
            total_pred_tok += int(win_r)

            elapsed = max(1e-6, time.time() - start_ts)
            progress.update(
                done_windows=done_windows,
                total_windows=total_windows,
                done_lines=len(pos_seen),
                total_lines=total_lines,
                elapsed_s=elapsed,
                pred_tokens=total_pred_tok,
                label=label,
            )
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    return out_by_pos
