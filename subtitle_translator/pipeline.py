# subtitle_translator/pipeline.py
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import Settings
from .json_parse import parse_model_json, get_returned_ids_and_missing
from .llama_client import call_llama_with_retries
from .prompts import build_system_prompt, render_prompt, render_user_json
from .splitback import split_group_translation_to_positions
from .validate import validate_focus_by_group_ids
from .windowing import Window, make_windows

logger = logging.getLogger(__name__)


def _input_by_pos(items: List[Dict[str, Any]]) -> Dict[int, str]:
    return {int(it["Position"]): (it.get("Line") or "") for it in items}


def _trunc(s: str, n: int = 800) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + f" ... [truncated {len(s) - n} chars]"


def _rough_token_estimate(chars: int) -> int:
    return max(1, int(chars / 4))


def format_eta(seconds: float) -> str:
    s = int(max(0, round(seconds)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _shrink_list(xs: List[Any], factor: float = 0.5) -> List[Any]:
    if len(xs) <= 1:
        return xs
    new_len = max(1, int(len(xs) * factor))
    return xs[:new_len]

def _extract_llama_token_stats(raw: Any) -> tuple[int, int, float] | None:
    if not isinstance(raw, dict):
        return None

    t = raw.get("timings")
    if isinstance(t, dict):
        prompt_n = t.get("prompt_n") or t.get("prompt_tokens") or t.get("n_prompt_tokens")
        pred_n = t.get("predicted_n") or t.get("predicted_tokens") or t.get("n_predicted_tokens")

        prompt_ms = t.get("prompt_ms") or t.get("prompt_eval_ms") or t.get("prompt_eval_time_ms")
        pred_ms = t.get("predicted_ms") or t.get("eval_ms") or t.get("eval_time_ms")

        if isinstance(prompt_n, int) and isinstance(pred_n, int):
            total_ms = 0.0
            if isinstance(prompt_ms, (int, float)):
                total_ms += float(prompt_ms)
            if isinstance(pred_ms, (int, float)):
                total_ms += float(pred_ms)
            if total_ms > 0:
                return int(prompt_n), int(pred_n), total_ms / 1000.0

    prompt_n = raw.get("tokens_evaluated") or raw.get("prompt_tokens")
    pred_n = raw.get("tokens_predicted") or raw.get("predicted_tokens")
    total_s = raw.get("total_time_s") or raw.get("total_time")

    if isinstance(prompt_n, int) and isinstance(pred_n, int) and isinstance(total_s, (int, float)) and float(total_s) > 0:
        return int(prompt_n), int(pred_n), float(total_s)

    return None


async def _translate_focus_window(
    ctx_before: List[Dict[str, Any]],
    focus: List[Dict[str, Any]],
    ctx_after: List[Dict[str, Any]],
    *,
    items: List[Dict[str, Any]],
    settings: Settings,
    system_prompt: str,
    duration_by_pos: Optional[Dict[int, float]] = None,
) -> Tuple[Dict[int, str], int, int]:
    in_by_pos = _input_by_pos(items)

    payload: List[Dict[str, Any]] = []
    expected_focus_ids: List[int] = []

    def add_groups(gs: List[Dict[str, Any]], role: str) -> None:
        for g in gs:
            gid = int(g["group_id"])
            text = (g.get("text") or "").strip()
            payload.append({"group_id": gid, "role": role, "text": text})
            if role == "translate":
                expected_focus_ids.append(gid)

    add_groups(ctx_before, "context")
    add_groups(focus, "translate")
    add_groups(ctx_after, "context")

    user_json = render_user_json(payload)
    prompt = render_prompt(settings.prompt_template, system_prompt, user_json)

    if logger.isEnabledFor(logging.DEBUG):
        focus_ids = [int(g["group_id"]) for g in focus]
        focus_text = " | ".join(_trunc((g.get("text") or "").strip(), 120) for g in focus[:3])
        logger.debug(
            "LLM request: tpl=%s ctxb=%d focus=%d ctxa=%d | focus_ids=%s..%s | prompt_chars=%d (~%d tok) user_json_chars=%d (~%d tok) | focus_text=%s",
            settings.prompt_template,
            len(ctx_before),
            len(focus),
            len(ctx_after),
            focus_ids[0] if focus_ids else None,
            focus_ids[-1] if focus_ids else None,
            len(prompt),
            _rough_token_estimate(len(prompt)),
            len(user_json),
            _rough_token_estimate(len(user_json)),
            focus_text,
        )

    raw, content = await call_llama_with_retries(prompt, settings=settings, retries=2)

    prompt_tok = 0
    pred_tok = 0
    stats = _extract_llama_token_stats(raw)
    if stats:
        prompt_tok, pred_tok, total_s = stats
        if logger.isEnabledFor(logging.DEBUG):
            tot = prompt_tok + pred_tok
            tps = (tot / total_s) if total_s > 0 else 0.0
            logger.debug("LLM timing: prompt_tok=%d pred_tok=%d total=%.2fs tok/s=%.2f", prompt_tok, pred_tok, total_s, tps)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("LLM raw keys=%s", list(raw.keys()) if isinstance(raw, dict) else type(raw))
        logger.debug("LLM content (head): %s", _trunc(content, 1200))

    try:
        model_json = parse_model_json(content)
    except Exception as e:
        logger.error(
            "JSON parse failed. focus_ids=%s prompt_chars=%d content_head=%s err=%r",
            [int(g["group_id"]) for g in focus],
            len(prompt),
            _trunc(content, 2000),
            e,
        )
        raise

    try:
        returned_ids, missing_ids, focus_lines_by_gid = get_returned_ids_and_missing(
            model_json,
            expected_focus_ids,
        )

        if missing_ids:
            raise ValueError(f"missing focus group_ids={missing_ids}")

    except Exception as e:
        # Use language-agnostic returned-id extraction for logging too
        returned_ids, _, _ = get_returned_ids_and_missing(
            model_json,
            expected_focus_ids,
        )
        logger.error(
            "Validation failed. expected_focus_ids=%s returned_ids=%s content_head=%s err=%r",
            sorted(expected_focus_ids),
            returned_ids,
            _trunc(content, 2000),
            e,
        )
        raise

    out_by_pos: Dict[int, str] = {}
    focus_by_gid = {int(g["group_id"]): g for g in focus}

    for gid, tr_text in focus_lines_by_gid.items():
        g = focus_by_gid.get(gid)
        if not g:
            continue
        positions = [int(p) for p in (g.get("positions") or [])]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Split-back gid=%d positions=%s tr_head=%s", gid, positions, _trunc(tr_text, 200))
        out_by_pos.update(
            split_group_translation_to_positions(
                positions,
                tr_text,
                in_by_pos,
                settings=settings,
                duration_by_pos=duration_by_pos,
            )
        )

    return out_by_pos, prompt_tok, pred_tok


async def _translate_entire_window(
    window: Window,
    *,
    items: List[Dict[str, Any]],
    settings: Settings,
    system_prompt: str,
    duration_by_pos: Optional[Dict[int, float]] = None,
) -> Tuple[Dict[int, str], int, int, int]:
    ctx_before_base = window.context_before
    ctx_after_base = window.context_after

    out_by_pos: Dict[int, str] = {}
    focus_remaining = list(window.focus)
    translated_focus_groups = 0

    win_prompt_tok = 0
    win_pred_tok = 0

    while focus_remaining:
        cur_focus = list(focus_remaining)
        attempt = 0

        while True:
            remainder_after = focus_remaining[len(cur_focus):]
            ctx_after = remainder_after + ctx_after_base

            try:
                chunk_out, p_tok, r_tok = await _translate_focus_window(
                    ctx_before=ctx_before_base,
                    focus=cur_focus,
                    ctx_after=ctx_after,
                    items=items,
                    settings=settings,
                    system_prompt=system_prompt,
                    duration_by_pos=duration_by_pos,
                )

                win_prompt_tok += int(p_tok)
                win_pred_tok += int(r_tok)

                out_by_pos.update(chunk_out)
                focus_remaining = focus_remaining[len(cur_focus):]
                translated_focus_groups += len(cur_focus)
                break

            except Exception as e:
                attempt += 1
                cur_ids = [int(g["group_id"]) for g in cur_focus]

                if attempt > settings.max_retries_per_window:
                    if len(cur_focus) > 1:
                        logger.warning(
                            "Chunk failed after retries; forcing shrink to 1 group. ids=%s..%s err=%r",
                            cur_ids[0] if cur_ids else None,
                            cur_ids[-1] if cur_ids else None,
                            e,
                        )
                        cur_focus = cur_focus[:1]
                        attempt = 0
                        continue

                    logger.error(
                        "Single-group chunk FAILED. group_id=%s text=%s err=%r",
                        cur_ids[0] if cur_ids else None,
                        _trunc((cur_focus[0].get("text") or "").strip() if cur_focus else "", 500),
                        e,
                    )
                    raise

                logger.warning(
                    "Chunk failed (attempt %d/%d) ids=%s..%s err=%r",
                    attempt,
                    settings.max_retries_per_window,
                    cur_ids[0] if cur_ids else None,
                    cur_ids[-1] if cur_ids else None,
                    e,
                )

                if settings.shrink_focus_on_retry and len(cur_focus) > 1:
                    cur_focus = _shrink_list(cur_focus, factor=0.5)

    return out_by_pos, translated_focus_groups, win_prompt_tok, win_pred_tok


async def translate_items_with_windows(
    *,
    items: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
    settings: Settings,
    src_lang: str,
    tgt_lang: str,
    duration_by_pos: Optional[Dict[int, float]] = None,
) -> Dict[int, str]:
    system_prompt = build_system_prompt(src_lang, tgt_lang)

    windows = make_windows(
        groups=groups,
        max_window_chars=settings.max_window_chars,
        pre_groups=settings.context_pre_groups,
        post_groups=settings.context_post_groups,
    )

    total_windows = len(windows)
    total_groups = len(groups)
    total_positions = len(items)

    out_by_pos: Dict[int, str] = {}
    pos_seen: set[int] = set()

    start_ts = time.time()
    done_windows = 0
    done_groups = 0

    total_prompt_tok = 0
    total_pred_tok = 0

    sem = asyncio.Semaphore(getattr(settings, "concurrency", 1) or 1)

    async def run_one(idx: int, w: Window):
        async with sem:
            res_by_pos, focus_groups_done, win_p, win_r = await _translate_entire_window(
                w,
                items=items,
                settings=settings,
                system_prompt=system_prompt,
                duration_by_pos=duration_by_pos,
            )
            return idx, res_by_pos, focus_groups_done, win_p, win_r

    tasks = [asyncio.create_task(run_one(i, w)) for i, w in enumerate(windows)]

    for fut in asyncio.as_completed(tasks):
        idx, res_by_pos, focus_groups_done, win_p, win_r = await fut

        out_by_pos.update(res_by_pos)
        for p in res_by_pos.keys():
            pos_seen.add(int(p))

        done_windows += 1
        done_groups += focus_groups_done

        total_prompt_tok += int(win_p)
        total_pred_tok += int(win_r)

        elapsed = max(1e-6, time.time() - start_ts)
        done_lines = len(pos_seen)

        lps = done_lines / elapsed
        lines_remaining = max(0, total_positions - done_lines)
        eta_s = (lines_remaining / lps) if lps > 0 else 0.0
        
        gen_tps = (total_pred_tok / elapsed) if total_pred_tok > 0 else 0.0

        logger.info(
            "Progress: windows %d/%d (%.1f%%) | groups ~%d/%d (%.1f%%) | lines %d/%d (%.1f%%) | ~%.1f tok/s | ETA ~%s",
            done_windows,
            total_windows,
            100.0 * done_windows / max(1, total_windows),
            done_groups,
            total_groups,
            100.0 * done_groups / max(1, total_groups),
            done_lines,
            total_positions,
            100.0 * done_lines / max(1, total_positions),
            gen_tps,
            format_eta(eta_s),
        )

    return out_by_pos


def build_1to1_groups(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    gid = 1
    for it in items:
        pos = int(it["Position"])
        text = (it.get("Line") or "").strip()
        groups.append({"group_id": gid, "positions": [pos], "text": text})
        gid += 1
    return groups
