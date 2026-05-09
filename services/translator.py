# subtitle_translator/services/translator.py
from __future__ import annotations

import logging
from typing import Any

from config import Settings
from domain.models import ModelResponseStats, SubtitleGroup, SubtitleLine
from llm.client import LlmClient
from llm.json_response import get_returned_ids_and_missing, parse_model_json
from llm.prompts import build_system_prompt, render_prompt, render_user_json
from subtitle.splitback import split_group_translation_to_positions

logger = logging.getLogger(__name__)


def _trunc(s: str, n: int = 800) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + f" ... [truncated {len(s) - n} chars]"


def _rough_token_estimate(chars: int) -> int:
    return max(1, int(chars / 4))


def _extract_stats(raw: Any, elapsed_s: float) -> ModelResponseStats | None:
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
                return ModelResponseStats(
                    prompt_tokens=int(prompt_n),
                    completion_tokens=int(pred_n),
                    elapsed_s=total_ms / 1000.0,
                )

    usage = raw.get("usage")
    if isinstance(usage, dict):
        prompt_n = usage.get("prompt_tokens")
        pred_n = usage.get("completion_tokens")
        if isinstance(prompt_n, int) and isinstance(pred_n, int) and elapsed_s > 0:
            return ModelResponseStats(
                prompt_tokens=int(prompt_n),
                completion_tokens=int(pred_n),
                elapsed_s=float(elapsed_s),
            )

    prompt_n = raw.get("tokens_evaluated") or raw.get("prompt_tokens")
    pred_n = raw.get("tokens_predicted") or raw.get("predicted_tokens") or raw.get("completion_tokens")
    total_s = raw.get("total_time_s") or raw.get("total_time") or elapsed_s
    if (
        isinstance(prompt_n, int)
        and isinstance(pred_n, int)
        and isinstance(total_s, (int, float))
        and float(total_s) > 0
    ):
        return ModelResponseStats(
            prompt_tokens=int(prompt_n),
            completion_tokens=int(pred_n),
            elapsed_s=float(total_s),
        )

    return None


class Translator:
    """
    Translates a focus chunk of subtitle groups using the configured LLM client.
    """

    def __init__(self, client: LlmClient, settings: Settings, src_lang: str, tgt_lang: str) -> None:
        self._client = client
        self._settings = settings
        self._system_prompt = build_system_prompt(src_lang, tgt_lang)
        self._input_by_pos: dict[int, str] = {}

    def set_items(self, items: list[SubtitleLine]) -> None:
        """Cache {position: text} for split-back lookups."""
        self._input_by_pos = {it.position: it.text for it in items}

    async def translate_chunk(
        self,
        focus: list[SubtitleGroup],
        ctx_before: list[SubtitleGroup],
        ctx_after: list[SubtitleGroup],
        *,
        duration_by_pos: dict[int, float],
    ) -> tuple[dict[int, str], int, int]:
        """
        Translate a focus chunk with surrounding context.
        Returns (translations_by_pos, prompt_tokens, pred_tokens).
        """
        import time

        payload: list[dict[str, Any]] = []
        expected_focus_ids: list[int] = []

        def add_groups(gs: list[SubtitleGroup], role: str) -> None:
            for g in gs:
                payload.append({"id": g.id, "role": role, "text": g.text})
                if role == "translate":
                    expected_focus_ids.append(g.id)

        add_groups(ctx_before, "context")
        add_groups(focus, "translate")
        add_groups(ctx_after, "context")

        user_json = render_user_json(payload)
        prompt = render_prompt(self._settings.prompt_template, self._system_prompt, user_json)

        if logger.isEnabledFor(logging.DEBUG):
            fids = [g.id for g in focus]
            logger.debug(
                "LLM request: focus=%d ids=%s..%s prompt_chars=%d (~%d tok)",
                len(focus),
                fids[0] if fids else None,
                fids[-1] if fids else None,
                len(prompt),
                _rough_token_estimate(len(prompt)),
            )

        t0 = time.time()
        response = await self._client.complete(prompt)
        elapsed = max(1e-6, time.time() - t0)

        stats = _extract_stats(response.raw, elapsed)
        prompt_tok = stats.prompt_tokens if stats else 0
        pred_tok = stats.completion_tokens if stats else 0

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("LLM content (head): %s", _trunc(response.content, 1200))

        try:
            model_json = parse_model_json(response.content)
        except Exception as e:
            logger.error(
                "JSON parse failed. focus_ids=%s content_head=%s err=%r",
                [g.id for g in focus],
                _trunc(response.content, 2000),
                e,
            )
            raise

        try:
            returned_ids, missing_ids, focus_lines_by_gid = get_returned_ids_and_missing(
                model_json, expected_focus_ids,
            )
            if missing_ids:
                raise ValueError(f"missing focus group_ids={missing_ids}")
        except Exception as e:
            returned_ids, _, _ = get_returned_ids_and_missing(model_json, expected_focus_ids)
            logger.error(
                "Validation failed. expected=%s returned=%s content_head=%s err=%r",
                sorted(expected_focus_ids),
                returned_ids,
                _trunc(response.content, 2000),
                e,
            )
            raise

        out_by_pos: dict[int, str] = {}
        focus_by_id = {g.id: g for g in focus}

        for gid, tr_text in focus_lines_by_gid.items():
            g = focus_by_id.get(gid)
            if not g:
                continue
            positions = list(g.positions)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Split-back gid=%d positions=%s tr_head=%s", gid, positions, _trunc(tr_text, 200))
            out_by_pos.update(
                split_group_translation_to_positions(
                    positions,
                    tr_text,
                    self._input_by_pos,
                    settings=self._settings,
                    duration_by_pos=duration_by_pos,
                )
            )

        return out_by_pos, prompt_tok, pred_tok
