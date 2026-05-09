# subtitle_translator/llm/llama_cpp.py
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from config import Settings
from .client import LlmResponse

logger = logging.getLogger(__name__)

_STOP_TOKENS = ["<end_of_turn>", "<start_of_turn>"]


def _extract_content(raw: dict[str, Any]) -> str:
    if isinstance(raw.get("content"), str):
        return raw["content"]
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            if isinstance(c0.get("text"), str):
                return c0["text"]
            msg = c0.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
    if isinstance(raw.get("text"), str):
        return raw["text"]
    raise ValueError(f"Unable to extract content from response keys={list(raw.keys())}")


class LlamaCppClient:
    """
    Client for llama.cpp /completion endpoint.

    Owns a persistent httpx.AsyncClient for connection reuse across requests.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(settings.http_timeout_s))

    async def complete(self, prompt: str) -> LlmResponse:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": self._settings.n_predict,
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "repeat_penalty": self._settings.repeat_penalty,
            "stop": _STOP_TOKENS,
        }

        last_exc: BaseException | None = None
        for attempt in range(3):
            try:
                r = await self._http.post(self._settings.llama_completion_url, json=payload)
                r.raise_for_status()
                raw = r.json()
                return LlmResponse(raw=raw, content=_extract_content(raw))
            except httpx.HTTPStatusError as e:
                body = "<unreadable>"
                try:
                    body = e.response.text
                except Exception:
                    pass
                logger.error("LLM HTTP error status=%s body=%s", e.response.status_code, body)
                raise
            except Exception as e:
                last_exc = e
                if attempt >= 2:
                    break
                delay = 0.75 * (2 ** attempt)
                logger.warning("llama.cpp call failed (attempt %d/3): %r; retrying in %.2fs", attempt + 1, e, delay)
                await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc

    async def aclose(self) -> None:
        await self._http.aclose()
