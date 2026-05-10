# subtitle_translator/llm/llama_cpp.py
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from config import Settings
from .client import LlmResponse, extract_content
from .prompts import get_profile

logger = logging.getLogger(__name__)


class LlamaCppClient:
    """
    Client for llama.cpp /completion endpoint.

    Owns a persistent httpx.AsyncClient for connection reuse across requests.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._stop_tokens = list(get_profile(settings.prompt_template).stop_tokens)
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(settings.http_timeout_s))

    async def complete(self, prompt: str) -> LlmResponse:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": self._settings.n_predict,
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "repeat_penalty": self._settings.repeat_penalty,
            "stop": self._stop_tokens,
        }

        last_exc: BaseException | None = None
        for attempt in range(3):
            try:
                r = await self._http.post(self._settings.llama_completion_url, json=payload)
                r.raise_for_status()
                raw = r.json()
                return LlmResponse(raw=raw, content=extract_content(raw))
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
