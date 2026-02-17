# subtitle_translator/llama_client.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

import httpx

from .config import Settings

logger = logging.getLogger(__name__)


def extract_llama_content(raw_json: Dict[str, Any]) -> str:
    """
    llama.cpp /completion responses vary a bit by build/config.
    Try common fields in order.
    """
    # Common: {"content": "..."}
    if isinstance(raw_json.get("content"), str):
        return raw_json["content"]

    # Some builds: {"choices":[{"text":"..."}]} or {"choices":[{"message":{"content":"..."}}]}
    choices = raw_json.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            if isinstance(c0.get("text"), str):
                return c0["text"]
            msg = c0.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]

    # Another common: {"text":"..."}
    if isinstance(raw_json.get("text"), str):
        return raw_json["text"]

    raise ValueError(f"Unable to extract content from llama response keys={list(raw_json.keys())}")


async def call_llama(prompt: str, *, settings: Settings) -> Tuple[Dict[str, Any], str]:
    """
    Call llama.cpp /completion.

    Returns:
      (raw_response_json, extracted_text_content)
    """
    payload = {
        "prompt": prompt,
        "n_predict": settings.n_predict,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "repeat_penalty": settings.repeat_penalty,
        # Add "stop": [...] here if your llama.cpp build supports it and you want hard stops.
    }

    timeout = httpx.Timeout(settings.http_timeout_s)

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(settings.llama_completion_url, json=payload)
        r.raise_for_status()
        raw = r.json()

    content = extract_llama_content(raw)
    return raw, content


async def call_llama_with_retries(
    prompt: str,
    *,
    settings: Settings,
    retries: int = 2,
    base_delay_s: float = 0.75,
) -> Tuple[Dict[str, Any], str]:
    """
    Retry wrapper for transient network/model errors.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            return await call_llama(prompt, settings=settings)
        except Exception as e:
            last_exc = e
            if attempt >= retries:
                break
            delay = base_delay_s * (2 ** attempt)
            logger.warning(
                "llama call failed (attempt %d/%d): %s; retrying in %.2fs",
                attempt + 1,
                retries + 1,
                repr(e),
                delay,
            )
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc
