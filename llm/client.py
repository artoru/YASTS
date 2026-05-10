# subtitle_translator/llm/client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from config import Settings


@dataclass(frozen=True)
class LlmResponse:
    raw: dict[str, Any]
    content: str


def extract_content(raw: dict[str, Any]) -> str:
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            if isinstance(c0.get("text"), str):
                return c0["text"]
            msg = c0.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
    if isinstance(raw.get("content"), str):
        return raw["content"]
    if isinstance(raw.get("text"), str):
        return raw["text"]
    raise ValueError(f"Unable to extract content from response keys={list(raw.keys())}")


@runtime_checkable
class LlmClient(Protocol):
    async def complete(self, prompt: str) -> LlmResponse: ...
    async def aclose(self) -> None: ...


def make_llm_client(settings: Settings) -> LlmClient:
    """Factory: create the right LlmClient implementation for settings.backend."""
    if settings.backend == "llamacpp":
        from .llama_cpp import LlamaCppClient
        return LlamaCppClient(settings)
    from .vllm import VllmClient
    return VllmClient(settings)
