# subtitle_translator/llm/client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from config import Settings


@dataclass(frozen=True)
class LlmResponse:
    raw: dict[str, Any]
    content: str


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
