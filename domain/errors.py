# subtitle_translator/domain/errors.py
from __future__ import annotations


class TranslationError(Exception):
    """Raised when a translation attempt cannot produce a usable result."""


class LlmError(Exception):
    """Raised when the LLM backend returns an unexpected error."""
