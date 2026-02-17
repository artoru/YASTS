# subtitle_translator/config.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Logging
    log_level: str = "INFO"

    # Llama.cpp endpoint + prompt template
    llama_completion_url: str = "http://127.0.0.1:8671/completion"
    prompt_template: str = "gemma3"  # gemma3 | llama3 | qwen3
    http_timeout_s: float = 120.0

    # Sampling
    n_predict: int = 2048
    temperature: float = 0.1
    top_p: float = 0.90
    repeat_penalty: float = 1.0

    # Phrase grouping (sentence-ish units)
    use_phrase_grouping: bool = True
    max_group_lines: int = 8
    max_group_chars: int = 360
    min_group_text_chars: int = 10
    min_group_words: int = 2

    # Split-back / display shaping
    split_max_line_chars: int = 42
    min_chunk_chars: int = 10

    # Context-window batching
    max_window_chars: int = 2000
    context_pre_groups: int = 2
    context_post_groups: int = 2
    max_retries_per_window: int = 2
    shrink_focus_on_retry: bool = True
    concurrency: int = 1

    # Default languages (CLI can override)
    src_lang: str = "English"
    tgt_lang: str = "Finnish"
