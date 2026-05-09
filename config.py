# subtitle_translator/config.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Logging
    log_level: str = "INFO"

    # Llama.cpp endpoint + prompt template
    llama_completion_url: str = "http://127.0.0.1:8180/v1/completions"
    prompt_template: str = "gemma3"  # gemma3 | gemma4 | llama3 | qwen3
    http_timeout_s: float = 180.0

    # Backend type (llamacpp vs vllm)
    backend: str = "vllm"  # "llamacpp" | "vllm"
    model_name: str = "gemma"  # model name sent to vLLM (--served-model-name)

    # Sampling
    n_predict: int = 2048
    temperature: float = 0.04
    top_p: float = 0.80
    repeat_penalty: float = 1.1

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
    context_pre_groups: int = 1
    context_post_groups: int = 1
    max_retries_per_window: int = 2
    shrink_focus_on_retry: bool = True
    concurrency: int = 1

    # Default languages (CLI can override)
    src_lang: str = "English"
    tgt_lang: str = "Finnish"

    # Review pass (used by yasts-review.py)
    review_include_source: bool = False
