# subtitle_translator/llm/prompts.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


def build_system_prompt(src_lang: str, tgt_lang: str) -> str:
    src_lang = (src_lang or "").strip() or "English"
    tgt_lang = (tgt_lang or "").strip() or "Finnish"

    return f"""
Translate subtitles from {src_lang} to {tgt_lang}.

Input is a JSON array:
- id: group id
- role: "context" or "translate"
- text: subtitle text

Use all items for context. Translate only role="translate".

Rules:
- Return one item per role="translate" item.
- Copy id exactly.
- Do not translate context items.
- Keep group boundaries; do not merge, split, or reorder.
- Preserve tags verbatim: <i>, <b>, {{\\an8}}.
- Translate fragments as fragments.
- Write natural, concise, idiomatic {tgt_lang} subtitles.
- Preserve tone, jokes, sarcasm, profanity, register, emotion, and character voice.
- Omit fillers such as "oh", "well", "uh", "um", "yeah", "you know".
- Return strict JSON only.

Format:
{{"translations":[{{"id":123,"line":"translated text"}}]}}
""".strip()


def render_user_json(groups_payload: list[dict[str, Any]]) -> str:
    return json.dumps(groups_payload, ensure_ascii=False, separators=(",", ":"))


def llama3_prompt(system_prompt: str, user_json: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_json}\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def qwen3_prompt(system_prompt: str, user_json: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{system_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_json}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def gemma3_prompt(system_prompt: str, user_json: str) -> str:
    return (
        "<start_of_turn>system\n"
        f"{system_prompt}\n"
        "<end_of_turn>\n"
        "<start_of_turn>user\n"
        f"{user_json}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def gemma4_prompt(system_prompt: str, user_json: str) -> str:
    system_prompt = (system_prompt or "").strip()
    user_json = (user_json or "").strip()
    return (
        "<|turn>system\n"
        f"{system_prompt}<turn|>\n"
        "<|turn>user\n"
        f"{user_json}<turn|>\n"
        "<|turn>model\n"
    )


@dataclass(frozen=True)
class PromptProfile:
    stop_tokens: tuple[str, ...]
    render: Callable[[str, str], str]


PROFILES: dict[str, PromptProfile] = {
    "gemma3": PromptProfile(
        stop_tokens=("<end_of_turn>", "<start_of_turn>"),
        render=gemma3_prompt,
    ),
    "gemma4": PromptProfile(
        stop_tokens=("<turn|>", "<|turn>"),
        render=gemma4_prompt,
    ),
    "llama3": PromptProfile(
        stop_tokens=("<|eot_id|>", "<|end_of_text|>"),
        render=llama3_prompt,
    ),
    "qwen3": PromptProfile(
        stop_tokens=("<|im_end|>", "<|im_start|>"),
        render=qwen3_prompt,
    ),
}

def get_profile(template: str) -> PromptProfile:
    key = (template or "").strip().lower()
    if key not in PROFILES:
        raise ValueError(f"Unknown prompt template {template!r}. Valid options: {sorted(PROFILES)}")
    return PROFILES[key]


def render_prompt(prompt_template: str, system_prompt: str, user_json: str) -> str:
    return get_profile(prompt_template).render(system_prompt, user_json)
