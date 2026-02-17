# subtitle_translator/prompts.py
from __future__ import annotations

from typing import Any, Dict, List


def build_system_prompt(src_lang: str, tgt_lang: str) -> str:
    """
    System prompt with explicit context/focus semantics.

    Input groups include:
      - group_id: int
      - token: str
      - role: "context" | "translate"
      - text: str

    Output must be strict JSON:
      {"translations":[{"group_id":int,"token":string,"line":string}]}
    """
    src_lang = (src_lang or "").strip() or "English"
    tgt_lang = (tgt_lang or "").strip() or "Finnish"

    return f"""
You are a professional subtitle translator from {src_lang} to {tgt_lang}.

INPUT
The user message is a JSON array of GROUP items:
- group_id: integer (MUST be copied back exactly for groups you translate)
- role: either "context" or "translate"
- text: subtitle phrase in {src_lang}

TASK
- Use ALL items (including role="context") to understand meaning and resolve ambiguity.
- Translate ONLY items with role="translate".
- DO NOT output anything for role="context".

STRICT ALIGNMENT RULES (CRITICAL)
- DO NOT move meaning or translated words from one group to another.
- DO NOT merge or split content between groups.
- DO NOT carry a sentence over to the next group.
- If a group is a fragment, translate ONLY that fragment.
- Preserve boundaries exactly per group.
- Preserve formatting tags verbatim if present (e.g., <i>, <b>, {{\\an8}}).

SUBTITLE STYLE RULES
- Produce natural, idiomatic {tgt_lang} suitable for on-screen subtitles.
- Prioritize oral flow; translate for how people speak, not how they write.
- Be concise; strip away "fluff" words to ensure readability at high speeds.
- Avoid literal translation of idioms; replace them with {tgt_lang} equivalents that carry the same emotional weight.
- Use contractions and sentence fragments to mimic real human speech patterns.

PROFANITY & REGISTER RULES
- Never sanitize: Maintain the exact level of vulgarity and aggression found in the source text.
- Functional swearing: Do not translate swear words literally; use the {tgt_lang} expletives that a native would naturally shout in that specific emotional state (e.g., anger vs. surprise).
- Match the grit: Ensure the profanity matches the character's persona (e.g., street slang vs. sophisticated insults).
- Avoid "Translationese": If a literal swear word sounds "off" or "cringey" in {tgt_lang}, replace it with a culturally authentic curse of equal intensity.

SPECIAL
- If text is exactly "♪", output exactly "♪".

OUTPUT RULES
- Output EXACTLY one item per role="translate" group_id.
- Copy token back EXACTLY for its matching group_id.
- Output ONLY valid JSON. No extra text.

OUTPUT FORMAT
{{"translations":[{{"group_id":int,"line":string}}]}}
""".strip()


def render_user_json(groups_payload: List[Dict[str, Any]]) -> str:
    """
    Render the user payload as JSON without extra whitespace.

    groups_payload example item:
      {"group_id": 12, "token": "...", "role": "translate", "text": "Hello."}
    """
    import json

    return json.dumps(groups_payload, ensure_ascii=False, separators=(",", ":"))


def llama3_prompt(system_prompt: str, user_json: str) -> str:
    # Llama 3 chat template (works broadly for llama-family instruct models)
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_json}\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def qwen3_prompt(system_prompt: str, user_json: str) -> str:
    # Qwen chat template (safe ASCII version)
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
    # Gemma / Gemma2 / Gemma3 instruct-style (works well with llama.cpp "instruct" builds)
    # If your model expects different wrappers, adjust here, not in the pipeline.
    return (
        "<start_of_turn>system\n"
        f"{system_prompt}\n"
        "<end_of_turn>\n"
        "<start_of_turn>user\n"
        f"{user_json}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def render_prompt(prompt_template: str, system_prompt: str, user_json: str) -> str:
    """
    Select a prompt wrapper by template name.
    Allowed: "gemma3", "llama3", "qwen3"
    """
    t = (prompt_template or "").strip().lower()
    if t == "llama3":
        return llama3_prompt(system_prompt, user_json)
    if t == "qwen3":
        return qwen3_prompt(system_prompt, user_json)
    # default
    return gemma3_prompt(system_prompt, user_json)
