# subtitle_translator/llm/json_response.py
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_BOM_RE = re.compile(r"^﻿")
_JSON_COMMENT_RE = re.compile(r"^\s*//.*$", re.MULTILINE)
_BAD_TRANSLATION_OBJECT_CLOSE_RE = re.compile(
    r'(\{\s*"id"\s*:\s*\d+\s*,\s*"line"\s*:\s*"(?:\\.|[^"\\])*"\s*)\](\s*,?)',
    re.DOTALL,
)
_FLAT_TRANSLATION_BOUNDARY_RE = re.compile(
    r'("line"\s*:\s*"(?:[^"\\]|\\.)*")\s*,\s*"id"\s*:',
    re.DOTALL,
)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_VENDOR_DIR = _PROJECT_ROOT / "vendor"

if _VENDOR_DIR.is_dir():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)


def _strip_wrappers(s: str) -> str:
    s = _BOM_RE.sub("", s or "")
    s = s.strip()
    if s.startswith("```"):
        s = _CODE_FENCE_RE.sub("", s).strip()
    return s


def strip_json_comments(s: str) -> str:
    return _JSON_COMMENT_RE.sub("", s)


def fix_bad_translation_object_closers(s: str) -> str:
    return _BAD_TRANSLATION_OBJECT_CLOSE_RE.sub(r"\1}\2", s)


def fix_flat_translation_objects(s: str) -> str:
    return _FLAT_TRANSLATION_BOUNDARY_RE.sub(r'\1},{"id":', s)


def find_first_json_substring(s: str) -> str | None:
    s = (s or "").strip()
    if not s:
        return None

    start = None
    open_ch = None
    for i, ch in enumerate(s):
        if ch == "{":
            start = i
            open_ch = "{"
            break
        if ch == "[":
            start = i
            open_ch = "["
            break
    if start is None or open_ch is None:
        return None

    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return s[start: j + 1]

    return s[start:]


def _coerce_object(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        if all(isinstance(x, dict) for x in obj):
            return {"translations": obj}
    raise ValueError(f"Top-level JSON is not an object: {type(obj).__name__}")


def _loads_object(s: str) -> dict[str, Any]:
    return _coerce_object(json.loads(s))


_VENDOR_PATH = str(Path(__file__).parent.parent / "vendor")


def _repair_json_object(s: str) -> dict[str, Any]:
    if _VENDOR_PATH not in sys.path:
        sys.path.insert(0, _VENDOR_PATH)
    try:
        from json_repair import repair_json
    except ImportError as e:
        raise ImportError(
            "Missing dependency 'json-repair'. Install it or place it in vendor/."
        ) from e
    repaired = repair_json(s, return_objects=True)
    return _coerce_object(repaired)


def parse_model_json(model_text: str) -> dict[str, Any]:
    """
    Robust JSON parse for model output.

    1. Strip common LLM wrappers and parse strict JSON.
    2. Repair the whole cleaned text with json-repair.
    3. Extract the first JSON-looking substring and parse strict JSON.
    4. Repair that substring with json-repair.
    """
    s = strip_json_comments(_strip_wrappers(model_text))
    s = fix_bad_translation_object_closers(s)
    s = fix_flat_translation_objects(s)

    try:
        return _loads_object(s)
    except Exception:
        pass

    try:
        return _repair_json_object(s)
    except Exception:
        pass

    sub = find_first_json_substring(s)
    if not sub:
        return _loads_object(s)

    try:
        return _loads_object(sub)
    except Exception:
        pass

    return _repair_json_object(sub)


def extract_group_translations(
    obj: dict[str, Any],
    expected_focus_ids: Iterable[int],
    *,
    translations_key: str = "translations",
) -> dict[int, str]:
    """
    Extract {group_id: translated_line} from model JSON without depending on
    translated key names. Finds the group id by matching any integer value
    against expected_focus_ids.
    """
    expected = set(int(x) for x in expected_focus_ids)
    out: dict[int, str] = {}

    translations = obj.get(translations_key)
    if not isinstance(translations, list):
        return out

    for item in translations:
        if not isinstance(item, dict):
            continue

        gid: int | None = None
        line: str | None = None

        v_line = item.get("line")
        if isinstance(v_line, str):
            line = v_line

        for v in item.values():
            if isinstance(v, int) and v in expected:
                gid = v
                break

        if line is None:
            for v in item.values():
                if isinstance(v, str):
                    line = v
                    break

        if gid is not None and line is not None:
            out[gid] = line

    return out


def get_returned_ids_and_missing(
    obj: dict[str, Any],
    expected_focus_ids: Sequence[int],
    *,
    translations_key: str = "translations",
) -> tuple[list[int], list[int], dict[int, str]]:
    mapping = extract_group_translations(obj, expected_focus_ids, translations_key=translations_key)
    returned_ids = sorted(mapping.keys())
    missing = [i for i in expected_focus_ids if i not in mapping]
    return returned_ids, missing, mapping
