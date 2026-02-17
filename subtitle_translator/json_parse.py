# subtitle_translator/json_parse.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_BOM_RE = re.compile(r"^\ufeff")
_JSON_COMMENT_RE = re.compile(r"^\s*//.*$", re.MULTILINE)

# Matches: "line"   :   "
_LINE_VALUE_START_RE = re.compile(r'"line"\s*:\s*"')


def _strip_wrappers(s: str) -> str:
    """
    Remove common wrappers: BOM, leading/trailing code fences, and whitespace.
    """
    s = _BOM_RE.sub("", s)
    s = s.strip()
    if s.startswith("```"):
        s = _CODE_FENCE_RE.sub("", s).strip()
    return s


def strip_json_comments(s: str) -> str:
    """
    Remove // comments (some models hallucinate them).
    """
    return _JSON_COMMENT_RE.sub("", s)


def find_first_json_substring(s: str) -> Optional[str]:
    """
    Find the first valid JSON object/array substring by bracket matching.
    Returns substring or None if nothing plausible found.
    """
    s = s.strip()
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
    if start is None:
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
                return s[start : j + 1]

    return None


def _is_escaped(s: str, i: int) -> bool:
    """
    True if s[i] is preceded by an odd number of backslashes.
    """
    bs = 0
    j = i - 1
    while j >= 0 and s[j] == "\\":
        bs += 1
        j -= 1
    return (bs % 2) == 1


def _repair_unescaped_quotes_in_line_fields(s: str) -> Tuple[str, int]:
    """
    Repair invalid JSON caused by unescaped quotes inside "line":"...".

    Strategy:
    - Find each occurrence of: "line" : "
    - Walk forward through the value string.
    - Any unescaped quote that is NOT a plausible terminator is escaped as \"
      A quote is considered a terminator if after it (skipping whitespace) comes ',' or '}'.

    Returns: (repaired_string, num_repairs_made)
    """
    repairs = 0
    out_parts = []
    pos = 0

    while True:
        m = _LINE_VALUE_START_RE.search(s, pos)
        if not m:
            out_parts.append(s[pos:])
            break

        # Copy everything up to and including the opening quote of the value
        start = m.start()
        out_parts.append(s[pos:start])
        out_parts.append(m.group(0))  # '"line"\s*:\s*"'
        i = m.end()  # position just after the opening quote

        # Now we're inside the "line" string value
        buf = []
        while i < len(s):
            ch = s[i]

            if ch == '"' and not _is_escaped(s, i):
                # Candidate: either terminator or an internal quote that must be escaped.
                k = i + 1
                while k < len(s) and s[k] in " \t\r\n":
                    k += 1

                if k < len(s) and s[k] in ",}":
                    # This is the closing quote of the JSON string value.
                    out_parts.append("".join(buf))
                    out_parts.append('"')
                    i += 1
                    break

                # Otherwise it's an internal quote -> escape it.
                buf.append('\\"')
                repairs += 1
                i += 1
                continue

            buf.append(ch)
            i += 1

        else:
            # We ran off the end without finding a terminator quote; just append the remainder.
            out_parts.append("".join(buf))
            pos = i
            break

        # Continue searching after the terminator quote we consumed.
        pos = i

    return "".join(out_parts), repairs


def _loads_object(s: str) -> Dict[str, Any]:
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON is not an object")
    return obj


def parse_model_json(model_text: str) -> Dict[str, Any]:
    """
    Robust JSON parse for local model output:
    - strip wrappers/code fences
    - remove // comments
    - try parse whole text
    - else: extract first JSON substring and parse
    - else: repair unescaped quotes inside "line" fields and parse again
    """
    s = _strip_wrappers(model_text)
    s = strip_json_comments(s)

    # 1) Try full parse
    try:
        return _loads_object(s)
    except json.JSONDecodeError:
        pass
    except Exception:
        # Non-decode errors (e.g., not an object) should still try substring path
        pass

    # 2) Try substring parse
    sub = find_first_json_substring(s)
    if not sub:
        # Re-raise the most recent exception by attempting once more (will throw)
        return _loads_object(s)

    try:
        return _loads_object(sub)
    except json.JSONDecodeError as e:
        # 3) Attempt targeted repair for unescaped quotes in "line" fields
        repaired, repairs = _repair_unescaped_quotes_in_line_fields(sub)
        if repairs > 0:
            return _loads_object(repaired)
        # Nothing repaired; re-raise original decode error
        raise e
