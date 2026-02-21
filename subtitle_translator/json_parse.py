# subtitle_translator/json_parse.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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
    Find the first plausible JSON object/array substring by bracket matching.
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
    out_parts: List[str] = []
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
        buf: List[str] = []
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


def _balance_json_delimiters(s: str) -> str:
    """
    If JSON appears truncated at the end (missing closing } or ]),
    append the necessary closing delimiters.

    Conservative:
      - Tracks nesting for { } and [ ] outside of strings.
      - Only appends closers if needed.
      - If the string ends inside a JSON string literal, does nothing.
      - If it sees mismatched closers, does nothing.
    """
    stack: List[str] = []
    in_str = False
    esc = False

    for ch in s:
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

        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]"):
            if stack and ch == stack[-1]:
                stack.pop()
            else:
                # mismatched closer -> don't try to fix
                return s

    # if ended inside a string, don't guess
    if in_str or not stack:
        return s

    return s + "".join(reversed(stack))


def parse_model_json(model_text: str) -> Dict[str, Any]:
    """
    Robust JSON parse for model output:
    - strip wrappers/code fences
    - remove // comments
    - try parse whole text
    - else: try appending missing closing braces/brackets to full text
    - else: extract first JSON substring and parse
    - else: try appending missing closing braces/brackets to substring
    - else: repair unescaped quotes inside "line" fields (+ balance again)
    """
    s = _strip_wrappers(model_text)
    s = strip_json_comments(s)

    # 1) Try full parse
    try:
        return _loads_object(s)
    except json.JSONDecodeError:
        pass
    except Exception:
        # still try repair/substring route
        pass

    # 1b) Try delimiter balancing on FULL text (fixes missing final top-level } / ])
    balanced_full = _balance_json_delimiters(s)
    if balanced_full != s:
        try:
            return _loads_object(balanced_full)
        except json.JSONDecodeError:
            pass
        except Exception:
            pass

    # 2) Try substring parse
    sub = find_first_json_substring(s)
    if not sub:
        # last attempt: raise a useful error from original cleaned string
        return _loads_object(s)

    try:
        return _loads_object(sub)
    except json.JSONDecodeError:
        pass

    # 3) Try delimiter balancing on substring (missing final } / ])
    balanced = _balance_json_delimiters(sub)
    if balanced != sub:
        try:
            return _loads_object(balanced)
        except json.JSONDecodeError:
            pass

    # 4) Repair unescaped quotes inside "line" fields, then balance again
    repaired, repairs = _repair_unescaped_quotes_in_line_fields(sub)
    if repairs > 0:
        repaired2 = _balance_json_delimiters(repaired)
        return _loads_object(repaired2)

    # last: raise
    return _loads_object(sub)


# ---------------------------
# Language-agnostic extraction helpers
# ---------------------------

def extract_group_translations(
    obj: Dict[str, Any],
    expected_focus_ids: Iterable[int],
    *,
    translations_key: str = "translations",
) -> Dict[int, str]:
    """
    Extract {group_id: translated_line} from model JSON WITHOUT depending on key names.

    Some models translate/rename schema keys (e.g. "group_id" -> "group_määrä").
    This finds the group id by matching ANY integer value against expected_focus_ids.

    Rules per translation object:
      - group_id: first int value that is in expected_focus_ids
      - line: prefer value under key "line" if it is a string; otherwise first string value
    """
    expected = set(int(x) for x in expected_focus_ids)
    out: Dict[int, str] = {}

    translations = obj.get(translations_key)
    if not isinstance(translations, list):
        return out

    for item in translations:
        if not isinstance(item, dict):
            continue

        gid: Optional[int] = None
        line: Optional[str] = None

        v_line = item.get("line")
        if isinstance(v_line, str):
            line = v_line

        # language-agnostic group id resolution
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
    obj: Dict[str, Any],
    expected_focus_ids: Sequence[int],
    *,
    translations_key: str = "translations",
) -> Tuple[List[int], List[int], Dict[int, str]]:
    """
    Convenience helper for validation.

    Returns:
      (returned_ids_sorted, missing_ids, mapping)
    """
    mapping = extract_group_translations(obj, expected_focus_ids, translations_key=translations_key)
    returned_ids = sorted(mapping.keys())
    missing = [i for i in expected_focus_ids if i not in mapping]
    return returned_ids, missing, mapping
