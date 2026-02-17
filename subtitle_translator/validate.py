# subtitle_translator/validate.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)


def _extract_translations(model_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    translations = model_json.get("translations")
    if not isinstance(translations, list):
        raise ValueError("model_json.translations is not a list")
    out: List[Dict[str, Any]] = []
    for t in translations:
        if isinstance(t, dict):
            out.append(t)
    return out


def _first_wins_group_map(translations: List[Dict[str, Any]]) -> Dict[int, str]:
    """
    Build group_id -> line map.
    First occurrence of group_id wins (duplicates are logged and ignored).
    """
    got: Dict[int, str] = {}
    dups: List[int] = []

    for t in translations:
        gid = t.get("group_id")
        line = t.get("line")

        if not isinstance(gid, int) or not isinstance(line, str):
            continue

        if gid in got:
            dups.append(gid)
            continue

        got[gid] = line

    if dups:
        logger.warning("Duplicate group_ids in model output (ignored): %s%s", dups[:8], " ..." if len(dups) > 8 else "")

    return got


def validate_focus_by_group_ids(
    expected_focus_ids: Iterable[int],
    model_json: Dict[str, Any],
    *,
    allow_extra: bool = True,
) -> Dict[int, str]:
    """
    Focus-only validator (tokenless):
    - requires all expected focus group_ids to be present
    - ignores extra group_ids by default (context leakage)
    - returns: focus group_id -> translated line
    """
    translations = _extract_translations(model_json)
    got = _first_wins_group_map(translations)

    exp_ids = [int(x) for x in expected_focus_ids]
    exp_set = set(exp_ids)
    got_set = set(got.keys())

    missing = sorted(exp_set - got_set)
    if missing:
        raise ValueError(f"missing focus group_ids={missing}")

    if not allow_extra:
        extra = sorted(got_set - exp_set)
        if extra:
            raise ValueError(f"extra group_ids={extra}")

    # Preserve expected order
    return {gid: got[gid] for gid in exp_ids}


def validate_strict_by_group_ids(
    expected_ids: Iterable[int],
    model_json: Dict[str, Any],
) -> Dict[int, str]:
    """
    Strict validator (tokenless):
    - requires exact set of group_ids (no missing, no extra)
    - returns: group_id -> translated line
    """
    translations = _extract_translations(model_json)
    got = _first_wins_group_map(translations)

    exp_ids = [int(x) for x in expected_ids]
    exp_set = set(exp_ids)
    got_set = set(got.keys())

    missing = sorted(exp_set - got_set)
    extra = sorted(got_set - exp_set)

    if missing or extra:
        raise ValueError(f"group id mismatch: missing={missing} extra={extra}")

    return {gid: got[gid] for gid in exp_ids}
