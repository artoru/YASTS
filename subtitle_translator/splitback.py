# subtitle_translator/splitback.py
from __future__ import annotations

from typing import Dict, List, Optional

from .config import Settings


def split_greedy(text: str, targets: List[int], min_chunk_chars: int) -> List[str]:
    s = (text or "").strip()
    if not targets:
        return [s]

    words = s.split()
    if not words:
        return [""] * len(targets)

    chunks: List[str] = []
    wi = 0
    for ti, target_len in enumerate(targets):
        if wi >= len(words):
            chunks.append("")
            continue

        if ti == len(targets) - 1:
            chunks.append(" ".join(words[wi:]).strip())
            wi = len(words)
            continue

        cur: List[str] = []
        cur_len = 0

        while wi < len(words):
            w = words[wi]
            add_len = len(w) + (1 if cur else 0)

            if cur and (cur_len + add_len) > target_len and cur_len >= min_chunk_chars:
                break

            cur.append(w)
            cur_len += add_len
            wi += 1

            if cur_len >= target_len and cur_len >= min_chunk_chars:
                break

        chunks.append(" ".join(cur).strip())

    return chunks


def _targets_from_source_lengths(
    positions: List[int],
    input_by_pos: Dict[int, str],
    *,
    settings: Settings,
) -> List[int]:
    targets: List[int] = []
    for p in positions:
        src = (input_by_pos.get(p) or "").strip()
        targets.append(min(settings.split_max_line_chars, max(8, len(src))))
    return targets


def _targets_weighted_by_duration(
    positions: List[int],
    translated_text: str,
    duration_by_pos: Dict[int, float],
    *,
    settings: Settings,
) -> List[int]:
    tr = (translated_text or "").strip()
    total_chars = max(1, len(tr))

    min_t = max(8, int(settings.min_chunk_chars))
    max_t = int(settings.split_max_line_chars)

    durs = [max(0.01, float(duration_by_pos.get(p, 0.0))) for p in positions]
    total_dur = sum(durs)
    if total_dur <= 0:
        return _targets_from_source_lengths(positions, {}, settings=settings)

    # Proportional initial targets with clamps
    targets: List[int] = []
    for dur in durs:
        share = dur / total_dur
        t = int(round(total_chars * share))
        t = max(min_t, t)
        t = min(max_t, t)
        targets.append(t)

    # Desired sum we try to approach (bounded by capacity)
    cap_total = min(total_chars, max_t * len(positions))
    cur_total = sum(targets)

    if cur_total < cap_total:
        budget = cap_total - cur_total
        order = sorted(range(len(positions)), key=lambda i: durs[i], reverse=True)

        # Only iterate while there's someone who can still grow
        while budget > 0:
            growable = [i for i in order if targets[i] < max_t]
            if not growable:
                break  # nothing can grow -> stop (prevents infinite loop)

            for idx in growable:
                if budget <= 0:
                    break
                if targets[idx] < max_t:
                    targets[idx] += 1
                    budget -= 1

    elif cur_total > cap_total:
        over = cur_total - cap_total
        order = sorted(range(len(positions)), key=lambda i: durs[i])  # shortest first

        while over > 0:
            shrinkable = [i for i in order if targets[i] > min_t]
            if not shrinkable:
                break  # nothing can shrink -> stop

            for idx in shrinkable:
                if over <= 0:
                    break
                if targets[idx] > min_t:
                    targets[idx] -= 1
                    over -= 1

    return targets


def split_group_translation_to_positions(
    positions: List[int],
    translated_group_text: str,
    input_by_pos: Dict[int, str],
    *,
    settings: Settings,
    duration_by_pos: Optional[Dict[int, float]] = None,
) -> Dict[int, str]:
    tr = (translated_group_text or "").strip()
    if tr == "♪":
        return {positions[0]: "♪"} if positions else {}

    if duration_by_pos and all(p in duration_by_pos for p in positions):
        targets = _targets_weighted_by_duration(positions, tr, duration_by_pos, settings=settings)
    else:
        targets = _targets_from_source_lengths(positions, input_by_pos, settings=settings)

    chunks = split_greedy(tr, targets, min_chunk_chars=settings.min_chunk_chars)

    return {p: (c or "").strip() for p, c in zip(positions, chunks)}
