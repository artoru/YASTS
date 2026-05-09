# subtitle_translator/subtitle/splitback.py
from __future__ import annotations

from typing import Sequence

from config import Settings

_BOUNDARY_SPLIT_CHARS = frozenset({",", ".", ":", ";", "-", "–", "—", "#", "!", "?", "…"})
_CLOSING_TRAILERS = '"\'“”‘’)]}»›'
_MAX_BOUNDARY_SHIFT_WORDS = 2


def split_greedy(text: str, targets: list[int], min_chunk_chars: int) -> list[str]:
    s = (text or "").strip()
    if not targets:
        return [s]

    words = s.split()
    if not words:
        return [""] * len(targets)

    chunks: list[str] = []
    wi = 0
    for ti, target_len in enumerate(targets):
        if wi >= len(words):
            chunks.append("")
            continue
        if ti == len(targets) - 1:
            chunks.append(" ".join(words[wi:]).strip())
            wi = len(words)
            continue

        cur: list[str] = []
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
    positions: list[int],
    input_by_pos: dict[int, str],
    *,
    settings: Settings,
) -> list[int]:
    return [
        min(settings.split_max_line_chars, max(8, len((input_by_pos.get(p) or "").strip())))
        for p in positions
    ]


def _targets_weighted_by_duration(
    positions: list[int],
    translated_text: str,
    duration_by_pos: dict[int, float],
    *,
    settings: Settings,
) -> list[int]:
    tr = (translated_text or "").strip()
    total_chars = max(1, len(tr))
    min_t = max(8, int(settings.min_chunk_chars))
    max_t = int(settings.split_max_line_chars)

    durs = [max(0.01, float(duration_by_pos.get(p, 0.0))) for p in positions]
    total_dur = sum(durs)
    if total_dur <= 0:
        return _targets_from_source_lengths(positions, {}, settings=settings)

    targets: list[int] = []
    for dur in durs:
        share = dur / total_dur
        t = int(round(total_chars * share))
        targets.append(max(min_t, min(max_t, t)))

    cap_total = min(total_chars, max_t * len(positions))
    cur_total = sum(targets)

    if cur_total < cap_total:
        budget = cap_total - cur_total
        order = sorted(range(len(positions)), key=lambda i: durs[i], reverse=True)
        while budget > 0:
            growable = [i for i in order if targets[i] < max_t]
            if not growable:
                break
            for idx in growable:
                if budget <= 0:
                    break
                if targets[idx] < max_t:
                    targets[idx] += 1
                    budget -= 1
    elif cur_total > cap_total:
        over = cur_total - cap_total
        order = sorted(range(len(positions)), key=lambda i: durs[i])
        while over > 0:
            shrinkable = [i for i in order if targets[i] > min_t]
            if not shrinkable:
                break
            for idx in shrinkable:
                if over <= 0:
                    break
                if targets[idx] > min_t:
                    targets[idx] -= 1
                    over -= 1

    return targets


def _plain_boundary_token(word: str) -> str:
    return (word or "").rstrip(_CLOSING_TRAILERS).strip()


def _ends_with_boundary_char(word: str) -> bool:
    w = _plain_boundary_token(word)
    return bool(w) and w[-1] in _BOUNDARY_SPLIT_CHARS


def _chunk_len(words: Sequence[str]) -> int:
    return len(" ".join(words).strip())


def _smooth_boundaries_on_punctuation(
    chunks: list[str],
    targets: Sequence[int],
    *,
    max_shift_words: int = _MAX_BOUNDARY_SHIFT_WORDS,
) -> list[str]:
    if len(chunks) <= 1:
        return chunks

    word_chunks = [c.split() for c in chunks]

    for i in range(len(word_chunks) - 1):
        left = word_chunks[i]
        right = word_chunks[i + 1]
        if not left or not right:
            continue

        combined = left + right
        cur_split = len(left)
        lo = max(1, cur_split - max_shift_words)
        hi = min(len(combined) - 1, cur_split + max_shift_words)

        best_split = cur_split
        best_key: tuple[int, int, int, int] | None = None

        for split_at in range(lo, hi + 1):
            if not _ends_with_boundary_char(combined[split_at - 1]):
                continue
            cand_left = combined[:split_at]
            cand_right = combined[split_at:]
            if not cand_left or not cand_right:
                continue
            left_len = _chunk_len(cand_left)
            right_len = _chunk_len(cand_right)
            target_left = int(targets[i]) if i < len(targets) else left_len
            target_right = int(targets[i + 1]) if (i + 1) < len(targets) else right_len
            length_penalty = abs(left_len - target_left) + abs(right_len - target_right)
            shift_penalty = abs(split_at - cur_split)
            current_bonus = 1 if split_at == cur_split else 0
            key = (-length_penalty, -shift_penalty, current_bonus, split_at)
            if best_key is None or key > best_key:
                best_key = key
                best_split = split_at

        if best_split != cur_split:
            word_chunks[i] = combined[:best_split]
            word_chunks[i + 1] = combined[best_split:]

    return [" ".join(words).strip() for words in word_chunks]


def split_group_translation_to_positions(
    positions: list[int],
    translated_group_text: str,
    input_by_pos: dict[int, str],
    *,
    settings: Settings,
    duration_by_pos: dict[int, float] | None = None,
) -> dict[int, str]:
    tr = (translated_group_text or "").strip()
    if tr == "♪":
        return {positions[0]: "♪"} if positions else {}

    if duration_by_pos and all(p in duration_by_pos for p in positions):
        targets = _targets_weighted_by_duration(positions, tr, duration_by_pos, settings=settings)
    else:
        targets = _targets_from_source_lengths(positions, input_by_pos, settings=settings)

    chunks = split_greedy(tr, targets, min_chunk_chars=settings.min_chunk_chars)
    chunks = _smooth_boundaries_on_punctuation(chunks, targets)
    return {p: (c or "").strip() for p, c in zip(positions, chunks)}
