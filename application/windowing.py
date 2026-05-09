# subtitle_translator/application/windowing.py
from __future__ import annotations

from dataclasses import dataclass

from domain.models import SubtitleGroup


@dataclass(frozen=True)
class Window:
    context_before: list[SubtitleGroup]
    focus: list[SubtitleGroup]
    context_after: list[SubtitleGroup]


def _group_cost_chars(g: SubtitleGroup) -> int:
    return len(g.text) + 48


def make_windows(
    groups: list[SubtitleGroup],
    max_window_chars: int,
    pre_groups: int,
    post_groups: int,
) -> list[Window]:
    """
    Split groups into windows under a char budget.

    Focus expands greedily until the next group would exceed the budget.
    Context is added best-effort around focus; focus always wins.
    """
    if max_window_chars <= 200:
        raise ValueError("max_window_chars too small to be useful")

    windows: list[Window] = []
    i = 0
    n = len(groups)

    while i < n:
        focus: list[SubtitleGroup] = []
        budget_used = 0
        j = i

        while j < n:
            cost = _group_cost_chars(groups[j])
            if focus and (budget_used + cost) > max_window_chars:
                break
            focus.append(groups[j])
            budget_used += cost
            j += 1

        if not focus:
            focus = [groups[i]]
            j = i + 1
            budget_used = _group_cost_chars(groups[i])

        ctx_before: list[SubtitleGroup] = []
        ctx_after: list[SubtitleGroup] = []
        remaining = max_window_chars - budget_used

        for k in range(i - 1, max(-1, i - pre_groups - 1), -1):
            if k < 0:
                break
            cost = _group_cost_chars(groups[k])
            if cost > remaining:
                break
            ctx_before.append(groups[k])
            remaining -= cost
        ctx_before.reverse()

        for k in range(j, min(n, j + post_groups)):
            cost = _group_cost_chars(groups[k])
            if cost > remaining:
                break
            ctx_after.append(groups[k])
            remaining -= cost

        windows.append(Window(context_before=ctx_before, focus=focus, context_after=ctx_after))
        i = j

    return windows
