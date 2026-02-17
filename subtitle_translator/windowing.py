# subtitle_translator/windowing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class Window:
    """
    A translation window consisting of:
      - context_before: groups included for semantic context only
      - focus: groups that MUST be translated and returned
      - context_after: groups included for semantic context only
    """
    context_before: List[Dict[str, Any]]
    focus: List[Dict[str, Any]]
    context_after: List[Dict[str, Any]]


def _group_cost_chars(g: Dict[str, Any]) -> int:
    # Small overhead per JSON object + token + metadata in the prompt.
    text = (g.get("text") or "")
    return len(text) + 48


def make_windows(
    groups: List[Dict[str, Any]],
    max_window_chars: int,
    pre_groups: int,
    post_groups: int,
) -> List[Window]:
    """
    Split groups into windows under a char budget.

    Notes:
    - Budget is applied to the entire payload (context + focus + overlap).
    - Focus segment expands greedily until adding the next focus group would exceed budget.
    - Context is added around focus but may be truncated if it would overflow the budget.
      (Focus always wins; context is "best effort".)
    """
    if max_window_chars <= 200:
        raise ValueError("max_window_chars too small to be useful")

    windows: List[Window] = []
    i = 0
    n = len(groups)

    while i < n:
        # 1) Grow focus span
        focus: List[Dict[str, Any]] = []
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
            # Force at least one group per window
            focus = [groups[i]]
            j = i + 1
            budget_used = _group_cost_chars(groups[i])

        # 2) Build pre/post context best-effort under remaining budget
        ctx_before: List[Dict[str, Any]] = []
        ctx_after: List[Dict[str, Any]] = []

        remaining = max_window_chars - budget_used

        # Try to add pre-context starting from closest preceding group backwards
        for k in range(i - 1, max(-1, i - pre_groups - 1), -1):
            if k < 0:
                break
            cost = _group_cost_chars(groups[k])
            if cost > remaining:
                break
            ctx_before.append(groups[k])
            remaining -= cost
        ctx_before.reverse()

        # Try to add post-context starting from closest following group forwards
        for k in range(j, min(n, j + post_groups)):
            cost = _group_cost_chars(groups[k])
            if cost > remaining:
                break
            ctx_after.append(groups[k])
            remaining -= cost

        windows.append(Window(context_before=ctx_before, focus=focus, context_after=ctx_after))
        i = j

    return windows


def shrink_focus(window: Window, factor: float = 0.5) -> Window:
    """
    Shrink the focus span (keep left portion) to reduce output length on retries.
    Context is kept the same.
    """
    if not (0.0 < factor < 1.0):
        raise ValueError("factor must be between 0 and 1")

    focus = window.focus
    if len(focus) <= 1:
        return window

    new_len = max(1, int(len(focus) * factor))
    new_focus = focus[:new_len]

    # Move the remainder of the original focus into post-context (closest context),
    # so the model still sees it but we don't require output for it.
    remainder = focus[new_len:]
    new_after = remainder + window.context_after

    return Window(context_before=window.context_before, focus=new_focus, context_after=new_after)
