"""Context window helpers for sliding-window + summary behavior."""

from __future__ import annotations

from typing import Dict, List


def compress_history(history: List[Dict], keep_recent: int = 8) -> List[Dict]:
    """
    Keep recent messages and summarize older content into a single system item.

    This is a lightweight context strategy suitable for local models.
    """
    if not history:
        return []
    if len(history) <= keep_recent:
        return history

    old_messages = history[:-keep_recent]
    recent_messages = history[-keep_recent:]

    snippets: List[str] = []
    for msg in old_messages:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        snippets.append(f"{role}: {content[:140]}")

    summary = " | ".join(snippets[:20])
    summary_message = {
        "role": "system",
        "content": f"Conversation summary of earlier turns: {summary}",
    }
    return [summary_message, *recent_messages]

