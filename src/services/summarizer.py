"""Conversation summarizer."""

import asyncio
from datetime import datetime, timedelta

from src.config import cfg
from src.utils.logger import log
from src.services.claude_ai import generate_summary
from src.services.database import (
    get_recent_conversation,
    get_message_count,
    get_recent_summaries,
    save_summary,
)

_COOLDOWN_HOURS = 12


def _summarize_sync(user_id: int) -> None:
    count = get_message_count(user_id, since_days=1)
    if count < cfg.summary_threshold:
        return

    # Cooldown: once past the threshold, EVERY message would otherwise trigger
    # a fresh (near-identical) summary and a Claude call. One per window is enough.
    latest = get_recent_summaries(user_id, 1)
    if latest:
        try:
            last_end = datetime.fromisoformat(latest[0]["period_end"].replace("Z", "+00:00"))
            age = datetime.utcnow() - last_end.replace(tzinfo=None)
            if age < timedelta(hours=_COOLDOWN_HOURS):
                return
        except (ValueError, KeyError, TypeError):
            pass  # unparseable timestamp — don't let it block summarizing

    log.info(f"Triggering summary — {count} messages in 24h")
    messages = get_recent_conversation(user_id, token_budget=12000)
    if len(messages) < 20:
        return

    half = len(messages) // 2
    to_summarize = messages[:half]
    summary = generate_summary(to_summarize)
    if not summary:
        return

    now = datetime.utcnow()
    save_summary(
        user_id, summary,
        period_start=(now - timedelta(days=1)).isoformat(),
        period_end=now.isoformat(),
        message_count=len(to_summarize),
    )
    log.info(f"Summary saved — {len(to_summarize)} messages condensed")


async def maybe_summarize(user_id: int) -> None:
    """Summarize if warranted. Runs in a worker thread — the DB reads and the
    Claude call are sync and would otherwise block the event loop for seconds."""
    try:
        await asyncio.to_thread(_summarize_sync, user_id)
    except Exception as e:
        log.error(f"Summarization error: {e}", exc_info=True)
