"""Conversation summarizer — triggers when message count exceeds threshold."""

from datetime import datetime, timedelta

from src.config import cfg
from src.utils.logger import log
from src.services.claude_ai import generate_summary
from src.services.database import (
    get_recent_conversation,
    get_message_count,
    save_summary,
)


async def maybe_summarize(user_id: int) -> None:
    """Check if a summary is needed and generate one if so."""
    try:
        count = get_message_count(user_id, since_days=1)
        if count < cfg.summary_threshold:
            return

        log.info(f"Triggering summary — {count} messages in 24h")

        messages = get_recent_conversation(user_id, token_budget=12000)
        if len(messages) < 20:
            return

        # Summarize the older half
        half = len(messages) // 2
        to_summarize = messages[:half]

        summary = generate_summary(to_summarize)
        if not summary:
            return

        now = datetime.utcnow()
        save_summary(
            user_id,
            summary,
            period_start=(now - timedelta(days=1)).isoformat(),
            period_end=now.isoformat(),
            message_count=len(to_summarize),
        )
        log.info(f"Summary saved — {len(to_summarize)} messages condensed")
    except Exception as e:
        log.error(f"Summarization error: {e}", exc_info=True)
