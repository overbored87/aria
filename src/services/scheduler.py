"""Scheduler — currently inactive.

All proactive messaging and sweep jobs have been removed.
Kept as a stub so imports don't break.
"""

from __future__ import annotations

from src.utils.logger import log


_bot_app = None


def init_scheduler(bot_app):
    """No-op scheduler init. Kept for compatibility."""
    global _bot_app
    _bot_app = bot_app
    log.info("Scheduler initialized (no active jobs)")
    return None


def schedule_reminder(user_id: int, dt, message: str) -> bool:
    """No-op — reminders disabled."""
    log.info(f"Reminder requested but scheduler is inactive: '{message[:50]}'")
    return False
