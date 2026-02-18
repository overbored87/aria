"""Proactive messaging scheduler — cron-based check-ins and follow-ups.

Uses APScheduler to trigger morning/evening check-ins and task follow-ups,
plus a periodic sweep for DB-scheduled messages.
"""

from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from src.config import cfg
from src.utils.logger import log
from src.utils.time_helpers import is_quiet_hours
from src.services.claude_ai import generate_proactive_message, generate_dashboard_insights
from src.services.database import (
    get_pending_scheduled_messages,
    mark_scheduled_sent,
    log_proactive_message,
    get_daily_proactive_count,
    get_memories_by_category,
)

_bot_app = None  # Will hold the telegram Application for sending
_scheduler: AsyncIOScheduler | None = None


def init_scheduler(bot_app) -> AsyncIOScheduler:
    """Set up cron jobs and return the scheduler."""
    global _bot_app, _scheduler
    _bot_app = bot_app
    _scheduler = AsyncIOScheduler()
    user_id = cfg.allowed_user_id
    tz = cfg.user_timezone

    # Morning check-in at 7:30 AM user time
    _scheduler.add_job(
        _send_proactive,
        CronTrigger(hour=7, minute=30, timezone=tz),
        args=[user_id, "morning_checkin"],
        id="morning_checkin",
        replace_existing=True,
    )

    # Evening check-in at 9:00 PM user time
    _scheduler.add_job(
        _send_proactive,
        CronTrigger(hour=21, minute=0, timezone=tz),
        args=[user_id, "evening_checkin"],
        id="evening_checkin",
        replace_existing=True,
    )

    # Task follow-up at 2:00 PM weekdays
    _scheduler.add_job(
        _send_task_followup,
        CronTrigger(hour=14, minute=0, day_of_week="mon-fri", timezone=tz),
        args=[user_id],
        id="task_followup",
        replace_existing=True,
    )

    # Daily dashboard insights at 8:00 AM
    _scheduler.add_job(
        _send_dashboard_insights,
        CronTrigger(hour=8, minute=0, timezone=tz),
        args=[user_id],
        id="dashboard_insights",
        replace_existing=True,
    )

    # Process DB-scheduled messages every 5 minutes
    _scheduler.add_job(
        _process_scheduled_messages,
        "interval",
        minutes=5,
        id="process_scheduled",
        replace_existing=True,
    )

    _scheduler.start()
    log.info("Proactive scheduler initialized with cron jobs")
    return _scheduler


# ─── Core Send Logic ────────────────────────────────────────


async def _send_proactive(user_id: int, msg_type: str, context: dict | None = None) -> None:
    """Generate and send a proactive message with safety checks."""
    try:
        if is_quiet_hours():
            log.info(f"Skipping proactive [{msg_type}] — quiet hours")
            return

        daily_count = get_daily_proactive_count(user_id)
        if daily_count >= cfg.max_proactive_per_day:
            log.info(f"Skipping proactive [{msg_type}] — daily limit ({daily_count})")
            return

        message = generate_proactive_message(user_id, msg_type, context)
        if not message:
            log.warning(f"No message generated for [{msg_type}]")
            return

        # Send via Telegram
        try:
            await _bot_app.bot.send_message(
                chat_id=user_id, text=message, parse_mode="Markdown"
            )
        except Exception:
            # Retry without markdown if formatting fails
            await _bot_app.bot.send_message(chat_id=user_id, text=message)

        log_proactive_message(user_id, msg_type)
        log.info(f"Proactive [{msg_type}] sent: {message[:60]}...")

    except Exception as e:
        log.error(f"Failed to send proactive [{msg_type}]: {e}", exc_info=True)


# ─── Task Follow-up ─────────────────────────────────────────


async def _send_task_followup(user_id: int) -> None:
    """Pick the highest-priority task/goal and follow up."""
    try:
        tasks = get_memories_by_category(user_id, "task")
        goals = get_memories_by_category(user_id, "goal")
        items = sorted(tasks + goals, key=lambda x: x.get("importance", 0), reverse=True)

        if not items:
            log.info("No tasks/goals to follow up on")
            return

        item = items[0]
        await _send_proactive(user_id, "task_followup", {"task": item["content"]})

    except Exception as e:
        log.error(f"Task follow-up error: {e}", exc_info=True)


# ─── Dashboard Insights ─────────────────────────────────────


async def _send_dashboard_insights(user_id: int) -> None:
    """Generate and send daily dashboard insights."""
    try:
        if is_quiet_hours():
            log.info("Skipping dashboard insights — quiet hours")
            return

        daily_count = get_daily_proactive_count(user_id)
        if daily_count >= cfg.max_proactive_per_day:
            log.info("Skipping dashboard insights — daily limit")
            return

        message = generate_dashboard_insights(user_id)
        if not message:
            log.info("No dashboard insights generated (not configured or no data)")
            return

        # Split on double newlines for multi-message feel
        import asyncio
        import re
        chunks = [c.strip() for c in re.split(r"\n\n+", message) if c.strip()]

        for i, chunk in enumerate(chunks):
            if i > 0:
                await asyncio.sleep(0.5)
            try:
                await _bot_app.bot.send_message(
                    chat_id=user_id, text=chunk, parse_mode="Markdown"
                )
            except Exception:
                await _bot_app.bot.send_message(chat_id=user_id, text=chunk)

        log_proactive_message(user_id, "dashboard_insights")
        log.info(f"Dashboard insights sent: {len(chunks)} messages")

    except Exception as e:
        log.error(f"Dashboard insights error: {e}", exc_info=True)


# ─── DB-Scheduled Messages ──────────────────────────────────


async def _process_scheduled_messages() -> None:
    """Sweep for pending scheduled messages and send them."""
    try:
        pending = get_pending_scheduled_messages()
        if not pending:
            return

        log.info(f"Processing {len(pending)} scheduled messages")

        for item in pending:
            uid = item["user_id"]

            if is_quiet_hours():
                log.info(f"Skipping scheduled {item['id']} — quiet hours")
                continue

            daily = get_daily_proactive_count(uid)
            if daily >= cfg.max_proactive_per_day:
                log.info(f"Skipping scheduled {item['id']} — daily limit")
                continue

            message = item.get("content")
            if not message:
                message = generate_proactive_message(
                    uid, item["type"], item.get("context", {})
                )

            if message:
                try:
                    await _bot_app.bot.send_message(
                        chat_id=uid, text=message, parse_mode="Markdown"
                    )
                except Exception:
                    await _bot_app.bot.send_message(chat_id=uid, text=message)
                log_proactive_message(uid, item["type"])

            mark_scheduled_sent(item["id"])
            log.info(f"Scheduled {item['id']} [{item['type']}] processed")

    except Exception as e:
        log.error(f"Error processing scheduled messages: {e}", exc_info=True)


# ─── Reminder Scheduling ────────────────────────────────────


def schedule_reminder(user_id: int, dt, message: str) -> bool:
    """Schedule a one-time reminder at a specific datetime.

    Args:
        user_id: Telegram user ID to send to
        dt: datetime object (timezone-aware) for when to fire
        message: The reminder text to send
    Returns:
        True if scheduled successfully
    """
    if _scheduler is None:
        log.error("Scheduler not initialized — can't schedule reminder")
        return False

    job_id = f"reminder_{user_id}_{dt.timestamp()}"

    _scheduler.add_job(
        _send_reminder,
        DateTrigger(run_date=dt),
        args=[user_id, message],
        id=job_id,
        replace_existing=True,
    )

    log.info(f"Reminder scheduled: '{message[:50]}' at {dt.isoformat()}")
    return True


async def _send_reminder(user_id: int, message: str) -> None:
    """Send a reminder message to the user."""
    try:
        if _bot_app is None:
            log.error("Bot app not available for reminder")
            return

        reminder_text = f"⏰ *Reminder*\n\n{message}"

        try:
            await _bot_app.bot.send_message(
                chat_id=user_id, text=reminder_text, parse_mode="Markdown"
            )
        except Exception:
            await _bot_app.bot.send_message(chat_id=user_id, text=f"⏰ Reminder\n\n{message}")

        log.info(f"Reminder sent to {user_id}: {message[:60]}")

    except Exception as e:
        log.error(f"Failed to send reminder: {e}", exc_info=True)

