"""Proactive messaging scheduler."""

from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import cfg
from src.utils.logger import log
from src.utils.time_helpers import is_quiet_hours
from src.services.claude_ai import generate_proactive_message
from src.services.database import (
    get_pending_scheduled_messages,
    mark_scheduled_sent,
    log_proactive_message,
    get_daily_proactive_count,
    get_memories_by_category,
)

_bot_app = None
_scheduler: AsyncIOScheduler | None = None


def init_scheduler(bot_app) -> AsyncIOScheduler:
    global _bot_app, _scheduler
    _bot_app = bot_app
    _scheduler = AsyncIOScheduler()
    user_id = cfg.allowed_user_id
    tz = cfg.user_timezone

    _scheduler.add_job(
        _send_proactive,
        CronTrigger(hour=7, minute=30, timezone=tz),
        args=[user_id, "morning_checkin"],
        id="morning_checkin", replace_existing=True,
    )

    _scheduler.add_job(
        _send_proactive,
        CronTrigger(hour=21, minute=0, timezone=tz),
        args=[user_id, "evening_checkin"],
        id="evening_checkin", replace_existing=True,
    )

    _scheduler.add_job(
        _send_task_followup,
        CronTrigger(hour=14, minute=0, day_of_week="mon-fri", timezone=tz),
        args=[user_id],
        id="task_followup", replace_existing=True,
    )

    _scheduler.add_job(
        _process_scheduled_messages,
        "interval", minutes=5,
        id="process_scheduled", replace_existing=True,
    )

    _scheduler.start()
    log.info("Proactive scheduler initialized with cron jobs")
    return _scheduler


async def _send_proactive(user_id: int, msg_type: str, context: dict | None = None) -> None:
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

        try:
            await _bot_app.bot.send_message(chat_id=user_id, text=message, parse_mode="Markdown")
        except Exception:
            await _bot_app.bot.send_message(chat_id=user_id, text=message)

        log_proactive_message(user_id, msg_type)
        log.info(f"Proactive [{msg_type}] sent: {message[:60]}...")

    except Exception as e:
        log.error(f"Failed to send proactive [{msg_type}]: {e}", exc_info=True)


async def _send_task_followup(user_id: int) -> None:
    try:
        tasks = get_memories_by_category(user_id, "task")
        goals = get_memories_by_category(user_id, "goal")
        items = sorted(tasks + goals, key=lambda x: x.get("importance", 0), reverse=True)

        if not items:
            log.info("No tasks/goals to follow up on")
            return

        await _send_proactive(user_id, "task_followup", {"task": items[0]["content"]})
    except Exception as e:
        log.error(f"Task follow-up error: {e}", exc_info=True)


async def _process_scheduled_messages() -> None:
    try:
        pending = get_pending_scheduled_messages()
        if not pending:
            return

        log.info(f"Processing {len(pending)} scheduled messages")

        for item in pending:
            uid = item["user_id"]

            if is_quiet_hours():
                continue

            daily = get_daily_proactive_count(uid)
            if daily >= cfg.max_proactive_per_day:
                continue

            message = item.get("content")
            if not message:
                message = generate_proactive_message(uid, item["type"], item.get("context", {}))

            if message:
                try:
                    await _bot_app.bot.send_message(chat_id=uid, text=message, parse_mode="Markdown")
                except Exception:
                    await _bot_app.bot.send_message(chat_id=uid, text=message)
                log_proactive_message(uid, item["type"])

            mark_scheduled_sent(item["id"])
            log.info(f"Scheduled {item['id']} [{item['type']}] processed")

    except Exception as e:
        log.error(f"Error processing scheduled messages: {e}", exc_info=True)
