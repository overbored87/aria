"""Timezone-aware helpers for Singapore time."""

from datetime import datetime
from zoneinfo import ZoneInfo

from src.config import cfg

_tz = ZoneInfo(cfg.user_timezone)


def now_user() -> datetime:
    return datetime.now(_tz)


def user_hour() -> int:
    return now_user().hour


def is_quiet_hours() -> bool:
    h = user_hour()
    if cfg.quiet_start > cfg.quiet_end:
        return h >= cfg.quiet_start or h < cfg.quiet_end
    return cfg.quiet_start <= h < cfg.quiet_end


def format_user_time(dt: datetime | None = None) -> str:
    dt = dt or now_user()
    return dt.strftime("%a %d %b, %I:%M %p")


def today_date_str() -> str:
    return now_user().strftime("%Y-%m-%d")


def time_of_day() -> str:
    h = user_hour()
    if h < 12:
        return "morning"
    if h < 17:
        return "afternoon"
    if h < 21:
        return "evening"
    return "night"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
