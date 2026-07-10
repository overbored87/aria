"""Timezone-aware helpers for Singapore time."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from src.config import cfg

_tz = ZoneInfo(cfg.user_timezone)


def now_user() -> datetime:
    """Current datetime in user's timezone."""
    return datetime.now(_tz)


def format_user_time(dt: datetime | None = None) -> str:
    """Human-readable time string in user's locale, including year."""
    dt = dt or now_user()
    return dt.strftime("%a %d %b %Y, %I:%M %p")


def utc_to_user(iso_str: str) -> str:
    """Convert a UTC ISO timestamp from Supabase to user-timezone formatted string."""
    try:
        # Supabase returns ISO like "2026-02-28T09:00:00+00:00" or "2026-02-28T09:00:00"
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone(_tz)
        return local.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso_str[:16].replace("T", " ")


def estimate_tokens(text: str) -> int:
    """Rough token estimate (≈4 chars/token)."""
    return max(1, len(text) // 4)
