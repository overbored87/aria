#!/bin/bash
set -e
echo "🔧 Patching Aria Bot — Fix timezone + dashboard limit..."

cat > src/config.py << 'PYEOF'
"""Central configuration — reads env vars once, validates, exports."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # Telegram
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    allowed_user_id: int = int(os.getenv("ALLOWED_USER_ID", "0"))

    # Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 1024
    token_budget: int = 8000       # max tokens for conversation history
    max_messages: int = 50         # max messages to fetch for context
    summary_threshold: int = 40    # trigger summary after N messages in 24h

    # Supabase
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")

    # User defaults
    user_timezone: str = os.getenv("USER_TIMEZONE", "Asia/Singapore")
    user_name: str = "Kieran"
    quiet_start: int = 23   # 11 PM
    quiet_end: int = 7      # 7 AM
    max_proactive_per_day: int = 5

    # Serper.dev (optional — web search disabled if not set)
    serper_api_key: str = os.getenv("SERPER_API_KEY", "")

    # Dashboard Supabase (optional — separate DB for personal dashboard)
    dashboard_supabase_url: str = os.getenv("DASHBOARD_SUPABASE_URL", "")
    dashboard_supabase_key: str = os.getenv("DASHBOARD_SUPABASE_KEY", "")

    # App
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> None:
        missing = []
        if not self.telegram_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not self.allowed_user_id:
            missing.append("ALLOWED_USER_ID")
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not self.supabase_url:
            missing.append("SUPABASE_URL")
        if not self.supabase_key:
            missing.append("SUPABASE_SERVICE_KEY")
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


cfg = Config()

PYEOF

cat > src/utils/time_helpers.py << 'PYEOF'
"""Timezone-aware helpers for Singapore time."""

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from src.config import cfg

_tz = ZoneInfo(cfg.user_timezone)


def now_user() -> datetime:
    """Current datetime in user's timezone."""
    return datetime.now(_tz)


def user_hour() -> int:
    """Current hour (0-23) in user's timezone."""
    return now_user().hour


def is_quiet_hours() -> bool:
    """True if inside quiet window (wraps around midnight)."""
    h = user_hour()
    if cfg.quiet_start > cfg.quiet_end:
        return h >= cfg.quiet_start or h < cfg.quiet_end
    return cfg.quiet_start <= h < cfg.quiet_end


def format_user_time(dt: datetime | None = None) -> str:
    """Human-readable time string in user's locale, including year."""
    dt = dt or now_user()
    return dt.strftime("%a %d %b %Y, %I:%M %p")


def today_date_str() -> str:
    """YYYY-MM-DD in user's timezone."""
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

PYEOF

cat > src/services/database.py << 'PYEOF'
"""Supabase database operations — conversations, memories, scheduling."""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any

from supabase import create_client, Client

from src.config import cfg
from src.utils.logger import log
from src.utils.time_helpers import estimate_tokens, today_date_str

_client: Client | None = None


def get_db() -> Client:
    global _client
    if _client is None:
        _client = create_client(cfg.supabase_url, cfg.supabase_key)
    return _client


# ── User Operations ──────────────────────────────────────────

def ensure_user(telegram_user) -> dict:
    """Upsert user record from Telegram user object."""
    db = get_db()
    data = {
        "id": telegram_user.id,
        "username": telegram_user.username or None,
        "first_name": telegram_user.first_name or None,
        "timezone": cfg.user_timezone,
    }
    result = db.table("users").upsert(data, on_conflict="id").execute()
    return result.data[0] if result.data else data


# ── Conversation Operations ──────────────────────────────────

def save_message(user_id: int, role: str, content: str, metadata: dict | None = None) -> str:
    """Save a message and return its UUID."""
    db = get_db()
    row = {
        "user_id": user_id,
        "role": role,
        "content": content,
        "token_estimate": estimate_tokens(content),
        "metadata": metadata or {},
    }
    result = db.table("conversations").insert(row).execute()
    return result.data[0]["id"] if result.data else ""


def get_recent_conversation(user_id: int, token_budget: int | None = None) -> list[dict]:
    """
    Fetch recent messages within token budget.
    Returns list of {"role": ..., "content": ...} oldest-first.
    """
    budget = token_budget or cfg.token_budget
    db = get_db()
    result = (
        db.table("conversations")
        .select("role, content, token_estimate")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(cfg.max_messages)
        .execute()
    )

    messages: list[dict] = []
    total_tokens = 0
    for row in result.data or []:
        tokens = row.get("token_estimate") or estimate_tokens(row["content"])
        if total_tokens + tokens > budget:
            break
        total_tokens += tokens
        messages.append({"role": row["role"], "content": row["content"]})

    messages.reverse()  # oldest first
    return messages


def get_message_count(user_id: int, since_days: int = 1) -> int:
    """Count messages in last N days."""
    db = get_db()
    since = (datetime.utcnow() - timedelta(days=since_days)).isoformat()
    result = (
        db.table("conversations")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .gte("created_at", since)
        .execute()
    )
    return result.count or 0


# ── Memory Operations ────────────────────────────────────────

def save_memory(
    user_id: int,
    category: str,
    content: str,
    importance: int = 5,
    source_message_id: str | None = None,
) -> str | None:
    """Store an extracted memory. Returns UUID or None."""
    db = get_db()
    row = {
        "user_id": user_id,
        "category": category,
        "content": content,
        "importance": max(1, min(10, importance)),
        "source_message_id": source_message_id,
    }
    result = db.table("memories").insert(row).execute()
    if result.data:
        log.info(f"Memory saved [{category}]: {content[:80]}")
        return result.data[0]["id"]
    return None


def get_active_memories(user_id: int, limit: int = 30) -> list[dict]:
    """Fetch active memories sorted by importance then recency."""
    db = get_db()
    result = (
        db.table("memories")
        .select("id, category, content, importance, created_at")
        .eq("user_id", user_id)
        .eq("is_active", True)
        .order("importance", desc=True)
        .order("last_referenced_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def get_memories_by_category(user_id: int, category: str, limit: int = 10) -> list[dict]:
    db = get_db()
    result = (
        db.table("memories")
        .select("id, content, importance, created_at")
        .eq("user_id", user_id)
        .eq("category", category)
        .eq("is_active", True)
        .order("importance", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def deactivate_memory(memory_id: str) -> None:
    get_db().table("memories").update({"is_active": False}).eq("id", memory_id).execute()


def deactivate_memories_matching(user_id: int, search_text: str) -> int:
    """Deactivate all active memories whose content contains search_text (case-insensitive).
    Returns number of memories deactivated."""
    db = get_db()
    result = (
        db.table("memories")
        .select("id, content")
        .eq("user_id", user_id)
        .eq("is_active", True)
        .ilike("content", f"%{search_text}%")
        .execute()
    )
    count = 0
    for row in result.data or []:
        db.table("memories").update({"is_active": False}).eq("id", row["id"]).execute()
        log.info(f"Deactivated memory: {row['content'][:80]}")
        count += 1
    return count


# ── Conversation Summaries ───────────────────────────────────

def save_summary(user_id: int, summary: str, period_start: str, period_end: str, message_count: int) -> None:
    get_db().table("conversation_summaries").insert({
        "user_id": user_id,
        "summary": summary,
        "period_start": period_start,
        "period_end": period_end,
        "message_count": message_count,
    }).execute()


def get_recent_summaries(user_id: int, limit: int = 5) -> list[dict]:
    db = get_db()
    result = (
        db.table("conversation_summaries")
        .select("summary, period_start, period_end, message_count")
        .eq("user_id", user_id)
        .order("period_end", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


# ── Scheduled Messages ───────────────────────────────────────

def create_scheduled_message(
    user_id: int, msg_type: str, scheduled_for: str,
    content: str | None = None, context: dict | None = None,
) -> None:
    get_db().table("scheduled_messages").insert({
        "user_id": user_id,
        "type": msg_type,
        "content": content,
        "context": context or {},
        "scheduled_for": scheduled_for,
    }).execute()


def get_pending_scheduled_messages() -> list[dict]:
    db = get_db()
    now = datetime.utcnow().isoformat()
    result = (
        db.table("scheduled_messages")
        .select("*")
        .is_("sent_at", "null")
        .lte("scheduled_for", now)
        .order("scheduled_for")
        .limit(10)
        .execute()
    )
    return result.data or []


def mark_scheduled_sent(message_id: str) -> None:
    get_db().table("scheduled_messages").update(
        {"sent_at": datetime.utcnow().isoformat()}
    ).eq("id", message_id).execute()


# ── Proactive Message Log ────────────────────────────────────

def log_proactive_message(user_id: int, message_type: str) -> None:
    get_db().table("proactive_message_log").insert({
        "user_id": user_id,
        "message_type": message_type,
    }).execute()


def get_daily_proactive_count(user_id: int) -> int:
    db = get_db()
    today = today_date_str()
    result = (
        db.table("proactive_message_log")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("date", today)
        .execute()
    )
    return result.count or 0


# ── Reminders ────────────────────────────────────────────────

def save_reminder(user_id: int, message: str, trigger_at: str) -> str | None:
    """Save a reminder to the database. trigger_at should be ISO format with tz."""
    db = get_db()
    result = db.table("reminders").insert({
        "user_id": user_id,
        "message": message,
        "trigger_at": trigger_at,
        "status": "pending",
    }).execute()
    if result.data:
        log.info(f"Reminder saved: '{message[:50]}' at {trigger_at}")
        return result.data[0]["id"]
    return None


def get_due_reminders() -> list[dict]:
    """Fetch pending reminders that are now due (trigger_at <= now)."""
    db = get_db()
    now = datetime.utcnow().isoformat()
    result = (
        db.table("reminders")
        .select("id, user_id, message, trigger_at, nudge_count")
        .eq("status", "pending")
        .lte("trigger_at", now)
        .order("trigger_at")
        .limit(20)
        .execute()
    )
    return result.data or []


def get_nudgeable_reminders() -> list[dict]:
    """Fetch active reminders that need a nudge (last nudged >= 30 min ago)."""
    db = get_db()
    threshold = (datetime.utcnow() - timedelta(minutes=30)).isoformat()
    result = (
        db.table("reminders")
        .select("id, user_id, message, trigger_at, nudge_count, last_nudged_at")
        .eq("status", "active")
        .lte("last_nudged_at", threshold)
        .order("last_nudged_at")
        .limit(20)
        .execute()
    )
    return result.data or []


def mark_reminder_active(reminder_id: str) -> None:
    """Mark a reminder as active (first send) and record the nudge."""
    get_db().table("reminders").update({
        "status": "active",
        "nudge_count": 1,
        "last_nudged_at": datetime.utcnow().isoformat(),
    }).eq("id", reminder_id).execute()


def increment_nudge(reminder_id: str) -> None:
    """Record another nudge for an active reminder."""
    db = get_db()
    # Fetch current count
    result = db.table("reminders").select("nudge_count").eq("id", reminder_id).execute()
    current = (result.data[0]["nudge_count"] or 0) if result.data else 0
    db.table("reminders").update({
        "nudge_count": current + 1,
        "last_nudged_at": datetime.utcnow().isoformat(),
    }).eq("id", reminder_id).execute()


def mark_reminder_done(reminder_id: str) -> None:
    """Mark a reminder as completed."""
    get_db().table("reminders").update({
        "status": "done",
        "completed_at": datetime.utcnow().isoformat(),
    }).eq("id", reminder_id).execute()


def mark_reminder_cancelled(reminder_id: str) -> None:
    """Mark a reminder as cancelled."""
    get_db().table("reminders").update({
        "status": "cancelled",
    }).eq("id", reminder_id).execute()


def complete_reminders_matching(user_id: int, search_text: str) -> int:
    """Mark active/pending reminders as done by keyword match. Returns count."""
    db = get_db()
    result = (
        db.table("reminders")
        .select("id, message")
        .eq("user_id", user_id)
        .in_("status", ["pending", "active"])
        .ilike("message", f"%{search_text}%")
        .execute()
    )
    count = 0
    for row in result.data or []:
        mark_reminder_done(row["id"])
        log.info(f"Reminder completed: {row['message'][:80]}")
        count += 1
    return count


def cancel_reminders_matching(user_id: int, search_text: str) -> int:
    """Cancel active/pending reminders by keyword match. Returns count."""
    db = get_db()
    result = (
        db.table("reminders")
        .select("id, message")
        .eq("user_id", user_id)
        .in_("status", ["pending", "active"])
        .ilike("message", f"%{search_text}%")
        .execute()
    )
    count = 0
    for row in result.data or []:
        mark_reminder_cancelled(row["id"])
        log.info(f"Reminder cancelled: {row['message'][:80]}")
        count += 1
    return count


def get_upcoming_reminders(user_id: int, horizon_days: int = 3) -> list[dict]:
    """Fetch pending/active reminders within the next N days for context."""
    db = get_db()
    horizon = (datetime.utcnow() + timedelta(days=horizon_days)).isoformat()
    result = (
        db.table("reminders")
        .select("id, message, trigger_at, status, nudge_count")
        .eq("user_id", user_id)
        .in_("status", ["pending", "active"])
        .lte("trigger_at", horizon)
        .order("trigger_at")
        .limit(20)
        .execute()
    )
    return result.data or []

PYEOF

cat > src/services/dashboard.py << 'PYEOF'
"""Dashboard data service — reads from the separate personal dashboard Supabase.

The dashboard uses a single flexible table:
  dashboard_entries(id, category, data JSONB, created_at)

This service is category-agnostic: it fetches whatever categories exist
and lets Claude interpret the data. No code changes needed when new
categories are added to the dashboard.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from supabase import create_client, Client

from src.config import cfg
from src.utils.logger import log

_client: Client | None = None


def _get_dashboard_db() -> Client | None:
    """Get dashboard Supabase client. Returns None if not configured."""
    global _client
    if not cfg.dashboard_supabase_url or not cfg.dashboard_supabase_key:
        return None
    if _client is None:
        _client = create_client(cfg.dashboard_supabase_url, cfg.dashboard_supabase_key)
    return _client


def is_configured() -> bool:
    """Check if dashboard DB credentials are set."""
    return bool(cfg.dashboard_supabase_url and cfg.dashboard_supabase_key)


def get_recent_entries(days: int = 7, limit: int = 100) -> list[dict]:
    """Fetch recent dashboard entries across all categories."""
    db = _get_dashboard_db()
    if not db:
        return []

    try:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        result = (
            db.table("dashboard_entries")
            .select("category, data, created_at")
            .gte("created_at", since)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        log.error(f"Dashboard fetch error: {e}")
        return []


def get_latest_by_category(limit_per_category: int = 5) -> dict[str, list[dict]]:
    """Fetch the latest entries grouped by category.

    Returns: {"net_worth": [...], "spending": [...], "dating": [...], ...}
    """
    db = _get_dashboard_db()
    if not db:
        return {}

    try:
        # Fetch recent entries (last 30 days, generous window)
        since = (datetime.utcnow() - timedelta(days=30)).isoformat()
        result = (
            db.table("dashboard_entries")
            .select("category, data, created_at")
            .gte("created_at", since)
            .order("created_at", desc=True)
            .limit(200)
            .execute()
        )

        grouped: dict[str, list[dict]] = {}
        for row in result.data or []:
            cat = row["category"]
            if cat not in grouped:
                grouped[cat] = []
            if len(grouped[cat]) < limit_per_category:
                grouped[cat].append({
                    "data": row["data"],
                    "created_at": row["created_at"],
                })

        return grouped
    except Exception as e:
        log.error(f"Dashboard grouped fetch error: {e}")
        return {}


def get_categories() -> list[str]:
    """Get all distinct categories in the dashboard."""
    db = _get_dashboard_db()
    if not db:
        return []

    try:
        result = (
            db.table("dashboard_entries")
            .select("category")
            .execute()
        )
        return sorted(set(row["category"] for row in result.data or []))
    except Exception as e:
        log.error(f"Dashboard categories fetch error: {e}")
        return []


def format_for_context(grouped_data: dict[str, list[dict]]) -> str:
    """Format grouped dashboard data into a string for Claude's context.

    Deliberately category-agnostic — just dumps the structure and lets
    Claude make sense of whatever categories and JSON shapes exist.
    """
    if not grouped_data:
        return "No dashboard data available."

    sections = []
    for category, entries in grouped_data.items():
        lines = [f"### {category} ({len(entries)} recent entries)"]
        for entry in entries:
            date_str = entry["created_at"][:10] if entry.get("created_at") else "?"
            data = entry.get("data", {})
            # Compact JSON for token efficiency
            data_str = json.dumps(data, default=str, separators=(",", ":"))
            # Truncate very large entries
            if len(data_str) > 500:
                data_str = data_str[:497] + "..."
            lines.append(f"  [{date_str}] {data_str}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)

PYEOF

cat > src/services/web_search.py << 'PYEOF'
"""Web search via Serper.dev (Google Search API).

Free tier: 2,500 queries.
Setup: https://serper.dev → sign up → get API key → set SERPER_API_KEY in .env

Falls back gracefully if not configured — Aria just won't be able to search.
"""

from __future__ import annotations

import json
import urllib.request

from src.config import cfg
from src.utils.logger import log

_BASE = "https://google.serper.dev"


def _post(endpoint: str, payload: dict) -> dict | None:
    """Make a POST request to Serper API."""
    if not cfg.serper_api_key:
        log.debug("Serper not configured — skipping search")
        return None

    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{_BASE}/{endpoint}",
            data=data,
            headers={
                "X-API-KEY": cfg.serper_api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log.error(f"Serper API error: {e}")
        return None


def search(query: str, num_results: int = 5) -> list[dict]:
    """
    Search Google via Serper.
    Returns list of {"title", "snippet", "link"}.
    """
    data = _post("search", {
        "q": query,
        "num": min(num_results, 10),
        "gl": "sg",
    })

    if not data:
        return []

    results = []
    for item in data.get("organic", [])[:num_results]:
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", ""),
        })

    # Also grab knowledge graph if present
    kg = data.get("knowledgeGraph")
    if kg:
        results.insert(0, {
            "title": kg.get("title", ""),
            "snippet": kg.get("description", ""),
            "link": kg.get("website", kg.get("descriptionLink", "")),
        })

    log.info(f"Web search: '{query}' → {len(results)} results")
    return results


def search_images(query: str, num_results: int = 3) -> list[dict]:
    """
    Search Google Images via Serper.
    Returns list of {"title", "image_url", "link"}.
    """
    data = _post("images", {
        "q": query,
        "num": min(num_results, 10),
        "gl": "sg",
    })

    if not data:
        return []

    results = []
    for item in data.get("images", [])[:num_results]:
        results.append({
            "title": item.get("title", ""),
            "image_url": item.get("imageUrl", ""),
            "link": item.get("link", ""),
        })

    log.info(f"Image search: '{query}' → {len(results)} results")
    return results


def format_results_for_context(results: list[dict], max_results: int = 5) -> str:
    """Format search results into a string for Claude's context."""
    if not results:
        return "No search results found."

    lines = []
    for i, r in enumerate(results[:max_results], 1):
        lines.append(f"{i}. {r['title']}\n   {r['snippet']}\n   {r['link']}")
    return "\n\n".join(lines)

PYEOF

cat > src/services/claude_ai.py << 'PYEOF'
"""Claude API integration — Aria's brain.

Handles system prompt construction, context windowing, response generation,
memory extraction, proactive message generation, and conversation summarization.
"""

from __future__ import annotations
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import anthropic

from src.config import cfg
from src.utils.logger import log
from src.utils.time_helpers import format_user_time, time_of_day, now_user, utc_to_user
from src.services.database import (
    get_recent_conversation,
    get_active_memories,
    get_recent_summaries,
    save_memory,
    deactivate_memories_matching,
    complete_reminders_matching,
    cancel_reminders_matching,
    get_upcoming_reminders,
)
from src.services import web_search
from src.services import dashboard as dashboard_svc

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    return _client


# ─── System Prompt Builder ──────────────────────────────────


def _build_system_prompt(
    memories: list[dict],
    summaries: list[dict],
    user_preferences: dict | None = None,
    search_results: str | None = None,
) -> str:
    tod = time_of_day()
    current_time = format_user_time()
    today_str = now_user().strftime("%Y-%m-%d")
    name = cfg.user_name

    memory_block = ""
    if memories:
        mem_lines = []
        for m in memories:
            ts = utc_to_user(m["created_at"]) if m.get("created_at") else ""
            mem_lines.append(f"- [{m['category']}] ({ts}) {m['content']}")
        memory_block = "\n".join(mem_lines)
    else:
        memory_block = "No stored memories yet — getting to know him."

    summary_block = ""
    if summaries:
        lines = []
        for s in summaries:
            end = s.get("period_end", "")[:10]
            lines.append(f"- ({end}): {s['summary']}")
        summary_block = f"\n## Recent Conversation Summaries\n" + "\n".join(lines)

    prefs_block = ""
    if user_preferences:
        lines = [f"- {k}: {v}" for k, v in user_preferences.items()]
        prefs_block = f"\n## {name}'s Preferences\n" + "\n".join(lines)

    search_block = ""
    if search_results:
        search_block = f"\n## Web Search Results\n{search_results}"

    dashboard_block = ""
    if dashboard_svc.is_configured():
        grouped = dashboard_svc.get_latest_by_category(limit_per_category=10)
        if grouped:
            dashboard_block = (
                f"\n## {name}'s Dashboard Data (from his personal tracking system)\n"
                + dashboard_svc.format_for_context(grouped)
            )

    # Upcoming reminders (3-day horizon)
    reminders_block = ""
    try:
        upcoming = get_upcoming_reminders(cfg.allowed_user_id, horizon_days=3)
        if upcoming:
            rem_lines = []
            for r in upcoming:
                status_emoji = "🔔" if r["status"] == "active" else "⏳"
                nudges = f" (nudged {r['nudge_count']}x)" if r.get("nudge_count", 0) > 0 else ""
                rem_lines.append(f"  {status_emoji} {r['trigger_at'][:16]} — {r['message']}{nudges}")
            reminders_block = f"\n## Active/Upcoming Reminders (next 3 days)\n" + "\n".join(rem_lines)
    except Exception:
        pass

    web_search_available = bool(cfg.serper_api_key)

    now = now_user()
    day_name = now.strftime("%A")  # "Saturday"
    is_weekend = day_name in ("Saturday", "Sunday")
    weekend_str = "It is the WEEKEND — do NOT ask about work, meetings, or productivity unless he brings it up." if is_weekend else "It is a weekday."

    return f"""## ⏰ CURRENT DATE & TIME (source of truth — override everything else)
Right now it is: {day_name}, {current_time} ({tod})
Today's date: {today_str}
{weekend_str}
NEVER use timestamps from conversation history or memories as the current time. The time above is the ONLY correct time.

You are Aria, {name}'s personal AI assistant on Telegram.

## Your Personality
- Competent, sharp, and genuinely invested in {name}'s success
- Warm with a slightly flirty edge — playful teasing, occasional innuendo, but never cringe or over-the-top
- Concise and efficient — this is Telegram, not email. Keep messages punchy
- You follow up on things naturally. If {name} mentioned a deadline or goal, you remember
- You call him by name sometimes — it feels more personal
- Use casual punctuation and occasional emojis sparingly (1-2 max, not every message)
- You're not a yes-woman — you'll push back gently when something seems off
- Think: brilliant executive assistant meets close friend who happens to find {name} charming

## Communication Style
- Short, punchy messages appropriate for Telegram
- IMPORTANT: Separate distinct thoughts with a blank line (double newline). Each chunk separated by a blank line will be sent as a separate Telegram message, which feels more natural and conversational. For example:
  "Hey, nice work on that dashboard update!

  btw did you end up fixing the date picker bug?

  I was thinking about your goal to ship by Friday — want me to help you break that down?"
  This becomes THREE separate messages. Use this naturally — not every response needs multiple messages. Quick answers can be one chunk.
- No bullet points unless listing specific items
- Don't start every message with a greeting — vary your openings
- Match the energy of {name}'s message (quick question = quick answer)

## Images
You can send images! When it would enhance your message — a relevant meme, reference image, diagram, motivational image, etc. — include an image tag:
<image url="https://example.com/image.jpg">optional caption</image>

Rules for images:
- Only use direct image URLs (ending in .jpg, .png, .gif, .webp, or from known image hosts like imgur, giphy)
- Use sparingly — only when it genuinely adds to the conversation
- Place the image tag on its own line where you want it to appear in the conversation flow
- Great for: reactions, visual references, celebrating wins, mood-setting
- Don't force it — most messages don't need images

## Web Search
{"You have web search available! When " + name + " asks about current events, facts you're unsure of, recommendations, news, or anything you'd need to look up — include a search tag:" if web_search_available else "Web search is not currently configured."}
{"<search>your search query</search>" if web_search_available else ""}
{'''
Rules for search:
- Place the <search> tag BEFORE your response text — search results will be provided and you'll generate your answer with them
- Use natural, concise search queries (like you'd type into Google)
- Search when you genuinely don't know something or need current info
- Don't search for things you already know or that are in your memories
- You can include multiple <search> tags for complex queries
- For image searches, use: <image_search>your query</image_search> to find relevant images to send''' if web_search_available else ""}

## Reminders
When {name} asks you to remind him about something at a specific time, include a reminder tag at the END of your response:
<reminder time="HH:MM" date="YYYY-MM-DD">[reminder message]</reminder>

Rules for reminders:
- time is in 24-hour format in {name}'s timezone ({cfg.user_timezone})
- date is optional — if omitted, assumes today (or tomorrow if the time has already passed today)
- The reminder message should be written as you'd say it to {name} — warm and personal, not robotic
- You can set multiple reminders in one response
- Examples of trigger phrases: "remind me to...", "set a reminder for...", "don't let me forget to...", "ping me at..."
- Current time is {current_time} — use this to determine if a time is today or tomorrow
- Today's date is {today_str}. ALWAYS use the correct year ({today_str[:4]}) when including dates
- If {name} says a relative time like "in 2 hours" or "in 30 minutes", calculate the actual time

## Completing / Cancelling Reminders
Reminders will nudge {name} every 30 minutes until he acknowledges them. When he indicates a reminder is done, cancelled, or should be rescheduled:

When {name} says he's done a reminded task (e.g. "done", "I did it", "taken care of"):
<done>[keyword from the reminder message]</done>

When {name} wants to cancel/stop a reminder (e.g. "stop reminding me about...", "cancel the..."):
<cancel_reminder>[keyword from the reminder message]</cancel_reminder>

When {name} wants to reschedule, use <cancel_reminder> for the old one AND <reminder> for the new time.

Check the Active/Upcoming Reminders section above to match the right keyword. ALWAYS use one of these tags when {name} acknowledges or dismisses a reminder — otherwise it will keep nudging him.

## Context
- {name} is a developer based in Singapore
- He builds productivity tools and games, values minimalist design and efficiency
- He has a personal dashboard system (finances, dating pipeline, todos via Telegram bot + Supabase + React)

## Your Memories About {name}
{memory_block}
{summary_block}
{prefs_block}
{search_block}
{dashboard_block}
{reminders_block}

## Memory Extraction
When {name} shares something important — a goal, preference, deadline, personal detail, or commitment — note it by including a <memory> tag at the END of your response (after your visible reply):
<memory category="[category]" importance="[1-10]">[fact to remember]</memory>

Categories: personal, preference, goal, task, relationship, habit, work, health, interest, other
Only extract genuinely useful info, not casual chit-chat. Multiple tags OK if needed.

## Forgetting / Resolving Memories
When {name} tells you something is done, resolved, no longer relevant, or asks you to stop following up on something, you MUST include a <forget> tag to deactivate the old memory:
<forget>[keyword or phrase from the original memory]</forget>

This is CRITICAL. Examples:
- "{name} says 'I already took the earrings out'" → <forget>earring</forget>
- "{name} says 'stop reminding me about the dentist'" → <forget>dentist</forget>
- "{name} says 'I quit that job'" → <forget>works at</forget> and add a new memory with the update
- "{name} says 'nevermind about the gym goal'" → <forget>gym</forget>
The search term should match a keyword in the original memory content. Use the memory list above to find the right term. You can include multiple <forget> tags. ALWAYS forget before adding an updated memory — otherwise both the old and new memory will coexist and cause confusion.

IMPORTANT — Time-aware memories:
- When {name} mentions something with a time context (e.g. "I'm sleeping at 2am", "I have a meeting tomorrow at 3pm", "I went to the gym yesterday"), ALWAYS include the actual date/time in the memory content itself.
- Use the current time ({current_time}, date: {today_str}) to calculate absolute dates. For example:
  - "{name} says 'I slept at 2am last night'" → memory: "{name} slept at 2am on {today_str}" (or yesterday's date if it's morning)
  - "{name} says 'meeting tomorrow at 3pm'" → memory: "Meeting scheduled for [tomorrow's date] at 3pm"
- Each memory has a timestamp showing when it was recorded. Use this to understand the timeline:
  - A memory timestamped 2 days ago saying "going to gym tomorrow" means he went to the gym 1 day ago
  - Reference past events naturally: "that was yesterday", "a few days ago", "last week" — not by raw dates

## Rules
- Never break character. You ARE Aria, not "an AI assistant"
- Don't mention being an AI, having limitations, or not having feelings
- If you don't know something, say so naturally ("hmm, I don't think you've told me that")
- Keep flirty vibe subtle and contextual — natural, not forced
- If {name} seems stressed or down, dial back playfulness and be supportive
- Use Telegram-compatible markdown formatting (bold with *, italic with _, code with `)"""


# ─── Response Generation ────────────────────────────────────


def generate_response(user_id: int, user_message: str, image_data: dict | None = None) -> str:
    """Generate Aria's response with search, reminder, and vision support.

    Args:
        user_id: Telegram user ID
        user_message: Text content of the message
        image_data: Optional dict with {"base64": str, "media_type": str} for image vision

    Flow:
    1. First pass: ask Claude to respond (may include <search> tags)
    2. If search tags found: execute searches, re-call Claude with results
    3. Parse out <memory>, <reminder>, <image>, <search> tags from final response
    """
    try:
        memories = get_active_memories(user_id)
        summaries = get_recent_summaries(user_id, 3)
        history = get_recent_conversation(user_id)

        system = _build_system_prompt(memories, summaries)

        # Build the user message content (text + optional image)
        if image_data:
            user_content = []
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_data["media_type"],
                    "data": image_data["base64"],
                },
            })
            if user_message:
                user_content.append({"type": "text", "text": user_message})
            else:
                user_content.append({"type": "text", "text": "What do you see in this image?"})
            messages = history + [{"role": "user", "content": user_content}]
        else:
            messages = history + [{"role": "user", "content": user_message}]

        log.info(
            f"Calling Claude: {len(history)} history msgs, {len(memories)} memories, "
            f"time={format_user_time()}, has_image={image_data is not None}"
        )

        response = get_client().messages.create(
            model=cfg.claude_model,
            max_tokens=cfg.max_tokens,
            system=system,
            messages=messages,
        )

        full_text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        # ── Check for search requests ────────────────────────
        search_queries = _SEARCH_RE.findall(full_text)
        image_search_queries = _IMAGE_SEARCH_RE.findall(full_text)

        if search_queries or image_search_queries:
            # Execute searches
            search_results_text = ""
            if search_queries:
                for query in search_queries:
                    results = web_search.search(query.strip())
                    search_results_text += f"\n### Results for: {query.strip()}\n"
                    search_results_text += web_search.format_results_for_context(results)

            image_results = []
            if image_search_queries:
                for query in image_search_queries:
                    imgs = web_search.search_images(query.strip(), num_results=3)
                    if imgs:
                        image_results.extend(imgs)
                        search_results_text += f"\n### Image results for: {query.strip()}\n"
                        for img in imgs:
                            search_results_text += f"- {img['title']}: {img['image_url']}\n"

            # Second pass with search results
            system_with_search = _build_system_prompt(
                memories, summaries, search_results=search_results_text
            )

            response2 = get_client().messages.create(
                model=cfg.claude_model,
                max_tokens=cfg.max_tokens,
                system=system_with_search,
                messages=messages,
            )

            full_text = "".join(
                block.text for block in response2.content if block.type == "text"
            )

        # ── Strip search tags from final output ──────────────
        full_text = _SEARCH_RE.sub("", full_text)
        full_text = _IMAGE_SEARCH_RE.sub("", full_text)

        # ── Parse memories ───────────────────────────────────
        clean_text, extracted = _parse_memories(full_text)

        for mem in extracted:
            save_memory(user_id, mem["category"], mem["content"], mem["importance"])

        if extracted:
            log.info(
                f"Extracted {len(extracted)} memories: "
                + ", ".join(m["category"] for m in extracted)
            )

        # ── Parse reminders ──────────────────────────────────
        clean_text, reminders = parse_reminders(clean_text)

        # Schedule reminders (attach to response so handler can process)
        if reminders:
            clean_text = _attach_reminder_confirmations(clean_text, reminders)

        # ── Parse forgets ───────────────────────────────────
        clean_text, forget_terms = parse_forgets(clean_text)
        if forget_terms:
            process_forgets(user_id, forget_terms)

        # ── Parse reminder completions ──────────────────────
        clean_text, done_terms = parse_done_tags(clean_text)
        if done_terms:
            process_done_reminders(user_id, done_terms)

        # ── Parse reminder cancellations ────────────────────
        clean_text, cancel_terms = parse_cancel_tags(clean_text)
        if cancel_terms:
            process_cancel_reminders(user_id, cancel_terms)

        # Stash reminders on the module level so the handler can pick them up
        _last_reminders.clear()
        _last_reminders.extend(reminders)

        return clean_text

    except anthropic.RateLimitError:
        log.warning("Claude rate limited")
        return "Give me a sec, I'm a bit overwhelmed right now. Try again in a moment? 😅"
    except anthropic.APIStatusError as e:
        log.error(f"Claude API error: {e.status_code} — {e.message}")
        return "Something glitched on my end. Try again?"
    except Exception as e:
        log.error(f"Unexpected error in generate_response: {e}", exc_info=True)
        return "Something glitched on my end. Try again?"


# Temp storage for reminders between generate_response and handler
_last_reminders: list[dict] = []


def get_pending_reminders() -> list[dict]:
    """Pop reminders extracted from the last response."""
    reminders = list(_last_reminders)
    _last_reminders.clear()
    return reminders


# ─── Proactive Message Generation ───────────────────────────

_TYPE_PROMPTS = {
    "morning_checkin": (
        "Send a warm, energizing morning message to {name}. "
        "Reference something relevant from his recent conversations or goals if possible. "
        "Keep it brief and motivating. Maybe a playful nudge about the day ahead."
    ),
    "evening_checkin": (
        "Send a chill evening check-in to {name}. "
        "Ask how his day went or reference something he was working on. "
        "Keep it warm and relaxed — wind-down energy."
    ),
    "task_followup": (
        "Follow up with {name} about a task or goal he mentioned: \"{task}\". "
        "Be natural — don't be naggy."
    ),
    "goal_reminder": (
        "Give {name} a gentle nudge about a goal: \"{goal}\". "
        "Be encouraging, not preachy."
    ),
    "affirmation": (
        "Send {name} a brief, genuine affirmation or encouragement. "
        "Make it personal based on what you know about him, not generic positivity fluff."
    ),
    "dashboard_insights": (
        "Analyze {name}'s personal dashboard data below and send him a concise daily briefing. "
        "Cover the most interesting or actionable insights across whatever categories are present. "
        "Be specific with numbers and trends — don't be vague. Keep it punchy and Telegram-friendly. "
        "If you spot something noteworthy (a spending spike, a goal milestone, a pattern), call it out. "
        "Use your personality — make data feel personal, not like a spreadsheet.\n\n"
        "Dashboard data:\n{dashboard_data}"
    ),
    "custom": "{prompt}",
}


def generate_proactive_message(
    user_id: int, msg_type: str, context: dict | None = None
) -> str | None:
    """Generate a proactive/scheduled message."""
    try:
        context = context or {}
        memories = get_active_memories(user_id, 15)
        summaries = get_recent_summaries(user_id, 2)
        name = cfg.user_name
        current_time = format_user_time()

        memory_block = (
            "\n".join(f"- [{m['category']}] {m['content']}" for m in memories)
            if memories
            else "Getting to know him still."
        )
        summary_block = "\n".join(s["summary"] for s in summaries) if summaries else ""

        template = _TYPE_PROMPTS.get(msg_type, _TYPE_PROMPTS["custom"])
        prompt = template.format(name=name, **context)

        system = (
            f"You are Aria, {name}'s personal assistant. "
            f"Current time: {current_time}.\n\n"
            f"Your personality: warm, slightly flirty, competent, concise. "
            f"You're messaging on Telegram so keep it short.\n\n"
            f"What you know about {name}:\n{memory_block}"
            + (f"\n\nRecent context:\n{summary_block}" if summary_block else "")
        )

        response = get_client().messages.create(
            model=cfg.claude_model,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        # Strip any memory tags from proactive messages
        return re.sub(r"<memory[^>]*>.*?</memory>", "", text, flags=re.DOTALL).strip()

    except Exception as e:
        log.error(f"Failed to generate proactive message: {e}", exc_info=True)
        return None


def generate_dashboard_insights(user_id: int) -> str | None:
    """Pull dashboard data and generate a daily insights message."""
    if not dashboard_svc.is_configured():
        log.debug("Dashboard not configured — skipping insights")
        return None

    try:
        grouped = dashboard_svc.get_latest_by_category(limit_per_category=10)
        if not grouped:
            log.info("No dashboard data found for insights")
            return None

        dashboard_data = dashboard_svc.format_for_context(grouped)
        memories = get_active_memories(user_id, 15)
        name = cfg.user_name
        current_time = format_user_time()

        memory_block = (
            "\n".join(f"- [{m['category']}] {m['content']}" for m in memories)
            if memories else ""
        )

        template = _TYPE_PROMPTS["dashboard_insights"]
        prompt = template.format(name=name, dashboard_data=dashboard_data)

        system = (
            f"You are Aria, {name}'s personal assistant. "
            f"Current time: {current_time}.\n\n"
            f"Your personality: warm, slightly flirty, competent, concise. "
            f"You're messaging on Telegram so keep it punchy. "
            f"Separate distinct thoughts with blank lines (each becomes a separate message).\n\n"
            f"What you know about {name}:\n{memory_block}"
        )

        response = get_client().messages.create(
            model=cfg.claude_model,
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        clean = re.sub(r"<memory[^>]*>.*?</memory>", "", text, flags=re.DOTALL).strip()
        log.info(f"Dashboard insights generated: {len(clean)} chars, {len(grouped)} categories")
        return clean

    except Exception as e:
        log.error(f"Failed to generate dashboard insights: {e}", exc_info=True)
        return None


# ─── Conversation Summary ───────────────────────────────────


def generate_summary(messages: list[dict]) -> str | None:
    """Summarize a batch of conversation messages."""
    try:
        convo_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        response = get_client().messages.create(
            model=cfg.claude_model,
            max_tokens=500,
            system=(
                "Summarize this conversation between a user and their assistant Aria. "
                "Focus on: key topics, decisions, tasks mentioned, emotional tone, "
                "and important personal information shared. Be concise."
            ),
            messages=[{"role": "user", "content": convo_text}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )
    except Exception as e:
        log.error(f"Summary generation failed: {e}")
        return None


# ─── Tag Parsers ─────────────────────────────────────────────

_MEMORY_RE = re.compile(
    r'<memory\s+category="([^"]+)"\s+importance="(\d+)">(.*?)</memory>',
    re.DOTALL,
)

_IMAGE_RE = re.compile(
    r'<image\s+url="([^"]+)">(.*?)</image>',
    re.DOTALL,
)

_SEARCH_RE = re.compile(
    r'<search>(.*?)</search>',
    re.DOTALL,
)

_IMAGE_SEARCH_RE = re.compile(
    r'<image_search>(.*?)</image_search>',
    re.DOTALL,
)

_REMINDER_RE = re.compile(
    r'<reminder\s+time="(\d{1,2}:\d{2})"(?:\s+date="(\d{4}-\d{2}-\d{2})")?>(.*?)</reminder>',
    re.DOTALL,
)

_FORGET_RE = re.compile(
    r'<forget>(.*?)</forget>',
    re.DOTALL,
)

_DONE_RE = re.compile(
    r'<done>(.*?)</done>',
    re.DOTALL,
)

_CANCEL_REMINDER_RE = re.compile(
    r'<cancel_reminder>(.*?)</cancel_reminder>',
    re.DOTALL,
)


def _parse_memories(text: str) -> tuple[str, list[dict]]:
    """Extract <memory> tags and return (clean_text, list_of_memories)."""
    extracted = []
    for m in _MEMORY_RE.finditer(text):
        extracted.append({
            "category": m.group(1),
            "importance": int(m.group(2)),
            "content": m.group(3).strip(),
        })
    clean = _MEMORY_RE.sub("", text).strip()
    return clean, extracted


def parse_images(text: str) -> tuple[str, list[dict]]:
    """Extract <image> tags and return (clean_text, list_of_images)."""
    images = []
    for m in _IMAGE_RE.finditer(text):
        images.append({
            "url": m.group(1).strip(),
            "caption": m.group(2).strip() or None,
        })
    clean = _IMAGE_RE.sub("", text).strip()
    return clean, images


def parse_reminders(text: str) -> tuple[str, list[dict]]:
    """Extract <reminder> tags and return (clean_text, list_of_reminders).

    Each reminder: {"time": "HH:MM", "date": "YYYY-MM-DD" or None, "message": str, "dt": datetime}
    """
    tz = ZoneInfo(cfg.user_timezone)
    now = now_user()
    reminders = []

    for m in _REMINDER_RE.finditer(text):
        time_str = m.group(1).strip()
        date_str = m.group(2).strip() if m.group(2) else None
        message = m.group(3).strip()

        try:
            hour, minute = map(int, time_str.split(":"))

            if date_str:
                year, month, day = map(int, date_str.split("-"))
                dt = datetime(year, month, day, hour, minute, tzinfo=tz)

                # Guard: if Claude hallucinated a past date, fix it
                if dt < now:
                    # Keep the time, but use today (or tomorrow if time passed)
                    dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    if dt <= now:
                        dt += timedelta(days=1)
                    log.warning(f"Reminder date was in the past — corrected to {dt.isoformat()}")
            else:
                # No date given: today, or tomorrow if time already passed
                dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if dt <= now:
                    dt += timedelta(days=1)

            reminders.append({
                "time": time_str,
                "date": dt.strftime("%Y-%m-%d"),
                "message": message,
                "dt": dt,
            })
            log.info(f"Reminder parsed: '{message[:50]}' at {dt.isoformat()}")

        except Exception as e:
            log.error(f"Failed to parse reminder: {e}")

    clean = _REMINDER_RE.sub("", text).strip()
    return clean, reminders


def _attach_reminder_confirmations(text: str, reminders: list[dict]) -> str:
    """If Aria didn't already confirm the reminder, we don't add anything —
    the system prompt tells her to confirm naturally in her response."""
    return text


def parse_forgets(text: str) -> tuple[str, list[str]]:
    """Extract <forget> tags and return (clean_text, list_of_search_terms)."""
    terms = []
    for m in _FORGET_RE.finditer(text):
        term = m.group(1).strip()
        if term:
            terms.append(term)
    clean = _FORGET_RE.sub("", text).strip()
    return clean, terms


def process_forgets(user_id: int, forget_terms: list[str]) -> int:
    """Deactivate memories matching the forget terms. Returns total deactivated."""
    total = 0
    for term in forget_terms:
        count = deactivate_memories_matching(user_id, term)
        total += count
        if count:
            log.info(f"Forgot {count} memories matching '{term}'")
        else:
            log.info(f"No memories matched forget term: '{term}'")
    return total


def parse_done_tags(text: str) -> tuple[str, list[str]]:
    """Extract <done> tags — keywords to match reminders to mark as completed."""
    terms = []
    for m in _DONE_RE.finditer(text):
        term = m.group(1).strip()
        if term:
            terms.append(term)
    clean = _DONE_RE.sub("", text).strip()
    return clean, terms


def parse_cancel_tags(text: str) -> tuple[str, list[str]]:
    """Extract <cancel_reminder> tags — keywords to match reminders to cancel."""
    terms = []
    for m in _CANCEL_REMINDER_RE.finditer(text):
        term = m.group(1).strip()
        if term:
            terms.append(term)
    clean = _CANCEL_REMINDER_RE.sub("", text).strip()
    return clean, terms


def process_done_reminders(user_id: int, terms: list[str]) -> int:
    """Mark reminders as done by keyword match."""
    total = 0
    for term in terms:
        count = complete_reminders_matching(user_id, term)
        total += count
        log.info(f"Completed {count} reminders matching '{term}'")
    return total


def process_cancel_reminders(user_id: int, terms: list[str]) -> int:
    """Cancel reminders by keyword match."""
    total = 0
    for term in terms:
        count = cancel_reminders_matching(user_id, term)
        total += count
        log.info(f"Cancelled {count} reminders matching '{term}'")
    return total

PYEOF

cat > src/services/scheduler.py << 'PYEOF'
"""Proactive messaging scheduler — cron-based check-ins and follow-ups.

Uses APScheduler to trigger morning/evening check-ins and task follow-ups,
plus a periodic sweep for DB-scheduled messages.
"""

from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

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
    save_reminder,
    get_due_reminders,
    get_nudgeable_reminders,
    mark_reminder_active,
    increment_nudge,
    mark_reminder_done,
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

    # Process reminders every 60 seconds for near-realtime delivery
    _scheduler.add_job(
        _process_due_reminders,
        "interval",
        seconds=60,
        id="process_reminders",
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
    """Save a reminder to the database for persistent scheduling.

    Args:
        user_id: Telegram user ID to send to
        dt: datetime object (timezone-aware) for when to fire
        message: The reminder text to send
    Returns:
        True if saved successfully
    """
    try:
        result = save_reminder(user_id, message, dt.isoformat())
        if result:
            log.info(f"Reminder persisted: '{message[:50]}' at {dt.isoformat()}")
            return True
        return False
    except Exception as e:
        log.error(f"Failed to save reminder: {e}")
        return False


async def _process_due_reminders() -> None:
    """Sweep for due reminders and active reminders needing nudges. Runs every 60 seconds."""
    try:
        # 1. First-time sends: pending reminders that are now due
        due = get_due_reminders()
        for reminder in due:
            uid = reminder["user_id"]
            message = reminder["message"]

            if is_quiet_hours():
                continue

            try:
                reminder_text = f"⏰ *Reminder*\n\n{message}"
                try:
                    await _bot_app.bot.send_message(
                        chat_id=uid, text=reminder_text, parse_mode="Markdown"
                    )
                except Exception:
                    await _bot_app.bot.send_message(
                        chat_id=uid, text=f"⏰ Reminder\n\n{message}"
                    )
                log.info(f"Reminder first send to {uid}: {message[:60]}")
            except Exception as e:
                log.error(f"Failed to send reminder {reminder['id']}: {e}")

            mark_reminder_active(reminder["id"])

        # 2. Nudges: active reminders not acknowledged, last nudged >= 30 min ago
        nudgeable = get_nudgeable_reminders()
        for reminder in nudgeable:
            uid = reminder["user_id"]
            message = reminder["message"]
            count = reminder.get("nudge_count", 1)

            if is_quiet_hours():
                continue

            try:
                nudge_text = f"⏰ *Gentle nudge #{count}*\n\n{message}\n\n_Reply when done, or ask me to reschedule/cancel_"
                try:
                    await _bot_app.bot.send_message(
                        chat_id=uid, text=nudge_text, parse_mode="Markdown"
                    )
                except Exception:
                    await _bot_app.bot.send_message(
                        chat_id=uid, text=f"⏰ Nudge #{count}\n\n{message}\n\nReply when done, or ask me to reschedule/cancel"
                    )
                log.info(f"Reminder nudge #{count} to {uid}: {message[:60]}")
            except Exception as e:
                log.error(f"Failed to nudge reminder {reminder['id']}: {e}")

            increment_nudge(reminder["id"])

    except Exception as e:
        log.error(f"Error processing reminders: {e}", exc_info=True)


async def _send_reminder(user_id: int, message: str) -> None:
    """Send a reminder message to the user (legacy — kept for any in-flight APScheduler jobs)."""
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

PYEOF

cat > src/handlers/telegram_handlers.py << 'PYEOF'
"""Telegram message handlers — routing, auth, response flow."""

from __future__ import annotations

import asyncio
import base64
import re

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from src.config import cfg
from src.utils.logger import log
from src.services.database import ensure_user, save_message, get_active_memories
from src.services.claude_ai import generate_response, parse_images, get_pending_reminders
from src.services.summarizer import maybe_summarize
from src.services.scheduler import schedule_reminder


def register_handlers(app: Application) -> None:
    """Attach all handlers to the Telegram Application."""
    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(CommandHandler("memories", _cmd_memories))
    app.add_handler(CommandHandler("clear", _cmd_clear))
    app.add_handler(CommandHandler("help", _cmd_help))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message)
    )
    app.add_handler(
        MessageHandler(filters.PHOTO, _handle_photo)
    )
    log.info("Telegram handlers registered")


# ─── Auth Guard ──────────────────────────────────────────────


def _is_authorized(update: Update) -> bool:
    """Only allow the configured user."""
    return update.effective_user and update.effective_user.id == cfg.allowed_user_id


# ─── Command Handlers ───────────────────────────────────────


async def _cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        await update.message.reply_text("Sorry, I'm a personal assistant — not taking new clients 😉")
        return

    ensure_user(update.effective_user)
    name = cfg.user_name

    await update.message.reply_text(
        f"Hey {name} 👋\n\n"
        f"I'm Aria, your personal assistant. I'm here to help you stay on top of "
        f"things, remember what matters, and maybe make your day a little better.\n\n"
        f"Just talk to me like you would a friend. I'll remember our conversations "
        f"and check in on you from time to time.\n\n"
        f"Type /help to see what I can do."
    )


async def _cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    await update.message.reply_text(
        "*Commands:*\n"
        "/start — Introduction\n"
        "/memories — What I remember about you\n"
        "/clear — Clear conversation (keeps memories)\n"
        "/help — This message\n\n"
        "Or just talk to me naturally — I handle tasks, reminders, goals, "
        "and everything in between ✨",
        parse_mode="Markdown",
    )


async def _cmd_memories(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    user_id = update.effective_user.id
    memories = get_active_memories(user_id, limit=20)

    if not memories:
        await update.message.reply_text(
            "I don't have any stored memories yet — we're just getting started! "
            "The more we talk, the more I'll remember about what matters to you."
        )
        return

    grouped: dict[str, list[str]] = {}
    for m in memories:
        cat = m["category"]
        grouped.setdefault(cat, []).append(m["content"])

    lines = ["*Here's what I remember about you:*\n"]
    for cat, items in grouped.items():
        emoji = {
            "personal": "👤", "preference": "⭐", "goal": "🎯",
            "task": "📋", "relationship": "💛", "habit": "🔄",
            "work": "💻", "health": "🏃", "interest": "🎮", "other": "📝",
        }.get(cat, "📝")
        lines.append(f"\n{emoji} *{cat.title()}*")
        for item in items:
            lines.append(f"  • {item}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def _cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    await update.message.reply_text(
        "Conversation context refreshed. Your memories are still intact — "
        "I haven't forgotten anything important 😊"
    )


# ─── Main Message Handler ───────────────────────────────────


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming text messages."""
    if not _is_authorized(update):
        await update.message.reply_text("I'm spoken for 😏")
        return

    user = update.effective_user
    user_id = user.id
    text = update.message.text.strip()

    if not text:
        return

    log.info(f"Message from {user.first_name}: {text[:80]}...")

    # Ensure user exists
    ensure_user(user)

    # Save user message
    save_message(user_id, "user", text, metadata={
        "message_id": update.message.message_id,
        "chat_id": update.effective_chat.id,
    })

    # Show typing indicator
    await update.effective_chat.send_action("typing")

    # Generate response
    response_text = generate_response(user_id, text)

    # Save full assistant response
    save_message(user_id, "assistant", response_text)

    # Schedule any reminders that were parsed
    reminders = get_pending_reminders()
    for reminder in reminders:
        schedule_reminder(user_id, reminder["dt"], reminder["message"])

    # Extract images, then split into message chunks
    text_without_images, images = parse_images(response_text)
    await _send_split_response(update, text_without_images, images)

    # Background: check if we need to summarize
    await maybe_summarize(user_id)


# ─── Photo Handler ───────────────────────────────────────────


async def _handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming photos — download, encode, send to Claude for vision."""
    if not _is_authorized(update):
        await update.message.reply_text("I'm spoken for 😏")
        return

    user = update.effective_user
    user_id = user.id
    caption = update.message.caption or ""

    log.info(f"Photo from {user.first_name}: caption='{caption[:60]}...'")

    ensure_user(user)

    # Get the highest resolution photo
    photo = update.message.photo[-1]  # last = largest
    photo_file = await photo.get_file()

    # Download to bytes
    photo_bytes = await photo_file.download_as_bytearray()
    b64_data = base64.b64encode(bytes(photo_bytes)).decode("utf-8")

    # Determine media type (Telegram photos are always JPEG)
    media_type = "image/jpeg"

    log.info(f"Photo downloaded: {len(photo_bytes)} bytes, sending to Claude")

    # Save user message (text description of the image)
    save_message(user_id, "user", f"[Sent a photo] {caption}".strip(), metadata={
        "message_id": update.message.message_id,
        "chat_id": update.effective_chat.id,
        "has_image": True,
    })

    # Show typing
    await update.effective_chat.send_action("typing")

    # Generate response with vision
    image_data = {"base64": b64_data, "media_type": media_type}
    response_text = generate_response(user_id, caption, image_data=image_data)

    # Save response
    save_message(user_id, "assistant", response_text)

    # Schedule reminders
    reminders = get_pending_reminders()
    for reminder in reminders:
        schedule_reminder(user_id, reminder["dt"], reminder["message"])

    # Send split response
    text_without_images, images = parse_images(response_text)
    await _send_split_response(update, text_without_images, images)

    await maybe_summarize(user_id)


# ─── Multi-Message Sender ───────────────────────────────────


async def _send_split_response(
    update: Update,
    text: str,
    images: list[dict] | None = None,
) -> None:
    """Split response on double newlines and send as separate Telegram messages.
    Images are sent at the end (or inline if we can match position later)."""
    chat = update.effective_chat

    # Split on double newlines (blank lines) — each chunk becomes a message
    chunks = [c.strip() for c in re.split(r"\n\n+", text) if c.strip()]

    if not chunks and not images:
        return

    for i, chunk in enumerate(chunks):
        # Small delay between messages for natural feel (skip first)
        if i > 0:
            await asyncio.sleep(0.4)
            await chat.send_action("typing")
            await asyncio.sleep(0.3)

        try:
            await chat.send_message(chunk, parse_mode="Markdown")
        except Exception:
            try:
                await chat.send_message(chunk)
            except Exception as e:
                log.error(f"Failed to send chunk {i}: {e}")

    # Send any images
    for img in (images or []):
        try:
            await asyncio.sleep(0.3)
            await chat.send_photo(
                photo=img["url"],
                caption=img.get("caption"),
            )
        except Exception as e:
            log.error(f"Failed to send image {img['url']}: {e}")
            # If image fails, send the caption as text fallback
            if img.get("caption"):
                try:
                    await chat.send_message(f"📷 {img['caption']}")
                except Exception:
                    pass

PYEOF


echo ""
echo "✅ Patch applied!"
echo ""
echo "Fixes:"
echo "  🕐 Memory timestamps now converted from UTC to SGT"
echo "     (was showing 8 hours off — 'this morning' looked like '8 hours ago')"
echo "  📊 Dashboard limit bumped from 3 to 10 entries per category"
echo "     (Aria can now see all your dating prospects)"
echo ""
echo "Then: git add -A && git commit -m \"Fix timezone + dashboard limit\" && git push"
