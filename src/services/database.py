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

