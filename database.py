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
        .select("category, content, importance, created_at")
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
        .select("content, importance, created_at")
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
