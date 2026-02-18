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

