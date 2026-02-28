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


# ── Wiki ─────────────────────────────────────────────────────


def get_wiki_titles() -> list[dict]:
    """Fetch all wiki page titles and slugs for context listing."""
    db = _get_dashboard_db()
    if not db:
        return []

    try:
        result = (
            db.table("wiki_pages")
            .select("title, slug, updated_at")
            .order("updated_at", desc=True)
            .limit(100)
            .execute()
        )
        return result.data or []
    except Exception as e:
        log.error(f"Wiki titles fetch error: {e}")
        return []


def search_wiki(query: str, limit: int = 5) -> list[dict]:
    """Search wiki pages by title or content (case-insensitive).
    Returns matching pages with full content."""
    db = _get_dashboard_db()
    if not db:
        return []

    try:
        # Search in title
        title_results = (
            db.table("wiki_pages")
            .select("title, slug, content, updated_at")
            .ilike("title", f"%{query}%")
            .limit(limit)
            .execute()
        )

        # Search in content
        content_results = (
            db.table("wiki_pages")
            .select("title, slug, content, updated_at")
            .ilike("content", f"%{query}%")
            .limit(limit)
            .execute()
        )

        # Deduplicate by slug
        seen = set()
        results = []
        for row in (title_results.data or []) + (content_results.data or []):
            if row["slug"] not in seen:
                seen.add(row["slug"])
                results.append(row)

        return results[:limit]
    except Exception as e:
        log.error(f"Wiki search error: {e}")
        return []


def get_wiki_page(slug: str) -> dict | None:
    """Fetch a single wiki page by slug."""
    db = _get_dashboard_db()
    if not db:
        return None

    try:
        result = (
            db.table("wiki_pages")
            .select("title, slug, content, updated_at")
            .eq("slug", slug)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as e:
        log.error(f"Wiki page fetch error: {e}")
        return None


def format_wiki_titles_for_context(titles: list[dict]) -> str:
    """Format wiki titles into a compact list for Claude's context."""
    if not titles:
        return ""
    lines = [f"  - {t['title']}" for t in titles]
    return "\n".join(lines)


def format_wiki_results_for_context(pages: list[dict]) -> str:
    """Format full wiki pages for Claude's context after a search."""
    if not pages:
        return "No wiki pages matched your search."

    sections = []
    for page in pages:
        content = page.get("content", "")
        # Truncate very long pages
        if len(content) > 2000:
            content = content[:2000] + "\n... (truncated)"
        sections.append(f"### {page['title']}\n{content}")

    return "\n\n".join(sections)

