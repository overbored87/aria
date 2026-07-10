"""Wiki service — reads and writes the wiki_pages table.

Backed by the same Supabase project as the personal dashboard, addressed
via the dashboard_* config keys.
"""

from __future__ import annotations

from datetime import datetime

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
            .limit(30)
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


def _markdown_to_html(text: str) -> str:
    """Basic markdown to HTML conversion for Tiptap compatibility."""
    import re as _re
    lines = text.split("\n")
    html_lines = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("")
            continue

        # Headers
        if stripped.startswith("### "):
            html_lines.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("## "):
            html_lines.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("# "):
            html_lines.append(f"<h1>{stripped[2:]}</h1>")
        # List items
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{stripped[2:]}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<p>{stripped}</p>")

    if in_list:
        html_lines.append("</ul>")

    result = "\n".join(html_lines)
    # Inline formatting
    result = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', result)
    result = _re.sub(r'\*(.+?)\*', r'<em>\1</em>', result)
    result = _re.sub(r'`(.+?)`', r'<code>\1</code>', result)
    return result


# ── Wiki Write Operations ────────────────────────────────────


def create_wiki_page(user_id: int, title: str, slug: str, content: str) -> dict | None:
    """Create a new wiki page, updating it if the slug already exists."""
    db = _get_dashboard_db()
    if not db:
        return None

    try:
        # Check if page already exists
        existing = db.table("wiki_pages").select("id").eq("slug", slug).execute()
        if existing.data:
            return update_wiki_page(slug=slug, content=content, title=title)

        result = db.table("wiki_pages").insert({
            "user_id": user_id,
            "title": title,
            "slug": slug,
            "content": content,
            "content_rendered": _markdown_to_html(content),
        }).execute()
        if result.data:
            log.info(f"Wiki page created: {title} ({slug})")
            return result.data[0]
        return None
    except Exception as e:
        log.error(f"Wiki page create error: {e}")
        return None


def update_wiki_page(slug: str, content: str, title: str | None = None) -> dict | None:
    """Update an existing wiki page's content (and optionally title)."""
    db = _get_dashboard_db()
    if not db:
        return None

    try:
        update_data = {
            "content": content,
            "content_rendered": _markdown_to_html(content),
            "updated_at": datetime.utcnow().isoformat(),
        }
        if title:
            update_data["title"] = title

        result = (
            db.table("wiki_pages")
            .update(update_data)
            .eq("slug", slug)
            .execute()
        )
        if result.data:
            log.info(f"Wiki page updated: {slug}")
            return result.data[0]
        return None
    except Exception as e:
        log.error(f"Wiki page update error: {e}")
        return None


def delete_wiki_page(slug: str) -> bool:
    """Delete a wiki page by slug."""
    db = _get_dashboard_db()
    if not db:
        return False

    try:
        result = (
            db.table("wiki_pages")
            .delete()
            .eq("slug", slug)
            .execute()
        )
        if result.data:
            log.info(f"Wiki page deleted: {slug}")
            return True
        log.warning(f"Wiki page not found for deletion: {slug}")
        return False
    except Exception as e:
        log.error(f"Wiki page delete error: {e}")
        return False


def format_wiki_titles_for_context(titles: list[dict]) -> str:
    """Format wiki titles into a compact list for Claude's context."""
    if not titles:
        return ""
    lines = [f"  - {t['title']}" for t in titles]
    return "\n".join(lines)

