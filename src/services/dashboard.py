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

# Stop words to filter out from auto-search keyword extraction
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "i", "me", "my", "mine", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "don", "now", "and", "but", "or",
    "if", "of", "as", "into", "also", "tell", "show", "give", "get",
    "know", "think", "want", "like", "make", "go", "see", "look",
    "find", "say", "said", "let", "help", "hey", "aria", "please",
    "thanks", "thank", "ok", "okay", "yeah", "yes", "no", "hi", "hello",
    "whats", "what's", "hows", "how's", "whos", "who's",
}


def extract_search_keywords(message: str) -> list[str]:
    """Extract meaningful keywords from a user message for wiki auto-search."""
    import re
    # Keep only alphanumeric and spaces
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", message.lower())
    words = clean.split()
    # Filter stop words and very short words
    keywords = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    return keywords


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


def auto_search_wiki(message: str, max_results: int = 3) -> list[dict]:
    """Automatically search wiki based on user message keywords.
    Returns matching pages ranked by relevance (number of keyword hits)."""
    db = _get_dashboard_db()
    if not db:
        return []

    keywords = extract_search_keywords(message)
    if not keywords:
        return []

    try:
        # Score each page by how many keywords match
        all_pages = {}  # slug -> {page_data, score}

        for keyword in keywords:
            # Search titles
            title_results = (
                db.table("wiki_pages")
                .select("title, slug, content, updated_at")
                .ilike("title", f"%{keyword}%")
                .limit(10)
                .execute()
            )
            for row in title_results.data or []:
                slug = row["slug"]
                if slug not in all_pages:
                    all_pages[slug] = {"page": row, "score": 0}
                all_pages[slug]["score"] += 2  # Title match worth more

            # Search content
            content_results = (
                db.table("wiki_pages")
                .select("title, slug, content, updated_at")
                .ilike("content", f"%{keyword}%")
                .limit(10)
                .execute()
            )
            for row in content_results.data or []:
                slug = row["slug"]
                if slug not in all_pages:
                    all_pages[slug] = {"page": row, "score": 0}
                all_pages[slug]["score"] += 1  # Content match

        if not all_pages:
            return []

        # Sort by score descending, return top results
        ranked = sorted(all_pages.values(), key=lambda x: x["score"], reverse=True)
        return [item["page"] for item in ranked[:max_results]]

    except Exception as e:
        log.error(f"Wiki auto-search error: {e}")
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
    """Create a new wiki page."""
    db = _get_dashboard_db()
    if not db:
        return None

    try:
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


def format_wiki_results_for_context(pages: list[dict]) -> str:
    """Format full wiki pages for Claude's context."""
    if not pages:
        return "No wiki pages matched."

    sections = []
    for page in pages:
        content = page.get("content", "")
        if len(content) > 3000:
            content = content[:3000] + "\n... (truncated)"
        sections.append(f"### {page['title']} (slug: {page['slug']})\n{content}")

    return "\n\n".join(sections)

