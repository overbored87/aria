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

