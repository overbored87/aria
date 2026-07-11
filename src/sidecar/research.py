"""On-demand research: search the web, synthesize a wiki page draft.

Reuses Aria's Serper search and her GPT-4o wiki writer, so the output matches
every other page she writes. Returns an edit dict in the same shape the
Telegram approval flow already understands — drafting a new page, or revising
an existing one when a page with the target slug already exists.
"""

import re
import uuid

from src.services import web_search
from src.services import dashboard as dashboard_svc
from src.services.claude_ai import _call_wiki_writer
from src.utils.logger import log


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "untitled"


def _research(topic: str) -> str:
    queries = [topic, f"{topic} latest", f"{topic} overview"]
    blocks = []
    for q in queries:
        results = web_search.search(q, num_results=5)
        if results:
            blocks.append(f"Search '{q}':\n" + web_search.format_results_for_context(results))
    return "\n\n".join(blocks) or "No search results were available."


def run_research_and_draft(topic: str, slug: str | None = None, title: str | None = None) -> dict:
    # Create vs update is decided by whether a target page already exists. An
    # explicit slug is authoritative (research into *that* page); otherwise fall
    # back to the slug we'd derive from the title/topic, so "research X" revises
    # an existing "X" page instead of silently forking a near-duplicate.
    existing = dashboard_svc.get_wiki_page(slug) if slug else None
    if existing is None and slug is None:
        derived = _slugify(title or topic)
        existing = dashboard_svc.get_wiki_page(derived)

    research = _research(topic)

    if existing:
        slug = existing["slug"]
        title = title or existing["title"]
        brief = (
            f"Revise the existing wiki page '{title}'. This is a REVISION, not a "
            "summary — treat it as a comprehensive, verbatim edit.\n\n"
            "Below is the current page followed by fresh research. Return the FULL "
            "updated page: preserve all existing content, sections, and detail "
            "verbatim, correcting only what's now outdated and folding in genuinely "
            "new information. Do NOT condense, shorten, summarize, or drop sections "
            "— the result must be at least as long and detailed as the current page. "
            "Ignore any default length limit; length follows the source. Note "
            "anything time-sensitive with its date.\n\n"
            f"--- CURRENT PAGE ---\n{existing['content']}\n\n"
            f"--- NEW RESEARCH ---\n{research}"
        )
        edit_type = "update"
    else:
        title = title or topic.strip().title()
        slug = slug or _slugify(topic)
        brief = (
            f"Write a wiki page titled '{title}'.\n\n"
            "Synthesize the research below into a coherent page — do not just list "
            "the sources. Note anything time-sensitive with its date. If the "
            "research is thin, say so briefly rather than padding.\n\n"
            f"Research:\n{research}"
        )
        edit_type = "create"

    content = _call_wiki_writer(brief)
    log.info(f"Research {edit_type} ready: {slug} ({len(content.split())} words)")
    return {"id": uuid.uuid4().hex[:8], "type": edit_type, "slug": slug, "title": title, "content": content}
