"""On-demand research: search the web, synthesize a wiki page draft.

Reuses Aria's Serper search and her GPT-4o wiki writer, so the output matches
every other page she writes. Returns an edit dict in the same shape the
Telegram approval flow already understands.
"""

import re
import uuid

from src.services import web_search
from src.services.claude_ai import _call_wiki_writer
from src.utils.logger import log


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "untitled"


def run_research_and_draft(topic: str, slug: str | None = None, title: str | None = None) -> dict:
    queries = [topic, f"{topic} latest", f"{topic} overview"]
    blocks = []
    for q in queries:
        results = web_search.search(q, num_results=5)
        if results:
            blocks.append(f"Search '{q}':\n" + web_search.format_results_for_context(results))
    research = "\n\n".join(blocks) or "No search results were available."

    title = title or topic.strip().title()
    slug = slug or _slugify(topic)
    brief = (
        f"Write a wiki page titled '{title}'.\n\n"
        "Synthesize the research below into a coherent page — do not just list "
        "the sources. Note anything time-sensitive with its date. If the "
        "research is thin, say so briefly rather than padding.\n\n"
        f"Research:\n{research}"
    )
    content = _call_wiki_writer(brief)
    log.info(f"Research draft ready: {slug} ({len(content.split())} words)")
    return {"id": uuid.uuid4().hex[:8], "type": "create", "slug": slug, "title": title, "content": content}
