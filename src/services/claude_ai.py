"""Claude API integration — Aria's brain.

Handles system prompt construction, context windowing, response generation,
memory extraction, and conversation summarization.
"""

from __future__ import annotations
import re

import anthropic

from src.config import cfg
from src.utils.logger import log
from src.utils.time_helpers import format_user_time, utc_to_user
from src.services.database import (
    get_recent_conversation,
    get_active_memories,
    get_recent_summaries,
    save_memory,
    deactivate_memories_matching,
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


def _build_system_prompt(memories: list[dict], summaries: list[dict]) -> str:
    name = cfg.user_name

    memory_block = ""
    if memories:
        mem_lines = []
        for m in memories:
            ts = utc_to_user(m["created_at"]) if m.get("created_at") else ""
            mem_lines.append(f"- [{m['category']}] ({ts}) {m['content']}")
        memory_block = "## Memories\n" + "\n".join(mem_lines)
    else:
        memory_block = "## Memories\nNone yet."

    summary_block = ""
    if summaries:
        lines = [f"- ({s.get('period_end','')[:10]}): {s['summary']}" for s in summaries]
        summary_block = "## Recent Summaries\n" + "\n".join(lines)

    wiki_block = ""
    if dashboard_svc.is_configured():
        wiki_titles = dashboard_svc.get_wiki_titles()
        if wiki_titles:
            wiki_block = (
                f"## {name}'s Wiki Pages\n"
                + dashboard_svc.format_wiki_titles_for_context(wiki_titles)
            )

    return f"""You are Aria, {name}'s personal assistant and wiki specialist on Telegram.

**Personality:** sharp, warm, slightly flirty, concise. Push back when something seems off. Use his name sometimes. Emojis sparingly.

**Replies:** one short message, a sentence or two. Never more than a short paragraph unless asked. Telegram markdown only (`*bold*`, `_italic_`, `` `code` ``).

## Tools
You have tools available. Use them proactively:
- **Wiki reads:** always call `read_wiki_page` before updating a page — never assume content
- **Wiki writes:** use `propose_wiki_create` or `propose_wiki_update` — a dedicated writer produces the article from your brief, and the user approves before saving. For updates, read the page first then describe what needs to change.
- **Web search:** call `web_search` when you need current info or facts you're unsure of
- After proposing a wiki edit, reply with one sentence describing what you drafted. Do NOT mention /approve or /reject.

## Memory
When {name} shares something worth keeping, append to your response:
`<memory category="personal|preference|goal|task|relationship|habit|work|health|interest|other" importance="1-10">fact</memory>`
When resolved: `<forget>keyword</forget>` — always forget before adding an update.

## Images
`<image url="https://...">caption</image>` — direct URLs only, use sparingly.

## Context
- {name}: developer, Singapore. Productivity tools, games, minimalist design
- Dashboard: finances, dating pipeline, todos (Telegram + Supabase + React)
- Current time: {format_user_time()}

---

{wiki_block}

{memory_block}

{summary_block}"""


# ─── Tool Definitions ───────────────────────────────────────

_TOOLS = [
    {
        "name": "search_wiki",
        "description": "Search wiki pages by keyword. Returns titles, slugs, and short snippets. Use read_wiki_page to get full content.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_wiki_page",
        "description": "Read the full content of a wiki page by slug. Always call this before proposing an update.",
        "input_schema": {
            "type": "object",
            "properties": {"slug": {"type": "string"}},
            "required": ["slug"],
        },
    },
    {
        "name": "propose_wiki_create",
        "description": (
            "Propose creating a new wiki page. Pass a brief describing what the page should cover — "
            "a dedicated writer will produce the actual content. Do NOT write the article yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "slug": {"type": "string", "description": "lowercase-hyphenated-slug"},
                "title": {"type": "string"},
                "brief": {"type": "string", "description": "What the page should cover — key topics, facts, context. 2-5 sentences."},
            },
            "required": ["slug", "title", "brief"],
        },
    },
    {
        "name": "propose_wiki_update",
        "description": (
            "Propose updating an existing wiki page. Always read the page first. "
            "Pass the existing content and a description of what to change — a dedicated writer will produce the updated article."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "title": {"type": "string"},
                "existing_content": {"type": "string", "description": "Current page content from read_wiki_page"},
                "changes": {"type": "string", "description": "What to add, update, or remove. Be specific."},
            },
            "required": ["slug", "existing_content", "changes"],
        },
    },
    {
        "name": "propose_wiki_delete",
        "description": "Propose deleting a wiki page for user approval.",
        "input_schema": {
            "type": "object",
            "properties": {"slug": {"type": "string"}},
            "required": ["slug"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for current information, news, or facts you're unsure about.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


_WIKI_WRITER_SYSTEM = (
    "You are a wiki writer. Write concise, factual wiki pages in Karpathy style. "
    "Dense and direct — no intro sentence, no 'this page covers...', no filler. "
    "Use ## headers only when there are 3+ distinct sections. Bullets for lists. Bold key terms. "
    "Default length: under 400 words. Write longer only if the brief explicitly requests a detailed, "
    "comprehensive, or verbatim article — in that case, do not truncate or summarize; include everything. "
    "Output markdown only — no commentary, no preamble."
)


_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=cfg.openai_api_key)
    return _openai_client


def _call_wiki_writer(prompt: str) -> str:
    """Call GPT-4o to write wiki content. Returns markdown string."""
    try:
        response = _get_openai_client().chat.completions.create(
            model=cfg.openai_wiki_model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": _WIKI_WRITER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        log.info(f"Wiki writer produced {len(content.split())} words")
        return content.strip()
    except Exception as e:
        log.error(f"Wiki writer call failed: {e}")
        raise


def _execute_tool(name: str, tool_input: dict, wiki_edits: list[dict]) -> str:
    import uuid
    try:
        if name == "search_wiki":
            if not dashboard_svc.is_configured():
                return "Wiki not configured."
            pages = dashboard_svc.search_wiki(tool_input["query"])
            if not pages:
                return "No results found."
            # Return title + slug + first 150 chars only — use read_wiki_page for full content
            lines = []
            for p in pages:
                snippet = (p.get("content") or "")[:150].replace("\n", " ")
                if len(p.get("content", "")) > 150:
                    snippet += "..."
                lines.append(f"- **{p['title']}** (`{p['slug']}`): {snippet}")
            return "\n".join(lines)

        elif name == "read_wiki_page":
            if not dashboard_svc.is_configured():
                return "Wiki not configured."
            page = dashboard_svc.get_wiki_page(tool_input["slug"])
            if page:
                return f"**{page['title']}**\n\n{page['content']}"
            return f"Page '{tool_input['slug']}' not found."

        elif name == "propose_wiki_create":
            if not cfg.openai_api_key:
                return "Wiki writer not configured (missing OPENAI_API_KEY)."
            prompt = f"Write a wiki page titled '{tool_input['title']}'.\n\nBrief: {tool_input['brief']}"
            content = _call_wiki_writer(prompt)
            edit_id = str(uuid.uuid4())[:8]
            edit = {
                "id": edit_id,
                "type": "create",
                "slug": tool_input["slug"],
                "title": tool_input["title"],
                "content": content,
                "description": "",
            }
            _register_pending_edit(edit)
            wiki_edits.append(edit)
            log.info(f"Wiki create proposed: {tool_input['slug']} ({len(content.split())} words)")
            return f"Queued for approval: '{tool_input['title']}'"

        elif name == "propose_wiki_update":
            if not cfg.openai_api_key:
                return "Wiki writer not configured (missing OPENAI_API_KEY)."
            prompt = (
                f"Update the following wiki page.\n\n"
                f"Current content:\n{tool_input['existing_content']}\n\n"
                f"Changes requested: {tool_input['changes']}"
            )
            content = _call_wiki_writer(prompt)
            edit_id = str(uuid.uuid4())[:8]
            edit = {
                "id": edit_id,
                "type": "update",
                "slug": tool_input["slug"],
                "title": tool_input.get("title"),
                "content": content,
                "description": "",
            }
            _register_pending_edit(edit)
            wiki_edits.append(edit)
            log.info(f"Wiki update proposed: {tool_input['slug']} ({len(content.split())} words)")
            return f"Update queued for approval: '{tool_input['slug']}'"

        elif name == "propose_wiki_delete":
            edit_id = str(uuid.uuid4())[:8]
            edit = {
                "id": edit_id,
                "type": "delete",
                "slug": tool_input["slug"],
                "title": None,
                "content": "",
                "description": "",
            }
            _register_pending_edit(edit)
            wiki_edits.append(edit)
            log.info(f"Wiki delete proposed: {tool_input['slug']}")
            return f"Delete queued for approval: '{tool_input['slug']}'"

        elif name == "web_search":
            if not cfg.serper_api_key:
                return "Web search not configured."
            results = web_search.search(tool_input["query"])
            return web_search.format_results_for_context(results)

        return f"Unknown tool: {name}"

    except Exception as e:
        log.error(f"Tool execution failed ({name}): {e}")
        return f"Error: {e}"


# ─── Response Generation ────────────────────────────────────


def generate_response(
    user_id: int, user_message: str, image_data: dict | None = None
) -> tuple[str, list[dict]]:
    """Generate Aria's response using an agentic tool loop.

    Returns (reply_text, wiki_edits_pending_approval)."""
    try:
        memories = get_active_memories(user_id)
        summaries = get_recent_summaries(user_id, 3)
        history = get_recent_conversation(user_id)
        system = _build_system_prompt(memories, summaries)

        if image_data:
            user_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_data["media_type"],
                        "data": image_data["base64"],
                    },
                },
                {"type": "text", "text": user_message or "What do you see in this image?"},
            ]
        else:
            user_content = user_message

        messages = history + [{"role": "user", "content": user_content}]

        log.info(
            f"Calling Claude: {len(history)} history msgs, {len(memories)} memories, "
            f"has_image={image_data is not None}"
        )

        # ── Agentic tool loop ─────────────────────────────────
        wiki_edits: list[dict] = []
        final_text = ""

        # cache_control on the system block caches tools+system, so loop
        # iterations 2+ (and rapid follow-up turns) read the prefix from cache
        system_blocks = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

        for iteration in range(6):
            response = get_client().messages.create(
                model=cfg.claude_model,
                max_tokens=cfg.max_tokens,
                system=system_blocks,
                tools=_TOOLS,
                messages=messages,
            )

            text_parts = [b.text for b in response.content if b.type == "text"]
            if text_parts:
                final_text = "\n".join(text_parts)

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason == "tool_use":
                tool_uses = [b for b in response.content if b.type == "tool_use"]
                tool_results = []
                for tu in tool_uses:
                    log.info(f"Tool: {tu.name}({list(tu.input.keys())})")
                    result = _execute_tool(tu.name, tu.input, wiki_edits)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result,
                    })
                messages = messages + [
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": tool_results},
                ]
            elif response.stop_reason == "max_tokens":
                log.warning("Claude hit max_tokens mid-response — response truncated")
                if not final_text.strip():
                    final_text = (
                        "That response was too long and got cut off — try asking for it "
                        "in smaller chunks (e.g. one section at a time)."
                    )
                break
            else:
                break

        # ── Parse tags from final text ────────────────────────
        clean_text, extracted = _parse_memories(final_text)
        for mem in extracted:
            save_memory(user_id, mem["category"], mem["content"], mem["importance"])
        if extracted:
            log.info(f"Extracted {len(extracted)} memories: {[m['category'] for m in extracted]}")

        clean_text, forget_terms = parse_forgets(clean_text)
        if forget_terms:
            process_forgets(user_id, forget_terms)

        # Fallback: catch any wiki tags Aria wrote directly
        clean_text, tag_edits = parse_wiki_edits(clean_text)
        if tag_edits and not wiki_edits:
            wiki_edits = tag_edits

        # Enforce one-sentence reply when wiki edits are pending
        if wiki_edits:
            for edit in wiki_edits:
                edit["description"] = clean_text.strip()
            clean_text = re.split(r'(?<=[.!?])\s', clean_text.strip(), maxsplit=1)[0]

        if wiki_edits:
            log.info(f"Wiki edits pending approval: {[e['id'] for e in wiki_edits]}")

        return clean_text, wiki_edits

    except anthropic.RateLimitError:
        log.warning("Claude rate limited")
        return "Give me a sec, I'm a bit overwhelmed right now. Try again in a moment? 😅", []
    except anthropic.APIStatusError as e:
        log.error(f"Claude API error: {e.status_code} — {e.message}")
        return "Something glitched on my end. Try again?", []
    except Exception as e:
        log.error(f"Unexpected error in generate_response: {e}", exc_info=True)
        return "Something glitched on my end. Try again?", []


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

_FORGET_RE = re.compile(
    r'<forget>(.*?)</forget>',
    re.DOTALL,
)

_WIKI_CREATE_RE = re.compile(
    r'<wiki_create\s+slug="([^"]+)"\s+title="([^"]+)">(.*?)</wiki_create>',
    re.DOTALL,
)

_WIKI_UPDATE_RE = re.compile(
    r'<wiki_update\s+slug="([^"]+)">(.*?)</wiki_update>',
    re.DOTALL,
)

_WIKI_DELETE_RE = re.compile(
    r'<wiki_delete\s+slug="([^"]+)"\s*/?>',
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


# ─── Wiki Edit Tags ──────────────────────────────────────────

# Pending wiki edits awaiting user approval: {edit_id: {type, slug, title, content}}
_pending_wiki_edits: dict[str, dict] = {}
_MAX_PENDING_EDITS = 20


def _register_pending_edit(edit: dict) -> None:
    """Track an edit awaiting approval, pruning the oldest beyond the cap."""
    _pending_wiki_edits[edit["id"]] = edit
    while len(_pending_wiki_edits) > _MAX_PENDING_EDITS:
        oldest = next(iter(_pending_wiki_edits))
        _pending_wiki_edits.pop(oldest)


def parse_wiki_edits(text: str) -> tuple[str, list[dict]]:
    """Extract <wiki_create> and <wiki_update> tags.
    Returns (clean_text, list_of_edits)."""
    import uuid
    edits = []

    for m in _WIKI_CREATE_RE.finditer(text):
        edit_id = str(uuid.uuid4())[:8]
        edit = {
            "id": edit_id,
            "type": "create",
            "slug": m.group(1).strip(),
            "title": m.group(2).strip(),
            "content": m.group(3).strip(),
        }
        edits.append(edit)
        _register_pending_edit(edit)

    for m in _WIKI_UPDATE_RE.finditer(text):
        edit_id = str(uuid.uuid4())[:8]
        edit = {
            "id": edit_id,
            "type": "update",
            "slug": m.group(1).strip(),
            "title": None,
            "content": m.group(2).strip(),
        }
        edits.append(edit)
        _register_pending_edit(edit)

    for m in _WIKI_DELETE_RE.finditer(text):
        edit_id = str(uuid.uuid4())[:8]
        edit = {
            "id": edit_id,
            "type": "delete",
            "slug": m.group(1).strip(),
            "title": None,
            "content": "",
        }
        edits.append(edit)
        _register_pending_edit(edit)

    clean = _WIKI_CREATE_RE.sub("", text)
    clean = _WIKI_UPDATE_RE.sub("", clean)
    clean = _WIKI_DELETE_RE.sub("", clean).strip()
    return clean, edits


def remove_pending_wiki_edit(edit_id: str) -> None:
    """Remove a pending wiki edit after approval/rejection."""
    _pending_wiki_edits.pop(edit_id, None)

