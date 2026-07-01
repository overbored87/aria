"""Claude API integration — Aria's brain.

Handles system prompt construction, context windowing, response generation,
memory extraction, proactive message generation, and conversation summarization.
"""

from __future__ import annotations
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import anthropic

from src.config import cfg
from src.utils.logger import log
from src.utils.time_helpers import format_user_time, time_of_day, now_user, utc_to_user
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
- **Wiki writes:** use `propose_wiki_create` or `propose_wiki_update` — the user approves before saving
- **Web search:** call `web_search` when you need current info or facts you're unsure of
- After proposing a wiki edit, reply with one sentence describing what you drafted. Do NOT mention /approve or /reject.

## Wiki
A dedicated writer handles article content — you just decide what to write and pass a brief. For updates, read the page first then describe what needs to change. After proposing, reply with one sentence summarising what you drafted.

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


def _call_wiki_writer(prompt: str) -> str:
    """Call GPT-4o to write wiki content. Returns markdown string."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=cfg.openai_api_key)
        response = client.chat.completions.create(
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
            _pending_wiki_edits[edit_id] = edit
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
            _pending_wiki_edits[edit_id] = edit
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
            _pending_wiki_edits[edit_id] = edit
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


def generate_response(user_id: int, user_message: str, image_data: dict | None = None) -> str:
    """Generate Aria's response using an agentic tool loop."""
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

        for iteration in range(6):
            response = get_client().messages.create(
                model=cfg.claude_model,
                max_tokens=cfg.max_tokens,
                system=system,
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

        _last_wiki_edits.clear()
        _last_wiki_edits.extend(wiki_edits)

        return clean_text

    except anthropic.RateLimitError:
        log.warning("Claude rate limited")
        return "Give me a sec, I'm a bit overwhelmed right now. Try again in a moment? 😅"
    except anthropic.APIStatusError as e:
        log.error(f"Claude API error: {e.status_code} — {e.message}")
        return "Something glitched on my end. Try again?"
    except Exception as e:
        log.error(f"Unexpected error in generate_response: {e}", exc_info=True)
        return "Something glitched on my end. Try again?"


# Temp storage for wiki edits between generate_response and handler
_last_wiki_edits: list[dict] = []


def get_pending_wiki_edits_from_response() -> list[dict]:
    """Pop wiki edits extracted from the last response."""
    edits = list(_last_wiki_edits)
    _last_wiki_edits.clear()
    return edits


# ─── Proactive Message Generation ───────────────────────────

_TYPE_PROMPTS = {
    "morning_checkin": (
        "Send a warm, energizing morning message to {name}. "
        "Reference something relevant from his recent conversations or goals if possible. "
        "Keep it brief and motivating. Maybe a playful nudge about the day ahead."
    ),
    "evening_checkin": (
        "Send a chill evening check-in to {name}. "
        "Ask how his day went or reference something he was working on. "
        "Keep it warm and relaxed — wind-down energy."
    ),
    "task_followup": (
        "Follow up with {name} about a task or goal he mentioned: \"{task}\". "
        "Be natural — don't be naggy."
    ),
    "goal_reminder": (
        "Give {name} a gentle nudge about a goal: \"{goal}\". "
        "Be encouraging, not preachy."
    ),
    "affirmation": (
        "Send {name} a brief, genuine affirmation or encouragement. "
        "Make it personal based on what you know about him, not generic positivity fluff."
    ),
    "dashboard_insights": (
        "Analyze {name}'s personal dashboard data below and send him a concise daily briefing. "
        "Cover the most interesting or actionable insights across whatever categories are present. "
        "Be specific with numbers and trends — don't be vague. Keep it punchy and Telegram-friendly. "
        "If you spot something noteworthy (a spending spike, a goal milestone, a pattern), call it out. "
        "Use your personality — make data feel personal, not like a spreadsheet.\n\n"
        "Dashboard data:\n{dashboard_data}"
    ),
    "custom": "{prompt}",
}


def generate_proactive_message(
    user_id: int, msg_type: str, context: dict | None = None
) -> str | None:
    """Generate a proactive/scheduled message."""
    try:
        context = context or {}
        memories = get_active_memories(user_id, 15)
        summaries = get_recent_summaries(user_id, 2)
        name = cfg.user_name
        current_time = format_user_time()

        memory_block = (
            "\n".join(f"- [{m['category']}] {m['content']}" for m in memories)
            if memories
            else "Getting to know him still."
        )
        summary_block = "\n".join(s["summary"] for s in summaries) if summaries else ""

        template = _TYPE_PROMPTS.get(msg_type, _TYPE_PROMPTS["custom"])
        prompt = template.format(name=name, **context)

        system = (
            f"You are Aria, {name}'s personal assistant. "
            f"Current time: {current_time}.\n\n"
            f"Your personality: warm, slightly flirty, competent, concise. "
            f"You're messaging on Telegram so keep it short.\n\n"
            f"What you know about {name}:\n{memory_block}"
            + (f"\n\nRecent context:\n{summary_block}" if summary_block else "")
        )

        response = get_client().messages.create(
            model=cfg.claude_model,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        # Strip any memory tags from proactive messages
        return re.sub(r"<memory[^>]*>.*?</memory>", "", text, flags=re.DOTALL).strip()

    except Exception as e:
        log.error(f"Failed to generate proactive message: {e}", exc_info=True)
        return None


def generate_dashboard_insights(user_id: int) -> str | None:
    """Pull dashboard data and generate a daily insights message."""
    if not dashboard_svc.is_configured():
        log.debug("Dashboard not configured — skipping insights")
        return None

    try:
        grouped = dashboard_svc.get_latest_by_category(limit_per_category=10)
        if not grouped:
            log.info("No dashboard data found for insights")
            return None

        dashboard_data = dashboard_svc.format_for_context(grouped)
        memories = get_active_memories(user_id, 15)
        name = cfg.user_name
        current_time = format_user_time()

        memory_block = (
            "\n".join(f"- [{m['category']}] {m['content']}" for m in memories)
            if memories else ""
        )

        template = _TYPE_PROMPTS["dashboard_insights"]
        prompt = template.format(name=name, dashboard_data=dashboard_data)

        system = (
            f"You are Aria, {name}'s personal assistant. "
            f"Current time: {current_time}.\n\n"
            f"Your personality: warm, slightly flirty, competent, concise. "
            f"You're messaging on Telegram so keep it punchy. "
            f"Separate distinct thoughts with blank lines (each becomes a separate message).\n\n"
            f"What you know about {name}:\n{memory_block}"
        )

        response = get_client().messages.create(
            model=cfg.claude_model,
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        clean = re.sub(r"<memory[^>]*>.*?</memory>", "", text, flags=re.DOTALL).strip()
        log.info(f"Dashboard insights generated: {len(clean)} chars, {len(grouped)} categories")
        return clean

    except Exception as e:
        log.error(f"Failed to generate dashboard insights: {e}", exc_info=True)
        return None


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


def _detect_wiki_intent(message: str, has_image: bool = False) -> str | None:
    """Detect if a message is asking to create/edit/delete a wiki page.
    Requires 'wiki' in text messages; for images, action words alone are enough."""
    msg = message.lower().strip()

    # Delete
    if any(d in msg for d in ["delete", "remove", "destroy", "get rid of"]):
        return "delete" if ("wiki" in msg or has_image) else None

    # Create
    if any(c in msg for c in ["create", "new", "start", "draft", "make", "write"]):
        return "create" if ("wiki" in msg or has_image) else None

    # Update
    if any(a in msg for a in ["update", "edit", "add", "append", "modify", "change",
                                "revise", "rewrite", "include", "put", "insert",
                                "note", "record", "save", "log", "track"]):
        return "update" if ("wiki" in msg or has_image) else None

    return None


def _wiki_writer_call(
    user_message: str,
    intent: str,
    wiki_context: str | None,
    aria_reply: str | None = None,
    image_data: dict | None = None,
) -> list[dict]:
    """Dedicated wiki writer agent. Returns parsed wiki edits."""
    try:
        context_parts = []
        if wiki_context:
            context_parts.append(f"Existing wiki content:\n{wiki_context}")
        if aria_reply:
            context_parts.append(f"Aria's summary of what to capture: {aria_reply}")

        context_block = ("\n\n" + "\n\n".join(context_parts)) if context_parts else ""

        CREATE_STYLE = (
            "Karpathy style. Max 400 words. "
            "Use ## headers only if there are 3+ distinct sections. "
            "Bullet points for lists, bold for key terms. "
            "Every sentence must earn its place — no intro, no summary, no 'this page covers...'. "
            "Start directly with the content."
        )
        EDIT_STYLE = (
            "You are merging new information into an existing wiki page. "
            "Preserve all existing content that is still accurate. "
            "Add or update only what has changed — do not rewrite for style. "
            "Keep the same structure unless restructuring genuinely improves it. "
            "Max 400 words total. No filler, no commentary about what changed."
        )
        if intent == "delete":
            system = (
                "Identify which wiki page to delete based on the user's request.\n"
                "Output ONLY this tag, nothing else:\n"
                '<wiki_delete slug="the-page-slug" />'
            )
        elif intent == "create":
            system = (
                f"Write a concise wiki page. {CREATE_STYLE}\n"
                "Output ONLY this tag, nothing else:\n"
                '<wiki_create slug="lowercase-slug" title="Page Title">content</wiki_create>'
            )
        else:
            system = (
                f"{EDIT_STYLE}\n"
                "Output the full updated page. Output ONLY this tag, nothing else:\n"
                '<wiki_update slug="existing-slug">full updated content</wiki_update>'
            )

        user_parts: list = []
        if image_data:
            user_parts.append({
                "type": "image",
                "source": {"type": "base64", "media_type": image_data["media_type"], "data": image_data["base64"]},
            })
        user_parts.append({"type": "text", "text": f"Request: {user_message}{context_block}"})

        response = get_client().messages.create(
            model=cfg.claude_model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user_parts}],
        )

        result_text = response.content[0].text.strip() if response.content else ""
        log.info(f"Wiki writer output: {result_text[:120]}")

        _, edits = parse_wiki_edits(result_text)
        if not edits:
            log.warning(f"Wiki writer returned no parseable tags; raw: {result_text[:200]}")
        # Attach Aria's conversational summary as the description
        for edit in edits:
            edit["description"] = aria_reply or ""
        return edits

    except Exception as e:
        log.error(f"Wiki writer call failed: {e}")
        return []

_IMAGE_RE = re.compile(
    r'<image\s+url="([^"]+)">(.*?)</image>',
    re.DOTALL,
)

_SEARCH_RE = re.compile(
    r'<search>(.*?)</search>',
    re.DOTALL,
)

_IMAGE_SEARCH_RE = re.compile(
    r'<image_search>(.*?)</image_search>',
    re.DOTALL,
)

_WIKI_SEARCH_RE = re.compile(
    r'<wiki_search>(.*?)</wiki_search>',
    re.DOTALL,
)

_REMINDER_RE = re.compile(
    r'<reminder\s+time="(\d{1,2}:\d{2})"(?:\s+date="(\d{4}-\d{2}-\d{2})")?>(.*?)</reminder>',
    re.DOTALL,
)

_FORGET_RE = re.compile(
    r'<forget>(.*?)</forget>',
    re.DOTALL,
)

_DONE_RE = re.compile(
    r'<done>(.*?)</done>',
    re.DOTALL,
)

_CANCEL_REMINDER_RE = re.compile(
    r'<cancel_reminder>(.*?)</cancel_reminder>',
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


def parse_reminders(text: str) -> tuple[str, list[dict]]:
    """Extract <reminder> tags and return (clean_text, list_of_reminders).

    Each reminder: {"time": "HH:MM", "date": "YYYY-MM-DD" or None, "message": str, "dt": datetime}
    """
    tz = ZoneInfo(cfg.user_timezone)
    now = now_user()
    reminders = []

    for m in _REMINDER_RE.finditer(text):
        time_str = m.group(1).strip()
        date_str = m.group(2).strip() if m.group(2) else None
        message = m.group(3).strip()

        try:
            hour, minute = map(int, time_str.split(":"))

            if date_str:
                year, month, day = map(int, date_str.split("-"))
                dt = datetime(year, month, day, hour, minute, tzinfo=tz)

                # Guard: if Claude hallucinated a past date, fix it
                if dt < now:
                    # Keep the time, but use today (or tomorrow if time passed)
                    dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    if dt <= now:
                        dt += timedelta(days=1)
                    log.warning(f"Reminder date was in the past — corrected to {dt.isoformat()}")
            else:
                # No date given: today, or tomorrow if time already passed
                dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if dt <= now:
                    dt += timedelta(days=1)

            reminders.append({
                "time": time_str,
                "date": dt.strftime("%Y-%m-%d"),
                "message": message,
                "dt": dt,
            })
            log.info(f"Reminder parsed: '{message[:50]}' at {dt.isoformat()}")

        except Exception as e:
            log.error(f"Failed to parse reminder: {e}")

    clean = _REMINDER_RE.sub("", text).strip()
    return clean, reminders


def _attach_reminder_confirmations(text: str, reminders: list[dict]) -> str:
    """If Aria didn't already confirm the reminder, we don't add anything —
    the system prompt tells her to confirm naturally in her response."""
    return text


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


def parse_done_tags(text: str) -> tuple[str, list[str]]:
    """Extract <done> tags — keywords to match reminders to mark as completed."""
    terms = []
    for m in _DONE_RE.finditer(text):
        term = m.group(1).strip()
        if term:
            terms.append(term)
    clean = _DONE_RE.sub("", text).strip()
    return clean, terms


def parse_cancel_tags(text: str) -> tuple[str, list[str]]:
    """Extract <cancel_reminder> tags — keywords to match reminders to cancel."""
    terms = []
    for m in _CANCEL_REMINDER_RE.finditer(text):
        term = m.group(1).strip()
        if term:
            terms.append(term)
    clean = _CANCEL_REMINDER_RE.sub("", text).strip()
    return clean, terms


def process_done_reminders(user_id: int, terms: list[str]) -> int:
    """Mark reminders as done by keyword match."""
    total = 0
    for term in terms:
        count = complete_reminders_matching(user_id, term)
        total += count
        log.info(f"Completed {count} reminders matching '{term}'")
    return total


def process_cancel_reminders(user_id: int, terms: list[str]) -> int:
    """Cancel reminders by keyword match."""
    total = 0
    for term in terms:
        count = cancel_reminders_matching(user_id, term)
        total += count
        log.info(f"Cancelled {count} reminders matching '{term}'")
    return total


# ─── Wiki Edit Tags ──────────────────────────────────────────

# Pending wiki edits awaiting user approval: {edit_id: {type, slug, title, content}}
_pending_wiki_edits: dict[str, dict] = {}


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
        _pending_wiki_edits[edit_id] = edit

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
        _pending_wiki_edits[edit_id] = edit

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
        _pending_wiki_edits[edit_id] = edit

    clean = _WIKI_CREATE_RE.sub("", text)
    clean = _WIKI_UPDATE_RE.sub("", clean)
    clean = _WIKI_DELETE_RE.sub("", clean).strip()
    return clean, edits


def get_pending_wiki_edit(edit_id: str) -> dict | None:
    """Retrieve a pending wiki edit by ID."""
    return _pending_wiki_edits.get(edit_id)


def remove_pending_wiki_edit(edit_id: str) -> None:
    """Remove a pending wiki edit after approval/rejection."""
    _pending_wiki_edits.pop(edit_id, None)

