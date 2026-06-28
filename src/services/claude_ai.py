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


def _build_system_prompt(
    memories: list[dict],
    summaries: list[dict],
    user_preferences: dict | None = None,
    search_results: str | None = None,
    wiki_context: str | None = None,
) -> str:
    name = cfg.user_name

    memory_block = ""
    if memories:
        mem_lines = []
        for m in memories:
            ts = utc_to_user(m["created_at"]) if m.get("created_at") else ""
            mem_lines.append(f"- [{m['category']}] ({ts}) {m['content']}")
        memory_block = "\n".join(mem_lines)
    else:
        memory_block = "No stored memories yet — getting to know him."

    summary_block = ""
    if summaries:
        lines = []
        for s in summaries:
            end = s.get("period_end", "")[:10]
            lines.append(f"- ({end}): {s['summary']}")
        summary_block = f"\n## Recent Conversation Summaries\n" + "\n".join(lines)

    prefs_block = ""
    if user_preferences:
        lines = [f"- {k}: {v}" for k, v in user_preferences.items()]
        prefs_block = f"\n## {name}'s Preferences\n" + "\n".join(lines)

    search_block = ""
    if search_results:
        search_block = f"\n## Web Search Results\n{search_results}"

    # Wiki: titles always loaded, content from auto-search passed in
    wiki_block = ""
    if dashboard_svc.is_configured():
        wiki_titles = dashboard_svc.get_wiki_titles()
        if wiki_titles:
            wiki_block = (
                f"\n## {name}'s Wiki Pages\n"
                + dashboard_svc.format_wiki_titles_for_context(wiki_titles)
            )
        if wiki_context:
            wiki_block += f"\n\n## Wiki Content (auto-loaded from matching pages)\n{wiki_context}"

    web_search_available = bool(cfg.serper_api_key)

    return f"""You are Aria, {name}'s personal assistant and wiki specialist on Telegram.

Your primary job is maintaining {name}'s personal knowledge wiki — writing and editing concise, factual entries. Think Karpathy-style: dense, no fluff, a few lines per page at most. Never write an essay.

**Personality:** sharp, warm, slightly flirty, concise. Push back when something seems off. Use his name sometimes. Emojis sparingly.

**Replies:** one short message, a sentence or two. Never more than a short paragraph unless asked. Telegram markdown only (`*bold*`, `_italic_`, `` `code` ``).

---

## Wiki

{name}'s wiki page titles and any relevant content are loaded below.
To load a specific page: `<wiki_search>keyword</wiki_search>`
When reading: summarise naturally, don't dump raw text.

**When writing or editing**, include the draft inline after a one-line acknowledgement:

Create: `<wiki_create slug="slug" title="Title">content</wiki_create>`
Update (full page): `<wiki_update slug="slug">content</wiki_update>`
Delete: `<wiki_delete slug="slug" />`

Wiki content rules: factual, a short paragraph at most, only markdown that earns it. Always include the tag — never acknowledge without the draft.
Do NOT mention /approve or /reject.

---

## Web Search
{"Place `<search>query</search>` BEFORE your response. Also `<image_search>query</image_search>` for images." if web_search_available else "Web search not configured."}

## Images
`<image url="https://...">caption</image>` — direct URLs only, use sparingly.

---

## Memory
Append to response when {name} shares something worth keeping:
`<memory category="personal|preference|goal|task|relationship|habit|work|health|interest|other" importance="1-10">fact</memory>`
When resolved: `<forget>keyword</forget>` — always forget before adding an update.

---

## Context
- {name}: developer, Singapore. Productivity tools, games, minimalist design
- Dashboard: finances, dating pipeline, todos (Telegram + Supabase + React)

---

{wiki_block}
{memory_block}
{summary_block}
{search_block}"""


# ─── Response Generation ────────────────────────────────────


def generate_response(user_id: int, user_message: str, image_data: dict | None = None) -> str:
    """Generate Aria's response with search, reminder, and vision support.

    Args:
        user_id: Telegram user ID
        user_message: Text content of the message
        image_data: Optional dict with {"base64": str, "media_type": str} for image vision

    Flow:
    1. First pass: ask Claude to respond (may include <search> tags)
    2. If search tags found: execute searches, re-call Claude with results
    3. Parse out <memory>, <reminder>, <image>, <search> tags from final response
    """
    try:
        memories = get_active_memories(user_id)
        summaries = get_recent_summaries(user_id, 3)
        history = get_recent_conversation(user_id)

        # Auto-search wiki based on user message keywords
        wiki_context = None
        if dashboard_svc.is_configured() and user_message:
            auto_wiki_pages = dashboard_svc.auto_search_wiki(user_message, max_results=3)
            if auto_wiki_pages:
                wiki_context = dashboard_svc.format_wiki_results_for_context(auto_wiki_pages)
                log.info(f"Wiki auto-search found {len(auto_wiki_pages)} pages: {[p['title'] for p in auto_wiki_pages]}")

        system = _build_system_prompt(memories, summaries, wiki_context=wiki_context)

        # Detect wiki edit intent (dedicated call will happen after main response)
        wiki_intent = _detect_wiki_intent(user_message, has_image=image_data is not None)

        # Build the user message content (text + optional image)
        if image_data:
            user_content = []
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_data["media_type"],
                    "data": image_data["base64"],
                },
            })
            if user_message:
                user_content.append({"type": "text", "text": user_message})
            else:
                user_content.append({"type": "text", "text": "What do you see in this image?"})
            messages = history + [{"role": "user", "content": user_content}]
        else:
            messages = history + [{"role": "user", "content": user_message}]

        log.info(
            f"Calling Claude: {len(history)} history msgs, {len(memories)} memories, "
            f"time={format_user_time()}, has_image={image_data is not None}"
        )

        response = get_client().messages.create(
            model=cfg.claude_model,
            max_tokens=cfg.max_tokens,
            system=system,
            messages=messages,
        )

        full_text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        # ── Check for search requests ────────────────────────
        search_queries = _SEARCH_RE.findall(full_text)
        image_search_queries = _IMAGE_SEARCH_RE.findall(full_text)
        wiki_search_queries = _WIKI_SEARCH_RE.findall(full_text)

        if search_queries or image_search_queries or wiki_search_queries:
            # Execute searches
            search_results_text = ""
            if search_queries:
                for query in search_queries:
                    results = web_search.search(query.strip())
                    search_results_text += f"\n### Results for: {query.strip()}\n"
                    search_results_text += web_search.format_results_for_context(results)

            image_results = []
            if image_search_queries:
                for query in image_search_queries:
                    imgs = web_search.search_images(query.strip(), num_results=3)
                    if imgs:
                        image_results.extend(imgs)
                        search_results_text += f"\n### Image results for: {query.strip()}\n"
                        for img in imgs:
                            search_results_text += f"- {img['title']}: {img['image_url']}\n"

            # Execute wiki searches
            wiki_results_text = ""
            if wiki_search_queries:
                for query in wiki_search_queries:
                    pages = dashboard_svc.search_wiki(query.strip())
                    wiki_results_text += f"\n### Wiki results for: {query.strip()}\n"
                    wiki_results_text += dashboard_svc.format_wiki_results_for_context(pages)
                if wiki_results_text:
                    search_results_text += f"\n## Wiki Content\n{wiki_results_text}"

            # Second pass with search results
            system_with_search = _build_system_prompt(
                memories, summaries, search_results=search_results_text, wiki_context=wiki_context
            )

            response2 = get_client().messages.create(
                model=cfg.claude_model,
                max_tokens=cfg.max_tokens,
                system=system_with_search,
                messages=messages,
            )

            full_text = "".join(
                block.text for block in response2.content if block.type == "text"
            )

        # ── Strip search tags from final output ──────────────
        full_text = _SEARCH_RE.sub("", full_text)
        full_text = _IMAGE_SEARCH_RE.sub("", full_text)
        full_text = _WIKI_SEARCH_RE.sub("", full_text)

        # ── Parse memories ───────────────────────────────────
        clean_text, extracted = _parse_memories(full_text)

        for mem in extracted:
            save_memory(user_id, mem["category"], mem["content"], mem["importance"])

        if extracted:
            log.info(
                f"Extracted {len(extracted)} memories: "
                + ", ".join(m["category"] for m in extracted)
            )

        # ── Parse forgets ───────────────────────────────────
        clean_text, forget_terms = parse_forgets(clean_text)
        if forget_terms:
            process_forgets(user_id, forget_terms)

        # ── Wiki edits — parse tags from Aria's response ─────
        clean_text, wiki_edits = parse_wiki_edits(clean_text)
        if wiki_edits:
            log.info(f"Wiki edits pending approval: {[e['id'] for e in wiki_edits]}")

        # Stash wiki edits for the handler
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
    if any(c in msg for c in ["create", "new", "start", "draft", "make"]):
        return "create" if ("wiki" in msg or has_image) else None

    # Update
    if any(a in msg for a in ["update", "edit", "add", "append", "modify", "change",
                                "revise", "rewrite", "include", "put", "insert",
                                "note", "record", "save", "write", "log", "track"]):
        return "update" if ("wiki" in msg or has_image) else None

    return None


def _generate_wiki_content(user_message: str, intent: str, wiki_context: str | None, aria_response: str | None = None) -> list[dict]:
    """Make a dedicated API call to generate wiki content. Returns list of edits."""
    import json as _json
    import uuid

    try:
        context_block = ""
        if wiki_context:
            context_block = f"\n\nExisting wiki content that may be relevant:\n{wiki_context}"
        if aria_response:
            context_block += f"\n\nYour assistant already drafted this response to the user (use this as the basis for the wiki content if it contains the relevant material):\n{aria_response}"

        if intent == "delete":
            system = (
                "You are a wiki manager. The user wants to delete a page. "
                "Based on the user's message and the available wiki pages, identify which page to delete. "
                "Respond with ONLY a JSON object, no other text:\n"
                '{"action": "delete", "slug": "the-page-slug", "title": "The Page Title"}'
            )
        elif intent == "create":
            system = (
                "You are a wiki content writer. The user wants to create a new wiki page. "
                "Write the full page content based on their request. "
                "Respond with ONLY a JSON object, no other text, no markdown fences:\n"
                '{"action": "create", "slug": "lowercase-hyphenated-slug", "title": "Page Title", "content": "Full page content here..."}'
                "\n\nWrite substantial, well-structured content in markdown. The content field should be the complete page."
            )
        else:  # update
            system = (
                "You are a wiki editor. The user wants to update an existing wiki page. "
                "Based on the existing page content and the user's request, produce the complete updated page. "
                "Respond with ONLY a JSON object, no other text, no markdown fences:\n"
                '{"action": "update", "slug": "existing-page-slug", "title": "Page Title", "content": "Complete updated page content..."}'
                "\n\nThe content field must contain the ENTIRE page (not just changes). Merge the existing content with the requested changes."
            )

        user_content = f"User's request: {user_message}{context_block}"

        response = get_client().messages.create(
            model=cfg.claude_model,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )

        result_text = response.content[0].text if response.content else ""

        # Strip markdown fences if present
        result_text = result_text.strip()
        if result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[-1]
        if result_text.endswith("```"):
            result_text = result_text.rsplit("```", 1)[0]
        result_text = result_text.strip()

        data = _json.loads(result_text)

        edit_id = str(uuid.uuid4())[:8]
        edit = {
            "id": edit_id,
            "type": data.get("action", intent),
            "slug": data.get("slug", ""),
            "title": data.get("title"),
            "content": data.get("content", ""),
        }

        if not edit["content"] and intent != "delete":
            log.warning(f"Wiki content call returned empty content for {edit['slug']!r}; raw: {result_text[:200]}")

        # Store in pending edits
        _pending_wiki_edits[edit_id] = edit
        log.info(f"Wiki content generated: {edit['type']} {edit['slug']}")
        return [edit]

    except _json.JSONDecodeError as e:
        log.error(f"Wiki content call returned invalid JSON: {e}; raw: {result_text[:200] if 'result_text' in dir() else '(not set)'}")
        # Try to extract from XML tags as fallback
        _, tag_edits = parse_wiki_edits(result_text)
        return tag_edits
    except Exception as e:
        log.error(f"Wiki content generation failed: {e}")
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

