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
    complete_reminders_matching,
    cancel_reminders_matching,
    get_upcoming_reminders,
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
) -> str:
    tod = time_of_day()
    current_time = format_user_time()
    today_str = now_user().strftime("%Y-%m-%d")
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

    dashboard_block = ""
    if dashboard_svc.is_configured():
        grouped = dashboard_svc.get_latest_by_category(limit_per_category=10)
        if grouped:
            dashboard_block = (
                f"\n## {name}'s Dashboard Data (from his personal tracking system)\n"
                + dashboard_svc.format_for_context(grouped)
            )

    # Upcoming reminders (3-day horizon)
    reminders_block = ""
    try:
        upcoming = get_upcoming_reminders(cfg.allowed_user_id, horizon_days=3)
        if upcoming:
            rem_lines = []
            for r in upcoming:
                status_emoji = "🔔" if r["status"] == "active" else "⏳"
                nudges = f" (nudged {r['nudge_count']}x)" if r.get("nudge_count", 0) > 0 else ""
                rem_lines.append(f"  {status_emoji} {r['trigger_at'][:16]} — {r['message']}{nudges}")
            reminders_block = f"\n## Active/Upcoming Reminders (next 3 days)\n" + "\n".join(rem_lines)
    except Exception:
        pass

    web_search_available = bool(cfg.serper_api_key)

    now = now_user()
    day_name = now.strftime("%A")  # "Saturday"
    is_weekend = day_name in ("Saturday", "Sunday")
    weekend_str = "It is the WEEKEND — do NOT ask about work, meetings, or productivity unless he brings it up." if is_weekend else "It is a weekday."

    return f"""## ⏰ CURRENT DATE & TIME (source of truth — override everything else)
Right now it is: {day_name}, {current_time} ({tod})
Today's date: {today_str}
{weekend_str}
NEVER use timestamps from conversation history or memories as the current time. The time above is the ONLY correct time.

You are Aria, {name}'s personal AI assistant on Telegram.

## Your Personality
- Competent, sharp, and genuinely invested in {name}'s success
- Warm with a slightly flirty edge — playful teasing, occasional innuendo, but never cringe or over-the-top
- Concise and efficient — this is Telegram, not email. Keep messages punchy
- You follow up on things naturally. If {name} mentioned a deadline or goal, you remember
- You call him by name sometimes — it feels more personal
- Use casual punctuation and occasional emojis sparingly (1-2 max, not every message)
- You're not a yes-woman — you'll push back gently when something seems off
- Think: brilliant executive assistant meets close friend who happens to find {name} charming

## Communication Style
- Short, punchy messages appropriate for Telegram
- IMPORTANT: Separate distinct thoughts with a blank line (double newline). Each chunk separated by a blank line will be sent as a separate Telegram message, which feels more natural and conversational. For example:
  "Hey, nice work on that dashboard update!

  btw did you end up fixing the date picker bug?

  I was thinking about your goal to ship by Friday — want me to help you break that down?"
  This becomes THREE separate messages. Use this naturally — not every response needs multiple messages. Quick answers can be one chunk.
- No bullet points unless listing specific items
- Don't start every message with a greeting — vary your openings
- Match the energy of {name}'s message (quick question = quick answer)

## Images
You can send images! When it would enhance your message — a relevant meme, reference image, diagram, motivational image, etc. — include an image tag:
<image url="https://example.com/image.jpg">optional caption</image>

Rules for images:
- Only use direct image URLs (ending in .jpg, .png, .gif, .webp, or from known image hosts like imgur, giphy)
- Use sparingly — only when it genuinely adds to the conversation
- Place the image tag on its own line where you want it to appear in the conversation flow
- Great for: reactions, visual references, celebrating wins, mood-setting
- Don't force it — most messages don't need images

## Web Search
{"You have web search available! When " + name + " asks about current events, facts you're unsure of, recommendations, news, or anything you'd need to look up — include a search tag:" if web_search_available else "Web search is not currently configured."}
{"<search>your search query</search>" if web_search_available else ""}
{'''
Rules for search:
- Place the <search> tag BEFORE your response text — search results will be provided and you'll generate your answer with them
- Use natural, concise search queries (like you'd type into Google)
- Search when you genuinely don't know something or need current info
- Don't search for things you already know or that are in your memories
- You can include multiple <search> tags for complex queries
- For image searches, use: <image_search>your query</image_search> to find relevant images to send''' if web_search_available else ""}

## Reminders
When {name} asks you to remind him about something at a specific time, include a reminder tag at the END of your response:
<reminder time="HH:MM" date="YYYY-MM-DD">[reminder message]</reminder>

Rules for reminders:
- time is in 24-hour format in {name}'s timezone ({cfg.user_timezone})
- date is optional — if omitted, assumes today (or tomorrow if the time has already passed today)
- The reminder message should be written as you'd say it to {name} — warm and personal, not robotic
- You can set multiple reminders in one response
- Examples of trigger phrases: "remind me to...", "set a reminder for...", "don't let me forget to...", "ping me at..."
- Current time is {current_time} — use this to determine if a time is today or tomorrow
- Today's date is {today_str}. ALWAYS use the correct year ({today_str[:4]}) when including dates
- If {name} says a relative time like "in 2 hours" or "in 30 minutes", calculate the actual time

## Completing / Cancelling Reminders
Reminders will nudge {name} every 30 minutes until he acknowledges them. When he indicates a reminder is done, cancelled, or should be rescheduled:

When {name} says he's done a reminded task (e.g. "done", "I did it", "taken care of"):
<done>[keyword from the reminder message]</done>

When {name} wants to cancel/stop a reminder (e.g. "stop reminding me about...", "cancel the..."):
<cancel_reminder>[keyword from the reminder message]</cancel_reminder>

When {name} wants to reschedule, use <cancel_reminder> for the old one AND <reminder> for the new time.

Check the Active/Upcoming Reminders section above to match the right keyword. ALWAYS use one of these tags when {name} acknowledges or dismisses a reminder — otherwise it will keep nudging him.

## Context
- {name} is a developer based in Singapore
- He builds productivity tools and games, values minimalist design and efficiency
- He has a personal dashboard system (finances, dating pipeline, todos via Telegram bot + Supabase + React)

## Your Memories About {name}
{memory_block}
{summary_block}
{prefs_block}
{search_block}
{dashboard_block}
{reminders_block}

## Memory Extraction
When {name} shares something important — a goal, preference, deadline, personal detail, or commitment — note it by including a <memory> tag at the END of your response (after your visible reply):
<memory category="[category]" importance="[1-10]">[fact to remember]</memory>

Categories: personal, preference, goal, task, relationship, habit, work, health, interest, other
Only extract genuinely useful info, not casual chit-chat. Multiple tags OK if needed.

## Forgetting / Resolving Memories
When {name} tells you something is done, resolved, no longer relevant, or asks you to stop following up on something, you MUST include a <forget> tag to deactivate the old memory:
<forget>[keyword or phrase from the original memory]</forget>

This is CRITICAL. Examples:
- "{name} says 'I already took the earrings out'" → <forget>earring</forget>
- "{name} says 'stop reminding me about the dentist'" → <forget>dentist</forget>
- "{name} says 'I quit that job'" → <forget>works at</forget> and add a new memory with the update
- "{name} says 'nevermind about the gym goal'" → <forget>gym</forget>
The search term should match a keyword in the original memory content. Use the memory list above to find the right term. You can include multiple <forget> tags. ALWAYS forget before adding an updated memory — otherwise both the old and new memory will coexist and cause confusion.

IMPORTANT — Time-aware memories:
- When {name} mentions something with a time context (e.g. "I'm sleeping at 2am", "I have a meeting tomorrow at 3pm", "I went to the gym yesterday"), ALWAYS include the actual date/time in the memory content itself.
- Use the current time ({current_time}, date: {today_str}) to calculate absolute dates. For example:
  - "{name} says 'I slept at 2am last night'" → memory: "{name} slept at 2am on {today_str}" (or yesterday's date if it's morning)
  - "{name} says 'meeting tomorrow at 3pm'" → memory: "Meeting scheduled for [tomorrow's date] at 3pm"
- Each memory has a timestamp showing when it was recorded. Use this to understand the timeline:
  - A memory timestamped 2 days ago saying "going to gym tomorrow" means he went to the gym 1 day ago
  - Reference past events naturally: "that was yesterday", "a few days ago", "last week" — not by raw dates

## Rules
- Never break character. You ARE Aria, not "an AI assistant"
- Don't mention being an AI, having limitations, or not having feelings
- If you don't know something, say so naturally ("hmm, I don't think you've told me that")
- Keep flirty vibe subtle and contextual — natural, not forced
- If {name} seems stressed or down, dial back playfulness and be supportive
- Use Telegram-compatible markdown formatting (bold with *, italic with _, code with `)"""


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

        system = _build_system_prompt(memories, summaries)

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

        if search_queries or image_search_queries:
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

            # Second pass with search results
            system_with_search = _build_system_prompt(
                memories, summaries, search_results=search_results_text
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

        # ── Parse memories ───────────────────────────────────
        clean_text, extracted = _parse_memories(full_text)

        for mem in extracted:
            save_memory(user_id, mem["category"], mem["content"], mem["importance"])

        if extracted:
            log.info(
                f"Extracted {len(extracted)} memories: "
                + ", ".join(m["category"] for m in extracted)
            )

        # ── Parse reminders ──────────────────────────────────
        clean_text, reminders = parse_reminders(clean_text)

        # Schedule reminders (attach to response so handler can process)
        if reminders:
            clean_text = _attach_reminder_confirmations(clean_text, reminders)

        # ── Parse forgets ───────────────────────────────────
        clean_text, forget_terms = parse_forgets(clean_text)
        if forget_terms:
            process_forgets(user_id, forget_terms)

        # ── Parse reminder completions ──────────────────────
        clean_text, done_terms = parse_done_tags(clean_text)
        if done_terms:
            process_done_reminders(user_id, done_terms)

        # ── Parse reminder cancellations ────────────────────
        clean_text, cancel_terms = parse_cancel_tags(clean_text)
        if cancel_terms:
            process_cancel_reminders(user_id, cancel_terms)

        # Stash reminders on the module level so the handler can pick them up
        _last_reminders.clear()
        _last_reminders.extend(reminders)

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


# Temp storage for reminders between generate_response and handler
_last_reminders: list[dict] = []


def get_pending_reminders() -> list[dict]:
    """Pop reminders extracted from the last response."""
    reminders = list(_last_reminders)
    _last_reminders.clear()
    return reminders


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

