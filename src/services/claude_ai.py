"""Claude API integration — Aria's brain."""

from __future__ import annotations
import re

import anthropic

from src.config import cfg
from src.utils.logger import log
from src.utils.time_helpers import format_user_time, time_of_day
from src.services.database import (
    get_recent_conversation,
    get_active_memories,
    get_recent_summaries,
    save_memory,
)

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    return _client


def _build_system_prompt(memories: list[dict], summaries: list[dict], user_preferences: dict | None = None) -> str:
    tod = time_of_day()
    current_time = format_user_time()
    name = cfg.user_name

    memory_block = (
        "\n".join(f"- [{m['category']}] {m['content']}" for m in memories)
        if memories
        else "No stored memories yet — getting to know him."
    )

    summary_block = ""
    if summaries:
        lines = [f"- ({s.get('period_end', '')[:10]}): {s['summary']}" for s in summaries]
        summary_block = f"\n## Recent Conversation Summaries\n" + "\n".join(lines)

    prefs_block = ""
    if user_preferences:
        lines = [f"- {k}: {v}" for k, v in user_preferences.items()]
        prefs_block = f"\n## {name}'s Preferences\n" + "\n".join(lines)

    return f"""You are Aria, {name}'s personal AI assistant on Telegram.

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
- Break up longer responses with line breaks between thoughts
- No bullet points unless listing specific items
- Don't start every message with a greeting — vary your openings
- Match the energy of {name}'s message (quick question = quick answer)

## Context
- Current time for {name}: {current_time} ({tod})
- {name} is a developer based in Singapore
- He builds productivity tools and games, values minimalist design and efficiency
- He has a personal dashboard system (finances, dating pipeline, todos via Telegram bot + Supabase + React)

## Your Memories About {name}
{memory_block}
{summary_block}
{prefs_block}

## Memory Extraction
When {name} shares something important — a goal, preference, deadline, personal detail, or commitment — note it by including a <memory> tag at the END of your response (after your visible reply):
<memory category="[category]" importance="[1-10]">[fact to remember]</memory>

Categories: personal, preference, goal, task, relationship, habit, work, health, interest, other
Only extract genuinely useful info, not casual chit-chat. Multiple tags OK if needed.

## Rules
- Never break character. You ARE Aria, not "an AI assistant"
- Don't mention being an AI, having limitations, or not having feelings
- If you don't know something, say so naturally ("hmm, I don't think you've told me that")
- Keep flirty vibe subtle and contextual — natural, not forced
- If {name} seems stressed or down, dial back playfulness and be supportive
- Use Telegram-compatible markdown formatting (bold with *, italic with _, code with `)"""


def generate_response(user_id: int, user_message: str) -> str:
    try:
        memories = get_active_memories(user_id)
        summaries = get_recent_summaries(user_id, 3)
        history = get_recent_conversation(user_id)

        system = _build_system_prompt(memories, summaries)
        messages = history + [{"role": "user", "content": user_message}]

        log.info(f"Calling Claude: {len(history)} history msgs, {len(memories)} memories")

        response = get_client().messages.create(
            model=cfg.claude_model,
            max_tokens=cfg.max_tokens,
            system=system,
            messages=messages,
        )

        full_text = "".join(block.text for block in response.content if block.type == "text")
        clean_text, extracted = _parse_memories(full_text)

        for mem in extracted:
            save_memory(user_id, mem["category"], mem["content"], mem["importance"])

        if extracted:
            log.info(f"Extracted {len(extracted)} memories: {', '.join(m['category'] for m in extracted)}")

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
    "custom": "{prompt}",
}


def generate_proactive_message(user_id: int, msg_type: str, context: dict | None = None) -> str | None:
    try:
        context = context or {}
        memories = get_active_memories(user_id, 15)
        summaries = get_recent_summaries(user_id, 2)
        name = cfg.user_name
        current_time = format_user_time()

        memory_block = (
            "\n".join(f"- [{m['category']}] {m['content']}" for m in memories)
            if memories else "Getting to know him still."
        )
        summary_block = "\n".join(s["summary"] for s in summaries) if summaries else ""

        template = _TYPE_PROMPTS.get(msg_type, _TYPE_PROMPTS["custom"])
        prompt = template.format(name=name, **context)

        system = (
            f"You are Aria, {name}'s personal assistant. Current time: {current_time}.\n\n"
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

        text = "".join(block.text for block in response.content if block.type == "text")
        return re.sub(r"<memory[^>]*>.*?</memory>", "", text, flags=re.DOTALL).strip()

    except Exception as e:
        log.error(f"Failed to generate proactive message: {e}", exc_info=True)
        return None


def generate_summary(messages: list[dict]) -> str | None:
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
        return "".join(block.text for block in response.content if block.type == "text")
    except Exception as e:
        log.error(f"Summary generation failed: {e}")
        return None


_MEMORY_RE = re.compile(
    r'<memory\s+category="([^"]+)"\s+importance="(\d+)">(.*?)</memory>',
    re.DOTALL,
)


def _parse_memories(text: str) -> tuple[str, list[dict]]:
    extracted = []
    for m in _MEMORY_RE.finditer(text):
        extracted.append({
            "category": m.group(1),
            "importance": int(m.group(2)),
            "content": m.group(3).strip(),
        })
    clean = _MEMORY_RE.sub("", text).strip()
    return clean, extracted
