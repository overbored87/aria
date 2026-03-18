#!/bin/bash
set -e
echo "🔧 Patching Aria Bot — Time-bound memory expiry..."
echo ""
echo "⚠️  FIRST: Run this SQL in Supabase SQL Editor:"
echo "────────────────────────────────────────────────"
echo "ALTER TABLE memories ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;"
echo ""
echo "CREATE INDEX IF NOT EXISTS idx_memories_expires"
echo "  ON memories(expires_at)"
echo "  WHERE is_active = TRUE AND expires_at IS NOT NULL;"
echo "────────────────────────────────────────────────"
echo ""
read -p "Press Enter once the SQL migration is done (or Ctrl+C to abort)..."
echo ""

# ── database.py ──────────────────────────────────────────────
# 1. save_memory: add expires_at parameter
# 2. get_active_memories: filter out expired memories, include expires_at in select
# 3. NEW: get_past_event_memories for reference context

cd "$(dirname "$0")"

# --- save_memory: add expires_at param ---
cat > /tmp/aria_patch_db.py << 'PYEOF'
import re, sys

with open("src/services/database.py", "r") as f:
    content = f.read()

# 1. Add expires_at param to save_memory
old_sig = '''def save_memory(
    user_id: int,
    category: str,
    content: str,
    importance: int = 5,
    source_message_id: str | None = None,
) -> str | None:
    """Store an extracted memory. Returns UUID or None."""
    db = get_db()
    row = {
        "user_id": user_id,
        "category": category,
        "content": content,
        "importance": max(1, min(10, importance)),
        "source_message_id": source_message_id,
    }
    result = db.table("memories").insert(row).execute()
    if result.data:
        log.info(f"Memory saved [{category}]: {content[:80]}")
        return result.data[0]["id"]
    return None'''

new_sig = '''def save_memory(
    user_id: int,
    category: str,
    content: str,
    importance: int = 5,
    source_message_id: str | None = None,
    expires_at: str | None = None,
) -> str | None:
    """Store an extracted memory. Returns UUID or None.

    Args:
        expires_at: ISO timestamp for time-bound memories. NULL = permanent.
    """
    db = get_db()
    row = {
        "user_id": user_id,
        "category": category,
        "content": content,
        "importance": max(1, min(10, importance)),
        "source_message_id": source_message_id,
    }
    if expires_at:
        row["expires_at"] = expires_at
    result = db.table("memories").insert(row).execute()
    if result.data:
        exp_str = f", expires={expires_at}" if expires_at else ""
        log.info(f"Memory saved [{category}]: {content[:80]}{exp_str}")
        return result.data[0]["id"]
    return None'''

content = content.replace(old_sig, new_sig)

# 2. Replace get_active_memories to filter expired + include expires_at
old_active = '''def get_active_memories(user_id: int, limit: int = 30) -> list[dict]:
    """Fetch active memories sorted by importance then recency."""
    db = get_db()
    result = (
        db.table("memories")
        .select("id, category, content, importance, created_at")
        .eq("user_id", user_id)
        .eq("is_active", True)
        .order("importance", desc=True)
        .order("last_referenced_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []'''

new_active = '''def get_active_memories(user_id: int, limit: int = 30) -> list[dict]:
    """Fetch active memories that are NOT expired.

    Returns permanent memories (expires_at IS NULL) and future events
    (expires_at > now). Sorted by importance then recency.
    """
    db = get_db()
    now = datetime.utcnow().isoformat()

    # Supabase doesn't support OR conditions with .is_() easily,
    # so we fetch all active and filter in Python
    result = (
        db.table("memories")
        .select("id, category, content, importance, created_at, expires_at")
        .eq("user_id", user_id)
        .eq("is_active", True)
        .order("importance", desc=True)
        .order("last_referenced_at", desc=True)
        .limit(limit * 2)  # fetch extra to account for filtering
        .execute()
    )

    memories = []
    for row in result.data or []:
        exp = row.get("expires_at")
        if exp is None or exp > now:
            memories.append(row)
            if len(memories) >= limit:
                break

    return memories


def get_past_event_memories(user_id: int, limit: int = 15, max_age_days: int = 30) -> list[dict]:
    """Fetch expired time-bound memories (past events) for reference context.

    Returns memories where expires_at is in the past but within max_age_days.
    Older than that, they\'re considered stale enough to drop from context.
    """
    db = get_db()
    now = datetime.utcnow().isoformat()
    cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()

    result = (
        db.table("memories")
        .select("id, category, content, importance, created_at, expires_at")
        .eq("user_id", user_id)
        .eq("is_active", True)
        .not_.is_("expires_at", "null")
        .lte("expires_at", now)
        .gte("expires_at", cutoff)
        .order("expires_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []'''

content = content.replace(old_active, new_active)

with open("src/services/database.py", "w") as f:
    f.write(content)

print("  ✅ database.py patched")
PYEOF

python3 /tmp/aria_patch_db.py

# --- claude_ai.py patches ---
cat > /tmp/aria_patch_claude.py << 'PYEOF'
import re, sys

with open("src/services/claude_ai.py", "r") as f:
    content = f.read()

# 1. Add get_past_event_memories import
content = content.replace(
    "from src.services.database import (\n    get_recent_conversation,\n    get_active_memories,\n    get_recent_summaries,",
    "from src.services.database import (\n    get_recent_conversation,\n    get_active_memories,\n    get_past_event_memories,\n    get_recent_summaries,"
)

# 2. Update _build_system_prompt signature to accept past_events
content = content.replace(
    '''def _build_system_prompt(
    memories: list[dict],
    summaries: list[dict],
    user_preferences: dict | None = None,
    search_results: str | None = None,
) -> str:''',
    '''def _build_system_prompt(
    memories: list[dict],
    summaries: list[dict],
    past_events: list[dict] | None = None,
    user_preferences: dict | None = None,
    search_results: str | None = None,
) -> str:'''
)

# 3. Add expires display to memory_block and add past_events_block after it
old_memory_block = '''    memory_block = ""
    if memories:
        mem_lines = []
        for m in memories:
            ts = utc_to_user(m["created_at"]) if m.get("created_at") else ""
            mem_lines.append(f"- [{m[\'category\']}] ({ts}) {m[\'content\']}")
        memory_block = "\\n".join(mem_lines)
    else:
        memory_block = "No stored memories yet — getting to know him."

    summary_block'''

new_memory_block = '''    memory_block = ""
    if memories:
        mem_lines = []
        for m in memories:
            ts = utc_to_user(m["created_at"]) if m.get("created_at") else ""
            exp = ""
            if m.get("expires_at"):
                exp = f" [expires: {utc_to_user(m[\'expires_at\'])}]"
            mem_lines.append(f"- [{m[\'category\']}] ({ts}) {m[\'content\']}{exp}")
        memory_block = "\\n".join(mem_lines)
    else:
        memory_block = "No stored memories yet — getting to know him."

    past_events_block = ""
    if past_events:
        pe_lines = []
        for m in past_events:
            ts = utc_to_user(m["created_at"]) if m.get("created_at") else ""
            exp = utc_to_user(m["expires_at"]) if m.get("expires_at") else ""
            pe_lines.append(f"- [{m[\'category\']}] ({ts}) {m[\'content\']} [event ended: {exp}]")
        past_events_block = (
            f"\\n## Past Events (for reference — these already happened, do NOT remind or follow up)\\n"
            + "\\n".join(pe_lines)
        )

    summary_block'''

content = content.replace(old_memory_block, new_memory_block)

# 4. Add past_events_block to system prompt template (after memory_block)
content = content.replace(
    "{memory_block}\n{summary_block}",
    "{memory_block}\n{past_events_block}\n{summary_block}"
)

# 5. Replace memory extraction instructions in system prompt
old_memory_instructions = '''## Memory Extraction
When {name} shares something important — a goal, preference, deadline, personal detail, or commitment — note it by including a <memory> tag at the END of your response (after your visible reply):
<memory category="[category]" importance="[1-10]">[fact to remember]</memory>

Categories: personal, preference, goal, task, relationship, habit, work, health, interest, other
Only extract genuinely useful info, not casual chit-chat. Multiple tags OK if needed.

## Forgetting / Resolving Memories
When {name} tells you something is done, resolved, no longer relevant, or asks you to stop following up on something, you MUST include a <forget> tag to deactivate the old memory:
<forget>[keyword or phrase from the original memory]</forget>

This is CRITICAL. Examples:
- "{name} says \'I already took the earrings out\'" → <forget>earring</forget>
- "{name} says \'stop reminding me about the dentist\'" → <forget>dentist</forget>
- "{name} says \'I quit that job\'" → <forget>works at</forget> and add a new memory with the update
- "{name} says \'nevermind about the gym goal\'" → <forget>gym</forget>
The search term should match a keyword in the original memory content. Use the memory list above to find the right term. You can include multiple <forget> tags. ALWAYS forget before adding an updated memory — otherwise both the old and new memory will coexist and cause confusion.

IMPORTANT — Time-aware memories:
- When {name} mentions something with a time context (e.g. "I\'m sleeping at 2am", "I have a meeting tomorrow at 3pm", "I went to the gym yesterday"), ALWAYS include the actual date/time in the memory content itself.
- Use the current time ({current_time}, date: {today_str}) to calculate absolute dates. For example:
  - "{name} says \'I slept at 2am last night\'" → memory: "{name} slept at 2am on {today_str}" (or yesterday\'s date if it\'s morning)
  - "{name} says \'meeting tomorrow at 3pm\'" → memory: "Meeting scheduled for [tomorrow\'s date] at 3pm"
- Each memory has a timestamp showing when it was recorded. Use this to understand the timeline:
  - A memory timestamped 2 days ago saying "going to gym tomorrow" means he went to the gym 1 day ago
  - Reference past events naturally: "that was yesterday", "a few days ago", "last week" — not by raw dates'''

new_memory_instructions = '''## Memory Extraction
When {name} shares something important — a goal, preference, deadline, personal detail, or commitment — note it by including a <memory> tag at the END of your response (after your visible reply):
<memory category="[category]" importance="[1-10]">[fact to remember]</memory>

Categories: personal, preference, goal, task, relationship, habit, work, health, interest, other
Only extract genuinely useful info, not casual chit-chat. Multiple tags OK if needed.

### Time-Bound vs Permanent Memories
For events, meetings, deadlines, or anything with a specific time — add an `expires` attribute:
<memory category="personal" importance="5" expires="YYYY-MM-DDTHH:MM">Meeting friend at 3pm on 2026-03-18</memory>

Rules for `expires`:
- ALWAYS include `expires` for time-bound events (meetings, appointments, deadlines, plans)
- The `expires` value should be when the event is OVER — not when it starts
  - 1-hour meeting at 3pm → expires="...T16:00"
  - Dinner at 7pm → expires="...T22:00"
  - All-day event → expires="...T23:59"
  - Deadline to submit by Friday → expires on Friday end of day
- If unclear how long the event lasts, default to 2 hours after start time
- Use {name}\'s timezone ({cfg.user_timezone}), format: YYYY-MM-DDTHH:MM
- Current time is {current_time}, today is {today_str} — use these to calculate dates
- NEVER include `expires` for permanent facts: preferences, habits, relationships, personal traits, ongoing goals
  - "likes to gym" → NO expires
  - "works at Company X" → NO expires
  - "gym session tomorrow at 7am" → expires="[tomorrow]T09:00"

After a time-bound memory expires, it moves to the "Past Events" section above. {name} can still reference it ("remember that friend I met last Sunday?") but you will NOT proactively bring it up or remind him about it.

IMPORTANT — Always include concrete dates/times in the memory content itself:
- "{name} says \'meeting tomorrow at 3pm\'" → memory: "Meeting scheduled for [tomorrow\'s date] at 3pm"
- "{name} says \'I slept at 2am last night\'" → memory: "{name} slept at 2am on [yesterday\'s date]"

## Forgetting / Resolving Memories
When {name} tells you something is done, resolved, no longer relevant, or asks you to stop following up on something, you MUST include a <forget> tag to deactivate the old memory:
<forget>[keyword or phrase from the original memory]</forget>

This is CRITICAL. Examples:
- "{name} says \'I already took the earrings out\'" → <forget>earring</forget>
- "{name} says \'stop reminding me about the dentist\'" → <forget>dentist</forget>
- "{name} says \'I quit that job\'" → <forget>works at</forget> and add a new memory with the update
- "{name} says \'nevermind about the gym goal\'" → <forget>gym</forget>
The search term should match a keyword in the original memory content. Use the memory list above to find the right term. You can include multiple <forget> tags. ALWAYS forget before adding an updated memory — otherwise both the old and new memory will coexist and cause confusion.

Note: For time-bound memories with `expires`, you usually do NOT need <forget> — they auto-transition to past events. Only use <forget> if {name} explicitly says something is cancelled or wrong.'''

content = content.replace(old_memory_instructions, new_memory_instructions)

# 6. Update _MEMORY_RE regex to capture optional expires
content = content.replace(
    """_MEMORY_RE = re.compile(
    r'<memory\\s+category="([^"]+)"\\s+importance="(\\d+)">(.*?)</memory>',
    re.DOTALL,
)""",
    """_MEMORY_RE = re.compile(
    r'<memory\\s+category="([^"]+)"\\s+importance="(\\d+)"(?:\\s+expires="([^"]+)")?\\s*>(.*?)</memory>',
    re.DOTALL,
)"""
)

# 7. Replace _parse_memories to handle expires
old_parse = '''def _parse_memories(text: str) -> tuple[str, list[dict]]:
    """Extract <memory> tags and return (clean_text, list_of_memories)."""
    extracted = []
    for m in _MEMORY_RE.finditer(text):
        extracted.append({
            "category": m.group(1),
            "importance": int(m.group(2)),
            "content": m.group(3).strip(),
        })
    clean = _MEMORY_RE.sub("", text).strip()
    return clean, extracted'''

new_parse = '''def _parse_memories(text: str) -> tuple[str, list[dict]]:
    """Extract <memory> tags and return (clean_text, list_of_memories).

    Now supports optional expires attribute for time-bound memories.
    """
    tz = ZoneInfo(cfg.user_timezone)
    extracted = []
    for m in _MEMORY_RE.finditer(text):
        expires_raw = m.group(3)  # None if not present
        expires_at = None

        if expires_raw:
            try:
                expires_raw = expires_raw.strip()
                dt = datetime.fromisoformat(expires_raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=tz)
                expires_at = dt.isoformat()
                log.info(f"Memory expires at: {expires_at}")
            except Exception as e:
                log.warning(f"Failed to parse memory expires \\'{expires_raw}\\': {e}")

        extracted.append({
            "category": m.group(1),
            "importance": int(m.group(2)),
            "content": m.group(4).strip(),
            "expires_at": expires_at,
        })
    clean = _MEMORY_RE.sub("", text).strip()
    return clean, extracted'''

content = content.replace(old_parse, new_parse)

# 8. Update save_memory call to pass expires_at
content = content.replace(
    "            save_memory(user_id, mem[\"category\"], mem[\"content\"], mem[\"importance\"])",
    "            save_memory(\n                user_id, mem[\"category\"], mem[\"content\"],\n                mem[\"importance\"], expires_at=mem.get(\"expires_at\"),\n            )"
)

# 9. Update generate_response to fetch past_events and pass to system prompt
content = content.replace(
    "        memories = get_active_memories(user_id)\n        summaries = get_recent_summaries(user_id, 3)\n        history = get_recent_conversation(user_id)\n\n        system = _build_system_prompt(memories, summaries)",
    "        memories = get_active_memories(user_id)\n        past_events = get_past_event_memories(user_id)\n        summaries = get_recent_summaries(user_id, 3)\n        history = get_recent_conversation(user_id)\n\n        system = _build_system_prompt(memories, summaries, past_events=past_events)"
)

# 10. Update the log line to include past events count
content = content.replace(
    '''        log.info(
            f"Calling Claude: {len(history)} history msgs, {len(memories)} memories, "
            f"time={format_user_time()}, has_image={image_data is not None}"
        )''',
    '''        log.info(
            f"Calling Claude: {len(history)} history msgs, {len(memories)} memories, "
            f"{len(past_events)} past events, "
            f"time={format_user_time()}, has_image={image_data is not None}"
        )'''
)

# 11. Update second-pass system prompt call to include past_events
content = content.replace(
    '''            system_with_search = _build_system_prompt(
                memories, summaries, search_results=search_results_text
            )''',
    '''            system_with_search = _build_system_prompt(
                memories, summaries, past_events=past_events,
                search_results=search_results_text,
            )'''
)

# 12. Update memory extraction log to show expires
content = content.replace(
    '''        if extracted:
            log.info(
                f"Extracted {len(extracted)} memories: "
                + ", ".join(m["category"] for m in extracted)
            )''',
    '''        if extracted:
            log.info(
                f"Extracted {len(extracted)} memories: "
                + ", ".join(
                    f"{m[\'category\']}{'(exp)' if m.get(\'expires_at\') else \'\'}"
                    for m in extracted
                )
            )'''
)

with open("src/services/claude_ai.py", "w") as f:
    f.write(content)

print("  ✅ claude_ai.py patched")
PYEOF

python3 /tmp/aria_patch_claude.py

# Cleanup
rm -f /tmp/aria_patch_db.py /tmp/aria_patch_claude.py

echo ""
echo "✅ Patch applied!"
echo ""
echo "Changes:"
echo "  🕐 memories.expires_at — nullable timestamp for time-bound memories"
echo "  📋 Active memories now exclude expired events"
echo "  📜 Expired events shown in separate 'Past Events' reference section (30-day window)"
echo "  🏷️  New <memory> tag format: expires=\"YYYY-MM-DDTHH:MM\" (optional)"
echo "  🧠 System prompt teaches Aria when to use expires vs not"
echo "  🚫 Aria won't proactively bring up past events, but can reference them if asked"
echo ""
echo "Then: git add -A && git commit -m \"Add time-bound memory expiry\" && git push"
