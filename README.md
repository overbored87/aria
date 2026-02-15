# Aria — Personal AI Assistant Telegram Bot

A Telegram bot powered by Claude Sonnet 4.5 with persistent memory, personality, and proactive messaging.

## Architecture

```
Telegram ←→ python-telegram-bot (polling)
                    ↓
              Message Handler
                    ↓
            ┌───────┴───────┐
            │  Claude API   │  ← System prompt + memories + conversation history
            │  (Sonnet 4.5) │
            └───────┬───────┘
                    ↓
              Supabase (PostgreSQL)
              ├── conversations    (message history)
              ├── memories         (extracted facts)
              ├── summaries        (compressed context)
              └── scheduled_msgs   (proactive messages)

        APScheduler (cron jobs)
              ├── Morning check-in    (7:30 AM SGT)
              ├── Evening check-in    (9:00 PM SGT)
              ├── Task follow-ups     (2:00 PM weekdays)
              └── DB scheduled sweep  (every 5 min)
```

## Setup

### 1. Create Telegram Bot

1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. `/newbot` → name it "Aria" → get your `TELEGRAM_BOT_TOKEN`
3. Message your bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your user ID in the `from.id` field

### 2. Set Up Supabase

1. Create a project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run the contents of `src/db/migration.sql`
3. Get your project URL and **service role** key from **Settings → API**

### 3. Environment Variables

Copy `.env.example` to `.env` and fill in all values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | From BotFather |
| `ALLOWED_USER_ID` | Your Telegram numeric user ID |
| `ANTHROPIC_API_KEY` | From console.anthropic.com |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Supabase service role key (not anon) |
| `USER_TIMEZONE` | IANA timezone (default: `Asia/Singapore`) |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### 4. Run Locally

```bash
pip install -r requirements.txt
python -m src.main
```

### 5. Deploy to Render

**Option A: Blueprint (recommended)**
1. Push to GitHub
2. Render → New → Blueprint → connect repo
3. `render.yaml` auto-configures everything
4. Add env vars in Render dashboard

**Option B: Manual**
1. Render → New → Background Worker
2. Runtime: Python
3. Build: `pip install -r requirements.txt`
4. Start: `python -m src.main`
5. Add all env vars
6. Instance type: **Starter** ($7/mo — always on)

> ⚠️ **Important:** Use a **Background Worker**, not a Web Service. The bot uses polling, not webhooks, so it doesn't need an HTTP port. Make sure only ONE instance runs to avoid duplicate message handling.

## How It Works

### Conversation Flow

1. User sends message → saved to `conversations` table
2. Recent messages fetched within token budget (8,000 tokens ≈ 50 msgs)
3. Active memories loaded (sorted by importance)
4. Conversation summaries loaded for older context
5. Claude generates response with full context + Aria personality
6. Response parsed for `<memory>` tags → extracted facts saved to `memories`
7. Clean response sent back to user
8. Background check: if 40+ messages in 24h, trigger conversation summary

### Memory System

Aria automatically extracts and stores important information from conversations:

- **Categories:** personal, preference, goal, task, relationship, habit, work, health, interest, other
- **Importance:** 1-10 scale (higher = more likely to be included in context)
- **Lifecycle:** Active memories included in every Claude API call; old/superseded ones can be deactivated

View your memories: `/memories`

### Proactive Messaging

| Schedule | Type | Description |
|---|---|---|
| 7:30 AM SGT | Morning check-in | Energizing start, references goals |
| 2:00 PM SGT (weekdays) | Task follow-up | Nudge about highest-priority task |
| 9:00 PM SGT | Evening check-in | Wind-down, ask about the day |

**Safety limits:**
- Max 3 proactive messages per day
- Silent during quiet hours (11 PM – 7 AM SGT)
- No duplicate types in one day

### Context Windowing

To avoid sending entire conversation history to Claude:

1. **Token budget:** Most recent messages up to ~8,000 tokens
2. **Summaries:** Older conversations compressed into summaries
3. **Memories:** Key facts stored separately, always available
4. **Result:** Claude gets recent detail + long-term context efficiently

## Commands

| Command | Description |
|---|---|
| `/start` | Introduction message |
| `/memories` | View stored memories |
| `/clear` | Reset conversation context (keeps memories) |
| `/help` | Show available commands |

## Customization

### Personality Tuning
Edit the system prompt in `src/services/claude_ai.py` → `_build_system_prompt()`

### Schedule Tuning
Edit cron triggers in `src/services/scheduler.py` → `init_scheduler()`

### Adding New Proactive Message Types
1. Add type to `scheduled_messages.type` CHECK constraint in migration
2. Add prompt template in `claude_ai.py` → `_TYPE_PROMPTS`
3. Add scheduler job in `scheduler.py`

## Troubleshooting

| Issue | Fix |
|---|---|
| Bot not responding | Check `ALLOWED_USER_ID` matches your Telegram ID |
| Duplicate messages | Ensure only ONE instance running on Render |
| Proactive msgs not sending | Check timezone config, quiet hours, daily limit |
| Claude errors | Verify `ANTHROPIC_API_KEY`, check rate limits |
| DB errors | Verify Supabase credentials, check migration ran |
