# Aria

Personal assistant Telegram bot (single-user, whitelisted) with long-term memory, a
personal-wiki specialist role, web search, and vision. Python 3.13, deployed on
Railway (`railway.toml`, Nixpacks, `python -m src.main`); `render.yaml`/`Procfile`
are leftovers from Render and not the live deploy path.

## Running locally

```
pip install -r requirements.txt
python -m src.main          # needs env vars (see below); reads .env locally
python -m compileall -q src # quick syntax check — repo has no tests
ruff check src --select F401,F811,F821   # lint used in practice (not pinned in requirements)
```

Careful running the bot locally while the Railway worker is up: Telegram allows only
one poller per bot token — a second `run_polling` causes 409 Conflict errors.

## Architecture

```
src/main.py                    entry point: config validate → handlers → sidecar thread → run_polling
src/config.py                  frozen dataclass `cfg`, reads env once; ALLOWED_USER_IDS frozenset
src/handlers/telegram_handlers.py  commands, auth guard, photo pipeline, wiki approve/reject
src/services/claude_ai.py      the brain: agentic tool loop + GPT-4o wiki writer
src/services/database.py       Aria's own Supabase tables (aria_-prefixed)
src/services/dashboard.py      wiki_pages read/write (name is legacy — it's the wiki service)
src/services/summarizer.py     maybe_summarize: condenses history past a threshold
src/services/web_search.py     Serper.dev search/images; no-op if SERPER_API_KEY unset
src/sidecar/                   FastAPI on $PORT in a daemon thread — Siren federation
src/db/*.sql                   schema + migration scripts (run manually in Supabase SQL Editor)
```

### Message flow

`_handle_message` (or `_handle_photo`, which resizes to ≤1024px JPEG first) runs
`generate_response` via `asyncio.to_thread` (Anthropic SDK is sync — never call it
directly from a handler). Messages are saved **after** generation so history doesn't
double-include the current turn. Then: `parse_images` → split-send (4096-char chunks,
Markdown with plain-text fallback) → `_queue_wiki_edits` → `maybe_summarize`.

Telegram sends an album as one update per photo (shared `media_group_id`);
`_handle_photo` buffers those in `_albums` and a debounced `_flush_album` task
(2s after the last arrival) downloads them all and makes a single Claude call —
so multi-screenshot wiki requests are seen whole, not answered per photo.

### Agentic loop (`generate_response`)

Claude (`cfg.claude_model`) loops on `stop_reason == "tool_use"` with tools:
`search_wiki`, `read_wiki_page`, `propose_wiki_create/update/delete`, `web_search`.
Returns `(clean_text, wiki_edits)`. `stop_reason == "max_tokens"` must stay explicitly
handled — it used to fall through silently and the bot sent nothing (the "Aria froze"
bug). System prompt uses `cache_control: ephemeral` for prompt caching.

Response text may carry inline tags parsed post-loop: `<memory category=...>`
(save memory), `<forget>` (deactivate memories), `<image>` (send photo).

### Wiki writes are two-phase

`propose_wiki_*` tools don't touch the DB. Long-form article prose is drafted by
GPT-4o (`_call_wiki_writer`, `openai` package) — Claude orchestrates, GPT-4o writes.
Proposed edits go to `_pending_wiki_edits` (capped at 20) and are returned to the
handler, which previews them in chat; `/approve` / `/reject` applies or discards
**all** pending edits. Pending state is module-level globals — fine single-user,
would break multi-user.

## Supabase

One consolidated project holds everything: Aria's four tables (`aria_users`,
`aria_conversations`, `aria_memories`, `aria_conversation_summaries` — see
`src/db/migrate_to_new_project.sql`) plus the dashboard/wiki tables (`wiki_pages`).
Both env-var pairs (`SUPABASE_*` and `DASHBOARD_SUPABASE_*`) now point at the **same
project**; they're kept separate in code because the keys may differ and
dashboard/wiki access is optional (`dashboard_svc.is_configured()`).

**Key-role gotcha:** despite the name, `SUPABASE_SERVICE_KEY` holds an **anon**-role
JWT, not a `service_role` key (decode the JWT's `role` claim to confirm). It works
only because the aria_ tables have **no RLS**, so anon can read/write them freely.
This matters for the wiki: `wiki_pages` may have RLS, so whether an approved edit
actually writes depends on `DASHBOARD_SUPABASE_KEY` having write access (a
`service_role` key, or an anon key plus a permissive write policy). If an edit
"succeeds" but the page doesn't change, check that first — `_apply_wiki_edit` now
reports honest failure and `update_wiki_page` logs a 0-row write, rather than
silently forking a duplicate page.

`src/db/migration.sql` is the historical pre-rename schema — don't run it; use
`migrate_to_new_project.sql` for a fresh setup.

There is **no reminders/scheduler/proactive-messaging feature** — that whole area
(apscheduler, `reminders`/`scheduled_messages`/`proactive_message_log` tables) was
removed in f464db4. Don't reintroduce apscheduler casually; the sidecar thread and
polling loop own the process's threading story.

## Sidecar (Siren federation)

FastAPI app in a daemon thread (uvicorn signal handlers disabled — polling owns the
main thread). Endpoints: `GET /health` (Railway healthcheck), `POST /invoke`
(auth: `X-Siren-Key` == `SIREN_API_KEY`), `GET /jobs/{id}`.

`/invoke` tools: `search_wiki` and `journal_write` (synchronous), and
`research_and_draft` (async).

**`journal_write`** persists a journal entry to Day Two (`src/services/journal.py`).
Day Two is a **separate Supabase project with RLS bound to `auth.uid()`**, so there
is no server-side DB write available — entries go over HTTP to its
`POST /api/journal` route, which holds the `service_role` key and owns the schema
(`JOURNAL_API_URL`/`JOURNAL_API_KEY`). Siren composes the entry in her own
conversation and this stores it **verbatim** — nothing here rewrites or summarises
it, deliberately (contrast the wiki writer, which drafts prose). Synchronous, with
no approval step: the text is the user's own words, already agreed in Siren's chat.
A failed write returns 502 rather than a false success, so Siren can retry instead
of losing a reflection. After the entry is safely stored, durable facts are
harvested into `aria_memories` in a background thread
(`claude_ai.extract_memories_from_entry`) — best-effort, never blocks or fails the
write. That memory harvest is the reason journaling routes through Aria at all
rather than Siren writing to Day Two directly.
The latter (`src/sidecar/research.py`) returns a `job_id` immediately, then a
daemon thread runs Serper search + the GPT-4o wiki writer, stores the draft in
the in-memory `jobs` store (`src/sidecar/jobs.py`, ephemeral — lost on restart),
and DMs the user via Aria's own bot token with inline **Save / Discard buttons**
(`callback_data` `wiki_ok:<job_id>` / `wiki_no:<job_id>`, handled by `_cb_wiki`).
The buttons carry the job id because a tapped `/wiki_approve <job_id>` text
command drops its argument — Telegram only copies up to the first space. The
typed `/wiki_approve` / `/wiki_reject` still work as a fallback and, with no arg,
act on the latest pending draft (`jobstore.latest_actionable_job`). Button taps
arrive as `callback_query` updates, so `run_polling`'s `allowed_updates` **must**
include `"callback_query"` (see `main.py`) or the buttons silently do nothing.

It drafts a new page, or **revises an existing one** if a page already exists at
the target slug (explicit `slug` arg, else the slug derived from the title/topic)
— in which case the writer is fed the current page content and briefed to revise
rather than overwrite (`edit["type"]` is `create` vs `update`). `_cmd_wiki_approve` reuses `_apply_wiki_edit` — the same
path as the live-chat `/approve` flow — so drafts never auto-save and Siren never
touches `wiki_pages` directly. Completion is also POSTed to Siren's `/events` for
oversight. See `PLAN-aria-sidecar.md`.

## Env vars

Set in Railway dashboard for prod (deployed services never read `.env`):

- Required: `TELEGRAM_BOT_TOKEN`, `ALLOWED_USER_ID`, `ANTHROPIC_API_KEY`,
  `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`
- Optional: `ALLOWED_USER_IDS` (comma-separated extras), `OPENAI_API_KEY` (wiki
  writer), `SERPER_API_KEY` (web search), `DASHBOARD_SUPABASE_URL`/`DASHBOARD_SUPABASE_KEY`
  (wiki), `SIREN_API_KEY`/`SIREN_BASE_URL` (sidecar), `JOURNAL_API_URL`/`JOURNAL_API_KEY`
  (Day Two journal ingest), `USER_TIMEZONE`, `LOG_LEVEL`, `PORT`

Model IDs and caps live in `config.py` as code defaults (`claude-sonnet-5`, `gpt-4o`,
`max_tokens=8192`), not env vars.

## Conventions

- Direct-to-main workflow: no PRs; push to `main` triggers Railway deploy.
- Keep blocking work off the event loop (`asyncio.to_thread` for DB-heavy or
  API-sync calls from handlers).
- Register new deployed services with Sentinel for monitoring.
