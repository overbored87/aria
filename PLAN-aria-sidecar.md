# Plan: Aria's sidecar — the first real federation sidecar

Executable implementation plan. Written for a fresh session with no prior
context — read this whole file, then work the tasks in order. Each task has
acceptance criteria; verify before moving on.

This is the first sidecar in the Siren federation. Its job is twofold:
1. **Unlock delegation** — let Siren call Aria over HTTP (the sidecar contract
   Siren already speaks), so `invoke_agent("aria", ...)` stops returning "no
   sidecar deployed."
2. **Ship on-demand research** — Aria researches a topic, drafts a wiki page,
   and routes it through her *existing* approval flow, in her own voice.

The design principle behind this (decided with the user): research that
produces a durable knowledge artifact belongs to Aria, because she owns the
wiki, already has web search, and has an approve/reject flow. It is NOT an
anonymous Siren subagent. Watchtower already proves the pattern (research →
`wiki_pages`); this is the on-demand, arbitrary-topic version.

## Architecture (read before touching code)

- **Co-hosting:** Aria runs `app.run_polling()` (blocking, PTB v21, main
  thread) as her bot. The sidecar is a FastAPI app served by uvicorn in a
  **daemon thread** started *before* `run_polling()`. They share the process
  (so the sidecar imports Aria's services directly) but have independent event
  loops. This avoids restructuring Aria's proven polling loop — the single
  biggest risk — into a merged-loop rewrite.
- **Non-blocking research:** `research_and_draft` is long (several web searches
  + a GPT-4o writeup). `/invoke` must NOT run it inline — it spawns a
  `threading.Thread`, stores a job, and returns `{"job_id": ...}` immediately.
  This is what lets the user keep conversing with Siren while Aria works.
- **Approval in Aria's voice:** when the job finishes, the sidecar thread
  sends the user a Telegram message *via Aria's bot token* (raw httpx, exactly
  how proactive messages and Siren's relay already work) with the draft
  preview and `/wiki_approve <job_id>`. The user approves in Aria's Telegram,
  and the existing `_apply_wiki_edit` saves it. Siren never impersonates Aria.
- **Oversight:** on completion the sidecar also POSTs to Siren's `/events` so
  Siren has a record, per the federation rule.
- **Minimal Siren change:** Siren's `invoke_agent`/`check_agent_job` are
  generic (`{agent, tool, args}`) — no Siren code changes. Siren only needs a
  DB row update: `agent_registry.base_url` + `capabilities` for aria.

## Current state (verified 2026-07-08 — re-verify by reading the files)

Repo: `C:\Users\User\Documents\Python 2026\Aria`.

- `src/main.py` — builds the PTB `Application`, calls `register_handlers(app)`,
  `init_scheduler(app)`, then `app.run_polling(drop_pending_updates=True,
  allowed_updates=["message"])`. This is the blocking main-thread call.
- `src/config.py` — frozen `Config` dataclass, `cfg` singleton. Reads env in
  field defaults. Has `telegram_token`, `allowed_user_id`, `anthropic_api_key`,
  `serper_api_key`, `openai_api_key`, `dashboard_supabase_url/_key`.
  `ALLOWED_USER_IDS` frozenset exported separately. `validate()` checks
  required vars.
- `src/services/web_search.py` — `search(query, num_results=5) -> list[{title,
  snippet, link}]`; `format_results_for_context(results, max_results=5) -> str`.
  No-ops gracefully if `serper_api_key` unset.
- `src/services/claude_ai.py` — has `_call_wiki_writer(prompt: str) -> str`
  (GPT-4o via OpenAI, `_WIKI_WRITER_SYSTEM`, needs `openai_api_key`). Also the
  edit dict shape used throughout: `{"id", "type": "create|update|delete",
  "slug", "title", "content", "description"}`.
- `src/services/dashboard.py` — reads the shared Dashbored Supabase.
  `search_wiki(query, limit=5) -> list[{title, slug, content, updated_at}]`;
  `get_wiki_page(slug) -> dict|None`; `create_wiki_page(user_id, title, slug,
  content)`; `update_wiki_page(slug, content, title=None)`. `is_configured()`.
- `src/handlers/telegram_handlers.py` — `register_handlers(app)`; module global
  `_pending_approval`; `_apply_wiki_edit(edit) -> bool` (dispatches to
  dashboard_svc create/update/delete; uses `cfg.allowed_user_id` as user_id);
  `_is_authorized(update) -> bool`; `_cmd_approve`/`_cmd_reject`.
- `requirements.txt` — python-telegram-bot==21.10, anthropic==0.43.0,
  openai>=1.0.0, supabase==2.11.0, apscheduler, python-dotenv, Pillow.
  No fastapi/uvicorn yet.
- Deploy: currently a **Render Background Worker** (`python main.py`,
  long-polling). Background Workers can't receive inbound HTTP — see Deployment.

Siren side (repo `C:\Users\User\Documents\Python 2026\Siren`, deployed on
Railway at `https://siren-production-8df8.up.railway.app`), for reference only:
- `app/agents.py` — `invoke_agent(agent, tool, args)` POSTs
  `{base_url}/invoke` with header `X-Siren-Key: <SIREN_API_KEY>`, 60s timeout,
  returns `resp.json()`. `check_agent_job(agent, job_id)` GETs
  `{base_url}/jobs/{job_id}`. `base_url` comes from `agent_registry`.
- `app/events.py` — `POST /events` (header `X-Siren-Key`), body `EventIn`:
  `{agent, event_type, summary, payload}`, inserts into `agent_events`.
- Sidecar contract (from Siren's CLAUDE.md): `GET /health`; `POST /invoke
  {tool, args}` header `X-Siren-Key` → `{result}` or `{job_id}`;
  `GET /jobs/{id}`.

## Hard constraints

- Do NOT restructure `app.run_polling()` or merge event loops. The sidecar is
  a separate thread with `install_signal_handlers` disabled (it's not the main
  thread). Aria's bot must keep working exactly as before.
- The sidecar authenticates every `/invoke` and `/jobs` call against
  `SIREN_API_KEY` (same value Siren sends). Reject with 401 otherwise. `/health`
  is unauthenticated.
- Research drafts are NEVER auto-saved. They always go through
  `/wiki_approve`. Siren never writes to the wiki; Aria never saves without the
  user's approval.
- Reuse existing functions (`web_search.search`, `_call_wiki_writer`,
  `dashboard_svc.*`, `_apply_wiki_edit`). Don't reimplement wiki writing or
  saving.
- After each Python change: `python -m py_compile <files>`.
- Phase A must be fully working and deployed before Phase B — prove Siren can
  reach Aria at all before adding the async research feature on top.

---

# Phase A — sidecar infrastructure (the unlock)

## Task 1 — dependencies

`requirements.txt` — append:

```
fastapi==0.115.6
uvicorn==0.34.0
httpx>=0.27
```

(httpx ships with the anthropic SDK already, but pin it explicitly since the
sidecar uses it directly.)

## Task 2 — config additions

`src/config.py` — add fields to `Config` (after `serper_api_key`):

```python
    # Sidecar (Siren federation)
    siren_api_key: str = os.getenv("SIREN_API_KEY", "")
    siren_base_url: str = os.getenv("SIREN_BASE_URL", "")  # for /events oversight
    sidecar_port: int = int(os.getenv("PORT", "8080"))     # host sets PORT
```

Do NOT add these to `validate()` — the bot must still boot if the sidecar env
is unset (e.g. local dev). The sidecar self-checks `siren_api_key` at request
time.

## Task 3 — job store: `src/sidecar/__init__.py` (empty) + `src/sidecar/jobs.py`

```python
"""In-memory job store for the sidecar's async work (research drafts).

Single-user, single-process, low-frequency — a dict behind a lock is enough.
Jobs are ephemeral (lost on restart); a dropped in-flight research job just
means the user re-asks. No persistence by design.
"""

import threading
import time
import uuid

_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_MAX = 50


def create_job(tool: str, args: dict) -> str:
    jid = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[jid] = {
            "job_id": jid,
            "tool": tool,
            "status": "running",  # running -> done | failed
            "result": None,
            "error": None,
            "edit": None,  # the drafted wiki edit, for /wiki_approve
            "created_at": time.time(),
        }
        while len(_jobs) > _MAX:
            _jobs.pop(next(iter(_jobs)))
    return jid


def get_job(jid: str) -> dict | None:
    with _lock:
        return dict(_jobs[jid]) if jid in _jobs else None


def finish_job(jid: str, *, result=None, edit=None, error: str | None = None) -> None:
    with _lock:
        if jid in _jobs:
            _jobs[jid]["status"] = "failed" if error else "done"
            _jobs[jid]["result"] = result
            _jobs[jid]["edit"] = edit
            _jobs[jid]["error"] = error
```

## Task 4 — the server: `src/sidecar/server.py`

Phase A implements `/health`, `/invoke` with ONE synchronous tool
(`search_wiki`, to prove the pipe), and `/jobs/{id}`. The research tool is
added in Phase B.

```python
"""Aria's sidecar HTTP server — the Siren federation contract.

Runs in a daemon thread alongside Aria's Telegram polling (see run.py).
Endpoints: GET /health, POST /invoke (X-Siren-Key auth), GET /jobs/{id}.
"""

import os

from fastapi import FastAPI, Header, HTTPException

from src.services import dashboard as dashboard_svc
from src.sidecar import jobs as jobstore

app = FastAPI(title="Aria sidecar")


def _auth(x_siren_key: str) -> None:
    expected = os.getenv("SIREN_API_KEY", "")
    if not expected or x_siren_key != expected:
        raise HTTPException(status_code=401, detail="bad or missing X-Siren-Key")


@app.get("/health")
def health():
    return {"status": "ok", "agent": "aria"}


@app.post("/invoke")
def invoke(body: dict, x_siren_key: str = Header(default="")):
    _auth(x_siren_key)
    tool = body.get("tool")
    args = body.get("args") or {}

    if tool == "search_wiki":
        if not dashboard_svc.is_configured():
            return {"result": "wiki not configured"}
        pages = dashboard_svc.search_wiki(args.get("query", ""))
        return {"result": [{"title": p["title"], "slug": p["slug"]} for p in pages]}

    # research_and_draft is added in Phase B.
    raise HTTPException(status_code=400, detail=f"unknown tool: {tool}")


@app.get("/jobs/{job_id}")
def get_job(job_id: str, x_siren_key: str = Header(default="")):
    _auth(x_siren_key)
    job = jobstore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="no such job")
    return job
```

## Task 5 — co-host it: `src/sidecar/run.py` + wire into `main.py`

`src/sidecar/run.py`:

```python
"""Start the sidecar's uvicorn server in a daemon thread.

Signal handlers are disabled because uvicorn is NOT on the main thread —
Aria's app.run_polling() owns the main thread and its signals. The thread is
daemon so it dies with the process.
"""

import threading

import uvicorn

from src.config import cfg
from src.sidecar.server import app
from src.utils.logger import log


def start_sidecar_in_thread() -> None:
    config = uvicorn.Config(app, host="0.0.0.0", port=cfg.sidecar_port, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # not the main thread
    threading.Thread(target=server.run, daemon=True, name="aria-sidecar").start()
    log.info(f"Sidecar starting on :{cfg.sidecar_port}")
```

`src/main.py` — start the sidecar just before `run_polling`:

```python
    register_handlers(app)
    scheduler = init_scheduler(app)

    from src.sidecar.run import start_sidecar_in_thread
    start_sidecar_in_thread()
```

Accept (Phase A):
- `python -m py_compile src/config.py src/sidecar/jobs.py src/sidecar/server.py src/sidecar/run.py src/main.py`.
- Local smoke test: set `SIREN_API_KEY=test` and `PORT=8090` in a shell, run
  `python -m src.main` (or however Aria is normally started). In another shell:
  - `curl localhost:8090/health` → `{"status":"ok","agent":"aria"}`.
  - `curl -X POST localhost:8090/invoke -H "X-Siren-Key: test" -H "Content-Type: application/json" -d '{"tool":"search_wiki","args":{"query":"siren"}}'` → `{"result":[...]}` (needs dashboard env set; empty list is fine).
  - Same call WITHOUT the header → 401.
  - Confirm the Telegram bot still responds to a normal message (polling
    unaffected).
- Commit ("Add Aria sidecar: Siren federation contract over HTTP"). Deploy
  (see Deployment), then register with Siren:
  `update agent_registry set base_url = '<aria-public-url>' where name = 'aria';`
  (run on Siren's Supabase project). Then from Siren (Telegram/web): "ask Aria
  to search the wiki for X" → Siren calls `invoke_agent("aria","search_wiki",
  {...})` and reports real results. THIS is the milestone — delegation works.

---

# Phase B — research → draft → approve

## Task 6 — research module: `src/sidecar/research.py`

```python
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
```

## Task 7 — notify helpers: extend `src/sidecar/server.py`

Add a Telegram-send helper (raw httpx, Aria's own bot token — her voice) and a
Siren-oversight helper, plus the background job runner. Add these imports and
functions to `server.py`:

```python
import httpx
import threading
from src.config import cfg
from src.sidecar.research import run_research_and_draft


def _tg_send(text: str) -> None:
    try:
        httpx.post(
            f"https://api.telegram.org/bot{cfg.telegram_token}/sendMessage",
            json={"chat_id": cfg.allowed_user_id, "text": text},
            timeout=20.0,
        )
    except Exception:
        pass  # notification is best-effort


def _log_to_siren(event_type: str, summary: str, payload: dict) -> None:
    if not cfg.siren_base_url or not cfg.siren_api_key:
        return
    try:
        httpx.post(
            f"{cfg.siren_base_url.rstrip('/')}/events",
            headers={"X-Siren-Key": cfg.siren_api_key},
            json={"agent": "aria", "event_type": event_type, "summary": summary, "payload": payload},
            timeout=15.0,
        )
    except Exception:
        pass


def _run_research_job(job_id: str, args: dict) -> None:
    topic = args.get("topic", "").strip()
    try:
        edit = run_research_and_draft(topic, args.get("slug"), args.get("title"))
        jobstore.finish_job(job_id, result=f"drafted '{edit['title']}'", edit=edit)
        words = len(edit["content"].split())
        _tg_send(
            f"🔎 I researched \"{topic}\" and drafted a {words}-word page, "
            f"\"{edit['title']}\".\n\n/wiki_approve {job_id}  to save\n"
            f"/wiki_reject {job_id}  to discard"
        )
        _log_to_siren("research_drafted", f"Drafted wiki page '{edit['title']}' from research on {topic}",
                      {"job_id": job_id, "slug": edit["slug"]})
    except Exception as e:
        jobstore.finish_job(job_id, error=str(e))
        _tg_send(f"⚠️ My research on \"{topic}\" hit a snag: {e}")
```

Then add the `research_and_draft` branch to `invoke()` (before the final
`raise`):

```python
    if tool == "research_and_draft":
        if not (args.get("topic") or "").strip():
            raise HTTPException(status_code=400, detail="topic is required")
        job_id = jobstore.create_job(tool, args)
        threading.Thread(target=_run_research_job, args=(job_id, args), daemon=True).start()
        return {"job_id": job_id}
```

## Task 8 — approval commands in `src/handlers/telegram_handlers.py`

Add two command handlers that reuse the existing `_apply_wiki_edit`. Add near
`_cmd_approve`:

```python
async def _cmd_wiki_approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    from src.sidecar import jobs as jobstore
    if not context.args:
        await update.message.reply_text("Usage: /wiki_approve <job_id>")
        return
    job = jobstore.get_job(context.args[0])
    if not job or not job.get("edit"):
        await update.message.reply_text("No such research draft (it may have expired).")
        return
    edit = job["edit"]
    ok = await asyncio.to_thread(_apply_wiki_edit, edit)
    await update.message.reply_text(
        f"{'✅ Saved' if ok else '❌ Failed'}: {edit.get('title') or edit['slug']}"
    )


async def _cmd_wiki_reject(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    await update.message.reply_text("Discarded. Nothing was saved.")
```

Register them in `register_handlers` (alongside the existing CommandHandlers):

```python
    app.add_handler(CommandHandler("wiki_approve", _cmd_wiki_approve))
    app.add_handler(CommandHandler("wiki_reject", _cmd_wiki_reject))
```

(Confirm `asyncio` and `CommandHandler` are already imported in this file —
they are, from the existing `/approve` handler.)

## Task 9 — register the capability with Siren

On Siren's Supabase project, update Aria's roster row so Siren knows research
exists (Siren surfaces `capabilities` via `list_agents` in her prompt):

```sql
update agent_registry
set capabilities = '["search_wiki","research_and_draft"]'::jsonb
where name = 'aria';
```

Optional Siren-side polish (repo `Siren`, `app/tools.py`): extend the
`invoke_agent` tool description to mention "ask Aria to research a topic and
draft a wiki page (returns a job_id; Aria messages the user her draft for
approval — no need to poll)". Only if Siren isn't already choosing it.

## Task 10 — verify Phase B end to end (ONE run)

This spends real Serper + GPT-4o + Anthropic calls; do it once.

1. `python -m py_compile` the changed files. Deploy Aria.
2. From Siren (voice or text): "Ask Aria to research <some current topic> and
   add it to my wiki."
3. Expect: Siren delegates and replies that Aria's on it (job_id returned
   fast; conversation stays responsive — try sending Siren another message
   while it runs, it should answer).
4. Within ~30-60s, Aria's OWN Telegram bot messages you the draft summary with
   `/wiki_approve <id>`.
5. `/wiki_approve <id>` in Aria's chat → "✅ Saved" → confirm the page exists
   in `wiki_pages` (and in the Wiki web app).
6. Check Siren has an `agent_events` row (`event_type='research_drafted'`) —
   ask Siren "what has Aria been up to?" (she reads `get_agent_events`).

---

## Deployment (decision required — flag to the user)

Aria is currently a **Render Background Worker**, which cannot receive inbound
HTTP. The sidecar needs a host that (a) accepts HTTP and (b) does not sleep on
idle (a sleeping host kills Aria's polling loop). Options, best first:

1. **Move Aria to Railway** (recommended) — matches Siren, no idle sleep,
   stable public URL. Add a `railway.toml` (`startCommand = "python -m
   src.main"` or Aria's existing entrypoint, `healthcheckPath = "/health"`),
   set all existing env vars plus `SIREN_API_KEY` (same value as Siren's) and
   `SIREN_BASE_URL=https://siren-production-8df8.up.railway.app`. Railway sets
   `PORT`.
2. **Render Web Service** (not Worker) on a paid instance — free web services
   sleep and would stop the bot. Health check `/health`.

Either way: the public URL goes into Siren's `agent_registry.base_url` for
aria (Task 5). `SIREN_API_KEY` must be identical on both sides. Confirm the
host choice with the user before deploying — it changes Aria's hosting.

## Out of scope (do not build)

- `propose_wiki_update`/research-into-existing-page — Phase B drafts new pages
  only (`type: "create"`, which falls back to update if the slug exists).
- Persisting jobs across restarts (they're intentionally ephemeral).
- MCP migration of the contract (a separate, later roadmap item).
- Any change to Aria's conversational behavior, memory, or scheduler.
- Multi-instance safety / job claim-locks — single instance assumed. If Aria
  is ever scaled out, revisit (the sidecar's in-memory jobs are per-process).
