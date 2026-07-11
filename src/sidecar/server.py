"""Aria's sidecar HTTP server — the Siren federation contract.

Runs in a daemon thread alongside Aria's Telegram polling (see run.py).
Endpoints: GET /health, POST /invoke (X-Siren-Key auth), GET /jobs/{id}.
"""

import os
import threading

import httpx
from fastapi import FastAPI, Header, HTTPException

from src.config import cfg
from src.services import dashboard as dashboard_svc
from src.sidecar import jobs as jobstore
from src.sidecar.research import run_research_and_draft

app = FastAPI(title="Aria sidecar")


def _tg_send(text: str) -> None:
    """Notify the user in Aria's own voice, via her bot token — never Siren's."""
    try:
        httpx.post(
            f"https://api.telegram.org/bot{cfg.telegram_token}/sendMessage",
            json={"chat_id": cfg.allowed_user_id, "text": text},
            timeout=20.0,
        )
    except Exception:
        pass  # notification is best-effort


def _log_to_siren(event_type: str, summary: str, payload: dict) -> None:
    """Federation oversight: tell Siren what happened, so she can see it in
    get_agent_events even though she never touched the wiki herself."""
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
            f"\U0001F50E I researched \"{topic}\" and drafted a {words}-word page, "
            f"\"{edit['title']}\".\n\n/wiki_approve {job_id}  to save\n"
            f"/wiki_reject {job_id}  to discard"
        )
        _log_to_siren(
            "research_drafted",
            f"Drafted wiki page '{edit['title']}' from research on {topic}",
            {"job_id": job_id, "slug": edit["slug"]},
        )
    except Exception as e:
        jobstore.finish_job(job_id, error=str(e))
        _tg_send(f"⚠️ My research on \"{topic}\" hit a snag: {e}")


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

    if tool == "research_and_draft":
        if not (args.get("topic") or "").strip():
            raise HTTPException(status_code=400, detail="topic is required")
        job_id = jobstore.create_job(tool, args)
        threading.Thread(target=_run_research_job, args=(job_id, args), daemon=True).start()
        return {"job_id": job_id}

    raise HTTPException(status_code=400, detail=f"unknown tool: {tool}")


@app.get("/jobs/{job_id}")
def get_job(job_id: str, x_siren_key: str = Header(default="")):
    _auth(x_siren_key)
    job = jobstore.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="no such job")
    return job
