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
