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
