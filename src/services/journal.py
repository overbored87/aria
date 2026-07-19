"""Day Two journal ingest.

Day Two lives in its own Supabase project with RLS bound to `auth.uid()`, so
there's no server-side DB write available to us — entries go through its
`POST /api/journal` route, which holds the service_role key and owns the
schema. We send text and get back an entry id.

Entries are passed through verbatim. Nothing in this module rewrites,
summarises, or reformats what the user said — the words are the artifact.
"""

from __future__ import annotations

import httpx

from src.config import cfg
from src.utils.logger import log

_TIMEOUT = 20.0


def is_configured() -> bool:
    return bool(cfg.journal_api_url and cfg.journal_api_key)


def write_entry(
    content: str,
    title: str | None = None,
    created_at: str | None = None,
    source: str = "siren",
) -> dict:
    """Persist a journal entry to Day Two.

    created_at: optional ISO timestamp, for writing up an earlier day.
    Returns the created entry ({id, created_at, title, ...}).
    Raises RuntimeError on any failure — the caller must not report success
    for an entry that didn't land.
    """
    if not is_configured():
        raise RuntimeError("Journal ingest not configured (JOURNAL_API_URL/KEY)")

    content = (content or "").strip()
    if not content:
        raise RuntimeError("Refusing to write an empty journal entry")

    payload: dict = {"content": content, "source": source}
    if title:
        payload["title"] = title
    if created_at:
        payload["created_at"] = created_at

    try:
        response = httpx.post(
            cfg.journal_api_url,
            json=payload,
            headers={"x-journal-key": cfg.journal_api_key},
            timeout=_TIMEOUT,
        )
    except httpx.RequestError as e:
        # Network-level failure — Siren should retry rather than drop the entry.
        raise RuntimeError(f"Journal API unreachable: {e}") from e

    # Redirects aren't followed on purpose: /login would answer 200 with HTML
    # and look like success. A 3xx here means the endpoint sits behind auth
    # middleware, or the URL is off (trailing slash, www, http vs https).
    if 300 <= response.status_code < 400:
        raise RuntimeError(
            f"Journal API redirected ({response.status_code}) to "
            f"{response.headers.get('location', 'unknown')} — the endpoint is "
            f"behind auth middleware, or JOURNAL_API_URL is wrong"
        )

    if response.status_code not in (200, 201):
        detail = response.text[:200]
        raise RuntimeError(f"Journal API returned {response.status_code}: {detail}")

    entry = response.json()
    log.info(
        f"Journal entry saved: {entry.get('id')} "
        f"({len(content.split())} words, source={source}"
        f"{', deduped' if entry.get('deduped') else ''})"
    )
    return entry
