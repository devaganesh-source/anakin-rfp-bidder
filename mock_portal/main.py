"""Local procurement mock; start with ``python mock-portal/main.py``.

MOCK_PORTAL_HOST defaults to 127.0.0.1; MOCK_PORTAL_PORT defaults to 8000.
The script loads the project's .env. Point the browser agent's MOCK_PORTAL_URL
at this server. All forms and assets are local; no external services are called.

POST /bids starts a draft. GET/POST /bids/{id}/security, /tech-specs, and
/pricing display/save answers. GET /bids/{id}/submit only reviews them.
Only POST /api/bids/{id}/approve unlocks POST /bids/{id}/submit. Both use the
review form's token and revision; approval also requires confirm=yes.
GET /api/bids/{id} exposes state for polling. Automation should stop at review
until a human approves. This trusted local demo has no user authentication.

Run one worker: drafts and approval Events live in memory and reset on restart.
"""

import asyncio
import os
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from secrets import compare_digest, token_urlsafe
from urllib.parse import parse_qs
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware


# Disabling the built-in documentation pages avoids their external CDN assets.
app = FastAPI(title="Procurement Mock Portal", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
SECTIONS = {
    "security": ("Security", "security", "Describe your security capabilities."),
    "tech-specs": ("Tech Specs", "tech_specs", "Describe the proposed technical solution."),
    "pricing": ("Pricing", "pricing", "Describe your pricing model and commercial terms."),
}
MAX_ANSWER_LENGTH = 10_000
MAX_FORM_BYTES = 128_000


@dataclass
class Bid:
    id: str = field(default_factory=lambda: uuid4().hex)
    token: str = field(default_factory=lambda: token_urlsafe(32))
    answers: dict[str, str] = field(default_factory=dict)
    revision: int = 0
    approval: asyncio.Event = field(default_factory=asyncio.Event)
    submitted: bool = False


bids: dict[str, Bid] = {}


def portal_error(status_code: int, code: str, message: str) -> HTTPException:
    """Return a stable JSON error envelope for automation and human clients."""
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


def get_bid(bid_id: str) -> Bid:
    bid = bids.get(bid_id)
    if bid is None:
        raise portal_error(
            404,
            "BID_NOT_READY",
            "The bid session is not staged or does not exist. Start a new bid from the home page.",
        )
    return bid


def get_section(page: str) -> tuple[str, str, str]:
    if page not in SECTIONS:
        raise HTTPException(404, "Unknown RFP section.")
    return SECTIONS[page]


def hidden_fields(bid: Bid) -> str:
    return (
        f'<input type="hidden" name="token" value="{bid.token}">'
        f'<input type="hidden" name="revision" value="{bid.revision}">'
    )


def document(title: str, content: str, bid: Bid | None = None) -> HTMLResponse:
    nav = (
        '<a class="rounded-lg px-3 py-2 text-sm font-medium text-slate-300 transition '
        'hover:bg-slate-800 hover:text-white" href="/">Home</a>'
    )
    if bid:
        for slug, label in [(s, data[0]) for s, data in SECTIONS.items()] + [
            ("submit", "Submit review")
        ]:
            nav += (
                f'<a class="rounded-lg px-3 py-2 text-sm font-medium text-slate-300 '
                f'transition hover:bg-slate-800 hover:text-white" href="/bids/{bid.id}/{slug}">{label}</a>'
            )
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.tailwindcss.com"></script>
<title>{escape(title)} | Procurement Mock Portal</title>
<style>
body {{ min-height: 100vh; font: 16px/1.5 system-ui, sans-serif; margin: 0;
        padding: 0 1rem; background: #020617; color: #f8fafc; }}
header {{ max-width: 1100px; margin: 0 auto; padding: 1.25rem 0; border-bottom: 1px solid #1e293b; }}
header p {{ color: #64748b; font-size: .8rem; }}
main {{ max-width: 1100px; margin: 3rem auto; padding: 2rem; background: #0f172a;
        border: 1px solid #1e293b; border-radius: .75rem; box-shadow: 0 10px 25px rgba(0,0,0,.2); }}
nav {{ display: flex; flex-wrap: wrap; gap: .25rem; margin-top: 1rem; }}
nav a {{ display: inline-block; color: #cbd5e1; text-decoration: none; }}
nav a:hover {{ color: #fff; background: #1e293b; }}
textarea {{ display: block; width: 100%; box-sizing: border-box; margin: .5rem 0 1rem;
            padding: .75rem; font: inherit; background: #020617; color: #e2e8f0;
            border: 1px solid #334155; border-radius: .5rem; }}
button {{ padding: .75rem 1.25rem; cursor: pointer; font: inherit; border-radius: .5rem; }}
button:disabled {{ cursor: not-allowed; opacity: .55; }}
form {{ margin: 1rem 0; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font: inherit; }}
code {{ overflow-wrap: anywhere; }}
</style></head><body class="min-h-screen bg-slate-950 text-slate-100 antialiased"><header><p>Procurement Mock Portal · Local testing only</p>
<nav class="flex flex-wrap items-center gap-1" aria-label="RFP sections">{nav}</nav></header>
<main class="rounded-xl border border-slate-800 bg-slate-900 shadow-lg"><h1 class="text-3xl font-semibold tracking-tight text-white">{escape(title)}</h1>{content}</main></body></html>""",
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; script-src 'self' https://cdn.tailwindcss.com 'unsafe-inline'; "
            "style-src 'unsafe-inline'; "
            "form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
        },
    )


async def read_form(request: Request) -> dict[str, str]:
    # Body reads yield so other requests can run; no bid state changes here.
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip()
    if media_type != "application/x-www-form-urlencoded":
        raise HTTPException(415, "Use application/x-www-form-urlencoded form data.")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_FORM_BYTES:
            raise HTTPException(413, "Form is too large.")
    try:
        fields = parse_qs(
            body.decode("utf-8"), keep_blank_values=True, max_num_fields=8,
            encoding="utf-8", errors="strict",
        )
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(422, "Invalid form encoding or too many fields.") from exc
    if any(len(values) != 1 for values in fields.values()):
        raise HTTPException(422, "Duplicate form fields are not allowed.")
    return {name: values[0] for name, values in fields.items()}


def validate_action(bid: Bid, form: dict[str, str]) -> None:
    if not compare_digest(form.get("token", "").encode("utf-8"), bid.token.encode("utf-8")):
        raise HTTPException(403, "Invalid form token. Reload the page.")
    if bid.submitted:
        raise HTTPException(409, "This mock bid is already submitted and cannot be changed.")
    if form.get("revision") != str(bid.revision):
        raise HTTPException(409, "Answers changed. Reload and review the current revision.")


def require_complete(bid: Bid) -> None:
    if any(not bid.answers.get(title, "").strip() for title, _, _ in SECTIONS.values()):
        raise HTTPException(422, "Complete Security, Tech Specs, and Pricing first.")


# Handlers share one event loop. After reading a form, validation and mutation
# contain no awaits, so they are atomic within the required single worker.
# The Event gates final submission; blocked POSTs fail immediately instead of
# waiting in the background and submitting later when approval arrives.
@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    return document(
        "RFP submission",
        '<p class="max-w-2xl text-base leading-7 text-slate-300">Complete Security, Tech Specs, and Pricing, then review your answers before the human approval gate.</p>'
        '<form method="post" action="/bids">'
        '<button id="start_bid" class="mt-6 inline-flex items-center justify-center rounded-lg bg-indigo-300 px-5 py-3 text-sm font-bold text-indigo-950 shadow-[0_0_28px_rgba(129,140,248,.2)] transition hover:bg-indigo-200" type="submit">Start new bid</button></form>',
    )


@app.post("/bids")
async def create_bid() -> RedirectResponse:
    bid = Bid()
    bids[bid.id] = bid
    return RedirectResponse(f"/bids/{bid.id}/security", status_code=303)


@app.get("/api/bids/{bid_id}")
async def bid_state(bid_id: str) -> dict:
    bid = get_bid(bid_id)
    status = "submitted" if bid.submitted else "approved" if bid.approval.is_set() else "draft"
    return {
        "id": bid.id, "status": status, "revision": bid.revision,
        "answers": dict(bid.answers), "approved": bid.approval.is_set(),
        "submitted": bid.submitted,
    }


@app.get("/bids/{bid_id}/submit", response_class=HTMLResponse)
async def review(bid_id: str) -> HTMLResponse:
    bid = get_bid(bid_id)
    content = "".join(
        f'<section class="mb-4 rounded-xl border border-slate-800 bg-slate-950/60 p-5 last:mb-0 sm:p-6"><div class="mb-3 flex items-center justify-between gap-3"><h2 class="text-lg font-semibold text-white">{title}</h2><span class="rounded-md border border-slate-800 bg-slate-900 px-2 py-1 text-[10px] font-bold uppercase tracking-[0.16em] text-slate-500">Review</span></div><pre class="text-sm leading-7 text-slate-300" id="review_{input_id}">'
        f'{escape(bid.answers.get(title, "Not provided"))}</pre></section>'
        for title, input_id, _ in SECTIONS.values()
    )
    if bid.submitted:
        content += '<p class="mt-6 rounded-lg border border-emerald-300/25 bg-emerald-300/[0.08] px-4 py-3 text-sm font-semibold text-emerald-200" id="submission_status" role="status">Mock bid submitted.</p>'
    else:
        ready = all(bid.answers.get(title) for title, _, _ in SECTIONS.values())
        locked = "" if bid.approval.is_set() else " disabled"
        status = "Approved for submission." if bid.approval.is_set() else "Locked: human approval required."
        status_class = "border-emerald-300/25 bg-emerald-300/[0.08] text-emerald-200" if bid.approval.is_set() else "border-amber-300/30 bg-amber-300/[0.08] text-amber-100 shadow-[0_0_24px_rgba(251,191,36,.08)]"
        content += f'<p class="mt-6 flex items-center gap-2 rounded-lg border px-4 py-3 text-sm font-semibold {status_class}" id="submission_status" role="status"><span class="h-2 w-2 rounded-full bg-current shadow-[0_0_10px_currentColor]"></span>{status}</p>'
        if ready and not bid.approval.is_set():
            content += (
                '<div class="mt-8 border-t border-slate-800 pt-6"><h2 class="text-xl font-semibold text-white">Human approval</h2><p class="mt-2 text-sm leading-7 text-slate-400">Review the answers above before approving. '
                'Browser automation must pause here for a human reviewer.</p>'
                f'<form method="post" action="/api/bids/{bid.id}/approve">{hidden_fields(bid)}'
                '<label class="mt-5 flex items-start gap-3 text-sm text-slate-300"><input class="mt-1 h-4 w-4 accent-indigo-400" id="confirm_approval" type="checkbox" name="confirm" '
                'value="yes" required> <span>I have reviewed and approve these answers.</span></label>'
                '<button class="mt-5 inline-flex items-center justify-center rounded-lg bg-indigo-300 px-5 py-3 text-sm font-bold text-indigo-950 shadow-[0_0_28px_rgba(129,140,248,.2)] transition hover:bg-indigo-200" id="approve_bid" type="submit">Approve reviewed answers</button></form></div>'
            )
        elif not ready:
            content += '<p class="mt-6 text-sm text-slate-400">Complete every section before requesting approval.</p>'
        content += (
            f'<form class="mt-8 border-t border-slate-800 pt-6" method="post" action="/bids/{bid.id}/submit">{hidden_fields(bid)}'
            f'<button class="inline-flex items-center justify-center rounded-lg border border-slate-700 bg-slate-800 px-5 py-3 text-sm font-semibold text-slate-200 transition hover:border-slate-600 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50" id="submit_bid" type="submit"{locked}>Submit mock bid</button></form>'
        )
    return document("Submit review", content, bid)


@app.post("/api/bids/{bid_id}/approve")
async def approve_bid(bid_id: str, request: Request) -> RedirectResponse:
    form = await read_form(request)
    bid = get_bid(bid_id)
    validate_action(bid, form)
    require_complete(bid)
    if form.get("confirm") != "yes":
        raise HTTPException(422, "Explicit human confirmation is required.")
    if bid.approval.is_set():
        raise portal_error(
            409,
            "BID_ALREADY_APPROVED",
            "This bid has already been approved; duplicate approval is not accepted.",
        )
    bid.approval.set()  # This endpoint is the only place approval can be granted.
    return RedirectResponse(f"/bids/{bid.id}/submit", status_code=303)


@app.post("/bids/{bid_id}/submit")
async def submit_bid(bid_id: str, request: Request) -> RedirectResponse:
    form = await read_form(request)
    bid = get_bid(bid_id)
    validate_action(bid, form)
    if not bid.approval.is_set():
        raise HTTPException(403, "Submission locked. A human must approve this revision first.")
    require_complete(bid)
    bid.submitted = True
    bid.approval.clear()  # Consume approval; replaying the POST cannot submit again.
    return RedirectResponse(f"/bids/{bid.id}/submit", status_code=303)


@app.get("/bids/{bid_id}/{page}", response_class=HTMLResponse)
async def section_page(bid_id: str, page: str) -> HTMLResponse:
    title, input_id, prompt = get_section(page)
    bid = get_bid(bid_id)
    readonly = " readonly" if bid.submitted else ""
    disabled = " disabled" if bid.submitted else ""
    content = (
        f'<form method="post" action="/bids/{bid.id}/{page}">{hidden_fields(bid)}'
        f'<label class="block text-sm font-semibold text-white" for="{input_id}">{title}</label><p class="mt-2 text-sm leading-7 text-slate-400" id="section_hint">{prompt}</p>'
        f'<textarea class="mt-5 block w-full rounded-lg border border-slate-700 bg-slate-950/80 p-4 text-sm leading-7 text-slate-200 outline-none transition placeholder:text-slate-600 focus:border-indigo-300/60 focus:ring-2 focus:ring-indigo-300/10 disabled:cursor-not-allowed disabled:opacity-60" id="{input_id}" name="{title}" rows="10" '
        f'aria-describedby="section_hint" maxlength="{MAX_ANSWER_LENGTH}" placeholder="Draft your response here..." required{readonly}>'
        f'{escape(bid.answers.get(title, ""))}</textarea>'
        f'<button class="mt-5 inline-flex items-center justify-center rounded-lg bg-indigo-300 px-5 py-3 text-sm font-bold text-indigo-950 shadow-[0_0_28px_rgba(129,140,248,.2)] transition hover:bg-indigo-200 disabled:cursor-not-allowed disabled:opacity-50" id="save_continue" type="submit"{disabled}>Save and continue</button></form>'
    )
    return document(title, content, bid)


@app.post("/bids/{bid_id}/{page}")
async def save_section(bid_id: str, page: str, request: Request) -> RedirectResponse:
    title, _, _ = get_section(page)
    form = await read_form(request)
    bid = get_bid(bid_id)
    validate_action(bid, form)
    answer = form.get(title, "").strip()
    if not answer or len(answer) > MAX_ANSWER_LENGTH:
        raise HTTPException(422, f"{title} requires 1 to {MAX_ANSWER_LENGTH} characters.")
    bid.answers[title] = answer
    bid.revision += 1
    bid.approval.clear()  # Saving any section requires a fresh human review.
    pages = [*SECTIONS, "submit"]
    return RedirectResponse(f"/bids/{bid.id}/{pages[pages.index(page) + 1]}", status_code=303)


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    port = int(os.environ.get("MOCK_PORTAL_PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("MOCK_PORTAL_PORT must be between 1 and 65535.")
    uvicorn.run(app, host=os.environ.get("MOCK_PORTAL_HOST", "127.0.0.1"), port=port, workers=1)
