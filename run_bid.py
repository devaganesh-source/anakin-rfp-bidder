"""Run the sample RFP through retrieval, reasoning, staging, and HITL review.

The script starts the FastAPI HITL app in the same process that owns the
in-memory manager. That keeps the printed dashboard session valid until the
process is stopped. It never approves the bid; the dashboard's explicit
approval call is the only path that releases final submission.
"""

import argparse
import asyncio
import datetime
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import uvicorn
from dotenv import load_dotenv

try:
    import requests
except ModuleNotFoundError:  # Keep the script runnable in the minimal project venv.
    requests = httpx

from actuation.hitl_manager import BidSnapshot, app as hitl_app, manager
from crawler import RFPCrawler
from reasoner.groq_reasoner import generate_section_answer
from reasoner.vector_store import build_index, search_index
from start_backend import ensure_port_available


COMPLIANCE_MARKDOWN_URL = "https://vercel.com/docs/security/compliance.md"
STATUS_SUMMARY_URL = "https://www.vercel-status.com/api/v2/summary.json"
LIVE_FETCH_TIMEOUT_SECONDS = 30
SUBMIT_TIMEOUT_SECONDS = 30


def fetch_live_evidence() -> dict:
    """Fetch uncached compliance and operational evidence from live endpoints."""
    headers = {"Cache-Control": "no-cache"}
    redirect_option = (
        {"follow_redirects": False}
        if requests is httpx
        else {"allow_redirects": False}
    )
    request_error = (
        httpx.HTTPError
        if requests is httpx
        else requests.exceptions.RequestException
    )

    print("[LIVE FETCH] Fetching compliance.md live...")
    request_started = time.perf_counter()
    try:
        compliance_response = requests.get(
            COMPLIANCE_MARKDOWN_URL,
            headers=headers,
            timeout=LIVE_FETCH_TIMEOUT_SECONDS,
            **redirect_option,
        )
        req1_time = round((time.perf_counter() - request_started) * 1000)
        print(f"HTTP {compliance_response.status_code} · text/markdown · {req1_time}ms")
        compliance_response.raise_for_status()
    except request_error as exc:
        raise RuntimeError("The live Vercel compliance fetch failed.") from exc

    compliance_markdown = compliance_response.text
    if not isinstance(compliance_markdown, str) or not compliance_markdown.strip():
        raise RuntimeError("The live Vercel compliance response contained no Markdown.")
    current_utc_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"Fetched at: {current_utc_time}")
    print()

    print("[LIVE FETCH] Fetching summary.json live...")
    request_started = time.perf_counter()
    try:
        status_response = requests.get(
            STATUS_SUMMARY_URL,
            headers=headers,
            timeout=LIVE_FETCH_TIMEOUT_SECONDS,
            **redirect_option,
        )
        req2_time = round((time.perf_counter() - request_started) * 1000)
        print(f"HTTP {status_response.status_code} · application/json · {req2_time}ms")
        status_response.raise_for_status()
        status_payload = status_response.json()
    except request_error as exc:
        raise RuntimeError("The live Vercel status fetch failed.") from exc
    except ValueError as exc:
        raise RuntimeError("The live Vercel status response was not valid JSON.") from exc

    if not isinstance(status_payload, dict):
        raise RuntimeError("The live Vercel status response had an unexpected structure.")
    page = status_payload.get("page")
    status = status_payload.get("status")
    incidents = status_payload.get("incidents")
    if not isinstance(page, dict) or not isinstance(status, dict) or not isinstance(incidents, list):
        raise RuntimeError("The live Vercel status response omitted required fields.")
    json_updated_at = page.get("updated_at")
    json_status_desc = status.get("description")
    if not isinstance(json_updated_at, str) or not isinstance(json_status_desc, str):
        raise RuntimeError("The live Vercel status response contained invalid fields.")
    incident_count = len(incidents)

    print(f"Publisher status updated: {json_updated_at}")
    print(f"Overall status: {json_status_desc}")
    print(f"Active incidents: {incident_count}")
    print()
    print("[AGENT] Generating RFP compliance matrix using live evidence...")

    live_context = (
        f"{compliance_markdown.strip()}\n\n"
        "## Live Vercel Operational Status\n"
        f"- Publisher status updated: {json_updated_at}\n"
        f"- Overall status: {json_status_desc}\n"
        f"- Active incidents: {incident_count}"
    )
    return {
        "fetched_at": current_utc_time,
        "compliance": {
            "url": COMPLIANCE_MARKDOWN_URL,
            "status_code": compliance_response.status_code,
            "response_time_ms": req1_time,
            "markdown": compliance_markdown,
        },
        "status": {
            "url": STATUS_SUMMARY_URL,
            "status_code": status_response.status_code,
            "response_time_ms": req2_time,
            "updated_at": json_updated_at,
            "description": json_status_desc,
            "incident_count": incident_count,
        },
        "live_context": live_context,
    }


def _required_environment() -> tuple[str, str, str, str]:
    """Return required credentials and configured local workflow URLs."""
    values = {
        name: os.environ.get(name, "").strip()
        for name in ("GROQ_API_KEY", "ANAKIN_API_KEY", "MOCK_PORTAL_URL", "RFP_SOURCE_URL")
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"Set {', '.join(missing)} in .env or the environment.")
    return (
        values["GROQ_API_KEY"],
        values["ANAKIN_API_KEY"],
        values["MOCK_PORTAL_URL"],
        values["RFP_SOURCE_URL"],
    )


def _hitl_port() -> int:
    """Parse and validate the local HITL API port before external work starts."""
    try:
        port = int(os.environ.get("HITL_PORT", "8001"))
    except ValueError as exc:
        raise RuntimeError("HITL_PORT must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("HITL_PORT must be between 1 and 65535.")
    return port


def _validate_local_service_ports(portal_url: str, hitl_port: int) -> None:
    """Reject a local portal/HITL configuration that would bind one port twice."""
    parsed = urlsplit(portal_url)
    portal_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if portal_port == hitl_port:
        raise RuntimeError(
            f"MOCK_PORTAL_URL uses port {portal_port}, which conflicts with HITL_PORT. "
            "Run the mock portal on its own local port (8000 by default)."
        )


async def _stage_bid_via_api(
    portal_url: str,
    answers: list[dict[str, str]],
) -> dict[str, Any]:
    """Create the local draft through JSON without probing an HTML form."""
    parsed_portal = urlsplit(portal_url)
    loopback_hosts = {"localhost", "127.0.0.1", "::1"}
    if (
        parsed_portal.scheme not in {"http", "https"}
        or (parsed_portal.hostname or "").lower() not in loopback_hosts
        or parsed_portal.username is not None
        or parsed_portal.password is not None
        or parsed_portal.path not in {"", "/"}
        or parsed_portal.query
        or parsed_portal.fragment
    ):
        raise ValueError("MOCK_PORTAL_URL must be a loopback HTTP(S) root URL.")

    answer_map = {item["section"]: item["answer"] for item in answers}
    create_url = f"{portal_url.rstrip('/')}/api/bids"
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(SUBMIT_TIMEOUT_SECONDS, connect=10),
            follow_redirects=False,
        ) as client:
            response = await client.post(create_url, json={"answers": answer_map})
    except httpx.RequestError as exc:
        raise RuntimeError("The local mock bid API could not stage the draft.") from exc

    if response.status_code != 201:
        raise RuntimeError(
            f"The local mock bid API returned HTTP {response.status_code} while staging."
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("The local mock bid API returned an invalid staging response.") from exc
    portal_bid_id = data.get("id") if isinstance(data, dict) else None
    if not isinstance(portal_bid_id, str) or len(portal_bid_id) != 32:
        raise RuntimeError("The local mock bid API returned an invalid bid ID.")

    return {
        "bid_id": portal_bid_id,
        "review_url": f"{portal_url.rstrip('/')}/bids/{portal_bid_id}/submit",
        "status": "awaiting_approval",
        "submitted": False,
    }


def _make_submit_action(staged_bid: dict[str, Any], _answers: list[dict[str, str]]):
    """Create the gated callback that verifies the frontend-owned submission."""
    portal_bid_id = staged_bid["bid_id"]
    review_parts = urlsplit(staged_bid["review_url"])
    portal_url = f"{review_parts.scheme}://{review_parts.netloc}"

    async def submit_action() -> bool:
        """Verify once, after HITLManager observes explicit human approval."""
        verify_url = f"{portal_url.rstrip('/')}/api/bids/{portal_bid_id}"
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(SUBMIT_TIMEOUT_SECONDS, connect=10),
                follow_redirects=False,
            ) as client:
                response = await client.get(verify_url)
        except httpx.RequestError as exc:
            raise RuntimeError("Submission verification request failed.") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("The local mock bid API returned an invalid status response.") from exc
        if response.status_code != 200 or not isinstance(data, dict) or data.get("submitted") is not True:
            raise RuntimeError("The Next.js submission was not confirmed by the local mock bid API.")
        return True

    return submit_action


async def process_section_concurrently(
    title: str,
    content: str,
    faiss_index: Any,
    chunks: list[str],
    semaphore: asyncio.Semaphore,
) -> dict[str, str]:
    """Retrieve evidence and reason over one section within the shared limit."""
    retrieved_context = await asyncio.to_thread(
        search_index,
        f"{title}\n{content}",
        faiss_index,
        chunks,
    )

    # Limit only the external Groq requests; local evidence retrieval can still
    # run concurrently while final submission remains blocked by HITL approval.
    async with semaphore:
        return await generate_section_answer(title, content, retrieved_context)


async def run_pipeline() -> BidSnapshot:
    """Build the proposal, stage it, register HITL state, and return its session."""
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    _, _, portal_url, rfp_source_url = _required_environment()

    # 1. CRAWL THE RFP
    crawler = RFPCrawler(rfp_source_url)
    RFP_SECTIONS = await crawler.crawl()
    print(f"DEBUG: Scraped {len(RFP_SECTIONS)} sections from RFP.")

    for idx, (title, content) in enumerate(RFP_SECTIONS.items()):
        print(f"[{idx}] {title}: {content[:60]}...")

    # 2. INGEST LIVE VENDOR KNOWLEDGE
    live_evidence = await asyncio.to_thread(fetch_live_evidence)
    live_context = live_evidence["live_context"]
    # The blocking fetches run in one worker thread so the event loop stays
    # responsive; final submission remains blocked by the HITL approval Event.
    chunks = [live_context]
    faiss_index = build_index(chunks)

    # 3. GENERATE AI ANSWERS WITH BOUNDED CONCURRENCY
    items = list(RFP_SECTIONS.items())
    semaphore = asyncio.Semaphore(1)
    tasks = [
        process_section_concurrently(
            title,
            content,
            faiss_index,
            chunks,
            semaphore,
        )
        for title, content in items
    ]
    print("[AGENT] Dispatching all RFP sections concurrently to the Groq reasoning engine...")
    results = await asyncio.gather(*tasks)
    answers = list(results)

    if not answers:
        print("\n[!] DRY RUN ACTIVE: No API calls made. Exiting before JSON staging.")
        return None

    print(f"DEBUG: Completed AI reasoning for {len(answers)} of {len(items)} sections.")

    proposal_answers = [dict(a) for a in answers]

    # 4. STAGE AND REGISTER
    print("DEBUG: Staging the finalized draft through the local mock JSON API...")
    staged = await _stage_bid_via_api(portal_url, proposal_answers)
    submit_action = _make_submit_action(staged, proposal_answers)
    
    # Pass the full dynamic requirement payload to the Next.js review UI.
    return await manager.register_staged_bid(
        staged,
        RFP_SECTIONS,        # Dynamic, unadulterated requirements
        proposal_answers,    # Dynamic, unadulterated answers
        submit_action,
    )


async def serve_app(serve: bool = True) -> None:
    """Run the pipeline and optionally keep the HITL API serving the session."""
    # Staging completes first; Uvicorn then owns the same event loop as the
    # in-memory HITL manager so polling and approval tasks remain available.
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    hitl_host = os.environ.get("HITL_HOST", "127.0.0.1")
    hitl_port = _hitl_port()
    _, _, portal_url, _ = _required_environment()
    _validate_local_service_ports(portal_url, hitl_port)
    if serve:
        ensure_port_available(hitl_host, hitl_port)

    snapshot = await run_pipeline()
    
    # Handle graceful exit if running in dry-run mode
    if not snapshot:
        return

    dashboard_url = os.environ.get("DASHBOARD_URL", "http://127.0.0.1:3000").rstrip("/")
    review_url = f"{dashboard_url}/?bid={snapshot.id}"
    print(f"Dashboard review URL: {review_url}")
    print(f"Session ID: {snapshot.id}")
    print(f"Status: {snapshot.status}")

    if not serve:
        return

    ensure_port_available(hitl_host, hitl_port)
    config = uvicorn.Config(hitl_app, host=hitl_host, port=hitl_port, log_level="info")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    except OSError as exc:
        raise RuntimeError(f"Could not bind HITL API to {hitl_host}:{hitl_port}.") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Register and print the session without keeping the HITL API alive.",
    )
    args = parser.parse_args()
    try:
        asyncio.run(serve_app(serve=not args.no_server))
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"run_bid failed: {exc}") from exc
