"""Run the sample RFP through retrieval, reasoning, staging, and HITL review.

The script starts the FastAPI HITL app in the same process that owns the
in-memory manager. That keeps the printed dashboard session valid until the
process is stopped. It never approves the bid; the dashboard's explicit
approval call is the only path that releases final submission.
"""

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import uvicorn
from dotenv import load_dotenv

from actuation.hitl_manager import BidSnapshot, app as hitl_app, manager
from crawler import RFPCrawler
from reasoner.groq_reasoner import process_all_sections
from reasoner.vector_store import build_index, load_and_chunk_docs
from start_backend import ensure_port_available


SUBMIT_TIMEOUT_SECONDS = 30
GROQ_BATCH_SIZE = 1
GROQ_429_MAX_ATTEMPTS = 3
GROQ_429_INITIAL_BACKOFF_SECONDS = 15


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

    # 2. LOAD KNOWLEDGE BASE
    chunks = load_and_chunk_docs(project_root / "sample-data")
    faiss_index = build_index(chunks)

    # 3. GENERATE BATCHED AI ANSWERS (WITH SELF-HEALING RATE LIMIT LOGIC)
    print("DEBUG: AI Reasoning Phase (Groq - Throttled Batches)...")
    answers = []
    rate_limit_errors = 0
    items = list(RFP_SECTIONS.items())
    total_batches = (len(items) + GROQ_BATCH_SIZE - 1) // GROQ_BATCH_SIZE
    
    for i in range(0, len(items), GROQ_BATCH_SIZE):
        batch_number = i // GROQ_BATCH_SIZE + 1
        
        if batch_number > 1:
            print("   Waiting 6s before the next Groq batch...")
            await asyncio.sleep(6)

        batch = dict(items[i:i + GROQ_BATCH_SIZE])
        print(f"-> Processing batch {batch_number} of {total_batches}...")
        
        for attempt in range(GROQ_429_MAX_ATTEMPTS):
            try:
                batch_answers = await process_all_sections(batch, faiss_index, chunks)
                answers.extend(batch_answers)
                print(f"<- Completed batch {batch_number} of {total_batches}.")
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < GROQ_429_MAX_ATTEMPTS - 1:
                    rate_limit_errors += 1
                    backoff_seconds = GROQ_429_INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    print(
                        "   [!] Groq rate limit reached. "
                        f"Backing off for {backoff_seconds}s before attempt "
                        f"{attempt + 2}/{GROQ_429_MAX_ATTEMPTS}..."
                    )
                    await asyncio.sleep(backoff_seconds)
                else:
                    raise

    if not answers:
        print("\n[!] DRY RUN ACTIVE: No API calls made. Exiting before JSON staging.")
        return None

    print(f"DEBUG: Completed AI reasoning for {len(answers)} of {len(items)} sections.")
    print(f"DEBUG: Groq HTTP 429 errors observed: {rate_limit_errors}.")
    
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
