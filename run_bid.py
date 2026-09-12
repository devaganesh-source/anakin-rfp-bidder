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
import re
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx
import uvicorn
from dotenv import load_dotenv

try:
    import requests
except ModuleNotFoundError:  # Keep the script runnable in the minimal project venv.
    requests = httpx

from actuation.anakin_client import crawl_rfp, stage_bid
from actuation.hitl_manager import BidSnapshot, app as hitl_app, manager
from reasoner.groq_reasoner import process_all_sections
from reasoner.vector_store import build_index
from start_backend import ensure_port_available


COMPLIANCE_MARKDOWN_URL = "https://vercel.com/docs/security/compliance.md"
STATUS_SUMMARY_URL = "https://www.vercel-status.com/api/v2/summary.json"
LIVE_FETCH_TIMEOUT_SECONDS = 30
SUBMIT_TIMEOUT_SECONDS = 30


def _normalize_anakin_rfp_sections(
    crawled_sections: Mapping[str, str],
) -> tuple[str, dict[str, str]]:
    """Expand Anakin Markdown headings into individual labelled requirements."""
    if not crawled_sections:
        raise RuntimeError("Anakin returned no RFP content.")
    document_title = "Untitled RFP"
    requirements: dict[str, str] = {}
    labelled_requirement = re.compile(
        r"(?ms)^\s*\*\*(?P<title>[^*\n]+?):\*\*\s*(?P<body>.*?)"
        r"(?=^\s*\*\*[^*\n]+?:\*\*|\Z)"
    )

    def clean(value: str) -> str:
        unescaped = re.sub(r"\\([\\`*_{}\[\]()#+.!-])", r"\1", value)
        return " ".join(unescaped.split())

    for heading, body in crawled_sections.items():
        clean_heading = clean(heading)
        if not body.strip():
            if document_title == "Untitled RFP" and clean_heading:
                document_title = clean_heading
            continue
        matches = list(labelled_requirement.finditer(body))
        if matches:
            for match in matches:
                title = clean(match.group("title"))
                requirement = clean(match.group("body"))
                if not title or not requirement or title in requirements:
                    raise RuntimeError("Anakin returned duplicate or empty RFP requirements.")
                requirements[title] = requirement
        elif clean_heading:
            requirement = clean(body)
            if not requirement or clean_heading in requirements:
                raise RuntimeError("Anakin returned duplicate or empty RFP sections.")
            requirements[clean_heading] = requirement

    if not requirements:
        raise RuntimeError("Anakin returned no usable RFP requirements.")
    return document_title, requirements


def _chunk_live_markdown(markdown: str) -> list[str]:
    """Split public Markdown into source-labelled sections for focused retrieval."""
    cleaned = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", markdown, count=1, flags=re.DOTALL)
    cleaned = re.sub(
        r"<!-- docsgraph:related -->.*?<!-- /docsgraph:related -->",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    headings = list(re.finditer(r"^(#{1,4})\s+(.+?)\s*$", cleaned, flags=re.MULTILINE))
    chunks: list[str] = []
    for position, heading in enumerate(headings):
        body_start = heading.end()
        body_end = headings[position + 1].start() if position + 1 < len(headings) else len(cleaned)
        body = cleaned[body_start:body_end].strip()
        if not body:
            continue
        title = heading.group(2).strip()
        chunks.append(
            f"Source URL: {COMPLIANCE_MARKDOWN_URL}\n"
            f"Section: {title}\n\n"
            f"{body}"
        )

    if not chunks:
        raise RuntimeError("The live Vercel compliance Markdown had no usable sections.")
    return chunks


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

    live_chunks = _chunk_live_markdown(compliance_markdown)
    live_chunks.append(
        f"Source URL: {STATUS_SUMMARY_URL}\n"
        "Section: Live Vercel Operational Status\n\n"
        f"- Publisher status updated: {json_updated_at}\n"
        f"- Overall status: {json_status_desc}\n"
        f"- Active incidents: {incident_count}"
    )
    print(f"[RAG] Indexed {len(live_chunks)} focused live-evidence chunks.")
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
        "live_chunks": live_chunks,
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


def _make_submit_action(staged_bid: dict[str, Any]):
    """Create the sole backend-owned action that approves and submits one revision."""
    portal_bid_id = staged_bid["bid_id"]
    review_parts = urlsplit(staged_bid["review_url"])
    portal_url = f"{review_parts.scheme}://{review_parts.netloc}"

    async def submit_action(finalized_answers: Mapping[str, str]) -> bool:
        """Submit once after HITLManager records the reviewed answer revision."""
        bid_api_url = f"{portal_url.rstrip('/')}/api/bids/{portal_bid_id}"
        answers = dict(finalized_answers)
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(SUBMIT_TIMEOUT_SECONDS, connect=10),
                follow_redirects=False,
            ) as client:
                approval = await client.post(
                    f"{bid_api_url}/approve",
                    json={
                        "answers": answers,
                        "authorized_representative": True,
                        "ungrounded_items_acknowledged": True,
                    },
                )
                if approval.status_code != 200:
                    raise RuntimeError(
                        f"The mock portal rejected approval with HTTP {approval.status_code}."
                    )
                submission = await client.post(
                    f"{bid_api_url}/submit", json={"answers": answers}
                )
                verification = await client.get(bid_api_url)
        except httpx.RequestError as exc:
            raise RuntimeError("The controlled mock submission request failed.") from exc

        try:
            submission_data = submission.json()
            verification_data = verification.json()
        except ValueError as exc:
            raise RuntimeError("The mock portal returned an invalid submission response.") from exc
        if (
            submission.status_code != 200
            or not isinstance(submission_data, dict)
            or submission_data.get("submitted") is not True
            or verification.status_code != 200
            or not isinstance(verification_data, dict)
            or verification_data.get("submitted") is not True
            or verification_data.get("answers") != answers
        ):
            raise RuntimeError("The controlled mock submission could not be verified.")
        return True

    return submit_action


async def run_pipeline(source_url: str | None = None) -> BidSnapshot | None:
    """Build the proposal, stage it, register HITL state, and return its session."""
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    _, anakin_api_key, portal_url, configured_rfp_url = _required_environment()
    rfp_source_url = source_url.strip() if isinstance(source_url, str) else configured_rfp_url
    if not rfp_source_url:
        raise RuntimeError("Provide an RFP source URL.")

    # 1. CRAWL THE RFP
    print(f"[ANAKIN CRAWL] Reading live RFP: {rfp_source_url}")
    crawled_sections = await crawl_rfp(rfp_source_url, anakin_api_key)
    rfp_title, rfp_sections = _normalize_anakin_rfp_sections(crawled_sections)
    RFP_SECTIONS = rfp_sections
    print(f"DEBUG: Scraped {len(RFP_SECTIONS)} sections from RFP.")

    for idx, (title, content) in enumerate(RFP_SECTIONS.items()):
        print(f"[{idx}] {title}: {content[:60]}...")

    # 2. INGEST LIVE VENDOR KNOWLEDGE
    live_evidence = await asyncio.to_thread(fetch_live_evidence)
    # The blocking fetches run in one worker thread so the event loop stays
    # responsive; final submission remains blocked by the HITL approval Event.
    chunks = live_evidence["live_chunks"]
    faiss_index = build_index(chunks)

    # 3. GENERATE AI ANSWERS IN ONE TOKEN-EFFICIENT BATCH
    items = list(RFP_SECTIONS.items())
    print("[AGENT] Sending one grounded RFP batch to the Groq reasoning engine...")
    answers = await process_all_sections(RFP_SECTIONS, faiss_index, chunks)

    if not answers:
        print("\n[!] DRY RUN ACTIVE: No API calls made. Exiting before JSON staging.")
        return None

    print(f"DEBUG: Completed AI reasoning for {len(answers)} of {len(items)} sections.")

    proposal_answers = [dict(a) for a in answers]

    # 4. STAGE AND REGISTER
    print("[ANAKIN BROWSER] Staging the finalized draft in the controlled mock portal...")
    staged = await stage_bid(portal_url, proposal_answers, anakin_api_key)
    submit_action = _make_submit_action(staged)
    
    # Pass the full dynamic requirement payload to the Next.js review UI.
    return await manager.register_staged_bid(
        staged,
        RFP_SECTIONS,        # Dynamic, unadulterated requirements
        proposal_answers,    # Dynamic, unadulterated answers
        submit_action,
        rfp_title=rfp_title,
        rfp_source_url=rfp_source_url,
        evidence_sources=(COMPLIANCE_MARKDOWN_URL, STATUS_SUMMARY_URL),
        evidence_fetched_at=live_evidence["fetched_at"],
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
