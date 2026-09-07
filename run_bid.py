"""Run the sample RFP through retrieval, reasoning, staging, and HITL review.

The script starts the FastAPI HITL app in the same process that owns the
in-memory manager. That keeps the printed dashboard session valid until the
process is stopped. It never approves the bid; the dashboard's explicit
approval call is the only path that releases final submission.
"""

import argparse
import asyncio
import html
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import aiohttp
import uvicorn
from dotenv import load_dotenv

from actuation.anakin_client import stage_bid
from actuation.hitl_manager import BidSnapshot, app as hitl_app, manager
from reasoner.groq_reasoner import process_all_sections
from reasoner.vector_store import build_index, load_and_chunk_docs
from start_backend import ensure_port_available


RFP_SECTIONS = {
    "Security": "Describe your encryption, compliance, support, and authentication capabilities.",
    "Tech Specs": "Describe the technical solution and integrations available for this proposal.",
    "Pricing": "Describe the pricing model and commercial terms for this proposal.",
}
SUBMIT_TIMEOUT_SECONDS = 30


def _required_environment() -> tuple[str, str, str]:
    """Return required credentials and the configured local portal URL."""
    values = {
        name: os.environ.get(name, "").strip()
        for name in ("GROQ_API_KEY", "ANAKIN_API_KEY", "MOCK_PORTAL_URL")
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"Set {', '.join(missing)} in .env or the environment.")
    return values["GROQ_API_KEY"], values["ANAKIN_API_KEY"], values["MOCK_PORTAL_URL"]


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


def _hidden_value(document: str, field_name: str) -> str:
    """Extract one hidden form value from the mock portal review page."""
    pattern = re.compile(
        r'<input\b(?=[^>]*\btype=["\']hidden["\'])'
        rf'(?=[^>]*\bname=["\']{re.escape(field_name)}["\'])'
        r'[^>]*\bvalue=["\']([^"\']*)["\']',
        re.IGNORECASE,
    )
    match = pattern.search(document)
    if match is None:
        raise RuntimeError(f"The mock portal review did not contain {field_name}.")
    return html.unescape(match.group(1))


def _make_submit_action(staged_bid: dict[str, Any], answers: list[dict[str, str]]):
    """Create the approval-gated action that submits this exact reviewed draft."""
    review_url = staged_bid["review_url"]
    expected_answers = {item["section"]: item["answer"].strip() for item in answers}
    expected_bid_id = staged_bid["bid_id"]

    async def submit_action() -> bool:
        """Submit once, after HITLManager has observed explicit human approval."""
        timeout = aiohttp.ClientTimeout(total=SUBMIT_TIMEOUT_SECONDS, connect=10)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(review_url, allow_redirects=False) as review:
                    if review.status != 200:
                        raise RuntimeError(f"Mock portal review returned HTTP {review.status}.")
                    document = await review.text()

                for section, answer in expected_answers.items():
                    if html.escape(answer) not in document:
                        raise RuntimeError(f"Reviewed answer for {section} no longer matches.")

                token = _hidden_value(document, "token")
                revision = _hidden_value(document, "revision")
                submit_url = urljoin(review_url, f"/bids/{expected_bid_id}/submit")
                async with session.post(
                    submit_url,
                    data={"token": token, "revision": revision},
                    allow_redirects=False,
                ) as submitted:
                    if submitted.status != 303:
                        raise RuntimeError(f"Mock portal submit returned HTTP {submitted.status}.")
                    location = submitted.headers.get("Location", "")
                    if urljoin(submit_url, location) != review_url:
                        raise RuntimeError("Mock portal redirected outside the reviewed draft.")

                async with session.get(review_url, allow_redirects=False) as final_review:
                    if final_review.status != 200:
                        raise RuntimeError("Could not confirm the mock portal submission.")
                    return "Mock bid submitted." in await final_review.text()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise RuntimeError("The mock portal submit request failed.") from exc

    return submit_action


async def run_pipeline() -> BidSnapshot:
    """Build the proposal, stage it, register HITL state, and return its session."""
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    _, anakin_api_key, portal_url = _required_environment()

    chunks = load_and_chunk_docs(project_root / "sample-data")
    faiss_index = build_index(chunks)

    # Retrieval is performed once, then all Groq section requests run concurrently.
    answers = await process_all_sections(RFP_SECTIONS, faiss_index, chunks)
    proposal_answers = [dict(answer) for answer in answers]

    staged = await stage_bid(portal_url, proposal_answers, anakin_api_key)
    submit_action = _make_submit_action(staged, proposal_answers)
    return await manager.register_staged_bid(
        staged,
        RFP_SECTIONS,
        proposal_answers,
        submit_action,
    )


async def main(serve: bool = True) -> None:
    """Run the pipeline and optionally keep the HITL API serving the session."""
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    hitl_host = os.environ.get("HITL_HOST", "127.0.0.1")
    hitl_port = _hitl_port()
    _, _, portal_url = _required_environment()
    _validate_local_service_ports(portal_url, hitl_port)
    if serve:
        # Preflight removes a stale local Python/Uvicorn listener before the
        # pipeline runs; the final Uvicorn bind remains guarded below as well.
        ensure_port_available(hitl_host, hitl_port)

    snapshot = await run_pipeline()
    dashboard_url = os.environ.get("DASHBOARD_URL", "http://127.0.0.1:3000").rstrip("/")
    review_url = f"{dashboard_url}/?bid={snapshot.id}"
    print(f"Dashboard review URL: {review_url}")
    print(f"Session ID: {snapshot.id}")
    print(f"Status: {snapshot.status}")

    if not serve:
        return

    # The pipeline uses the separate MOCK_PORTAL_URL service; this bind is the
    # HITL review API only, so the two local FastAPI apps never share a port.
    ensure_port_available(hitl_host, hitl_port)
    config = uvicorn.Config(hitl_app, host=hitl_host, port=hitl_port, log_level="info")
    server = uvicorn.Server(config)
    # The API server, approval waiter, and dashboard polling share this event loop.
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
        asyncio.run(main(serve=not args.no_server))
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"run_bid failed: {exc}") from exc
