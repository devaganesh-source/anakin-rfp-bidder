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
import httpx
import uvicorn
from dotenv import load_dotenv

from actuation.anakin_client import stage_bid
from actuation.hitl_manager import BidSnapshot, app as hitl_app, manager
from crawler import RFPCrawler
from reasoner.groq_reasoner import process_all_sections
from reasoner.vector_store import build_index, load_and_chunk_docs
from start_backend import ensure_port_available


SUBMIT_TIMEOUT_SECONDS = 30


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


def _hidden_value(document: str, field_name: str) -> str:
    """Extract one hidden form value from the mock portal review page robustly."""
    tag_pattern = re.compile(rf'<input[^>]*name=["\']{re.escape(field_name)}["\'][^>]*>', re.IGNORECASE)
    tag_match = tag_pattern.search(document)
    
    if not tag_match:
        raise RuntimeError(f"The mock portal review did not contain {field_name}.")
        
    val_pattern = re.compile(r'value=["\']([^"\']*)["\']', re.IGNORECASE)
    val_match = val_pattern.search(tag_match.group(0))
    
    if not val_match:
        raise RuntimeError(f"The mock portal {field_name} input had no value attribute.")
        
    return html.unescape(val_match.group(1))


def _make_submit_action(staged_bid: dict[str, Any], answers: list[dict[str, str]]):
    """Create the approval-gated action that submits this exact reviewed draft."""
    review_url = staged_bid["review_url"]
    portal_bid_id = staged_bid["bid_id"]
    review_parts = urlsplit(review_url)
    portal_url = f"{review_parts.scheme}://{review_parts.netloc}"

    async def submit_action() -> bool:
        """Submit once, after HITLManager has observed explicit human approval."""
        timeout = aiohttp.ClientTimeout(total=SUBMIT_TIMEOUT_SECONDS, connect=10)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(review_url, allow_redirects=False) as review:
                    if review.status != 200:
                        raise RuntimeError(f"Mock portal review returned HTTP {review.status}.")
                    document = await review.text()

                # IDEMPOTENCY CHECK: Skip POST if the headless browser already submitted it
                if "Mock bid submitted" in document:
                    print("DEBUG: Bid was already submitted by the staging browser. Skipping POST.")
                else:
                    token = _hidden_value(document, "token")
                    revision = _hidden_value(document, "revision")
                    
                    # 2. FIRE THE FINAL SUBMISSION
                    submit_url = urljoin(review_url, f"/bids/{portal_bid_id}/submit")
                    async with session.post(
                        submit_url,
                        data={"token": token, "revision": revision},
                        allow_redirects=False,
                    ) as submitted:
                        if submitted.status not in (200, 302, 303):
                            raise RuntimeError(f"Mock portal submit returned HTTP {submitted.status}.")
                        location = submitted.headers.get("Location", "")
                        if urljoin(submit_url, location) != review_url:
                            raise RuntimeError("Mock portal redirected outside the reviewed draft.")

                verify_url = f"{portal_url.rstrip('/')}/api/bids/{portal_bid_id}"
                print(f"DEBUG: Verifying submission at {verify_url}")
                try:
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(SUBMIT_TIMEOUT_SECONDS, connect=10),
                        follow_redirects=False,
                    ) as verify_client:
                        verify_response = await verify_client.get(verify_url)
                except httpx.RequestError as exc:
                    raise RuntimeError("Submission verification request failed.") from exc

                raw_response = verify_response.text
                try:
                    data = verify_response.json()
                except ValueError as exc:
                    raise RuntimeError(
                        f"Submission was not confirmed. Raw response: {raw_response}"
                    ) from exc
                if (
                    verify_response.status_code != 200
                    or not isinstance(data, dict)
                    or data.get("submitted") is not True
                ):
                    raise RuntimeError(
                        f"Submission was not confirmed. Raw response: {raw_response}"
                    )
                return True
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise RuntimeError("The mock portal submit request failed.") from exc

    return submit_action


async def run_pipeline() -> BidSnapshot:
    """Build the proposal, stage it, register HITL state, and return its session."""
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    _, anakin_api_key, portal_url, rfp_source_url = _required_environment()

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
    items = list(RFP_SECTIONS.items())
    batch_size = 2  # Reduced to protect TPM limit
    
    for i in range(0, len(items), batch_size):
        batch = dict(items[i:i + batch_size])
        print(f"-> Processing batch {i // batch_size + 1} of {(len(items) + batch_size - 1) // batch_size}...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                batch_answers = await process_all_sections(batch, faiss_index, chunks)
                answers.extend(batch_answers)
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < max_retries - 1:
                    print(f"   [!] Groq TPM limit reached. Backing off for 12 seconds (Attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(12)
                else:
                    raise exc
                    
        if i + batch_size < len(items):
            await asyncio.sleep(5)
    
    proposal_answers = [dict(a) for a in answers]

    # --- THE SPLIT HANDOFF ---
    print("DEBUG: Performing split handoff for legacy mock portal routing...")
    
    # 1. Prepare the legacy payload for the robot browser
    # The Mock Portal HTML form still only has 3 input boxes. We must aggregate
    # the answers so the headless browser knows where to type them.
    mapped_answers_dict = {
        "Security": [],
        "Tech Specs": [],
        "Pricing": []
    }

    for item in proposal_answers:
        key = item["section"]
        val = item["answer"]
        key_upper = key.upper()
        
        # Heuristic routing based on section title keywords
        if any(kw in key_upper for kw in ("SEC", "SECURITY", "COMPLIANCE", "AUDIT", "ACCESS", "AUTH", "SSO")):
            mapped_answers_dict["Security"].append(f"[{key}]: {val}")
        elif any(kw in key_upper for kw in ("PRICE", "PRICING", "COST", "COMMERCIAL", "FEE")):
            mapped_answers_dict["Pricing"].append(f"[{key}]: {val}")
        else:
            # Default everything else to Tech Specs
            mapped_answers_dict["Tech Specs"].append(f"[{key}]: {val}")

    legacy_browser_payload = [
        {"section": k, "answer": "\n\n".join(v) if v else "Not explicitly requested in this RFP."}
        for k, v in mapped_answers_dict.items()
    ]

    # 4. STAGE AND REGISTER
    # We pass the aggregated 3-field payload to the robot browser so it doesn't crash on invalid form fields
    staged = await stage_bid(portal_url, legacy_browser_payload, anakin_api_key)
    submit_action = _make_submit_action(staged, legacy_browser_payload)
    
    # But we pass the raw, full 16-item payload to the Next.js UI!
    return await manager.register_staged_bid(
        staged,
        RFP_SECTIONS,        # Dynamic, unadulterated requirements
        proposal_answers,    # Dynamic, unadulterated answers
        submit_action,
    )


async def main(serve: bool = True) -> None:
    """Run the pipeline and optionally keep the HITL API serving the session."""
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    hitl_host = os.environ.get("HITL_HOST", "127.0.0.1")
    hitl_port = _hitl_port()
    _, _, portal_url, _ = _required_environment()
    _validate_local_service_ports(portal_url, hitl_port)
    if serve:
        ensure_port_available(hitl_host, hitl_port)

    snapshot = await run_pipeline()
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
        asyncio.run(main(serve=not args.no_server))
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"run_bid failed: {exc}") from exc