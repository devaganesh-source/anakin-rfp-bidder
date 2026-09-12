"""Bounded resilience checks for the local RFP bidder.

Run with ``python test_resilience.py``. The 15-run batch uses deterministic
local doubles for Groq and Anakin, so it never spends credentials or depends on
an external service. It executes ``run_bid.run_pipeline`` and the real HITL
registration path. Failure checks call the real parser, crawler, and approval
handlers. No test approves a batch bid; it stops at the human approval boundary.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from itertools import count
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from aiohttp import web
from fastapi import FastAPI
from fastapi import Response

import run_bid
from actuation.anakin_client import AnakinCrawlError, crawl_rfp
from actuation.hitl_manager import HITLManager
from mock_portal import main as mock_portal
from reasoner import groq_reasoner
from reasoner.groq_reasoner import GroqReasonerError, process_all_sections


BATCH_RUNS = 15
CHECK_TIMEOUT_SECONDS = 5.0
LOGGER = logging.getLogger("resilience")
LOCAL_PORTAL_URL = "http://127.0.0.1:8000"
TEST_RFP_SECTIONS = {
    "1.1 Encryption": "Describe encryption controls.",
    "1.2 Service Health": "Report current service health.",
}


@dataclass
class CheckResult:
    name: str
    status: str
    latency_ms: float
    error_code: str | None = None
    message: str | None = None


def _error_code(error: BaseException) -> str:
    """Extract the stable code exposed by the application error contracts."""
    return str(getattr(error, "code", "UNEXPECTED_ERROR"))


async def _timed_check(name: str, operation) -> CheckResult:
    """Run one check with a hard timeout and convert failures into a result."""
    # Each operation is bounded independently; the suite runs checks sequentially
    # while any HITL waiter remains blocked on its approval Event.
    started = time.perf_counter()
    try:
        await asyncio.wait_for(operation(), timeout=CHECK_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        elapsed = (time.perf_counter() - started) * 1000
        result = CheckResult(name, "FAIL", elapsed, "TEST_TIMEOUT", "Check exceeded its hard timeout.")
        LOGGER.error("FAIL %-42s latency_ms=%.2f code=%s", name, elapsed, result.error_code)
        return result
    except Exception as exc:  # The runner must report one failed check and continue.
        elapsed = (time.perf_counter() - started) * 1000
        result = CheckResult(name, "FAIL", elapsed, _error_code(exc), str(exc))
        LOGGER.error("FAIL %-42s latency_ms=%.2f code=%s message=%s", name, elapsed, result.error_code, exc)
        return result
    elapsed = (time.perf_counter() - started) * 1000
    result = CheckResult(name, "PASS", elapsed)
    LOGGER.info("PASS %-42s latency_ms=%.2f", name, elapsed)
    return result


async def _run_batch() -> list[CheckResult]:
    """Run the real orchestrator fifteen times with deterministic service edges."""
    bid_counter = count(1)

    async def fake_reasoner(sections, _faiss_index, _chunks):
        # All section calls complete concurrently, matching the real reasoner contract.
        await asyncio.sleep(0)
        return [
            {
                "section": title,
                "answer": f"Local resilience answer for {title}.",
                "source_snippet": "local capability fixture",
            }
            for title in sections
        ]

    async def fake_crawl(_url, _anakin_api_key):
        return {
            "Fixture RFP": "",
            "1\\. Requirements": (
                "**1.1 Encryption:** Describe encryption controls.\n\n"
                "**1.2 Service Health:** Report current service health."
            ),
        }

    def fake_live_evidence():
        return {
            "fetched_at": "2026-09-12T00:00:00+00:00",
            "live_chunks": ["local capability fixture"],
        }

    async def fake_stage(portal_url, _answers, _anakin_api_key):
        bid_id = f"{next(bid_counter):032x}"
        return {
            "bid_id": bid_id,
            "review_url": f"{portal_url}/bids/{bid_id}/submit",
            "status": "awaiting_approval",
            "submitted": False,
        }

    def fake_submit_action(_staged_bid):
        async def submit_action(_finalized_answers) -> bool:
            await asyncio.sleep(0)
            return True

        return submit_action

    results: list[CheckResult] = []
    environment = {
        "GROQ_API_KEY": "resilience-test-key",
        "ANAKIN_API_KEY": "resilience-test-key",
        "MOCK_PORTAL_URL": LOCAL_PORTAL_URL,
        "RFP_SOURCE_URL": "https://example.com/rfp",
    }
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, environment))
            stack.enter_context(patch.object(run_bid, "crawl_rfp", new=fake_crawl))
            stack.enter_context(
                patch.object(run_bid, "fetch_live_evidence", new=fake_live_evidence)
            )
            stack.enter_context(patch.object(run_bid, "build_index", return_value=object()))
            stack.enter_context(patch.object(run_bid, "process_all_sections", new=fake_reasoner))
            stack.enter_context(patch.object(run_bid, "stage_bid", new=fake_stage))
            stack.enter_context(patch.object(run_bid, "_make_submit_action", new=fake_submit_action))
            for iteration in range(1, BATCH_RUNS + 1):
                name = f"batch iteration {iteration:02d}/{BATCH_RUNS}"

                async def operation():
                    snapshot = await run_bid.run_pipeline()
                    if snapshot.status != "awaiting_approval" or snapshot.submitted:
                        raise AssertionError("Pipeline did not stop at the HITL approval boundary.")

                results.append(await _timed_check(name, operation))
    finally:
        # Cancel all approval waiters without approving or submitting a batch bid.
        # The waiters share the event loop but do not block the next iteration.
        await run_bid.manager.aclose()
    return results


async def _scenario_empty_or_missing_section() -> None:
    """Empty mappings and empty section text must fail before retrieval/API work."""
    for sections, expected_code in (
        ({}, "RFP_SECTIONS_MISSING"),
        ({"Security": ""}, "RFP_SECTION_INVALID"),
    ):
        try:
            await process_all_sections(sections, object(), [])
        except GroqReasonerError as exc:
            if exc.code != expected_code or not exc.payload["message"]:
                raise AssertionError(f"Unexpected validation payload: {exc.payload}") from exc
        else:
            raise AssertionError("Invalid RFP sections were accepted.")


async def _scenario_malformed_groq_json() -> None:
    """Malformed model JSON must become a typed parser failure, not a crash."""
    result = groq_reasoner._validate_answer("{not-json", "Security", ["evidence"])
    if not result["answer"].startswith("CAPABILITY_NOT_FOUND"):
        raise AssertionError("Malformed Groq JSON was accepted.")


async def _scenario_multi_excerpt_grounding() -> None:
    """Rendered links and multiple exact excerpts pass; fabrication fails closed."""
    context = [
        "Encryption at rest uses AES-256. Data in transit uses **TLS 1.3**. "
        "More information is available at [security.example.com](https://security.example.com/). "
        "Customers may enable [Secure Compute (available on Enterprise plans)]"
        "(/docs/networking/secure-compute)."
    ]
    valid = groq_reasoner._validate_answer(
        json.dumps({
            "is_supported": True,
            "evidence_citations": [
                "Encryption at rest uses AES-256.",
                "Data in transit uses TLS 1.3.",
                "More information is available at https://security.example.com/.",
                "Customers may enable Secure Compute (available on Enterprise plans).",
            ],
            "answer": json.dumps({
                "encryption_at_rest": "AES-256",
                "encryption_in_transit": "TLS 1.3",
            }),
        }),
        "Encryption",
        context,
    )
    if valid["answer"].startswith("CAPABILITY_NOT_FOUND"):
        raise AssertionError("Valid multi-excerpt grounding was rejected.")
    if "{" in valid["answer"] or "Encryption at rest: AES-256" not in valid["answer"]:
        raise AssertionError("Serialized answer text was not made reviewer-readable.")
    invalid = groq_reasoner._validate_answer(
        json.dumps({
            "is_supported": True,
            "evidence_citations": ["The service uses quantum encryption."],
            "answer": "The service uses quantum encryption.",
        }),
        "Encryption",
        context,
    )
    if not invalid["answer"].startswith("CAPABILITY_NOT_FOUND"):
        raise AssertionError("Fabricated evidence passed validation.")
    subparts = groq_reasoner._requirement_subparts(
        "Describe penetration testing, static analysis, continuous scanning, and resilience monitoring."
    )
    if len(subparts) != 4 or "continuous scanning" not in subparts:
        raise AssertionError(f"Compound requirement was not decomposed: {subparts!r}")
    extracted = groq_reasoner._best_source_statement(
        "continuous cloud security scanning",
        "Source URL: https://example.com\nSection: Infrastructure\n\n"
        "Access is controlled. Continuous cloud security scanning and alerting are enabled.",
    )
    if extracted != "Continuous cloud security scanning and alerting are enabled.":
        raise AssertionError(f"Unexpected extractive coverage sentence: {extracted!r}")
    if groq_reasoner._subpart_is_explicit(
        "storage separation", "Backups run every two hours and are retained for 30 days."
    ):
        raise AssertionError("Missing storage separation was treated as explicit coverage.")
    if not groq_reasoner._subpart_is_explicit(
        "storage separation", "Backups are stored separately in a storage service."
    ):
        raise AssertionError("Explicit storage separation was not recognized.")
    if groq_reasoner._subpart_is_explicit(
        "relevant plan conditions", "BAAs are available to eligible customers."
    ):
        raise AssertionError("Vague eligibility was accepted as a specific plan condition.")
    if not groq_reasoner._subpart_is_explicit(
        "relevant plan conditions", "BAAs are available to Pro and Enterprise customers."
    ):
        raise AssertionError("Explicit Pro and Enterprise eligibility was not recognized.")
    plan_source = (
        "Source URL: https://example.com\nSection: HIPAA\n\n"
        "Pro teams can purchase the HIPAA BAA add-on. "
        "Enterprise teams can request a BAA or add Secure Compute to their plan."
    )
    plan_statement = groq_reasoner._best_source_statement(
        "any relevant plan conditions", plan_source
    )
    if "Pro teams" not in plan_statement or "Enterprise teams" not in plan_statement:
        raise AssertionError(f"Plan conditions were not fully extracted: {plan_statement!r}")
    backup_source = (
        "Source URL: https://example.com\nSection: Data backup\n\n"
        "Backups are persisted for 30 days and globally replicated. "
        "Backups are periodically tested by the engineering team."
    )
    restored = groq_reasoner._best_source_statement("restoration testing", backup_source)
    if restored != "Backups are periodically tested by the engineering team.":
        raise AssertionError(f"Restoration-test evidence was not normalized: {restored!r}")
    focused_soc = [
        "Vercel has a SOC 2 Type 2 attestation for Security, Confidentiality, and Availability. "
        "More information is available at security.vercel.com."
    ]
    pruned = groq_reasoner._filter_answer_to_evidence(
        "Vercel has a SOC 2 Type 2 attestation. Supporting documents are in the ENX TISAX portal.",
        "Identify the SOC attestation and supporting documentation.",
        focused_soc,
    )
    if "SOC 2 Type 2" not in pruned or "ENX" in pruned or "TISAX" in pruned:
        raise AssertionError(f"Irrelevant named claim was not removed: {pruned!r}")
    pruned_encryption = groq_reasoner._filter_answer_to_evidence(
        "Data at rest uses AES-256. Centralized IAM regulates production access.",
        "Specify the encryption algorithm and protocol for data at rest and in transit.",
        ["Data at rest uses AES-256 and data in transit uses TLS 1.3. Centralized IAM is used."],
    )
    if "AES-256" not in pruned_encryption or "IAM" in pruned_encryption:
        raise AssertionError(f"Unrequested control sentence survived: {pruned_encryption!r}")


async def _scenario_groq_timeout() -> None:
    """A slow fake SDK call must be cancelled by the reasoner's bounded timeout."""

    class SlowCompletions:
        async def create(self, **_kwargs):
            await asyncio.sleep(0.2)

    class SlowChat:
        completions = SlowCompletions()

    class SlowClient:
        chat = SlowChat()

        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

    with (
        patch.dict(os.environ, {"GROQ_API_KEY": "resilience-test-key"}),
        patch.object(groq_reasoner, "AsyncGroq", SlowClient),
        patch.object(groq_reasoner, "REQUEST_TIMEOUT_SECONDS", 0.05),
    ):
        try:
            await groq_reasoner.generate_section_answer("Security", "Need encryption", ["evidence"])
        except GroqReasonerError as exc:
            if exc.code != "GROQ_TIMEOUT" or not exc.payload["message"]:
                raise AssertionError(f"Unexpected timeout payload: {exc.payload}") from exc
        else:
            raise AssertionError("Slow Groq call was not timed out.")


async def _scenario_structured_output_retry() -> None:
    """One Groq constrained-decoding failure is retried, then validated."""
    calls = 0

    class FlakyCompletions:
        async def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise groq_reasoner.groq.BadRequestError(
                    "strict schema generation failed",
                    response=httpx.Response(
                        400,
                        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
                    ),
                    body={"error": {"code": "json_validate_failed"}},
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=json.dumps({
                        "is_supported": True,
                        "evidence_citations": ["Encryption at rest uses AES-256."],
                        "answer": "Encryption at rest uses AES-256.",
                    })),
                )]
            )

    class FlakyClient:
        chat = SimpleNamespace(completions=FlakyCompletions())

        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

    async def no_delay(_seconds):
        return None

    with (
        patch.dict(os.environ, {"GROQ_API_KEY": "resilience-test-key"}),
        patch.object(groq_reasoner, "AsyncGroq", FlakyClient),
        patch.object(groq_reasoner.asyncio, "sleep", new=no_delay),
    ):
        answer = await groq_reasoner.generate_section_answer(
            "Encryption", "State encryption at rest.", ["Encryption at rest uses AES-256."]
        )
    if calls != 2 or answer["answer"].startswith("CAPABILITY_NOT_FOUND"):
        raise AssertionError("Structured-output retry did not recover exactly once.")


async def _scenario_anakin_crawl_404() -> None:
    """A local 404 responder exercises the crawler's non-2xx path safely."""
    # The temporary server handles one request; cleanup closes its socket before
    # this scenario returns to the sequential suite.
    application = web.Application()

    async def blocked(_request):
        return web.Response(status=404, text="blocked")

    application.router.add_route("*", "/crawl", blocked)
    runner = web.AppRunner(application)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    previous_endpoint = os.environ.get("ANAKIN_CRAWL_API_URL")
    os.environ["ANAKIN_CRAWL_API_URL"] = f"http://127.0.0.1:{port}/crawl"
    os.environ["ANAKIN_API_KEY"] = "resilience-test-key"
    try:
        try:
            await crawl_rfp("http://127.0.0.1:9000/rfp", "resilience-test-key")
        except AnakinCrawlError as exc:
            if exc.code != "ANAKIN_CRAWL_HTTP_ERROR" or exc.status_code != 404:
                raise AssertionError(f"Unexpected crawl payload: {exc.payload}") from exc
        else:
            raise AssertionError("404 crawl response was accepted.")
    finally:
        if previous_endpoint is None:
            os.environ.pop("ANAKIN_CRAWL_API_URL", None)
        else:
            os.environ["ANAKIN_CRAWL_API_URL"] = previous_endpoint
        await runner.cleanup()


def _asgi_app(manager: HITLManager) -> FastAPI:
    """Mount one isolated manager router for real HTTP-level approval checks."""
    application = FastAPI()
    application.include_router(manager.router)
    return application


async def _register_test_bid(manager: HITLManager, submit_action) -> str:
    """Register a valid staged receipt against the configured local portal."""
    portal_bid_id = "b" * 32
    portal_url = os.environ.get("MOCK_PORTAL_URL", LOCAL_PORTAL_URL)
    await manager.register_staged_bid(
        {
            "bid_id": portal_bid_id,
            "review_url": f"{portal_url}/bids/{portal_bid_id}/submit",
            "status": "awaiting_approval",
            "submitted": False,
        },
        TEST_RFP_SECTIONS,
        [
            {
                "section": title,
                "answer": "A reviewed local answer.",
                "source_snippet": "local capability fixture",
            }
            for title in TEST_RFP_SECTIONS
        ],
        submit_action,
        rfp_title="Fixture RFP",
        rfp_source_url="https://example.com/rfp",
        evidence_sources=("https://example.com/evidence",),
        evidence_fetched_at="2026-09-12T00:00:00+00:00",
    )
    return next(reversed(manager._bids))


async def _scenario_duplicate_approval() -> None:
    """The HITL and mock-portal approval routes reject duplicates and run once."""
    manager = HITLManager(submit_timeout_seconds=1)
    calls = 0

    async def submit_action(_answers) -> bool:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return True

    try:
        bid_id = await _register_test_bid(manager, submit_action)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_asgi_app(manager)),
            base_url=LOCAL_PORTAL_URL,
        ) as client:
            approval_payload = {
                "answers": {
                    title: "A reviewed local answer." for title in TEST_RFP_SECTIONS
                },
                "authorized_representative": True,
                "ungrounded_items_acknowledged": True,
                "human_resolved_sections": [],
            }
            first = await client.post(f"/api/bids/{bid_id}/approve", json=approval_payload)
            second = await client.post(f"/api/bids/{bid_id}/approve", json=approval_payload)
        if first.status_code != 202:
            raise AssertionError(f"First approval returned HTTP {first.status_code}.")
        if second.status_code != 409 or second.json()["detail"]["code"] != "BID_ALREADY_APPROVED":
            raise AssertionError(f"Duplicate approval was not rejected: {second.text}")
        await asyncio.wait_for(manager._bids[bid_id].task, timeout=1)
        if calls != 1:
            raise AssertionError(f"Submit action ran {calls} times.")

        portal_answers = {"Security": "A reviewed local answer."}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_portal.app),
            base_url=LOCAL_PORTAL_URL,
        ) as client:
            created = await client.post("/api/bids", json={"answers": portal_answers})
            portal_id = created.json()["id"]
            first_portal = await client.post(
                f"/api/bids/{portal_id}/approve",
                json={
                    "answers": portal_answers,
                    "authorized_representative": True,
                    "ungrounded_items_acknowledged": True,
                },
            )
            changed_portal = await client.post(
                f"/api/bids/{portal_id}/approve",
                json={
                    "answers": {"Security": "Changed after approval."},
                    "authorized_representative": True,
                    "ungrounded_items_acknowledged": True,
                },
            )
        if first_portal.status_code != 200 or changed_portal.status_code != 409:
            raise AssertionError((first_portal.text, changed_portal.text))
    finally:
        await manager.aclose()


async def _scenario_approval_before_staging() -> None:
    """The HITL and mock-portal routes reject approval before staging."""
    manager = HITLManager()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_asgi_app(manager)),
            base_url=LOCAL_PORTAL_URL,
        ) as client:
            response = await client.post(
                "/api/bids/not-staged/approve",
                json={
                    "answers": {"Security": "Reviewed answer."},
                    "authorized_representative": True,
                    "ungrounded_items_acknowledged": True,
                    "human_resolved_sections": [],
                },
            )
        if response.status_code != 404 or response.json()["detail"]["code"] != "BID_NOT_READY":
            raise AssertionError(f"Early approval was not rejected: {response.text}")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_portal.app),
            base_url=LOCAL_PORTAL_URL,
        ) as client:
            portal_response = await client.post(
                "/api/bids/not-staged/approve",
                json={
                    "answers": {"Security": "Reviewed answer."},
                    "authorized_representative": True,
                    "ungrounded_items_acknowledged": True,
                },
            )
        if portal_response.status_code != 404 or portal_response.json()["detail"]["code"] != "BID_NOT_READY":
            raise AssertionError(f"Mock portal early approval was not rejected: {portal_response.text}")
    finally:
        await manager.aclose()


async def run_suite() -> list[CheckResult]:
    """Run the batch and all bounded failure scenarios."""
    if not os.environ.get("MOCK_PORTAL_URL"):
        os.environ["MOCK_PORTAL_URL"] = LOCAL_PORTAL_URL
    results = await _run_batch()
    scenarios = (
        ("scenario 1: empty or missing RFP section", _scenario_empty_or_missing_section),
        ("scenario 2: malformed Groq JSON", _scenario_malformed_groq_json),
        ("scenario 3: multi-excerpt grounding", _scenario_multi_excerpt_grounding),
        ("scenario 4: Groq timeout", _scenario_groq_timeout),
        ("scenario 5: strict-output retry", _scenario_structured_output_retry),
        ("scenario 6: Anakin crawl 404", _scenario_anakin_crawl_404),
        ("scenario 7: duplicate approval", _scenario_duplicate_approval),
        ("scenario 8: approval before staging", _scenario_approval_before_staging),
    )
    for name, operation in scenarios:
        results.append(await _timed_check(name, operation))
    return results


async def main() -> int:
    results = await run_suite()
    passed = sum(result.status == "PASS" for result in results)
    failed = len(results) - passed
    print(json.dumps({"passed": passed, "failed": failed, "results": [asdict(result) for result in results]}, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        raise SystemExit(asyncio.run(main()))
    except (KeyboardInterrupt, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Resilience suite failed to start: {exc}") from exc
