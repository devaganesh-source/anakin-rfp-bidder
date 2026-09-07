"""Bounded resilience checks for the local RFP bidder.

Run with ``python test_resilience.py``. The 15-run batch uses deterministic
local doubles for Groq and Anakin by default, so it never spends credentials
or depends on an external service. Use ``--live`` to run the same batch through
the configured services. Both modes execute ``run_bid.run_pipeline`` and the
real HITL registration path. Failure checks call the real parser, crawler, and
approval handlers. No test approves a batch bid; the batch stops at the human
approval boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from itertools import count
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


async def _run_batch(live: bool = False) -> list[CheckResult]:
    """Run the real orchestrator fifteen times with optional live service edges."""
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

    async def fake_stage(portal_url, _answers, _anakin_api_key):
        bid_id = f"{next(bid_counter):032x}"
        return {
            "bid_id": bid_id,
            "review_url": f"{portal_url}/bids/{bid_id}/submit",
            "status": "awaiting_approval",
            "submitted": False,
        }

    def fake_submit_action(_staged_bid, _answers):
        async def submit_action() -> bool:
            await asyncio.sleep(0)
            return True

        return submit_action

    results: list[CheckResult] = []
    environment = {
        "GROQ_API_KEY": "resilience-test-key",
        "ANAKIN_API_KEY": "resilience-test-key",
        "MOCK_PORTAL_URL": LOCAL_PORTAL_URL,
    }
    try:
        with ExitStack() as stack:
            if not live:
                stack.enter_context(patch.dict(os.environ, environment))
                stack.enter_context(
                    patch.object(run_bid, "load_and_chunk_docs", return_value=["local capability fixture"])
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
    try:
        groq_reasoner._validate_answer("{not-json", "Security", ["evidence"])
    except GroqReasonerError as exc:
        if exc.code != "GROQ_INVALID_JSON" or exc.payload["code"] != exc.code:
            raise AssertionError(f"Unexpected Groq parsing payload: {exc.payload}") from exc
    else:
        raise AssertionError("Malformed Groq JSON was accepted.")


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
        run_bid.RFP_SECTIONS,
        [
            {
                "section": title,
                "answer": "A reviewed local answer.",
                "source_snippet": "local capability fixture",
            }
            for title in run_bid.RFP_SECTIONS
        ],
        submit_action,
    )
    return next(reversed(manager._bids))


async def _scenario_duplicate_approval() -> None:
    """The HITL and mock-portal approval routes reject duplicates and run once."""
    manager = HITLManager(submit_timeout_seconds=1)
    calls = 0

    async def submit_action() -> bool:
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
            first = await client.post(f"/api/bids/{bid_id}/approve")
            second = await client.post(f"/api/bids/{bid_id}/approve")
        if first.status_code != 202:
            raise AssertionError(f"First approval returned HTTP {first.status_code}.")
        if second.status_code != 409 or second.json()["detail"]["code"] != "BID_ALREADY_APPROVED":
            raise AssertionError(f"Duplicate approval was not rejected: {second.text}")
        await asyncio.wait_for(manager._bids[bid_id].task, timeout=1)
        if calls != 1:
            raise AssertionError(f"Submit action ran {calls} times.")

        portal_bid = mock_portal.Bid()
        portal_bid.answers = {
            title: "A reviewed local answer." for title, _, _ in mock_portal.SECTIONS.values()
        }
        mock_portal.bids[portal_bid.id] = portal_bid
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=mock_portal.app),
                base_url=LOCAL_PORTAL_URL,
            ) as client:
                first_portal = await client.post(
                    f"/api/bids/{portal_bid.id}/approve",
                    data={"token": portal_bid.token, "revision": "0", "confirm": "yes"},
                )
                second_portal = await client.post(
                    f"/api/bids/{portal_bid.id}/approve",
                    data={"token": portal_bid.token, "revision": "0", "confirm": "yes"},
                )
            if first_portal.status_code != 303:
                raise AssertionError(f"Mock portal first approval returned HTTP {first_portal.status_code}.")
            if (
                second_portal.status_code != 409
                or second_portal.json()["detail"]["code"] != "BID_ALREADY_APPROVED"
            ):
                raise AssertionError(f"Mock portal duplicate approval was not rejected: {second_portal.text}")
        finally:
            mock_portal.bids.pop(portal_bid.id, None)
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
            response = await client.post("/api/bids/not-staged/approve")
        if response.status_code != 404 or response.json()["detail"]["code"] != "BID_NOT_READY":
            raise AssertionError(f"Early approval was not rejected: {response.text}")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_portal.app),
            base_url=LOCAL_PORTAL_URL,
        ) as client:
            portal_response = await client.post(
                "/api/bids/not-staged/approve",
                data={"token": "missing", "revision": "0", "confirm": "yes"},
            )
        if portal_response.status_code != 404 or portal_response.json()["detail"]["code"] != "BID_NOT_READY":
            raise AssertionError(f"Mock portal early approval was not rejected: {portal_response.text}")
    finally:
        await manager.aclose()


async def run_suite(live: bool = False) -> list[CheckResult]:
    """Run the batch and all six bounded failure scenarios."""
    if not os.environ.get("MOCK_PORTAL_URL"):
        os.environ["MOCK_PORTAL_URL"] = LOCAL_PORTAL_URL
    results = await _run_batch(live=live)
    scenarios = (
        ("scenario 1: empty or missing RFP section", _scenario_empty_or_missing_section),
        ("scenario 2: malformed Groq JSON", _scenario_malformed_groq_json),
        ("scenario 3: Groq timeout", _scenario_groq_timeout),
        ("scenario 4: Anakin crawl 404", _scenario_anakin_crawl_404),
        ("scenario 5: duplicate approval", _scenario_duplicate_approval),
        ("scenario 6: approval before staging", _scenario_approval_before_staging),
    )
    for name, operation in scenarios:
        results.append(await _timed_check(name, operation))
    return results


async def main(live: bool = False) -> int:
    results = await run_suite(live=live)
    passed = sum(result.status == "PASS" for result in results)
    failed = len(results) - passed
    print(json.dumps({"passed": passed, "failed": failed, "results": [asdict(result) for result in results]}, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the 15 batch iterations through configured Groq and Anakin services.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        raise SystemExit(asyncio.run(main(live=args.live)))
    except (KeyboardInterrupt, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Resilience suite failed to start: {exc}") from exc
