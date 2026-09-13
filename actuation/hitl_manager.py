"""In-memory approval gate for the orchestration API; run with one worker.

After ``stage_bid`` succeeds, call ``await manager.register_staged_bid(...)``
with its receipt, the requirements, reasoner answers, and an async submit_action.
Poll GET /api/bids/{id}; only a human's POST /api/bids/{id}/approve releases
the waiting action. That approval atomically records the reviewed answer revision.

submit_action must return True only after confirming final submission on the
MOCK_PORTAL_URL draft. It must verify the reviewed answers and honor the portal's
token/revision approval protocol, use timeouts, and never retry. The existing
Anakin stage_bid closes its browser; it is not a resumable submission action.
There is deliberately no default action that pretends a bid was submitted.

Serve ``actuation.hitl_manager:app`` or include ``router`` in an orchestration
app and await ``manager.aclose()`` in its lifespan teardown. Do not mount this
router on the mock portal: that app owns separate routes at the same paths.
This trusted local demo has no authentication; state resets on process restart.
"""

import asyncio
import inspect
import logging
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from actuation.anakin_client import _mock_origin


LOGGER = logging.getLogger(__name__)


BidStatus = Literal["awaiting_approval", "submitting", "submitted", "manual_intervention"]
SubmitAction = Callable[[Mapping[str, str]], Awaitable[bool]]


def _http_error(status_code: int, code: str, message: str) -> HTTPException:
    """Return one stable error envelope for dashboard and resilience clients."""
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


class ReviewSection(BaseModel):
    """One side-by-side requirement/answer row, with its grounding evidence."""

    model_config = ConfigDict(frozen=True)
    section: str
    requirement: str
    answer: str
    source_snippet: str


class BidSnapshot(BaseModel):
    """Public polling data; never expose callbacks, tasks, or approval Events."""

    model_config = ConfigDict(frozen=True)
    id: str
    portal_bid_id: str
    review_url: str
    status: BidStatus
    comparison: tuple[ReviewSection, ...]
    approved: bool
    submitted: bool
    manual_intervention_required: bool
    error: str | None
    rfp_title: str
    rfp_source_url: str
    evidence_sources: tuple[str, ...]
    evidence_fetched_at: str
    human_override_sections: tuple[str, ...]
    human_resolved_sections: tuple[str, ...]


class BidApprovalRequest(BaseModel):
    """The reviewed answer revision and human attestations that unlock submission."""

    model_config = ConfigDict(extra="forbid", strict=True)
    answers: dict[str, str]
    authorized_representative: bool
    ungrounded_items_acknowledged: bool
    human_resolved_sections: list[str] = Field(default_factory=list)


class BidGenerationRequest(BaseModel):
    """An optional live RFP URL selected by the dashboard operator."""

    model_config = ConfigDict(extra="forbid", strict=True)
    source_url: str | None = Field(default=None, max_length=2048)


@dataclass
class _StagedBid:
    id: str
    portal_bid_id: str
    review_url: str
    comparison: tuple[ReviewSection, ...]
    submit_action: SubmitAction
    approval: asyncio.Event = field(default_factory=asyncio.Event)
    status: BidStatus = "awaiting_approval"
    error: str | None = None
    task: asyncio.Task[None] | None = None
    rfp_title: str = "Untitled RFP"
    rfp_source_url: str = ""
    evidence_sources: tuple[str, ...] = ()
    evidence_fetched_at: str = ""
    human_override_sections: tuple[str, ...] = ()
    human_resolved_sections: tuple[str, ...] = ()


def _review_rows(
    requirements: Mapping[str, str], answers: Sequence[Mapping[str, str]],
) -> tuple[ReviewSection, ...]:
    """Copy and align by title; never silently drop missing or duplicate sections."""
    if not requirements or any(
        not isinstance(title, str) or not title.strip()
        or not isinstance(text, str) or not text.strip()
        for title, text in requirements.items()
    ):
        raise ValueError("Requirements must contain nonempty section and text strings.")
    by_section: dict[str, ReviewSection] = {}
    for item in answers:
        if not isinstance(item, Mapping):
            raise ValueError("Each proposal answer must be a section/answer object.")
        title, answer = item.get("section"), item.get("answer")
        snippet = item.get("source_snippet", "")
        if (
            not isinstance(title, str) or title not in requirements or title in by_section
            or not isinstance(answer, str) or not answer.strip()
            or not isinstance(snippet, str)
        ):
            raise ValueError("Provide one nonempty answer per requirement, with a string snippet.")
        by_section[title] = ReviewSection(
            section=title, requirement=requirements[title], answer=answer,
            source_snippet=snippet,
        )
    if set(by_section) != set(requirements):
        raise ValueError("Every RFP section must have a proposal answer before review.")
    return tuple(by_section[title] for title in requirements)


class HITLManager:
    """Own one immutable review and one approval waiter for each staged draft.

    Use only from one event loop. Each bid runs independently; state checks and
    transitions contain no awaits, so duplicate approvals cannot race. The Event
    blocks the sole submit action, not polling requests or work on other bids.
    """

    def __init__(self, submit_timeout_seconds: float = 60.0) -> None:
        if not math.isfinite(submit_timeout_seconds) or submit_timeout_seconds <= 0:
            raise ValueError("submit_timeout_seconds must be finite and positive.")
        self._submit_timeout = submit_timeout_seconds
        self._bids: dict[str, _StagedBid] = {}
        self._portal_ids: set[str] = set()
        self._closed = False
        self.router = APIRouter(prefix="/api/bids", tags=["HITL"])
        self.router.add_api_route(
            "/{bid_id}", self._poll, methods=["GET"], response_model=BidSnapshot,
        )
        self.router.add_api_route(
            "/{bid_id}/approve", self._approve, methods=["POST"],
            response_model=BidSnapshot, status_code=202,
        )

    async def register_staged_bid(
        self, staged_bid: Mapping[str, Any], requirements: Mapping[str, str],
        answers: Sequence[Mapping[str, str]], submit_action: SubmitAction,
        *,
        rfp_title: str = "Untitled RFP",
        rfp_source_url: str = "",
        evidence_sources: Sequence[str] = (),
        evidence_fetched_at: str = "",
    ) -> BidSnapshot:
        """Register only after staging reaches review; return the dashboard ID.

        submit_action is an async callable, not an already-running Task or
        coroutine. Capture the same saved draft and answers in that callable.
        Registering the same portal draft twice is rejected, including on failure.
        """
        # Registration is atomic on this loop; the new task yields at Event.wait().
        if self._closed:
            raise RuntimeError("The HITL manager is closed.")
        if staged_bid.get("status") != "awaiting_approval" or staged_bid.get("submitted") is not False:
            raise ValueError("A successful, unsubmitted staging receipt is required.")
        portal_id = staged_bid.get("bid_id")
        if not isinstance(portal_id, str) or not re.fullmatch(r"[0-9a-f]{32}", portal_id):
            raise ValueError("The receipt must contain a mock portal bid ID.")
        review_url = staged_bid.get("review_url")
        if not isinstance(review_url, str):
            raise ValueError("The receipt must contain a mock portal review URL.")
        suffix = f"/bids/{portal_id}/submit"
        origin = _mock_origin(review_url.removesuffix(suffix))
        if review_url != origin + suffix:
            raise ValueError("The review URL must match the configured mock portal draft.")
        if portal_id in self._portal_ids:
            raise ValueError("This portal draft is already tracked; it cannot be registered again.")
        if not (
            inspect.iscoroutinefunction(submit_action)
            or inspect.iscoroutinefunction(getattr(submit_action, "__call__", None))
        ):
            raise ValueError("submit_action must be an async callable, not a running action.")
        metadata = (rfp_title, rfp_source_url, evidence_fetched_at)
        if any(not isinstance(value, str) or not value.strip() for value in metadata):
            raise ValueError("RFP title, source URL, and evidence timestamp are required.")
        source_urls = tuple(evidence_sources)
        if not source_urls or any(
            not isinstance(value, str) or not value.strip() for value in source_urls
        ):
            raise ValueError("At least one evidence source URL is required.")
        bid = _StagedBid(
            id=uuid4().hex, portal_bid_id=portal_id, review_url=review_url,
            comparison=_review_rows(requirements, answers), submit_action=submit_action,
            rfp_title=rfp_title.strip(), rfp_source_url=rfp_source_url.strip(),
            evidence_sources=source_urls, evidence_fetched_at=evidence_fetched_at.strip(),
        )
        bid.task = asyncio.create_task(self._wait_and_submit(bid), name=f"hitl-{bid.id}")
        self._bids[bid.id] = bid
        self._portal_ids.add(portal_id)
        return self._snapshot(bid)

    def _get_bid(self, bid_id: str) -> _StagedBid:
        bid = self._bids.get(bid_id)
        if bid is None:
            raise _http_error(404, "BID_NOT_READY", "The bid session is not staged or does not exist.")
        return bid

    @staticmethod
    def _snapshot(bid: _StagedBid) -> BidSnapshot:
        return BidSnapshot(
            id=bid.id, portal_bid_id=bid.portal_bid_id, review_url=bid.review_url,
            status=bid.status, comparison=bid.comparison, approved=bid.approval.is_set(),
            submitted=bid.status == "submitted",
            manual_intervention_required=bid.status == "manual_intervention", error=bid.error,
            rfp_title=bid.rfp_title, rfp_source_url=bid.rfp_source_url,
            evidence_sources=bid.evidence_sources, evidence_fetched_at=bid.evidence_fetched_at,
            human_override_sections=bid.human_override_sections,
            human_resolved_sections=bid.human_resolved_sections,
        )

    async def _poll(self, bid_id: str, response: Response) -> BidSnapshot:
        # No await: each response is a coherent snapshot while other bid tasks run.
        response.headers["Cache-Control"] = "no-store"
        return self._snapshot(self._get_bid(bid_id))

    async def _approve(
        self, bid_id: str, payload: BidApprovalRequest, response: Response
    ) -> BidSnapshot:
        """Accept an explicit human approval and schedule the one final action."""
        # No await between checking state and setting the Event: one POST wins.
        bid = self._get_bid(bid_id)
        response.headers["Cache-Control"] = "no-store"
        if self._closed:
            raise _http_error(503, "HITL_SHUTTING_DOWN", "The HITL manager is shutting down.")
        if bid.status == "manual_intervention":
            raise _http_error(
                409,
                "BID_MANUAL_INTERVENTION_REQUIRED",
                bid.error or "Manual intervention is required.",
            )
        if bid.status in {"submitting", "submitted"}:
            raise _http_error(
                409,
                "BID_ALREADY_APPROVED",
                "This bid has already been approved; duplicate approval is not accepted.",
            )
        if bid.status == "awaiting_approval":
            if not payload.authorized_representative or not payload.ungrounded_items_acknowledged:
                raise _http_error(
                    422,
                    "HUMAN_ATTESTATION_REQUIRED",
                    "Both required human certifications must be accepted.",
                )
            expected_sections = {row.section for row in bid.comparison}
            if set(payload.answers) != expected_sections or any(
                not isinstance(answer, str) or not answer.strip() or len(answer.strip()) > 10_000
                for answer in payload.answers.values()
            ):
                raise _http_error(
                    422,
                    "REVIEW_REVISION_INVALID",
                    "Provide one nonempty finalized answer for every RFP requirement.",
                )
            resolved_sections = tuple(dict.fromkeys(payload.human_resolved_sections))
            if any(section not in expected_sections for section in resolved_sections):
                raise _http_error(
                    422,
                    "RESOLUTION_SECTION_INVALID",
                    "Human-resolved sections must belong to this bid.",
                )
            if bid.task is None or bid.task.done():
                bid.status = "manual_intervention"
                bid.error = "Approval waiter is unavailable; manual intervention required."
                raise _http_error(409, "BID_WAITER_UNAVAILABLE", bid.error)
            reviewed_rows: list[ReviewSection] = []
            overrides: list[str] = []
            for row in bid.comparison:
                reviewed_answer = payload.answers[row.section].strip()
                if reviewed_answer != row.answer:
                    overrides.append(row.section)
                reviewed_rows.append(row.model_copy(update={"answer": reviewed_answer}))
            bid.comparison = tuple(reviewed_rows)
            bid.human_override_sections = tuple(overrides)
            bid.human_resolved_sections = resolved_sections
            bid.status = "submitting"
            bid.approval.set()  # Only this POST endpoint may release the submission gate.
        response.status_code = 200 if bid.status == "submitted" else 202
        return self._snapshot(bid)

    async def _wait_and_submit(self, bid: _StagedBid) -> None:
        # Only this task can invoke the callback; Event.wait blocks immediately
        # before submission, and yields so polling and other bids remain live.
        try:
            await bid.approval.wait()
            async with asyncio.timeout(self._submit_timeout):
                submitted = await bid.submit_action(
                    {row.section: row.answer for row in bid.comparison}
                )
            if submitted is not True:
                raise RuntimeError("The submission action did not confirm completion.")
            bid.status = "submitted"
        except asyncio.CancelledError:
            bid.status = "manual_intervention"
            bid.error = (
                "Submission interrupted; outcome unknown. Check the mock portal manually."
                if bid.approval.is_set() else "Approval wait cancelled; submission abandoned."
            )
            raise
        except Exception as exc:
            # STOP MASKING ERRORS: Pass the actual exception string to the dashboard
            bid.status = "manual_intervention"
            bid.error = f"Submission failed: {str(exc)}"
            raise

    async def aclose(self) -> None:
        """Cancel and drain owned tasks without ever granting approval."""
        # Mark closed before yielding; cancel all tasks together, then drain them.
        self._closed = True
        tasks = []
        for bid in self._bids.values():
            if bid.task is not None and not bid.task.done():
                # Record failure even if a task is cancelled before its first step.
                bid.status = "manual_intervention"
                bid.error = (
                    "Submission interrupted; outcome unknown. Check the mock portal manually."
                    if bid.approval.is_set() else "Approval wait cancelled; no submission was attempted."
                )
                bid.task.cancel()
                tasks.append(bid.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


manager = HITLManager()
router = manager.router
_generation_lock = asyncio.Lock()


async def _generate_bid(
    response: Response, payload: BidGenerationRequest | None = None
) -> BidSnapshot:
    """Run one proposal pipeline from the dashboard and return its review state."""
    response.headers["Cache-Control"] = "no-store"
    if _generation_lock.locked():
        raise _http_error(
            409,
            "BID_GENERATION_IN_PROGRESS",
            "Another bid is already being generated. Wait for it to finish before starting another.",
        )

    # Import lazily to keep the HITL manager reusable without creating an import
    # cycle: run_bid imports this module for the shared manager instance.
    from run_bid import run_pipeline

    async with _generation_lock:
        try:
            # Retrieval and one token-efficient reasoning batch happen inside
            # the orchestrator; this lock ensures one request owns browser staging.
            snapshot = await run_pipeline(payload.source_url if payload else None)
            if snapshot is None:
                raise RuntimeError("Dry-run mode did not create a review session.")
            return snapshot
        except Exception as exc:
            LOGGER.exception("Dashboard bid generation failed")
            raise _http_error(
                502,
                "BID_GENERATION_FAILED",
                f"Bid generation stopped before a review session was created ({type(exc).__name__}).",
            ) from exc


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Request tasks and bid waiters share one loop; shutdown drains owned waiters.
    try:
        yield
    finally:
        await manager.aclose()


app = FastAPI(title="RFP Bid Approval", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(router)
app.add_api_route(
    "/api/bids/generate",
    _generate_bid,
    methods=["POST"],
    response_model=BidSnapshot,
    status_code=201,
    tags=["HITL"],
)


@app.get("/api/health", include_in_schema=False)
async def health() -> dict[str, str]:
    """Confirm that the public orchestration process is accepting requests."""
    return {"status": "ok", "service": "anakin-rfp-bidder"}


# Production images include a static Next.js export. API routes are registered
# first so the dashboard catch-all can never shadow orchestration or approval.
DASHBOARD_EXPORT_DIR = Path(__file__).resolve().parents[1] / "dashboard" / "out"
if DASHBOARD_EXPORT_DIR.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=DASHBOARD_EXPORT_DIR, html=True),
        name="dashboard",
    )
