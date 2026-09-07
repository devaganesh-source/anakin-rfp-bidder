"""In-memory approval gate for the orchestration API; run with one worker.

After ``stage_bid`` succeeds, call ``await manager.register_staged_bid(...)``
with its receipt, the requirements, reasoner answers, and an async submit_action.
Poll GET /api/bids/{id}; only a human's POST /api/bids/{id}/approve releases
the waiting action. Review data is immutable: changed answers need a new draft.

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
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict

from actuation.anakin_client import _mock_origin


BidStatus = Literal["awaiting_approval", "submitting", "submitted", "manual_intervention"]
SubmitAction = Callable[[], Awaitable[bool]]


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
        bid = _StagedBid(
            id=uuid4().hex, portal_bid_id=portal_id, review_url=review_url,
            comparison=_review_rows(requirements, answers), submit_action=submit_action,
        )
        bid.task = asyncio.create_task(self._wait_and_submit(bid), name=f"hitl-{bid.id}")
        self._bids[bid.id] = bid
        self._portal_ids.add(portal_id)
        return self._snapshot(bid)

    def _get_bid(self, bid_id: str) -> _StagedBid:
        bid = self._bids.get(bid_id)
        if bid is None:
            raise HTTPException(404, "Unknown bid.")
        return bid

    @staticmethod
    def _snapshot(bid: _StagedBid) -> BidSnapshot:
        return BidSnapshot(
            id=bid.id, portal_bid_id=bid.portal_bid_id, review_url=bid.review_url,
            status=bid.status, comparison=bid.comparison, approved=bid.approval.is_set(),
            submitted=bid.status == "submitted",
            manual_intervention_required=bid.status == "manual_intervention", error=bid.error,
        )

    async def _poll(self, bid_id: str, response: Response) -> BidSnapshot:
        # No await: each response is a coherent snapshot while other bid tasks run.
        response.headers["Cache-Control"] = "no-store"
        return self._snapshot(self._get_bid(bid_id))

    async def _approve(self, bid_id: str, response: Response) -> BidSnapshot:
        """Accept an explicit human approval and schedule the one final action."""
        # No await between checking state and setting the Event: one POST wins.
        bid = self._get_bid(bid_id)
        response.headers["Cache-Control"] = "no-store"
        if self._closed:
            raise HTTPException(503, "The HITL manager is shutting down.")
        if bid.status == "manual_intervention":
            raise HTTPException(409, bid.error)
        if bid.status == "awaiting_approval":
            if bid.task is None or bid.task.done():
                bid.status = "manual_intervention"
                bid.error = "Approval waiter is unavailable; manual intervention required."
                raise HTTPException(409, bid.error)
            bid.status = "submitting"
            bid.approval.set()  # Only this POST endpoint may release the submission gate.
        response.status_code = 200 if bid.status == "submitted" else 202
        return self._snapshot(bid)

    async def _wait_and_submit(self, bid: _StagedBid) -> None:
        # Only this task can invoke the callback; Event.wait blocks immediately
        # before submission, and yields so polling and other bids remain responsive.
        try:
            await bid.approval.wait()
            async with asyncio.timeout(self._submit_timeout):
                confirmed = await bid.submit_action()
            if confirmed is not True:
                raise RuntimeError("The submit action did not confirm submission.")
            bid.status = "submitted"
        except asyncio.CancelledError:
            bid.status = "manual_intervention"
            bid.error = (
                "Submission interrupted; outcome unknown. Check the mock portal manually."
                if bid.approval.is_set() else "Approval wait cancelled; no submission was attempted."
            )
            raise
        except Exception as exc:
            # A timeout or transport failure can occur after the portal accepts a
            # submit. Never retry or claim it was not submitted; require inspection.
            bid.status = "manual_intervention"
            bid.error = (
                f"Submission was not confirmed ({type(exc).__name__}). "
                "Check the mock portal manually before taking further action."
            )

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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Request tasks and bid waiters share one loop; shutdown drains owned waiters.
    try:
        yield
    finally:
        await manager.aclose()


app = FastAPI(title="RFP Bid Approval", lifespan=_lifespan)
app.include_router(router)
