"""Local, JSON-only procurement API for the Next.js review dashboard.

The service stores mock bids in memory and must run with one worker. Drafts are
lost when the process restarts. It never targets or submits to an external site.

The final action is deliberately two-step: a human must first POST the exact
reviewed answers to ``/api/bids/{id}/approve`` and may only then POST the same
answers to ``/api/bids/{id}/submit``.
"""

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


app = FastAPI(title="Procurement Mock API", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SECTION_LENGTH = 256
MAX_ANSWER_LENGTH = 10_000


@dataclass
class Bid:
    id: str = field(default_factory=lambda: uuid4().hex)
    answers: dict[str, str] = field(default_factory=dict)
    revision: int = 0
    approval: asyncio.Event = field(default_factory=asyncio.Event)
    submitted: bool = False


class BidAnswersPayload(BaseModel):
    """Final answer text keyed by the RFP section shown in the dashboard."""

    model_config = ConfigDict(extra="forbid")
    answers: dict[str, str]


class BidApprovalPayload(BidAnswersPayload):
    """The attestations and exact answer revision approved by the reviewer."""

    authorized_representative: bool
    ungrounded_items_acknowledged: bool


class BidCreatePayload(BaseModel):
    """Optional initial answers for local API clients and fixtures."""

    model_config = ConfigDict(extra="forbid")
    answers: dict[str, str] = Field(default_factory=dict)


bids: dict[str, Bid] = {}


def portal_error(status_code: int, code: str, message: str) -> HTTPException:
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
            "The bid session is not staged or does not exist.",
        )
    return bid


def normalize_answers(answers: dict[str, str], *, require_answers: bool = True) -> dict[str, str]:
    """Validate and copy untrusted answer dictionaries before storing them."""
    if require_answers and not answers:
        raise HTTPException(422, "Provide at least one finalized answer.")

    normalized: dict[str, str] = {}
    for section, answer in answers.items():
        clean_section = section.strip()
        clean_answer = answer.strip()
        if not clean_section or len(clean_section) > MAX_SECTION_LENGTH:
            raise HTTPException(
                422,
                f"Section names require 1 to {MAX_SECTION_LENGTH} characters.",
            )
        if not clean_answer or len(clean_answer) > MAX_ANSWER_LENGTH:
            raise HTTPException(
                422,
                f"{clean_section} requires 1 to {MAX_ANSWER_LENGTH} characters.",
            )
        if clean_section in normalized:
            raise HTTPException(422, f"Duplicate section after normalization: {clean_section}.")
        normalized[clean_section] = clean_answer
    return normalized


def state_payload(bid: Bid) -> dict:
    approved = bid.approval.is_set()
    status = "submitted" if bid.submitted else "approved" if approved else "draft"
    return {
        "id": bid.id,
        "status": status,
        "revision": bid.revision,
        "answers": dict(bid.answers),
        "approved": approved,
        "submitted": bid.submitted,
    }


def create_demo_bid(bid_id: str) -> Bid:
    """Create a local draft for demos and in-process tests."""
    bid = Bid(id=bid_id)
    bids[bid.id] = bid
    return bid


# Request parsing happens before these handlers run. Their validation and state
# transitions contain no awaits, so one worker cannot expose a partial mutation.
@app.get("/")
async def health() -> dict:
    return {"status": "ok", "service": "procurement-mock-api"}


@app.post("/api/bids", status_code=201)
async def create_bid(payload: BidCreatePayload | None = None) -> dict:
    initial_answers = normalize_answers(
        payload.answers if payload is not None else {},
        require_answers=False,
    )
    bid = Bid(answers=initial_answers)
    bids[bid.id] = bid
    return state_payload(bid)


@app.get("/api/bids/{bid_id}")
async def bid_state(bid_id: str) -> dict:
    return state_payload(get_bid(bid_id))


@app.post("/api/bids/{bid_id}/answers")
async def update_bid_answers(bid_id: str, payload: BidAnswersPayload) -> dict:
    """Save edits as a draft; changing text invalidates any earlier approval."""
    bid = get_bid(bid_id)
    if bid.submitted:
        raise portal_error(409, "BID_ALREADY_SUBMITTED", "A submitted bid is read-only.")

    answers = normalize_answers(payload.answers)
    if answers != bid.answers:
        bid.answers = answers
        bid.revision += 1
        bid.approval.clear()
    return state_payload(bid)


@app.post("/api/bids/{bid_id}/approve")
async def approve_bid(bid_id: str, payload: BidApprovalPayload) -> dict:
    """Record explicit human approval for one exact finalized answer set."""
    bid = get_bid(bid_id)
    if bid.submitted:
        raise portal_error(409, "BID_ALREADY_SUBMITTED", "A submitted bid is read-only.")
    if not payload.authorized_representative or not payload.ungrounded_items_acknowledged:
        raise HTTPException(422, "Both required human certifications must be accepted.")

    answers = normalize_answers(payload.answers)
    if bid.approval.is_set():
        if answers == bid.answers:
            return state_payload(bid)
        raise portal_error(
            409,
            "BID_ALREADY_APPROVED",
            "A different answer set is already approved for this bid revision.",
        )
    if answers != bid.answers:
        bid.answers = answers
        bid.revision += 1
    bid.approval.set()  # This explicit endpoint is the only submission unlock.
    return state_payload(bid)


@app.post("/api/bids/{bid_id}/submit")
async def submit_bid(bid_id: str, payload: BidAnswersPayload) -> dict:
    """Persist and submit the exact answer set covered by human approval."""
    bid = get_bid(bid_id)
    if bid.submitted:
        return {
            "status": "success",
            "message": "Bid officially submitted",
            "id": bid.id,
            "submitted": True,
        }
    if not bid.approval.is_set():
        raise portal_error(
            403,
            "HUMAN_APPROVAL_REQUIRED",
            "Submission is locked until an authorized human approves this revision.",
        )

    answers = normalize_answers(payload.answers)
    if answers != bid.answers:
        raise portal_error(
            409,
            "ANSWERS_CHANGED_AFTER_APPROVAL",
            "Final answers changed after approval. Save and approve them again.",
        )

    bid.answers = answers
    bid.submitted = True
    bid.approval.clear()  # Consume approval so it cannot authorize another action.
    return {
        "status": "success",
        "message": "Bid officially submitted",
        "id": bid.id,
        "submitted": True,
    }


@app.get("/bids/{bid_id}/submit")
async def legacy_submission_state(bid_id: str) -> dict:
    """JSON-only compatibility status for already-issued staging receipts."""
    bid = get_bid(bid_id)
    payload = state_payload(bid)
    if bid.submitted:
        payload["message"] = "Mock bid submitted. Bid officially submitted."
    return payload


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    port = int(os.environ.get("MOCK_PORTAL_PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("MOCK_PORTAL_PORT must be between 1 and 65535.")
    uvicorn.run(
        app,
        host=os.environ.get("MOCK_PORTAL_HOST", "127.0.0.1"),
        port=port,
        workers=1,
    )
