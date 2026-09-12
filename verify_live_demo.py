"""Run one credentialed end-to-end audit against the controlled mock portal."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

import run_bid
from actuation.hitl_manager import app, manager


async def verify(approve: bool = False) -> None:
    """Print grounding coverage and optionally exercise the explicit approval gate."""
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    try:
        snapshot = await run_bid.run_pipeline()
        if snapshot is None:
            raise RuntimeError("The live pipeline returned no review snapshot.")
        rows = list(snapshot.comparison)
        supported = [
            row for row in rows
            if not row.answer.startswith("CAPABILITY_NOT_FOUND") and row.source_snippet.strip()
        ]
        print("E2E_RESPONSE_AUDIT_BEGIN")
        print(json.dumps({
            "rfp_title": snapshot.rfp_title,
            "rfp_source_url": snapshot.rfp_source_url,
            "evidence_sources": list(snapshot.evidence_sources),
            "evidence_fetched_at": snapshot.evidence_fetched_at,
            "status_before_approval": snapshot.status,
            "requirements": len(rows),
            "grounded": len(supported),
            "responses": [
                {
                    "section": row.section,
                    "grounded": row in supported,
                    "answer": row.answer,
                    "source_snippet": row.source_snippet,
                }
                for row in rows
            ],
        }, indent=2, ensure_ascii=False))
        print("E2E_RESPONSE_AUDIT_END")

        if not approve:
            print("approval=SKIPPED (pass --approve to authorize the mock submission)")
            return

        # The CLI flag is an explicit human action; it releases exactly one test revision.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8001",
        ) as client:
            approval = await client.post(
                f"/api/bids/{snapshot.id}/approve",
                json={
                    "answers": {row.section: row.answer for row in rows},
                    "authorized_representative": True,
                    "ungrounded_items_acknowledged": True,
                    "human_resolved_sections": [],
                },
            )
            print(f"approval_http={approval.status_code}")
            approval.raise_for_status()
            for _ in range(100):
                response = await client.get(f"/api/bids/{snapshot.id}")
                response.raise_for_status()
                payload = response.json()
                if payload["status"] in {"submitted", "manual_intervention"}:
                    print(f"final_hitl_status={payload['status']}")
                    print(f"final_hitl_error={payload.get('error')}")
                    if payload["status"] != "submitted":
                        raise RuntimeError(payload.get("error") or "Submission failed.")
                    break
                await asyncio.sleep(0.05)
            else:
                raise RuntimeError("Submission state did not settle.")

        portal_url = snapshot.review_url.split("/bids/", 1)[0]
        async with httpx.AsyncClient(base_url=portal_url) as portal:
            response = await portal.get(f"/api/bids/{snapshot.portal_bid_id}")
            response.raise_for_status()
            print(f"portal_submitted={response.json().get('submitted')}")
    finally:
        await manager.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Explicitly authorize and verify one submission to the local mock portal.",
    )
    arguments = parser.parse_args()
    asyncio.run(verify(approve=arguments.approve))
