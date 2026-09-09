"""Run two local RFPs through the production pipeline and print captured JSON."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import run_bid


PROJECT_ROOT = Path(__file__).resolve().parent
INPUTS = (
    ("A", "http://127.0.0.1:8080/mock_rfp.html"),
    ("B", "http://127.0.0.1:8080/mock_rfp_b.html"),
)


async def main() -> None:
    """Capture crawl and reasoner values while the real pipeline stages each bid."""
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(PROJECT_ROOT / ".env")
    original_crawler = run_bid.RFPCrawler
    original_process = run_bid.process_all_sections
    current: dict[str, Any] = {}

    class CapturingCrawler(original_crawler):
        async def crawl(self) -> dict[str, str]:
            sections = await super().crawl()
            current["rfp_sections"] = dict(sections)
            return sections

    async def capture_process(*args: Any, **kwargs: Any) -> list[dict[str, str]]:
        answers = await original_process(*args, **kwargs)
        current.setdefault("generated_answers", []).extend(dict(answer) for answer in answers)
        return answers

    run_bid.RFPCrawler = CapturingCrawler
    run_bid.process_all_sections = capture_process
    results: dict[str, Any] = {}
    selected_case = os.environ.get("VERIFY_CASE", "").strip().upper()
    inputs = INPUTS if not selected_case else tuple(
        item for item in INPUTS if item[0] == selected_case
    )
    if selected_case and not inputs:
        raise ValueError("VERIFY_CASE must be A or B.")
    try:
        for label, source_url in inputs:
            current = {"source_url": source_url, "generated_answers": []}
            os.environ["RFP_SOURCE_URL"] = source_url
            snapshot = await run_bid.run_pipeline()
            current["staged_snapshot"] = snapshot.model_dump(mode="json")
            results[label] = current
    finally:
        run_bid.RFPCrawler = original_crawler
        run_bid.process_all_sections = original_process
        await run_bid.manager.aclose()

    print("RAW_PIPELINE_OUTPUT_BEGIN")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("RAW_PIPELINE_OUTPUT_END")


if __name__ == "__main__":
    asyncio.run(main())
