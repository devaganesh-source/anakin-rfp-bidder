"""Read RFPs through Anakin Crawl; failures require manual intervention."""

import asyncio
import json
import os
import re
from typing import Any
from urllib.parse import quote, urlsplit

import aiohttp


CRAWL_API_URL = "https://api.anakin.io/v1/crawl"
REQUEST_TIMEOUT_SECONDS = 30
CRAWL_TIMEOUT_SECONDS = 120
MAX_POLLS = 20
POLL_INTERVAL_SECONDS = 2


class AnakinCrawlError(RuntimeError):
    """A failed or incomplete crawl that the caller should flag for review."""

    def __init__(self, message: str) -> None:
        super().__init__(f"{message} Manual intervention required.")


def _split_markdown_sections(markdown: str) -> dict[str, str]:
    """Split ATX (# through ######) and Setext headings into section bodies.

    Preamble text uses 'Introduction'. Repeated titles receive numbered
    suffixes. Fenced code is preserved without treating its contents as headings.
    """
    sections: dict[str, str] = {}
    title: str | None = None
    body: list[str] = []
    fence = ""
    skip_underline = False
    lines = markdown.splitlines()

    def save_section() -> None:
        text = "\n".join(body).strip()
        if title is None and not text:
            return
        base = title or "Introduction"
        key = base
        suffix = 2
        while key in sections:
            key = f"{base} ({suffix})"
            suffix += 1
        sections[key] = text

    for position, line in enumerate(lines):
        if skip_underline:
            skip_underline = False
            continue
        if fence:
            body.append(line)
            if re.fullmatch(rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*", line):
                fence = ""
            continue
        opening = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if opening and not (opening[1][0] == "`" and "`" in opening[2]):
            fence = opening[1]
            body.append(line)
            continue

        heading = re.match(r"^ {0,3}#{1,6}(?:[ \t]+(.*)|[ \t]*)$", line)
        heading_title: str | None = None
        if heading:
            heading_title = re.sub(r"(?:[ \t]+|^)#+[ \t]*$", "", heading[1] or "").strip()
        elif (
            line.strip()
            and not re.match(r"^(?: {4}|\t| {0,3}(?:>|[-+*][ \t]|\d+[.)][ \t]))", line)
            and position + 1 < len(lines)
            and re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*", lines[position + 1])
        ):
            heading_title = line.strip()
            skip_underline = True

        if heading_title is not None:
            save_section()
            title = heading_title or "Untitled"
            body = []
        else:
            body.append(line)

    save_section()
    return sections


async def _request_json(
    session: aiohttp.ClientSession,
    method: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Make one request using the session timeout; never retry failures."""
    # Awaited I/O yields to other tasks; each crawl calls this sequentially.
    try:
        async with session.request(
            method, endpoint, json=payload, allow_redirects=False
        ) as response:
            response.raise_for_status()
            if not 200 <= response.status < 300:
                raise AnakinCrawlError(f"Anakin returned HTTP {response.status}.")
            data = await response.json()
    except asyncio.TimeoutError as exc:
        raise AnakinCrawlError("Anakin request timed out.") from exc
    except aiohttp.ContentTypeError as exc:
        raise AnakinCrawlError("Anakin returned a non-JSON response.") from exc
    except aiohttp.ClientResponseError as exc:
        raise AnakinCrawlError(f"Anakin returned HTTP {exc.status}.") from exc
    except aiohttp.ClientError as exc:
        raise AnakinCrawlError("Could not complete the Anakin HTTP request.") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AnakinCrawlError("Anakin returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise AnakinCrawlError("Anakin returned an unexpected JSON structure.")
    return data


def _extract_markdown(result: dict[str, Any]) -> str:
    """Require successful page results so an incomplete RFP is never accepted."""
    pages = result.get("results")
    if not isinstance(pages, list) or not pages:
        raise AnakinCrawlError("The completed crawl contains no page results.")
    documents: list[str] = []
    for page in pages:
        if not isinstance(page, dict) or page.get("status") != "completed":
            raise AnakinCrawlError("A crawled page failed or is incomplete.")
        markdown = page.get("markdown")
        if not isinstance(markdown, str) or not markdown.strip():
            raise AnakinCrawlError("A crawled page contains no usable Markdown.")
        documents.append(markdown)
    return "\n\n".join(documents)


async def crawl_rfp(url: str, anakin_api_key: str) -> dict[str, str]:
    """Submit a one-page crawl, poll within fixed limits, and return sections.

    Pass the API key from os.environ['ANAKIN_API_KEY']; it is never logged.
    ANAKIN_CRAWL_API_URL may override the API endpoint for a local mock.
    Invalid arguments raise ValueError. HTTP, timeout, job, and response
    failures raise AnakinCrawlError; failed requests are never retried.
    """
    parsed_url = urlsplit(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError("RFP URL must be an absolute HTTP or HTTPS URL.")
    if not anakin_api_key.strip():
        raise ValueError("Provide ANAKIN_API_KEY from the environment.")

    endpoint = os.environ.get("ANAKIN_CRAWL_API_URL", CRAWL_API_URL).rstrip("/")
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    # Each invocation has its own session. Polls run sequentially and sleep
    # cooperatively, so callers may run separate crawls concurrently; no lock is used.
    try:
        async with asyncio.timeout(CRAWL_TIMEOUT_SECONDS):
            async with aiohttp.ClientSession(
                headers={"X-API-Key": anakin_api_key}, timeout=timeout
            ) as session:
                result = await _request_json(
                    session, "POST", endpoint, {"url": url, "maxPages": 1}
                )
                if result.get("status") not in ("pending", "processing"):
                    raise AnakinCrawlError("Anakin did not accept the crawl job.")
                job_id = result.get("jobId")
                if not isinstance(job_id, str) or not job_id.strip():
                    raise AnakinCrawlError("Anakin did not return a valid crawl job ID.")

                for attempt in range(MAX_POLLS):
                    if attempt:
                        await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    result = await _request_json(
                        session, "GET", f"{endpoint}/{quote(job_id, safe='')}"
                    )
                    status = result.get("status")
                    if status == "completed":
                        return _split_markdown_sections(_extract_markdown(result))
                    if status not in ("pending", "processing"):
                        raise AnakinCrawlError("The crawl failed or returned an unexpected status.")
    except asyncio.TimeoutError as exc:
        raise AnakinCrawlError("The crawl exceeded its overall time limit.") from exc

    raise AnakinCrawlError("The crawl did not finish within the polling limit.")
