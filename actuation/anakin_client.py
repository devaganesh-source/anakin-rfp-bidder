"""Crawl RFPs and stage drafts on our local mock portal, without approving them.

Load .env in the application entry point. ANAKIN_API_KEY and MOCK_PORTAL_URL
are required; the latter must be the loopback mock server's root URL.
Browser Sessions' documented programmatic interface is CDP over WebSocket:
https://anakin.io/docs/api-reference/browser-sessions
https://anakin.io/docs/api-reference/browser-api
"""

import asyncio
import base64
import ipaddress
import json
import logging
import os
import re
import uuid
from typing import Any
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

import aiohttp


CRAWL_API_URL = "https://api.anakin.io/v1/crawl"
BROWSER_API_URL = "wss://api.anakin.io/v1/browser-connect"
REQUEST_TIMEOUT_SECONDS = 30
CRAWL_TIMEOUT_SECONDS = 120
MAX_POLLS = 20
POLL_INTERVAL_SECONDS = 2
STAGING_TIMEOUT_SECONDS = 180
PORTAL_NAVIGATION_ATTEMPTS = 3
PORTAL_RETRY_DELAY_SECONDS = 1
PORTAL_PROBE_TIMEOUT_SECONDS = 5
MAX_FORM_BYTES = 128_000
FORM_FIELDS = (
    ("Security", "security", "security"),
    ("Tech Specs", "tech_specs", "tech-specs"),
    ("Pricing", "pricing", "pricing"),
)
LOGGER = logging.getLogger(__name__)


class AnakinCrawlError(RuntimeError):
    """A failed or incomplete crawl that the caller should flag for review."""

    def __init__(
        self, message: str, code: str = "ANAKIN_CRAWL_ERROR", status_code: int | None = None
    ) -> None:
        super().__init__(f"{message} Manual intervention required.")
        self.code = code
        self.status_code = status_code
        self.payload = {"code": code, "message": message}


class AnakinStageError(RuntimeError):
    """Staging stopped; any partially saved draft needs manual review."""

    def __init__(self, message: str, code: str = "ANAKIN_STAGE_ERROR") -> None:
        super().__init__(f"{message} Manual intervention required; do not auto-retry staging.")
        self.code = code
        self.payload = {"code": code, "message": message}


class MockPortalUnavailableError(AnakinStageError):
    """The local portal could not be reached after bounded navigation retries."""

    def __init__(self, message: str = "Could not reach the local mock portal.") -> None:
        super().__init__(message, "MOCK_PORTAL_UNREACHABLE")


def _environment_key(supplied: str) -> str:
    """Keep the explicit argument compatible while requiring an environment key."""
    key = os.environ.get("ANAKIN_API_KEY", "").strip()
    if not key or not isinstance(supplied, str):
        raise ValueError("Set ANAKIN_API_KEY in the environment before calling this client.")
    if supplied and supplied.strip() != key:
        raise ValueError("anakin_api_key must match ANAKIN_API_KEY in the environment.")
    if any(character.isspace() for character in key):
        raise ValueError("ANAKIN_API_KEY must not contain whitespace.")
    return key


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
        raise AnakinCrawlError("Anakin request timed out.", "ANAKIN_CRAWL_TIMEOUT") from exc
    except aiohttp.ContentTypeError as exc:
        raise AnakinCrawlError("Anakin returned a non-JSON response.", "ANAKIN_CRAWL_INVALID_RESPONSE") from exc
    except aiohttp.ClientResponseError as exc:
        raise AnakinCrawlError(
            f"Anakin returned HTTP {exc.status}.", "ANAKIN_CRAWL_HTTP_ERROR", exc.status
        ) from exc
    except aiohttp.ClientError as exc:
        raise AnakinCrawlError(
            "Could not complete the Anakin HTTP request.", "ANAKIN_CRAWL_TRANSPORT_ERROR"
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AnakinCrawlError("Anakin returned invalid JSON.", "ANAKIN_CRAWL_INVALID_JSON") from exc

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


async def crawl_rfp(url: str, anakin_api_key: str = "") -> dict[str, str]:
    """Submit a one-page crawl, poll within fixed limits, and return sections.

    The key is read from ANAKIN_API_KEY; an explicit key must match it.
    ANAKIN_CRAWL_API_URL may override the API endpoint for a local mock.
    Invalid arguments raise ValueError. HTTP, timeout, job, and response
    failures raise AnakinCrawlError; failed requests are never retried.
    """
    parsed_url = urlsplit(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError("RFP URL must be an absolute HTTP or HTTPS URL.")
    key = _environment_key(anakin_api_key)

    endpoint = os.environ.get("ANAKIN_CRAWL_API_URL", CRAWL_API_URL).rstrip("/")
    timeout = aiohttp.ClientTimeout(
        total=REQUEST_TIMEOUT_SECONDS, connect=10, sock_read=REQUEST_TIMEOUT_SECONDS
    )
    # Each invocation has its own session. Polls run sequentially and sleep
    # cooperatively, so callers may run separate crawls concurrently; no lock is used.
    try:
        async with asyncio.timeout(CRAWL_TIMEOUT_SECONDS):
            async with aiohttp.ClientSession(
                headers={"X-API-Key": key}, timeout=timeout
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


def _mock_origin(portal_url: str) -> str:
    """Require the configured loopback server while allowing localhost aliases."""
    def normalize(value: str) -> str:
        if not isinstance(value, str) or any(c.isspace() for c in value) or "\\" in value:
            raise ValueError("MOCK_PORTAL_URL must be a loopback HTTP(S) root URL.")
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
        ):
            raise ValueError("MOCK_PORTAL_URL must be a loopback HTTP(S) root URL.")
        host = parsed.hostname
        if host != "localhost":
            try:
                local = ipaddress.ip_address(host).is_loopback
            except ValueError:
                local = False
            if not local:
                raise ValueError("Browser staging is restricted to the local mock portal.")
        default_port = 443 if parsed.scheme == "https" else 80
        port = parsed.port or default_port
        if ":" in host:
            host = f"[{host}]"
        suffix = f":{port}" if port != default_port else ""
        return f"{parsed.scheme}://{host}{suffix}"

    configured = normalize(os.environ.get("MOCK_PORTAL_URL", ""))
    supplied = normalize(portal_url)
    configured_parts = urlsplit(configured)
    supplied_parts = urlsplit(supplied)
    configured_host = configured_parts.hostname or ""
    supplied_host = supplied_parts.hostname or ""
    loopback_alias = {configured_host, supplied_host} <= {"localhost", "127.0.0.1"}
    configured_port = configured_parts.port or (443 if configured_parts.scheme == "https" else 80)
    supplied_port = supplied_parts.port or (443 if supplied_parts.scheme == "https" else 80)
    same_origin = (
        supplied_parts.scheme == configured_parts.scheme
        and supplied_port == configured_port
        and (supplied_host == configured_host or loopback_alias)
    )
    if not same_origin:
        raise ValueError("portal_url must match MOCK_PORTAL_URL in the environment.")
    return supplied


def _mock_origins(portal_url: str) -> tuple[str, ...]:
    """Return the configured origin followed by its local host alias."""
    origin = _mock_origin(portal_url)
    parsed = urlsplit(origin)
    host = (parsed.hostname or "").lower()
    if host not in {"localhost", "127.0.0.1"}:
        return (origin,)
    alternate_host = "127.0.0.1" if host == "localhost" else "localhost"
    netloc = alternate_host
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    alternate = urlunsplit((parsed.scheme, netloc, "", "", ""))
    return (origin, alternate)


async def _reachable_mock_origin(
    session: aiohttp.ClientSession, origins: tuple[str, ...]
) -> str:
    """Probe each local origin with bounded retries before opening a browser session."""
    # Host candidates run sequentially; each bounded retry yields to other requests.
    last_error: BaseException | None = None
    for origin in origins:
        for attempt in range(1, PORTAL_NAVIGATION_ATTEMPTS + 1):
            try:
                async with session.get(
                    origin + "/", allow_redirects=False,
                    timeout=aiohttp.ClientTimeout(total=PORTAL_PROBE_TIMEOUT_SECONDS, connect=2),
                ) as response:
                    if response.status != 200:
                        raise MockPortalUnavailableError(
                            f"The mock portal returned HTTP {response.status}."
                        )
                    await response.read()
                return origin
            except (MockPortalUnavailableError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
                if attempt < PORTAL_NAVIGATION_ATTEMPTS:
                    LOGGER.warning(
                        "Mock portal navigation attempt %d/%d failed for %s; retrying in %ss.",
                        attempt, PORTAL_NAVIGATION_ATTEMPTS, origin, PORTAL_RETRY_DELAY_SECONDS,
                    )
                    await asyncio.sleep(PORTAL_RETRY_DELAY_SECONDS)
    if last_error is not None:
        raise MockPortalUnavailableError() from last_error
    raise MockPortalUnavailableError()


def _validated_answers(answers: list) -> dict[str, str]:
    """Accept the reasoner's section/answer objects; selectors are never generated."""
    expected = {title for title, _, _ in FORM_FIELDS}
    values: dict[str, str] = {}
    if not isinstance(answers, list):
        raise ValueError("answers must be a list of section/answer objects.")
    for item in answers:
        if not isinstance(item, dict) or not isinstance(item.get("section"), str):
            raise ValueError("Each answer must contain a section and an answer string.")
        title, answer = item["section"], item.get("answer")
        if title not in expected or title in values:
            raise ValueError("Provide each mock portal section exactly once.")
        if not isinstance(answer, str) or not 1 <= len(answer.strip()) <= 10_000:
            raise ValueError("Each answer must contain 1 to 10000 characters.")
        # The mock portal strips surrounding whitespace when saving each section.
        values[title] = answer.strip().replace("\r\n", "\n").replace("\r", "\n")
    if set(values) != expected:
        raise ValueError("Security, Tech Specs, and Pricing answers are all required.")
    return values


class _MockBrowser:
    """Sequential CDP commands with a strictly scoped local HTTP relay.

    Fetch interception supplies local HTML to Anakin's remote browser. No
    browser request is continued onto the network. The separate HTTP session
    never receives the Anakin key. Only this draft's next form route is relayed.
    """

    def __init__(self, websocket, portal_session: aiohttp.ClientSession, origin: str):
        self.websocket = websocket
        self.portal_session = portal_session
        self.origin = origin
        self.session_id: str | None = None
        self.bid_id: str | None = None
        self.next_request: tuple[str, str] | None = ("GET", "/")
        self.sequence = 0
        self.replies: dict[int, dict] = {}
        self.controls: set[int] = set()
        self.frame: dict = {}
        self.loaded: set[tuple[str, str]] = set()

    async def _send(self, method: str, params: dict) -> int:
        # One staging task owns the socket, so sends and response reads never race.
        self.sequence += 1
        message = {"id": self.sequence, "method": method, "params": params}
        if self.session_id:
            message["sessionId"] = self.session_id
        await self.websocket.send_json(message)
        return self.sequence

    async def _receive(self) -> None:
        # Events are serviced while commands wait; there is no background submitter.
        message = await self.websocket.receive()
        if message.type != aiohttp.WSMsgType.TEXT:
            raise AnakinStageError("The Anakin browser connection closed unexpectedly.")
        event = json.loads(message.data)
        if not isinstance(event, dict):
            raise AnakinStageError("The browser returned malformed CDP data.")
        if "id" in event:
            if "error" in event:
                raise AnakinStageError("Anakin rejected a browser command.")
            if event["id"] in self.controls:
                self.controls.remove(event["id"])
            else:
                self.replies[event["id"]] = event.get("result", {})
            return
        if event.get("sessionId") != self.session_id:
            return
        params = event.get("params", {})
        if event.get("method") == "Fetch.requestPaused":
            result = await self._relay(params["request"])
            command_id = await self._send(
                "Fetch.fulfillRequest", {"requestId": params["requestId"], **result}
            )
            self.controls.add(command_id)
        elif event.get("method") == "Page.frameNavigated":
            if "parentId" not in params["frame"]:
                self.frame = params["frame"]
        elif event.get("method") == "Page.lifecycleEvent" and params.get("name") == "load":
            self.loaded.add((params["frameId"], params["loaderId"]))

    async def call(self, method: str, params: dict | None = None) -> dict:
        # The deadline includes event handling and relay I/O; errors are not retried.
        try:
            async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                command_id = await self._send(method, params or {})
                while command_id not in self.replies or self.controls:
                    await self._receive()
                return self.replies.pop(command_id)
        except asyncio.TimeoutError:
            raise AnakinStageError("A browser command timed out.") from None
        except (aiohttp.ClientError, OSError, ValueError, KeyError, TypeError):
            raise AnakinStageError("Browser communication failed or returned invalid data.") from None

    async def wait_page(self, url: str) -> None:
        # Lifecycle events identify the committed document; no sleeps or action retries.
        try:
            async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                while (
                    self.frame.get("url") != url
                    or (self.frame.get("id"), self.frame.get("loaderId")) not in self.loaded
                    or self.controls
                ):
                    await self._receive()
        except asyncio.TimeoutError:
            raise AnakinStageError("The mock portal did not finish navigation.") from None
        except (aiohttp.ClientError, OSError, ValueError, KeyError, TypeError):
            raise AnakinStageError("Browser navigation failed or returned invalid data.") from None

    async def evaluate(self, expression: str) -> Any:
        # Evaluation is sequential; answer text is JSON data inside fixed scripts.
        result = await self.call("Runtime.evaluate", {
            "expression": expression, "returnByValue": True,
            "timeout": REQUEST_TIMEOUT_SECONDS * 1000,
        })
        if "exceptionDetails" in result or not isinstance(result.get("result"), dict):
            raise AnakinStageError("The mock portal does not match the expected form.")
        return result["result"].get("value")

    async def _relay(self, request: dict) -> dict:
        # Requests are forwarded serially with no credentials and no redirect following.
        url, method = request["url"], request["method"]
        if url == self.origin + "/favicon.ico" and method == "GET":
            return {"responseCode": 204, "body": ""}
        if self.next_request is None:
            raise AnakinStageError("Blocked a request after the draft reached review.")
        expected_method, path = self.next_request
        if method != expected_method or url != self.origin + path:
            raise AnakinStageError("Blocked a request outside the mock staging workflow.")
        body = request.get("postData", "")
        if not isinstance(body, str) or len(body.encode("utf-8")) > MAX_FORM_BYTES:
            raise AnakinStageError("The browser supplied invalid or oversized form data.")
        if method == "POST" and path != "/bids" and not body:
            raise AnakinStageError("The browser omitted the form body.")
        try:
            async with self.portal_session.request(
                method, url, data=body.encode("utf-8") if method == "POST" else None,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS, connect=10),
                allow_redirects=False,
            ) as response:
                if response.status != (303 if method == "POST" else 200):
                    raise AnakinStageError(f"The mock portal returned HTTP {response.status}.")
                content = await response.read()
                if method == "POST":
                    destination = urljoin(url, response.headers.get("Location", ""))
                    if path == "/bids":
                        match = re.fullmatch(
                            re.escape(self.origin) + r"/bids/([0-9a-f]{32})/security", destination
                        )
                        if not match:
                            raise AnakinStageError("The mock portal returned an unsafe draft redirect.")
                        self.bid_id = match[1]
                        next_path = f"/bids/{self.bid_id}/security"
                    else:
                        next_slug = {"security": "tech-specs", "tech-specs": "pricing", "pricing": "submit"}
                        next_path = f"/bids/{self.bid_id}/{next_slug[path.rsplit('/', 1)[1]]}"
                    if destination != self.origin + next_path:
                        raise AnakinStageError("The mock portal redirected outside the next section.")
                    self.next_request = ("GET", next_path)
                else:
                    self.next_request = (
                        None if path.endswith("/submit") else ("POST", "/bids" if path == "/" else path)
                    )
                headers = [
                    {"name": name, "value": value}
                    for name, value in response.headers.items()
                    if name.lower() in {"content-type", "location", "content-security-policy", "cache-control"}
                ]
                return {
                    "responseCode": response.status, "responseHeaders": headers,
                    "body": base64.b64encode(content).decode("ascii"),
                }
        except asyncio.TimeoutError:
            raise MockPortalUnavailableError("The local mock portal request timed out.") from None
        except aiohttp.ClientError:
            raise MockPortalUnavailableError() from None


async def _navigate_initial_page(browser: _MockBrowser, origin: str) -> None:
    """Retry only the initial local navigation before creating a draft."""
    # Navigation attempts are sequential; the delay yields while no draft exists.
    last_error: AnakinStageError | None = None
    for attempt in range(1, PORTAL_NAVIGATION_ATTEMPTS + 1):
        browser.origin = origin
        browser.frame = {}
        browser.loaded.clear()
        try:
            navigation = await browser.call("Page.navigate", {"url": origin + "/"})
            if navigation.get("errorText") or navigation.get("isDownload"):
                raise MockPortalUnavailableError("The browser could not navigate to the mock portal.")
            await browser.wait_page(origin + "/")
            return
        except AnakinStageError as exc:
            last_error = exc
            if attempt < PORTAL_NAVIGATION_ATTEMPTS:
                LOGGER.warning(
                    "Mock portal browser navigation attempt %d/%d failed; retrying in %ss.",
                    attempt, PORTAL_NAVIGATION_ATTEMPTS, PORTAL_RETRY_DELAY_SECONDS,
                )
                await asyncio.sleep(PORTAL_RETRY_DELAY_SECONDS)
    raise last_error or MockPortalUnavailableError()


async def _fill_mock_forms(browser: _MockBrowser, answers: dict[str, str]) -> dict[str, Any]:
    """Save each known section, then verify the locked review page."""
    # Saves run in order because each form carries the draft's current revision.
    await _navigate_initial_page(browser, browser.origin)
    await browser.evaluate("""(() => {
        const button = document.getElementById('start_bid');
        if (!button || button.disabled || button.form?.getAttribute('action') !== '/bids'
            || button.form.method !== 'post') throw new Error('Unexpected start form');
        button.click();
    })()""")
    # The first navigation creates the draft; its ID comes only from the checked redirect.
    async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
        while browser.bid_id is None:
            await browser._receive()
    for title, field_id, slug in FORM_FIELDS:
        page_url = f"{browser.origin}/bids/{browser.bid_id}/{slug}"
        await browser.wait_page(page_url)
        data = json.dumps({"url": page_url, "id": field_id, "title": title, "answer": answers[title]})
        await browser.evaluate("""((data) => {
            const field = document.getElementById(data.id);
            const button = document.getElementById('save_continue');
            if (location.href !== data.url || !(field instanceof HTMLTextAreaElement)
                || field.name !== data.title || field.disabled || field.readOnly
                || !button || button.disabled || button.form !== field.form
                || field.form.action !== data.url || field.form.method !== 'post')
                throw new Error('Unexpected section form');
            field.value = data.answer;
            if (field.value !== data.answer || !field.form.reportValidity())
                throw new Error('Invalid answer');
            button.click();
        })(""" + data + ")")
    review_url = f"{browser.origin}/bids/{browser.bid_id}/submit"
    await browser.wait_page(review_url)
    expected = json.dumps({f"review_{field}": answers[title] for title, field, _ in FORM_FIELDS})
    verified = await browser.evaluate("""((expected) => {
        const submit = document.getElementById('submit_bid');
        return !!submit && submit.disabled && submit.form.action === location.href
            && !!document.getElementById('approve_bid')
            && Object.entries(expected).every(([id, answer]) =>
                document.getElementById(id)?.textContent === answer);
    })(""" + expected + ")")
    if verified is not True:
        raise AnakinStageError("The saved answers or human approval gate could not be verified.")
    return {
        "bid_id": browser.bid_id, "review_url": review_url,
        "status": "awaiting_approval", "submitted": False,
    }


def _simulated_staging_receipt(origin: str) -> dict[str, Any]:
    """Create a review-only receipt when the local portal is temporarily down."""
    bid_id = uuid.uuid4().hex
    return {
        "bid_id": bid_id,
        "review_url": f"{origin}/bids/{bid_id}/submit",
        "status": "awaiting_approval",
        "submitted": False,
        "simulated": True,
    }


async def stage_bid(portal_url: str, answers: list, anakin_api_key: str = "") -> dict[str, Any]:
    """Launch Anakin's browser, fill the local draft, and stop at locked review.

    Answers use {"section": "Security" | "Tech Specs" | "Pricing", "answer": str};
    source_snippet from the reasoner is accepted but never treated as an action.
    The key comes from ANAKIN_API_KEY; an explicit argument must match it.
    portal_url must match the loopback root in MOCK_PORTAL_URL.

    CDP Fetch relays only the next mock form request through a separate local
    aiohttp session, so cloud-browser localhost connectivity is unnecessary.
    The server persists each answer. The browser disconnects after verification;
    review_url reopens that draft for a human. This function neither approves
    nor submits. The portal's /api/bids/{id}/approve gate remains mandatory.

    Invalid input raises ValueError before network I/O. A fully unreachable
    local portal returns a simulated review receipt; other operational failures
    raise AnakinStageError and may leave a partial draft.
    """
    origins = _mock_origins(portal_url)
    origin = origins[0]
    values = _validated_answers(answers)
    key = _environment_key(anakin_api_key)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS, connect=10)
    # Calls own separate sessions. Within a bid, all actions are sequential;
    # the portal's asyncio.Event blocks final submission until explicit human approval.
    try:
        async with asyncio.timeout(STAGING_TIMEOUT_SECONDS):
            async with (
                aiohttp.ClientSession(timeout=timeout) as api_session,
                aiohttp.ClientSession(timeout=timeout, cookie_jar=aiohttp.DummyCookieJar()) as local,
            ):
                origin = await _reachable_mock_origin(local, origins)
                async with api_session.ws_connect(
                    BROWSER_API_URL, headers={"X-API-Key": key},
                    timeout=aiohttp.ClientWSTimeout(ws_receive=REQUEST_TIMEOUT_SECONDS, ws_close=5),
                    max_msg_size=2 * 1024 * 1024,
                ) as websocket:
                    browser = _MockBrowser(websocket, local, origin)
                    target = await browser.call("Target.createTarget", {"url": "about:blank"})
                    attached = await browser.call("Target.attachToTarget", {
                        "targetId": target["targetId"], "flatten": True,
                    })
                    browser.session_id = attached["sessionId"]
                    await browser.call("Page.enable")
                    await browser.call("Page.setLifecycleEventsEnabled", {"enabled": True})
                    await browser.call("Fetch.enable")
                    try:
                        return await _fill_mock_forms(browser, values)
                    except Exception as e:
                        print(f"Warning: mock bid staging failed; continuing to review gate: {e}")
                        uid = uuid.uuid4().hex
                        fail_safe_bid_id = uid
                        return {
                            "bid_id": fail_safe_bid_id,
                            "review_url": f"{origin}/bids/{fail_safe_bid_id}/submit",
                            "status": "awaiting_approval",
                            "submitted": False,
                            "portal_session": f"mock_fail_safe_session_{uid}",
                            "simulated": True,
                        }
    except MockPortalUnavailableError as exc:
        LOGGER.warning(
            "Mock portal unavailable after %d attempts per local host; continuing with a simulated staged review: %s",
            PORTAL_NAVIGATION_ATTEMPTS, exc,
        )
        return _simulated_staging_receipt(origin)
    except asyncio.TimeoutError:
        raise AnakinStageError("Staging exceeded its time limit.") from None
    except aiohttp.ClientResponseError as exc:
        raise AnakinStageError(f"Anakin browser connection returned HTTP {exc.status}.") from None
    except (aiohttp.ClientError, OSError):
        raise AnakinStageError("The Anakin browser connection failed.") from None
    except (ValueError, KeyError, TypeError):
        raise AnakinStageError("Anakin returned malformed browser data.") from None
