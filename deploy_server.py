"""Run the public app and its private mock portal in one deployment container."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn


PORTAL_START_TIMEOUT_SECONDS = 30.0
PROCESS_STOP_TIMEOUT_SECONDS = 5.0


def _port_from_environment(name: str, default: int) -> int:
    """Read and validate one TCP port from the deployment environment."""
    try:
        port = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError(f"{name} must be between 1 and 65535.")
    return port


def _wait_for_mock_portal(process: subprocess.Popen[bytes], url: str) -> None:
    """Block startup until the private portal responds or exits."""
    deadline = time.monotonic() + PORTAL_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"The private mock portal exited during startup with code {return_code}."
            )
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(0.2)
    raise RuntimeError("The private mock portal did not become ready in time.")


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    """Gracefully stop the private child process, then bound shutdown."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)


def run() -> None:
    """Start one private portal and one public single-worker API server."""
    public_port = _port_from_environment("PORT", 10000)
    portal_port = _port_from_environment("MOCK_PORTAL_PORT", 8000)
    if public_port == portal_port:
        raise RuntimeError("PORT and MOCK_PORTAL_PORT must be different.")

    portal_url = f"http://127.0.0.1:{portal_port}"
    configured_portal_url = os.environ.setdefault("MOCK_PORTAL_URL", portal_url).rstrip("/")
    if configured_portal_url not in {
        portal_url,
        f"http://localhost:{portal_port}",
    }:
        raise RuntimeError(
            "Deployment MOCK_PORTAL_URL must target the private loopback mock portal."
        )
    os.environ["MOCK_PORTAL_HOST"] = "127.0.0.1"
    os.environ["MOCK_PORTAL_PORT"] = str(portal_port)

    # The private portal and public API are separate processes. Only the public
    # API binds Render's PORT; the portal remains unreachable from the internet.
    portal_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "mock_portal.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(portal_port),
            "--workers",
            "1",
        ]
    )
    try:
        _wait_for_mock_portal(portal_process, portal_url + "/")
        from actuation.hitl_manager import app

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=public_port,
            workers=1,
            proxy_headers=True,
            forwarded_allow_ips="*",
            log_level="info",
        )
    finally:
        _stop_process(portal_process)


if __name__ == "__main__":
    try:
        run()
    except (OSError, RuntimeError) as exc:
        raise SystemExit(f"Deployment startup failed: {exc}") from exc
