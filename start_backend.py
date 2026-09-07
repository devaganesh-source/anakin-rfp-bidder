"""Start the local HITL FastAPI service after reclaiming its Windows port.

The mock portal and the HITL API are separate local services. The portal uses
``MOCK_PORTAL_PORT`` (8000 by default); this runner owns ``HITL_PORT`` (8001
by default). Only a Python/Uvicorn listener is eligible for cleanup. An
unrelated process is never force-stopped.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import time
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from actuation.hitl_manager import app


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
PORT_RELEASE_TIMEOUT_SECONDS = 5.0


def _port_is_bound(host: str, port: int) -> bool:
    """Return whether the configured bind address is currently unavailable."""
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    for family, socket_type, protocol, _, address in addresses:
        with socket.socket(family, socket_type, protocol) as probe:
            try:
                probe.bind(address)
            except OSError:
                return True
    return False


def _listener_pids_windows(port: int) -> set[int]:
    """Read LISTENING TCP owners without requiring third-party packages."""
    result = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not inspect TCP listeners: {result.stderr.strip()}")

    pids: set[int] = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0].upper() != "TCP":
            continue
        local_address, state, pid_text = fields[1], fields[-2], fields[-1]
        if state.upper() != "LISTENING" or not local_address.rsplit(":", 1)[-1] == str(port):
            continue
        try:
            pids.add(int(pid_text))
        except ValueError:
            continue
    return pids


def _process_name_windows(pid: int) -> str:
    """Return one process image name using same-user PowerShell inspection."""
    result = subprocess.run(
        [
            "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
            f"(Get-Process -Id {pid} -ErrorAction Stop).ProcessName",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not inspect process {pid}: {result.stderr.strip()}")
    return result.stdout.strip().splitlines()[-1].lower() if result.stdout.strip() else ""


def _is_killable_process(name: str) -> bool:
    """Recognize Python and Uvicorn image names with or without .exe."""
    normalized = name.lower().removesuffix(".exe")
    return normalized in {"uvicorn", "python", "pythonw", "py"} or normalized.startswith("python")


def ensure_port_available(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    release_timeout: float = PORT_RELEASE_TIMEOUT_SECONDS,
) -> None:
    """Reclaim a stale local Python/Uvicorn listener or fail clearly.

    Port ownership is checked before Uvicorn starts. On Windows, all owners
    must be eligible Python/Uvicorn processes before any one is stopped; this
    avoids partially changing the machine when an unrelated process is found.
    """
    if not 1 <= port <= 65535:
        raise ValueError("HITL_PORT must be between 1 and 65535.")
    if release_timeout <= 0:
        raise ValueError("release_timeout must be positive.")
    if not _port_is_bound(host, port):
        return
    if os.name != "nt":
        raise RuntimeError(f"Port {port} is already in use; automatic cleanup is supported on Windows only.")

    pids = _listener_pids_windows(port)
    if not pids:
        raise RuntimeError(f"Port {port} is in use, but its listener could not be identified.")
    owners = {pid: _process_name_windows(pid) for pid in pids}
    current_pid = os.getpid()
    unsafe = {
        pid: name for pid, name in owners.items()
        if pid == current_pid or not _is_killable_process(name)
    }
    if unsafe:
        details = ", ".join(f"PID {pid} ({name or 'unknown'})" for pid, name in unsafe.items())
        raise RuntimeError(f"Port {port} is owned by {details}; refusing to stop it.")

    for pid in owners:
        result = subprocess.run(
            [
                "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                f"Stop-Process -Id {pid} -Force -ErrorAction Stop",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Could not stop stale process {pid}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

    deadline = time.monotonic() + release_timeout
    while _port_is_bound(host, port) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _port_is_bound(host, port):
        raise RuntimeError(f"Port {port} did not become available after stopping stale listeners.")


def _configured_port() -> int:
    """Parse the same HITL port setting used by run_bid.py."""
    try:
        return int(os.environ.get("HITL_PORT", str(DEFAULT_PORT)))
    except ValueError as exc:
        raise ValueError("HITL_PORT must be an integer.") from exc


def run() -> None:
    """Load local configuration, reclaim the port, and serve one worker."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    host = os.environ.get("HITL_HOST", DEFAULT_HOST)
    port = _configured_port()
    ensure_port_available(host, port)
    uvicorn.run(app, host=host, port=port, workers=1, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        run()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Backend startup failed: {exc}") from exc
