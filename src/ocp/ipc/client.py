"""Synchronous UDS client used by the CLI."""
from __future__ import annotations

import json
import socket
import struct
from pathlib import Path
from typing import Any

from .. import paths
from .protocol import MAX_BODY_BYTES, PROTOCOL_VERSION


class IPCClientError(Exception):
    pass


class DaemonDown(IPCClientError):
    pass


def call(
    cmd: str,
    args: dict[str, Any] | None = None,
    socket_path: Path | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Send a single request and return the parsed response dict.

    Raises DaemonDown if the daemon is not reachable; IPCClientError otherwise.
    """
    sock_path = socket_path or paths.SOCKET_PATH
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        try:
            s.connect(str(sock_path))
        except (FileNotFoundError, ConnectionRefusedError) as e:
            raise DaemonDown(f"daemon not reachable at {sock_path}: {e}") from e
        body = json.dumps(
            {"v": PROTOCOL_VERSION, "id": "cli", "cmd": cmd, "args": args or {}},
            separators=(",", ":"),
        ).encode()
        s.sendall(struct.pack(">I", len(body)) + body)
        header = _recv_exact(s, 4)
        (length,) = struct.unpack(">I", header)
        if length == 0 or length > MAX_BODY_BYTES:
            raise IPCClientError(f"bad response frame length: {length}")
        resp_body = _recv_exact(s, length)
        try:
            return json.loads(resp_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise IPCClientError(f"bad response JSON: {e}") from e
    finally:
        s.close()


def _recv_exact(s: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise IPCClientError("connection closed mid-frame")
        buf.extend(chunk)
    return bytes(buf)
