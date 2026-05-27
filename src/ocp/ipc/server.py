"""UDS server with SO_PEERCRED-based identity extraction."""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import struct
from pathlib import Path
from typing import Awaitable, Callable

from .. import auth
from .protocol import (
    Request,
    Response,
    read_frame,
    write_frame,
    ProtocolError,
)

log = logging.getLogger(__name__)

Handler = Callable[[auth.Caller, Request], Awaitable[Response]]

# Linux SO_PEERCRED: struct ucred { pid_t pid; uid_t uid; gid_t gid; }
_UCRED_FMT = "iII"
_UCRED_SIZE = struct.calcsize(_UCRED_FMT)


def _peercred(sock: socket.socket) -> tuple[int, int, int]:
    data = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _UCRED_SIZE)
    pid, uid, gid = struct.unpack(_UCRED_FMT, data)
    return pid, uid, gid


class UDSServer:
    def __init__(self, socket_path: Path, handler: Handler):
        self._socket_path = Path(socket_path)
        self._handler = handler
        self._server: asyncio.base_events.Server | None = None

    async def start(self) -> None:
        # Remove stale socket if present.
        try:
            self._socket_path.unlink()
        except FileNotFoundError:
            pass
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._server = await asyncio.start_unix_server(
            self._on_client, path=str(self._socket_path)
        )
        # World-connectable; auth is in-daemon.
        os.chmod(self._socket_path, 0o666)
        log.info("IPC listening on %s", self._socket_path)

    async def serve_forever(self) -> None:
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
            self._server = None
        try:
            self._socket_path.unlink()
        except FileNotFoundError:
            pass

    async def _on_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        sock: socket.socket | None = writer.get_extra_info("socket")
        if sock is None:
            writer.close()
            return
        try:
            pid, uid, gid = _peercred(sock)
        except OSError as e:
            log.warning("failed SO_PEERCRED: %s", e)
            writer.close()
            return
        caller = auth.caller_from_creds(pid, uid, gid)
        try:
            try:
                body = await read_frame(reader)
            except (asyncio.IncompleteReadError, ProtocolError) as e:
                log.debug("malformed request from %s: %s", caller.user, e)
                return
            try:
                req = Request.from_bytes(body)
            except ProtocolError as e:
                resp = Response.failure("?", "E_BAD_REQUEST", str(e))
            else:
                try:
                    resp = await self._handler(caller, req)
                except Exception as e:  # noqa: BLE001 - last-resort guard
                    log.exception(
                        "handler crashed cmd=%s user=%s", req.cmd, caller.user
                    )
                    resp = Response.failure(
                        req.id, "E_BAD_REQUEST", f"server error: {e}"
                    )
            try:
                await write_frame(writer, resp.to_bytes())
            except (ConnectionResetError, BrokenPipeError):
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
