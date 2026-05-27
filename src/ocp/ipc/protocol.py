"""Wire protocol: framing + request/response dataclasses."""
from __future__ import annotations

import asyncio
import json
import struct
import uuid
from dataclasses import dataclass, field
from typing import Any


PROTOCOL_VERSION = 1
MAX_BODY_BYTES = 1 << 20  # 1 MiB


class ProtocolError(Exception):
    pass


@dataclass
class Request:
    cmd: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    v: int = PROTOCOL_VERSION

    def to_bytes(self) -> bytes:
        return _encode({"v": self.v, "id": self.id, "cmd": self.cmd, "args": self.args})

    @classmethod
    def from_bytes(cls, body: bytes) -> "Request":
        d = _decode(body)
        if not isinstance(d, dict) or "cmd" not in d:
            raise ProtocolError("request missing cmd")
        v = int(d.get("v", PROTOCOL_VERSION))
        if v != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol version: {v}")
        args = d.get("args") or {}
        if not isinstance(args, dict):
            raise ProtocolError("request.args must be an object")
        return cls(
            cmd=str(d["cmd"]),
            args=args,
            id=str(d.get("id") or uuid.uuid4()),
            v=v,
        )


@dataclass
class Response:
    id: str
    ok: bool
    data: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    v: int = PROTOCOL_VERSION

    @classmethod
    def success(cls, req_id: str, data: dict[str, Any] | None = None) -> "Response":
        return cls(id=req_id, ok=True, data=data or {})

    @classmethod
    def failure(
        cls,
        req_id: str,
        code: str,
        msg: str,
        data: dict[str, Any] | None = None,
    ) -> "Response":
        err: dict[str, Any] = {"code": code, "msg": msg}
        if data:
            err["data"] = data
        return cls(id=req_id, ok=False, error=err)

    def to_bytes(self) -> bytes:
        out: dict[str, Any] = {"v": self.v, "id": self.id, "ok": self.ok}
        if self.ok:
            out["data"] = self.data or {}
        else:
            out["error"] = self.error or {}
        return _encode(out)

    @classmethod
    def from_bytes(cls, body: bytes) -> "Response":
        d = _decode(body)
        if not isinstance(d, dict) or "ok" not in d:
            raise ProtocolError("response missing ok")
        return cls(
            id=str(d.get("id", "")),
            ok=bool(d["ok"]),
            data=d.get("data"),
            error=d.get("error"),
            v=int(d.get("v", PROTOCOL_VERSION)),
        )


def _encode(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def _decode(body: bytes) -> dict:
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"invalid JSON: {e}") from e


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    header = await reader.readexactly(4)
    (length,) = struct.unpack(">I", header)
    if length == 0 or length > MAX_BODY_BYTES:
        raise ProtocolError(f"frame length out of range: {length}")
    return await reader.readexactly(length)


async def write_frame(writer: asyncio.StreamWriter, body: bytes) -> None:
    if len(body) > MAX_BODY_BYTES:
        raise ProtocolError(f"body too large: {len(body)}")
    writer.write(struct.pack(">I", len(body)) + body)
    await writer.drain()
