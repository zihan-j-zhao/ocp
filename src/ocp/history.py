"""Append-only JSONL history with bounded-cap compaction + in-memory ring."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

COMPACT_OVERSHOOT = 1.2


@dataclass
class HistoryEntry:
    ts: float
    uid: int
    user: str
    cmd: str
    args: dict[str, Any]
    ok: bool = True
    error: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HistoryEntry":
        return cls(
            ts=float(d["ts"]),
            uid=int(d["uid"]),
            user=str(d["user"]),
            cmd=str(d["cmd"]),
            args=d.get("args") or {},
            ok=bool(d.get("ok", True)),
            error=d.get("error"),
            note=d.get("note"),
        )


class HistoryStore:
    """In-memory ring backed by an append-only file.

    The daemon is the sole writer (a single asyncio task drains a queue), so no
    inter-process locking is needed.
    """

    def __init__(self, path: Path, max_entries: int):
        self._path = Path(path)
        self._max = int(max_entries)
        self._ring: deque[HistoryEntry] = deque(maxlen=self._max)
        self._lines_on_disk = 0
        self._queue: asyncio.Queue[HistoryEntry | None] = asyncio.Queue()
        self._writer_task: asyncio.Task | None = None

    @property
    def max_entries(self) -> int:
        return self._max

    def set_max_entries(self, n: int) -> None:
        if n < 10:
            raise ValueError("max_entries must be >= 10")
        self._ring = deque(self._ring, maxlen=n)
        self._max = n

    def load(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch(mode=0o644)
            self._lines_on_disk = 0
            return
        loaded: list[HistoryEntry] = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    loaded.append(HistoryEntry.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                    log.warning("dropping malformed history line: %s", e)
        self._lines_on_disk = len(loaded)
        if len(loaded) > self._max:
            loaded = loaded[-self._max:]
        self._ring.clear()
        self._ring.extend(loaded)

    def start_writer(self) -> asyncio.Task:
        assert self._writer_task is None, "writer already started"
        self._writer_task = asyncio.create_task(
            self._run_writer(), name="history-writer"
        )
        return self._writer_task

    async def stop_writer(self) -> None:
        if self._writer_task is None:
            return
        await self._queue.put(None)
        try:
            await self._writer_task
        finally:
            self._writer_task = None

    def enqueue(self, entry: HistoryEntry) -> None:
        """Non-blocking; caller should be inside the asyncio loop."""
        self._queue.put_nowait(entry)

    def record(
        self,
        *,
        uid: int,
        user: str,
        cmd: str,
        args: dict[str, Any] | None = None,
        ok: bool = True,
        error: str | None = None,
        note: str | None = None,
        ts: float | None = None,
    ) -> None:
        self.enqueue(
            HistoryEntry(
                ts=ts if ts is not None else time.time(),
                uid=uid, user=user, cmd=cmd, args=args or {},
                ok=ok, error=error, note=note,
            )
        )

    def query(
        self,
        *,
        n: int | None = None,
        since_s: float | None = None,
        user: str | None = None,
        uid: int | None = None,
    ) -> list[HistoryEntry]:
        items = list(self._ring)
        if since_s is not None:
            cutoff = time.time() - since_s
            items = [e for e in items if e.ts >= cutoff]
        if user is not None:
            items = [e for e in items if e.user == user]
        if uid is not None:
            items = [e for e in items if e.uid == uid]
        if n is not None and n > 0:
            items = items[-n:]
        return items

    # --- internals ---------------------------------------------------------

    async def _run_writer(self) -> None:
        try:
            while True:
                entry = await self._queue.get()
                if entry is None:
                    break
                self._ring.append(entry)
                try:
                    self._append_to_disk(entry)
                except OSError as e:
                    log.error("history write failed: %s", e)
                if self._lines_on_disk > int(self._max * COMPACT_OVERSHOOT):
                    try:
                        self._compact()
                    except OSError as e:
                        log.error("history compaction failed: %s", e)
        finally:
            # Drain any leftover items so a graceful shutdown doesn't lose them.
            while not self._queue.empty():
                entry = self._queue.get_nowait()
                if entry is None:
                    continue
                self._ring.append(entry)
                try:
                    self._append_to_disk(entry)
                except OSError:
                    pass

    def _append_to_disk(self, entry: HistoryEntry) -> None:
        line = json.dumps(entry.to_dict(), separators=(",", ":")) + "\n"
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)
        self._lines_on_disk += 1

    def _compact(self) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for entry in self._ring:
                f.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, self._path)
        self._lines_on_disk = len(self._ring)


def format_relative(ts: float, now: float | None = None) -> str:
    now = now if now is not None else time.time()
    delta = int(now - ts)
    if delta < 5:
        return "now"
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"
