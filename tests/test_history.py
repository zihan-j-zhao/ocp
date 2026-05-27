import asyncio
import pytest

from ocp.history import HistoryStore


pytestmark = pytest.mark.asyncio


async def _drain(h: HistoryStore) -> None:
    """Wait until the writer queue is empty (cooperative)."""
    while not h._queue.empty():  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(0.01)
    # Allow one more cycle for the awaited get() inside the writer.
    await asyncio.sleep(0.01)


async def test_append_query(tmp_path):
    h = HistoryStore(tmp_path / "h.jsonl", max_entries=100)
    h.load()
    h.start_writer()
    try:
        h.record(uid=1001, user="alice", cmd="pause", args={"gpu": 0})
        h.record(uid=0, user="(auto)", cmd="worker_spawned", args={"gpu": 0})
        await _drain(h)
    finally:
        await h.stop_writer()
    entries = h.query(n=10)
    assert [e.cmd for e in entries] == ["pause", "worker_spawned"]


async def test_compaction_bounds_disk(tmp_path):
    p = tmp_path / "h.jsonl"
    h = HistoryStore(p, max_entries=10)
    h.load()
    h.start_writer()
    try:
        for i in range(50):
            h.record(uid=0, user="(auto)", cmd="t", args={"i": i})
        await _drain(h)
    finally:
        await h.stop_writer()
    # Ring holds the last 10.
    assert len(h.query()) == 10
    # On-disk file is bounded by ~max_entries * 1.2.
    on_disk = sum(1 for _ in p.read_text().splitlines())
    assert on_disk <= 12


async def test_reload_drops_oldest(tmp_path):
    p = tmp_path / "h.jsonl"
    h = HistoryStore(p, max_entries=5)
    h.load()
    h.start_writer()
    try:
        for i in range(20):
            h.record(uid=0, user="(auto)", cmd="t", args={"i": i})
        await _drain(h)
    finally:
        await h.stop_writer()
    h2 = HistoryStore(p, max_entries=5)
    h2.load()
    entries = h2.query()
    assert len(entries) == 5
    assert entries[-1].args["i"] == 19


async def test_filter_mine(tmp_path):
    h = HistoryStore(tmp_path / "h.jsonl", max_entries=100)
    h.load()
    h.start_writer()
    try:
        h.record(uid=1001, user="alice", cmd="pause", args={})
        h.record(uid=2002, user="bob", cmd="pause", args={})
        h.record(uid=0, user="(auto)", cmd="worker_spawned", args={})
        await _drain(h)
    finally:
        await h.stop_writer()
    mine = h.query(uid=1001)
    assert [e.user for e in mine] == ["alice"]
