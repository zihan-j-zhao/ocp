import asyncio
import pytest

from ocp.config import Config, ThresholdsCfg, WorkerCfg
from ocp.controller import Controller
from ocp.history import HistoryStore
from ocp.lease import LeaseStore
from ocp.monitor.base import GpuSample


pytestmark = pytest.mark.asyncio


class FakeMonitor:
    def __init__(self, samples_by_gpu: dict[int, list[GpuSample]]):
        self._samples = samples_by_gpu
        self._idx = {g: 0 for g in samples_by_gpu}

    def open(self): pass
    def close(self): pass
    def enumerate(self): return list(self._samples)
    def total_mem_mb(self, gpu): return 80000

    def sample(self, gpu):
        i = self._idx[gpu]
        out = self._samples[gpu][min(i, len(self._samples[gpu]) - 1)]
        self._idx[gpu] = i + 1
        return out


class FakeWorkerManager:
    def __init__(self):
        self.spawned: list[int] = []
        self.stopped: list[int] = []
        self._meta: dict[int, int] = {}

    def replace_config(self, cfg): pass
    def get(self, gpu): return None
    def running_gpus(self): return list(self._meta)
    def our_pids(self): return set(self._meta.values())

    def pid_for(self, gpu):
        return self._meta.get(gpu)

    def spawn(self, gpu):
        self.spawned.append(gpu)
        self._meta[gpu] = 99000 + gpu
        return None

    async def stop(self, gpu, timeout=5.0):
        self.stopped.append(gpu)
        self._meta.pop(gpu, None)

    async def stop_all(self, timeout=5.0):
        for g in list(self._meta):
            await self.stop(g)


def _ctrl(tmp_path, cfg, monitor, *, hist=None) -> tuple[Controller, FakeWorkerManager, HistoryStore]:
    leases = LeaseStore(tmp_path / "state.json")
    hist = hist or HistoryStore(tmp_path / "h.jsonl", max_entries=100)
    hist.load()
    wm = FakeWorkerManager()
    # Use a fake monotonic clock so debounce is fully deterministic.
    clock = [0.0]
    def t(): return clock[0]
    ctrl = Controller(cfg=cfg, monitor=monitor, leases=leases,
                      workers=wm, history=hist, time_fn=t)
    ctrl._fake_clock = clock  # type: ignore[attr-defined]
    return ctrl, wm, hist


def _cfg(*, idle_debounce_s=4, busy_debounce_s=2) -> Config:
    cfg = Config()
    cfg.thresholds = ThresholdsCfg(
        util_low=10, util_high=80, mem_low=20, mem_high=80,
        idle_debounce_s=idle_debounce_s, busy_debounce_s=busy_debounce_s,
    )
    cfg.worker = WorkerCfg()
    return cfg


async def _advance(ctrl: Controller, seconds: float, ticks: int = 1) -> None:
    clock = ctrl._fake_clock  # type: ignore[attr-defined]
    step = seconds / ticks
    for _ in range(ticks):
        clock[0] += step
        await ctrl.tick_once()
        # Yield so tasks scheduled by the controller (e.g. _schedule_stop) run.
        await asyncio.sleep(0)


async def test_spawn_after_idle_debounce(tmp_path):
    idle = GpuSample(gpu=0, util_pct=0, mem_used_mb=100, mem_total_mb=80000)
    mon = FakeMonitor({0: [idle]})
    ctrl, wm, hist = _ctrl(tmp_path, _cfg(idle_debounce_s=4), mon)
    hist.start_writer()
    try:
        ctrl.set_watched([0])
        # First tick establishes the baseline; dt=0 so streak doesn't grow.
        await _advance(ctrl, 0.0)
        assert wm.spawned == []
        # Three more ticks of 2s each = 6s idle, well past debounce.
        for _ in range(3):
            await _advance(ctrl, 2.0)
        assert wm.spawned == [0]
    finally:
        await hist.stop_writer()


async def test_no_spawn_when_busy(tmp_path):
    busy = GpuSample(
        gpu=0, util_pct=50, mem_used_mb=40000, mem_total_mb=80000,
        compute_pids=(11111,),
    )
    mon = FakeMonitor({0: [busy]})
    ctrl, wm, hist = _ctrl(tmp_path, _cfg(), mon)
    hist.start_writer()
    try:
        ctrl.set_watched([0])
        for _ in range(5):
            await _advance(ctrl, 2.0)
        assert wm.spawned == []
    finally:
        await hist.stop_writer()


async def test_yield_on_foreign_pid(tmp_path):
    idle = GpuSample(gpu=0, util_pct=0, mem_used_mb=100, mem_total_mb=80000)
    busy = GpuSample(
        gpu=0, util_pct=80, mem_used_mb=40000, mem_total_mb=80000,
        compute_pids=(12345,),
    )
    # First idle so we spawn; then a foreign PID arrives.
    mon = FakeMonitor({0: [idle, idle, idle, idle, busy, busy, busy]})
    ctrl, wm, hist = _ctrl(tmp_path, _cfg(idle_debounce_s=2, busy_debounce_s=1), mon)
    hist.start_writer()
    try:
        ctrl.set_watched([0])
        for _ in range(8):
            await _advance(ctrl, 1.0)
        assert wm.spawned == [0]
        assert 0 in wm.stopped
    finally:
        await hist.stop_writer()


async def test_pause_prevents_spawn(tmp_path):
    idle = GpuSample(gpu=0, util_pct=0, mem_used_mb=100, mem_total_mb=80000)
    mon = FakeMonitor({0: [idle]})
    ctrl, wm, hist = _ctrl(tmp_path, _cfg(idle_debounce_s=1), mon)
    hist.start_writer()
    try:
        ctrl.set_watched([0])
        await ctrl.pause(uid=1001, user="alice", gpus=[0], duration_s=60)
        for _ in range(5):
            await _advance(ctrl, 2.0)
        assert wm.spawned == []
    finally:
        await hist.stop_writer()


async def test_pause_tears_down_running_worker(tmp_path):
    idle = GpuSample(gpu=0, util_pct=0, mem_used_mb=100, mem_total_mb=80000)
    mon = FakeMonitor({0: [idle]})
    ctrl, wm, hist = _ctrl(tmp_path, _cfg(idle_debounce_s=1), mon)
    hist.start_writer()
    try:
        ctrl.set_watched([0])
        for _ in range(3):
            await _advance(ctrl, 2.0)
        assert wm.spawned == [0]
        # Now pause: the worker should be torn down.
        await ctrl.pause(uid=1001, user="alice", gpus=[0], duration_s=60)
        await asyncio.sleep(0)  # let _schedule_stop's task run
        assert 0 in wm.stopped
    finally:
        await hist.stop_writer()
