"""Per-device decision policy with per-GPU lease integration."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from .config import Config
from .history import HistoryStore
from .lease import LeaseError, LeaseStore
from .monitor.base import GpuSample

log = logging.getLogger(__name__)


@dataclass
class DeviceState:
    gpu: int
    idle_streak_s: float = 0.0
    busy_streak_s: float = 0.0
    last_sample: GpuSample | None = None
    last_decision: str = "init"


class Controller:
    def __init__(
        self,
        *,
        cfg: Config,
        monitor,
        leases: LeaseStore,
        workers,
        history: HistoryStore,
        time_fn=time.monotonic,
    ):
        self._cfg = cfg
        self._monitor = monitor
        self._leases = leases
        self._workers = workers
        self._history = history
        self._time = time_fn
        self._states: dict[int, DeviceState] = {}
        self._lock = asyncio.Lock()
        self._watched: list[int] = []
        self._last_tick_at: float | None = None

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    @property
    def watched(self) -> list[int]:
        return list(self._watched)

    def state(self, gpu: int) -> DeviceState | None:
        return self._states.get(gpu)

    def states(self) -> dict[int, DeviceState]:
        return dict(self._states)

    def set_watched(self, gpus: list[int]) -> None:
        self._watched = list(gpus)
        for g in gpus:
            self._states.setdefault(g, DeviceState(gpu=g))
        for g in list(self._states):
            if g not in gpus:
                self._states.pop(g, None)

    def replace_config(self, cfg: Config) -> None:
        self._cfg = cfg

    async def tick_once(self) -> None:
        """Sample each watched GPU and apply policy. Holds the controller lock."""
        async with self._lock:
            now = self._time()
            wall_now = time.time()
            dt = (now - self._last_tick_at) if self._last_tick_at is not None else 0.0
            self._last_tick_at = now

            for expired in self._leases.sweep_expired(now=wall_now):
                self._history.record(
                    uid=0, user="(auto)", cmd="lease_expired",
                    args={"gpu": expired.gpu},
                    note=f"holder was {expired.user}",
                )
            for gpu in self._watched:
                try:
                    sample = self._monitor.sample(gpu)
                except Exception as e:  # noqa: BLE001
                    log.warning("monitor sample failed gpu=%d: %s", gpu, e)
                    self._history.record(
                        uid=0, user="(auto)", cmd="nvml_error",
                        args={"gpu": gpu}, ok=False, error=str(e),
                    )
                    continue
                self._apply(gpu, sample, dt)

    def _apply(self, gpu: int, sample: GpuSample, dt: float) -> None:
        st = self._states.setdefault(gpu, DeviceState(gpu=gpu))
        st.last_sample = sample

        if sample.error:
            st.last_decision = "monitor_error"
            return

        cfg = self._cfg
        held = self._leases.covers(gpu)
        our_pid = self._workers.pid_for(gpu)
        foreign = [p for p in sample.compute_pids if p != our_pid]

        if held:
            if our_pid is not None:
                # Defensive: pause() should have torn this down already.
                log.warning(
                    "worker present on gpu=%d under active lease; tearing down", gpu
                )
                self._schedule_stop(gpu)
            st.idle_streak_s = 0.0
            st.busy_streak_s = 0.0
            st.last_decision = "leased"
            return

        if foreign:
            st.busy_streak_s += dt
            st.idle_streak_s = 0.0
            if (
                our_pid is not None
                and st.busy_streak_s >= cfg.thresholds.busy_debounce_s
            ):
                self._schedule_stop(gpu)
                self._history.record(
                    uid=0, user="(auto)", cmd="worker_yielded",
                    args={"gpu": gpu, "foreign_pids": foreign[:8]},
                )
                st.last_decision = "yielded"
            else:
                st.last_decision = "external"
            return

        # No foreign PIDs.
        st.busy_streak_s = 0.0
        is_quiet = (
            sample.util_pct < cfg.thresholds.util_low
            and sample.mem_used_pct < cfg.thresholds.mem_low
        )
        if our_pid is None:
            if is_quiet:
                st.idle_streak_s += dt
            else:
                st.idle_streak_s = 0.0
            if (
                st.idle_streak_s >= cfg.thresholds.idle_debounce_s
                and cfg.worker.workloads
            ):
                self._workers.spawn(gpu)
                self._history.record(
                    uid=0, user="(auto)", cmd="worker_spawned",
                    args={"gpu": gpu},
                    note=f"idle {int(st.idle_streak_s)}s",
                )
                st.idle_streak_s = 0.0
                st.last_decision = "spawned"
            else:
                st.last_decision = "waiting_idle" if is_quiet else "active_no_worker"
        else:
            st.last_decision = "occupied_by_us"

    def _schedule_stop(self, gpu: int) -> None:
        asyncio.create_task(self._workers.stop(gpu), name=f"stop-worker-{gpu}")

    # --- IPC-facing operations --------------------------------------------

    async def pause(self, *, uid: int, user: str, gpus: list[int], duration_s: int):
        """Acquire-or-extend per-GPU leases. Tears down our workers on covered GPUs."""
        async with self._lock:
            unknown = [g for g in gpus if g not in self._watched]
            if unknown:
                raise LeaseError(f"unknown gpus: {unknown}")
            leases = self._leases.acquire(
                uid=uid, user=user, gpus=gpus, duration_s=duration_s
            )
            for g in gpus:
                if self._workers.pid_for(g) is not None:
                    self._schedule_stop(g)
            return leases

    async def resume(self, *, uid: int, gpus: list[int], is_root: bool):
        async with self._lock:
            return self._leases.release(uid=uid, gpus=gpus, is_root=is_root)
