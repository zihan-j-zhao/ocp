"""Spawn / supervise ocp-worker subprocesses, one per idle GPU."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

log = logging.getLogger(__name__)


@dataclass
class WorkerProc:
    gpu: int
    pid: int
    started_at: float


OnExit = Callable[[int, int], Awaitable[None]]


class WorkerManager:
    """Per-GPU subprocess pool.

    Not thread-safe and not asyncio-safe to mutate concurrently: the controller
    serializes calls under its lock.
    """

    def __init__(self, worker_cfg, on_exit: OnExit | None = None):
        self._cfg = worker_cfg
        self._procs: dict[int, subprocess.Popen] = {}
        self._meta: dict[int, WorkerProc] = {}
        self._on_exit = on_exit
        self._watchers: dict[int, asyncio.Task] = {}

    def replace_config(self, worker_cfg) -> None:
        self._cfg = worker_cfg

    def get(self, gpu: int) -> WorkerProc | None:
        return self._meta.get(gpu)

    def running_gpus(self) -> list[int]:
        return list(self._procs)

    def pid_for(self, gpu: int) -> int | None:
        m = self._meta.get(gpu)
        return m.pid if m else None

    def our_pids(self) -> set[int]:
        return {m.pid for m in self._meta.values()}

    def spawn(self, gpu: int) -> WorkerProc:
        if gpu in self._procs:
            return self._meta[gpu]
        argv = self._build_argv(gpu)
        log.info("spawning worker gpu=%d: %s", gpu, " ".join(argv))
        proc = subprocess.Popen(
            argv,
            stdout=sys.stderr,
            stderr=sys.stderr,
            close_fds=True,
            start_new_session=True,
        )
        self._procs[gpu] = proc
        meta = WorkerProc(gpu=gpu, pid=proc.pid, started_at=time.time())
        self._meta[gpu] = meta
        self._watchers[gpu] = asyncio.create_task(
            self._watch(gpu, proc), name=f"watch-worker-{gpu}"
        )
        return meta

    async def stop(self, gpu: int, timeout: float = 5.0) -> None:
        proc = self._procs.get(gpu)
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(_wait_proc(proc), timeout=timeout)
            except asyncio.TimeoutError:
                log.warning("worker gpu=%d did not exit on SIGTERM; killing", gpu)
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await _wait_proc(proc)
        self._cleanup(gpu)

    async def stop_all(self, timeout: float = 5.0) -> None:
        await asyncio.gather(
            *(self.stop(g, timeout) for g in list(self._procs)),
            return_exceptions=True,
        )

    def _cleanup(self, gpu: int) -> None:
        self._procs.pop(gpu, None)
        self._meta.pop(gpu, None)
        task = self._watchers.pop(gpu, None)
        if task is not None and not task.done():
            task.cancel()

    async def _watch(self, gpu: int, proc: subprocess.Popen) -> None:
        try:
            rc = await _wait_proc(proc)
        except asyncio.CancelledError:
            return
        # Still in our map => exit was not requested by stop().
        if self._procs.get(gpu) is proc:
            log.warning("worker gpu=%d exited unexpectedly rc=%d", gpu, rc)
            self._cleanup(gpu)
            if self._on_exit is not None:
                try:
                    await self._on_exit(gpu, rc)
                except Exception:
                    log.exception("on_exit callback failed")

    def _build_argv(self, gpu: int) -> list[str]:
        cfg = self._cfg
        argv = [
            sys.executable, "-m", "ocp.worker.main",
            "--gpu", str(gpu),
            "--workloads", ",".join(cfg.workloads),
            "--util-target", str(cfg.util_target),
            "--mem-noise-frac", str(cfg.mem_noise_frac),
            "--mem-noise-period-s", str(cfg.mem_noise_period_s),
            "--util-noise-frac", str(cfg.util_noise_frac),
            "--util-noise-period-s", str(cfg.util_noise_period_s),
        ]
        if cfg.mem_mb is not None:
            argv += ["--mem-mb", str(cfg.mem_mb)]
        else:
            argv += ["--mem-frac", str(cfg.mem_frac)]
        if cfg.nice is not None:
            argv += ["--nice", str(cfg.nice)]
        return argv


async def _wait_proc(proc: subprocess.Popen, poll_s: float = 0.1) -> int:
    while True:
        rc = proc.poll()
        if rc is not None:
            return rc
        await asyncio.sleep(poll_s)
