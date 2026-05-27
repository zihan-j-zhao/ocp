"""pynvml-backed monitor. Refuses to manage MIG-enabled cards (see PLAN §2)."""
from __future__ import annotations

import logging

from .base import GpuSample

log = logging.getLogger(__name__)


class NvmlOpenError(RuntimeError):
    pass


class NvmlMonitor:
    def __init__(self) -> None:
        self._pynvml = None
        self._handles: dict[int, object] = {}
        self._mig_indices: set[int] = set()
        self._total_mb: dict[int, int] = {}

    def open(self) -> None:
        try:
            import pynvml
        except ImportError as e:
            raise NvmlOpenError(f"pynvml not installed: {e}") from e
        try:
            pynvml.nvmlInit()
        except Exception as e:  # noqa: BLE001 - NVML may raise its own type
            raise NvmlOpenError(f"NVML init failed: {e}") from e
        self._pynvml = pynvml

    def close(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            self._pynvml = None
        self._handles.clear()

    def _handle(self, gpu: int):
        if self._pynvml is None:
            raise RuntimeError("NVML not opened")
        h = self._handles.get(gpu)
        if h is not None:
            return h
        h = self._pynvml.nvmlDeviceGetHandleByIndex(gpu)
        # Refuse MIG-enabled cards (out of scope).
        try:
            current, _pending = self._pynvml.nvmlDeviceGetMigMode(h)
            if current == self._pynvml.NVML_DEVICE_MIG_ENABLE:
                self._mig_indices.add(gpu)
                raise RuntimeError(
                    f"gpu {gpu} has MIG enabled; OCP does not manage MIG devices"
                )
        except Exception as e:  # NVMLError or AttributeError on older bindings
            if "NotSupported" in str(e) or "AttributeError" in repr(type(e)):
                pass
            elif "MIG enabled" in str(e):
                raise
            else:
                log.debug("gpu %d MIG query failed (ignored): %s", gpu, e)
        self._handles[gpu] = h
        return h

    def enumerate(self) -> list[int]:
        if self._pynvml is None:
            return []
        try:
            n = self._pynvml.nvmlDeviceGetCount()
        except Exception as e:  # noqa: BLE001
            log.warning("nvmlDeviceGetCount failed: %s", e)
            return []
        out: list[int] = []
        for i in range(n):
            try:
                self._handle(i)
            except RuntimeError:
                continue
            out.append(i)
        return out

    def total_mem_mb(self, gpu: int) -> int:
        cached = self._total_mb.get(gpu)
        if cached is not None:
            return cached
        info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle(gpu))
        total = int(info.total // (1024 * 1024))
        self._total_mb[gpu] = total
        return total

    def sample(self, gpu: int) -> GpuSample:
        try:
            h = self._handle(gpu)
            util = self._pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                procs = self._pynvml.nvmlDeviceGetComputeRunningProcesses_v3(h)
            except AttributeError:
                procs = self._pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            pids = tuple(int(p.pid) for p in procs)
            return GpuSample(
                gpu=gpu,
                util_pct=int(util.gpu),
                mem_used_mb=int(mem.used // (1024 * 1024)),
                mem_total_mb=int(mem.total // (1024 * 1024)),
                compute_pids=pids,
            )
        except Exception as e:  # noqa: BLE001
            return GpuSample(
                gpu=gpu, util_pct=0, mem_used_mb=0,
                mem_total_mb=0, compute_pids=(), error=str(e),
            )
