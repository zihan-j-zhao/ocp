"""Monitor interface and sample dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GpuSample:
    gpu: int
    util_pct: int
    mem_used_mb: int
    mem_total_mb: int
    compute_pids: tuple[int, ...] = ()
    error: str | None = None

    @property
    def mem_used_pct(self) -> float:
        if self.mem_total_mb <= 0:
            return 0.0
        return 100.0 * self.mem_used_mb / self.mem_total_mb


class GpuMonitor(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def enumerate(self) -> list[int]: ...
    def sample(self, gpu: int) -> GpuSample: ...
    def total_mem_mb(self, gpu: int) -> int: ...
