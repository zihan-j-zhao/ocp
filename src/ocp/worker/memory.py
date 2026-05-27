"""Memory controller: periodically reallocates a CUDA tensor sized above the mean target.

The size distribution is uniform on `[target, target * (1 + 2 * noise_frac)]`,
so the time-averaged held bytes is `target * (1 + noise_frac)` — strictly above
the configured target. This is the invariant the monitor relies on so it never
sees its own worker as idle (PLAN §6.3).
"""
from __future__ import annotations

import asyncio
import logging
import random
import sys

log = logging.getLogger(__name__)

# Leave headroom for the compute controller and the CUDA caching allocator.
_VRAM_CAP_FRAC = 0.95


class MemoryController:
    def __init__(
        self,
        *,
        gpu: int,
        mem_frac: float | None,
        mem_mb: int | None,
        noise_frac: float,
        period_s: float,
    ):
        self.gpu = gpu
        self.mem_frac = mem_frac
        self.mem_mb = mem_mb
        self.noise_frac = max(0.0, noise_frac)
        self.period_s = max(0.5, period_s)

    def _target_bytes(self, total_bytes: int) -> int:
        if self.mem_mb is not None:
            return self.mem_mb * 1024 * 1024
        frac = self.mem_frac if self.mem_frac is not None else 0.5
        return int(total_bytes * frac)

    def _random_size(self, target_bytes: int, cap_bytes: int) -> int:
        if self.noise_frac == 0:
            return min(target_bytes, cap_bytes)
        upper = min(int(target_bytes * (1.0 + 2.0 * self.noise_frac)), cap_bytes)
        if upper <= target_bytes:
            return min(target_bytes, cap_bytes)
        return random.randint(target_bytes, upper)

    async def run(self, stop: asyncio.Event) -> None:
        try:
            import torch
        except ImportError:
            log.error("torch not installed; memory controller cannot run")
            return
        if not torch.cuda.is_available():
            log.error("torch reports no CUDA; memory controller cannot run")
            return

        device = torch.device("cuda", 0)  # CUDA_VISIBLE_DEVICES already remaps
        _free, total = torch.cuda.mem_get_info(device)
        target = self._target_bytes(total)
        cap = int(total * _VRAM_CAP_FRAC)
        if target > cap:
            log.warning("memory target %.2f GiB exceeds cap %.2f GiB; clamping",
                        target / 2**30, cap / 2**30)
            target = cap

        tensor = None
        try:
            while not stop.is_set():
                size = self._random_size(target, cap)
                tensor = None
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    tensor = torch.empty(size, dtype=torch.uint8, device=device)
                    # Touch the buffer so it actually backs into VRAM (not lazy).
                    tensor[::max(1, size // 64)].fill_(0)
                except RuntimeError as e:
                    log.warning("alloc %d bytes failed: %s; shrinking cap", size, e)
                    cap = max(int(cap * 0.9), target)
                    await _wait(stop, 1.0)
                    continue
                print(
                    f"mem: held {size / (1024**3):.2f} GiB "
                    f"(target {target / (1024**3):.2f} GiB)",
                    file=sys.stderr, flush=True,
                )
                await _wait(stop, self.period_s)
        finally:
            tensor = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


async def _wait(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass
