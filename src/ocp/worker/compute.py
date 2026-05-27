"""Compute controller: BF16 gemm loop with size-noise and a duty-cycle dial.

Util target is reached by varying the work/sleep duty cycle of the gemm loop;
the matrix size is jittered around a base value to keep the workload realistic
and to ensure time-averaged util exceeds the configured target (PLAN §6.3).
"""
from __future__ import annotations

import asyncio
import logging
import random
import sys
import time

log = logging.getLogger(__name__)

_BASE_SIZE = 4096
_CYCLE_S = 0.1


class ComputeController:
    def __init__(self, *, gpu: int, util_target: int, noise_frac: float, period_s: float):
        self.gpu = gpu
        self.util_target = max(0, min(100, util_target))
        self.noise_frac = max(0.0, noise_frac)
        self.period_s = max(0.5, period_s)

    def _random_size(self, base: int) -> int:
        if self.noise_frac == 0:
            return base
        upper = int(base * (1.0 + 2.0 * self.noise_frac))
        return random.randint(base, max(base, upper))

    async def run(self, stop: asyncio.Event) -> None:
        try:
            import torch
        except ImportError:
            log.error("torch not installed; compute controller cannot run")
            return
        if not torch.cuda.is_available():
            log.error("torch reports no CUDA; compute controller cannot run")
            return

        device = torch.device("cuda", 0)
        if self.util_target <= 0:
            await stop.wait()
            return

        dtype = torch.bfloat16
        size = _BASE_SIZE
        try:
            a = torch.randn(size, size, dtype=dtype, device=device)
            b = torch.randn(size, size, dtype=dtype, device=device)
        except RuntimeError as e:
            log.error("compute controller initial alloc failed: %s", e)
            return

        # Mean util is target * 1.05 to stay strictly above the dial.
        duty = max(0.05, min(1.0, (self.util_target / 100.0) * 1.05))
        last_reshape = time.monotonic()
        step = 0
        try:
            while not stop.is_set():
                now = time.monotonic()
                if now - last_reshape >= self.period_s:
                    new_size = self._random_size(_BASE_SIZE)
                    try:
                        a = torch.randn(new_size, new_size, dtype=dtype, device=device)
                        b = torch.randn(new_size, new_size, dtype=dtype, device=device)
                        size = new_size
                    except RuntimeError as e:
                        log.warning("compute reshape to %d failed: %s", new_size, e)
                    last_reshape = now

                work_end = time.monotonic() + _CYCLE_S * duty
                while time.monotonic() < work_end and not stop.is_set():
                    _ = torch.matmul(a, b)

                sleep_for = max(0.0, _CYCLE_S * (1.0 - duty))
                if sleep_for > 0:
                    try:
                        torch.cuda.synchronize(device)
                    except Exception:
                        pass
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=sleep_for)
                    except asyncio.TimeoutError:
                        pass

                step += 1
                if step % 50 == 0:
                    print(
                        f"util: duty={duty:.2f} size={size}",
                        file=sys.stderr, flush=True,
                    )
        finally:
            del a, b
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
