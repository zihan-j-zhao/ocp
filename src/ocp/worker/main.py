"""ocp-worker entry point.

Sets `OCP::Worker` as the process title, then runs the memory and compute
controllers concurrently in the same process. Both honor SIGTERM/SIGINT.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys


PROC_TITLE = "OCP::Worker"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("ocp-worker")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument(
        "--workloads", default="mem,util",
        help="comma-list subset of {mem, util}",
    )
    mem = p.add_mutually_exclusive_group()
    mem.add_argument("--mem-frac", type=float)
    mem.add_argument("--mem-mb", type=int)
    p.add_argument("--util-target", type=int, default=90)
    p.add_argument("--mem-noise-frac", type=float, default=0.05)
    p.add_argument("--mem-noise-period-s", type=float, default=5.0)
    p.add_argument("--util-noise-frac", type=float, default=0.05)
    p.add_argument("--util-noise-period-s", type=float, default=5.0)
    p.add_argument("--nice", type=int, default=None)
    return p.parse_args(argv)


def _set_proctitle() -> None:
    try:
        from setproctitle import setproctitle
        setproctitle(PROC_TITLE)
    except ImportError:
        pass


def _set_nice(nice: int | None) -> None:
    if nice is None:
        return
    try:
        os.nice(nice)
    except (OSError, ValueError):
        pass


async def _run(args: argparse.Namespace) -> int:
    # Bind to the requested device via CUDA_VISIBLE_DEVICES so subsequent
    # `cuda:0` references point at the right GPU regardless of host topology.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    workloads = {w.strip() for w in args.workloads.split(",") if w.strip()}
    unknown = workloads - {"mem", "util"}
    if unknown:
        print(f"ocp-worker: unknown workloads: {sorted(unknown)}", file=sys.stderr)
        return 2

    stop = asyncio.Event()
    tasks: list[asyncio.Task] = []

    if "mem" in workloads:
        from .memory import MemoryController
        mem = MemoryController(
            gpu=args.gpu,
            mem_frac=args.mem_frac,
            mem_mb=args.mem_mb,
            noise_frac=args.mem_noise_frac,
            period_s=args.mem_noise_period_s,
        )
        tasks.append(asyncio.create_task(mem.run(stop), name="ocp-worker-mem"))

    if "util" in workloads:
        from .compute import ComputeController
        comp = ComputeController(
            gpu=args.gpu,
            util_target=args.util_target,
            noise_frac=args.util_noise_frac,
            period_s=args.util_noise_period_s,
        )
        tasks.append(asyncio.create_task(comp.run(stop), name="ocp-worker-util"))

    if not tasks:
        print("ocp-worker: no workloads selected; exiting", file=sys.stderr)
        return 0

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()
    for t in tasks:
        t.cancel()
    for t in tasks:
        try:
            await t
        except asyncio.CancelledError:
            pass
    return 0


def main(argv: list[str] | None = None) -> int:
    _set_proctitle()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s ocp-worker[%(process)d] %(message)s",
        stream=sys.stderr,
    )
    args = _parse_args(argv)
    _set_nice(args.nice)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
