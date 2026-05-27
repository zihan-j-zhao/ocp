"""ocpd — the OCP system daemon."""
from __future__ import annotations

import argparse
import asyncio
import errno
import fcntl
import logging
import os
import signal
import sys
import time
from pathlib import Path

from . import auth, config as cfg_mod, logging_setup, paths
from .controller import Controller
from .history import HistoryStore
from .ipc.protocol import Request, Response
from .ipc.server import UDSServer
from .lease import LeaseStore, PauseHeld
from .monitor.nvml import NvmlMonitor, NvmlOpenError
from .worker.manager import WorkerManager

log = logging.getLogger("ocpd")


ANY_CMDS = {"GET_STATUS", "HISTORY_GET", "CONFIG_LIST"}
WHITELIST_CMDS = {"PAUSE", "RESUME"}
ROOT_CMDS = {"CONFIG_RELOAD"}
ALL_CMDS = ANY_CMDS | WHITELIST_CMDS | ROOT_CMDS


class Pidfile:
    def __init__(self, path: Path):
        self._path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                try:
                    pid = self._path.read_text().strip()
                except OSError:
                    pid = "?"
                raise RuntimeError(f"ocpd already running (pid={pid})") from e
            raise
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass


class Daemon:
    def __init__(self, cfg: cfg_mod.Config):
        self.cfg = cfg
        self._monitor = NvmlMonitor()
        self._leases = LeaseStore(paths.STATE_PATH)
        self._history = HistoryStore(paths.HISTORY_PATH, cfg.history.max_entries)
        self._workers = WorkerManager(cfg.worker, on_exit=self._on_worker_exit)
        self._controller = Controller(
            cfg=cfg, monitor=self._monitor, leases=self._leases,
            workers=self._workers, history=self._history,
        )
        self._server = UDSServer(paths.SOCKET_PATH, self._dispatch)
        self._pidfile = Pidfile(paths.PIDFILE_PATH)
        self._started_at = time.time()
        self._nvml_ok = False
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    # --- lifecycle --------------------------------------------------------

    async def run(self) -> int:
        self._pidfile.acquire()
        try:
            self._setup_signals()
            try:
                self._monitor.open()
                self._nvml_ok = True
            except NvmlOpenError as e:
                log.warning("NVML unavailable: %s; monitoring disabled", e)
            self._controller.set_watched(self._resolve_watched())
            self._history.load()
            self._history.start_writer()
            restored = self._leases.load()
            self._history.record(
                uid=0, user="(auto)", cmd="daemon_started",
                args={
                    "pid": os.getpid(),
                    "watched": self._controller.watched,
                    "restored_leases": len(restored),
                },
            )
            await self._server.start()
            self._tasks.append(asyncio.create_task(
                self._server.serve_forever(), name="ipc-server"))
            self._tasks.append(asyncio.create_task(
                self._monitor_loop(), name="monitor"))
            self._tasks.append(asyncio.create_task(
                self._lease_expiry_loop(), name="lease-expiry"))
            await self._stop.wait()
            return 0
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        log.info("shutting down")
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()
        try:
            await self._server.stop()
        except Exception:
            log.exception("server stop failed")
        await self._workers.stop_all()
        self._leases.persist()
        self._history.record(
            uid=0, user="(auto)", cmd="daemon_stopped",
            args={"pid": os.getpid()},
        )
        await self._history.stop_writer()
        try:
            self._monitor.close()
        except Exception:
            pass
        self._pidfile.release()

    def _setup_signals(self) -> None:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, self._stop.set)
        loop.add_signal_handler(signal.SIGINT, self._stop.set)
        loop.add_signal_handler(signal.SIGHUP, self._on_sighup)

    def _on_sighup(self) -> None:
        asyncio.create_task(self._reload_config(actor=None), name="sighup-reload")

    def _resolve_watched(self) -> list[int]:
        cfg = self.cfg
        if cfg.gpu.indices:
            # Trust explicit config. Devices NVML cannot see still appear in
            # status with a monitor_error sample so users notice.
            return list(cfg.gpu.indices)
        return self._monitor.enumerate() if self._nvml_ok else []

    # --- background tasks -------------------------------------------------

    async def _monitor_loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    await self._controller.tick_once()
                except Exception:
                    log.exception("controller tick failed")
                try:
                    await asyncio.wait_for(
                        self._stop.wait(),
                        timeout=self.cfg.general.poll_interval_s,
                    )
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass

    async def _lease_expiry_loop(self) -> None:
        try:
            while not self._stop.is_set():
                leases = self._leases.all()
                if not leases:
                    sleep_for = 30.0
                else:
                    sleep_for = max(0.0, min(l.expires_at for l in leases) - time.time())
                    sleep_for = min(sleep_for, 30.0)
                if sleep_for > 0:
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=sleep_for)
                    except asyncio.TimeoutError:
                        pass
                if self._stop.is_set():
                    return
                async with self._controller.lock:
                    for expired in self._leases.sweep_expired():
                        self._history.record(
                            uid=0, user="(auto)", cmd="lease_expired",
                            args={"gpu": expired.gpu},
                            note=f"holder was {expired.user}",
                        )
        except asyncio.CancelledError:
            pass

    async def _on_worker_exit(self, gpu: int, rc: int) -> None:
        self._history.record(
            uid=0, user="(auto)", cmd="worker_crashed",
            args={"gpu": gpu, "rc": rc}, ok=False,
            error=f"unexpected exit code {rc}",
        )

    # --- IPC dispatch -----------------------------------------------------

    async def _dispatch(self, caller: auth.Caller, req: Request) -> Response:
        cmd = req.cmd.upper()
        if cmd not in ALL_CMDS:
            return Response.failure(req.id, "E_BAD_REQUEST", f"unknown cmd: {req.cmd}")
        if cmd in ROOT_CMDS and not auth.is_root(caller):
            return Response.failure(
                req.id, "E_NOT_ROOT", f"{cmd.lower()} requires root"
            )
        if (
            cmd in WHITELIST_CMDS
            and not auth.is_whitelisted(caller, self.cfg.multiuser.whitelist)
        ):
            return Response.failure(
                req.id, "E_NOT_WHITELISTED",
                f"you are not whitelisted; allowed: {self._format_allowed()}",
                {"allowed": list(self.cfg.multiuser.whitelist)},
            )
        try:
            if cmd == "GET_STATUS":
                return Response.success(req.id, await self._handle_status())
            if cmd == "HISTORY_GET":
                return Response.success(req.id, await self._handle_history(caller, req))
            if cmd == "CONFIG_LIST":
                return Response.success(req.id, {"config": self.cfg.to_dict()})
            if cmd == "CONFIG_RELOAD":
                return await self._handle_config_reload(caller, req)
            if cmd == "PAUSE":
                return await self._handle_pause(caller, req)
            if cmd == "RESUME":
                return await self._handle_resume(caller, req)
            return Response.failure(req.id, "E_BAD_REQUEST", f"unhandled cmd: {cmd}")
        except Exception as e:  # noqa: BLE001
            log.exception("handler error cmd=%s user=%s", cmd, caller.user)
            return Response.failure(req.id, "E_BAD_REQUEST", f"internal error: {e}")

    def _format_allowed(self) -> str:
        wl = self.cfg.multiuser.whitelist
        return ", ".join(str(e) for e in wl) if wl else "(only root)"

    async def _handle_status(self) -> dict:
        gpus = []
        for gpu, st in sorted(self._controller.states().items()):
            sample = st.last_sample
            worker = self._workers.get(gpu)
            lease = self._leases.get(gpu)
            gpus.append({
                "gpu": gpu,
                "util_pct": sample.util_pct if sample else None,
                "mem_used_mb": sample.mem_used_mb if sample else None,
                "mem_total_mb": sample.mem_total_mb if sample else None,
                "compute_pids": list(sample.compute_pids) if sample else [],
                "monitor_error": sample.error if sample else None,
                "worker": (
                    {"pid": worker.pid,
                     "uptime_s": int(time.time() - worker.started_at)}
                    if worker else None
                ),
                "lease": (
                    {"user": lease.user, "uid": lease.uid,
                     "remaining_s": lease.remaining_s(),
                     "expires_at": lease.expires_at}
                    if lease else None
                ),
                "decision": st.last_decision,
            })
        return {
            "daemon": {
                "pid": os.getpid(),
                "uptime_s": int(time.time() - self._started_at),
                "socket": str(paths.SOCKET_PATH),
                "nvml_ok": self._nvml_ok,
            },
            "watched": self._controller.watched,
            "gpus": gpus,
        }

    async def _handle_history(self, caller: auth.Caller, req: Request) -> dict:
        args = req.args or {}
        n = args.get("n")
        since_s = args.get("since_s")
        user = args.get("user")
        uid = caller.uid if args.get("mine") else args.get("uid")
        entries = self._history.query(
            n=int(n) if n else None,
            since_s=float(since_s) if since_s else None,
            user=str(user) if user else None,
            uid=int(uid) if uid is not None else None,
        )
        return {"entries": [e.to_dict() for e in entries]}

    async def _handle_config_reload(self, caller: auth.Caller, req: Request) -> Response:
        try:
            new_cfg = cfg_mod.load()
        except cfg_mod.ConfigError as e:
            self._history.record(
                uid=caller.uid, user=caller.user, cmd="config_reload",
                ok=False, error="E_BAD_CONFIG", note=str(e),
            )
            return Response.failure(req.id, "E_BAD_CONFIG", str(e))
        await self._apply_new_config(new_cfg, actor=caller)
        return Response.success(req.id, {"config": new_cfg.to_dict()})

    async def _reload_config(self, *, actor: auth.Caller | None) -> None:
        try:
            new_cfg = cfg_mod.load()
        except cfg_mod.ConfigError as e:
            self._history.record(
                uid=actor.uid if actor else 0,
                user=actor.user if actor else "(auto)",
                cmd="config_reload",
                ok=False, error="E_BAD_CONFIG", note=str(e),
            )
            log.error("config reload failed: %s", e)
            return
        await self._apply_new_config(new_cfg, actor=actor)

    async def _apply_new_config(
        self, new_cfg: cfg_mod.Config, *, actor: auth.Caller | None
    ) -> None:
        old = self.cfg
        async with self._controller.lock:
            self.cfg = new_cfg
            self._controller.replace_config(new_cfg)
            self._workers.replace_config(new_cfg.worker)
            self._history.set_max_entries(new_cfg.history.max_entries)
            self._controller.set_watched(self._resolve_watched())
        self._history.record(
            uid=actor.uid if actor else 0,
            user=actor.user if actor else "(auto)",
            cmd="config_reload", ok=True,
            note=_diff_summary(old, new_cfg),
        )

    async def _handle_pause(self, caller: auth.Caller, req: Request) -> Response:
        args = req.args or {}
        gpus = args.get("gpus")
        duration_s = args.get("duration_s")
        if (
            not isinstance(gpus, list)
            or not gpus
            or any(isinstance(g, bool) or not isinstance(g, int) for g in gpus)
        ):
            return Response.failure(
                req.id, "E_BAD_GPU_SET", "gpus must be a non-empty list of ints"
            )
        if len(set(gpus)) != len(gpus):
            return Response.failure(req.id, "E_BAD_GPU_SET", "duplicate gpu in request")
        watched = set(self._controller.watched)
        bad = [g for g in gpus if g not in watched]
        if bad:
            return Response.failure(
                req.id, "E_BAD_GPU_SET",
                f"gpus not in watched set: {bad}",
                {"watched": sorted(watched)},
            )
        if not isinstance(duration_s, (int, float)) or duration_s <= 0:
            return Response.failure(req.id, "E_BAD_DURATION", "duration_s must be > 0")
        duration_s = int(duration_s)
        if duration_s < self.cfg.pause.min_duration_s:
            return Response.failure(
                req.id, "E_PAUSE_TOO_SHORT",
                f"duration < {self.cfg.pause.min_duration_s}s",
                {"min_s": self.cfg.pause.min_duration_s},
            )
        if duration_s > self.cfg.pause.max_duration_s:
            return Response.failure(
                req.id, "E_PAUSE_TOO_LONG",
                f"duration > {self.cfg.pause.max_duration_s}s",
                {"max_s": self.cfg.pause.max_duration_s},
            )
        try:
            leases = await self._controller.pause(
                uid=caller.uid, user=caller.user,
                gpus=list(gpus), duration_s=duration_s,
            )
        except PauseHeld as e:
            self._history.record(
                uid=caller.uid, user=caller.user, cmd="pause",
                args={"gpus": gpus, "duration_s": duration_s}, ok=False,
                error="E_PAUSE_HELD",
                note=", ".join(
                    f"gpu={c['gpu']} held by {c['holder']}" for c in e.conflicts
                ),
            )
            holder_msg = "; ".join(
                f"gpu {c['gpu']} reserved by {c['holder']} ({c['remaining_s']}s left)"
                for c in e.conflicts
            )
            return Response.failure(
                req.id, "E_PAUSE_HELD", holder_msg, {"conflicts": e.conflicts}
            )
        self._history.record(
            uid=caller.uid, user=caller.user, cmd="pause",
            args={"gpus": gpus, "duration_s": duration_s}, ok=True,
            note=(
                f"expires at "
                f"{time.strftime('%H:%M:%S', time.localtime(leases[0].expires_at))}"
            ),
        )
        return Response.success(
            req.id,
            {"leases": [
                {"user": l.user, "uid": l.uid, "gpu": l.gpu,
                 "expires_at": l.expires_at, "remaining_s": l.remaining_s()}
                for l in leases
            ]},
        )

    async def _handle_resume(self, caller: auth.Caller, req: Request) -> Response:
        args = req.args or {}
        gpus = args.get("gpus")
        if (
            not isinstance(gpus, list)
            or not gpus
            or any(isinstance(g, bool) or not isinstance(g, int) for g in gpus)
        ):
            return Response.failure(
                req.id, "E_BAD_GPU_SET", "gpus must be a non-empty list of ints"
            )
        if len(set(gpus)) != len(gpus):
            return Response.failure(req.id, "E_BAD_GPU_SET", "duplicate gpu in request")
        released, errors = await self._controller.resume(
            uid=caller.uid, gpus=list(gpus), is_root=auth.is_root(caller),
        )
        for g in released:
            self._history.record(
                uid=caller.uid, user=caller.user, cmd="resume",
                args={"gpu": g}, ok=True,
            )
        for err in errors:
            self._history.record(
                uid=caller.uid, user=caller.user, cmd="resume",
                args={"gpu": err["gpu"]}, ok=False, error=err["code"],
            )
        if released:
            return Response.success(req.id, {"released": released, "errors": errors})
        code = errors[0]["code"] if errors else "E_NO_PAUSE"
        return Response.failure(req.id, code, "no GPUs released", {"errors": errors})


def _diff_summary(old, new) -> str:
    changed = [
        section for section in (
            "general", "gpu", "thresholds", "worker",
            "multiuser", "pause", "history",
        )
        if getattr(old, section) != getattr(new, section)
    ]
    return f"changed sections: {', '.join(changed) or 'none'}"


# --- entry point -----------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("ocpd")
    p.add_argument(
        "--foreground", action="store_true",
        help="run in the foreground (default — provided for clarity)",
    )
    p.add_argument(
        "--config", type=Path, default=None, help="override config dir",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.config is not None:
        os.environ["OCP_CONFIG_DIR"] = str(args.config.parent)
        from importlib import reload
        reload(paths)
    try:
        cfg = cfg_mod.load()
    except cfg_mod.ConfigError as e:
        print(f"ocpd: bad config: {e}", file=sys.stderr)
        return 2
    logging_setup.setup(cfg.general.log_level)
    daemon = Daemon(cfg)
    try:
        return asyncio.run(daemon.run())
    except RuntimeError as e:
        print(f"ocpd: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
