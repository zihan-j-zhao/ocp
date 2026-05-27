"""ocp — the CLI client."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer

from . import paths
from .duration import DurationError, format_duration, parse_duration
from .gpuspec import GpuSpecError, parse_gpu_spec
from .history import format_relative
from .ipc.client import DaemonDown, IPCClientError, call

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="OCP — Occupy Compute Process",
)
daemon_app = typer.Typer(no_args_is_help=True, help="manage the ocpd daemon")
config_app = typer.Typer(no_args_is_help=True, help="inspect / reload config")
app.add_typer(daemon_app, name="daemon")
app.add_typer(config_app, name="config")


_JSON_OPT = typer.Option(False, "--json", help="emit machine-readable JSON")
_SOCK_OPT = typer.Option(None, "--socket", help="override UDS path")


def _fail(msg: str, *, code: str | None = None, exit_code: int = 1) -> None:
    if code:
        typer.echo(f"error: {msg}\n       ({code})", err=True)
    else:
        typer.echo(f"error: {msg}", err=True)
    raise typer.Exit(exit_code)


def _err_info(resp: dict) -> tuple[str, str]:
    err = resp.get("error") or {}
    return str(err.get("code", "E_UNKNOWN")), str(err.get("msg", "unknown error"))


def _do_call(
    cmd: str,
    args: dict | None = None,
    *,
    socket_path: Optional[Path] = None,
) -> dict:
    try:
        return call(cmd, args, socket_path=socket_path)
    except DaemonDown as e:
        _fail(f"daemon is not running ({e}). Try: sudo ocp daemon on")
    except IPCClientError as e:
        _fail(f"IPC error: {e}")


def _print_json(resp: dict) -> None:
    json.dump(resp, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")


# --- daemon -----------------------------------------------------------------

def _read_pidfile() -> int | None:
    try:
        return int(paths.PIDFILE_PATH.read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _require_root(action: str) -> None:
    if os.geteuid() != 0:
        _fail(f"ocp daemon {action} requires root (try: sudo ocp daemon {action})")


@daemon_app.command("on")
def daemon_on() -> None:
    """Start the daemon (root only, idempotent)."""
    _require_root("on")
    pid = _read_pidfile()
    if pid and _pid_alive(pid):
        typer.echo(f"ok: ocpd already running (pid {pid})")
        return
    subprocess.Popen(
        [sys.executable, "-m", "ocp.daemon"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(50):
        time.sleep(0.1)
        pid = _read_pidfile()
        if pid and _pid_alive(pid):
            typer.echo(f"ok: ocpd started (pid {pid})")
            return
    _fail("ocpd may have failed to start; check /var/log/ocp/ocpd.log or run `ocpd` in the foreground")


@daemon_app.command("off")
def daemon_off() -> None:
    """Stop the daemon (root only)."""
    _require_root("off")
    pid = _read_pidfile()
    if not pid or not _pid_alive(pid):
        typer.echo("ok: ocpd is not running")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        typer.echo("ok: ocpd is not running")
        return
    for _ in range(50):
        if not _pid_alive(pid):
            typer.echo("ok: ocpd stopped")
            return
        time.sleep(0.1)
    _fail(f"ocpd (pid {pid}) did not exit within 5s")


@daemon_app.command("status")
def daemon_status() -> None:
    """Check whether the daemon is running (anyone)."""
    pid = _read_pidfile()
    if pid and _pid_alive(pid):
        typer.echo(f"up    pid={pid}    socket={paths.SOCKET_PATH}")
    else:
        typer.echo(f"down  socket={paths.SOCKET_PATH}")


# --- status / pause / resume -----------------------------------------------

@app.command("status")
def status_cmd(
    as_json: bool = _JSON_OPT,
    socket: Optional[Path] = _SOCK_OPT,
) -> None:
    resp = _do_call("GET_STATUS", socket_path=socket)
    if not resp.get("ok"):
        code, msg = _err_info(resp); _fail(msg, code=code)
    if as_json:
        _print_json(resp); return
    data = resp.get("data") or {}
    d = data.get("daemon") or {}
    typer.echo(
        f"daemon:  up (pid {d.get('pid')}, "
        f"uptime {format_duration(int(d.get('uptime_s', 0)))})"
    )
    if not d.get("nvml_ok", True):
        typer.echo("         NVML unavailable; monitoring is disabled")
    gpus = data.get("gpus") or []
    if not gpus:
        typer.echo("(no GPUs being watched)")
        return
    for g in gpus:
        line = f"gpu {g['gpu']}: "
        if g.get("monitor_error"):
            line += f"NVML error: {g['monitor_error']}"
        else:
            util = g.get("util_pct")
            mem_u_mb = g.get("mem_used_mb") or 0
            mem_t_mb = g.get("mem_total_mb") or 0
            line += (
                f"util {util:>3}%, "
                f"mem {mem_u_mb / 1024:>5.1f} GB / {mem_t_mb / 1024:>5.1f} GB"
            )
            worker = g.get("worker")
            line += "  | worker: " + (
                f"up pid={worker['pid']}" if worker else "down"
            )
        lease = g.get("lease")
        if lease:
            line += (
                f"  | paused by {lease['user']} (uid {lease['uid']}) "
                f"for {format_duration(int(lease['remaining_s']))} more"
            )
        typer.echo(line)


@app.command("pause")
def pause_cmd(
    gpu: str = typer.Argument(..., help="GPU index or comma-list (e.g. '7' or '0,2')"),
    duration: str = typer.Argument(..., help="duration (e.g. 10m, 1h30m)"),
    as_json: bool = _JSON_OPT,
    socket: Optional[Path] = _SOCK_OPT,
) -> None:
    try:
        gpus = parse_gpu_spec(gpu)
    except GpuSpecError as e:
        _fail(str(e))
    try:
        dur_s = parse_duration(duration)
    except DurationError as e:
        _fail(str(e))
    resp = _do_call("PAUSE", {"gpus": gpus, "duration_s": dur_s}, socket_path=socket)
    if as_json:
        _print_json(resp); return
    if not resp.get("ok"):
        code, msg = _err_info(resp); _fail(msg, code=code)
    leases = (resp.get("data") or {}).get("leases") or []
    for lease in leases:
        exp = time.strftime("%H:%M:%S", time.localtime(lease["expires_at"]))
        typer.echo(
            f"ok: paused gpu {lease['gpu']} until {exp} "
            f"({format_duration(int(lease['remaining_s']))})"
        )


@app.command("resume")
def resume_cmd(
    gpu: str = typer.Argument(..., help="GPU index or comma-list"),
    as_json: bool = _JSON_OPT,
    socket: Optional[Path] = _SOCK_OPT,
) -> None:
    try:
        gpus = parse_gpu_spec(gpu)
    except GpuSpecError as e:
        _fail(str(e))
    resp = _do_call("RESUME", {"gpus": gpus}, socket_path=socket)
    if as_json:
        _print_json(resp); return
    if not resp.get("ok"):
        code, msg = _err_info(resp); _fail(msg, code=code)
    data = resp.get("data") or {}
    released = data.get("released") or []
    errors = data.get("errors") or []
    if released:
        typer.echo(f"ok: released gpu(s): {', '.join(map(str, released))}")
    for err in errors:
        typer.echo(f"  gpu {err['gpu']}: {err.get('code')}", err=True)


@app.command("history")
def history_cmd(
    n: int = typer.Option(20, "-n", help="number of entries"),
    since: Optional[str] = typer.Option(None, "--since", help="e.g. 1h, 30m"),
    user: Optional[str] = typer.Option(None, "--user"),
    mine: bool = typer.Option(False, "--mine"),
    as_json: bool = _JSON_OPT,
    socket: Optional[Path] = _SOCK_OPT,
) -> None:
    args: dict = {"n": n}
    if since:
        try:
            args["since_s"] = parse_duration(since)
        except DurationError as e:
            _fail(str(e))
    if user:
        args["user"] = user
    if mine:
        args["mine"] = True
    resp = _do_call("HISTORY_GET", args, socket_path=socket)
    if not resp.get("ok"):
        code, msg = _err_info(resp); _fail(msg, code=code)
    if as_json:
        _print_json(resp); return
    entries = (resp.get("data") or {}).get("entries") or []
    now = time.time()
    for e in entries:
        ts = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
        rel = format_relative(e["ts"], now)
        suffix = ""
        if not e.get("ok", True):
            suffix = f"  [{e.get('error')}]"
        note = e.get("note")
        if note:
            suffix += f"  ({note})"
        typer.echo(
            f"{ts}  ({rel:>7})  {e['user']:>10}  "
            f"{e['cmd']} {_fmt_args(e.get('args') or {})}{suffix}"
        )


def _fmt_args(args: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in args.items())


@config_app.command("list")
def config_list(
    as_json: bool = _JSON_OPT,
    socket: Optional[Path] = _SOCK_OPT,
) -> None:
    resp = _do_call("CONFIG_LIST", socket_path=socket)
    if not resp.get("ok"):
        code, msg = _err_info(resp); _fail(msg, code=code)
    if as_json:
        _print_json(resp); return
    cfg = (resp.get("data") or {}).get("config") or {}
    json.dump(cfg, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")


@config_app.command("reload")
def config_reload(socket: Optional[Path] = _SOCK_OPT) -> None:
    resp = _do_call("CONFIG_RELOAD", socket_path=socket)
    if not resp.get("ok"):
        code, msg = _err_info(resp); _fail(msg, code=code)
    typer.echo("ok: config reloaded")


@config_app.command("path")
def config_path_cmd() -> None:
    typer.echo(str(paths.CONFIG_PATH))


if __name__ == "__main__":
    app()
