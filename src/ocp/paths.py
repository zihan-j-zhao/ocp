"""System paths used by the daemon, CLI, and worker.

All paths are derived from environment variables when set (handy for tests).
Production defaults: the daemon runs as root, with config in root's home and
runtime/log state under `/run/ocp` and `/var/log/ocp`.
"""
from __future__ import annotations

import os
from pathlib import Path


def _path(env_var: str, default: str | Path) -> Path:
    return Path(os.environ.get(env_var) or default)


def _root_home() -> Path:
    try:
        import pwd
        return Path(pwd.getpwuid(0).pw_dir)
    except (KeyError, ImportError):
        return Path("/root")


RUN_DIR = _path("OCP_RUN_DIR", "/run/ocp")
CONFIG_DIR = _path("OCP_CONFIG_DIR", _root_home() / ".ocp")
LOG_DIR = _path("OCP_LOG_DIR", "/var/log/ocp")

SOCKET_PATH = RUN_DIR / "ocpd.sock"
PIDFILE_PATH = RUN_DIR / "ocpd.pid"
STATE_PATH = RUN_DIR / "state.json"

CONFIG_PATH = CONFIG_DIR / "config.toml"

HISTORY_PATH = LOG_DIR / "history.jsonl"
