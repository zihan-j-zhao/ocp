"""TOML config loading + validation.

The daemon is the sole writer of nothing here: this module only reads and
validates `/etc/ocp/config.toml`. See PLAN.md §8.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import paths


class ConfigError(ValueError):
    pass


@dataclass
class GeneralCfg:
    poll_interval_s: float = 2.0
    log_level: str = "INFO"


@dataclass
class GpuCfg:
    indices: list[int] = field(default_factory=list)  # [] = all visible


@dataclass
class ThresholdsCfg:
    util_low: int = 5
    mem_low: int = 10
    idle_debounce_s: int = 30


@dataclass
class WorkerCfg:
    workloads: list[str] = field(default_factory=lambda: ["mem", "util"])
    mem_frac: float = 0.5
    mem_mb: int | None = None
    mem_noise_frac: float = 0.05
    mem_noise_period_s: float = 5.0
    util_target: int = 90
    util_noise_frac: float = 0.05
    util_noise_period_s: float = 5.0
    nice: int = 19
    restart_backoff_s: float = 5.0


@dataclass
class MultiuserCfg:
    # Entries may be str (username) or int (uid).
    whitelist: list = field(default_factory=list)


@dataclass
class PauseCfg:
    min_duration_s: int = 30
    max_duration_s: int = 7200


@dataclass
class HistoryCfg:
    max_entries: int = 1000


@dataclass
class Config:
    general: GeneralCfg = field(default_factory=GeneralCfg)
    gpu: GpuCfg = field(default_factory=GpuCfg)
    thresholds: ThresholdsCfg = field(default_factory=ThresholdsCfg)
    worker: WorkerCfg = field(default_factory=WorkerCfg)
    multiuser: MultiuserCfg = field(default_factory=MultiuserCfg)
    pause: PauseCfg = field(default_factory=PauseCfg)
    history: HistoryCfg = field(default_factory=HistoryCfg)

    def to_dict(self) -> dict:
        return asdict(self)


_VALID_WORKLOADS = {"mem", "util"}
_KNOWN_SECTIONS = {
    "general", "gpu", "thresholds", "worker",
    "multiuser", "pause", "history",
}


def _load_section(raw: dict, section: str, cls):
    sub = raw.get(section, {}) or {}
    if not isinstance(sub, dict):
        raise ConfigError(f"[{section}] must be a table")
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(sub) - valid
    if unknown:
        raise ConfigError(f"[{section}] unknown keys: {sorted(unknown)}")
    try:
        return cls(**sub)
    except TypeError as e:
        raise ConfigError(f"[{section}] {e}") from e


def _validate(cfg: Config) -> None:
    t = cfg.thresholds
    if not 0 <= t.util_low <= 100:
        raise ConfigError("thresholds.util_low must be in [0, 100]")
    if not 0 <= t.mem_low <= 100:
        raise ConfigError("thresholds.mem_low must be in [0, 100]")
    if t.idle_debounce_s <= 0:
        raise ConfigError("thresholds.idle_debounce_s must be > 0")

    w = cfg.worker
    bad = set(w.workloads) - _VALID_WORKLOADS
    if bad:
        raise ConfigError(f"worker.workloads contains unknown entries: {sorted(bad)}")
    if w.mem_mb is None and not (0 < w.mem_frac < 1):
        raise ConfigError("worker.mem_frac must be in (0, 1) when mem_mb is unset")
    if w.mem_mb is not None and w.mem_mb <= 0:
        raise ConfigError("worker.mem_mb must be > 0")
    if not (0 <= w.util_target <= 100):
        raise ConfigError("worker.util_target must be in [0, 100]")
    if w.mem_noise_frac < 0 or w.util_noise_frac < 0:
        raise ConfigError("worker noise fractions must be >= 0")
    if w.mem_noise_period_s <= 0 or w.util_noise_period_s <= 0:
        raise ConfigError("worker noise periods must be > 0")

    p = cfg.pause
    if p.min_duration_s <= 0:
        raise ConfigError("pause.min_duration_s must be > 0")
    if p.max_duration_s < p.min_duration_s:
        raise ConfigError("pause.max_duration_s must be >= min_duration_s")

    if cfg.history.max_entries < 10:
        raise ConfigError("history.max_entries must be >= 10")

    g = cfg.gpu
    if any((not isinstance(i, int) or isinstance(i, bool) or i < 0)
           for i in g.indices):
        raise ConfigError("gpu.indices must be a list of non-negative integers")

    for entry in cfg.multiuser.whitelist:
        if isinstance(entry, bool) or not isinstance(entry, (str, int)):
            raise ConfigError(f"multiuser.whitelist entry not a str or int: {entry!r}")

    if cfg.general.poll_interval_s <= 0:
        raise ConfigError("general.poll_interval_s must be > 0")


def load(path: Path | None = None) -> Config:
    """Load and validate the config. Returns defaults if the file is absent."""
    path = path or paths.CONFIG_PATH
    if not path.exists():
        cfg = Config()
        _validate(cfg)
        return cfg
    try:
        raw = tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, OSError) as e:
        raise ConfigError(f"parse error in {path}: {e}") from e
    unknown = set(raw) - _KNOWN_SECTIONS
    if unknown:
        raise ConfigError(f"unknown top-level tables: {sorted(unknown)}")
    cfg = Config(
        general=_load_section(raw, "general", GeneralCfg),
        gpu=_load_section(raw, "gpu", GpuCfg),
        thresholds=_load_section(raw, "thresholds", ThresholdsCfg),
        worker=_load_section(raw, "worker", WorkerCfg),
        multiuser=_load_section(raw, "multiuser", MultiuserCfg),
        pause=_load_section(raw, "pause", PauseCfg),
        history=_load_section(raw, "history", HistoryCfg),
    )
    _validate(cfg)
    return cfg
