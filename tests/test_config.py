import pytest
from pathlib import Path
from ocp import config


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(text)
    return p


def test_defaults_valid():
    config._validate(config.Config())  # no raise


def test_load_minimal(tmp_path):
    p = _write(tmp_path, "[general]\npoll_interval_s = 1.0\n")
    cfg = config.load(p)
    assert cfg.general.poll_interval_s == 1.0
    assert cfg.thresholds.util_low == 5  # default


def test_unknown_top_level(tmp_path):
    p = _write(tmp_path, "[whatever]\nfoo = 1\n")
    with pytest.raises(config.ConfigError):
        config.load(p)


def test_unknown_key_in_section(tmp_path):
    p = _write(tmp_path, "[worker]\nfoo = 1\n")
    with pytest.raises(config.ConfigError):
        config.load(p)


def test_bad_thresholds(tmp_path):
    p = _write(tmp_path, "[thresholds]\nutil_low = 50\nutil_high = 30\n")
    with pytest.raises(config.ConfigError):
        config.load(p)


def test_bad_workload(tmp_path):
    p = _write(tmp_path, "[worker]\nworkloads = ['foo']\n")
    with pytest.raises(config.ConfigError):
        config.load(p)


def test_bad_mem_frac(tmp_path):
    p = _write(tmp_path, "[worker]\nmem_frac = 1.5\n")
    with pytest.raises(config.ConfigError):
        config.load(p)


def test_pause_min_gt_max(tmp_path):
    p = _write(tmp_path, "[pause]\nmin_duration_s = 100\nmax_duration_s = 30\n")
    with pytest.raises(config.ConfigError):
        config.load(p)


def test_whitelist_mixed(tmp_path):
    p = _write(tmp_path, '[multiuser]\nwhitelist = ["alice", 1042]\n')
    cfg = config.load(p)
    assert cfg.multiuser.whitelist == ["alice", 1042]


def test_absent_file_returns_defaults(tmp_path):
    p = tmp_path / "nope.toml"
    cfg = config.load(p)
    assert cfg.general.log_level == "INFO"
