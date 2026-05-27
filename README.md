# OCP — Occupy Compute Process

A client/server tool that watches GPU utilization & memory and, when a device is
idle, launches a worker process to keep it occupied. A CLI client talks to a
background daemon. See [PLAN.md](PLAN.md) for the full design.

## Requirements

- Linux (systemd is optional — see the auto-start note in [Install](#install))
- NVIDIA driver ≥ R570 (Blackwell-capable; A100 unaffected)
- Python ≥ 3.10
- For the worker process: PyTorch ≥ 2.6 with CUDA 12.8

## Install

The model is simple:

- **Root** installs the package, starts and stops the daemon, and owns the
  config file at `/root/.ocp/config.toml`.
- **Any user** can install the same package (to get the CLI) and talk to the
  running daemon: `status`, `history`, `pause`, `resume`, `config list`. Only
  root can `daemon on|off` or `config reload`.
- Only one daemon runs per host (enforced by an `flock` on the pidfile).

### As root: install + start the daemon

```bash
sudo pip install '.[worker]'        # installs ocp, ocpd, ocp-worker
sudo mkdir -p /root/.ocp            # holds the config
sudo $EDITOR /root/.ocp/config.toml # optional — defaults are sensible
sudo ocp daemon on                  # forks ocpd; creates /run/ocp + /var/log/ocp
```

That's it. The daemon mkdirs its own runtime (`/run/ocp/`) and log
(`/var/log/ocp/`) directories on first start.

To stop:

```bash
sudo ocp daemon off
```

#### Optional: auto-start at boot via systemd

Drop in the unit file shipped in `packaging/` and enable it:

```bash
sudo install -m 0644 packaging/ocpd.service /etc/systemd/system/ocpd.service
sudo systemctl daemon-reload
sudo systemctl enable --now ocpd.service
```

(`ocp daemon on/off` then still works — both paths just send signals to the
same `ocpd` process.)

### As a regular user: install + use

```bash
pip install '.'                # or pip install -e '.[test]' for dev
ocp daemon status              # any user: is the daemon up?
ocp status                     # per-GPU view
ocp pause 7 10m                # only succeeds if you're in the whitelist
ocp history
```

`ocp daemon on|off` from a non-root user is rejected up-front with a clear
error message. Everything else works as long as the daemon is running.

To pause/resume, root must add you to the whitelist:

```bash
sudo $EDITOR /root/.ocp/config.toml      # add yourself under [multiuser].whitelist
sudo ocp config reload
```

### Development / unprivileged testing

If you want to run the daemon as yourself (e.g. for hacking), redirect the
runtime/config/log dirs to a writable location and start the daemon directly:

```bash
export OCP_RUN_DIR=$HOME/.local/run/ocp
export OCP_CONFIG_DIR=$HOME/.config/ocp
export OCP_LOG_DIR=$HOME/.local/share/ocp
python -m ocp.daemon &
ocp status
```

The two scripts in [`scripts/`](scripts/) show this pattern end-to-end.

## Usage

### Command reference

| Command                         | Who           | What it does                                            |
| ------------------------------- | ------------- | ------------------------------------------------------- |
| `ocp daemon on`                 | **root**      | Start the daemon (idempotent).                          |
| `ocp daemon off`                | **root**      | Stop the daemon (SIGTERM, graceful).                    |
| `ocp daemon status`             | anyone        | Print whether the daemon is up + its pid + socket path. |
| `ocp status`                    | anyone        | Per-GPU snapshot: util, memory, worker, active lease.   |
| `ocp pause <gpu> <duration>`    | whitelisted   | Reserve one or more GPUs for a duration (e.g. `5m`).    |
| `ocp resume <gpu>`              | whitelisted   | Release a reservation early (you must be the holder).   |
| `ocp history [-n N] [--mine]`   | anyone        | Recent events (pause/resume/spawn/yield/…).             |
| `ocp config list`               | anyone        | Print the in-memory config snapshot.                    |
| `ocp config reload`             | **root**      | Re-read `/root/.ocp/config.toml` and swap atomically.   |
| `ocp config path`               | anyone        | Print the config file path.                             |

Add `--json` to any IPC command for machine-readable output.

GPU spec accepts a single index or a comma-list: `7`, `0,2,3`. Durations are
`5s` / `90s` / `5m` / `1h` / `1h30m` (combinations allowed; no fractions).

### Example session

```bash
# --- on the box, as root ---
sudo ocp daemon on
# ok: ocpd started (pid 12345)

# --- as alice (whitelisted) ---
ocp status
# daemon:  up (pid 12345, uptime 2m)
# gpu 0:   util  12%, mem  2.4 GB /  80.0 GB  | worker: down
# gpu 7:   util   0%, mem  0.5 GB /  80.0 GB  | worker: down

ocp pause 7 10m
# ok: paused gpu 7 until 14:42 (10m)

ocp status
# gpu 7:   util   0%, mem  0.5 GB /  80.0 GB  | worker: down  | paused by alice for 9m 58s more

# --- as bob (whitelisted) — collision ---
ocp pause 7 5m
# error: gpu 7 reserved by alice (598s left)
#        (E_PAUSE_HELD)

# bob grabs a different idle GPU instead
ocp pause 0 5m
# ok: paused gpu 0 until 14:38 (5m)

ocp history -n 4
# 14:33  (now    )           bob   pause gpus=[0] duration_s=300
# 14:32  (1m ago)         alice    pause gpus=[7] duration_s=600
# 14:32  (1m ago)           bob   pause gpus=[7] duration_s=300   [E_PAUSE_HELD]
# 14:30  (3m ago)        (auto)   daemon_started pid=12345

# alice finishes early
ocp resume 7
# ok: released gpu(s): 7
```

### Config (`/root/.ocp/config.toml`)

The daemon starts with sensible defaults if the file is absent. Show the
defaults at any time with `ocp config list`. Example, with the keys you'll most
often touch:

```toml
[gpu]
indices = []                 # [] = all visible GPUs. Same-model only (no MIG).

[thresholds]
util_low      = 5            # %
util_high     = 30
mem_low       = 10           # % of total
mem_high      = 50
idle_debounce_s = 30         # how long a GPU must look idle before we spawn
busy_debounce_s = 5          # how quickly we yield when a real user appears

[worker]
workloads          = ["mem", "util"]   # subset of {"mem", "util"}; [] disables
mem_frac           = 0.5               # mean fraction of VRAM to hold
util_target        = 90                # 0–100, mean SM utilization
mem_noise_period_s = 5                 # reallocate every ~5s
util_noise_period_s = 5

[multiuser]
# Users allowed to run pause/resume. Match by name first, then numeric uid.
# Root is always allowed implicitly.
whitelist = ["alice", "bob", 1042]

[pause]
min_duration_s = 30          # shortest lease accepted
max_duration_s = 7200        # hard ceiling (2h)
```

Workflow to change anything:

```bash
sudo $EDITOR /root/.ocp/config.toml
sudo ocp config reload     # validates first; old snapshot stays on failure
```

### Authorization (recap)

| Tier         | Who                                      | Can run                                          |
| ------------ | ---------------------------------------- | ------------------------------------------------ |
| Anyone       | every local user                         | `status`, `history`, `config list`, `config path`, `daemon status` |
| Whitelisted  | uid 0, or uid/name in `multiuser.whitelist` | + `pause`, `resume`                              |
| Root         | uid 0 only                               | + `daemon on/off`, `config reload`               |

Caller identity is taken from the kernel via `SO_PEERCRED` — clients cannot
spoof it.

## Uninstall

### Root install

```bash
sudo ocp daemon off
# If you enabled the systemd unit:
sudo systemctl disable --now ocpd.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/ocpd.service
sudo systemctl daemon-reload 2>/dev/null || true
# Drop the package and its state:
sudo pip uninstall -y ocp
sudo rm -rf /root/.ocp /run/ocp /var/log/ocp
```

### Regular-user install

```bash
pip uninstall -y ocp
```

## Layout

See `src/ocp/` for the package and `tests/` for the test suite.
The optional systemd unit lives in `packaging/`.
Helper scripts (smoke test, canonical session) live in `scripts/`.
