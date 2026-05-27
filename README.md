# OCP — Occupy Compute Process

A client/server tool that watches GPU utilization & memory and, when a device is idle, launches a worker process to keep it occupied. A CLI client talks to a background daemon.

**Requirements**

- Linux
- Python ≥ 3.10
- PyTorch ≥ 2.6 with CUDA 12.8

---

## Quick Start

### **Root** User Actions

#### Setup

```bash
# 1. clone the repo
git clone https://github.com/zihan-j-zhao/ocp.git
cd ocp

# 2. create the environment (not necessarily conda)
conda create -yn ocp python=3.12
conda activate ocp

# 3. install ocp
pip install '.[worker]'

# 4. copy default config
mkdir -p ~/.ocp          # /root/.ocp
cp config.toml ~/.ocp/   # /root/.ocp/config.toml

# 5. start the daemon
ocp daemon on
```

Now the daemon should be up. Verify with:

```bash
ocp daemon status
ocp status                       # per-GPU snapshot
```

#### Whitelisting

By default, only root has control over the daemon process. However, you may grant some control to users by listing their user IDs or usernames in the `multiuser.whitelist` field in the config file. After the change, you may reload the config by

```bash
ocp config reload
```


#### Teardown

When you want to tear down the service, simply do

```bash
ocp daemon off
```

This command will shut down all worker processes gracefully.


### Normal User Actions

#### Setup

```bash
# 1. clone the repo
git clone https://github.com/zihan-j-zhao/ocp.git
cd ocp

# 2. create the environment (not necessarily conda)
conda create -yn ocp python=3.12
conda activate ocp

# 3. install ocp
pip install '.[worker]'
```

#### Request GPUs

When you want to submit a job to some GPUs but notice ocp workers are occupying them, you can let daemon stop monitoring those GPUs for a short period of time by

```bash
ocp pause 0,1,2,3 10m  # pause 10 minutes on GPU0-3
```

Within the next 10 minutes, you can start your own job, and the daemon won't spawn another worker process to hold the resources you requested.

---

## Full List of Commands

All commands talk to the running daemon over a local UNIX socket. Caller
identity is taken from the kernel (`SO_PEERCRED`) and cannot be spoofed.

| Command                                  | Who           | What it does                                            |
| ---------------------------------------- | ------------- | ------------------------------------------------------- |
| `ocp daemon on`                          | **root**      | Start the daemon (idempotent).                          |
| `ocp daemon off`                         | **root**      | Stop the daemon (SIGTERM, graceful — tears down workers).|
| `ocp daemon status`                      | anyone        | Print whether the daemon is up, its pid, and socket path.|
| `ocp status`                             | anyone        | Per-GPU snapshot: util, memory, worker, active lease.   |
| `ocp pause <gpu> <duration>`             | whitelisted   | Reserve one or more GPUs for a duration (e.g. `5m`).    |
| `ocp resume <gpu>`                       | whitelisted   | Release a reservation early (you must be the holder).   |
| `ocp history [-n N] [--since D] [--user U] [--mine]` | anyone | Recent events (pause/resume/spawn/…).                   |
| `ocp config list`                        | anyone        | Print the in-memory config snapshot.                    |
| `ocp config reload`                      | **root**      | Re-read `/root/.ocp/config.toml` and swap atomically.   |
| `ocp config path`                        | anyone        | Print the config file path.                             |

**Arguments**

- **GPU spec.** A single index or comma-list — `7`, `0,2,3`.
- **Duration.** Combinations of `s` / `m` / `h`, e.g. `5s`, `90s`, `5m`,
  `1h30m`. No fractions.
- **`--json`.** Append to any IPC command for machine-readable output.
- **`--socket <path>`.** Override the UDS path (useful when running an
  unprivileged daemon under `$OCP_RUN_DIR`).

**Authorization tiers**

| Tier         | Who                                          | Can run                                                            |
| ------------ | -------------------------------------------- | ------------------------------------------------------------------ |
| Anyone       | every local user                             | `status`, `history`, `config list`, `config path`, `daemon status` |
| Whitelisted  | uid 0, or uid/name in `multiuser.whitelist`  | + `pause`, `resume`                                                |
| Root         | uid 0 only                                   | + `daemon on/off`, `config reload`                                 |

---

## FAQs

**The daemon won't yield when I start my job. Why?**
By design. OCP no longer auto-yields based on foreign GPU activity — you must
`ocp pause <gpu> <duration>` first. The pause command tears down the OCP
worker on the requested devices and prevents new spawns for the duration of
your lease.

**What if I forget to pause?**
Your job and the OCP worker will share the GPU. Util/mem readings will
overlap, and your job may OOM if the worker's `mem_frac` leaves too little
headroom. Always `ocp pause` first.

**How does the daemon decide a GPU is idle enough to occupy?**
Each tick (`general.poll_interval_s`, default 2s) it samples util and memory.
A GPU counts as quiet when *both* `util_pct < thresholds.util_low` **and**
`mem_used_pct < thresholds.mem_low`. After `thresholds.idle_debounce_s`
seconds of sustained quiet, it spawns a worker.

**What workload does the worker actually run?**
Two cooperating asyncio controllers in one process. The memory controller
allocates and periodically reallocates a CUDA tensor at a randomized size
above `worker.mem_frac` of VRAM. The compute controller runs a `bfloat16`
matmul loop with a duty cycle calibrated to `worker.util_target`. Both
deliberately overshoot their targets by ~5% so the daemon's monitor never
samples its own worker as idle. Disable individually via `worker.workloads`
(e.g. `["mem"]` for memory-only).

**Can I run the daemon as a non-root user?**
For development, yes. Set `OCP_RUN_DIR`, `OCP_CONFIG_DIR`, `OCP_LOG_DIR` to
writable locations and start it directly:

```bash
export OCP_RUN_DIR=$HOME/.local/run/ocp
export OCP_CONFIG_DIR=$HOME/.config/ocp
export OCP_LOG_DIR=$HOME/.local/share/ocp
python -m ocp.daemon
```

`ocp daemon on/off` still refuses non-root callers — that gate is only on the
client subcommand. See [`scripts/canonical_session.sh`](scripts/canonical_session.sh)
for a full example.

**Does it support MIG?**
No. MIG-enabled devices are detected and skipped at NVML open time.

**What happens to my pause if the daemon restarts?**
Active leases are persisted to `$OCP_RUN_DIR/state.json` after every change
and reloaded on startup. A pause from before a daemon restart is still
honored — only the unexpired remainder.

**Can I run two daemons on one host?**
No. The pidfile is `flock`-protected; a second `ocpd` will exit with an error.

**I changed `config.toml`. How do I apply it without restarting?**
`sudo ocp config reload`. The new config is validated before being swapped in;
on failure the running snapshot stays as-is. Caveat: already-running workers
keep their old `[worker]` settings — changes take effect on the *next* spawn.

**Where are the logs?**
`/var/log/ocp/ocpd.log` by default (or `$OCP_LOG_DIR/ocpd.log`). Bump
`general.log_level` to `"DEBUG"` for more detail. Worker stderr lines
(`mem: held X GiB …`, `util: duty=Y size=Z`) also land there.

**How do I see what just happened?**
`ocp history -n 50` (add `--mine` to filter to your own actions, `--since 1h`
to bound by time, `--json` for machine output). Events include
`worker_spawned`, `lease_expired`, `pause`, `resume`, and any errors.

**How do I uninstall?**

```bash
sudo ocp daemon off
sudo pip uninstall -y ocp
sudo rm -rf /root/.ocp /var/log/ocp     # config + logs (optional)
```

For a regular-user install: `pip uninstall -y ocp`.
