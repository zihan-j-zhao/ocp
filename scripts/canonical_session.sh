#!/usr/bin/env bash
# Canonical user session under a tmp prefix, runs unprivileged.
# Mirrors the documented session (ocp daemon on / pause / history / off), but
# `ocp daemon on|off` now requires root in the production model, so this
# script invokes `python -m ocp.daemon` directly and sends SIGTERM to stop.
set -u

TMP=$(mktemp -d)
mkdir -p "$TMP/etc" "$TMP/run" "$TMP/log"

# Watch GPU 7 only (a fake config — useful to test pause without a real GPU 7).
# Whitelist the current user so pause/resume work without sudo.
cat > "$TMP/etc/config.toml" <<EOF
[gpu]
indices = [7]

[pause]
min_duration_s = 1
max_duration_s = 7200

[multiuser]
whitelist = ["$USER"]
EOF

export OCP_RUN_DIR="$TMP/run"
export OCP_CONFIG_DIR="$TMP/etc"
export OCP_LOG_DIR="$TMP/log"

cleanup() {
    [[ -n "${DAEMON_PID:-}" ]] && kill -TERM "$DAEMON_PID" 2>/dev/null
    wait "${DAEMON_PID:-}" 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT

echo "==> starting daemon (python -m ocp.daemon — bypasses root check)"
python -m ocp.daemon &
DAEMON_PID=$!
sleep 1.5

echo "==> ocp daemon status"
ocp daemon status

echo "==> ocp pause 7 10m"
ocp pause 7 10m

echo "==> ocp status"
ocp status

echo "==> ocp history"
ocp history -n 10

echo "==> stopping daemon (SIGTERM)"
kill -TERM "$DAEMON_PID" 2>/dev/null
wait "$DAEMON_PID" 2>/dev/null
echo "done"
