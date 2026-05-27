#!/usr/bin/env bash
# End-to-end smoke test: bring up a daemon under a tmp prefix and exercise the CLI.
set -u

TMP=$(mktemp -d)
mkdir -p "$TMP/etc" "$TMP/run" "$TMP/log"

export OCP_RUN_DIR="$TMP/run"
export OCP_CONFIG_DIR="$TMP/etc"
export OCP_LOG_DIR="$TMP/log"

cleanup() {
    [[ -n "${DAEMON_PID:-}" ]] && kill -TERM "$DAEMON_PID" 2>/dev/null
    wait "${DAEMON_PID:-}" 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT

echo "tmp dir: $TMP"
echo "==> starting daemon"
python -m ocp.daemon &
DAEMON_PID=$!
sleep 1.5

echo "==> ocp daemon status"
ocp daemon status

echo "==> ocp status"
ocp status

echo "==> ocp history"
ocp history -n 5

echo "==> ocp config list (first 20 lines)"
ocp config list | head -20

echo "==> stopping daemon"
kill -TERM "$DAEMON_PID" 2>/dev/null
wait "$DAEMON_PID" 2>/dev/null
echo "done"
