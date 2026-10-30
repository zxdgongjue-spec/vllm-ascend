#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/z00945955/IO/vllm-ascend}"
PY=/usr/local/python3.12.13/bin/python
TRACE_DIR="${1:-}"
if [ -z "$TRACE_DIR" ] && [ -f /tmp/dsa_capacity_minimal_tp2_trace_dir ]; then
  TRACE_DIR="$(cat /tmp/dsa_capacity_minimal_tp2_trace_dir)"
fi
if [ -z "$TRACE_DIR" ] || [ ! -d "$TRACE_DIR" ]; then
  echo "ERROR: Trace directory not found. Start the service with run_dsa_capacity_minimal_tp2.sh first."
  exit 2
fi

PORT="$(cat "$TRACE_DIR/port.txt")"
TARGET_LAYER="$(cat "$TRACE_DIR/target_layer.txt")"
BASE_CAPACITY="$(cat "$TRACE_DIR/base_capacity.txt")"
TARGET_CAPACITY="$(cat "$TRACE_DIR/target_capacity.txt")"
SERVER_LOG="$TRACE_DIR/server.log"
PLOG_DIR="$TRACE_DIR/plog"
SERVER_TRACE_DIR="$TRACE_DIR/server-trace"
SERVER_PID="$(cat "$TRACE_DIR/server.pid")"

cd "$REPO"

cleanup_server_tree() {
  echo "CLEANUP_SERVER_TREE: pid=$SERVER_PID"
  "$PY" - "$SERVER_PID" <<'PY' || true
import os
import signal
import sys
import time
from pathlib import Path

root = int(sys.argv[1])
children = {}
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    pid = int(proc.name)
    try:
        stat = (proc / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        ppid = int(fields[1])
        children.setdefault(ppid, []).append(pid)
    except (OSError, ValueError, IndexError):
        continue

order = []
seen = set()
stack = [root]
while stack:
    pid = stack.pop()
    if pid in seen:
        continue
    seen.add(pid)
    order.append(pid)
    stack.extend(children.get(pid, []))

for pid in reversed(order):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        print("CLEANUP_PERMISSION_DENIED pid=", pid)
time.sleep(2)
for pid in reversed(order):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        print("CLEANUP_PERMISSION_DENIED pid=", pid)
print("CLEANUP_SIGNALLED_PIDS =", order)
PY
}

FATAL_PATTERN='EngineCore failed to start|Worker failed with error|ERR00100|listen the port [0-9]+ failed|already been bound|HRaSocketListenStart|CreateChannels failed|HcclAllocAlgResourceAiv.*ret\[4\]'

echo "===== 1. Wait for service ====="
echo "TRACE_DIR=$TRACE_DIR"
echo "PORT=$PORT"
echo "TARGET_LAYER=$TARGET_LAYER"
echo "BASE_CAPACITY=$BASE_CAPACITY"
echo "TARGET_CAPACITY=$TARGET_CAPACITY"

READY=0
for I in $(seq 1 120); do
  if grep -ERiq "$FATAL_PATTERN" "$SERVER_LOG" "$PLOG_DIR" 2>/dev/null; then
    echo "ERROR: fatal engine/worker/HCCL startup failure detected before API health."
    echo "===== MATCHING FATAL LOGS ====="
    grep -ERni "$FATAL_PATTERN" "$SERVER_LOG" "$PLOG_DIR" 2>/dev/null | tail -n 120 || true
    echo "===== SERVER LOG TAIL ====="
    tail -n 240 "$SERVER_LOG" || true
    cleanup_server_tree
    exit 7
  fi
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    READY=1
    echo "API_SERVER_READY=1"
    echo "WAIT_ITERATION=$I"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: API server process exited before becoming healthy."
    tail -n 160 "$SERVER_LOG" || true
    cleanup_server_tree
    exit 3
  fi
  sleep 5
done

if [ "$READY" -ne 1 ]; then
  echo "ERROR: API server did not become healthy within 10 minutes."
  echo "===== PLOG ERROR TAIL ====="
  grep -ERni 'ERROR|already been bound|listen the port|CreateChannels failed' "$PLOG_DIR" 2>/dev/null | tail -n 120 || true
  tail -n 160 "$SERVER_LOG" || true
  cleanup_server_tree
  exit 4
fi

echo
echo "===== 2. Verify configured capacity was actually allocated ====="
if ! grep -F "DSA_LAYER_RESIDENT_CAPACITY_ALLOCATED: layer=$TARGET_LAYER tokens=$TARGET_CAPACITY" "$SERVER_LOG" >/dev/null; then
  echo "ERROR: did not find the actual per-layer allocation confirmation in server.log."
  echo "Expected: DSA_LAYER_RESIDENT_CAPACITY_ALLOCATED: layer=$TARGET_LAYER tokens=$TARGET_CAPACITY"
  tail -n 200 "$SERVER_LOG" || true
  cleanup_server_tree
  exit 5
fi
grep -F "DSA_LAYER_RESIDENT_CAPACITY" "$SERVER_LOG" | tail -n 20

echo
echo "===== 3. Send one fixed request: 12K input / 64 output ====="
if "$PY" benchmarks/dsa/capacity_trace/request_dsa_capacity_trace.py \
    --base-url "http://127.0.0.1:$PORT" \
    --model glm-capacity \
    --input-tokens 12000 \
    --output-tokens 64 \
    --seed 1024 \
    --timeout 600 \
    --trace-dir "$TRACE_DIR"; then
  echo "REQUEST_STATUS=PASS"
else
  RC=$?
  echo "REQUEST_STATUS=FAIL rc=$RC"
  tail -n 200 "$SERVER_LOG" || true
  cleanup_server_tree
  exit "$RC"
fi

echo
echo "===== 4. Verify service remains healthy ====="
if ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "ERROR: service became unhealthy after the request."
  tail -n 200 "$SERVER_LOG" || true
  cleanup_server_tree
  exit 6
fi
echo "POST_REQUEST_HEALTH=PASS"

echo
echo "===== 5. Analyze trace if the offload path emitted records ====="
if find "$SERVER_TRACE_DIR" -maxdepth 1 -type f -name 'topk_trace_rank*_pid*.jsonl' -print -quit | grep -q .; then
  "$PY" benchmarks/dsa/analyze_dsa_resident_capacity.py \
    --trace-dir "$SERVER_TRACE_DIR" \
    --output-dir "$TRACE_DIR/analysis" \
    --capacities "$BASE_CAPACITY,$TARGET_CAPACITY"
  echo "TRACE_ANALYSIS=PASS"
else
  echo "TRACE_ANALYSIS=NOT_RUN"
  echo "WARNING: request and allocation checks passed, but no TopK trace JSONL was emitted by this offload path."
  echo "Do not treat this as a performance result; inspect observer-hook coverage separately."
fi

echo
echo "MINIMAL_LAYER_CAPACITY_SMOKE=PASS"
echo "TRACE_DIR=$TRACE_DIR"
echo "SERVER_LOG=$SERVER_LOG"
