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
SERVER_TRACE_DIR="$TRACE_DIR/server-trace"
SERVER_PID="$(cat "$TRACE_DIR/server.pid")"

cd "$REPO"

echo "===== 1. Wait for service ====="
echo "TRACE_DIR=$TRACE_DIR"
echo "PORT=$PORT"
echo "TARGET_LAYER=$TARGET_LAYER"
echo "BASE_CAPACITY=$BASE_CAPACITY"
echo "TARGET_CAPACITY=$TARGET_CAPACITY"

READY=0
for I in $(seq 1 120); do
  if grep -Eq 'EngineCore failed to start|Worker failed with error|ERR00100 PTA call acl api failed' "$SERVER_LOG" 2>/dev/null; then
    echo "ERROR: fatal engine/worker startup failure detected before API health."
    tail -n 240 "$SERVER_LOG" || true
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
    exit 3
  fi
  sleep 5
done

if [ "$READY" -ne 1 ]; then
  echo "ERROR: API server did not become healthy within 10 minutes."
  tail -n 160 "$SERVER_LOG" || true
  exit 4
fi

echo
echo "===== 2. Verify configured capacity was actually allocated ====="
if ! grep -F "DSA_LAYER_RESIDENT_CAPACITY_ALLOCATED: layer=$TARGET_LAYER tokens=$TARGET_CAPACITY" "$SERVER_LOG" >/dev/null; then
  echo "ERROR: did not find the actual per-layer allocation confirmation in server.log."
  echo "Expected: DSA_LAYER_RESIDENT_CAPACITY_ALLOCATED: layer=$TARGET_LAYER tokens=$TARGET_CAPACITY"
  tail -n 200 "$SERVER_LOG" || true
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
  exit "$RC"
fi

echo
echo "===== 4. Verify service remains healthy ====="
if ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "ERROR: service became unhealthy after the request."
  tail -n 200 "$SERVER_LOG" || true
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
