#!/usr/bin/env bash
set -euo pipefail

# Minimal P0 runtime smoke test for a single-layer resident-capacity override.
# This is a debug/correctness run, not a performance benchmark:
# keep_device_kv_cache=true retains the normal device KV cache.

REPO="${REPO:-/home/z00945955/IO/vllm-ascend}"
MODEL="${MODEL:-/home/z00945955/glm5-mtp-4l}"
PORT="${PORT:-8081}"
LOCAL_IP="${LOCAL_IP:-}"
VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-6,7}"
TARGET_LAYER="${TARGET_LAYER:-model.layers.0.self_attn.attn}"
BASE_CAPACITY="${BASE_CAPACITY:-4096}"
TARGET_CAPACITY="${TARGET_CAPACITY:-6144}"
PY=/usr/local/python3.12.13/bin/python
STAMP="$(date +%Y%m%d-%H%M%S)"
RESULT_ROOT="${RESULT_ROOT:-/home/z00945955/IO/results}"
TRACE_DIR="${TRACE_DIR:-$RESULT_ROOT/dsa-capacity-minimal-tp2-$STAMP-$$}"
SERVER_LOG="$TRACE_DIR/server.log"

cd "$REPO"

echo "===== 1. Preflight ====="
test -x "$PY" || { echo "ERROR: Python not found: $PY"; exit 2; }
test -f "$MODEL/config.json" || { echo "ERROR: model config not found: $MODEL/config.json"; exit 2; }
grep -q 'layer_topk_buffer_sizes' vllm_ascend/ascend_config.py || {
  echo "ERROR: current checkout does not contain per-layer capacity support."
  exit 2
}

if [ -z "$LOCAL_IP" ]; then
  LOCAL_IP=$("$PY" - <<'PY'
import socket
ip = ""
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(("1.1.1.1", 80))
    ip = sock.getsockname()[0]
    sock.close()
except OSError:
    pass
if not ip:
    import subprocess
    try:
        ip = subprocess.check_output(["hostname", "-I"], text=True).split()[0]
    except Exception:
        pass
print(ip)
PY
)
fi
if ! [[ "$LOCAL_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "ERROR: could not determine a valid local IP; rerun with LOCAL_IP=<IP>."
  exit 2
fi

"$PY" - "$REPO" "$MODEL/config.json" "$PORT" "$BASE_CAPACITY" "$TARGET_CAPACITY" <<'PY'
import importlib
import json
import socket
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2])
port, baseline, target = map(int, sys.argv[3:])
module = importlib.import_module("vllm_ascend")
loaded = Path(module.__file__).resolve()
print("ASCEND_SOURCE =", loaded)
if not loaded.is_relative_to(repo):
    raise SystemExit(f"STOP: Ascend source is outside expected repo: {loaded}")

config = json.loads(config_path.read_text())
sections = [config, config.get("text_config", {}), config.get("language_config", {}), config.get("model_config", {})]
layers = next((x.get("num_hidden_layers") for x in sections if isinstance(x, dict) and x.get("num_hidden_layers") is not None), None)
print("MODEL_NUM_HIDDEN_LAYERS =", layers)
if layers is not None and int(layers) > 4:
    raise SystemExit("STOP: this smoke script is intended for the 4-layer TP2 model.")

if baseline <= 0 or target <= baseline:
    raise SystemExit("STOP: TARGET_CAPACITY must be greater than BASE_CAPACITY.")
if baseline % 128 or target % 128:
    raise SystemExit("STOP: both capacities must be multiples of block_size=128.")

with socket.socket() as sock:
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        raise SystemExit(f"STOP: port {port} is occupied; choose another PORT or stop that service.")
PY

mkdir -p "$TRACE_DIR/plog" "$TRACE_DIR/server-trace"
printf '%s\n' "$TRACE_DIR" > /tmp/dsa_capacity_minimal_tp2_trace_dir
printf '%s\n' "$PORT" > "$TRACE_DIR/port.txt"
printf '%s\n' "$TARGET_LAYER" > "$TRACE_DIR/target_layer.txt"
printf '%s\n' "$BASE_CAPACITY" > "$TRACE_DIR/base_capacity.txt"
printf '%s\n' "$TARGET_CAPACITY" > "$TRACE_DIR/target_capacity.txt"

"$PY" - "$TRACE_DIR/additional_config.json" "$TARGET_LAYER" "$BASE_CAPACITY" "$TARGET_CAPACITY" <<'PY'
import json
import sys
from pathlib import Path

out, layer, baseline, target = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
config = {
    "ascend_compilation_config": {"enable_super_kernel": False},
    "enable_dsa_cp": False,
    "sparse_kv_offload_config": {
        "enabled": True,
        "topk_buffer_size": baseline,
        "layer_topk_buffer_sizes": {layer: target},
        "dram_size_per_dp_GB": 128,
        "keep_device_kv_cache": True,
        "fused_op_type": "none",
        "use_fused_overlap": False,
    },
}
Path(out).write_text(json.dumps(config, separators=(",", ":")) + "\n", encoding="utf-8")
print("ADDITIONAL_CONFIG_FILE =", out)
print("BASE_CAPACITY =", baseline)
print("TARGET_LAYER =", layer)
print("TARGET_CAPACITY =", target)
PY

ADDITIONAL_CONFIG="$(cat "$TRACE_DIR/additional_config.json")"

# Do not delete or reuse another service's Prometheus multiprocess directory.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export HCCL_IF_IP="$LOCAL_IP"
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
export HCCL_BUFFSIZE=400
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_ASYNC_ERROR_HANDLING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_PROCESS_LOG_PATH="$TRACE_DIR/plog"
export ASCEND_GLOBAL_LOG_LEVEL=3
export PROMETHEUS_MULTIPROC_DIR="/dev/shm/vllm_metrics_dsa_capacity_${STAMP}_$$"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
export VLLM_ASCEND_DSA_CAPACITY_TRACE_DIR="$TRACE_DIR/server-trace"
export VLLM_ASCEND_DSA_CAPACITY_TRACE_MAX_RECORDS=10000
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1

echo
echo "===== 2. Launch debug offload service ====="
echo "REPO=$REPO"
echo "MODEL=$MODEL"
echo "PORT=$PORT"
echo "LOCAL_IP=$LOCAL_IP"
echo "VISIBLE_DEVICES=$VISIBLE_DEVICES"
echo "TRACE_DIR=$TRACE_DIR"
echo "SERVER_LOG=$SERVER_LOG"
echo "CAPACITY_MODE=single-layer static expansion; keep_device_kv_cache=true"
echo "NOTE: this run validates configuration/allocation/correctness only; do not use its latency as a performance result."

nohup vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --distributed-executor-backend mp \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --data-parallel-address "$LOCAL_IP" \
  --data-parallel-rpc-port 13390 \
  --tensor-parallel-size 2 \
  --seed 1024 \
  --served-model-name glm-capacity \
  --enable-expert-parallel \
  --max-num-seqs 1 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.95 \
  --quantization ascend \
  --dtype bfloat16 \
  --enforce-eager \
  --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}' \
  --attention_config.indexer_kv_dtype fp8 \
  --additional_config "$ADDITIONAL_CONFIG" \
  > "$SERVER_LOG" 2>&1 < /dev/null &

SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$TRACE_DIR/server.pid"
printf '%s\n' "$SERVER_PID" > /tmp/dsa_capacity_minimal_tp2_server_pid

echo "SERVER_PID=$SERVER_PID"
echo "SERVER_LOG=$SERVER_LOG"
echo "TRACE_DIR=$TRACE_DIR"
echo "Launch command returned; service is running in the background."
echo "Next: bash benchmarks/dsa/capacity_trace/validate_dsa_capacity_minimal_tp2.sh"
