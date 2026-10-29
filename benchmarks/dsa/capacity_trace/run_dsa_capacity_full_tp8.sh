#!/usr/bin/env bash
set -euo pipefail

REPO=/home/z00945955/IO/vllm-ascend
MODEL="${MODEL:-/home/z00945955/GLM-5.3-w4a4c8-mxfp4}"
PORT="${PORT:-8080}"
LOCAL_IP="${LOCAL_IP:-141.62.38.16}"
NIC_NAME="${NIC_NAME:-enp34s0f1}"
PY=/usr/local/python3.12.13/bin/python

cd "$REPO"

TRACE_SCRIPT=benchmarks/dsa/capacity_trace/run_dsa_capacity_trace.sh
REQUEST_SCRIPT=benchmarks/dsa/capacity_trace/request_dsa_capacity_trace.py
ANALYZER_SCRIPT=benchmarks/dsa/analyze_dsa_resident_capacity.py

# 1. Check the model, committed collection scripts, analyzer, and observer hook.
test -f "$MODEL/config.json"
test -f "$TRACE_SCRIPT"
test -f "$REQUEST_SCRIPT"
test -f "$ANALYZER_SCRIPT"
grep -q 'observe_sfa_topk' vllm_ascend/attention/sfa_v1.py

# 2. Ensure the active Python import resolves to this checkout and the model
#    is not an accidentally reduced-layer model.
"$PY" - "$REPO" "$MODEL/config.json" "$PORT" <<'PY'
import importlib
import json
import socket
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2])
port = int(sys.argv[3])

module = importlib.import_module("vllm_ascend")
loaded = Path(module.__file__).resolve()
print("ASCEND_SOURCE =", loaded)

if not loaded.is_relative_to(repo):
    raise SystemExit(
        f"STOP: loaded Ascend source is outside expected repo: {loaded}"
    )

config = json.loads(config_path.read_text())
sections = [
    config,
    config.get("text_config", {}),
    config.get("language_config", {}),
    config.get("model_config", {}),
]
layers = next(
    (
        item.get("num_hidden_layers")
        for item in sections
        if isinstance(item, dict)
        and item.get("num_hidden_layers") is not None
    ),
    None,
)
print("MODEL_NUM_HIDDEN_LAYERS =", layers)

if layers is not None and int(layers) <= 4:
    raise SystemExit("STOP: this appears to be a reduced-layer model")

with socket.socket() as sock:
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        raise SystemExit(
            f"STOP: port {port} is occupied; stop the previous service first"
        )
PY

# 3. Create an isolated trace directory for this run.
TRACE_DIR="/home/z00945955/IO/results/dsa-capacity-full-tp8-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$TRACE_DIR/plog"
printf '%s\n' "$TRACE_DIR" > /tmp/dsa_capacity_full_tp8_trace_dir

export TRACE_DIR
export VLLM_ASCEND_DSA_CAPACITY_TRACE_MAX_RECORDS=10000
export ASCEND_PROCESS_LOG_PATH="$TRACE_DIR/plog"
export ASCEND_GLOBAL_LOG_LEVEL=3

# 4. NPU and communication environment. Preserve the previously used values.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=0

export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
# TEMP_DISABLED: export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=400
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1

echo "TRACE_DIR=$TRACE_DIR"
echo "MODEL=$MODEL"
echo "TP=8"
echo "OBSERVATION_ONLY=1"
echo "TASK_QUEUE_ENABLE=$TASK_QUEUE_ENABLE"
echo "ASYNC_SCHEDULING=not-enabled-by-this-script"
echo "HCCL_IF_IP=unset-by-this-script"

# 5. Start the full model in eager mode for observation. Do not use this run
#    for performance comparisons; trace mode copies TopK data to host.
exec bash "$TRACE_SCRIPT" -- \
  vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --distributed-executor-backend mp \
    --data-parallel-size 1 \
    --data-parallel-size-local 1 \
    --data-parallel-address "$LOCAL_IP" \
    --data-parallel-rpc-port 13389 \
    --tensor-parallel-size 8 \
    --seed 1024 \
    --served-model-name glm \
    --enable-expert-parallel \
    --max-num-seqs 1 \
    --max-model-len 131072 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enforce-eager \
    --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}' \
    --kv-cache-dtype fp8 \
    --attention_config.indexer_kv_dtype fp8 \
    --additional_config '{"ascend_compilation_config":{"enable_super_kernel":false},"enable_cpu_binding":"True","multistream_overlap_shared_expert":true,"enable_dsa_cp":true}' \
    2>&1 | tee "$TRACE_DIR/server.log"
