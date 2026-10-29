# DSA minimal per-layer resident-capacity P0

## Purpose

This is the first runtime-capacity implementation, deliberately limited to one static per-layer override on the existing Sparse KV offload path.

- Global baseline: `sparse_kv_offload_config.topk_buffer_size`.
- Optional override: `sparse_kv_offload_config.layer_topk_buffer_sizes`.
- Unlisted layers keep the global baseline.
- Overrides are expansion-only and must be multiples of the cache block size.
- The first implementation rejects `fused_op_type=fused_copy_sfa` and `use_fused_overlap=true`.
- No TopK, eviction algorithm, Attention math, or online resizing logic is changed.

The runtime validates the configured layer names against the registered offload layers, allocates each layer's K/V resident buffers at its own capacity, builds layer-specific slot tables and resident block tables, and passes the matching capacity into the LRU planner. The server log contains an explicit allocation confirmation.

## Important scope limitation

The currently used native observation service does not enable Sparse KV offload by itself. A capacity override only takes effect when `sparse_kv_offload_config.enabled=true` selects the offload backend.

The smoke launcher therefore uses the existing 4-layer TP2 model and the offload debug setting `keep_device_kv_cache=true`. This retains the normal device KV cache and adds the resident buffers; it is for configuration/allocation/correctness validation only. Do not use its latency as a performance result or claim that it reduces HBM usage.

The offload path requires the supported A3 runtime and compiled Sparse KV offload operators. This first P0 also disables DSA context parallelism and fused offload modes. The script deliberately does not set `--kv-cache-dtype fp8`; the main-cache path should use BF16. It passes `--dtype bfloat16` and leaves KV cache dtype at its model-dtype default.

## Files changed

- `vllm_ascend/ascend_config.py`: static layer capacity config, validation and lookup.
- `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py`: per-layer buffer allocation, LRU tables/workspaces and actual allocation checks.
- `vllm_ascend/attention/sfa_kv_offload.py`: layer-specific resident tensor views and block tables.
- `vllm_ascend/worker/model_runner_v1.py`: pass the exact layer name into buffer allocation.
- `tests/ut/test_v2_sparse_offload_config.py`: config validation tests.
- `tests/ut/distributed/kv_transfer/sparse_kv_offload/test_sparse_kv_offload_manager.py`: mocked allocation-shape test.
- `run_dsa_capacity_minimal_tp2.sh`: starts an isolated 4-layer TP2 debug service in the background.
- `validate_dsa_capacity_minimal_tp2.sh`: waits for health, checks actual allocation, sends one request and optionally analyzes TopK trace.

The existing full-model TP8 launcher and observer-only startup scripts are not modified.

## 1. Pull and run unit tests

From the repository root:

```bash
cd /home/z00945955/IO/vllm-ascend
git fetch origin
git checkout experiment/dsa-layer-resident-capacity-minimal-p0
git pull --ff-only origin experiment/dsa-layer-resident-capacity-minimal-p0

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/local/python3.12.13/bin/python -m pytest -q \
  tests/ut/test_v2_sparse_offload_config.py \
  tests/ut/distributed/kv_transfer/sparse_kv_offload/test_sparse_kv_offload_manager.py

/usr/local/python3.12.13/bin/python -m py_compile \
  vllm_ascend/ascend_config.py \
  vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py \
  vllm_ascend/attention/sfa_kv_offload.py \
  vllm_ascend/worker/model_runner_v1.py

bash -n benchmarks/dsa/capacity_trace/run_dsa_capacity_minimal_tp2.sh
bash -n benchmarks/dsa/capacity_trace/validate_dsa_capacity_minimal_tp2.sh
```

These tests validate configuration rules and the per-layer buffer allocation shape. The NPU smoke test below is still required to prove the actual A3 runtime path.

## 2. Launch the minimal TP2 service

Stop any existing service using NPU 6,7 first. The script uses port 8081 so it does not collide with the existing port-8080 service. It returns after starting the service in the background; it does not replace or edit existing launch scripts.

```bash
cd /home/z00945955/IO/vllm-ascend
bash benchmarks/dsa/capacity_trace/run_dsa_capacity_minimal_tp2.sh
```

Defaults:

- Model: `/home/z00945955/glm5-mtp-4l`
- TP=2, visible devices `6,7`
- Baseline capacity: 4096 tokens
- Override: `model.layers.0.self_attn.attn = 6144` tokens
- Port: 8081
- Fixed request: 12,000 input tokens, 64 output tokens

You can select another target layer or larger expansion without editing the script:

```bash
TARGET_LAYER=model.layers.0.self_attn.attn BASE_CAPACITY=4096 TARGET_CAPACITY=6144 \
  bash benchmarks/dsa/capacity_trace/run_dsa_capacity_minimal_tp2.sh
```

The target layer must be an exact registered host-resident SFA layer name. If it is not, startup fails with a list of registered layers rather than silently ignoring the override.

## 3. Validate allocation and send the fixed request

```bash
cd /home/z00945955/IO/vllm-ascend
bash benchmarks/dsa/capacity_trace/validate_dsa_capacity_minimal_tp2.sh
```

The validator:

1. Waits up to 10 minutes for `/health`.
2. Requires the log marker `DSA_LAYER_RESIDENT_CAPACITY_ALLOCATED` for the exact target layer and requested capacity. This checks the actual tensor allocation, not just config parsing.
3. Sends one fixed 12K/64 request and checks that the server remains healthy.
4. Runs the offline analyzer if TopK trace files were emitted. If no trace files were emitted, it reports that separately; request/allocation PASS does not imply trace analysis PASS.

Useful log checks:

```bash
TRACE_DIR="$(cat /tmp/dsa_capacity_minimal_tp2_trace_dir)"
grep -F 'DSA_LAYER_RESIDENT_CAPACITY' "$TRACE_DIR/server.log"
tail -n 120 "$TRACE_DIR/server.log"
```

## PASS / STOP criteria

**P0 PASS**

- Unit tests and syntax checks pass.
- Server reaches healthy state.
- The log confirms the target layer's actual K/V buffer shapes use 6144 tokens while unlisted layers retain 4096.
- The fixed request returns the requested prompt and completion token counts.
- Service remains healthy after the request.

**STOP**

- Config parser rejects the model or an unsupported mode.
- Target layer is not a registered offload layer.
- Allocated K/V buffer shape differs from the configured capacity.
- Service fails, request returns HTTP 500, or output token counts are wrong.
- Any runtime correctness failure appears when the override is enabled.

## What this P0 does not prove

- It does not prove a production P/D-offload performance gain.
- `keep_device_kv_cache=true` is debug-only and retains the device KV cache.
- A single successful request proves the code path is connected, not that the capacity planner is optimal.
- TP2/4-layer results cannot be generalized to the full 78-layer TP8 service.
- Performance comparison is a later phase after this smoke test passes, using the real target backend and matched physical HBM budgets.
