# DSA 4-layer resident-capacity observation (P0)

## Baseline and scope

- Repository: zxdgongjue-spec/vllm-ascend
- Base tag: v0.30.0rc2
- Base commit: 8fde6d4e335fe300e2b949aab0e31659cce0811d
- Branch: experiment/dsa-layer-resident-capacity-4l-observe-p0
- Runtime hook: SFA TopK selection in vllm_ascend/attention/sfa_v1.py, immediately before the selected indices are passed to the sparse-attention execution path.

The observer is disabled unless VLLM_ASCEND_DSA_CAPACITY_TRACE_DIR is set before vLLM starts. It records only pure-decode batches and skips profile/capture contexts where detectable. It does not alter TopK indices, resident-cache state, eviction, KV transfers, or attention inputs. Trace mode copies indices to CPU and writes JSONL, so its latency must not be used as the native performance baseline.

## Files

- vllm_ascend/attention/dsa_capacity_observer.py: opt-in observer, one JSONL record per SFA layer forward.
- benchmarks/dsa/request_dsa_capacity_trace.py: uses the server /tokenize endpoint to build an exact token-ID prompt and sends one fixed-length completion request.
- benchmarks/dsa/analyze_dsa_resident_capacity.py: emits per-layer summaries and an offline LRU capacity curve; optional dynamic-programming allocation under a per-process token budget.
- benchmarks/dsa/run_dsa_capacity_trace.sh: exports the trace environment before starting the server.
- tests/ut/attention/test_dsa_capacity_observer.py and test_dsa_capacity_analyzer.py: unit coverage.

## 4-layer validation procedure

1. Fetch the branch into the checkout that will actually be imported by the running vLLM process. Do not switch the source checkout underneath a live server. Verify vllm_ascend.__file__ points to that checkout before testing.
2. Run the server with the trace environment set before Python starts. Use the existing 4-layer model and existing TP=2 settings. For the trace-only run, add the global --enforce-eager option to avoid graph-capture interactions. Keep all other model and DSA settings unchanged.
3. Example trace directory:
   /home/z00945955/IO/results/dsa-capacity-4l-YYYYMMDD-HHMMSS
4. Launch the existing vllm serve command through the wrapper:
   TRACE_DIR=/absolute/path/to/trace-dir bash benchmarks/dsa/run_dsa_capacity_trace.sh -- vllm serve /home/z00945955/glm5-mtp-4l [existing server arguments, including --enforce-eager]
5. After /health returns 200, issue one request:
   python benchmarks/dsa/request_dsa_capacity_trace.py --base-url http://127.0.0.1:8080 --model glm --input-tokens 12000 --output-tokens 64 --trace-dir /absolute/path/to/trace-dir
6. Analyze:
   python benchmarks/dsa/analyze_dsa_resident_capacity.py --trace-dir /absolute/path/to/trace-dir --capacities 0,1024,2048,4096,6144,8192,10240,12288
7. Optional candidate allocation, after confirming the target-layer count and the intended per-process budget:
   python benchmarks/dsa/analyze_dsa_resident_capacity.py --trace-dir /absolute/path/to/trace-dir --capacities 0,1024,2048,4096,6144,8192,10240,12288 --total-budget 24576 --allocation-scope target

The 24576 example represents 4 target layers × 6144 entries; replace it if the target-layer count or budget differs. Allocation defaults to target layers only. Use --allocation-scope all only if you explicitly want MTP layers included. The budget is measured in raw index entries per worker process, not GiB. The optional allocation optimizes simulated hits on the captured trace only; it is a candidate for a later live experiment, not a runtime policy change.

## Expected outputs

- manifest_rank*_pid*.json
- topk_trace_rank*_pid*.jsonl
- client_request.json and client_response.json
- layer_report.csv
- capacity_curve.csv
- summary.json
- capacity_allocation.csv when --total-budget is supplied

## Validation commands

Run from repository root:

- PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/local/python3.12.13/bin/python -m pytest -q --confcutdir=tests/ut/attention tests/ut/attention/test_dsa_capacity_observer.py tests/ut/attention/test_dsa_capacity_analyzer.py
- /usr/local/python3.12.13/bin/python -m py_compile vllm_ascend/attention/dsa_capacity_observer.py vllm_ascend/attention/sfa_v1.py benchmarks/dsa/request_dsa_capacity_trace.py benchmarks/dsa/analyze_dsa_resident_capacity.py
- bash -n benchmarks/dsa/run_dsa_capacity_trace.sh

## Pass/fail criteria

PASS:
- the uninstrumented baseline still starts and serves a request;
- the observer-disabled branch produces no trace files and no TopK changes;
- the trace-enabled run returns the requested prompt and completion token counts;
- JSONL records are valid and identify rank, PID, target/MTP layer, sequence length, query positions, raw TopK IDs, block size, and cache tensor metadata;
- analyzer produces non-empty layer_report.csv and capacity_curve.csv.

STOP:
- no trace files after a decode request;
- malformed JSONL or unsupported TopK shape;
- model source path does not point to the intended RC2 checkout;
- max selected index is outside the observed sequence domain and the index semantics have not been confirmed;
- any inference error appears only when observation is enabled.

This P0 is an observation and offline-analysis change only. It does not claim that simulated LRU misses equal live offload misses, physical HBM occupancy, or end-to-end latency.
