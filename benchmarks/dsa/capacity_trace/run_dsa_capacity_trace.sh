#!/usr/bin/env bash
set -euo pipefail

TRACE_DIR="$(printenv TRACE_DIR 2>/dev/null || true)"
if [ -z "$TRACE_DIR" ]; then
    TRACE_DIR="$PWD/results-dsa-capacity-4l-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$TRACE_DIR"
export VLLM_ASCEND_DSA_CAPACITY_TRACE_DIR="$TRACE_DIR"

MAX_RECORDS="$(printenv VLLM_ASCEND_DSA_CAPACITY_TRACE_MAX_RECORDS 2>/dev/null || true)"
if [ -z "$MAX_RECORDS" ]; then
    export VLLM_ASCEND_DSA_CAPACITY_TRACE_MAX_RECORDS=10000
fi

if [ "$#" -lt 2 ] || [ "$1" != "--" ]; then
    echo "Usage: TRACE_DIR=/absolute/trace/dir $0 -- vllm serve MODEL [existing server arguments...]" >&2
    echo "For trace collection, include the global --enforce-eager option." >&2
    exit 2
fi
shift

echo "[DSA_CAPACITY_TRACE] enabled=1"
echo "[DSA_CAPACITY_TRACE] dir=$VLLM_ASCEND_DSA_CAPACITY_TRACE_DIR"
echo "[DSA_CAPACITY_TRACE] max_records=$VLLM_ASCEND_DSA_CAPACITY_TRACE_MAX_RECORDS"
exec "$@"
