# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in, read-only SFA TopK trace for offline DSA resident-capacity analysis.

Set VLLM_ASCEND_DSA_CAPACITY_TRACE_DIR before starting vLLM. The observer
records the raw TopK indices consumed by SFA on pure decode batches only.
It never changes TopK, cache state, eviction, KV movement, or attention inputs.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

TRACE_ENV = "VLLM_ASCEND_DSA_CAPACITY_TRACE_DIR"
TRACE_MAX_RECORDS_ENV = "VLLM_ASCEND_DSA_CAPACITY_TRACE_MAX_RECORDS"
TRACE_DIR = os.environ.get(TRACE_ENV, "").strip()
DSA_CAPACITY_TRACE_ENABLED = bool(TRACE_DIR)

try:
    _MAX_RECORDS = max(1, int(os.environ.get(TRACE_MAX_RECORDS_ENV, "10000")))
except (TypeError, ValueError):
    _MAX_RECORDS = 10000

_LOCK = threading.Lock()
_TRACE_SEQ = 0
_RECORD_COUNT = 0
_MANIFEST_WRITTEN = False
_WARNED = False


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, RuntimeError):
        return default


def _to_list(value: Any) -> Any:
    """Copy a tensor to host only in explicitly enabled trace mode."""
    if value is None:
        return None
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _normalize_topk_rows(topk_indices: Any) -> list[list[int]]:
    """Normalize [T,K] or [T,1,K] indices to a list of query rows."""
    shape_value = getattr(topk_indices, "shape", ())
    try:
        shape = tuple(int(dim) for dim in shape_value)
    except (TypeError, ValueError):
        shape = ()

    values = _to_list(topk_indices)
    if not isinstance(values, list):
        raise ValueError("TopK indices did not convert to a list")

    if len(shape) == 3:
        if shape[1] != 1:
            raise ValueError(f"Unsupported TopK shape {shape}; expected [T,1,K]")
        values = [row[0] for row in values]
    elif len(shape) > 3:
        raise ValueError(f"Unsupported TopK shape {shape}; expected [T,K] or [T,1,K]")

    if values and not isinstance(values[0], (list, tuple)):
        values = [values]

    rows: list[list[int]] = []
    for row in values:
        if not isinstance(row, (list, tuple)):
            raise ValueError("Malformed TopK row")
        rows.append([_safe_int(item, -1) for item in row])
    return rows


def _tensor_metadata(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    result: dict[str, Any] = {}
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            result["shape"] = [int(dim) for dim in shape]
        except (TypeError, ValueError):
            result["shape"] = str(shape)
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        result["dtype"] = str(dtype)
    numel_fn = getattr(value, "numel", None)
    element_size_fn = getattr(value, "element_size", None)
    try:
        numel = int(numel_fn()) if callable(numel_fn) else None
        element_size = int(element_size_fn()) if callable(element_size_fn) else None
        if numel is not None:
            result["numel"] = numel
        if element_size is not None:
            result["element_size_bytes"] = element_size
        if numel is not None and element_size is not None:
            result["storage_bytes"] = numel * element_size
    except (TypeError, ValueError, RuntimeError):
        pass
    return result


def _cache_metadata(kv_cache: Any) -> list[dict[str, Any] | None]:
    if kv_cache is None:
        return []
    tensors = list(kv_cache) if isinstance(kv_cache, (list, tuple)) else [kv_cache]
    return [_tensor_metadata(tensor) for tensor in tensors]


def _rank_label() -> str:
    for name in ("RANK", "RANK_ID", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip().replace("/", "_")
    return "unknown"


def _is_graph_capturing(forward_context: Any) -> bool:
    if bool(getattr(forward_context, "in_profile_run", False)):
        return True
    try:
        import torch

        checker = getattr(getattr(torch, "npu", None), "is_current_stream_capturing", None)
        if callable(checker):
            return bool(checker())
    except Exception:
        # The launch instructions require eager mode for trace collection.
        # A missing/unsupported capture probe must not break inference.
        return False
    return False


def _warn_once(message: str) -> None:
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    try:
        sys.stderr.write("[DSA_CAPACITY_OBSERVER] " + message + "\n")
    except Exception:
        pass


def _trace_paths() -> tuple[Path, Path]:
    root = Path(TRACE_DIR)
    rank = _rank_label()
    pid = os.getpid()
    return (
        root / f"topk_trace_rank{rank}_pid{pid}.jsonl",
        root / f"manifest_rank{rank}_pid{pid}.json",
    )


def _write_manifest(trace_path: Path, manifest_path: Path) -> None:
    global _MANIFEST_WRITTEN
    if _MANIFEST_WRITTEN or manifest_path.exists():
        _MANIFEST_WRITTEN = True
        return
    manifest = {
        "schema_version": 1,
        "kind": "dsa_capacity_trace_manifest",
        "pid": os.getpid(),
        "rank": _rank_label(),
        "trace_file": trace_path.name,
        "trace_env": TRACE_ENV,
        "max_records": _MAX_RECORDS,
        "created_unix_ns": time.time_ns(),
        "capture_scope": "pure_decode_only",
        "index_semantics": "raw_sfa_topk_indices_before_attention",
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    _MANIFEST_WRITTEN = True


def observe_sfa_topk(
    *,
    layer_name: str,
    topk_indices: Any,
    metadata: Any,
    kv_cache: Any = None,
    is_mtp_layer: bool = False,
    forward_context: Any = None,
) -> None:
    """Append one decode-only TopK record; all failures are non-fatal."""
    global _TRACE_SEQ, _RECORD_COUNT
    if not DSA_CAPACITY_TRACE_ENABLED:
        return

    try:
        if _is_graph_capturing(forward_context):
            return

        num_prefills = _safe_int(getattr(metadata, "num_prefills", 0))
        num_decodes = _safe_int(getattr(metadata, "num_decodes", 0))
        if num_prefills != 0 or num_decodes <= 0:
            return

        rows = _normalize_topk_rows(topk_indices)
        num_actual_tokens = _safe_int(
            getattr(metadata, "num_actual_tokens", len(rows)), len(rows)
        )
        if num_actual_tokens <= 0:
            return
        rows = rows[:num_actual_tokens]
        if not rows:
            return

        seq_lens = _to_list(getattr(metadata, "seq_lens", None)) or []
        positions = _to_list(getattr(metadata, "positions", None))
        query_start_loc = _to_list(getattr(metadata, "query_start_loc", None))
        if positions is not None:
            positions = positions[:num_actual_tokens]
        if query_start_loc is not None:
            query_start_loc = [int(value) for value in query_start_loc]
        seq_lens = [int(value) for value in seq_lens]

        block_size = _safe_int(getattr(metadata, "block_size", 0))
        record = {
            "schema_version": 1,
            "kind": "topk_trace",
            "trace_seq": None,
            "timestamp_unix_ns": time.time_ns(),
            "timestamp_monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "rank": _rank_label(),
            "layer_name": str(layer_name),
            "is_mtp_layer": bool(is_mtp_layer),
            "phase": "decode",
            "num_prefills": num_prefills,
            "num_decodes": num_decodes,
            "num_actual_tokens": num_actual_tokens,
            "num_query_rows": len(rows),
            "topk_width": max((len(row) for row in rows), default=0),
            "topk_indices": rows,
            "seq_lens": seq_lens,
            "positions": positions,
            "query_start_loc": query_start_loc,
            "block_size": block_size or None,
            "seq_len_tail_tokens": (
                [value % block_size for value in seq_lens] if block_size > 0 else None
            ),
            "seq_len_block_count": (
                [(value + block_size - 1) // block_size for value in seq_lens]
                if block_size > 0
                else None
            ),
            "kv_cache_metadata": _cache_metadata(kv_cache),
            "index_semantics": "raw_sfa_topk_indices_before_attention",
        }

        with _LOCK:
            if _RECORD_COUNT >= _MAX_RECORDS:
                return
            trace_path, manifest_path = _trace_paths()
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            _write_manifest(trace_path, manifest_path)
            record["trace_seq"] = _TRACE_SEQ
            with trace_path.open("a", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, separators=(",", ":"))
                stream.write("\n")
            _TRACE_SEQ += 1
            _RECORD_COUNT += 1
    except Exception as exc:
        # Observation must never change inference success/failure semantics.
        _warn_once(f"skipped a record after {type(exc).__name__}: {exc}")
