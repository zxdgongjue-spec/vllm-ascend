#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze raw SFA TopK traces and simulate per-layer LRU resident capacities.

The simulator is offline and approximate. It treats each query row's TopK list
as an ordered access sequence, matching the captured index order. Its hit/miss
counts are not live runtime offload counters and must not be reported as measured
TPOT or physical HBM bytes.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Any


DEFAULT_CAPACITIES = (0, 1024, 2048, 4096, 6144, 8192, 10240, 12288)


def parse_capacities(value: str) -> list[int]:
    try:
        capacities = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("capacities must be comma-separated integers") from exc
    if not capacities or capacities[0] < 0:
        raise argparse.ArgumentTypeError("capacities must contain non-negative integers")
    return capacities


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _request_index(row_index: int, query_start_loc: list[int], request_count: int) -> int:
    if len(query_start_loc) >= 2:
        for request_index in range(len(query_start_loc) - 1):
            if query_start_loc[request_index] <= row_index < query_start_loc[request_index + 1]:
                return request_index
    if request_count <= 0:
        return -1
    return min(row_index, request_count - 1)


def record_to_query_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    topk_rows = record.get("topk_indices")
    if not isinstance(topk_rows, list):
        return []
    seq_lens = _int_list(record.get("seq_lens"))
    positions = _int_list(record.get("positions"))
    query_start_loc = _int_list(record.get("query_start_loc"))
    block_size = int(record.get("block_size") or 0)
    result = []
    for row_index, row in enumerate(topk_rows):
        if not isinstance(row, list):
            continue
        indices = []
        for value in row:
            try:
                indices.append(int(value))
            except (TypeError, ValueError):
                indices.append(-1)
        request_index = _request_index(row_index, query_start_loc, len(seq_lens))
        seq_len = seq_lens[request_index] if 0 <= request_index < len(seq_lens) else None
        position = positions[row_index] if row_index < len(positions) else None
        result.append({
            "indices": indices,
            "seq_len": seq_len,
            "position": position,
            "block_size": block_size,
        })
    return result


def simulate_lru(query_rows: list[list[int]], capacity: int) -> dict[str, int | float]:
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    resident: OrderedDict[int, None] = OrderedDict()
    hits = 0
    misses = 0
    accesses = 0
    duplicate_within_query = 0

    for row in query_rows:
        seen: set[int] = set()
        for token_id in row:
            if token_id < 0:
                continue
            if token_id in seen:
                duplicate_within_query += 1
                continue
            seen.add(token_id)
            accesses += 1
            if capacity == 0:
                misses += 1
                continue
            if token_id in resident:
                hits += 1
                resident.move_to_end(token_id)
            else:
                misses += 1
                resident[token_id] = None
                if len(resident) > capacity:
                    resident.popitem(last=False)

    return {
        "accesses": accesses,
        "hits": hits,
        "misses": misses,
        "hit_rate": (hits / accesses) if accesses else 0.0,
        "miss_rate": (misses / accesses) if accesses else 0.0,
        "misses_per_query": (misses / len(query_rows)) if query_rows else 0.0,
        "final_resident_entries": len(resident),
        "duplicate_within_query": duplicate_within_query,
    }


def read_trace(trace_dir: Path) -> tuple[list[dict[str, Any]], list[str], int]:
    paths = sorted(glob.glob(str(trace_dir / "topk_trace_rank*_pid*.jsonl")))
    if not paths:
        raise SystemExit(
            "No topk_trace_rank*_pid*.jsonl files found. Set "
            "VLLM_ASCEND_DSA_CAPACITY_TRACE_DIR before starting the server, "
            "then issue a decode request."
        )
    records: list[dict[str, Any]] = []
    malformed = 0
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if isinstance(value, dict) and value.get("kind") == "topk_trace":
                    value["_trace_file"] = Path(path).name
                    value["_line_number"] = line_number
                    records.append(value)
    records.sort(key=lambda item: (
        str(item.get("rank", "unknown")),
        int(item.get("pid", -1)),
        str(item.get("layer_name", "")),
        int(item.get("trace_seq") or 0),
    ))
    return records, paths, malformed


def _group_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("rank", "unknown")),
        str(record.get("pid", "unknown")),
        str(record.get("layer_name", "unknown")),
    )


def _summarize_group(
    rank: str,
    pid: str,
    layer_name: str,
    records: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    seq_lens = [int(row["seq_len"]) for row in query_rows if row["seq_len"] is not None]
    indices = [
        int(token_id)
        for row in query_rows
        for token_id in row["indices"]
        if int(token_id) >= 0
    ]
    unique_per_query = [
        len({int(token_id) for token_id in row["indices"] if int(token_id) >= 0})
        for row in query_rows
    ]
    block_sizes = sorted({
        int(row["block_size"]) for row in query_rows if int(row["block_size"]) > 0
    })
    max_index = max(indices) if indices else None
    max_seq_len = max(seq_lens) if seq_lens else None
    return {
        "rank": rank,
        "pid": pid,
        "layer_name": layer_name,
        "is_mtp_layer": any(bool(record.get("is_mtp_layer")) for record in records),
        "trace_records": len(records),
        "query_rows": len(query_rows),
        "topk_width_max": max((len(row["indices"]) for row in query_rows), default=0),
        "raw_valid_topk_items": len(indices),
        "unique_accesses": sum(unique_per_query),
        "duplicate_topk_items": len(indices) - sum(unique_per_query),
        "negative_padding_items": sum(
            1 for row in query_rows for token_id in row["indices"] if int(token_id) < 0
        ),
        "mean_unique_selected_per_query": (
            statistics.mean(unique_per_query) if unique_per_query else 0.0
        ),
        "mean_seq_len": statistics.mean(seq_lens) if seq_lens else None,
        "max_seq_len": max_seq_len,
        "max_selected_index": max_index,
        "max_index_ge_max_seq_len": (
            bool(max_index is not None and max_seq_len is not None and max_index >= max_seq_len)
        ),
        "block_size": block_sizes[0] if len(block_sizes) == 1 else (
            ",".join(str(value) for value in block_sizes) if block_sizes else None
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def allocate_capacity(
    group_rows: dict[tuple[str, str, str], list[dict[str, Any]]],
    capacities: list[int],
    capacity_curves: dict[tuple[str, str, str, int], dict[str, Any]],
    total_budget: int,
) -> list[dict[str, Any]]:
    """Find the best discrete per-layer allocation under a per-process token budget."""
    process_keys = sorted({(key[0], key[1]) for key in group_rows})
    output: list[dict[str, Any]] = []
    for rank, pid in process_keys:
        layers = sorted(
            key for key in group_rows if key[0] == rank and key[1] == pid
        )
        dp: dict[int, tuple[int, list[tuple[tuple[str, str, str], int]]]] = {0: (0, [])}
        for layer_key in layers:
            next_dp: dict[int, tuple[int, list[tuple[tuple[str, str, str], int]]]] = {}
            for used, (hits_so_far, choices) in dp.items():
                for capacity in capacities:
                    new_used = used + capacity
                    if new_used > total_budget:
                        continue
                    curve = capacity_curves[(layer_key[0], layer_key[1], layer_key[2], capacity)]
                    new_hits = hits_so_far + int(curve["hits"])
                    candidate = (new_hits, choices + [(layer_key, capacity)])
                    current = next_dp.get(new_used)
                    if current is None or new_hits > current[0]:
                        next_dp[new_used] = candidate
            dp = next_dp
        if not dp:
            continue
        used, (hits, choices) = max(dp.items(), key=lambda item: (item[1][0], item[0]))
        for layer_key, capacity in choices:
            curve = capacity_curves[(layer_key[0], layer_key[1], layer_key[2], capacity)]
            output.append({
                "rank": layer_key[0],
                "pid": layer_key[1],
                "layer_name": layer_key[2],
                "total_budget_tokens_per_process": total_budget,
                "allocated_budget_tokens_per_process": used,
                "capacity_tokens": capacity,
                "predicted_accesses": curve["accesses"],
                "predicted_hits": curve["hits"],
                "predicted_misses": curve["misses"],
                "predicted_hit_rate": curve["hit_rate"],
                "allocation_objective_total_hits": hits,
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--capacities",
        type=parse_capacities,
        default=list(DEFAULT_CAPACITIES),
        help="Comma-separated token capacities per layer; default includes 0, 1024, 2048, 4096, 6144, 8192, 10240, 12288",
    )
    parser.add_argument(
        "--total-budget",
        type=int,
        default=None,
        help="Optional total resident-token budget per worker process, allocated across selected layers",
    )
    parser.add_argument(
        "--allocation-scope",
        choices=("target", "all"),
        default="target",
        help="Layers included in optional allocation: target only by default, or target plus MTP",
    )
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else trace_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.total_budget is not None and args.total_budget < 0:
        parser.error("--total-budget must be non-negative")

    records, trace_files, malformed = read_trace(trace_dir)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(_group_key(record), []).append(record)

    layer_rows: list[dict[str, Any]] = []
    query_rows_by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    curve_rows: list[dict[str, Any]] = []
    capacity_curves: dict[tuple[str, str, str, int], dict[str, Any]] = {}

    for key, group_records in sorted(groups.items()):
        group_records.sort(key=lambda item: int(item.get("trace_seq") or 0))
        query_rows = [
            row
            for record in group_records
            for row in record_to_query_rows(record)
        ]
        query_rows_by_group[key] = query_rows
        layer_rows.append(_summarize_group(*key, group_records, query_rows))
        raw_rows = [row["indices"] for row in query_rows]
        for capacity in args.capacities:
            metrics = simulate_lru(raw_rows, capacity)
            curve = {
                "rank": key[0],
                "pid": key[1],
                "layer_name": key[2],
                "is_mtp_layer": any(bool(record.get("is_mtp_layer")) for record in group_records),
                "capacity_tokens": capacity,
                "query_rows": len(raw_rows),
                **metrics,
            }
            curve_rows.append(curve)
            capacity_curves[(key[0], key[1], key[2], capacity)] = curve

    _write_csv(output_dir / "layer_report.csv", layer_rows)
    _write_csv(output_dir / "capacity_curve.csv", curve_rows)

    allocation_rows: list[dict[str, Any]] = []
    if args.total_budget is not None:
        if args.allocation_scope == "target":
            allocation_groups = {
                key: value
                for key, value in query_rows_by_group.items()
                if not any(bool(record.get("is_mtp_layer")) for record in groups[key])
            }
        else:
            allocation_groups = query_rows_by_group
        allocation_keys = set(allocation_groups)
        allocation_curves = {
            key: value for key, value in capacity_curves.items() if key[:3] in allocation_keys
        }
        allocation_rows = allocate_capacity(
            allocation_groups, args.capacities, allocation_curves, args.total_budget
        )
        _write_csv(output_dir / "capacity_allocation.csv", allocation_rows)

    summary = {
        "schema_version": 1,
        "trace_dir": str(trace_dir),
        "trace_files": [Path(path).name for path in trace_files],
        "trace_file_count": len(trace_files),
        "topk_trace_record_count": len(records),
        "malformed_jsonl_line_count": malformed,
        "rank_pid_layer_group_count": len(groups),
        "layer_names": sorted({key[2] for key in groups}),
        "target_layer_names": sorted({
            key[2] for key, values in groups.items()
            if not any(bool(record.get("is_mtp_layer")) for record in values)
        }),
        "mtp_layer_names": sorted({
            key[2] for key, values in groups.items()
            if any(bool(record.get("is_mtp_layer")) for record in values)
        }),
        "capacities_tokens_per_layer": args.capacities,
        "total_budget_tokens_per_process": args.total_budget,
        "allocation_scope": args.allocation_scope if args.total_budget is not None else None,
        "allocation_row_count": len(allocation_rows),
        "simulator": {
            "name": "offline_ordered_lru",
            "capacity_unit": "raw TopK index entries",
            "within_query_order": "captured TopK output order",
            "prefill_included": False,
            "runtime_hit_miss_measured": False,
            "physical_hbm_bytes_measured": False,
            "tpot_measured": False,
        },
        "warnings": [
            "This is a trace-driven LRU estimate, not a measurement of live resident-cache hits or offload misses.",
            "Validate that raw SFA TopK indices represent the token identity domain intended for capacity simulation.",
            "If target and MTP layers differ, use layer_report.csv to separate them before making a capacity decision.",
            "Observation mode copies indices to CPU and writes JSONL; do not use its latency as the performance baseline.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "trace_files": len(trace_files),
        "records": len(records),
        "rank_pid_layer_groups": len(groups),
        "target_layers": len(summary["target_layer_names"]),
        "mtp_layers": len(summary["mtp_layer_names"]),
        "outputs": [
            str(output_dir / "layer_report.csv"),
            str(output_dir / "capacity_curve.csv"),
            str(output_dir / "summary.json"),
        ] + ([str(output_dir / "capacity_allocation.csv")] if allocation_rows else []),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
