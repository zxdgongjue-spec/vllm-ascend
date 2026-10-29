#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only audit for rank/layer TopK trace validity and DSA-CP metadata."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


TRACE_PATTERN = "topk_trace_rank*_pid*.jsonl"
DEFAULT_SAMPLE_LAYERS = (
    "model.layers.3.self_attn.attn,"
    "model.layers.4.self_attn.attn,"
    "model.layers.5.self_attn.attn,"
    "model.layers.6.self_attn.attn,"
    "model.layers.10.self_attn.attn,"
    "model.layers.11.self_attn.attn,"
    "model.layers.78.self_attn.attn"
)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _record_rows(record: dict[str, Any]) -> list[list[int]]:
    value = record.get("topk_indices")
    if not isinstance(value, list):
        return []
    rows: list[list[int]] = []
    for row in value:
        if not isinstance(row, list):
            continue
        rows.append([_as_int(item, -1) for item in row])
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rank_sort(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):08d}")
    except ValueError:
        return (1, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, help="Existing trace directory; input is never modified")
    parser.add_argument("--output-dir", default=None, help="Defaults to TRACE_DIR/analysis-trace-audit")
    parser.add_argument(
        "--sample-layers",
        default=DEFAULT_SAMPLE_LAYERS,
        help="Comma-separated layer names to print for rank-by-rank comparison",
    )
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else trace_dir / "analysis-trace-audit"
    )
    if not trace_dir.is_dir():
        raise SystemExit(f"STOP: trace directory does not exist: {trace_dir}")

    paths = sorted(glob.glob(str(trace_dir / TRACE_PATTERN)))
    if not paths:
        raise SystemExit(f"STOP: no {TRACE_PATTERN} files found under {trace_dir}")

    records: list[dict[str, Any]] = []
    malformed_lines = 0
    per_file: list[dict[str, Any]] = []
    for path in paths:
        count = 0
        with open(path, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue
                if not isinstance(value, dict) or value.get("kind") != "topk_trace":
                    continue
                value["_audit_file"] = Path(path).name
                value["_audit_line"] = line_number
                records.append(value)
                count += 1
        per_file.append({"file": Path(path).name, "record_count": count})

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    rank_groups: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    layer_names: set[str] = set()
    total_items = 0
    total_valid = 0
    total_rows = 0
    total_valid_rows = 0
    for record in records:
        rank = str(record.get("rank", "unknown"))
        pid = str(record.get("pid", "unknown"))
        layer = str(record.get("layer_name", "unknown"))
        key = (rank, pid, layer)
        if not groups[key]:
            rank_groups[(rank, pid)].append(key)
        groups[key].append(record)
        layer_names.add(layer)
        rows = _record_rows(record)
        total_rows += len(rows)
        total_items += sum(len(row) for row in rows)
        total_valid += sum(value >= 0 for row in rows for value in row)
        total_valid_rows += sum(any(value >= 0 for value in row) for row in rows)

    layer_rows: list[dict[str, Any]] = []
    for (rank, pid, layer), group_records in sorted(
        groups.items(), key=lambda item: (_rank_sort(item[0][0]), item[0][1], item[0][2])
    ):
        all_rows: list[list[int]] = []
        for record in group_records:
            all_rows.extend(_record_rows(record))
        item_count = sum(len(row) for row in all_rows)
        valid_items = sum(value >= 0 for row in all_rows for value in row)
        valid_rows = sum(any(value >= 0 for value in row) for row in all_rows)
        local_expected = sorted({
            _as_int(record["local_query_rows_expected"])
            for record in group_records
            if record.get("local_query_rows_expected") is not None
        })
        query_metadata = sorted({
            _json_cell(record.get("indexer_actual_seq_lengths_query"))
            for record in group_records
        })
        key_metadata = sorted({
            _json_cell(record.get("indexer_actual_seq_lengths_key"))
            for record in group_records
        })
        dropped_padding_rows = sum(
            _as_int(record.get("query_rows_dropped_as_parallel_padding"))
            for record in group_records
        )
        first_record = group_records[0]
        first_rows = _record_rows(first_record)
        positions = first_record.get("positions")
        layer_rows.append({
            "rank": rank,
            "pid": pid,
            "layer_name": layer,
            "is_mtp_layer": any(bool(record.get("is_mtp_layer")) for record in group_records),
            "trace_records": len(group_records),
            "query_rows": len(all_rows),
            "topk_widths": _json_cell(sorted({len(row) for row in all_rows})),
            "topk_item_count": item_count,
            "valid_topk_items": valid_items,
            "negative_index_items": item_count - valid_items,
            "negative_index_fraction": (item_count - valid_items) / item_count if item_count else 0.0,
            "valid_query_rows": valid_rows,
            "all_negative_query_rows": len(all_rows) - valid_rows,
            "all_rows_invalid": valid_items == 0,
            "rows_with_no_valid_topk": len(all_rows) - valid_rows,
            "local_query_rows_expected_values": _json_cell(local_expected),
            "indexer_query_lengths_values": _json_cell(query_metadata),
            "indexer_key_lengths_values": _json_cell(key_metadata),
            "query_rows_dropped_as_parallel_padding": dropped_padding_rows,
            "num_actual_tokens_values": _json_cell(sorted({
                _as_int(record.get("num_actual_tokens"), -1) for record in group_records
            })),
            "num_query_rows_values": _json_cell(sorted({
                _as_int(record.get("num_query_rows"), -1) for record in group_records
            })),
            "first_record_seq_lens": _json_cell(first_record.get("seq_lens")),
            "first_record_positions": _json_cell(positions),
            "first_record_query_start_loc": _json_cell(first_record.get("query_start_loc")),
            "first_record_global_query_start_loc": _json_cell(first_record.get("global_query_start_loc")),
            "first_row_first_16_topk": _json_cell(first_rows[0][:16] if first_rows else []),
            "trace_file": str(first_record.get("_audit_file", "")),
        })

    rank_rows: list[dict[str, Any]] = []
    for (rank, pid), keys in sorted(rank_groups.items(), key=lambda item: (_rank_sort(item[0][0]), item[0][1])):
        rank_layer_rows = [row for row in layer_rows if row["rank"] == rank and row["pid"] == pid]
        rank_records = [record for record in records if str(record.get("rank", "unknown")) == rank and str(record.get("pid", "unknown")) == pid]
        rank_rows.append({
            "rank": rank,
            "pid": pid,
            "trace_records": len(rank_records),
            "layer_groups": len(keys),
            "layers_with_any_valid_topk": sum(not row["all_rows_invalid"] for row in rank_layer_rows),
            "layers_all_rows_invalid": sum(row["all_rows_invalid"] for row in rank_layer_rows),
            "valid_topk_items": sum(row["valid_topk_items"] for row in rank_layer_rows),
            "negative_index_items": sum(row["negative_index_items"] for row in rank_layer_rows),
            "negative_index_fraction": (
                sum(row["negative_index_items"] for row in rank_layer_rows)
                / max(1, sum(row["topk_item_count"] for row in rank_layer_rows))
            ),
            "missing_layers_vs_trace_union": _json_cell(sorted(layer_names - {key[2] for key in keys})),
        })

    sample_layers = {value.strip() for value in args.sample_layers.split(",") if value.strip()}
    samples = [
        row for row in layer_rows
        if row["layer_name"] in sample_layers
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "rank_summary.csv", rank_rows)
    _write_csv(output_dir / "rank_layer_summary.csv", layer_rows)
    (output_dir / "summary.json").write_text(
        json.dumps({
            "schema_version": 1,
            "audit_status": "PASS",
            "trace_dir": str(trace_dir),
            "output_dir": str(output_dir),
            "trace_file_count": len(paths),
            "record_count": len(records),
            "malformed_jsonl_line_count": malformed_lines,
            "rank_pid_count": len(rank_groups),
            "rank_layer_group_count": len(groups),
            "union_layer_count": len(layer_names),
            "query_row_count": total_rows,
            "topk_item_count": total_items,
            "valid_topk_item_count": total_valid,
            "negative_index_item_count": total_items - total_valid,
            "negative_index_fraction": (total_items - total_valid) / total_items if total_items else 0.0,
            "valid_query_rows": total_valid_rows,
            "all_invalid_rank_layer_groups": sum(row["all_rows_invalid"] for row in layer_rows),
            "trace_files": per_file,
            "rank_summary": rank_rows,
            "sample_layers": sorted(sample_layers),
            "sample_rank_layer_groups": samples,
            "interpretation": (
                "This audit describes captured values only. It does not decide whether negative "
                "indices are legal padding/sentinels for the current DSA-CP path."
            ),
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": "PASS",
        "data_quality_status": "PARTIAL" if total_valid == 0 or any(row["all_rows_invalid"] for row in layer_rows) or total_valid < total_items else "PASS",
        "trace_files": len(paths),
        "records": len(records),
        "rank_pid_groups": len(rank_groups),
        "rank_layer_groups": len(groups),
        "layers_in_trace_union": len(layer_names),
        "valid_topk_items": total_valid,
        "negative_index_items": total_items - total_valid,
        "negative_index_fraction": (total_items - total_valid) / total_items if total_items else 0.0,
        "outputs": [
            str(output_dir / "summary.json"),
            str(output_dir / "rank_summary.csv"),
            str(output_dir / "rank_layer_summary.csv"),
        ],
        "rank_summary": rank_rows,
        "sample_rank_layer_groups": samples,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
