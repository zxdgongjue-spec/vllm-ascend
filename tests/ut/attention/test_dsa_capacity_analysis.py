# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import csv
import json

from benchmarks.dsa import analyze_dsa_resident_capacity as analysis


def _query_row(indices):
    return {
        "indices": indices,
        "seq_len": 12000,
        "position": 11999,
        "block_size": 128,
    }


def test_simulate_lru_ignores_negative_padding():
    metrics = analysis.simulate_lru([[10, -1, 20], [-1, -1]], capacity=1)
    assert metrics["accesses"] == 2
    assert metrics["misses"] == 2
    assert metrics["hits"] == 0


def test_group_with_all_negative_topk_is_unusable():
    summary = analysis._summarize_group(
        "0",
        "123",
        "model.layers.0.self_attn",
        [{"is_mtp_layer": False}],
        [_query_row([-1, -1, -1])],
    )
    assert summary["analysis_status"] == "NO_VALID_TOPK"
    assert summary["analysis_usable"] is False
    assert summary["raw_valid_topk_items"] == 0
    assert summary["rows_without_valid_topk"] == 1


def test_group_with_all_negative_query_row_is_marked_partial():
    summary = analysis._summarize_group(
        "0",
        "123",
        "model.layers.0.self_attn",
        [{"is_mtp_layer": False}],
        [_query_row([10, -1, 20]), _query_row([-1, -1, -1])],
    )
    assert summary["analysis_status"] == "PARTIAL_NO_VALID_ROWS"
    assert summary["analysis_usable"] is False


def test_group_with_padding_but_valid_indices_remains_usable():
    summary = analysis._summarize_group(
        "0",
        "123",
        "model.layers.0.self_attn",
        [{"is_mtp_layer": False}],
        [_query_row([10, -1, 20])],
    )
    assert summary["analysis_status"] == "VALID_WITH_PADDING"
    assert summary["analysis_usable"] is True
    assert summary["raw_valid_topk_items"] == 2
    assert summary["negative_padding_items"] == 1


def test_main_blanks_invalid_capacity_curves_and_excludes_allocation(tmp_path, monkeypatch):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    record = {
        "kind": "topk_trace",
        "trace_seq": 0,
        "pid": 123,
        "rank": "0",
        "layer_name": "model.layers.0.self_attn",
        "is_mtp_layer": False,
        "topk_indices": [[-1, -1, -1]],
        "seq_lens": [12000],
        "positions": [11999],
        "query_start_loc": [0, 1],
        "block_size": 128,
    }
    (trace_dir / "topk_trace_rank0_pid123.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_dsa_resident_capacity.py",
            "--trace-dir",
            str(trace_dir),
            "--capacities",
            "0,1",
            "--total-budget",
            "1",
        ],
    )

    assert analysis.main() == 0
    summary = json.loads((trace_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["data_quality_status"] == "FAIL"
    assert summary["unusable_capacity_group_count"] == 1

    with (trace_dir / "capacity_curve.csv").open(encoding="utf-8", newline="") as stream:
        curves = list(csv.DictReader(stream))
    assert len(curves) == 2
    assert all(row["analysis_usable"] == "False" for row in curves)
    assert all(row["accesses"] == "" and row["misses"] == "" for row in curves)

    allocation_path = trace_dir / "capacity_allocation.csv"
    assert allocation_path.exists()
    assert allocation_path.read_text(encoding="utf-8") == ""
