# SPDX-License-Identifier: Apache-2.0
from benchmarks.dsa.analyze_dsa_resident_capacity import record_to_query_rows, simulate_lru


def test_lru_simulator_counts_hits_and_misses():
    result = simulate_lru([[1, 2], [2, 3]], capacity=2)
    assert result["accesses"] == 4
    assert result["hits"] == 1
    assert result["misses"] == 3
    assert result["hit_rate"] == 0.25
    assert result["final_resident_entries"] == 2


def test_zero_capacity_misses_every_unique_access():
    result = simulate_lru([[1, 1, -1, 2], [2, 3]], capacity=0)
    assert result["accesses"] == 4
    assert result["hits"] == 0
    assert result["misses"] == 4


def test_record_rows_keep_sequence_and_query_positions():
    record = {
        "topk_indices": [[1, 2, -1], [2, 3, -1]],
        "seq_lens": [10],
        "positions": [8, 9],
        "query_start_loc": [0, 2],
        "block_size": 16,
    }
    rows = record_to_query_rows(record)
    assert len(rows) == 2
    assert rows[0]["seq_len"] == 10
    assert rows[1]["position"] == 9
    assert rows[0]["block_size"] == 16
