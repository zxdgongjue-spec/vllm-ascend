# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from types import SimpleNamespace

from vllm_ascend.attention import dsa_capacity_observer as observer


class FakeTensor:
    def __init__(self, values, shape, dtype="int32", element_size=4):
        self.values = values
        self.shape = shape
        self.dtype = dtype
        self._element_size = element_size

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values

    def numel(self):
        def count(value):
            if isinstance(value, (list, tuple)):
                return sum(count(item) for item in value)
            return 1
        return count(self.values)

    def element_size(self):
        return self._element_size


def reset_observer(monkeypatch, trace_dir):
    monkeypatch.setattr(observer, "TRACE_DIR", str(trace_dir))
    monkeypatch.setattr(observer, "DSA_CAPACITY_TRACE_ENABLED", True)
    monkeypatch.setattr(observer, "_TRACE_SEQ", 0)
    monkeypatch.setattr(observer, "_RECORD_COUNT", 0)
    monkeypatch.setattr(observer, "_MANIFEST_WRITTEN", False)
    monkeypatch.setattr(observer, "_WARNED", False)
    monkeypatch.setattr(observer, "_MAX_RECORDS", 100)


def test_normalize_topk_3d_shape():
    tensor = FakeTensor(
        [[[1, 2, -1]], [[2, 3, -1]]],
        (2, 1, 3),
    )
    assert observer._normalize_topk_rows(tensor) == [[1, 2, -1], [2, 3, -1]]


def test_observer_writes_decode_record(tmp_path, monkeypatch):
    reset_observer(monkeypatch, tmp_path)
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=1,
        num_actual_tokens=2,
        seq_lens=FakeTensor([10], (1,)),
        positions=FakeTensor([8, 9], (2,)),
        query_start_loc=FakeTensor([0, 2], (2,)),
        block_size=16,
    )
    topk = FakeTensor(
        [[[1, 2, -1]], [[2, 3, -1]]],
        (2, 1, 3),
    )
    cache = FakeTensor([0] * 8, (2, 4), dtype="float8_e4m3fn", element_size=1)

    observer.observe_sfa_topk(
        layer_name="model.layers.0.self_attn",
        topk_indices=topk,
        metadata=metadata,
        kv_cache=(cache,),
        is_mtp_layer=False,
    )

    trace_files = list(tmp_path.glob("topk_trace_rank*_pid*.jsonl"))
    assert len(trace_files) == 1
    record = json.loads(trace_files[0].read_text(encoding="utf-8").splitlines()[0])
    assert record["kind"] == "topk_trace"
    assert record["topk_indices"] == [[1, 2, -1], [2, 3, -1]]
    assert record["topk_item_count"] == 6
    assert record["valid_topk_items"] == 4
    assert record["negative_padding_items"] == 2
    assert record["negative_padding_fraction"] == 2 / 6
    assert record["valid_topk_rows"] == 2
    assert record["rows_without_valid_topk"] == 0
    assert record["seq_lens"] == [10]
    assert record["positions"] == [8, 9]
    assert record["block_size"] == 16
    assert record["is_mtp_layer"] is False
    assert record["kv_cache_metadata"][0]["storage_bytes"] == 8
    assert len(list(tmp_path.glob("manifest_rank*_pid*.json"))) == 1


def test_observer_skips_prefill(tmp_path, monkeypatch):
    reset_observer(monkeypatch, tmp_path)
    metadata = SimpleNamespace(
        num_prefills=1,
        num_decodes=0,
        num_actual_tokens=2,
        seq_lens=FakeTensor([10], (1,)),
        positions=FakeTensor([8, 9], (2,)),
        query_start_loc=FakeTensor([0, 2], (2,)),
        block_size=16,
    )
    topk = FakeTensor([[1, 2], [2, 3]], (2, 2))
    observer.observe_sfa_topk(
        layer_name="model.layers.0.self_attn",
        topk_indices=topk,
        metadata=metadata,
    )
    assert list(tmp_path.glob("topk_trace_rank*_pid*.jsonl")) == []



def test_rank_label_prefers_initialized_distributed_rank(monkeypatch):
    for name in ("RANK", "RANK_ID", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(observer, "_distributed_rank_label", lambda: "7")
    assert observer._rank_label() == "7"


def test_rank_label_falls_back_to_local_rank(monkeypatch):
    monkeypatch.setattr(observer, "_distributed_rank_label", lambda: None)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("RANK_ID", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert observer._rank_label() == "3"
