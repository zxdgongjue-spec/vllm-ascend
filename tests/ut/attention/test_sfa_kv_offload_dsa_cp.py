"""Regression tests for the DSA-CP + Sparse KV offload composition."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_ascend.attention.attention_v1 import AscendAttentionState  # noqa: E402
from vllm_ascend.attention.context_parallel.sfa_cp import (  # noqa: E402
    AscendSFADSACPImpl,
    AscendSFADSACPMetadata,
)
from vllm_ascend.attention.sfa_kv_offload import (  # noqa: E402
    AscendSFAKVOffloadImpl,
    AscendSFAOffloadMetadata,
)
from vllm_ascend.attention.sfa_kv_offload_cp import (  # noqa: E402
    AscendSFAKVOffloadDSACPImpl,
    AscendSFAKVOffloadDSACPMetadataBuilder,
    AscendSFAOffloadDSACPMetadata,
)
from vllm_ascend.attention.sfa_v1 import AscendSFABackend  # noqa: E402
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (  # noqa: E402
    SparseKVOffloadManager,
)


def test_dsa_cp_offload_composition_preserves_both_base_implementations():
    mro = AscendSFAKVOffloadDSACPImpl.__mro__
    assert AscendSFAKVOffloadImpl in mro
    assert AscendSFADSACPImpl in mro
    assert mro.index(AscendSFAKVOffloadImpl) < mro.index(AscendSFADSACPImpl)
    metadata_mro = AscendSFAOffloadDSACPMetadata.__mro__
    assert AscendSFAOffloadMetadata in metadata_mro
    assert AscendSFADSACPMetadata in metadata_mro
    assert AscendSFAKVOffloadDSACPImpl.supports_dsa_cp is True


def test_sfa_backend_selects_dsa_cp_offload_pair():
    config = SimpleNamespace(
        enable_dsa_cp=True,
        sparse_kv_offload_config=SimpleNamespace(enabled=True),
    )
    with patch("vllm_ascend.attention.sfa_v1.get_ascend_config", return_value=config):
        assert AscendSFABackend.get_impl_cls() is AscendSFAKVOffloadDSACPImpl
        assert AscendSFABackend.get_builder_cls() is AscendSFAKVOffloadDSACPMetadataBuilder


@pytest.mark.parametrize(
    ("local_start", "local_end", "expected_tokens", "expected_map"),
    [
        (0, 2, 2, [0, 1]),
        (2, 4, 2, [2, 3]),
        (4, 6, 0, []),
    ],
)
def test_dsa_cp_builder_localizes_decode_token_rows(local_start, local_end, expected_tokens, expected_map):
    builder = AscendSFAKVOffloadDSACPMetadataBuilder.__new__(AscendSFAKVOffloadDSACPMetadataBuilder)
    builder.use_fused_copy_sfa = False
    builder.decode_threshold = 1
    builder.is_pd_decode_consumer = False
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.DecodeOnly,
        dsa_cp_context=SimpleNamespace(local_start=local_start, local_end_with_pad=local_end),
    )
    common = SimpleNamespace(
        req_ids_tensor=torch.tensor([10, 11]),
        token_to_req=torch.tensor([0, 1, 0, 1]),
    )
    with patch(
        "vllm_ascend.attention.sfa_kv_offload.split_decodes_and_prefills",
        return_value=(2, 0, 4, 0),
    ):
        builder._populate_offload_metadata(metadata, common)

    assert metadata.num_decodes == 2
    assert metadata.num_prefills == 0
    assert metadata.num_decode_tokens == expected_tokens
    assert metadata.token_to_req.tolist() == expected_map


def test_dsa_cp_rank_without_local_decode_rows_still_is_decode_only():
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.DecodeOnly,
        num_decodes=2,
        num_prefills=0,
        num_decode_tokens=0,
    )
    assert AscendSFAKVOffloadDSACPImpl._is_decode_only(metadata)


def test_dsa_cp_decode_stores_gathered_global_kv_to_host_pool():
    impl = AscendSFAKVOffloadDSACPImpl.__new__(AscendSFAKVOffloadDSACPImpl)
    impl.qk_rope_head_dim = 2
    impl.kv_lora_rank = 3
    impl.layer_name = "model.layers.0.self_attn.attn"
    impl._current_layer_name = None
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.DecodeOnly,
        num_decodes=2,
        num_prefills=0,
        num_decode_tokens=0,
        num_actual_tokens=2,
    )
    gathered = torch.arange(10, dtype=torch.float32).view(2, 5)
    slots = torch.tensor([17, 18], dtype=torch.int64)
    manager = MagicMock()

    with (
        patch("vllm_ascend.attention.sfa_kv_offload_cp.get_sparse_kv_offload_manager", return_value=manager),
        patch.object(AscendSFAKVOffloadImpl, "_cpu_cache_pair", return_value=("cpu-k", "cpu-v")),
        patch.object(AscendSFAKVOffloadImpl, "_in_graph_runtime", return_value=False),
    ):
        impl._store_parallel_kv(
            k_pe=None,
            k_nope=None,
            knope_scale=None,
            fused_kv_no_split=gathered,
            kv_ag_handles=[],
            kv_cache=None,
            slot_mapping_sfa=slots,
            attn_metadata=metadata,
            full_gather_o_proj_enabled=False,
        )

    kwargs = manager.offload_new_kv.call_args.kwargs
    assert kwargs["has_prefill"] is False
    assert torch.equal(kwargs["slot_mapping"], slots)
    assert tuple(kwargs["k"].shape) == (2, 1, 3)
    assert tuple(kwargs["v"].shape) == (2, 1, 2)


def test_zero_local_decode_rank_still_joins_tp_barrier():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.max_num_topk_rows = 8
    manager.tp_size = 2
    manager.tp_group = MagicMock()
    manager.layer_name_to_offload_id = {"layer": 0}
    real_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("device", None)
        return real_empty(*args, **kwargs)

    with patch(
        "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager.torch.empty",
        side_effect=cpu_empty,
    ):
        manager.onload_topk_kv(
            layer_name="layer",
            num_tokens=0,
            num_reqs=1,
            block_table=None,
            topk_indices_npu=None,
            current_slots_npu=None,
            req_ids_npu=None,
            stable_prefix_lens_npu=None,
            token_to_req_npu=None,
        )

    manager.tp_group.broadcast.assert_called_once()


def test_dsa_cp_prefill_rejects_missing_device_main_cache_before_native_write():
    impl = AscendSFAKVOffloadDSACPImpl.__new__(AscendSFAKVOffloadDSACPImpl)
    metadata = SimpleNamespace(attn_state=object(), num_prefills=1, num_decodes=0)
    with pytest.raises(RuntimeError, match="keep_device_kv_cache=True"):
        impl.exec_kv(
            kv_no_split=torch.empty((1, 1)),
            cos=torch.empty((1, 1)),
            sin=torch.empty((1, 1)),
            kv_cache=(None, None),
            slots=torch.empty((1,), dtype=torch.int64),
            attn_metadata=metadata,
        )
