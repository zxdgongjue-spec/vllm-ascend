"""DSA-CP-aware composition of SFA and Sparse KV offload.

The DSA-CP implementation owns token sharding and the TP KV gather. Sparse
KV offload owns the shared Host KV pool and per-rank resident selection.
This module composes those paths without changing the DSA-CP attention math.
"""

from dataclasses import dataclass

import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.sfa_cp import (
    AscendSFADSACPImpl,
    AscendSFADSACPMetadata,
    AscendSFADSACPMetadataBuilder,
)
from vllm_ascend.attention.sfa_kv_offload import (
    AscendSFAKVOffloadImpl,
    AscendSFAKVOffloadMetadataBuilder,
    AscendSFAOffloadMetadata,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    get_sparse_kv_offload_manager,
)


@dataclass
class AscendSFAOffloadDSACPMetadata(AscendSFAOffloadMetadata, AscendSFADSACPMetadata):
    """Metadata carrying both offload state and DSA-CP token-shard state."""


class AscendSFAKVOffloadDSACPMetadataBuilder(
    AscendSFAKVOffloadMetadataBuilder,
    AscendSFADSACPMetadataBuilder,
):
    """Compose the offload metadata extension with DSA-CP metadata building."""

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
        metadata_cls: type[AscendSFAOffloadMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls or AscendSFAOffloadDSACPMetadata,
            supports_dcp_with_varlen,
        )

    def _populate_offload_metadata(
        self,
        metadata: AscendSFAOffloadDSACPMetadata,
        common_attn_metadata,
        draft_index: int | None = None,
    ) -> AscendSFAOffloadDSACPMetadata:
        metadata = super()._populate_offload_metadata(metadata, common_attn_metadata, draft_index)
        context = metadata.dsa_cp_context
        if context is None:
            raise RuntimeError("DSA-CP Sparse KV offload metadata is missing dsa_cp_context.")

        # split_decodes_and_prefills returns global token counts. The offload
        # selection path runs on each rank's local token shard, so retain only
        # the intersection of that shard with the leading global decode range.
        global_decode_tokens = int(metadata.num_decode_tokens)
        local_start = min(max(int(context.local_start), 0), global_decode_tokens)
        local_end = min(max(int(context.local_end_with_pad), local_start), global_decode_tokens)
        metadata.num_decode_tokens = local_end - local_start
        if metadata.token_to_req is not None:
            metadata.token_to_req = metadata.token_to_req[local_start:local_end]
        return metadata


class AscendSFAKVOffloadDSACPImpl(AscendSFAKVOffloadImpl, AscendSFADSACPImpl):
    """SFA offload implementation retaining DSA-CP's token-sharded semantics."""

    supports_dsa_cp = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.qk_rope_head_dim <= 0:
            raise NotImplementedError(
                "DSA-CP Sparse KV offload currently requires a RoPE SFA main KV layout."
            )

    @staticmethod
    def _is_decode_only(attn_metadata) -> bool:
        # num_decode_tokens is rank-local in the composed metadata and may be
        # zero on a rank that still participates in a global decode batch.
        return (
            attn_metadata.attn_state in (AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding)
            and int(getattr(attn_metadata, "num_prefills", 0) or 0) == 0
            and int(getattr(attn_metadata, "num_decodes", 0) or 0) > 0
        )

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata,
    ):
        if self._is_decode_only(attn_metadata):
            # Decode queries are token-sharded. Compute each rank's local K/V
            # without writing a partial row into the NPU main cache; the DSA-CP
            # KV all-gather below reconstructs the complete token batch.
            k_nope, k_pe = self._compute_kv_only(kv_no_split, cos, sin)
            return k_pe, k_nope

        # Prefill/mixed batches retain the native DSA-CP local cache write.
        # The full gathered cache is committed to Host KV in _store_parallel_kv.
        return super(AscendSFAKVOffloadImpl, self).exec_kv(
            kv_no_split, cos, sin, kv_cache, slots, attn_metadata
        )

    def _store_parallel_kv(
        self,
        k_pe,
        k_nope,
        knope_scale,
        fused_kv_no_split,
        kv_ag_handles,
        kv_cache,
        slot_mapping_sfa,
        attn_metadata,
        full_gather_o_proj_enabled,
    ):
        manager = get_sparse_kv_offload_manager()
        layer_name = self._offload_layer_name()

        if self._is_decode_only(attn_metadata):
            # DSA-CP's all-gather restores the global token order on every rank.
            # TP0 alone commits this complete batch to the shared Host KV pool;
            # each rank then performs its own local TopK/LRU resident selection.
            for handle in kv_ag_handles:
                handle.wait()
            if full_gather_o_proj_enabled:
                self._all_gather_o_proj_full_weight()
            if fused_kv_no_split is None:
                raise RuntimeError("DSA-CP Sparse KV decode did not produce gathered main KV.")

            num_actual_tokens = int(attn_metadata.num_actual_tokens)
            gathered = fused_kv_no_split[:num_actual_tokens]
            k_pe_full, k_nope_full = gathered.split(
                [self.qk_rope_head_dim, self.kv_lora_rank],
                dim=-1,
            )
            k_nope_full = k_nope_full.view(num_actual_tokens, 1, self.kv_lora_rank)
            k_pe_full = k_pe_full.view(num_actual_tokens, 1, self.qk_rope_head_dim)
            k_cache_cpu, v_cache_cpu = self._cpu_cache_pair(manager, layer_name)
            manager.offload_new_kv(
                layer_name=layer_name,
                slot_mapping=slot_mapping_sfa[:num_actual_tokens],
                k_cache_cpu=k_cache_cpu,
                v_cache_cpu=v_cache_cpu,
                k_cache_npu=None,
                v_cache_npu=None,
                k=k_nope_full,
                v=k_pe_full,
                has_prefill=False,
                capturing=self._in_graph_runtime(),
            )
            return None, None

        # Prefill/mixed: let DSA-CP gather and store the complete global KV in
        # the retained NPU cache first, then let TP0 mirror those rows to Host KV.
        result = super(AscendSFAKVOffloadImpl, self)._store_parallel_kv(
            k_pe,
            k_nope,
            knope_scale,
            fused_kv_no_split,
            kv_ag_handles,
            kv_cache,
            slot_mapping_sfa,
            attn_metadata,
            full_gather_o_proj_enabled,
        )
        if kv_cache is None or kv_cache[0] is None or kv_cache[1] is None:
            raise RuntimeError(
                "DSA-CP Sparse KV prefill requires keep_device_kv_cache=True so the gathered "
                "global KV can be staged to the Host KV pool."
            )
        k_cache_cpu, v_cache_cpu = self._cpu_cache_pair(manager, layer_name)
        manager.offload_new_kv(
            layer_name=layer_name,
            slot_mapping=slot_mapping_sfa,
            k_cache_cpu=k_cache_cpu,
            v_cache_cpu=v_cache_cpu,
            k_cache_npu=kv_cache[0],
            v_cache_npu=kv_cache[1],
            k=None,
            v=None,
            has_prefill=True,
            capturing=self._in_graph_runtime(),
        )
        return result
