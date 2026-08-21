# Copyright (c) 2026 LightSeek Foundation

"""One-grid SM100 producer for mixed dense-FP8 and packed-R31 MLA."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from tokenspeed_mla.mla_decode_fp8 import (
    BlackwellMultiHeadLatentAttentionForwardFP8,
    SplitKVKernelArgs,
)


# Keep this coupled to the documented SplitKVKernelArgs ABI.  The mixed grid
# needs the child scheduler's declared Z extent before entering either body.
_SPLIT_KV_ARG_INDEX = 26


class BlackwellMixedFP8R31Producer:
    """Dispatch one complete FP8 or R31 producer mainloop per CTA.

    The two child kernels retain their independently typed TMA, MMA, scale,
    RoPE, softmax, and epilogue paths.  This wrapper only combines their grid:
    hot split CTAs precede cold split CTAs on grid Z.  Both children alias one
    raw shared-memory allocation sized to the larger child storage contract.
    """

    def __init__(
        self,
        hot_kernel: BlackwellMultiHeadLatentAttentionForwardFP8,
        cold_kernel: BlackwellMultiHeadLatentAttentionForwardFP8,
    ) -> None:
        if hot_kernel.use_tq_e2m1:
            raise ValueError("the hot mixed producer must use dense FP8")
        cold_is_normalized = cold_kernel.use_tq_r31_rope
        cold_is_physical = cold_kernel.use_tq_r31_physical_split_score
        if not (
            cold_kernel.use_tq_e2m1
            and (cold_is_normalized != cold_is_physical)
        ):
            raise ValueError(
                "the cold mixed producer must use exactly one packed-R31 format"
            )
        if cold_is_physical and not (
            cold_kernel.tq_r31_physical_split_score_dual_tmem
            and not cold_kernel.tq_r31_physical_split_score_lookahead
        ):
            raise ValueError(
                "the physical cold mixed producer requires dual TMEM without "
                "lookahead"
            )
        for name, hot_value, cold_value in (
            ("acc_dtype", hot_kernel.acc_dtype, cold_kernel.acc_dtype),
            ("lse_dtype", hot_kernel.lse_dtype, cold_kernel.lse_dtype),
            (
                "max_active_clusters",
                hot_kernel.max_active_clusters,
                cold_kernel.max_active_clusters,
            ),
            (
                "skip_correction_threshold",
                hot_kernel.skip_correction_threshold,
                cold_kernel.skip_correction_threshold,
            ),
            (
                "is_persistent",
                hot_kernel.is_persistent,
                cold_kernel.is_persistent,
            ),
            ("is_var_seq", hot_kernel.is_var_seq, cold_kernel.is_var_seq),
            (
                "is_var_split_kv",
                hot_kernel.is_var_split_kv,
                cold_kernel.is_var_split_kv,
            ),
            ("producer_only", hot_kernel.producer_only, cold_kernel.producer_only),
            ("is_causal", hot_kernel.is_causal, cold_kernel.is_causal),
            (
                "threads_per_cta",
                hot_kernel.threads_per_cta,
                cold_kernel.threads_per_cta,
            ),
            (
                "cluster_shape_mnk",
                hot_kernel.cluster_shape_mnk,
                cold_kernel.cluster_shape_mnk,
            ),
            (
                "mma_qk_tiler_mn",
                hot_kernel.mma_qk_tiler_mn,
                cold_kernel.mma_qk_tiler_mn,
            ),
            (
                "mma_pv_tiler_mn",
                hot_kernel.mma_pv_tiler_mn,
                cold_kernel.mma_pv_tiler_mn,
            ),
            ("page_size", hot_kernel.page_size, cold_kernel.page_size),
            ("fold_sq_factor", hot_kernel.fold_sq_factor, cold_kernel.fold_sq_factor),
            ("num_heads", hot_kernel.num_heads, cold_kernel.num_heads),
            ("seq_len_q", hot_kernel.seq_len_q, cold_kernel.seq_len_q),
            ("cp_world", hot_kernel.cp_world, cold_kernel.cp_world),
            (
                "use_runtime_causal_bound",
                hot_kernel.use_runtime_causal_bound,
                cold_kernel.use_runtime_causal_bound,
            ),
        ):
            if hot_value != cold_value:
                raise ValueError(
                    f"mixed producer requires equal {name}, got "
                    f"hot={hot_value!r} cold={cold_value!r}"
                )
        if not hot_kernel.producer_only:
            raise ValueError("mixed child kernels must be producer-only")
        if hot_kernel.is_persistent:
            raise ValueError("mixed producer requires non-persistent child kernels")
        if not hot_kernel.is_var_seq:
            raise ValueError("mixed producer requires variable-sequence child kernels")
        if hot_kernel.is_var_split_kv:
            raise ValueError("mixed producer requires fixed scalar split counts")
        if hot_kernel.cp_world != 1:
            raise ValueError("mixed producer currently requires cp_world=1")
        if not (hot_kernel.is_causal and hot_kernel.use_runtime_causal_bound):
            raise ValueError("mixed child kernels require runtime causal bounds")
        if hot_kernel.cluster_shape_mnk != (1, 1, 1):
            raise ValueError("mixed producer currently requires one-CTA M64 kernels")
        self.hot_kernel = hot_kernel
        self.cold_kernel = cold_kernel

    @cute.kernel
    def mixed_split_kv_kernel(
        self,
        hot_args: SplitKVKernelArgs,
        cold_args: SplitKVKernelArgs,
        HotStorage: cutlass.Constexpr,
        ColdStorage: cutlass.Constexpr,
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        gdimx, gdimy, _ = cute.arch.grid_dim()
        split_index = cute.arch.make_warp_uniform(bidz)
        hot_splits = hot_args[_SPLIT_KV_ARG_INDEX]
        cold_splits = cold_args[_SPLIT_KV_ARG_INDEX]

        # Allocate one union-sized region. Allocating child structs separately,
        # even in divergent branches, would charge their sum statically.
        storage_bytes = max(
            HotStorage.size_in_bytes(),
            ColdStorage.size_in_bytes(),
        )
        storage_alignment = max(
            HotStorage.__alignof__(),
            ColdStorage.__alignof__(),
        )
        smem = utils.SmemAllocator()
        raw_storage = smem.allocate(storage_bytes, storage_alignment)

        if split_index < hot_splits:
            hot_storage = HotStorage(raw_storage)
            self.hot_kernel._split_kv_kernel_impl(
                hot_args,
                hot_storage,
                (bidx, bidy, split_index),
                (gdimx, gdimy, hot_splits),
            )
        else:
            cold_storage = ColdStorage(raw_storage)
            cold_split_index = split_index - hot_splits
            self.cold_kernel._split_kv_kernel_impl(
                cold_args,
                cold_storage,
                (bidx, bidy, cold_split_index),
                (gdimx, gdimy, cold_splits),
            )

    @cute.jit
    def __call__(
        self,
        hot_q_latent: cute.Tensor,
        hot_q_rope: cute.Tensor,
        hot_c_latent: cute.Tensor,
        hot_c_rope: cute.Tensor,
        hot_page_table: cute.Tensor,
        hot_workspace: cute.Tensor,
        hot_split_kv: cutlass.Int32,
        hot_cache_seqs: cute.Tensor,
        hot_causal_seqs: cute.Tensor,
        cold_q_latent: cute.Tensor,
        cold_q_rope: cute.Tensor,
        cold_c_latent: cute.Tensor,
        cold_c_rope: cute.Tensor,
        cold_page_table: cute.Tensor,
        cold_workspace: cute.Tensor,
        cold_split_kv: cutlass.Int32,
        cold_cache_seqs: cute.Tensor,
        cold_causal_seqs: cute.Tensor,
        output_sentinel: cute.Tensor,
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        cold_scale: cute.Tensor,
        cold_rope_residual: cute.Tensor,
        cold_fault_status: Optional[cute.Tensor],
        stream: cuda.CUstream,
        use_pdl: cutlass.Constexpr = True,
    ):
        hot_args, hot_grid, HotStorage = self.hot_kernel(
            hot_q_latent,
            hot_q_rope,
            hot_c_latent,
            hot_c_rope,
            hot_page_table,
            output_sentinel,
            None,
            hot_workspace,
            hot_split_kv,
            hot_cache_seqs,
            hot_causal_seqs,
            None,
            softmax_scale,
            output_scale,
            stream,
            use_pdl,
            None,
            None,
            None,
            None,
            True,
        )
        cold_args, cold_grid, ColdStorage = self.cold_kernel(
            cold_q_latent,
            cold_q_rope,
            cold_c_latent,
            cold_c_rope,
            cold_page_table,
            output_sentinel,
            None,
            cold_workspace,
            cold_split_kv,
            cold_cache_seqs,
            cold_causal_seqs,
            None,
            softmax_scale,
            output_scale,
            stream,
            use_pdl,
            cold_scale,
            None,
            cold_rope_residual,
            cold_fault_status,
            True,
        )
        self.mixed_split_kv_kernel(
            hot_args,
            cold_args,
            HotStorage,
            ColdStorage,
        ).launch(
            grid=(hot_grid[0], hot_grid[1], hot_split_kv + cold_split_kv),
            block=[self.hot_kernel.threads_per_cta, 1, 1],
            cluster=self.hot_kernel.cluster_shape_mnk,
            smem=max(
                HotStorage.size_in_bytes(),
                ColdStorage.size_in_bytes(),
            ),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=use_pdl,
        )


__all__ = ["BlackwellMixedFP8R31Producer"]
