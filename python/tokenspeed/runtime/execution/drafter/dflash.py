# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import importlib.util
import json
import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig

from tokenspeed.runtime.distributed.comm_ops import all_gather_into_tensor
from tokenspeed.runtime.execution.cache_loc_kernel import compute_out_cache_loc_uniform
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, _lm_head_matmul
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import get_global_server_args
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.model_runner import ModelRunner
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput

logger = get_colorful_logger(__name__)


class DFlash(BaseDrafter):
    """DFlash block drafter.

    Prefer the native TokenSpeed draft runner when a draft model config is
    available. The local HF fallback is kept for bring-up and compatibility
    with older launch paths.
    """

    def __init__(
        self,
        spec_num_tokens: int,
        spec_num_steps: int,
        page_size: int,
        draft_model_runner: ModelRunner | None = None,
        req_to_page: torch.Tensor | None = None,
        attn_backend=None,
        token_to_kv_pool=None,
        runtime_states: RuntimeStates | None = None,
        input_buffers: InputBuffers | None = None,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__(
            spec_num_tokens=spec_num_tokens,
            spec_num_steps=spec_num_steps,
            draft_model_runner=draft_model_runner,
            runtime_states=runtime_states,
            input_buffers=input_buffers,
            page_size=page_size,
            req_to_page=req_to_page,
            attn_backend=attn_backend,
            token_to_kv_pool=token_to_kv_pool,
            vocab_size=vocab_size,
        )
        server_args = get_global_server_args()
        if not server_args.speculative_draft_model_path:
            raise ValueError("DFLASH requires --speculative-draft-model-path.")

        self.draft_path = server_args.speculative_draft_model_path
        self.device = (
            torch.device(draft_model_runner.device)
            if draft_model_runner is not None
            else torch.device(server_args.device)
        )
        self.native = draft_model_runner is not None
        if self.native:
            self.model = draft_model_runner.model
            self.dflash_module = None
        else:
            self.model, self.dflash_module = self._load_draft_model(self.draft_path)
            self.model.to(self.device).eval()

        cfg = self.model.config
        dflash_cfg = getattr(cfg, "dflash_config", {}) or {}
        self.target_layer_ids = [int(x) for x in dflash_cfg.get("target_layer_ids", [])]
        if not self.target_layer_ids:
            raise ValueError("DFLASH draft config must define dflash_config.target_layer_ids.")
        self.mask_token_id = int(dflash_cfg.get("mask_token_id"))
        self.block_size = int(getattr(cfg, "block_size", spec_num_tokens))
        if self.block_size != int(spec_num_tokens):
            logger.warning(
                "DFLASH block size mismatch: checkpoint block_size=%s, runtime speculative_num_draft_tokens=%s.",
                self.block_size,
                spec_num_tokens,
            )
        self.hidden_size = int(getattr(cfg, "hidden_size"))
        self._init_native_buffers()
        self._kv_cache: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._seq_lens: dict[int, int] = {}
        self._greedy_gathered_max: torch.Tensor | None = None
        self._greedy_gathered_ids: torch.Tensor | None = None
        self._greedy_gather_cap = 0
        self._batched_kv_weight: torch.Tensor | None = None
        self._batched_kv_bias: torch.Tensor | None = None
        if self.native:
            self._init_batched_kv_projector()

    def _init_native_buffers(self) -> None:
        if not self.native:
            return
        if self.input_buffers is None:
            raise ValueError("Native DFLASH requires input buffers.")
        if self.req_to_page is None:
            raise ValueError("Native DFLASH requires req_to_page.")
        if self.attn_backend is None or self.token_to_kv_pool is None:
            raise ValueError("Native DFLASH requires draft attention components.")

        max_bs = self.input_buffers.max_bs
        self.draft_seq_lens_buf = torch.zeros_like(self.input_buffers.seq_lens_buf)
        self.draft_out_cache_loc_buf = torch.empty(
            (max_bs * self.spec_num_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.draft_input_lengths_buf = torch.full(
            (max_bs,),
            self.spec_num_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.draft_extend_seq_lens_cpu = torch.full(
            (max_bs,),
            self.spec_num_tokens,
            dtype=torch.int32,
            pin_memory=True,
        )
        self.block_offsets = torch.arange(
            self.spec_num_tokens, dtype=torch.int64, device=self.device
        )
        self.block_ids_buf = torch.empty(
            (max_bs, self.spec_num_tokens), dtype=torch.int32, device=self.device
        )
        self.block_positions_buf = torch.empty(
            (max_bs, self.spec_num_tokens), dtype=torch.int64, device=self.device
        )

    def _init_batched_kv_projector(self) -> None:
        weights = []
        biases = []
        saw_bias = False
        for layer_idx, layer in enumerate(self.draft_model_runner.model.layers):
            attn = layer.self_attn
            qkv_proj = attn.qkv_proj
            weight = getattr(qkv_proj, "weight", None)
            if weight is None:
                logger.info(
                    "DFLASH batched KV projection disabled: qkv_proj has no plain weight at layer %s.",
                    layer_idx,
                )
                self._batched_kv_weight = None
                self._batched_kv_bias = None
                return

            kv_slice = slice(attn.q_size, attn.q_size + 2 * attn.kv_size)
            weights.append(weight[kv_slice])

            bias = getattr(qkv_proj, "bias", None)
            if bias is not None:
                saw_bias = True
                biases.append(bias[kv_slice])
            else:
                biases.append(None)

        self._batched_kv_weight = torch.stack(weights, dim=0).contiguous()
        if saw_bias:
            zero_bias = torch.zeros(
                (self._batched_kv_weight.shape[-2],),
                dtype=self._batched_kv_weight.dtype,
                device=self._batched_kv_weight.device,
            )
            self._batched_kv_bias = torch.stack(
                [b if b is not None else zero_bias for b in biases], dim=0
            ).contiguous()
        logger.info(
            "DFLASH batched KV projection enabled. n_layers=%s, kv_width=%s.",
            self._batched_kv_weight.shape[0],
            self._batched_kv_weight.shape[1],
        )

    def _kv_capacity(self, needed: int) -> int:
        block = 2048
        return ((needed + block - 1) // block) * block

    def _load_draft_model(self, model_path: str):
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        cfg._attn_implementation = "eager"

        module_path = os.path.join(model_path, "dflash.py")
        spec = importlib.util.spec_from_file_location("tokenspeed_local_dflash", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load DFlash module from {module_path}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model = module.DFlashDraftModel(cfg)
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            shard_names = sorted(set(weight_map.values()))
        else:
            shard_names = ["model.safetensors"]

        for shard_name in shard_names:
            state = load_file(os.path.join(model_path, shard_name), device="cpu")
            model.load_state_dict(state, strict=False)
            del state

        return model, module

    def bind_target_model(self, target_model) -> None:
        language_model = getattr(target_model, "language_model", target_model)
        self.target_model = target_model
        self.target_language_model = language_model
        self.embed_tokens = target_model.get_input_embeddings()
        self.lm_head = target_model.lm_head
        self.logits_processor = language_model.logits_processor

    def _repeat_kv(self, x: torch.Tensor, num_groups: int) -> torch.Tensor:
        if num_groups == 1:
            return x
        return x.repeat_interleave(num_groups, dim=1)

    def _append_target_hidden(
        self,
        pool_idx: int,
        target_hidden: torch.Tensor,
    ) -> None:
        if target_hidden.numel() == 0:
            return
        target_hidden = target_hidden.to(device=self.device, dtype=self.model.fc.weight.dtype)
        start_pos = self._seq_lens.get(pool_idx, 0)
        positions = torch.arange(
            start_pos,
            start_pos + target_hidden.shape[0],
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            ctx_hidden = self.model.hidden_norm(self.model.fc(target_hidden))
            ctx_hidden_b = ctx_hidden.unsqueeze(0)

            if pool_idx not in self._kv_cache:
                self._kv_cache[pool_idx] = []

            for layer_idx, layer in enumerate(self.model.layers):
                attn = layer.self_attn
                num_kv_heads = int(attn.config.num_key_value_heads)
                k = attn.k_proj(ctx_hidden_b)
                v = attn.v_proj(ctx_hidden_b)
                k = k.view(1, -1, num_kv_heads, attn.head_dim)
                v = v.view(1, -1, num_kv_heads, attn.head_dim)
                k = attn.k_norm(k).transpose(1, 2)
                v = v.transpose(1, 2)

                cos, sin = self.model.rotary_emb(ctx_hidden_b, positions.unsqueeze(0))
                dummy_q = torch.empty_like(k)
                _, k = self.dflash_module.apply_rotary_pos_emb(dummy_q, k, cos, sin)

                if layer_idx >= len(self._kv_cache[pool_idx]):
                    capacity = self._kv_capacity(start_pos + target_hidden.shape[0])
                    k_buf = torch.empty(
                        (k.shape[0], k.shape[1], capacity, k.shape[3]),
                        dtype=k.dtype,
                        device=k.device,
                    )
                    v_buf = torch.empty(
                        (v.shape[0], v.shape[1], capacity, v.shape[3]),
                        dtype=v.dtype,
                        device=v.device,
                    )
                    self._kv_cache[pool_idx].append((k_buf, v_buf))
                else:
                    k_buf, v_buf = self._kv_cache[pool_idx][layer_idx]
                    needed = start_pos + target_hidden.shape[0]
                    if k_buf.shape[2] < needed:
                        capacity = self._kv_capacity(needed)
                        new_k = torch.empty(
                            (k_buf.shape[0], k_buf.shape[1], capacity, k_buf.shape[3]),
                            dtype=k_buf.dtype,
                            device=k_buf.device,
                        )
                        new_v = torch.empty(
                            (v_buf.shape[0], v_buf.shape[1], capacity, v_buf.shape[3]),
                            dtype=v_buf.dtype,
                            device=v_buf.device,
                        )
                        new_k[:, :, :start_pos, :].copy_(k_buf[:, :, :start_pos, :])
                        new_v[:, :, :start_pos, :].copy_(v_buf[:, :, :start_pos, :])
                        self._kv_cache[pool_idx][layer_idx] = (new_k, new_v)
                        k_buf, v_buf = new_k, new_v

                k_buf, v_buf = self._kv_cache[pool_idx][layer_idx]
                end_pos = start_pos + target_hidden.shape[0]
                k_buf[:, :, start_pos:end_pos, :].copy_(k)
                v_buf[:, :, start_pos:end_pos, :].copy_(v)

        self._seq_lens[pool_idx] = start_pos + int(target_hidden.shape[0])

    def _draft_attention(
        self,
        attn,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
    ) -> torch.Tensor:
        num_heads = int(attn.config.num_attention_heads)
        num_kv_heads = int(attn.config.num_key_value_heads)
        q = attn.q_proj(hidden_states)
        k_noise = attn.k_proj(hidden_states)
        v_noise = attn.v_proj(hidden_states)

        q = q.view(1, -1, num_heads, attn.head_dim)
        k_noise = k_noise.view(1, -1, num_kv_heads, attn.head_dim)
        v_noise = v_noise.view(1, -1, num_kv_heads, attn.head_dim)
        q = attn.q_norm(q).transpose(1, 2)
        k_noise = attn.k_norm(k_noise).transpose(1, 2)
        v_noise = v_noise.transpose(1, 2)

        cos, sin = self.model.rotary_emb(hidden_states, positions.unsqueeze(0))
        q, k_noise = self.dflash_module.apply_rotary_pos_emb(q, k_noise, cos, sin)
        k = torch.cat([prefix_k, k_noise], dim=2)
        v = torch.cat([prefix_v, v_noise], dim=2)

        num_groups = num_heads // num_kv_heads
        k = self._repeat_kv(k, num_groups)
        v = self._repeat_kv(v, num_groups)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).reshape(1, hidden_states.shape[1], -1)
        return attn.o_proj(attn_output)

    def _greedy_sample_from_vocab_parallel_head(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self.lm_head, "weight") or not hasattr(
            self.lm_head, "shard_indices"
        ):
            metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE)
            logits = self.logits_processor._get_logits(
                hidden_states, self.lm_head, metadata
            )
            return torch.argmax(logits, dim=-1).to(torch.int32)

        shard = self.lm_head.shard_indices
        weight = self.lm_head.weight
        hidden_states = hidden_states.to(weight.dtype)

        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        chunk_len = int(hidden_states.shape[0])
        use_fused_lm_head = getattr(self.logits_processor, "_use_fused_lm_head", False)
        if num_org > 0:
            if use_fused_lm_head:
                base_logits = _lm_head_matmul(hidden_states, weight[:num_org])
            else:
                base_logits = torch.matmul(hidden_states, weight[:num_org].T)
            local_max, local_arg = torch.max(base_logits, dim=-1)
        else:
            local_max = torch.full(
                (chunk_len,),
                torch.finfo(weight.dtype).min,
                dtype=weight.dtype,
                device=hidden_states.device,
            )
            local_arg = torch.zeros(
                (chunk_len,), dtype=torch.int64, device=hidden_states.device
            )

        if num_added > 0:
            added_start = num_org_padded
            added_end = num_org_padded + num_added
            added_weight = weight[added_start:added_end]
            if use_fused_lm_head:
                added_logits = _lm_head_matmul(hidden_states, added_weight)
            else:
                added_logits = torch.matmul(hidden_states, added_weight.T)
            added_max, added_arg = torch.max(added_logits, dim=-1)
            use_added = added_max > local_max
            local_max = torch.where(use_added, added_max, local_max)
            local_arg = torch.where(
                use_added,
                added_arg.to(local_arg.dtype) + num_org_padded,
                local_arg,
            )

        if num_added == 0:
            global_ids = local_arg + org_vocab_start
        else:
            global_ids = torch.empty(
                (chunk_len,), dtype=torch.int64, device=hidden_states.device
            )
            is_base = local_arg < num_org
            global_ids[is_base] = org_vocab_start + local_arg[is_base]
            global_ids[~is_base] = added_vocab_start + (
                local_arg[~is_base] - num_org_padded
            )

        tp_size = int(self.logits_processor.tp_size)
        if tp_size == 1:
            return global_ids.to(torch.int32)

        needed = tp_size * chunk_len
        if (
            self._greedy_gather_cap < needed
            or self._greedy_gathered_max is None
            or self._greedy_gathered_ids is None
            or self._greedy_gathered_max.dtype != local_max.dtype
            or self._greedy_gathered_max.device != hidden_states.device
        ):
            self._greedy_gathered_max = torch.empty(
                (needed,), dtype=local_max.dtype, device=hidden_states.device
            )
            self._greedy_gathered_ids = torch.empty(
                (needed,), dtype=global_ids.dtype, device=hidden_states.device
            )
            self._greedy_gather_cap = needed

        gathered_max = self._greedy_gathered_max[:needed]
        gathered_ids = self._greedy_gathered_ids[:needed]
        all_gather_into_tensor(
            gathered_max,
            local_max.contiguous(),
            self.logits_processor.tp_rank,
            self.logits_processor.tp_group,
        )
        all_gather_into_tensor(
            gathered_ids,
            global_ids.contiguous(),
            self.logits_processor.tp_rank,
            self.logits_processor.tp_group,
        )

        gathered_max = gathered_max.view(tp_size, chunk_len)
        gathered_ids = gathered_ids.view(tp_size, chunk_len)
        best_rank = torch.argmax(gathered_max, dim=0).unsqueeze(0)
        return torch.gather(gathered_ids, 0, best_rank).view(-1).to(torch.int32)

    def _draft_one(self, pool_idx: int, current_token: int) -> torch.Tensor:
        if pool_idx not in self._kv_cache:
            raise RuntimeError("DFLASH draft cache is missing for request pool slot.")
        seq_len = self._seq_lens.get(pool_idx, 0)
        block_ids = torch.full(
            (self.spec_num_tokens,),
            self.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        block_ids[0] = int(current_token)
        positions = torch.arange(
            seq_len,
            seq_len + self.spec_num_tokens,
            dtype=torch.long,
            device=self.device,
        )

        with torch.inference_mode():
            hidden_states = self.embed_tokens(block_ids).unsqueeze(0).to(
                dtype=self.model.fc.weight.dtype
            )
            for layer, (prefix_k, prefix_v) in zip(
                self.model.layers, self._kv_cache[pool_idx], strict=True
            ):
                live_prefix_k = prefix_k[:, :, :seq_len, :]
                live_prefix_v = prefix_v[:, :, :seq_len, :]
                sliding_window = getattr(layer.self_attn, "sliding_window", None)
                if sliding_window is not None and live_prefix_k.shape[2] > int(
                    sliding_window
                ):
                    live_prefix_k = live_prefix_k[:, :, -int(sliding_window) :, :]
                    live_prefix_v = live_prefix_v[:, :, -int(sliding_window) :, :]
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
                attn_out = self._draft_attention(
                    layer.self_attn,
                    hidden_states,
                    positions,
                    live_prefix_k,
                    live_prefix_v,
                )
                hidden_states = residual + attn_out
                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                hidden_states = layer.mlp(hidden_states)
                hidden_states = residual + hidden_states
            hidden_states = self.model.norm(hidden_states)
            draft_hidden = hidden_states[:, 1:, :].reshape(
                self.spec_num_tokens - 1, self.hidden_size
            )
            return self._greedy_sample_from_vocab_parallel_head(draft_hidden)

    @nvtx_range("dflash_update_cache", color="purple")
    def _update_cache_from_target(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        accept_lengths: torch.Tensor,
    ) -> None:
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DFLASH requires target hidden states.")
        if hidden.shape[0] != base_ctx.input_num_tokens:
            raise RuntimeError(
                "DFLASH hidden-state/token mismatch: "
                f"hidden_tokens={hidden.shape[0]}, input_tokens={base_ctx.input_num_tokens}."
            )

        bs = base_ctx.bs
        lengths = self.input_buffers.input_lengths_buf[:bs].detach().cpu().tolist()
        req_pool_indices = (
            self.input_buffers.req_pool_indices_buf[:bs].detach().cpu().tolist()
        )
        chunks = torch.split(hidden, [int(x) for x in lengths], dim=0)

        for row, (pool_idx, chunk) in enumerate(zip(req_pool_indices, chunks, strict=True)):
            pool_idx = int(pool_idx)
            if row < base_ctx.num_extends:
                self._kv_cache.pop(pool_idx, None)
                self._seq_lens[pool_idx] = 0
                self._append_target_hidden(pool_idx, chunk.contiguous())
            else:
                accept_len = int(accept_lengths[row].item())
                self._append_target_hidden(pool_idx, chunk[:accept_len].contiguous())

    @nvtx_range("dflash_update_native_cache", color="purple")
    def _update_native_cache_from_target(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        accept_lengths: torch.Tensor,
    ) -> None:
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DFLASH requires target hidden states.")
        if hidden.shape[0] != base_ctx.input_num_tokens:
            raise RuntimeError(
                "DFLASH hidden-state/token mismatch: "
                f"hidden_tokens={hidden.shape[0]}, input_tokens={base_ctx.input_num_tokens}."
            )

        bs = base_ctx.bs
        lengths = self.input_buffers.input_lengths_buf[:bs].to(torch.int64)
        req_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
        positions = self.input_buffers.positions_buf[: base_ctx.input_num_tokens]
        cache_locs = self.input_buffers.out_cache_loc_buf[: base_ctx.input_num_tokens]

        if (
            base_ctx.num_extends == 0
            and torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        ):
            old_lens = self.runtime_states.valid_cache_lengths.index_select(
                0, req_pool_indices
            )
            self.draft_seq_lens_buf[:bs].copy_(
                old_lens.to(torch.int32) + accept_lengths[:bs].to(torch.int32)
            )
            self._write_native_cache(hidden, positions, cache_locs)
            return

        hidden_chunks = torch.split(hidden, lengths.detach().cpu().tolist(), dim=0)
        pos_chunks = torch.split(positions, lengths.detach().cpu().tolist(), dim=0)
        loc_chunks = torch.split(cache_locs, lengths.detach().cpu().tolist(), dim=0)

        selected_hidden = []
        selected_positions = []
        selected_cache_locs = []
        new_seq_lens = torch.empty((bs,), dtype=torch.int32, device=self.device)

        for row, (chunk, pos_chunk, loc_chunk) in enumerate(
            zip(hidden_chunks, pos_chunks, loc_chunks, strict=True)
        ):
            if row < base_ctx.num_extends:
                take = int(chunk.shape[0])
            else:
                take = int(accept_lengths[row].item())
            if take <= 0:
                pool_idx = req_pool_indices[row]
                new_seq_lens[row] = self.runtime_states.valid_cache_lengths[pool_idx]
                continue

            chunk = chunk[:take].contiguous()
            pos_chunk = pos_chunk[:take].contiguous()
            loc_chunk = loc_chunk[:take].contiguous()
            selected_hidden.append(chunk)
            selected_positions.append(pos_chunk)
            selected_cache_locs.append(loc_chunk)
            new_seq_lens[row] = (pos_chunk[-1] + 1).to(torch.int32)

        self.draft_seq_lens_buf[:bs].copy_(new_seq_lens)
        if not selected_hidden:
            return

        target_hidden = torch.cat(selected_hidden, dim=0)
        target_positions = torch.cat(selected_positions, dim=0)
        target_cache_locs = torch.cat(selected_cache_locs, dim=0)
        self._write_native_cache(target_hidden, target_positions, target_cache_locs)

    def _write_native_cache(
        self,
        target_hidden: torch.Tensor,
        target_positions: torch.Tensor,
        target_cache_locs: torch.Tensor,
    ) -> None:
        target_hidden = target_hidden.to(
            device=self.device,
            dtype=self.draft_model_runner.model.fc.weight.dtype,
        )
        with torch.inference_mode():
            ctx_hidden = self.draft_model_runner.model.project_target_hidden(
                target_hidden
            )
            if self._batched_kv_weight is not None:
                self._write_native_cache_batched(
                    ctx_hidden, target_positions, target_cache_locs
                )
                return
            for layer in self.draft_model_runner.model.layers:
                attn = layer.self_attn
                k, v = attn.kv_proj_only(ctx_hidden)
                k = attn.apply_k_norm(k)
                k = attn.apply_k_rope(target_positions, k)
                k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                v = v.view(-1, attn.num_kv_heads, attn.head_dim)
                self.token_to_kv_pool.set_kv_buffer(
                    attn.attn,
                    target_cache_locs,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )

    def _write_native_cache_batched(
        self,
        ctx_hidden: torch.Tensor,
        target_positions: torch.Tensor,
        target_cache_locs: torch.Tensor,
    ) -> None:
        kv_all = torch.einsum(
            "th,loh->lto", ctx_hidden, self._batched_kv_weight
        )
        if self._batched_kv_bias is not None:
            kv_all = kv_all + self._batched_kv_bias[:, None, :]

        for layer_idx, layer in enumerate(self.draft_model_runner.model.layers):
            attn = layer.self_attn
            k, v = kv_all[layer_idx].split([attn.kv_size, attn.kv_size], dim=-1)
            k = attn.apply_k_norm(k)
            k = attn.apply_k_rope(target_positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            self.token_to_kv_pool.set_kv_buffer(
                attn.attn,
                target_cache_locs,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
            )

    @staticmethod
    def _current_tokens_from_output(
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
        num_extends: int,
        spec_num_tokens: int,
    ) -> torch.Tensor:
        bs = accept_lengths.shape[0]
        current = torch.empty((bs,), dtype=torch.int32, device=output_tokens.device)
        if num_extends > 0:
            current[:num_extends] = output_tokens[:num_extends]
        num_decodes = bs - num_extends
        if num_decodes > 0:
            offsets = (
                torch.arange(num_decodes, dtype=torch.int64, device=output_tokens.device)
                * spec_num_tokens
                - 1
                + num_extends
            )
            current[num_extends:] = output_tokens[
                offsets + accept_lengths[num_extends:].to(torch.int64)
            ]
        return current

    def get_candidates(self, base_ctx: ForwardContext) -> torch.Tensor | None:
        num_extends = base_ctx.num_extends
        num_decodes = base_ctx.bs - num_extends
        if num_decodes == 0:
            return None
        num_decode_tokens = num_decodes * self.spec_num_tokens
        num_prefill_tokens = base_ctx.input_num_tokens - num_decode_tokens
        return self.input_buffers.input_ids_buf[
            num_prefill_tokens : base_ctx.input_num_tokens
        ].reshape(num_decodes, self.spec_num_tokens)

    def draft(self, current_tokens: torch.Tensor) -> torch.Tensor:
        if self.native:
            return self._draft_native(current_tokens)

        bs = current_tokens.shape[0]
        req_pool_indices = (
            self.input_buffers.req_pool_indices_buf[:bs].detach().cpu().tolist()
        )
        next_tokens = torch.empty(
            (bs, self.spec_num_tokens), dtype=torch.int32, device=self.device
        )
        next_tokens[:, 0] = current_tokens.to(torch.int32)
        current_cpu = current_tokens.detach().cpu().tolist()
        for row, (pool_idx, token_id) in enumerate(
            zip(req_pool_indices, current_cpu, strict=True)
        ):
            next_tokens[row, 1:] = self._draft_one(int(pool_idx), int(token_id))
        return next_tokens

    @nvtx_range("dflash_native_draft", color="purple")
    def _draft_native(self, current_tokens: torch.Tensor) -> torch.Tensor:
        bs = current_tokens.shape[0]
        req_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
        prefix_lens = self.draft_seq_lens_buf[:bs].clone()
        seq_lens_after = self.draft_seq_lens_buf[:bs]
        seq_lens_after.copy_(prefix_lens + int(self.spec_num_tokens))

        block_ids = self.block_ids_buf[:bs]
        block_ids.fill_(int(self.mask_token_id))
        block_ids[:, 0].copy_(current_tokens.to(torch.int32))
        block_positions = self.block_positions_buf[:bs]
        block_positions.copy_(prefix_lens.to(torch.int64).unsqueeze(1) + self.block_offsets)

        cache_locs = self.draft_out_cache_loc_buf[: bs * self.spec_num_tokens]
        compute_out_cache_loc_uniform(
            out_cache_loc_ptr=cache_locs,
            req_pool_indices=req_pool_indices,
            uniform_input_length=self.spec_num_tokens,
            cache_start=prefix_lens,
            req_to_pages=self.req_to_page,
            page_size=self.page_size,
        )

        self.attn_backend.init_forward_metadata(
            bs=bs,
            num_extends=0,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens_after,
            req_to_page=self.req_to_page,
            forward_mode=ForwardMode.DECODE,
            extend_seq_lens_cpu=self.draft_extend_seq_lens_cpu[:bs],
        )

        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_page=self.req_to_page,
            bs=bs,
            num_extends=0,
            input_num_tokens=bs * self.spec_num_tokens,
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

        flat_ids = block_ids.reshape(-1)
        input_embeds = self.embed_tokens(flat_ids)
        with torch.inference_mode():
            logits_output = self.draft_model_runner.forward(
                ctx=ctx,
                input_ids=flat_ids,
                positions=block_positions.reshape(-1),
                out_cache_loc=cache_locs,
                input_lengths=self.draft_input_lengths_buf[:bs],
                captured_hidden_states=None,
                input_embeds=input_embeds,
            )

        draft_hidden = logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("Native DFLASH draft model did not return hidden states.")
        draft_hidden = draft_hidden.view(bs, self.spec_num_tokens, self.hidden_size)

        next_tokens = torch.empty(
            (bs, self.spec_num_tokens), dtype=torch.int32, device=self.device
        )
        next_tokens[:, 0] = current_tokens.to(torch.int32)
        sampled = self._greedy_sample_from_vocab_parallel_head(
            draft_hidden[:, 1:, :].reshape(-1, self.hidden_size)
        )
        next_tokens[:, 1:] = sampled.view(bs, self.spec_num_tokens - 1)
        return next_tokens

    @nvtx_range("drafter:dflash", color="purple")
    def run(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self, "target_model"):
            raise RuntimeError("DFLASH drafter is not bound to a target model.")
        if self.native:
            self._update_native_cache_from_target(base_ctx, logits_output, accept_lengths)
        else:
            self._update_cache_from_target(base_ctx, logits_output, accept_lengths)
        current_tokens = self._current_tokens_from_output(
            output_tokens,
            accept_lengths,
            base_ctx.num_extends,
            self.spec_num_tokens,
        )
        return self.draft(current_tokens)
