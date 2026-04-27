from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from alexandros.adaptive_depth import AdaptiveDepthController
from alexandros.attention_mla import MLAAttention
from alexandros.bitlinear import make_linear
from alexandros.blocks import RMSNorm
from alexandros.configuration_alexandros import AlexandrosConfig
from alexandros.diffusion import MaskedDiffusionScheduler
from alexandros.initialization import (
    initialize_embedding,
    initialize_linear,
    initialize_norm,
)
from alexandros.kv_cache import TurboQuantPacket, validate_turboquant_packet
from alexandros.latent_reasoning import LatentDiffusionReasoner, LatentThoughtVAE
from alexandros.moe import MoEFeedForward, MoEStats
from alexandros.ssm_gated_deltanet import GatedDeltaNetBlock
from alexandros.ssm_mamba2 import Mamba2Block
from alexandros.ssm_matrix_deltanet import MatrixGatedDeltaNetBlock
from alexandros.training.objectives import IGNORE_INDEX
from alexandros.ttt import TTTState

StopSequenceInput = (
    list[list[int] | tuple[int, ...]] | tuple[list[int] | tuple[int, ...], ...] | None
)
AttentionCache = dict[str, torch.Tensor | TurboQuantPacket]

_INPUT_ID_DTYPES = {torch.int32, torch.int64}


def _load_safetensors_torch() -> Any:
    try:
        return importlib.import_module("safetensors.torch")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "safe_serialization=True requires the optional 'safetensors' package"
        ) from exc


class GenerationMode(str, Enum):
    AUTOREGRESSIVE = "autoregressive"
    BLOCK_DIFFUSION = "block_diffusion"
    LATENT_REASONING = "latent_reasoning"
    HYBRID = "hybrid"


@dataclass
class AlexandrosModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: list[AttentionCache | None] | None = None
    past_ssm_states: list[torch.Tensor | None] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    moe_stats: tuple[MoEStats | None, ...] | None = None
    halting: torch.Tensor | None = None


@dataclass
class AlexandrosCausalLMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    past_key_values: list[AttentionCache | None] | None = None
    past_ssm_states: list[torch.Tensor | None] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    moe_stats: tuple[MoEStats | None, ...] | None = None


class AlexandrosBlock(nn.Module):
    def __init__(self, config: AlexandrosConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_attention = config.is_attention_layer(layer_idx)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if self.is_attention:
            self.mixer = MLAAttention(config)
        elif config.linear_mixer_backend == "gated_deltanet":
            self.mixer = GatedDeltaNetBlock(config)
        elif config.linear_mixer_backend == "matrix_deltanet":
            self.mixer = MatrixGatedDeltaNetBlock(config)
        elif config.linear_mixer_backend == "mamba2":
            self.mixer = Mamba2Block(config)
        else:  # Defensive guard; AlexandrosConfig validates this.
            raise ValueError("unsupported linear_mixer_backend")
        self.moe = MoEFeedForward(config)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: AttentionCache | None = None,
        use_cache: bool = False,
        ssm_state: torch.Tensor | None = None,
        diffusion_timestep: int | torch.Tensor | None = None,
        diffusion_token_state: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, AttentionCache | None, torch.Tensor | None, MoEStats | None
    ]:
        residual = hidden_states
        normed = self.input_norm(hidden_states)
        cache = None
        next_ssm_state = None
        if self.is_attention:
            is_causal = (
                diffusion_timestep is None
                or self.config.diffusion_attention_mask_mode == "causal"
            )
            mixed, cache = self.mixer(
                normed,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                is_causal=is_causal,
            )
        else:
            mixed, next_ssm_state = self.mixer(
                normed,
                state=ssm_state,
                attention_mask=attention_mask,
            )
        hidden_states = residual + self.residual_dropout(mixed)
        moe_out = self.moe(
            self.post_norm(hidden_states),
            diffusion_timestep=diffusion_timestep,
            diffusion_token_state=diffusion_token_state,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + self.residual_dropout(moe_out)
        return hidden_states, cache, next_ssm_state, self.moe.last_stats


class AlexandrosModel(nn.Module):
    config_class = AlexandrosConfig

    def __init__(self, config: AlexandrosConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [AlexandrosBlock(config, idx) for idx in range(config.num_hidden_layers)]
        )
        self.loop_index_embeddings = (
            nn.Embedding(config.recurrent_depth, config.hidden_size)
            if config.use_loop_index_embeddings
            else None
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.adaptive_depth = (
            AdaptiveDepthController(config) if config.enable_adaptive_depth else None
        )
        self.last_stack_stats: dict[str, int | bool] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_embedding(
            self.embed_tokens,
            zero_index=self.config.pad_token_id,
        )
        initialize_norm(self.norm)
        if self.loop_index_embeddings is not None:
            initialize_embedding(self.loop_index_embeddings)
        if self.adaptive_depth is not None:
            self.adaptive_depth.reset_parameters()

    def _past_sequence_length(
        self, past_key_values: list[AttentionCache | None] | None
    ) -> int:
        if not past_key_values:
            return 0
        lengths = []
        for past in past_key_values:
            if past is None:
                continue
            if "c_kv" in past:
                c_kv = past["c_kv"]
                if torch.is_tensor(c_kv):
                    lengths.append(c_kv.size(1))
            elif "c_kv_packet" in past:
                packet = past["c_kv_packet"]
                if isinstance(packet, TurboQuantPacket):
                    lengths.append(packet.q.size(1))
        return max(lengths, default=0)

    def _validate_sequence_length(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[AttentionCache | None] | None = None,
    ) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in _INPUT_ID_DTYPES:
            raise ValueError("input_ids must be an integer tensor of token IDs")
        if input_ids.size(0) == 0:
            raise ValueError("input_ids batch size must be > 0")
        if input_ids.size(1) == 0:
            raise ValueError("input_ids sequence length must be > 0")
        if (
            input_ids.min().item() < 0
            or input_ids.max().item() >= self.config.vocab_size
        ):
            raise ValueError("input_ids contain token IDs outside [0, vocab_size)")
        total_length = input_ids.size(1) + self._past_sequence_length(past_key_values)
        if total_length > self.config.max_position_embeddings:
            raise ValueError(
                "sequence length "
                f"{total_length} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )

    def _validate_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        past_key_values: list[AttentionCache | None] | None = None,
    ) -> None:
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if attention_mask.size(0) != input_ids.size(0):
            raise ValueError("attention_mask batch size must match input_ids")
        past_len = self._past_sequence_length(past_key_values)
        valid_lengths = {input_ids.size(1), input_ids.size(1) + past_len}
        if attention_mask.size(1) not in valid_lengths:
            raise ValueError(
                "attention_mask length must match current input length or "
                "full cached key/value length"
            )
        if (
            torch.is_floating_point(attention_mask)
            and not torch.isfinite(attention_mask).all()
        ):
            raise ValueError("attention_mask must contain only finite 0/1 values")
        valid_mask_values = attention_mask.eq(0) | attention_mask.eq(1)
        if not valid_mask_values.all():
            raise ValueError("attention_mask must contain only 0/1 or bool values")
        bool_mask = attention_mask.to(dtype=torch.bool)
        if not bool_mask.any(dim=1).all():
            raise ValueError(
                "attention_mask must contain at least one non-pad token per row"
            )
        current_mask = bool_mask[:, -input_ids.size(1) :]
        if not current_mask[:, 0].all():
            raise ValueError(
                "left-padded current inputs are not supported by the reference causal attention"
            )
        if bool_mask.size(1) > 1 and (bool_mask[:, 1:] & ~bool_mask[:, :-1]).any():
            raise ValueError("attention_mask must be right-padded")

    def _validate_past_state_shapes(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[AttentionCache | None],
        past_ssm_states: list[torch.Tensor | None],
    ) -> None:
        batch = input_ids.size(0)
        for layer_idx, past in enumerate(past_key_values):
            if past is None:
                continue
            if "c_kv" in past:
                c_kv = past["c_kv"]
                if not torch.is_tensor(c_kv):
                    raise ValueError(
                        f"past_key_values[{layer_idx}]['c_kv'] must be a tensor"
                    )
                if (
                    c_kv.ndim != 3
                    or c_kv.size(0) != batch
                    or c_kv.size(-1) != self.config.kv_lora_rank
                ):
                    raise ValueError(
                        "past_key_values c_kv entries must have shape "
                        "[batch, cached_sequence, kv_lora_rank]"
                    )
                cached_sequence = c_kv.size(1)
            elif "c_kv_packet" in past:
                packet = past["c_kv_packet"]
                if not isinstance(packet, TurboQuantPacket):
                    raise ValueError(
                        f"past_key_values[{layer_idx}]['c_kv_packet'] must be a TurboQuantPacket"
                    )
                validate_turboquant_packet(
                    packet,
                    expected_batch=batch,
                    expected_last_dim=self.config.kv_lora_rank,
                )
                if packet.q.ndim != 3:
                    raise ValueError(
                        "TurboQuant cache packet q entries must have shape "
                        "[batch, cached_sequence, kv_lora_rank]"
                    )
                cached_sequence = packet.q.size(1)
            else:
                raise ValueError(
                    "past_key_values entries must contain c_kv or c_kv_packet"
                )
            if self.config.mla_d_r > 0:
                k_rope = past.get("k_rope")
                if not torch.is_tensor(k_rope) or (
                    k_rope.ndim != 3
                    or k_rope.size(0) != batch
                    or k_rope.size(1) != cached_sequence
                    or k_rope.size(-1) != self.config.mla_d_r
                ):
                    raise ValueError(
                        "past_key_values k_rope entries must have shape "
                        "[batch, cached_sequence, mla_rope_dim]"
                    )
        for layer_idx, state in enumerate(past_ssm_states):
            if state is None:
                continue
            layer = self.layers[layer_idx]
            if layer.is_attention or not hasattr(layer.mixer, "recurrent_state_shape"):
                raise ValueError(
                    f"past_ssm_states[{layer_idx}] is only valid for recurrent mixer layers"
                )
            expected_shape = layer.mixer.recurrent_state_shape(batch)
            if not torch.is_tensor(state) or tuple(state.shape) != expected_shape:
                raise ValueError(
                    f"past_ssm_states[{layer_idx}] must have shape {expected_shape}"
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[AttentionCache | None] | None = None,
        past_ssm_states: list[torch.Tensor | None] | None = None,
        use_cache: bool = False,
        diffusion_timestep: int | torch.Tensor | None = None,
        ttt_state: TTTState | None = None,
        output_hidden_states: bool = False,
    ) -> AlexandrosModelOutput:
        self._validate_sequence_length(input_ids, past_key_values)
        if use_cache and diffusion_timestep is not None:
            raise ValueError("use_cache is not supported for diffusion forwards")
        recurrent_cache_requested = use_cache or any(
            past is not None for past in (past_key_values or ())
        )
        recurrent_state_requested = any(
            state is not None for state in (past_ssm_states or ())
        )
        if (
            self.config.uses_recurrent_stack
            and self.config.recurrent_depth > 1
            and (recurrent_cache_requested or recurrent_state_requested)
        ):
            raise ValueError(
                "cache reuse is not supported when staged recurrent stack "
                "uses recurrent_depth > 1"
            )
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        self._validate_attention_mask(attention_mask, input_ids, past_key_values)
        hidden_states = self.embed_tokens(input_ids)
        all_hidden: list[torch.Tensor] = [hidden_states] if output_hidden_states else []
        next_cache: list[AttentionCache | None] = []
        next_ssm_states: list[torch.Tensor | None] = []
        moe_stats: list[MoEStats | None] = []
        past_key_values = past_key_values or [None] * len(self.layers)
        past_ssm_states = past_ssm_states or [None] * len(self.layers)
        if len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values length must match num_hidden_layers")
        if len(past_ssm_states) != len(self.layers):
            raise ValueError("past_ssm_states length must match num_hidden_layers")
        self._validate_past_state_shapes(input_ids, past_key_values, past_ssm_states)
        diffusion_token_state = (
            input_ids.eq(self.config.mask_token_id)
            if diffusion_timestep is not None and self.config.moe_token_state_routing
            else None
        )
        position_ids = None
        if self.config.moe_position_routing:
            past_len = self._past_sequence_length(past_key_values)
            position_ids = (
                torch.arange(
                    past_len,
                    past_len + input_ids.size(1),
                    device=input_ids.device,
                    dtype=torch.long,
                )
                .view(1, -1)
                .expand(input_ids.size(0), -1)
            )

        layer_executions = 0
        recurrent_loop_count = 0

        def run_layer(layer_idx: int) -> None:
            nonlocal hidden_states
            nonlocal layer_executions
            layer = self.layers[layer_idx]
            past = past_key_values[layer_idx]
            ssm_state = past_ssm_states[layer_idx]
            hidden_states, cache, next_ssm_state, stats = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past,
                use_cache=use_cache,
                ssm_state=ssm_state,
                diffusion_timestep=diffusion_timestep,
                diffusion_token_state=diffusion_token_state,
                position_ids=position_ids,
            )
            if output_hidden_states:
                all_hidden.append(hidden_states)
            next_cache[layer_idx] = cache
            next_ssm_states[layer_idx] = next_ssm_state
            moe_stats[layer_idx] = stats
            layer_executions += 1

        next_cache = [None] * len(self.layers)
        next_ssm_states = [None] * len(self.layers)
        moe_stats = [None] * len(self.layers)
        if self.config.uses_recurrent_stack:
            prelude_end = self.config.prelude_layers
            recurrent_start, recurrent_end = self.config.recurrent_layer_range
            for layer_idx in range(prelude_end):
                run_layer(layer_idx)
            for loop_idx in range(self.config.recurrent_depth):
                recurrent_loop_count += 1
                if (
                    self.loop_index_embeddings is not None
                    and diffusion_timestep is None
                ):
                    loop_ids = torch.full(
                        (hidden_states.size(0),),
                        loop_idx,
                        dtype=torch.long,
                        device=hidden_states.device,
                    )
                    hidden_states = hidden_states + self.loop_index_embeddings(
                        loop_ids
                    ).unsqueeze(1)
                for layer_idx in range(recurrent_start, recurrent_end):
                    run_layer(layer_idx)
            for layer_idx in range(recurrent_end, len(self.layers)):
                run_layer(layer_idx)
        else:
            for layer_idx in range(len(self.layers)):
                run_layer(layer_idx)
        self.last_stack_stats = {
            "uses_recurrent_stack": self.config.uses_recurrent_stack,
            "recurrent_depth": self.config.recurrent_depth
            if self.config.uses_recurrent_stack
            else 0,
            "recurrent_loop_count": recurrent_loop_count,
            "layer_executions": layer_executions,
            "loop_index_embeddings_applied": bool(
                self.loop_index_embeddings is not None
                and self.config.uses_recurrent_stack
                and diffusion_timestep is None
            ),
        }

        halting = None
        if self.adaptive_depth is not None:
            hidden_states, halting = self.adaptive_depth(hidden_states)
        hidden_states = self.norm(hidden_states)
        if ttt_state is not None:
            hidden_states = ttt_state.apply(hidden_states)
        return AlexandrosModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache if use_cache else None,
            past_ssm_states=next_ssm_states if use_cache else None,
            hidden_states=tuple(all_hidden) if output_hidden_states else None,
            moe_stats=tuple(moe_stats),
            halting=halting,
        )


class AlexandrosPreTrainedMixin:
    config: AlexandrosConfig
    tokenizer_metadata_name = "tokenizer_metadata.json"
    checkpoint_metadata_name = "checkpoint_metadata.json"

    def _generation_config(self) -> dict[str, Any]:
        return {
            "bos_token_id": self.config.bos_token_id,
            "eos_token_id": self.config.eos_token_id,
            "pad_token_id": self.config.pad_token_id,
            "max_length": self.config.max_position_embeddings,
            "max_new_tokens": 16,
            "do_sample": False,
            "temperature": 1.0,
            "top_k": None,
            "top_p": None,
            "repetition_penalty": 1.0,
            "block_diffusion_steps": self.config.diffusion_steps,
            "block_diffusion_block_size": None,
            "block_diffusion_confidence_schedule": "median",
            "block_diffusion_remask_low_confidence": False,
        }

    def _validate_tokenizer_metadata(self, tokenizer_metadata: dict[str, Any]) -> None:
        if not isinstance(tokenizer_metadata, dict):
            raise ValueError("tokenizer_metadata must be a JSON-serializable object")
        try:
            json.dumps(tokenizer_metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("tokenizer_metadata must be JSON serializable") from exc
        vocab_size = tokenizer_metadata.get("vocab_size")
        if vocab_size is not None:
            if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
                raise ValueError("tokenizer_metadata vocab_size must be an integer")
            if vocab_size != self.config.vocab_size:
                raise ValueError(
                    "tokenizer_metadata vocab_size must match config.vocab_size"
                )
        special_sources = [tokenizer_metadata]
        special_tokens = tokenizer_metadata.get("special_tokens")
        if special_tokens is not None:
            if not isinstance(special_tokens, dict):
                raise ValueError("tokenizer_metadata special_tokens must be an object")
            special_sources.append(special_tokens)
        for source in special_sources:
            for field in (
                "pad_token_id",
                "bos_token_id",
                "eos_token_id",
                "mask_token_id",
            ):
                if field not in source:
                    continue
                value = source[field]
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(f"tokenizer_metadata {field} must be an integer")
                if value != getattr(self.config, field):
                    raise ValueError(
                        f"tokenizer_metadata {field} must match config.{field}"
                    )

    def _tokenizer_metadata_payload(
        self,
        tokenizer_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        self._validate_tokenizer_metadata(tokenizer_metadata)
        return {
            "format": "open-alexandros-tokenizer-metadata",
            "format_version": 1,
            "tokenizer": tokenizer_metadata,
            "config": {
                "vocab_size": self.config.vocab_size,
                "pad_token_id": self.config.pad_token_id,
                "bos_token_id": self.config.bos_token_id,
                "eos_token_id": self.config.eos_token_id,
                "mask_token_id": self.config.mask_token_id,
            },
        }

    def _checkpoint_metadata_payload(
        self,
        checkpoint_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(checkpoint_metadata, dict):
            raise ValueError("checkpoint_metadata must be a JSON-serializable object")
        try:
            json.dumps(checkpoint_metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint_metadata must be JSON serializable") from exc
        return {
            "format": "open-alexandros-checkpoint-metadata",
            "format_version": 1,
            "model_class": self.__class__.__name__,
            "checkpoint": checkpoint_metadata,
        }

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        safe_serialization: bool = False,
        tokenizer_metadata: dict[str, Any] | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        tokenizer_payload = (
            self._tokenizer_metadata_payload(tokenizer_metadata)
            if tokenizer_metadata is not None
            else None
        )
        checkpoint_payload = (
            self._checkpoint_metadata_payload(checkpoint_metadata)
            if checkpoint_metadata is not None
            else None
        )
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(path)
        weights_name = (
            "model.safetensors" if safe_serialization else "pytorch_model.bin"
        )
        if safe_serialization:
            _load_safetensors_torch().save_file(self.state_dict(), path / weights_name)
        else:
            torch.save(self.state_dict(), path / weights_name)
        (path / "generation_config.json").write_text(
            json.dumps(self._generation_config(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if tokenizer_metadata is not None:
            (path / self.tokenizer_metadata_name).write_text(
                json.dumps(
                    tokenizer_payload,
                    indent=2,
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        if checkpoint_metadata is not None:
            (path / self.checkpoint_metadata_name).write_text(
                json.dumps(
                    checkpoint_payload,
                    indent=2,
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        extra = {
            "checkpoint_format_version": 1,
            "format": "open-alexandros-reference",
            "model_class": self.__class__.__name__,
            "weights": weights_name,
            "safe_serialization": safe_serialization,
            "preferred_future_weights": "model.safetensors",
            "hf_compatibility": {
                "transformers_required": False,
                "config_class": self.config.__class__.__name__,
                "model_class": self.__class__.__name__,
                "auto_config_registered": False,
                "auto_model_registered": False,
                "requires_trust_remote_code": False,
            },
            "tokenizer_metadata": self.tokenizer_metadata_name
            if tokenizer_metadata is not None
            else None,
            "checkpoint_metadata": self.checkpoint_metadata_name
            if checkpoint_metadata is not None
            else None,
        }
        (path / "alexandros_extra.json").write_text(
            json.dumps(extra, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def load_tokenizer_metadata(path: str | Path) -> dict[str, Any] | None:
        path = Path(path)
        metadata_path = path
        if path.is_dir():
            metadata_path = path / AlexandrosPreTrainedMixin.tokenizer_metadata_name
        if not metadata_path.exists():
            return None
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("format") != "open-alexandros-tokenizer-metadata":
            raise ValueError("unsupported Alexandros tokenizer metadata format")
        if payload.get("format_version", 1) > 1:
            raise ValueError("unsupported Alexandros tokenizer metadata version")
        tokenizer = payload.get("tokenizer")
        if not isinstance(tokenizer, dict):
            raise ValueError("Alexandros tokenizer metadata payload is malformed")
        return tokenizer

    @staticmethod
    def load_checkpoint_metadata(path: str | Path) -> dict[str, Any] | None:
        path = Path(path)
        metadata_path = path
        if path.is_dir():
            metadata_path = path / AlexandrosPreTrainedMixin.checkpoint_metadata_name
        if not metadata_path.exists():
            return None
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("format") != "open-alexandros-checkpoint-metadata":
            raise ValueError("unsupported Alexandros checkpoint metadata format")
        if payload.get("format_version", 1) > 1:
            raise ValueError("unsupported Alexandros checkpoint metadata version")
        checkpoint = payload.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise ValueError("Alexandros checkpoint metadata payload is malformed")
        return checkpoint

    @staticmethod
    def _allowed_handoff_prefixes(
        source_class: str | None,
        target_class: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if source_class in {None, target_class}:
            return (), ()
        latent_prefixes = ("latent_vae.", "latent_reasoner.")
        if (
            source_class == "AlexandrosForCausalLM"
            and target_class == "AlexandrosForDiffusionLM"
        ):
            return latent_prefixes, ()
        if (
            source_class == "AlexandrosForDiffusionLM"
            and target_class == "AlexandrosForCausalLM"
        ):
            return (), latent_prefixes
        raise ValueError(
            f"cannot load {source_class} checkpoint into {target_class} without a handoff rule"
        )

    @staticmethod
    def _is_allowed_state_dict_delta(name: str, prefixes: tuple[str, ...]) -> bool:
        return any(name.startswith(prefix) for prefix in prefixes)

    def _load_state_dict_for_handoff(
        self,
        state: dict[str, torch.Tensor],
        *,
        source_class: str | None,
    ) -> None:
        target_class = self.__class__.__name__
        allowed_missing, allowed_unexpected = self._allowed_handoff_prefixes(
            source_class,
            target_class,
        )
        if not allowed_missing and not allowed_unexpected:
            self.load_state_dict(state)
            return
        result = self.load_state_dict(state, strict=False)
        bad_missing = [
            name
            for name in result.missing_keys
            if not self._is_allowed_state_dict_delta(name, allowed_missing)
        ]
        bad_unexpected = [
            name
            for name in result.unexpected_keys
            if not self._is_allowed_state_dict_delta(name, allowed_unexpected)
        ]
        if bad_missing or bad_unexpected:
            details = []
            if bad_missing:
                details.append(f"missing={bad_missing}")
            if bad_unexpected:
                details.append(f"unexpected={bad_unexpected}")
            raise ValueError(
                "checkpoint state dict is incompatible: " + "; ".join(details)
            )

    @classmethod
    def from_pretrained(
        cls, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> Any:
        path = Path(path)
        config = AlexandrosConfig.from_pretrained(path)
        model = cls(config)
        extra_path = path / "alexandros_extra.json"
        if extra_path.exists():
            extra = json.loads(extra_path.read_text(encoding="utf-8"))
            if extra.get("checkpoint_format_version", 1) > 1:
                raise ValueError("unsupported Alexandros checkpoint format version")
        else:
            extra = {}
        source_class = extra.get("model_class")
        if source_class is not None and not isinstance(source_class, str):
            raise ValueError("alexandros_extra.json model_class must be a string")
        weights_name = extra.get("weights")
        if weights_name is None:
            weights_name = (
                "model.safetensors"
                if (path / "model.safetensors").exists()
                else "pytorch_model.bin"
            )
        weights_path = path / weights_name
        if weights_path.suffix == ".safetensors":
            state = _load_safetensors_torch().load_file(
                weights_path, device=str(map_location)
            )
            model._load_state_dict_for_handoff(state, source_class=source_class)
            return model
        try:
            state = torch.load(
                weights_path,
                map_location=map_location,
                weights_only=True,
            )
        except TypeError:
            state = torch.load(weights_path, map_location=map_location)
        model._load_state_dict_for_handoff(state, source_class=source_class)
        return model


class AlexandrosForCausalLM(nn.Module, AlexandrosPreTrainedMixin):
    def __init__(self, config: AlexandrosConfig) -> None:
        super().__init__()
        self.config = config
        self.model = AlexandrosModel(config)
        self.lm_head = make_linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            variant=config.variant,
            activation_bits=config.bitnet_activation_bits,
        )
        initialize_linear(self.lm_head)
        if config.tie_word_embeddings and config.variant == "heavy":
            self.lm_head.weight = self.model.embed_tokens.weight  # type: ignore[attr-defined]

    def _validate_generation_length(
        self, input_ids: torch.Tensor, max_new_tokens: int
    ) -> None:
        self._validate_max_new_tokens(max_new_tokens)
        attention_mask = self._validate_generation_prompt(input_ids)
        if max_new_tokens == 0:
            return
        prompt_length = int(attention_mask.sum(dim=1).max().item())
        total_length = prompt_length + max_new_tokens
        if total_length > self.config.max_position_embeddings:
            raise ValueError(
                "requested generation length "
                f"{total_length} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )

    def _validate_max_new_tokens(self, max_new_tokens: int) -> None:
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
            raise ValueError("max_new_tokens must be a non-negative integer")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")

    def _validate_generation_steps(self, steps: int, field: str) -> int:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError(f"{field} must be a positive integer")
        return steps

    def _validate_optional_positive_int(
        self,
        value: int | None,
        field: str,
    ) -> int | None:
        if value is None:
            return None
        return self._validate_generation_steps(value, field)

    def _validate_generation_prompt(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.model._validate_sequence_length(input_ids)
        attention_mask = input_ids.ne(self.config.pad_token_id)
        self.model._validate_attention_mask(attention_mask, input_ids)
        return attention_mask

    def _has_padded_generation_prompts(self, input_ids: torch.Tensor) -> bool:
        attention_mask = self._validate_generation_prompt(input_ids)
        return not attention_mask.all().item()

    def _generate_variable_length_batch(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        generate_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        attention_mask = self._validate_generation_prompt(input_ids)
        lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
        outputs = []
        for row_idx, length in enumerate(lengths.tolist()):
            compact_prompt = input_ids[row_idx : row_idx + 1, :length]
            outputs.append(
                self.generate(
                    compact_prompt,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                ).squeeze(0)
            )
        max_length = max(output.numel() for output in outputs)
        padded = input_ids.new_full(
            (input_ids.size(0), max_length),
            self.config.pad_token_id,
        )
        for row_idx, output in enumerate(outputs):
            padded[row_idx, : output.numel()] = output
        return padded

    def _normalize_stop_sequences(
        self,
        stop_sequences: StopSequenceInput,
    ) -> tuple[tuple[int, ...], ...]:
        if stop_sequences is None:
            return ()
        normalized = []
        for sequence in stop_sequences:
            values = tuple(
                self._validate_token_id(token, "stop_sequences") for token in sequence
            )
            if not values:
                raise ValueError("stop_sequences cannot contain empty sequences")
            normalized.append(values)
        return tuple(normalized)

    def _validate_token_id(self, token: int, field: str) -> int:
        if not isinstance(token, int) or isinstance(token, bool):
            raise ValueError(f"{field} must contain integer token IDs")
        if token < 0 or token >= self.config.vocab_size:
            raise ValueError(f"{field} contain token IDs outside [0, vocab_size)")
        return token

    def _normalize_stop_token_ids(
        self,
        stop_token_ids: list[int] | tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        if stop_token_ids is None:
            return ()
        return tuple(
            self._validate_token_id(token, "stop_token_ids") for token in stop_token_ids
        )

    def _normalize_eos_token_id(self, eos_token_id: int | None) -> int | None:
        if eos_token_id is None:
            return self.config.eos_token_id
        return self._validate_token_id(eos_token_id, "eos_token_id")

    def _matches_stop_sequences(
        self,
        generated: torch.Tensor,
        stop_sequences: tuple[tuple[int, ...], ...],
    ) -> bool:
        if not stop_sequences:
            return False
        batch_matches = []
        for row in generated:
            row_match = False
            for sequence in stop_sequences:
                if len(sequence) > row.numel():
                    continue
                suffix = torch.tensor(sequence, device=row.device, dtype=row.dtype)
                if torch.equal(row[-len(sequence) :], suffix):
                    row_match = True
                    break
            batch_matches.append(row_match)
        return all(batch_matches)

    def _select_next_token(
        self,
        logits: torch.Tensor,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        if not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("temperature must be finite and > 0")
        if top_k is not None:
            if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
                raise ValueError("top_k must be a positive integer when provided")
        if top_p is not None and (
            not math.isfinite(float(top_p)) or not 0.0 < top_p <= 1.0
        ):
            raise ValueError(
                "top_p must be finite and satisfy 0 < top_p <= 1 when provided"
            )
        logits = self._mask_ungeneratable_logits(logits)
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)
        scaled = logits / temperature
        if top_k is not None and top_k < scaled.size(-1):
            values, _ = torch.topk(scaled, top_k, dim=-1)
            cutoff = values[:, -1].unsqueeze(-1)
            scaled = scaled.masked_fill(scaled < cutoff, torch.finfo(scaled.dtype).min)
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            remove = sorted_probs.cumsum(dim=-1) > top_p
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(
                remove, torch.finfo(scaled.dtype).min
            )
            scaled = torch.full_like(scaled, torch.finfo(scaled.dtype).min)
            scaled.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _mask_ungeneratable_logits(self, logits: torch.Tensor) -> torch.Tensor:
        masked = logits.clone()
        masked[..., self.config.pad_token_id] = torch.finfo(masked.dtype).min
        return masked

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        repetition_penalty: float,
    ) -> torch.Tensor:
        if not math.isfinite(float(repetition_penalty)) or repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be finite and > 0")
        if repetition_penalty == 1.0:
            return logits
        adjusted = logits.clone()
        for batch_idx in range(generated.size(0)):
            token_ids = generated[batch_idx].unique()
            token_logits = adjusted[batch_idx, token_ids]
            adjusted[batch_idx, token_ids] = torch.where(
                token_logits < 0,
                token_logits * repetition_penalty,
                token_logits / repetition_penalty,
            )
        return adjusted

    def _should_stop(
        self,
        next_token: torch.Tensor,
        eos_token_id: int | None,
        stop_token_ids: list[int] | tuple[int, ...] | None,
        *,
        generated: torch.Tensor | None = None,
        stop_sequences: tuple[tuple[int, ...], ...] = (),
    ) -> bool:
        stop_ids = []
        if eos_token_id is not None:
            stop_ids.append(eos_token_id)
        if stop_token_ids is not None:
            stop_ids.extend(stop_token_ids)
        if not stop_ids:
            if generated is None:
                return False
            return self._matches_stop_sequences(generated, stop_sequences)
        stop = torch.tensor(stop_ids, device=next_token.device, dtype=next_token.dtype)
        single_token_stop = torch.isin(next_token, stop).all().item()
        if single_token_stop:
            return True
        if generated is None:
            return False
        return self._matches_stop_sequences(generated, stop_sequences)

    def _apply_stop_postprocess(
        self,
        generated: torch.Tensor,
        *,
        prompt_length: int,
        eos_token_id: int | None,
        stop_token_ids: list[int] | tuple[int, ...] | None,
        stop_sequences: tuple[tuple[int, ...], ...],
    ) -> torch.Tensor:
        stop_ids = []
        if eos_token_id is not None:
            stop_ids.append(eos_token_id)
        if stop_token_ids is not None:
            stop_ids.extend(int(token) for token in stop_token_ids)
        if not stop_ids and not stop_sequences:
            return generated

        processed = generated.clone()
        row_ends: list[int] = []
        for row in processed:
            stop_end: int | None = None
            for idx in range(prompt_length, row.numel()):
                token = int(row[idx].item())
                if token in stop_ids:
                    stop_end = idx + 1
                    break
                for sequence in stop_sequences:
                    seq_len = len(sequence)
                    start = idx + 1 - seq_len
                    if start < 0:
                        continue
                    suffix = torch.tensor(sequence, device=row.device, dtype=row.dtype)
                    if torch.equal(row[start : idx + 1], suffix):
                        stop_end = idx + 1
                        break
                if stop_end is not None:
                    break
            if stop_end is None:
                row_ends.append(row.numel())
                continue
            if stop_end < row.numel():
                row[stop_end:] = self.config.pad_token_id
            row_ends.append(stop_end)
        trim_to = max(row_ends, default=processed.size(1))
        return processed[:, :trim_to]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: list[AttentionCache | None] | None = None,
        past_ssm_states: list[torch.Tensor | None] | None = None,
        use_cache: bool = False,
        diffusion_timestep: int | torch.Tensor | None = None,
        ttt_state: TTTState | None = None,
        output_hidden_states: bool = False,
    ) -> AlexandrosCausalLMOutput:
        if ttt_state is not None and diffusion_timestep is not None:
            raise ValueError(
                "TTT state is only supported for non-diffusion causal forwards"
            )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            past_ssm_states=past_ssm_states,
            use_cache=use_cache,
            diffusion_timestep=diffusion_timestep,
            ttt_state=ttt_state,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if labels.dtype != torch.long:
                raise ValueError(
                    f"labels must be torch.long token IDs or {IGNORE_INDEX}"
                )
            if labels.size(1) < 2:
                raise ValueError("causal LM labels require sequence length >= 2")
            if not labels[:, 1:].ne(IGNORE_INDEX).any():
                raise ValueError(
                    "causal LM labels must include at least one non-ignored target"
                )
            invalid_labels = labels.ne(IGNORE_INDEX) & (
                labels.lt(0) | labels.ge(self.config.vocab_size)
            )
            if invalid_labels.any():
                raise ValueError(
                    f"labels contain token IDs outside [0, vocab_size) or {IGNORE_INDEX}"
                )
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return AlexandrosCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            past_ssm_states=outputs.past_ssm_states,
            hidden_states=outputs.hidden_states,
            moe_stats=outputs.moe_stats,
        )

    @torch.no_grad()
    def generate_cached(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        eos_token_id: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        stop_token_ids: list[int] | tuple[int, ...] | None = None,
        stop_sequences: StopSequenceInput = None,
    ) -> torch.Tensor:
        self._validate_max_new_tokens(max_new_tokens)
        if max_new_tokens == 0:
            return input_ids
        if self._has_padded_generation_prompts(input_ids):
            return self._generate_variable_length_batch(
                input_ids,
                max_new_tokens=max_new_tokens,
                generate_kwargs={
                    "eos_token_id": eos_token_id,
                    "use_cache": True,
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "stop_token_ids": stop_token_ids,
                    "stop_sequences": stop_sequences,
                },
            )
        self._validate_generation_length(input_ids, max_new_tokens)
        eos_token_id = self._normalize_eos_token_id(eos_token_id)
        normalized_stop_token_ids = self._normalize_stop_token_ids(stop_token_ids)
        normalized_stop_sequences = self._normalize_stop_sequences(stop_sequences)
        generated = input_ids
        attention_mask = generated.ne(self.config.pad_token_id)
        output = self(generated, attention_mask=attention_mask, use_cache=True)
        past_key_values = output.past_key_values
        past_ssm_states = output.past_ssm_states
        for step in range(max_new_tokens):
            logits = output.logits[:, -1, :]
            logits = self._apply_repetition_penalty(
                logits, generated, repetition_penalty
            )
            next_token = self._select_next_token(
                logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated = torch.cat([generated, next_token], dim=1)
            if self._should_stop(
                next_token,
                eos_token_id,
                normalized_stop_token_ids,
                generated=generated,
                stop_sequences=normalized_stop_sequences,
            ):
                break
            if step == max_new_tokens - 1:
                break
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, dtype=torch.bool)],
                dim=1,
            )
            output = self(
                next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                past_ssm_states=past_ssm_states,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            past_ssm_states = output.past_ssm_states
        return generated

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        eos_token_id: int | None = None,
        use_cache: bool = False,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        stop_token_ids: list[int] | tuple[int, ...] | None = None,
        stop_sequences: StopSequenceInput = None,
    ) -> torch.Tensor:
        self._validate_max_new_tokens(max_new_tokens)
        if max_new_tokens == 0:
            return input_ids
        if self._has_padded_generation_prompts(input_ids):
            return self._generate_variable_length_batch(
                input_ids,
                max_new_tokens=max_new_tokens,
                generate_kwargs={
                    "eos_token_id": eos_token_id,
                    "use_cache": use_cache,
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "stop_token_ids": stop_token_ids,
                    "stop_sequences": stop_sequences,
                },
            )
        self._validate_generation_length(input_ids, max_new_tokens)
        if use_cache:
            return self.generate_cached(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_token_ids=stop_token_ids,
                stop_sequences=stop_sequences,
            )
        eos_token_id = self._normalize_eos_token_id(eos_token_id)
        normalized_stop_token_ids = self._normalize_stop_token_ids(stop_token_ids)
        normalized_stop_sequences = self._normalize_stop_sequences(stop_sequences)
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self(generated).logits[:, -1, :]
            logits = self._apply_repetition_penalty(
                logits, generated, repetition_penalty
            )
            next_token = self._select_next_token(
                logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated = torch.cat([generated, next_token], dim=1)
            if self._should_stop(
                next_token,
                eos_token_id,
                normalized_stop_token_ids,
                generated=generated,
                stop_sequences=normalized_stop_sequences,
            ):
                break
        return generated


class AlexandrosForDiffusionLM(AlexandrosForCausalLM):
    def __init__(self, config: AlexandrosConfig) -> None:
        super().__init__(config)
        self.scheduler = MaskedDiffusionScheduler(config)
        self.latent_vae = LatentThoughtVAE(config)
        self.latent_reasoner = LatentDiffusionReasoner(config)

    def _masked_diffusion_loss_from_noisy(
        self,
        *,
        input_ids: torch.Tensor,
        timestep: int | torch.Tensor,
        noisy: torch.Tensor,
        mask: torch.Tensor,
    ) -> AlexandrosCausalLMOutput:
        masked_count = mask.sum()
        if masked_count.item() == 0:
            raise ValueError(
                "diffusion_loss requires at least one non-pad token to mask"
            )
        labels = input_ids.masked_fill(~mask, IGNORE_INDEX)
        output = self(noisy, diffusion_timestep=timestep)
        logits = output.logits
        token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view_as(input_ids)
        weights = self.scheduler.loss_weight(timestep).to(
            device=input_ids.device, dtype=token_loss.dtype
        )
        while weights.ndim < token_loss.ndim:
            weights = weights.unsqueeze(-1)
        loss_sum = (token_loss * weights * mask.to(dtype=token_loss.dtype)).sum()
        output.loss = loss_sum / masked_count.to(dtype=loss_sum.dtype)
        return output

    def _rao_blackwellized_diffusion_loss_from_noisy(
        self,
        *,
        input_ids: torch.Tensor,
        timestep: int | torch.Tensor,
        noisy: torch.Tensor,
    ) -> AlexandrosCausalLMOutput:
        target_mask = input_ids.ne(self.config.pad_token_id)
        target_count = int(target_mask.sum().item())
        if target_count == 0:
            raise ValueError(
                "diffusion_loss requires at least one non-pad token to mask"
            )
        rows, cols = target_mask.nonzero(as_tuple=True)
        timestep_grid = self.scheduler.timestep_grid(timestep, input_ids).to(
            device=input_ids.device
        )
        target_timesteps = timestep_grid[rows, cols]
        target_labels = input_ids[rows, cols].to(dtype=torch.long)
        chunk_size = self.config.diffusion_rb_chunk_size or target_count
        output = self(noisy, diffusion_timestep=timestep)
        loss_sum = output.logits.new_tensor(0.0)

        for start in range(0, target_count, chunk_size):
            end = min(start + chunk_size, target_count)
            chunk_rows = rows[start:end]
            chunk_cols = cols[start:end]
            variants = noisy[chunk_rows].clone()
            local = torch.arange(end - start, device=input_ids.device)
            variants[local, chunk_cols] = self.config.mask_token_id
            chunk_timesteps = target_timesteps[start:end]
            chunk_labels = target_labels[start:end]
            chunk_output = self(variants, diffusion_timestep=chunk_timesteps)
            target_logits = chunk_output.logits[local, chunk_cols]
            token_loss = F.cross_entropy(
                target_logits,
                chunk_labels,
                reduction="none",
            )
            weights = self.scheduler.loss_weight(chunk_timesteps).to(
                device=input_ids.device,
                dtype=token_loss.dtype,
            )
            loss_sum = loss_sum + (token_loss * weights).sum().to(dtype=loss_sum.dtype)
            output.moe_stats = chunk_output.moe_stats
        output.loss = loss_sum / output.logits.new_tensor(float(target_count))
        return output

    def diffusion_loss(
        self,
        input_ids: torch.Tensor,
        timestep: int | torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> AlexandrosCausalLMOutput:
        if timestep is None:
            timestep = torch.randint(
                0,
                self.config.diffusion_steps,
                (input_ids.size(0),),
                device=input_ids.device,
                generator=generator,
            )
        noisy, mask = self.scheduler.add_noise(input_ids, timestep, generator=generator)
        if self.config.diffusion_objective == "rao_blackwellized":
            return self._rao_blackwellized_diffusion_loss_from_noisy(
                input_ids=input_ids,
                timestep=timestep,
                noisy=noisy,
            )
        return self._masked_diffusion_loss_from_noisy(
            input_ids=input_ids,
            timestep=timestep,
            noisy=noisy,
            mask=mask,
        )

    @torch.no_grad()
    def generate_block_diffusion(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        steps: int | None = None,
        block_size: int | None = None,
        confidence_schedule: str = "median",
        remask_low_confidence: bool = False,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
        stop_token_ids: list[int] | tuple[int, ...] | None = None,
        stop_sequences: StopSequenceInput = None,
    ) -> torch.Tensor:
        self._validate_max_new_tokens(max_new_tokens)
        if max_new_tokens == 0:
            return input_ids
        self._validate_generation_length(input_ids, max_new_tokens)
        eos_token_id = self._normalize_eos_token_id(eos_token_id)
        normalized_stop_token_ids = self._normalize_stop_token_ids(stop_token_ids)
        normalized_stop_sequences = self._normalize_stop_sequences(stop_sequences)
        steps = (
            self.config.diffusion_steps
            if steps is None
            else self._validate_generation_steps(steps, "steps")
        )
        block_size = self._validate_optional_positive_int(block_size, "block_size")
        if confidence_schedule not in {"median", "linear", "all"}:
            raise ValueError("confidence_schedule must be one of: median, linear, all")
        if not isinstance(remask_low_confidence, bool):
            raise ValueError("remask_low_confidence must be a boolean")
        generated = input_ids
        total_prompt_length = input_ids.size(1)
        remaining = max_new_tokens
        while remaining > 0:
            current_block_size = (
                remaining if block_size is None else min(block_size, remaining)
            )
            block = torch.full(
                (input_ids.size(0), current_block_size),
                self.config.mask_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            generated = torch.cat([generated, block], dim=1)
            block_start = generated.size(1) - current_block_size
            generated = self._denoise_block_diffusion_block(
                generated,
                block_start=block_start,
                steps=steps,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                confidence_schedule=confidence_schedule,
                remask_low_confidence=remask_low_confidence,
            )
            remaining -= current_block_size
        return self._apply_stop_postprocess(
            generated,
            prompt_length=total_prompt_length,
            eos_token_id=eos_token_id,
            stop_token_ids=normalized_stop_token_ids,
            stop_sequences=normalized_stop_sequences,
        )

    def _block_diffusion_threshold(
        self,
        confidence: torch.Tensor,
        *,
        step: int,
        steps: int,
        confidence_schedule: str,
    ) -> torch.Tensor | float:
        if confidence.numel() == 0:
            return 1.0
        if confidence_schedule == "all":
            return torch.finfo(confidence.dtype).min
        if confidence_schedule == "median":
            return torch.quantile(confidence, 0.5)
        denoise_index = steps - step - 1
        commit_fraction = min(1.0, max(0.0, float(denoise_index + 1) / float(steps)))
        quantile = max(0.0, min(1.0, 1.0 - commit_fraction))
        return torch.quantile(confidence, quantile)

    def _denoise_block_diffusion_block(
        self,
        generated: torch.Tensor,
        *,
        block_start: int,
        steps: int,
        do_sample: bool,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        confidence_schedule: str,
        remask_low_confidence: bool,
    ) -> torch.Tensor:
        block_positions = torch.zeros_like(generated, dtype=torch.bool)
        block_positions[:, block_start:] = True
        for step in reversed(range(steps)):
            logits = self(generated, diffusion_timestep=step).logits.clone()
            logits = self._mask_ungeneratable_logits(logits)
            logits[..., self.config.mask_token_id] = torch.finfo(logits.dtype).min
            masked = (
                block_positions
                if remask_low_confidence
                else (block_positions & generated.eq(self.config.mask_token_id))
            )
            flat_logits = logits.reshape(-1, logits.size(-1))
            tokens = self._select_next_token(
                flat_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ).view_as(generated)
            probs = (logits / temperature).softmax(dim=-1)
            confidence = probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
            if step == 0:
                commit = masked
            else:
                threshold = self._block_diffusion_threshold(
                    confidence[masked],
                    step=step,
                    steps=steps,
                    confidence_schedule=confidence_schedule,
                )
                commit = masked & confidence.ge(threshold)
            if remask_low_confidence and step > 0:
                block_tokens = torch.where(
                    commit,
                    tokens,
                    torch.full_like(generated, self.config.mask_token_id),
                )
                generated = torch.where(block_positions, block_tokens, generated)
            else:
                generated = torch.where(commit, tokens, generated)
            if not generated[:, block_start:].eq(self.config.mask_token_id).any():
                break
        return generated

    @torch.no_grad()
    def generate_latent_reasoning(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        latent_steps: int | None = None,
    ) -> torch.Tensor:
        self._validate_max_new_tokens(max_new_tokens)
        if max_new_tokens == 0:
            return input_ids
        self._validate_generation_length(input_ids, max_new_tokens)
        if latent_steps is not None:
            self._validate_generation_steps(latent_steps, "latent_steps")
        outputs = self.model(input_ids)
        vae = self.latent_vae(outputs.last_hidden_state)
        refined = self.latent_reasoner(vae.latents, steps=latent_steps)
        latent_hidden = self.latent_reasoner.decode_to_hidden(refined).mean(
            dim=1, keepdim=True
        )
        logits = self.lm_head(outputs.last_hidden_state[:, -1:, :] + latent_hidden)
        logits = self._mask_ungeneratable_logits(logits)
        next_token = logits.argmax(dim=-1)
        generated = torch.cat([input_ids, next_token], dim=1)
        if max_new_tokens <= 1:
            return generated
        return self.generate(generated, max_new_tokens=max_new_tokens - 1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        mode: GenerationMode | str = GenerationMode.AUTOREGRESSIVE,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._validate_max_new_tokens(max_new_tokens)
        if max_new_tokens == 0:
            return input_ids
        mode = GenerationMode(mode)
        if mode != GenerationMode.AUTOREGRESSIVE and "use_cache" in kwargs:
            if kwargs.pop("use_cache"):
                raise ValueError(
                    "use_cache is only supported for autoregressive generation"
                )
        if self._has_padded_generation_prompts(input_ids):
            return self._generate_variable_length_batch(
                input_ids,
                max_new_tokens=max_new_tokens,
                generate_kwargs={"mode": mode, **kwargs},
            )
        if mode == GenerationMode.BLOCK_DIFFUSION:
            return self.generate_block_diffusion(
                input_ids, max_new_tokens=max_new_tokens, **kwargs
            )
        if mode == GenerationMode.LATENT_REASONING:
            return self.generate_latent_reasoning(
                input_ids, max_new_tokens=max_new_tokens, **kwargs
            )
        if mode == GenerationMode.HYBRID:
            latent_steps = kwargs.pop("latent_steps", None)
            diffusion_steps = kwargs.pop("steps", None)
            block_size = kwargs.pop("block_size", None)
            confidence_schedule = kwargs.pop("confidence_schedule", "median")
            remask_low_confidence = kwargs.pop("remask_low_confidence", False)
            if kwargs:
                unknown = ", ".join(sorted(kwargs))
                raise TypeError(f"unsupported hybrid generation arguments: {unknown}")
            primed = self.generate_latent_reasoning(
                input_ids,
                max_new_tokens=1,
                latent_steps=latent_steps,
            )
            return self.generate_block_diffusion(
                primed,
                max_new_tokens=max(0, max_new_tokens - 1),
                steps=diffusion_steps,
                block_size=block_size,
                confidence_schedule=confidence_schedule,
                remask_low_confidence=remask_low_confidence,
            )
        return super().generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)
