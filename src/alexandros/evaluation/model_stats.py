from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from alexandros.configuration_alexandros import AlexandrosConfig
from alexandros.moe import MoEFeedForward


@dataclass(frozen=True)
class ParameterReport:
    total_parameters: int
    trainable_parameters: int
    routed_expert_parameters: int
    active_routed_expert_parameters: int

    @property
    def active_parameters_per_token(self) -> int:
        return (
            self.total_parameters
            - self.routed_expert_parameters
            + self.active_routed_expert_parameters
        )


@dataclass(frozen=True)
class CacheMemoryReport:
    attention_layers: int
    standard_mha_elements_per_token: int
    mla_elements_per_token: int
    mla_compressed_elements_per_token: int
    mla_rope_elements_per_token: int
    standard_kv_bits: int
    mla_kv_bits: int
    turboquant_mla_bits: int

    @property
    def mla_compression_ratio(self) -> float:
        return self.standard_kv_bits / max(self.mla_kv_bits, 1)

    @property
    def turboquant_compression_ratio(self) -> float:
        return self.standard_kv_bits / max(self.turboquant_mla_bits, 1)


@dataclass(frozen=True)
class FlopReport:
    batch_size: int
    sequence_length: int
    attention_layers: int
    linear_mixer_layers: int
    prefill_flops: int
    decode_token_flops: int

    @property
    def prefill_tflops(self) -> float:
        return self.prefill_flops / 1e12


@dataclass(frozen=True)
class MoEReport:
    layers_with_stats: int
    mean_load_entropy: float
    min_expert_load: float
    max_expert_load: float
    timestep_tracked_selections: int
    timestep_load_entropy: tuple[float, ...]
    noisy_step_load_entropy: float
    polish_step_load_entropy: float
    noisy_timestep_tracked_selections: int
    polish_timestep_tracked_selections: int


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        return sum(param.numel() for param in parameters if param.requires_grad)
    return sum(param.numel() for param in parameters)


def summarize_parameters(model: nn.Module) -> ParameterReport:
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    routed_total = 0
    active_routed = 0
    for module in model.modules():
        if not isinstance(module, MoEFeedForward):
            continue
        expert_sizes = [count_parameters(expert) for expert in module.routed_experts]
        routed_total += sum(expert_sizes)
        active_routed += sum(expert_sizes[: module.top_k])
    return ParameterReport(
        total_parameters=total,
        trainable_parameters=trainable,
        routed_expert_parameters=routed_total,
        active_routed_expert_parameters=active_routed,
    )


def count_attention_layers(config: AlexandrosConfig) -> int:
    return len(config.attention_layers())


def summarize_moe_stats(model: nn.Module) -> MoEReport:
    loads = []
    timestep_load: torch.Tensor | None = None
    timestep_count: torch.Tensor | None = None
    diffusion_steps = 0
    for module in model.modules():
        if not isinstance(module, MoEFeedForward):
            continue
        if module.last_stats is not None:
            loads.append(module.last_stats.expert_load.float())
        module_load = module.timestep_expert_load.detach().float()
        module_count = module.timestep_expert_count.detach().float()
        diffusion_steps = max(diffusion_steps, module.config.diffusion_steps)
        if timestep_load is None or timestep_count is None:
            timestep_load = module_load.clone()
            timestep_count = module_count.clone()
        else:
            max_steps = max(timestep_load.size(0), module_load.size(0))
            max_experts = max(timestep_load.size(1), module_load.size(1))
            expanded_load = timestep_load.new_zeros(max_steps, max_experts)
            expanded_count = timestep_count.new_zeros(max_steps)
            expanded_load[: timestep_load.size(0), : timestep_load.size(1)] = (
                timestep_load
            )
            expanded_count[: timestep_count.size(0)] = timestep_count
            expanded_load[: module_load.size(0), : module_load.size(1)] += (
                module_load.to(expanded_load.device)
            )
            expanded_count[: module_count.size(0)] += module_count.to(
                expanded_count.device
            )
            timestep_load = expanded_load
            timestep_count = expanded_count
    timestep_diagnostics = _summarize_timestep_load(
        timestep_load,
        timestep_count,
        diffusion_steps=diffusion_steps,
    )
    if not loads:
        return MoEReport(
            layers_with_stats=0,
            mean_load_entropy=0.0,
            min_expert_load=0.0,
            max_expert_load=0.0,
            **timestep_diagnostics,
        )
    stacked = torch.stack(loads)
    entropy = -(stacked.clamp_min(1e-12) * stacked.clamp_min(1e-12).log()).sum(dim=-1)
    return MoEReport(
        layers_with_stats=len(loads),
        mean_load_entropy=float(entropy.mean().item()),
        min_expert_load=float(stacked.min().item()),
        max_expert_load=float(stacked.max().item()),
        **timestep_diagnostics,
    )


def _summarize_timestep_load(
    timestep_load: torch.Tensor | None,
    timestep_count: torch.Tensor | None,
    *,
    diffusion_steps: int,
) -> dict[str, object]:
    if timestep_load is None or timestep_count is None:
        return {
            "timestep_tracked_selections": 0,
            "timestep_load_entropy": (),
            "noisy_step_load_entropy": 0.0,
            "polish_step_load_entropy": 0.0,
            "noisy_timestep_tracked_selections": 0,
            "polish_timestep_tracked_selections": 0,
        }

    totals = timestep_load.sum(dim=-1)
    probs = timestep_load / totals.clamp_min(1.0).unsqueeze(-1)
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = torch.where(totals > 0, entropy, torch.zeros_like(entropy))
    tracked = timestep_count > 0
    if diffusion_steps > 0:
        step_ids = torch.arange(timestep_count.size(0), device=timestep_count.device)
        valid_steps = step_ids < diffusion_steps
        mask_probability = (step_ids.float() + 1.0) / float(diffusion_steps)
        noisy_mask = tracked & valid_steps & mask_probability.gt(0.5)
        polish_mask = tracked & valid_steps & mask_probability.le(0.5)
    else:
        noisy_mask = torch.zeros_like(tracked)
        polish_mask = torch.zeros_like(tracked)

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        if not mask.any():
            return 0.0
        return float(values[mask].mean().item())

    reported_entropy = entropy[:diffusion_steps] if diffusion_steps > 0 else entropy
    return {
        "timestep_tracked_selections": int(timestep_count.sum().item()),
        "timestep_load_entropy": tuple(
            float(value) for value in reported_entropy.cpu().tolist()
        ),
        "noisy_step_load_entropy": masked_mean(entropy, noisy_mask),
        "polish_step_load_entropy": masked_mean(entropy, polish_mask),
        "noisy_timestep_tracked_selections": int(
            timestep_count[noisy_mask].sum().item()
        ),
        "polish_timestep_tracked_selections": int(
            timestep_count[polish_mask].sum().item()
        ),
    }


def estimate_cache_memory(
    config: AlexandrosConfig,
    *,
    batch_size: int = 1,
    sequence_length: int | None = None,
    dtype_bits: int = 16,
) -> CacheMemoryReport:
    sequence_length = (
        config.max_position_embeddings if sequence_length is None else sequence_length
    )
    _validate_positive_int(batch_size, "batch_size")
    _validate_positive_int(sequence_length, "sequence_length")
    _validate_positive_int(dtype_bits, "dtype_bits")
    attention_layers = count_attention_layers(config)
    standard_elements_per_token = config.standard_mha_elements_per_token
    mla_compressed_elements_per_token = config.mla_d_c
    mla_rope_elements_per_token = config.mla_d_r
    mla_elements_per_token = config.mla_elements_per_token
    standard_per_layer = (
        batch_size * sequence_length * standard_elements_per_token * dtype_bits
    )
    mla_per_layer = batch_size * sequence_length * mla_elements_per_token * dtype_bits
    # TurboQuant compresses only c_kv. The optional k_rope cache remains
    # uncompressed until a positional-cache compressor is specified.
    turboquant_per_layer = (
        batch_size
        * sequence_length
        * mla_compressed_elements_per_token
        * config.turboquant_bits
        + batch_size * sequence_length * 16
        + batch_size * sequence_length * mla_rope_elements_per_token * dtype_bits
    )
    return CacheMemoryReport(
        attention_layers=attention_layers,
        standard_mha_elements_per_token=standard_elements_per_token,
        mla_elements_per_token=mla_elements_per_token,
        mla_compressed_elements_per_token=mla_compressed_elements_per_token,
        mla_rope_elements_per_token=mla_rope_elements_per_token,
        standard_kv_bits=standard_per_layer * attention_layers,
        mla_kv_bits=mla_per_layer * attention_layers,
        turboquant_mla_bits=turboquant_per_layer * attention_layers,
    )


def estimate_flops(
    config: AlexandrosConfig,
    *,
    batch_size: int = 1,
    sequence_length: int | None = None,
    include_lm_head: bool = True,
) -> FlopReport:
    """Return a conservative dense reference FLOP estimate.

    The estimate counts multiply-adds as two FLOPs. It respects
    `moe_sparse_dispatch` for routed expert work and remains a planning
    estimate, not a kernel benchmark.
    """

    sequence_length = (
        config.max_position_embeddings if sequence_length is None else sequence_length
    )
    _validate_positive_int(batch_size, "batch_size")
    _validate_positive_int(sequence_length, "sequence_length")
    attention_layers = count_attention_layers(config)
    linear_mixer_layers = config.num_hidden_layers - attention_layers
    h = config.hidden_size
    r = config.kv_lora_rank
    heads = config.num_attention_heads
    head_dim = config.head_dim
    k_nope_dim = heads * config.mla_d_nope
    k_rope_dim = config.mla_d_r
    t = sequence_length
    b = batch_size

    attention_projection_terms = (
        h * h  # q projection
        + h * r  # compressed KV down projection
        + r * k_nope_dim  # no-RoPE key reconstruction
        + h * k_rope_dim  # positional RoPE key projection
        + r * h  # value reconstruction
        + h * h  # output projection
    )
    attention_linear = 2 * b * t * attention_projection_terms
    attention_scores = 4 * b * heads * t * t * head_dim
    attention_total = attention_layers * (attention_linear + attention_scores)

    if config.linear_mixer_backend == "matrix_deltanet":
        mixer_linear = 2 * b * t * (4 * h * h + 2 * h * config.num_attention_heads)
        mixer_elementwise = (
            b
            * t
            * config.num_attention_heads
            * config.head_dim
            * (config.head_dim * 6 + 8)
        )
    elif config.linear_mixer_backend == "mamba2":
        mixer_linear = 2 * b * t * (6 * h * h)
        mixer_elementwise = b * t * h * 14
    else:
        mixer_linear = 2 * b * t * (6 * h * h)
        mixer_elementwise = b * t * h * 12
    mixer_total = linear_mixer_layers * (mixer_linear + mixer_elementwise)

    expert_hidden = config.moe_expert_hidden_size
    active_routed_experts = (
        config.moe_top_k if config.moe_sparse_dispatch else config.moe_num_experts
    )
    routed_expert = 2 * b * t * active_routed_experts * (3 * h * expert_hidden)
    shared_expert = (
        2
        * b
        * t
        * config.moe_num_shared_experts
        * (3 * h * expert_hidden * config.moe_top_k)
    )
    router = 2 * b * t * h * config.moe_num_experts
    moe_total = config.num_hidden_layers * (router + routed_expert + shared_expert)

    lm_head = 2 * b * t * h * config.vocab_size if include_lm_head else 0
    prefill = attention_total + mixer_total + moe_total + lm_head

    decode_attention_scores = 4 * b * heads * t * head_dim
    decode_attention = attention_layers * (
        2 * b * attention_projection_terms + decode_attention_scores
    )
    if config.linear_mixer_backend == "matrix_deltanet":
        decode_mixer_linear = 2 * b * (4 * h * h + 2 * h * config.num_attention_heads)
        decode_mixer_elementwise = (
            b * config.num_attention_heads * config.head_dim * (config.head_dim * 6 + 8)
        )
    else:
        decode_mixer_elementwise_per_hidden = (
            14 if config.linear_mixer_backend == "mamba2" else 12
        )
        decode_mixer_linear = 2 * b * (6 * h * h)
        decode_mixer_elementwise = b * h * decode_mixer_elementwise_per_hidden
    decode_mixer = linear_mixer_layers * (
        decode_mixer_linear + decode_mixer_elementwise
    )
    decode_moe = config.num_hidden_layers * (
        2 * b * h * config.moe_num_experts
        + 2 * b * active_routed_experts * (3 * h * expert_hidden)
        + 2
        * b
        * config.moe_num_shared_experts
        * (3 * h * expert_hidden * config.moe_top_k)
    )
    decode_lm_head = 2 * b * h * config.vocab_size if include_lm_head else 0

    return FlopReport(
        batch_size=b,
        sequence_length=t,
        attention_layers=attention_layers,
        linear_mixer_layers=linear_mixer_layers,
        prefill_flops=int(prefill),
        decode_token_flops=int(
            decode_attention + decode_mixer + decode_moe + decode_lm_head
        ),
    )


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
