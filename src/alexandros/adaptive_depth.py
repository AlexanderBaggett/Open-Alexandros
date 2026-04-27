from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from alexandros.bitlinear import make_linear
from alexandros.configuration_alexandros import AlexandrosConfig
from alexandros.initialization import initialize_linear, residual_projection_std


@dataclass(frozen=True)
class AdaptiveDepthStats:
    halting_sum: torch.Tensor
    remainder: torch.Tensor
    n_updates: torch.Tensor
    ponder_cost: torch.Tensor
    average_loop_count: float


class DepthWiseLoRAAdapter(nn.Module):
    """Low-rank residual adapter owned by one adaptive-depth loop."""

    def __init__(self, config: AlexandrosConfig, rank: int) -> None:
        super().__init__()
        self.rank = rank
        linear_kwargs = {
            "variant": config.variant,
            "activation_bits": config.bitnet_activation_bits,
        }
        self.down = make_linear(config.hidden_size, rank, bias=False, **linear_kwargs)
        self.up = make_linear(rank, config.hidden_size, bias=False, **linear_kwargs)
        self.reset_parameters(config)

    def reset_parameters(self, config: AlexandrosConfig) -> None:
        initialize_linear(self.down)
        initialize_linear(self.up, std=residual_projection_std(config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(torch.tanh(self.down(x)))


class AdaptiveDepthController(nn.Module):
    """OpenMythos-inspired ACT loop for bounded latent refinement."""

    def __init__(self, config: AlexandrosConfig) -> None:
        super().__init__()
        self.config = config
        linear_kwargs = {
            "variant": config.variant,
            "activation_bits": config.bitnet_activation_bits,
        }
        self.transition = make_linear(
            config.hidden_size, config.hidden_size, **linear_kwargs
        )
        self.halt_proj = make_linear(config.hidden_size, 1, **linear_kwargs)
        self.depth_lora_adapters = nn.ModuleList(
            [
                DepthWiseLoRAAdapter(config, config.depth_lora_rank_for_loop(idx))
                for idx in range(config.max_depth_iters)
            ]
        )
        self.last_stats: AdaptiveDepthStats | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_linear(self.transition, std=residual_projection_std(self.config))
        initialize_linear(self.halt_proj)
        for adapter in self.depth_lora_adapters:
            adapter.reset_parameters(self.config)

    def _loop_embedding(self, x: torch.Tensor, loop_idx: int) -> torch.Tensor:
        dim = min(x.size(-1), 32)
        pos = torch.arange(dim, device=x.device, dtype=x.dtype)
        freqs = torch.exp(-math.log(10000.0) * pos / max(dim, 1))
        signal = torch.sin((loop_idx + 1) * freqs)
        padded = torch.zeros_like(x)
        padded[..., :dim] = signal
        return x + padded

    def _validate_hidden_states(self, hidden_states: torch.Tensor) -> None:
        if not torch.is_tensor(hidden_states) or not hidden_states.is_floating_point():
            raise ValueError("hidden_states must be a floating-point tensor")
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, hidden_size]"
            )
        if hidden_states.size(0) == 0:
            raise ValueError("hidden_states batch size must be > 0")
        if hidden_states.size(1) == 0:
            raise ValueError("hidden_states sequence length must be > 0")
        if hidden_states.size(-1) != self.config.hidden_size:
            raise ValueError("hidden_states last dimension must match hidden_size")
        if not torch.isfinite(hidden_states).all():
            raise ValueError("hidden_states must contain only finite values")

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_hidden_states(hidden_states)
        running = hidden_states
        weighted = torch.zeros_like(hidden_states)
        halting_sum = torch.zeros(
            hidden_states.shape[:-1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        remainder = torch.zeros_like(halting_sum)
        n_updates = torch.zeros_like(halting_sum)
        threshold = float(self.config.act_threshold)
        for loop_idx in range(self.config.max_depth_iters):
            candidate = self._loop_embedding(running, loop_idx)
            update = torch.tanh(self.transition(candidate))
            update = update + self.depth_lora_adapters[loop_idx](update)
            p = torch.sigmoid(self.halt_proj(update)).squeeze(-1)
            still_running_bool = halting_sum < threshold
            still_running = still_running_bool.to(hidden_states.dtype)
            remaining_to_threshold = (threshold - halting_sum).clamp_min(0.0)
            is_last = loop_idx == self.config.max_depth_iters - 1
            new_halted = still_running_bool & ((halting_sum + p) >= threshold)
            if is_last:
                new_halted = still_running_bool
            continuing = still_running_bool & ~new_halted
            weight = torch.where(
                new_halted,
                remaining_to_threshold,
                torch.where(continuing, p, torch.zeros_like(p)),
            )
            step_remainder = torch.where(
                new_halted, remaining_to_threshold, torch.zeros_like(p)
            )
            remainder = torch.where(new_halted, step_remainder, remainder)
            weighted = weighted + update * weight.unsqueeze(-1)
            halting_sum = (halting_sum + weight).clamp(max=threshold)
            n_updates = n_updates + still_running
            running = torch.where(still_running_bool.unsqueeze(-1), update, running)
        fallback = hidden_states * (1.0 - halting_sum).unsqueeze(-1)
        ponder_cost = n_updates.mean() * float(self.config.act_ponder_cost)
        self.last_stats = AdaptiveDepthStats(
            halting_sum=halting_sum.detach(),
            remainder=remainder.detach(),
            n_updates=n_updates.detach(),
            ponder_cost=ponder_cost.detach(),
            average_loop_count=float(n_updates.detach().mean().item()),
        )
        return weighted + fallback, halting_sum
