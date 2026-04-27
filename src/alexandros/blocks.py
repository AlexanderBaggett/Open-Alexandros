from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from alexandros.bitlinear import make_linear
from alexandros.configuration_alexandros import AlexandrosConfig
from alexandros.initialization import (
    initialize_linear,
    initialize_norm,
    residual_projection_std,
)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_norm(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


class SwiGLUExpert(nn.Module):
    def __init__(
        self, config: AlexandrosConfig, hidden_size: int | None = None
    ) -> None:
        super().__init__()
        self.config = config
        hidden_size = hidden_size or config.moe_expert_hidden_size
        self.gate_proj = make_linear(
            config.hidden_size,
            hidden_size,
            bias=False,
            variant=config.variant,
            activation_bits=config.bitnet_activation_bits,
        )
        self.up_proj = make_linear(
            config.hidden_size,
            hidden_size,
            bias=False,
            variant=config.variant,
            activation_bits=config.bitnet_activation_bits,
        )
        self.down_proj = make_linear(
            hidden_size,
            config.hidden_size,
            bias=False,
            variant=config.variant,
            activation_bits=config.bitnet_activation_bits,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_linear(self.gate_proj)
        initialize_linear(self.up_proj)
        initialize_linear(self.down_proj, std=residual_projection_std(self.config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
