from __future__ import annotations

import math

import torch
from torch import nn

from alexandros.bitlinear import BitLinear
from alexandros.configuration_alexandros import AlexandrosConfig

DEFAULT_INITIALIZER_STD = 0.02


def residual_projection_std(config: AlexandrosConfig) -> float:
    return DEFAULT_INITIALIZER_STD / math.sqrt(2.0 * config.num_hidden_layers)


def initialize_linear(
    module: nn.Module, *, std: float = DEFAULT_INITIALIZER_STD
) -> None:
    if isinstance(module, BitLinear):
        weight = module.weight
        bias = module.bias
    elif isinstance(module, nn.Linear):
        weight = module.weight
        bias = module.bias
    else:
        raise TypeError("initialize_linear expects nn.Linear or BitLinear")
    with torch.no_grad():
        weight.normal_(mean=0.0, std=std)
        if bias is not None:
            bias.zero_()


def initialize_embedding(
    module: nn.Embedding,
    *,
    std: float = DEFAULT_INITIALIZER_STD,
    zero_index: int | None = None,
) -> None:
    with torch.no_grad():
        module.weight.normal_(mean=0.0, std=std)
        if zero_index is not None and 0 <= zero_index < module.num_embeddings:
            module.weight[zero_index].zero_()


def initialize_norm(module: nn.Module) -> None:
    with torch.no_grad():
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        if weight is not None:
            weight.fill_(1.0)
        if bias is not None:
            bias.zero_()
