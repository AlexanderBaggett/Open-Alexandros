from __future__ import annotations

import torch

import _bootstrap  # noqa: F401
from alexandros.bitlinear import BitLinear

layer = BitLinear(8, 4)
ternary = layer.ternary_weight()
scale = layer.weight.abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
unique_levels = torch.round(ternary / scale).unique().tolist()
print({"levels": sorted(float(level) for level in unique_levels)})
