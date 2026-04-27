from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

_CODE_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_activation_bits(activation_bits: int) -> None:
    if not isinstance(activation_bits, int) or isinstance(activation_bits, bool):
        raise ValueError("activation_bits must be an integer")
    if activation_bits < 2:
        raise ValueError("activation_bits must be >= 2")


def _validate_code_tensor(codes: torch.Tensor, *, name: str = "ternary codes") -> None:
    if not torch.is_tensor(codes):
        raise ValueError(f"{name} must be a tensor")
    if codes.dtype not in _CODE_DTYPES:
        raise ValueError(f"{name} must be an integer tensor")
    if codes.numel() and (codes.min().item() < 0 or codes.max().item() > 2):
        raise ValueError(f"{name} must use values 0, 1, or 2")


class BitLinear(nn.Module):
    """BitNet b1.58-style linear layer with straight-through ternary weights.

    The layer keeps a full-precision trainable shadow weight, but forward passes
    use ternary {-1, 0, +1} weights with absmean scaling and per-token absmax
    activation quantization. This is a reference QAT implementation, not a
    packed inference kernel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_bits: int = 8,
    ) -> None:
        super().__init__()
        _validate_positive_int("in_features", in_features)
        _validate_positive_int("out_features", out_features)
        if not isinstance(bias, bool):
            raise ValueError("bias must be a boolean")
        _validate_activation_bits(activation_bits)
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def ternary_weight(self) -> torch.Tensor:
        ternary, scale = ternary_codes_and_scales(self.weight)
        quantized = ternary * scale
        return self.weight + (quantized - self.weight).detach()

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        _validate_activation_bits(self.activation_bits)
        qmax = float(2 ** (self.activation_bits - 1) - 1)
        scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / qmax
        quantized = torch.round(x / scale).clamp(-qmax, qmax) * scale
        return x + (quantized - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.quantize_activation(x), self.ternary_weight(), self.bias)


def make_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    variant: str = "heavy",
    activation_bits: int = 8,
) -> nn.Module:
    if variant == "lite":
        return BitLinear(
            in_features,
            out_features,
            bias=bias,
            activation_bits=activation_bits,
        )
    return nn.Linear(in_features, out_features, bias=bias)


def ternary_codes_and_scales(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
    normalized = weight / scale
    ternary = torch.where(
        normalized > 0.5,
        torch.ones_like(normalized),
        torch.where(
            normalized < -0.5,
            -torch.ones_like(normalized),
            torch.zeros_like(normalized),
        ),
    )
    return ternary, scale


def ternary_to_codes(ternary: torch.Tensor) -> torch.Tensor:
    return torch.where(
        ternary > 0,
        torch.ones_like(ternary, dtype=torch.uint8),
        torch.where(
            ternary < 0,
            torch.full_like(ternary, 2, dtype=torch.uint8),
            torch.zeros_like(ternary, dtype=torch.uint8),
        ),
    )


def codes_to_ternary(codes: torch.Tensor) -> torch.Tensor:
    _validate_code_tensor(codes)
    signed = codes.to(torch.int16)
    return torch.where(
        signed.eq(1),
        torch.ones_like(signed),
        torch.where(signed.eq(2), -torch.ones_like(signed), torch.zeros_like(signed)),
    ).to(torch.int8)


def pack_ternary_codes(codes: torch.Tensor) -> tuple[torch.Tensor, int]:
    _validate_code_tensor(codes)
    flat = codes.reshape(-1).to(dtype=torch.uint8, device="cpu")
    if flat.numel() == 0:
        return torch.empty(0, dtype=torch.uint8), 0
    padding = (-flat.numel()) % 4
    if padding:
        flat = torch.cat([flat, torch.zeros(padding, dtype=torch.uint8)])
    grouped = flat.view(-1, 4)
    packed = (
        grouped[:, 0]
        | (grouped[:, 1] << 2)
        | (grouped[:, 2] << 4)
        | (grouped[:, 3] << 6)
    )
    return packed.contiguous(), padding


def unpack_ternary_codes(
    packed: torch.Tensor,
    *,
    shape: tuple[int, int],
    padding: int = 0,
) -> torch.Tensor:
    if not isinstance(padding, int) or isinstance(padding, bool):
        raise ValueError("padding must be an integer")
    if padding < 0 or padding > 3:
        raise ValueError("padding must be in [0, 3]")
    if not torch.is_tensor(packed):
        raise ValueError("packed ternary codes must be a tensor")
    if packed.dtype != torch.uint8:
        raise ValueError("packed ternary codes must be a uint8 tensor")
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
            for dim in shape
        )
    ):
        raise ValueError("shape must be a tuple of two non-negative integers")
    bytes_cpu = packed.to(dtype=torch.uint8, device="cpu").reshape(-1)
    codes = torch.stack(
        [
            bytes_cpu & 0b11,
            (bytes_cpu >> 2) & 0b11,
            (bytes_cpu >> 4) & 0b11,
            (bytes_cpu >> 6) & 0b11,
        ],
        dim=1,
    ).reshape(-1)
    if padding:
        codes = codes[:-padding]
    expected = shape[0] * shape[1]
    if codes.numel() != expected:
        raise ValueError("packed ternary code count does not match target shape")
    if codes.numel() and codes.max().item() > 2:
        raise ValueError("packed ternary codes contain reserved values")
    return codes.view(shape)


def pack_bitlinear_layer(layer: BitLinear) -> dict[str, Any]:
    ternary, scale = ternary_codes_and_scales(layer.weight.detach())
    codes = ternary_to_codes(ternary)
    packed, padding = pack_ternary_codes(codes)
    return {
        "format": "alexandros-packed-bitlinear-layer",
        "format_version": 1,
        "encoding": "2bit_ternary_0_zero_1_pos_2_neg",
        "weight_shape": tuple(layer.weight.shape),
        "packed_weight": packed,
        "padding": padding,
        "scale": scale.detach().cpu().to(torch.float32).squeeze(-1),
        "bias": None
        if layer.bias is None
        else layer.bias.detach().cpu().to(torch.float32),
        "activation_bits": layer.activation_bits,
    }


def export_packed_bitlinear_state(model: nn.Module) -> dict[str, Any]:
    layers = {
        name: pack_bitlinear_layer(module)
        for name, module in model.named_modules()
        if isinstance(module, BitLinear)
    }
    return {
        "format": "alexandros-packed-bitlinear",
        "format_version": 1,
        "encoding": "2bit_ternary_0_zero_1_pos_2_neg",
        "layers": layers,
    }


def save_packed_bitlinear_state(model: nn.Module, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_packed_bitlinear_state(model), out_path)
