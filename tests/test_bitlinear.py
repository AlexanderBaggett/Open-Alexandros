from __future__ import annotations

import pytest
import torch

from alexandros import AlexandrosConfig
from alexandros.bitlinear import (
    BitLinear,
    codes_to_ternary,
    pack_ternary_codes,
    ternary_codes_and_scales,
    unpack_ternary_codes,
)


def test_bitlinear_forward_uses_finite_quantized_values() -> None:
    layer = BitLinear(4, 3, activation_bits=8)
    x = torch.randn(2, 5, 4)

    out = layer(x)
    ternary, scale = ternary_codes_and_scales(layer.weight)

    assert out.shape == (2, 5, 3)
    assert torch.isfinite(out).all()
    assert torch.isfinite(scale).all()
    assert set(ternary.unique().tolist()).issubset({-1.0, 0.0, 1.0})


def test_bitlinear_uses_rowwise_absmean_scaling_and_ternary_thresholds() -> None:
    weight = torch.tensor(
        [
            [-2.0, -0.49, 0.49, 2.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    ternary, scale = ternary_codes_and_scales(weight)

    expected_scale = torch.tensor([[1.2450], [1e-6]])
    expected_ternary = torch.tensor([[-1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(scale, expected_scale)
    assert torch.equal(ternary, expected_ternary)


def test_bitlinear_uses_per_token_absmax_activation_scaling() -> None:
    layer = BitLinear(3, 2, activation_bits=3)
    x = torch.tensor(
        [
            [0.0, 1.4, -3.0],
            [2.0, -0.7, 0.4],
        ]
    )

    quantized = layer.quantize_activation(x)

    expected = torch.tensor(
        [
            [0.0, 1.0, -3.0],
            [2.0, -2.0 / 3.0, 2.0 / 3.0],
        ]
    )
    assert torch.allclose(quantized, expected, atol=1e-6)


def test_bitlinear_ste_sends_gradients_to_shadow_weight() -> None:
    layer = BitLinear(3, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [0.25, -0.5, 1.0],
                    [-1.0, 0.5, 0.25],
                ]
            )
        )
    x = torch.tensor([[1.0, -2.0, 0.5]], requires_grad=True)

    loss = layer(x).sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert layer.weight.grad.abs().sum().item() > 0.0
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_features": 0, "out_features": 3},
        {"in_features": 4, "out_features": 0},
        {"in_features": 4, "out_features": 3, "activation_bits": 1},
        {"in_features": 4, "out_features": 3, "activation_bits": True},
        {"in_features": 4, "out_features": 3, "bias": 1},
    ],
)
def test_bitlinear_rejects_invalid_constructor_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        BitLinear(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_codes",
    [
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([False, True]),
        torch.tensor([0, 1, 3], dtype=torch.uint8),
        torch.tensor([-1, 0, 1], dtype=torch.int64),
    ],
)
def test_ternary_code_helpers_reject_invalid_code_tensors(
    bad_codes: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        codes_to_ternary(bad_codes)
    with pytest.raises(ValueError):
        pack_ternary_codes(bad_codes)


def test_packed_ternary_codes_round_trip_without_silent_coercion() -> None:
    codes = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.uint8)

    packed, padding = pack_ternary_codes(codes)
    restored = unpack_ternary_codes(packed, shape=tuple(codes.shape), padding=padding)

    assert torch.equal(restored, codes.cpu())
    assert torch.equal(
        codes_to_ternary(restored), torch.tensor([[0, 1, -1, 0, 1]], dtype=torch.int8)
    )
    with pytest.raises(ValueError, match="uint8"):
        unpack_ternary_codes(packed.float(), shape=tuple(codes.shape), padding=padding)
    with pytest.raises(ValueError, match="padding"):
        unpack_ternary_codes(packed, shape=tuple(codes.shape), padding=True)  # type: ignore[arg-type]


def test_lite_config_rejects_one_bit_activation_quantization() -> None:
    cfg = AlexandrosConfig(
        variant="lite",
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        max_position_embeddings=16,
        moe_num_experts=1,
        moe_top_k=1,
        moe_expert_hidden_size=8,
        kv_lora_rank=4,
        latent_dim=4,
        latent_slots=1,
        diffusion_steps=1,
        depth_lora_rank=1,
        ttt_rank=1,
    )
    data = cfg.to_dict()
    data["bitnet_activation_bits"] = 1

    with pytest.raises(ValueError, match="bitnet_activation_bits"):
        AlexandrosConfig.from_dict(data)
