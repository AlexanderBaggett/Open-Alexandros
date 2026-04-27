from __future__ import annotations

import pytest
import torch

from alexandros.kv_cache import (
    TurboQuantKVCache,
    TurboQuantPacket,
    validate_turboquant_packet,
)


def test_turboquant_round_trip_preserves_shape_dtype_and_bounds() -> None:
    x = torch.randn(2, 4, 8, dtype=torch.float32)
    cache = TurboQuantKVCache(bits=4, use_qjl=True, seed=23)

    packet = cache.compress(x)
    restored = cache.decompress(packet)

    assert restored.shape == x.shape
    assert restored.dtype == x.dtype
    assert packet.q.dtype == torch.int8
    assert packet.q.abs().max().item() <= 7
    assert packet.qjl_sign is not None
    assert packet.packet_format_version == 1
    assert packet.qjl_projection_seed == cache.seed
    assert packet.qjl_residual_norm is not None
    assert packet.qjl_residual_norm.shape == packet.scale.shape
    assert packet.qjl_residual_norm.ge(0).all()
    assert packet.estimated_bits > packet.q.numel() * packet.bits
    assert torch.isfinite(restored).all()


def test_turboquant_attention_scores_match_decompressed_scalar_cache() -> None:
    torch.manual_seed(11)
    keys = torch.randn(2, 5, 8)
    query = torch.randn(2, 3, 8)
    cache = TurboQuantKVCache(bits=4, use_qjl=False, seed=29)
    packet = cache.compress(keys)

    estimated = cache.estimate_attention_scores(query, packet, use_qjl=False)
    restored = cache.decompress(packet)
    expected = torch.matmul(query, restored.transpose(-1, -2))

    torch.testing.assert_close(estimated, expected, atol=1e-5, rtol=1e-5)


def test_turboquant_qjl_attention_scores_improve_constructed_residual() -> None:
    cache = TurboQuantKVCache(bits=4, use_qjl=True, seed=31)
    q = torch.tensor([[[1, -2, 0, 3], [2, 1, -1, 0]]], dtype=torch.int8)
    scale = torch.ones(1, 2, 1)
    residual_sign = torch.tensor(
        [[[1, -1, 1, -1], [-1, -1, 1, 1]]],
        dtype=torch.float32,
    )
    residual_magnitude = 0.125
    residual = residual_sign * residual_magnitude
    projection = cache._qjl_projection(
        q.shape,
        q.device,
        torch.float32,
        cache.seed,
    )
    packet = TurboQuantPacket(
        q=q,
        scale=scale,
        bits=4,
        original_dtype=torch.float32,
        rotation_seed=cache.seed,
        qjl_sign=(residual * projection).sign().to(torch.int8),
        qjl_projection_seed=cache.seed,
        qjl_residual_norm=residual.norm(dim=-1, keepdim=True),
    )
    query = torch.randn(1, 3, 4)
    rotation = cache._rotation(4, q.device, torch.float32, cache.seed)
    exact_keys = (q.to(torch.float32) * scale + residual) @ rotation.transpose(-1, -2)
    exact_scores = torch.matmul(query, exact_keys.transpose(-1, -2))

    scalar_scores = cache.estimate_attention_scores(query, packet, use_qjl=False)
    qjl_scores = cache.estimate_attention_scores(query, packet, use_qjl=True)

    scalar_error = (scalar_scores - exact_scores).abs().mean()
    qjl_error = (qjl_scores - exact_scores).abs().mean()
    assert qjl_error < scalar_error
    torch.testing.assert_close(qjl_scores, exact_scores, atol=1e-5, rtol=1e-5)


def test_turboquant_attention_score_estimator_rejects_invalid_inputs() -> None:
    cache = TurboQuantKVCache(bits=4, use_qjl=False, seed=37)
    packet = cache.compress(torch.randn(1, 2, 4))

    with pytest.raises(ValueError, match="floating-point"):
        cache.estimate_attention_scores(torch.ones(1, 1, 4, dtype=torch.long), packet)
    with pytest.raises(ValueError, match="shape"):
        cache.estimate_attention_scores(torch.randn(1, 4), packet)
    with pytest.raises(ValueError, match="batch"):
        cache.estimate_attention_scores(torch.randn(2, 1, 4), packet)
    with pytest.raises(ValueError, match="last dimension"):
        cache.estimate_attention_scores(torch.randn(1, 1, 5), packet)
    with pytest.raises(ValueError, match="QJL"):
        cache.estimate_attention_scores(torch.randn(1, 1, 4), packet, use_qjl=True)


@pytest.mark.parametrize(
    "bad_input",
    [
        torch.ones(2, 3, dtype=torch.long),
        torch.tensor(float("nan")),
        torch.empty(2, 0),
        torch.tensor([[float("inf")]], dtype=torch.float32),
    ],
)
def test_turboquant_compress_rejects_invalid_inputs(bad_input: torch.Tensor) -> None:
    cache = TurboQuantKVCache(bits=4)

    with pytest.raises(ValueError):
        cache.compress(bad_input)


@pytest.mark.parametrize(
    ("packet", "message"),
    [
        (
            TurboQuantPacket(
                q=torch.zeros(1, 2, dtype=torch.int8),
                scale=torch.ones(1, 1),
                bits=1,
                original_dtype=torch.float32,
                rotation_seed=1,
            ),
            "bits",
        ),
        (
            TurboQuantPacket(
                q=torch.zeros(1, 2, dtype=torch.int8),
                scale=torch.ones(1, 2),
                bits=4,
                original_dtype=torch.float32,
                rotation_seed=1,
            ),
            "scale shape",
        ),
        (
            TurboQuantPacket(
                q=torch.full((1, 2), 8, dtype=torch.int8),
                scale=torch.ones(1, 1),
                bits=4,
                original_dtype=torch.float32,
                rotation_seed=1,
            ),
            "bit range",
        ),
        (
            TurboQuantPacket(
                q=torch.zeros(1, 2, dtype=torch.int8),
                scale=torch.ones(1, 1),
                bits=4,
                original_dtype=torch.float32,
                rotation_seed=1,
                qjl_sign=torch.full((1, 2), 2, dtype=torch.int8),
                qjl_projection_seed=1,
                qjl_residual_norm=torch.ones(1, 1),
            ),
            "qjl_sign",
        ),
        (
            TurboQuantPacket(
                q=torch.zeros(1, 2, dtype=torch.int8),
                scale=torch.ones(1, 1),
                bits=4,
                original_dtype=torch.float32,
                rotation_seed=1,
                qjl_sign=torch.zeros((1, 2), dtype=torch.int8),
            ),
            "qjl_projection_seed",
        ),
        (
            TurboQuantPacket(
                q=torch.zeros(1, 2, dtype=torch.int8),
                scale=torch.ones(1, 1),
                bits=4,
                original_dtype=torch.float32,
                rotation_seed=1,
                qjl_sign=torch.zeros((1, 2), dtype=torch.int8),
                qjl_projection_seed=1,
                qjl_residual_norm=torch.ones(1, 2),
            ),
            "qjl_residual_norm shape",
        ),
        (
            TurboQuantPacket(
                q=torch.zeros(1, 2, dtype=torch.int8),
                scale=torch.ones(1, 1),
                bits=4,
                original_dtype=torch.float32,
                rotation_seed=1,
                packet_format_version=2,
            ),
            "format version",
        ),
    ],
)
def test_turboquant_packet_validation_rejects_malformed_packets(
    packet: TurboQuantPacket,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_turboquant_packet(packet)
