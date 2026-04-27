from __future__ import annotations

import pytest
import torch

from alexandros import AlexandrosConfig
from alexandros.ssm_gated_deltanet import GatedDeltaNetBlock


def tiny_config() -> AlexandrosConfig:
    return AlexandrosConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        linear_attention_ratio=2,
        moe_num_experts=2,
        moe_num_shared_experts=1,
        moe_top_k=1,
        moe_expert_hidden_size=16,
        kv_lora_rank=4,
        latent_dim=8,
        latent_slots=2,
        diffusion_steps=4,
        mask_token_id=3,
        ttt_rank=2,
    )


def test_gated_deltanet_recurrent_state_contract_and_chunk_equivalence() -> None:
    cfg = tiny_config()
    block = GatedDeltaNetBlock(cfg).eval()
    x = torch.randn(2, 5, cfg.hidden_size)

    full_y, full_state = block(x)
    first_y, first_state = block(x[:, :3])
    second_y, second_state = block(x[:, 3:], state=first_state)
    chunked_y = torch.cat([first_y, second_y], dim=1)

    assert block.recurrent_state_shape(batch_size=2) == (2, cfg.hidden_size)
    assert full_state.shape == (2, cfg.hidden_size)
    assert full_state.requires_grad is False
    assert torch.allclose(chunked_y, full_y, atol=1e-6)
    assert torch.allclose(second_state, full_state, atol=1e-6)


def test_gated_deltanet_attention_mask_preserves_state_on_masked_tokens() -> None:
    cfg = tiny_config()
    block = GatedDeltaNetBlock(cfg).eval()
    x = torch.randn(1, 3, cfg.hidden_size)
    mask_first_only = torch.tensor([[1, 0, 0]], dtype=torch.long)

    masked_y, masked_state = block(x, attention_mask=mask_first_only)
    first_y, first_state = block(x[:, :1])

    assert torch.allclose(masked_y[:, :1], first_y, atol=1e-6)
    assert torch.count_nonzero(masked_y[:, 1:]).item() == 0
    assert torch.allclose(masked_state, first_state, atol=1e-6)


@pytest.mark.parametrize(
    ("bad_x", "message"),
    [
        (torch.ones(2, 16), "shape"),
        (torch.empty(0, 1, 16), "batch size"),
        (torch.empty(1, 0, 16), "sequence length"),
        (torch.ones(1, 2, 15), "hidden_size"),
        (torch.ones(1, 2, 16, dtype=torch.long), "floating-point"),
        (torch.full((1, 2, 16), float("nan")), "finite"),
    ],
)
def test_gated_deltanet_rejects_invalid_hidden_states(
    bad_x: torch.Tensor,
    message: str,
) -> None:
    block = GatedDeltaNetBlock(tiny_config())

    with pytest.raises(ValueError, match=message):
        block(bad_x)


@pytest.mark.parametrize(
    ("bad_state", "message"),
    [
        (torch.ones(1, 15), "shape"),
        (torch.ones(1, 16, dtype=torch.float64), "dtype"),
        (torch.ones(1, 16, dtype=torch.long), "floating-point"),
        (torch.full((1, 16), float("inf")), "finite"),
    ],
)
def test_gated_deltanet_rejects_invalid_recurrent_state(
    bad_state: torch.Tensor,
    message: str,
) -> None:
    cfg = tiny_config()
    block = GatedDeltaNetBlock(cfg)
    x = torch.randn(1, 2, cfg.hidden_size)

    with pytest.raises(ValueError, match=message):
        block(x, state=bad_state)


@pytest.mark.parametrize(
    ("bad_mask", "message"),
    [
        (torch.ones(2, 2), "shape"),
        (torch.ones(1, 1), "shorter"),
        (torch.tensor([[1, 2]]), "0/1"),
        (torch.tensor([[1.0, float("nan")]]), "finite"),
    ],
)
def test_gated_deltanet_rejects_invalid_attention_mask(
    bad_mask: torch.Tensor,
    message: str,
) -> None:
    cfg = tiny_config()
    block = GatedDeltaNetBlock(cfg)
    x = torch.randn(1, 2, cfg.hidden_size)

    with pytest.raises(ValueError, match=message):
        block(x, attention_mask=bad_mask)


def test_gated_deltanet_rejects_invalid_state_shape_request() -> None:
    block = GatedDeltaNetBlock(tiny_config())

    with pytest.raises(ValueError, match="batch_size"):
        block.recurrent_state_shape(0)
