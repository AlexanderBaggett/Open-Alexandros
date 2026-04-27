from __future__ import annotations

import pytest
import torch
from torch import nn

from alexandros import AlexandrosConfig
from alexandros.moe import MoEFeedForward


def tiny_config() -> AlexandrosConfig:
    return AlexandrosConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        linear_attention_ratio=2,
        moe_num_experts=3,
        moe_num_shared_experts=1,
        moe_top_k=2,
        moe_expert_hidden_size=12,
        kv_lora_rank=4,
        latent_dim=8,
        latent_slots=2,
        diffusion_steps=4,
        mask_token_id=3,
        ttt_rank=2,
    )


def test_moe_normalizes_router_weights_and_tracks_timestep_load() -> None:
    cfg = tiny_config()
    moe = MoEFeedForward(cfg)
    x = torch.randn(2, 3, cfg.hidden_size)

    out = moe(x, diffusion_timestep=torch.tensor([0, 1], dtype=torch.long))

    assert out.shape == x.shape
    assert moe.last_stats is not None
    selected = moe.last_stats.selected_experts
    assert selected.shape == (2, 3, cfg.moe_top_k)
    assert moe.last_stats.expert_load.sum().isclose(torch.tensor(1.0))
    assert moe.timestep_expert_count[0].item() > 0
    assert moe.timestep_expert_count[1].item() > 0


def test_moe_router_bias_changes_selection_not_mixture_weights() -> None:
    cfg = tiny_config()
    cfg.moe_num_shared_experts = 0
    cfg.router_bias_clip = 10.0
    moe = MoEFeedForward(cfg)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[:, 0] = torch.tensor([2.0, 1.0, 0.0])
        moe.router_bias.copy_(torch.tensor([-5.0, 0.0, 5.0]))
    x = torch.zeros(1, 1, cfg.hidden_size)
    x[..., 0] = 1.0

    moe(x)

    assert moe.last_stats is not None
    assert moe.last_stats.selected_experts.tolist() == [[[2, 1]]]
    selected_base_probs = torch.sigmoid(torch.tensor([0.0, 1.0]))
    expected_weights = selected_base_probs / selected_base_probs.sum()
    assert torch.allclose(
        moe.last_stats.routed_weights[0, 0],
        expected_weights,
        atol=1e-6,
    )
    biased_probs = torch.sigmoid(torch.tensor([5.0, 1.0]))
    biased_weights = biased_probs / biased_probs.sum()
    assert not torch.allclose(moe.last_stats.routed_weights[0, 0], biased_weights)


def test_moe_token_state_routing_changes_masked_unmasked_selection() -> None:
    cfg = tiny_config()
    cfg.moe_top_k = 1
    cfg.moe_num_shared_experts = 0
    cfg.moe_token_state_routing = True
    moe = MoEFeedForward(cfg)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.timestep_router_bias.weight.zero_()
        moe.token_state_router_bias.weight.zero_()
        moe.token_state_router_bias.weight[0, 0] = 5.0
        moe.token_state_router_bias.weight[1, 1] = 5.0
    x = torch.zeros(1, 2, cfg.hidden_size)
    token_state = torch.tensor([[0, 1]], dtype=torch.long)

    moe(x, diffusion_timestep=0, diffusion_token_state=token_state)

    assert moe.last_stats is not None
    assert moe.last_stats.selected_experts.tolist() == [[[0], [1]]]
    assert "router_bias" not in dict(moe.named_parameters())
    assert moe.router_bias.requires_grad is False


def test_moe_position_routing_changes_bucket_selection() -> None:
    cfg = tiny_config()
    cfg.moe_top_k = 1
    cfg.moe_num_shared_experts = 0
    cfg.moe_position_routing = True
    cfg.moe_position_buckets = 4
    cfg.max_position_embeddings = 8
    moe = MoEFeedForward(cfg)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.timestep_router_bias.weight.zero_()
        moe.token_state_router_bias.weight.zero_()
        moe.position_router_bias.weight.zero_()
        moe.position_router_bias.weight[0, 0] = 5.0
        moe.position_router_bias.weight[2, 1] = 5.0
        moe.position_router_bias.weight[3, 2] = 5.0
    x = torch.zeros(1, 3, cfg.hidden_size)
    position_ids = torch.tensor([[0, 4, 7]], dtype=torch.long)

    moe(x, diffusion_timestep=0, position_ids=position_ids)

    assert moe.last_stats is not None
    assert moe.last_stats.selected_experts.tolist() == [[[0], [1], [2]]]
    assert "router_bias" not in dict(moe.named_parameters())
    assert moe.router_bias.requires_grad is False


def test_moe_routed_weights_are_normalized_sigmoid_not_softmax() -> None:
    cfg = tiny_config()
    cfg.moe_num_shared_experts = 0
    moe = MoEFeedForward(cfg)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[:, 0] = torch.tensor([2.0, 1.0, 0.0])
    x = torch.zeros(1, 1, cfg.hidden_size)
    x[..., 0] = 1.0

    moe(x)

    assert moe.last_stats is not None
    assert moe.last_stats.selected_experts.tolist() == [[[0, 1]]]
    selected_logits = torch.tensor([2.0, 1.0])
    sigmoid_weights = torch.sigmoid(selected_logits)
    sigmoid_weights = sigmoid_weights / sigmoid_weights.sum()
    softmax_weights = torch.softmax(selected_logits, dim=0)
    assert torch.allclose(
        moe.last_stats.routed_weights[0, 0], sigmoid_weights, atol=1e-6
    )
    assert not torch.allclose(moe.last_stats.routed_weights[0, 0], softmax_weights)


class _ScaleExpert(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def test_moe_shared_experts_are_always_on_and_combined_separately() -> None:
    cfg = tiny_config()
    cfg.moe_top_k = 1
    cfg.moe_num_shared_experts = 1
    moe = MoEFeedForward(cfg)
    moe.routed_experts = nn.ModuleList(
        [_ScaleExpert(1.0), _ScaleExpert(2.0), _ScaleExpert(3.0)]
    )
    moe.shared_experts = nn.ModuleList([_ScaleExpert(10.0)])
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0, 0] = 1.0
    x = torch.zeros(1, 1, cfg.hidden_size)
    x[..., 0] = 2.0

    out = moe(x)

    assert moe.last_stats is not None
    assert moe.last_stats.selected_experts.tolist() == [[[0]]]
    assert torch.allclose(
        moe.last_stats.routed_weights, torch.ones_like(moe.last_stats.routed_weights)
    )
    assert torch.allclose(out, x * 11.0)


@pytest.mark.parametrize(
    ("bad_timestep", "message"),
    [
        (True, "integer value"),
        (torch.tensor([0.5]), "integer values"),
        (torch.tensor([float("nan")]), "finite"),
        (torch.tensor([], dtype=torch.long), "cannot be empty"),
        (torch.tensor([4], dtype=torch.long), "diffusion_steps"),
        (torch.zeros((2, 2, 1), dtype=torch.long), "scalar"),
    ],
)
def test_moe_rejects_invalid_diffusion_timestep_inputs(
    bad_timestep,
    message: str,
) -> None:
    cfg = tiny_config()
    moe = MoEFeedForward(cfg)
    x = torch.randn(2, 3, cfg.hidden_size)

    with pytest.raises(ValueError, match=message):
        moe(x, diffusion_timestep=bad_timestep)


@pytest.mark.parametrize(
    ("bad_state", "message"),
    [
        (torch.tensor([[0.5]]), "integer"),
        (torch.tensor([[float("nan")]]), "finite"),
        (torch.tensor([[2]], dtype=torch.long), "0/1"),
        (torch.empty(0, dtype=torch.long), "empty"),
        (torch.zeros((2, 2, 1), dtype=torch.long), "scalar"),
    ],
)
def test_moe_rejects_invalid_diffusion_token_state_inputs(
    bad_state: torch.Tensor,
    message: str,
) -> None:
    cfg = tiny_config()
    cfg.moe_token_state_routing = True
    moe = MoEFeedForward(cfg)
    x = torch.randn(2, 3, cfg.hidden_size)

    with pytest.raises(ValueError, match=message):
        moe(x, diffusion_timestep=0, diffusion_token_state=bad_state)


@pytest.mark.parametrize(
    ("bad_positions", "message"),
    [
        (torch.tensor([[0.5]], dtype=torch.float32), "integer"),
        (torch.tensor([[float("nan")]], dtype=torch.float32), "finite"),
        (torch.tensor([[-1]], dtype=torch.long), "max_position_embeddings"),
        (torch.tensor([[32]], dtype=torch.long), "max_position_embeddings"),
        (torch.empty(0, dtype=torch.long), "empty"),
        (torch.zeros((2, 2, 1), dtype=torch.long), "scalar"),
    ],
)
def test_moe_rejects_invalid_position_ids(
    bad_positions: torch.Tensor,
    message: str,
) -> None:
    cfg = tiny_config()
    cfg.moe_position_routing = True
    moe = MoEFeedForward(cfg)
    x = torch.randn(2, 3, cfg.hidden_size)

    with pytest.raises(ValueError, match=message):
        moe(x, diffusion_timestep=0, position_ids=bad_positions)


def test_moe_router_bias_update_moves_overloaded_expert_down() -> None:
    cfg = tiny_config()
    cfg.router_bias_clip = 0.25
    moe = MoEFeedForward(cfg)
    before = moe.router_bias.clone()

    moe.update_router_bias(torch.tensor([0.9, 0.05, 0.05]), rate=0.5)

    assert moe.router_bias[0] < before[0]
    assert moe.router_bias[1] > before[1]
    assert moe.router_bias[2] > before[2]
    assert moe.router_bias.abs().max() <= cfg.router_bias_clip
