from __future__ import annotations

import pytest
import torch

from alexandros import AlexandrosConfig, AlexandrosForCausalLM
from alexandros.adaptive_depth import AdaptiveDepthController
from alexandros.evaluation import adaptive_depth_toy_benchmark


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
        max_depth_iters=3,
        act_threshold=0.75,
        act_ponder_cost=0.2,
        depth_lora_rank=2,
        ttt_rank=2,
    )


def set_halting_bias(controller: AdaptiveDepthController, value: float) -> None:
    with torch.no_grad():
        controller.halt_proj.weight.zero_()
        assert controller.halt_proj.bias is not None
        controller.halt_proj.bias.fill_(value)


def test_adaptive_depth_records_act_stats_and_ponder_cost() -> None:
    cfg = tiny_config()
    controller = AdaptiveDepthController(cfg)
    set_halting_bias(controller, 100.0)
    hidden = torch.randn(2, 4, cfg.hidden_size)

    output, halting = controller(hidden)
    stats = controller.last_stats

    assert output.shape == hidden.shape
    assert halting.shape == hidden.shape[:-1]
    assert stats is not None
    torch.testing.assert_close(halting, torch.full_like(halting, cfg.act_threshold))
    torch.testing.assert_close(stats.halting_sum, halting)
    torch.testing.assert_close(
        stats.remainder, torch.full_like(halting, cfg.act_threshold)
    )
    torch.testing.assert_close(stats.n_updates, torch.ones_like(halting))
    assert stats.average_loop_count == pytest.approx(1.0)
    assert stats.ponder_cost.item() == pytest.approx(cfg.act_ponder_cost)
    assert torch.isfinite(output).all()


def test_adaptive_depth_uses_depth_wise_lora_ranks() -> None:
    cfg = tiny_config()
    cfg.depth_lora_ranks = [1, 2, 3]
    controller = AdaptiveDepthController(cfg)

    assert [adapter.rank for adapter in controller.depth_lora_adapters] == [1, 2, 3]
    assert controller.depth_lora_adapters[0].down.weight.shape == (1, cfg.hidden_size)
    assert controller.depth_lora_adapters[1].down.weight.shape == (2, cfg.hidden_size)
    assert controller.depth_lora_adapters[2].up.weight.shape == (cfg.hidden_size, 3)
    assert cfg.depth_lora_rank_for_loop(2) == 3
    with pytest.raises(ValueError, match="loop_idx"):
        cfg.depth_lora_rank_for_loop(3)


def test_adaptive_depth_lora_is_loop_specific() -> None:
    cfg = tiny_config()
    cfg.max_depth_iters = 2
    cfg.act_threshold = 1.0
    cfg.depth_lora_ranks = [1, 1]
    controller = AdaptiveDepthController(cfg)
    set_halting_bias(controller, -100.0)
    with torch.no_grad():
        controller.transition.weight.zero_()
        assert controller.transition.bias is not None
        controller.transition.bias.fill_(1.0)
        for adapter in controller.depth_lora_adapters:
            adapter.down.weight.fill_(1.0)
            adapter.up.weight.zero_()
        controller.depth_lora_adapters[0].up.weight.fill_(0.25)
        controller.depth_lora_adapters[1].up.weight.fill_(0.75)
    hidden = torch.ones(1, 2, cfg.hidden_size)

    output, _ = controller(hidden)

    assert torch.isfinite(output).all()
    assert controller.last_stats is not None
    assert controller.last_stats.average_loop_count == pytest.approx(2.0)
    assert not torch.allclose(output, hidden)


def test_adaptive_depth_toy_benchmark_reports_cost_and_rank() -> None:
    cfg = tiny_config()
    cfg.vocab_size = 32
    cfg.enable_adaptive_depth = True
    model = AlexandrosForCausalLM(cfg).eval()

    report = adaptive_depth_toy_benchmark(model)

    assert report.target_rank >= 1
    assert 0.0 <= report.target_probability <= 1.0
    assert report.average_loop_count >= 1.0
    assert report.ponder_cost >= 0.0
    assert report.elapsed_ms >= 0.0


def test_adaptive_depth_toy_benchmark_requires_enabled_controller() -> None:
    cfg = tiny_config()
    cfg.enable_adaptive_depth = False
    model = AlexandrosForCausalLM(cfg)

    with pytest.raises(TypeError, match="enable_adaptive_depth"):
        adaptive_depth_toy_benchmark(model)


def test_adaptive_depth_last_iteration_forces_remainder() -> None:
    cfg = tiny_config()
    controller = AdaptiveDepthController(cfg)
    set_halting_bias(controller, -100.0)
    hidden = torch.randn(1, 3, cfg.hidden_size)

    _, halting = controller(hidden)
    stats = controller.last_stats

    assert stats is not None
    torch.testing.assert_close(halting, torch.full_like(halting, cfg.act_threshold))
    torch.testing.assert_close(
        stats.n_updates, torch.full_like(halting, cfg.max_depth_iters)
    )
    assert stats.average_loop_count == pytest.approx(float(cfg.max_depth_iters))
    assert stats.ponder_cost.item() == pytest.approx(
        cfg.max_depth_iters * cfg.act_ponder_cost
    )


@pytest.mark.parametrize(
    ("bad_hidden", "message"),
    [
        (torch.ones(2, 16), "shape"),
        (torch.empty(0, 1, 16), "batch size"),
        (torch.empty(1, 0, 16), "sequence length"),
        (torch.ones(1, 2, 15), "hidden_size"),
        (torch.ones(1, 2, 16, dtype=torch.long), "floating-point"),
        (torch.full((1, 2, 16), float("nan")), "finite"),
    ],
)
def test_adaptive_depth_rejects_invalid_hidden_states(
    bad_hidden: torch.Tensor,
    message: str,
) -> None:
    controller = AdaptiveDepthController(tiny_config())

    with pytest.raises(ValueError, match=message):
        controller(bad_hidden)


@pytest.mark.parametrize(
    ("ranks", "message"),
    [
        ((1, 2, 3), "list"),
        ([1, 2], "length"),
        ([1, 0, 1], "> 0"),
        ([1, 2, 99], "hidden_size"),
    ],
)
def test_adaptive_depth_rejects_invalid_depth_lora_ranks(
    ranks: object,
    message: str,
) -> None:
    cfg = tiny_config()
    data = cfg.to_dict()
    data["depth_lora_ranks"] = ranks

    with pytest.raises(ValueError, match=message):
        AlexandrosConfig.from_dict(data)
