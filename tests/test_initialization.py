from __future__ import annotations

import torch

from alexandros import AlexandrosConfig, AlexandrosForCausalLM
from alexandros.bitlinear import BitLinear
from alexandros.initialization import DEFAULT_INITIALIZER_STD, residual_projection_std
from alexandros.latent_reasoning import LatentDiffusionReasoner
from alexandros.moe import MoEFeedForward


def tiny_config(variant: str = "heavy") -> AlexandrosConfig:
    return AlexandrosConfig(
        variant=variant,
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


def test_model_initialization_sets_pad_embedding_norms_and_tied_head() -> None:
    cfg = tiny_config("heavy")
    cfg.tie_word_embeddings = True
    model = AlexandrosForCausalLM(cfg)

    assert torch.equal(
        model.model.embed_tokens.weight[cfg.pad_token_id],
        torch.zeros_like(model.model.embed_tokens.weight[cfg.pad_token_id]),
    )
    assert model.model.embed_tokens.weight[cfg.bos_token_id].abs().sum().item() > 0.0
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert torch.allclose(
        model.model.norm.weight, torch.ones_like(model.model.norm.weight)
    )
    for layer in model.model.layers:
        assert torch.allclose(
            layer.input_norm.weight, torch.ones_like(layer.input_norm.weight)
        )
        assert torch.allclose(
            layer.post_norm.weight, torch.ones_like(layer.post_norm.weight)
        )


def test_moe_timestep_router_bias_starts_neutral_and_resets_state() -> None:
    cfg = tiny_config()
    moe = MoEFeedForward(cfg)

    assert torch.count_nonzero(moe.timestep_router_bias.weight).item() == 0
    assert torch.count_nonzero(moe.router_bias).item() == 0
    assert torch.count_nonzero(moe.router_load_ema).item() == 0
    assert moe.router_load_ema_steps.item() == 0

    with torch.no_grad():
        moe.timestep_router_bias.weight.fill_(1.0)
        moe.router_bias.fill_(1.0)
        moe.router_load_ema.fill_(1.0)
        moe.router_load_ema_steps.fill_(3)
        moe.timestep_expert_load.fill_(1.0)
        moe.timestep_expert_count.fill_(1.0)
    moe.last_stats = object()  # type: ignore[assignment]

    moe.reset_parameters()

    assert torch.count_nonzero(moe.timestep_router_bias.weight).item() == 0
    assert torch.count_nonzero(moe.router_bias).item() == 0
    assert torch.count_nonzero(moe.router_load_ema).item() == 0
    assert moe.router_load_ema_steps.item() == 0
    assert torch.count_nonzero(moe.timestep_expert_load).item() == 0
    assert torch.count_nonzero(moe.timestep_expert_count).item() == 0
    assert moe.last_stats is None


def test_residual_projection_std_is_scaled_by_depth() -> None:
    cfg = tiny_config()

    assert (
        residual_projection_std(cfg)
        == DEFAULT_INITIALIZER_STD / (2.0 * cfg.num_hidden_layers) ** 0.5
    )
    assert residual_projection_std(cfg) < DEFAULT_INITIALIZER_STD


def test_lite_bitlinear_shadow_weights_are_explicitly_initialized() -> None:
    cfg = tiny_config("lite")
    model = AlexandrosForCausalLM(cfg)
    bitlinear_layers = [
        module for module in model.modules() if isinstance(module, BitLinear)
    ]

    assert bitlinear_layers
    for layer in bitlinear_layers:
        assert torch.isfinite(layer.weight).all()
        assert layer.weight.abs().sum().item() > 0.0
        if layer.bias is not None:
            assert torch.count_nonzero(layer.bias).item() == 0


def test_latent_reasoner_initialization_resets_norms_and_timestep_embeddings() -> None:
    cfg = tiny_config()
    reasoner = LatentDiffusionReasoner(cfg)

    assert torch.allclose(
        reasoner.attn_norm.weight, torch.ones_like(reasoner.attn_norm.weight)
    )
    assert torch.allclose(
        reasoner.attn_norm.bias, torch.zeros_like(reasoner.attn_norm.bias)
    )
    assert torch.allclose(
        reasoner.ffn_norm.weight, torch.ones_like(reasoner.ffn_norm.weight)
    )
    assert torch.allclose(
        reasoner.ffn_norm.bias, torch.zeros_like(reasoner.ffn_norm.bias)
    )
    assert torch.isfinite(reasoner.timestep_embed.weight).all()
    assert reasoner.timestep_embed.weight.abs().sum().item() > 0.0
