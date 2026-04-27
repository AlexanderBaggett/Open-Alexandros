from __future__ import annotations

import pytest

from alexandros import AlexandrosConfig, AlexandrosForCausalLM, AlexandrosForDiffusionLM
from alexandros.training import apply_trainability, trainable_parameters


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
        moe_expert_hidden_size=12,
        kv_lora_rank=4,
        latent_dim=8,
        latent_slots=2,
        diffusion_steps=4,
        mask_token_id=3,
        ttt_rank=2,
    )


def trainable_names(model) -> set[str]:
    return {name for name, param in model.named_parameters() if param.requires_grad}


def test_trainability_phase_defaults_match_smoke_contracts() -> None:
    cfg = tiny_config()
    causal = AlexandrosForCausalLM(cfg)
    ar_report = apply_trainability(causal, phase="ar")
    assert ar_report.trainable_parameters == sum(
        param.numel() for param in causal.parameters()
    )
    assert ar_report.frozen_parameters == 0

    diffusion = AlexandrosForDiffusionLM(cfg)
    diffusion_report = apply_trainability(diffusion, phase="diffusion")
    diffusion_names = trainable_names(diffusion)
    assert diffusion_report.trainable_parameters > 0
    assert any(name.startswith("model.") for name in diffusion_names)
    assert any(name.startswith("lm_head.") for name in diffusion_names)
    assert not any(name.startswith("latent_vae.") for name in diffusion_names)
    assert not any(name.startswith("latent_reasoner.") for name in diffusion_names)

    latent = AlexandrosForDiffusionLM(cfg)
    latent_report = apply_trainability(latent, phase="latent")
    latent_names = trainable_names(latent)
    assert latent_report.trainable_parameters > 0
    assert any(name.startswith("model.") for name in latent_names)
    assert any(name.startswith("latent_vae.") for name in latent_names)
    assert any(name.startswith("latent_reasoner.") for name in latent_names)
    assert not any(name.startswith("lm_head.") for name in latent_names)

    ttt = AlexandrosForCausalLM(cfg)
    ttt_report = apply_trainability(ttt, phase="ttt")
    assert ttt_report.trainable_parameters == 0
    assert list(trainable_parameters(ttt)) == []


@pytest.mark.parametrize(
    ("scope", "expected_prefixes"),
    [
        ("backbone_only", ("model.",)),
        ("head_only", ("lm_head.",)),
        ("latent_only", ("latent_vae.", "latent_reasoner.")),
    ],
)
def test_trainability_explicit_scopes(
    scope: str, expected_prefixes: tuple[str, ...]
) -> None:
    model = AlexandrosForDiffusionLM(tiny_config())

    report = apply_trainability(model, phase="latent", scope=scope)
    names = trainable_names(model)

    assert report.trainable_parameters > 0
    assert names
    assert all(name.startswith(expected_prefixes) for name in names)


def test_trainability_rejects_unknown_phase_and_scope() -> None:
    model = AlexandrosForCausalLM(tiny_config())

    with pytest.raises(ValueError, match="unknown training phase"):
        apply_trainability(model, phase="missing")
    with pytest.raises(ValueError, match="trainable scope"):
        apply_trainability(model, phase="ar", scope="missing")
