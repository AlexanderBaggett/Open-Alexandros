from __future__ import annotations

import pytest
import torch

from alexandros import AlexandrosConfig
from alexandros.latent_reasoning import LatentDiffusionReasoner, LatentThoughtVAE


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
        latent_slots=3,
        latent_update_clip=0.5,
        diffusion_steps=4,
        mask_token_id=3,
        ttt_rank=2,
    )


def test_latent_vae_outputs_finite_slots_and_reconstruction() -> None:
    cfg = tiny_config()
    vae = LatentThoughtVAE(cfg).eval()
    hidden = torch.randn(2, 5, cfg.hidden_size)

    out = vae(hidden)

    assert out.latents.shape == (2, cfg.latent_slots, cfg.latent_dim)
    assert out.reconstruction.shape == (2, cfg.latent_slots, cfg.hidden_size)
    assert out.mu.shape == (2, cfg.latent_dim)
    assert out.logvar.shape == (2, cfg.latent_dim)
    assert torch.isfinite(out.kl_loss)
    assert torch.isfinite(out.latents).all()


@pytest.mark.parametrize(
    ("bad_hidden", "message"),
    [
        (torch.ones(2, 16), "shape"),
        (torch.empty(0, 1, 16), "batch size"),
        (torch.empty(1, 0, 16), "sequence length"),
        (torch.ones(1, 2, 15), "hidden_size"),
        (torch.ones(1, 2, 16, dtype=torch.long), "floating-point"),
        (torch.full((1, 2, 16), float("inf")), "finite"),
    ],
)
def test_latent_vae_rejects_invalid_hidden_states(
    bad_hidden: torch.Tensor,
    message: str,
) -> None:
    vae = LatentThoughtVAE(tiny_config())

    with pytest.raises(ValueError, match=message):
        vae(bad_hidden)


def test_latent_reasoner_refines_and_decodes_finite_latents() -> None:
    cfg = tiny_config()
    reasoner = LatentDiffusionReasoner(cfg)
    latents = torch.randn(2, cfg.latent_slots, cfg.latent_dim)

    refined = reasoner(latents, steps=2)
    decoded = reasoner.decode_to_hidden(refined)

    assert refined.shape == latents.shape
    assert decoded.shape == (2, cfg.latent_slots, cfg.hidden_size)
    assert torch.isfinite(refined).all()
    assert torch.isfinite(decoded).all()
    assert (refined - latents).norm(dim=-1).max() <= cfg.latent_update_clip + 1e-5
    assert reasoner.last_stats["steps_requested"] == 2
    assert reasoner.last_stats["steps_run"] == 2
    assert reasoner.last_stats["halted"] is False


def test_latent_reasoner_adaptive_compute_halts_on_small_update() -> None:
    cfg = tiny_config()
    cfg.latent_adaptive_threshold = 1e-3
    reasoner = LatentDiffusionReasoner(cfg)
    with torch.no_grad():
        for parameter in reasoner.parameters():
            parameter.zero_()
    latents = torch.randn(2, cfg.latent_slots, cfg.latent_dim)

    refined = reasoner(latents, steps=5)

    assert torch.equal(refined, latents)
    assert reasoner.last_stats["steps_requested"] == 5
    assert reasoner.last_stats["steps_run"] == 1
    assert reasoner.last_stats["halted"] is True
    assert reasoner.last_stats["last_update_norm"] == pytest.approx(0.0)


def test_latent_reasoner_uses_bidirectional_slot_attention() -> None:
    cfg = tiny_config()
    cfg.latent_update_clip = 10.0
    reasoner = LatentDiffusionReasoner(cfg).eval()
    with torch.no_grad():
        reasoner.timestep_embed.weight.zero_()
        reasoner.q_proj.weight.zero_()  # type: ignore[attr-defined]
        reasoner.k_proj.weight.zero_()  # type: ignore[attr-defined]
        reasoner.v_proj.weight.copy_(torch.eye(cfg.latent_dim))  # type: ignore[attr-defined]
        reasoner.attn_out_proj.weight.copy_(torch.eye(cfg.latent_dim))  # type: ignore[attr-defined]
        reasoner.in_proj.weight.zero_()  # type: ignore[attr-defined]
        reasoner.in_proj.bias.zero_()  # type: ignore[attr-defined]
        reasoner.out_proj.weight.zero_()  # type: ignore[attr-defined]
        reasoner.out_proj.bias.zero_()  # type: ignore[attr-defined]
    latents_a = torch.zeros(1, cfg.latent_slots, cfg.latent_dim)
    latents_b = latents_a.clone()
    latents_a[0, 0, 0] = 1.0
    latents_b[0, 0, 0] = 1.0
    latents_b[0, 1, 1] = 5.0

    refined_a = reasoner(latents_a, steps=1)
    refined_b = reasoner(latents_b, steps=1)

    assert reasoner.num_latent_heads == 4
    assert not torch.allclose(refined_a[:, 0], refined_b[:, 0])
    assert torch.isfinite(refined_b).all()


def test_latent_reasoner_selects_divisible_attention_head_count() -> None:
    cfg = tiny_config()
    cfg.latent_dim = 10
    cfg.num_attention_heads = 4
    reasoner = LatentDiffusionReasoner(cfg)

    assert reasoner.num_latent_heads == 2
    assert cfg.latent_dim % reasoner.num_latent_heads == 0


@pytest.mark.parametrize(
    ("bad_latents", "message"),
    [
        (torch.ones(2, 8), "shape"),
        (torch.empty(0, 1, 8), "batch size"),
        (torch.empty(1, 0, 8), "slot"),
        (torch.ones(1, 2, 7), "latent_dim"),
        (torch.ones(1, 2, 8, dtype=torch.long), "floating-point"),
        (torch.full((1, 2, 8), float("nan")), "finite"),
    ],
)
def test_latent_reasoner_rejects_invalid_latents(
    bad_latents: torch.Tensor,
    message: str,
) -> None:
    reasoner = LatentDiffusionReasoner(tiny_config())

    with pytest.raises(ValueError, match=message):
        reasoner(bad_latents)
    with pytest.raises(ValueError, match=message):
        reasoner.decode_to_hidden(bad_latents)


@pytest.mark.parametrize("bad_steps", [0, -1, True, 1.5])
def test_latent_reasoner_rejects_invalid_steps(bad_steps: object) -> None:
    cfg = tiny_config()
    reasoner = LatentDiffusionReasoner(cfg)
    latents = torch.randn(1, cfg.latent_slots, cfg.latent_dim)

    with pytest.raises(ValueError, match="steps"):
        reasoner(latents, steps=bad_steps)  # type: ignore[arg-type]


def test_latent_adaptive_threshold_must_be_nonnegative() -> None:
    cfg = tiny_config()
    data = cfg.to_dict()
    data["latent_adaptive_threshold"] = -1e-3

    with pytest.raises(ValueError, match="latent_adaptive_threshold"):
        AlexandrosConfig.from_dict(data)
