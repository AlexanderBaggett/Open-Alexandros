from __future__ import annotations

import pytest
import torch

from alexandros import AlexandrosConfig, AlexandrosForDiffusionLM
from alexandros.diffusion import MaskedDiffusionScheduler
from alexandros.training import (
    make_block_diffusion_training_batch,
    make_diffusion_training_batch,
)


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


def test_scheduler_masks_at_least_one_non_pad_token_per_row() -> None:
    cfg = tiny_config()
    scheduler = MaskedDiffusionScheduler(cfg)
    input_ids = torch.tensor(
        [
            [cfg.bos_token_id, 4, cfg.pad_token_id],
            [cfg.bos_token_id, 5, 6],
        ],
        dtype=torch.long,
    )

    noisy, mask = scheduler.add_noise(
        input_ids, timestep=0, generator=torch.Generator().manual_seed(0)
    )

    assert mask.any(dim=1).all()
    assert not mask[input_ids.eq(cfg.pad_token_id)].any()
    assert noisy[mask].eq(cfg.mask_token_id).all()


def test_scheduler_loss_weighting_schemes_are_explicit() -> None:
    cfg = tiny_config()
    scheduler = MaskedDiffusionScheduler(cfg)
    timesteps = torch.tensor([0, cfg.diffusion_steps - 1], dtype=torch.long)

    uniform = scheduler.loss_weight(timesteps, scheme="uniform")
    mask_prob = scheduler.loss_weight(timesteps, scheme="mask_prob")
    inverse = scheduler.loss_weight(timesteps, scheme="inverse_mask_prob")

    assert torch.equal(uniform, torch.ones_like(uniform))
    assert torch.allclose(mask_prob, torch.tensor([0.25, 1.0]))
    assert torch.allclose(inverse, torch.tensor([4.0, 1.0]))
    with pytest.raises(ValueError, match="weighting scheme"):
        scheduler.loss_weight(timesteps, scheme="unknown")


def test_scheduler_timestep_grid_matches_input_shape() -> None:
    cfg = tiny_config()
    scheduler = MaskedDiffusionScheduler(cfg)
    input_ids = torch.tensor(
        [
            [cfg.bos_token_id, 4, 5],
            [cfg.bos_token_id, 6, cfg.pad_token_id],
        ],
        dtype=torch.long,
    )

    scalar = scheduler.timestep_grid(1, input_ids)
    per_row = scheduler.timestep_grid(torch.tensor([0, 3]), input_ids)
    per_token = scheduler.timestep_grid(torch.tensor([[0, 1, 2], [3, 2, 1]]), input_ids)

    assert scalar.shape == input_ids.shape
    assert scalar.eq(1).all()
    assert per_row.tolist() == [[0, 0, 0], [3, 3, 3]]
    assert per_token.tolist() == [[0, 1, 2], [3, 2, 1]]


@pytest.mark.parametrize(
    ("bad_input", "message"),
    [
        (torch.ones((1, 3), dtype=torch.float32), "integer tensor"),
        (torch.empty((0, 3), dtype=torch.long), "batch size"),
        (torch.empty((1, 0), dtype=torch.long), "sequence length"),
        (torch.tensor([[1, 32]], dtype=torch.long), "outside"),
    ],
)
def test_scheduler_rejects_invalid_token_tensors(
    bad_input: torch.Tensor,
    message: str,
) -> None:
    scheduler = MaskedDiffusionScheduler(tiny_config())

    with pytest.raises(ValueError, match=message):
        scheduler.add_noise(bad_input, timestep=0)


def test_diffusion_training_batch_labels_only_masked_tokens() -> None:
    cfg = tiny_config()
    input_ids = torch.tensor([[cfg.bos_token_id, 4, 5, 6]], dtype=torch.long)

    batch = make_diffusion_training_batch(
        input_ids,
        cfg,
        torch=torch,
        timesteps=torch.tensor([cfg.diffusion_steps - 1]),
        generator=torch.Generator().manual_seed(1),
    )

    assert batch.masked_token_count.item() == batch.mask.sum().item()
    assert batch.noisy_input_ids[batch.mask].eq(cfg.mask_token_id).all()
    assert batch.labels[batch.mask].eq(input_ids[batch.mask]).all()
    assert batch.labels[~batch.mask].eq(-100).all()


def test_rao_blackwellized_diffusion_matches_masked_objective_at_full_mask() -> None:
    torch.manual_seed(1234)
    cfg = tiny_config()
    masked_cfg = AlexandrosConfig.from_dict(cfg.to_dict())
    rb_cfg = AlexandrosConfig.from_dict(cfg.to_dict())
    rb_cfg.diffusion_objective = "rao_blackwellized"
    masked = AlexandrosForDiffusionLM(masked_cfg).eval()
    rb = AlexandrosForDiffusionLM(rb_cfg).eval()
    rb.load_state_dict(masked.state_dict())
    input_ids = torch.tensor(
        [
            [cfg.bos_token_id, 4, 5, 6],
            [cfg.bos_token_id, 7, 8, cfg.pad_token_id],
        ],
        dtype=torch.long,
    )
    timestep = torch.tensor([cfg.diffusion_steps - 1, cfg.diffusion_steps - 1])

    with torch.no_grad():
        masked_out = masked.diffusion_loss(input_ids, timestep=timestep)
        rb_out = rb.diffusion_loss(input_ids, timestep=timestep)

    assert masked_out.loss is not None
    assert rb_out.loss is not None
    assert rb_out.logits.shape == masked_out.logits.shape
    assert torch.allclose(rb_out.logits, masked_out.logits, atol=1e-6)
    assert torch.allclose(rb_out.loss, masked_out.loss, atol=1e-6)


def test_rao_blackwellized_diffusion_chunking_is_loss_equivalent() -> None:
    torch.manual_seed(5678)
    cfg = tiny_config()
    unchunked_cfg = AlexandrosConfig.from_dict(cfg.to_dict())
    chunked_cfg = AlexandrosConfig.from_dict(cfg.to_dict())
    unchunked_cfg.diffusion_objective = "rao_blackwellized"
    chunked_cfg.diffusion_objective = "rao_blackwellized"
    chunked_cfg.diffusion_rb_chunk_size = 2
    unchunked = AlexandrosForDiffusionLM(unchunked_cfg).eval()
    chunked = AlexandrosForDiffusionLM(chunked_cfg).eval()
    chunked.load_state_dict(unchunked.state_dict())
    input_ids = torch.tensor(
        [[cfg.bos_token_id, 4, 5, 6, cfg.pad_token_id]],
        dtype=torch.long,
    )
    timestep = torch.tensor([1])

    with torch.no_grad():
        unchunked_out = unchunked.diffusion_loss(
            input_ids,
            timestep=timestep,
            generator=torch.Generator().manual_seed(99),
        )
        chunked_out = chunked.diffusion_loss(
            input_ids,
            timestep=timestep,
            generator=torch.Generator().manual_seed(99),
        )

    assert unchunked_out.loss is not None
    assert chunked_out.loss is not None
    assert torch.allclose(chunked_out.loss, unchunked_out.loss, atol=1e-6)


def test_bidirectional_diffusion_masked_logits_depend_on_right_context() -> None:
    torch.manual_seed(123)
    cfg = tiny_config()
    cfg.num_hidden_layers = 1
    cfg.attention_layer_indices = [0]
    cfg.diffusion_attention_mask_mode = "bidirectional"
    bidirectional = AlexandrosForDiffusionLM(cfg).eval()
    causal_cfg = AlexandrosConfig.from_dict(cfg.to_dict())
    causal_cfg.diffusion_attention_mask_mode = "causal"
    causal = AlexandrosForDiffusionLM(causal_cfg).eval()
    causal.load_state_dict(bidirectional.state_dict())

    input_a = torch.tensor(
        [[cfg.bos_token_id, 8, cfg.mask_token_id, 10]], dtype=torch.long
    )
    input_b = torch.tensor(
        [[cfg.bos_token_id, 8, cfg.mask_token_id, 11]], dtype=torch.long
    )

    with torch.no_grad():
        causal_a = causal(input_a, diffusion_timestep=0).logits[:, 2, :]
        causal_b = causal(input_b, diffusion_timestep=0).logits[:, 2, :]
        bidir_a = bidirectional(input_a, diffusion_timestep=0).logits[:, 2, :]
        bidir_b = bidirectional(input_b, diffusion_timestep=0).logits[:, 2, :]

    assert torch.allclose(causal_a, causal_b, atol=1e-6)
    assert not torch.allclose(bidir_a, bidir_b, atol=1e-6)


def test_block_diffusion_training_batch_keeps_mask_contiguous() -> None:
    cfg = tiny_config()
    input_ids = torch.tensor([[cfg.bos_token_id, 4, 5, 6, 7]], dtype=torch.long)

    batch = make_block_diffusion_training_batch(
        input_ids,
        cfg,
        torch=torch,
        block_size=2,
        timesteps=torch.tensor([1]),
        generator=torch.Generator().manual_seed(2),
    )

    start = int(batch.block_starts[0].item())
    expected_span = torch.zeros_like(batch.mask)
    expected_span[:, start : start + batch.block_size] = True
    assert not batch.mask[~expected_span].any()
    assert batch.mask[:, start : start + batch.block_size].any()
