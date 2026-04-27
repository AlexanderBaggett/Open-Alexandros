from __future__ import annotations

import pytest
import torch

from alexandros import AlexandrosConfig, AlexandrosForDiffusionLM
from alexandros.inference import (
    GenerationRequest,
    generate_from_request,
    reorder_generation_cache,
)
from alexandros.modeling_alexandros import GenerationMode


def tiny_config() -> AlexandrosConfig:
    return AlexandrosConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=64,
        linear_attention_ratio=2,
        moe_num_experts=4,
        moe_num_shared_experts=1,
        moe_top_k=2,
        moe_expert_hidden_size=24,
        kv_lora_rank=8,
        latent_dim=16,
        latent_slots=3,
        diffusion_steps=3,
        mask_token_id=3,
        ttt_rank=4,
    )


def test_generation_cache_reorder_rejects_non_integer_beam_indices() -> None:
    cache = [{"c_kv": torch.randn(2, 3, 4)}]
    states = [torch.randn(2, 4)]

    for bad_beam_idx in (torch.tensor([0.0, 1.0]), torch.tensor([True, False])):
        with pytest.raises(ValueError, match="integer tensor"):
            reorder_generation_cache(
                past_key_values=cache,
                past_ssm_states=states,
                beam_idx=bad_beam_idx,
            )


def test_generation_cache_reorder_preserves_batch_ordering() -> None:
    cache = [
        {
            "c_kv": torch.arange(24).view(2, 3, 4),
            "k_rope": torch.arange(12).view(2, 3, 2),
        }
    ]
    states = [torch.arange(8).view(2, 4)]

    reordered_cache, reordered_states = reorder_generation_cache(
        past_key_values=cache,
        past_ssm_states=states,
        beam_idx=torch.tensor([1, 0], dtype=torch.long),
    )

    assert reordered_cache is not None
    assert reordered_states is not None
    assert torch.equal(reordered_cache[0]["c_kv"][0], cache[0]["c_kv"][1])
    assert torch.equal(reordered_cache[0]["k_rope"][0], cache[0]["k_rope"][1])
    assert torch.equal(reordered_states[0][0], states[0][1])


def test_generation_request_blocks_invalid_mode_controls() -> None:
    with pytest.raises(ValueError, match="use_cache"):
        GenerationRequest(
            input_ids=[[1, 4]],
            mode=GenerationMode.BLOCK_DIFFUSION,
            use_cache=True,
        )
    with pytest.raises(ValueError, match="sampling controls"):
        GenerationRequest(
            input_ids=[[1, 4]],
            mode=GenerationMode.LATENT_REASONING,
            do_sample=True,
        )


def test_generate_from_request_accepts_objects_and_dicts() -> None:
    cfg = tiny_config()
    model = AlexandrosForDiffusionLM(cfg).eval()
    request = GenerationRequest(
        input_ids=[[cfg.bos_token_id, 8, 9]],
        max_new_tokens=2,
        mode=GenerationMode.AUTOREGRESSIVE,
    )

    with torch.no_grad():
        object_output = generate_from_request(model, request)
        dict_output = generate_from_request(model, request.to_dict())

    assert object_output.shape == (1, 5)
    assert dict_output.shape == (1, 5)
    assert torch.equal(object_output[:, :3], torch.tensor([[cfg.bos_token_id, 8, 9]]))
    assert torch.equal(dict_output[:, :3], torch.tensor([[cfg.bos_token_id, 8, 9]]))
