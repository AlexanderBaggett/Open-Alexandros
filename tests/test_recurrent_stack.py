from __future__ import annotations

import pytest
import torch

from alexandros import AlexandrosConfig, AlexandrosForCausalLM


def tiny_config() -> AlexandrosConfig:
    return AlexandrosConfig(
        vocab_size=48,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=32,
        linear_attention_ratio=2,
        moe_num_experts=2,
        moe_num_shared_experts=1,
        moe_top_k=1,
        moe_expert_hidden_size=16,
        kv_lora_rank=8,
        latent_dim=12,
        latent_slots=2,
        diffusion_steps=3,
        ttt_rank=4,
        depth_lora_rank=4,
    )


def test_default_stack_remains_uniform() -> None:
    cfg = tiny_config()

    assert not cfg.uses_recurrent_stack
    assert cfg.recurrent_layer_range == (0, 0)

    model = AlexandrosForCausalLM(cfg).eval()
    input_ids = torch.randint(4, cfg.vocab_size, (2, 5))
    output = model(input_ids, output_hidden_states=True)

    assert output.logits.shape == (2, 5, cfg.vocab_size)
    assert output.hidden_states is not None
    assert len(output.hidden_states) == cfg.num_hidden_layers + 1
    assert model.model.last_stack_stats["uses_recurrent_stack"] is False
    assert model.model.last_stack_stats["layer_executions"] == cfg.num_hidden_layers


def test_staged_recurrent_stack_repeats_recurrent_layers() -> None:
    cfg = tiny_config()
    cfg.prelude_layers = 1
    cfg.recurrent_layers = 2
    cfg.coda_layers = 1
    cfg.recurrent_depth = 3
    cfg.use_loop_index_embeddings = True
    model = AlexandrosForCausalLM(cfg).eval()
    input_ids = torch.randint(4, cfg.vocab_size, (2, 5))

    output = model(input_ids, output_hidden_states=True)

    expected_executions = (
        cfg.prelude_layers
        + (cfg.recurrent_layers * cfg.recurrent_depth)
        + cfg.coda_layers
    )
    assert output.logits.shape == (2, 5, cfg.vocab_size)
    assert output.hidden_states is not None
    assert len(output.hidden_states) == expected_executions + 1
    assert model.model.last_stack_stats == {
        "uses_recurrent_stack": True,
        "recurrent_depth": cfg.recurrent_depth,
        "recurrent_loop_count": cfg.recurrent_depth,
        "layer_executions": expected_executions,
        "loop_index_embeddings_applied": True,
    }


def test_staged_recurrent_stack_disables_loop_embeddings_for_diffusion() -> None:
    cfg = tiny_config()
    cfg.prelude_layers = 1
    cfg.recurrent_layers = 2
    cfg.coda_layers = 1
    cfg.recurrent_depth = 2
    cfg.use_loop_index_embeddings = True
    model = AlexandrosForCausalLM(cfg).eval()
    input_ids = torch.randint(4, cfg.vocab_size, (1, 5))

    output = model(input_ids, diffusion_timestep=torch.tensor([1]))

    assert torch.isfinite(output.logits).all()
    assert model.model.last_stack_stats["loop_index_embeddings_applied"] is False


def test_staged_recurrent_stack_rejects_cache_reuse_when_depth_repeats() -> None:
    cfg = tiny_config()
    cfg.prelude_layers = 1
    cfg.recurrent_layers = 2
    cfg.coda_layers = 1
    cfg.recurrent_depth = 2
    model = AlexandrosForCausalLM(cfg).eval()
    input_ids = torch.randint(4, cfg.vocab_size, (1, 5))

    with pytest.raises(ValueError, match="cache reuse"):
        model(input_ids, use_cache=True)
    with pytest.raises(ValueError, match="cache reuse"):
        model.generate(input_ids, max_new_tokens=1, use_cache=True)


def test_staged_recurrent_stack_depth_one_can_use_cache() -> None:
    cfg = tiny_config()
    cfg.prelude_layers = 1
    cfg.recurrent_layers = 2
    cfg.coda_layers = 1
    cfg.recurrent_depth = 1
    model = AlexandrosForCausalLM(cfg).eval()
    input_ids = torch.randint(4, cfg.vocab_size, (1, 5))

    output = model(input_ids, use_cache=True)

    assert output.past_key_values is not None
    assert output.past_ssm_states is not None
    assert len(output.past_key_values) == cfg.num_hidden_layers
    assert len(output.past_ssm_states) == cfg.num_hidden_layers


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"prelude_layers": 1, "recurrent_layers": 1, "coda_layers": 1}, "must equal"),
        ({"prelude_layers": 2, "recurrent_layers": 0, "coda_layers": 2}, "> 0"),
        ({"prelude_layers": -1}, "prelude_layers"),
        ({"recurrent_depth": 0}, "recurrent_depth"),
        ({"use_loop_index_embeddings": 1}, "use_loop_index_embeddings"),
    ],
)
def test_staged_recurrent_stack_config_validation(
    updates: dict[str, object],
    message: str,
) -> None:
    data = tiny_config().to_dict()
    data.update(updates)

    with pytest.raises(ValueError, match=message):
        AlexandrosConfig.from_dict(data)
