from __future__ import annotations

import pytest
import torch

from alexandros.configuration_alexandros import AlexandrosConfig
from alexandros.hf_compat import (
    AlexandrosHFConfig,
    AlexandrosHFForCausalLM,
    register_alexandros_with_transformers,
    transformers_available,
)


def tiny_config() -> AlexandrosConfig:
    return AlexandrosConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=16,
        linear_attention_ratio=1,
        moe_num_experts=2,
        moe_num_shared_experts=1,
        moe_top_k=1,
        moe_expert_hidden_size=16,
        kv_lora_rank=8,
        latent_dim=8,
        latent_slots=2,
        diffusion_steps=2,
        depth_lora_rank=4,
        ttt_rank=4,
    )


def test_hf_compat_has_helpful_error_without_optional_dependency() -> None:
    if transformers_available():
        pytest.skip("transformers is installed; optional HF path is exercised directly")
    with pytest.raises(ImportError, match="pip install -e .*\\[hf\\]"):
        AlexandrosHFConfig()
    with pytest.raises(ImportError, match="pip install -e .*\\[hf\\]"):
        register_alexandros_with_transformers()


@pytest.mark.skipif(
    not transformers_available(),
    reason="optional transformers dependency is not installed",
)
def test_hf_config_round_trips_alexandros_config_when_available() -> None:
    cfg = tiny_config()
    hf_cfg = AlexandrosHFConfig.from_alexandros_config(cfg)
    assert hf_cfg.model_type == "alexandros"
    assert hf_cfg.to_alexandros_config().to_dict() == cfg.to_dict()


@pytest.mark.skipif(
    not transformers_available(),
    reason="optional transformers dependency is not installed",
)
def test_hf_causal_wrapper_forward_when_available() -> None:
    cfg = tiny_config()
    hf_cfg = AlexandrosHFConfig.from_alexandros_config(cfg)
    model = AlexandrosHFForCausalLM(hf_cfg).eval()
    input_ids = torch.randint(4, cfg.vocab_size, (2, 5))
    labels = input_ids.clone()
    output = model(input_ids=input_ids, labels=labels)
    assert output.logits.shape == (2, 5, cfg.vocab_size)
    assert output.loss is not None
    assert torch.isfinite(output.loss)


@pytest.mark.skipif(
    not transformers_available(),
    reason="optional transformers dependency is not installed",
)
def test_hf_auto_registration_when_available() -> None:
    register_alexandros_with_transformers()
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = tiny_config()
    config_kwargs = cfg.to_dict()
    config_kwargs.pop("model_type")
    hf_cfg = AutoConfig.for_model("alexandros", **config_kwargs)
    assert isinstance(hf_cfg, AlexandrosHFConfig)
    model = AutoModelForCausalLM.from_config(hf_cfg)
    assert isinstance(model, AlexandrosHFForCausalLM)


@pytest.mark.skipif(
    not transformers_available(),
    reason="optional transformers dependency is not installed",
)
def test_hf_native_save_load_round_trip_when_available(tmp_path) -> None:
    register_alexandros_with_transformers()
    from transformers import AutoModelForCausalLM

    cfg = tiny_config()
    model = AlexandrosHFForCausalLM(
        AlexandrosHFConfig.from_alexandros_config(cfg)
    ).eval()
    model.save_pretrained(tmp_path)

    restored = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    input_ids = torch.randint(4, cfg.vocab_size, (1, 4))
    original = model(input_ids=input_ids).logits
    loaded = restored(input_ids=input_ids).logits

    assert isinstance(restored, AlexandrosHFForCausalLM)
    assert torch.allclose(original, loaded)
