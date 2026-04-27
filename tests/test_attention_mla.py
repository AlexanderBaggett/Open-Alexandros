from __future__ import annotations

import pytest
import torch

from alexandros import AlexandrosConfig, AlexandrosForCausalLM
from alexandros.attention_mla import MLAAttention
from alexandros.kv_cache import TurboQuantPacket


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


def tiny_paper_mla_config() -> AlexandrosConfig:
    cfg = tiny_config()
    cfg.mla_rope_dim = 2
    return AlexandrosConfig.from_dict(cfg.to_dict())


@pytest.mark.parametrize(
    ("mla_rope_dim", "message"),
    [
        (-1, "mla_rope_dim"),
        (1, "even"),
        (4, "smaller than head_dim"),
        (True, "mla_rope_dim"),
    ],
)
def test_config_rejects_invalid_mla_rope_dim(
    mla_rope_dim: object,
    message: str,
) -> None:
    data = tiny_config().to_dict()
    data["mla_rope_dim"] = mla_rope_dim

    with pytest.raises(ValueError, match=message):
        AlexandrosConfig.from_dict(data)


def test_mla_cached_and_uncached_outputs_match_in_eval_mode() -> None:
    cfg = tiny_config()
    attention = MLAAttention(cfg).eval()
    x = torch.randn(2, 4, cfg.hidden_size)

    full, _ = attention(x, use_cache=False)
    prefix, cache = attention(x[:, :3], use_cache=True)
    suffix, next_cache = attention(
        x[:, 3:],
        past_key_value=cache,
        use_cache=True,
        attention_mask=torch.ones(2, 4, dtype=torch.bool),
    )

    assert prefix.shape == (2, 3, cfg.hidden_size)
    assert next_cache is not None
    assert torch.allclose(full[:, 3:], suffix, atol=1e-5, rtol=1e-5)


def test_mla_paper_cache_layout_stores_separate_rope_keys() -> None:
    cfg = tiny_paper_mla_config()
    attention = MLAAttention(cfg).eval()
    x = torch.randn(2, 4, cfg.hidden_size)

    full, _ = attention(x, use_cache=False)
    prefix, cache = attention(x[:, :3], use_cache=True)
    suffix, next_cache = attention(
        x[:, 3:],
        past_key_value=cache,
        use_cache=True,
        attention_mask=torch.ones(2, 4, dtype=torch.bool),
    )

    assert prefix.shape == (2, 3, cfg.hidden_size)
    assert cache is not None
    assert next_cache is not None
    assert cache["c_kv"].shape == (2, 3, cfg.mla_d_c)
    assert cache["k_rope"].shape == (2, 3, cfg.mla_d_r)
    assert next_cache["c_kv"].shape == (2, 4, cfg.mla_d_c)
    assert next_cache["k_rope"].shape == (2, 4, cfg.mla_d_r)
    assert torch.allclose(full[:, 3:], suffix, atol=1e-5, rtol=1e-5)


def test_mla_turboquant_compresses_only_c_kv_with_rope_cache() -> None:
    cfg = tiny_paper_mla_config()
    cfg.use_turboquant_cache = True
    attention = MLAAttention(AlexandrosConfig.from_dict(cfg.to_dict())).eval()
    x = torch.randn(1, 3, cfg.hidden_size)

    _, cache = attention(x, use_cache=True)

    assert cache is not None
    assert isinstance(cache["c_kv_packet"], TurboQuantPacket)
    assert "c_kv" not in cache
    assert cache["k_rope"].shape == (1, 3, cfg.mla_d_r)


def test_mla_paper_cache_layout_matches_cached_model_logits() -> None:
    cfg = AlexandrosConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=32,
        linear_attention_ratio=1,
        attention_layer_indices=[0],
        moe_num_experts=2,
        moe_num_shared_experts=1,
        moe_top_k=1,
        moe_expert_hidden_size=16,
        kv_lora_rank=4,
        mla_rope_dim=2,
        latent_dim=8,
        latent_slots=2,
        diffusion_steps=4,
        mask_token_id=3,
        ttt_rank=2,
    )
    model = AlexandrosForCausalLM(cfg).eval()
    input_ids = torch.tensor([[cfg.bos_token_id, 4, 5, cfg.eos_token_id]])

    full = model(input_ids, use_cache=False)
    prefix = model(input_ids[:, :3], use_cache=True)
    suffix = model(
        input_ids[:, 3:],
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
        past_key_values=prefix.past_key_values,
        past_ssm_states=prefix.past_ssm_states,
        use_cache=True,
    )

    assert torch.allclose(full.logits[:, 3:], suffix.logits, atol=1e-5, rtol=1e-5)


def test_mla_sdpa_backend_matches_eager_reference_in_eval_mode() -> None:
    eager_cfg = tiny_config()
    sdpa_cfg = tiny_config()
    sdpa_cfg.attention_backend = "sdpa"
    eager = MLAAttention(eager_cfg).eval()
    sdpa = MLAAttention(sdpa_cfg).eval()
    sdpa.load_state_dict(eager.state_dict())
    x = torch.randn(2, 5, eager_cfg.hidden_size)
    attention_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ],
        dtype=torch.bool,
    )

    eager_full, _ = eager(x, attention_mask=attention_mask)
    sdpa_full, _ = sdpa(x, attention_mask=attention_mask)
    eager_prefix, eager_cache = eager(x[:, :3], use_cache=True)
    sdpa_prefix, sdpa_cache = sdpa(x[:, :3], use_cache=True)
    eager_suffix, _ = eager(
        x[:, 3:],
        attention_mask=attention_mask,
        past_key_value=eager_cache,
        use_cache=True,
    )
    sdpa_suffix, _ = sdpa(
        x[:, 3:],
        attention_mask=attention_mask,
        past_key_value=sdpa_cache,
        use_cache=True,
    )

    assert torch.allclose(sdpa_full, eager_full, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sdpa_prefix, eager_prefix, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sdpa_suffix, eager_suffix, atol=1e-5, rtol=1e-5)


def test_mla_flash_backend_requires_cuda_on_cpu() -> None:
    if torch.cuda.is_available():
        pytest.skip("CPU flash-backend error contract only")
    cfg = tiny_config()
    cfg.attention_backend = "flash"
    attention = MLAAttention(cfg).eval()
    x = torch.randn(1, 2, cfg.hidden_size)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        attention(x)


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
def test_mla_rejects_invalid_hidden_states(
    bad_hidden: torch.Tensor,
    message: str,
) -> None:
    attention = MLAAttention(tiny_config())

    with pytest.raises(ValueError, match=message):
        attention(bad_hidden)


@pytest.mark.parametrize(
    ("bad_mask", "message"),
    [
        (torch.ones(2, 3), "batch size"),
        (torch.ones(1, 3), "length"),
        (torch.tensor([[1.0, float("nan")]]), "finite"),
        (torch.tensor([[1.0, 0.5]]), "0/1"),
        (torch.zeros(1, 2, dtype=torch.bool), "non-pad"),
    ],
)
def test_mla_rejects_invalid_attention_masks(
    bad_mask: torch.Tensor,
    message: str,
) -> None:
    cfg = tiny_config()
    attention = MLAAttention(cfg)
    x = torch.randn(1, 2, cfg.hidden_size)

    with pytest.raises(ValueError, match=message):
        attention(x, attention_mask=bad_mask)


@pytest.mark.parametrize(
    ("cache", "message"),
    [
        ({"c_kv": torch.randn(1, 4)}, "shape"),
        ({"c_kv": torch.randn(2, 1, 4)}, "batch size"),
        ({"c_kv": torch.empty(1, 0, 4)}, "sequence length"),
        ({"c_kv": torch.randn(1, 1, 5)}, "kv_lora_rank"),
        ({"c_kv": torch.ones(1, 1, 4, dtype=torch.long)}, "floating-point"),
        ({"c_kv": torch.full((1, 1, 4), float("inf"))}, "finite"),
    ],
)
def test_mla_rejects_invalid_cache_tensors(
    cache: dict[str, torch.Tensor],
    message: str,
) -> None:
    cfg = tiny_config()
    attention = MLAAttention(cfg)
    x = torch.randn(1, 1, cfg.hidden_size)

    with pytest.raises(ValueError, match=message):
        attention(x, past_key_value=cache)


def test_mla_paper_cache_layout_rejects_missing_or_bad_rope_cache() -> None:
    cfg = tiny_paper_mla_config()
    attention = MLAAttention(cfg)
    x = torch.randn(1, 1, cfg.hidden_size)

    with pytest.raises(ValueError, match="k_rope"):
        attention(x, past_key_value={"c_kv": torch.randn(1, 1, cfg.kv_lora_rank)})
    with pytest.raises(ValueError, match="sequence length"):
        attention(
            x,
            past_key_value={
                "c_kv": torch.randn(1, 1, cfg.kv_lora_rank),
                "k_rope": torch.randn(1, 2, cfg.mla_d_r),
            },
        )
    with pytest.raises(ValueError, match="c_kv"):
        attention(x, past_key_value={"k_rope": torch.randn(1, 1, cfg.mla_d_r)})


def test_mla_rejects_cache_for_non_causal_attention() -> None:
    attention = MLAAttention(tiny_config())
    x = torch.randn(1, 2, attention.hidden_size)

    with pytest.raises(ValueError, match="non-causal"):
        attention(x, use_cache=True, is_causal=False)
