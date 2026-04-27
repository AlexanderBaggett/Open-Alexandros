from __future__ import annotations

import pytest
import torch
from torch import nn

from alexandros import AlexandrosConfig, AlexandrosModel
from alexandros.ssm_gated_deltanet import GatedDeltaNetBlock
from alexandros.ssm_mamba2 import Mamba2Block
from alexandros.ssm_matrix_deltanet import MatrixGatedDeltaNetBlock


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


@pytest.mark.parametrize("block_cls", [GatedDeltaNetBlock, Mamba2Block])
def test_recurrent_mixer_state_contract_and_chunk_equivalence(
    block_cls: type[nn.Module],
) -> None:
    cfg = tiny_config()
    block = block_cls(cfg).eval()
    x = torch.randn(2, 5, cfg.hidden_size)

    full_y, full_state = block(x)
    first_y, first_state = block(x[:, :3])
    second_y, second_state = block(x[:, 3:], state=first_state)
    chunked_y = torch.cat([first_y, second_y], dim=1)

    assert block.recurrent_state_shape(batch_size=2) == (2, cfg.hidden_size)
    assert full_state.shape == (2, cfg.hidden_size)
    assert full_state.requires_grad is False
    assert torch.allclose(chunked_y, full_y, atol=1e-6)
    assert torch.allclose(second_state, full_state, atol=1e-6)


@pytest.mark.parametrize("block_cls", [GatedDeltaNetBlock, Mamba2Block])
def test_recurrent_mixer_attention_mask_preserves_state_on_masked_tokens(
    block_cls: type[nn.Module],
) -> None:
    cfg = tiny_config()
    block = block_cls(cfg).eval()
    x = torch.randn(1, 3, cfg.hidden_size)
    mask_first_only = torch.tensor([[1, 0, 0]], dtype=torch.long)

    masked_y, masked_state = block(x, attention_mask=mask_first_only)
    first_y, first_state = block(x[:, :1])

    assert torch.allclose(masked_y[:, :1], first_y, atol=1e-6)
    assert torch.count_nonzero(masked_y[:, 1:]).item() == 0
    assert torch.allclose(masked_state, first_state, atol=1e-6)


@pytest.mark.parametrize(
    "block_cls", [GatedDeltaNetBlock, MatrixGatedDeltaNetBlock, Mamba2Block]
)
@pytest.mark.parametrize(
    ("bad_x", "message"),
    [
        (torch.ones(2, 16), "shape"),
        (torch.empty(0, 1, 16), "batch size"),
        (torch.empty(1, 0, 16), "sequence length"),
        (torch.ones(1, 2, 15), "hidden_size"),
        (torch.ones(1, 2, 16, dtype=torch.long), "floating-point"),
        (torch.full((1, 2, 16), float("nan")), "finite"),
    ],
)
def test_recurrent_mixers_reject_invalid_hidden_states(
    block_cls: type[nn.Module],
    bad_x: torch.Tensor,
    message: str,
) -> None:
    block = block_cls(tiny_config())

    with pytest.raises(ValueError, match=message):
        block(bad_x)


@pytest.mark.parametrize("block_cls", [GatedDeltaNetBlock, Mamba2Block])
@pytest.mark.parametrize(
    ("bad_state", "message"),
    [
        (torch.ones(1, 15), "shape"),
        (torch.ones(1, 16, dtype=torch.float64), "dtype"),
        (torch.ones(1, 16, dtype=torch.long), "floating-point"),
        (torch.full((1, 16), float("inf")), "finite"),
    ],
)
def test_recurrent_mixers_reject_invalid_recurrent_state(
    block_cls: type[nn.Module],
    bad_state: torch.Tensor,
    message: str,
) -> None:
    cfg = tiny_config()
    block = block_cls(cfg)
    x = torch.randn(1, 2, cfg.hidden_size)

    with pytest.raises(ValueError, match=message):
        block(x, state=bad_state)


@pytest.mark.parametrize(
    "block_cls", [GatedDeltaNetBlock, MatrixGatedDeltaNetBlock, Mamba2Block]
)
@pytest.mark.parametrize(
    ("bad_mask", "message"),
    [
        (torch.ones(2, 2), "shape"),
        (torch.ones(1, 1), "shorter"),
        (torch.tensor([[1, 2]]), "0/1"),
        (torch.tensor([[1.0, float("nan")]]), "finite"),
    ],
)
def test_recurrent_mixers_reject_invalid_attention_mask(
    block_cls: type[nn.Module],
    bad_mask: torch.Tensor,
    message: str,
) -> None:
    cfg = tiny_config()
    block = block_cls(cfg)
    x = torch.randn(1, 2, cfg.hidden_size)

    with pytest.raises(ValueError, match=message):
        block(x, attention_mask=bad_mask)


@pytest.mark.parametrize(
    "block_cls", [GatedDeltaNetBlock, MatrixGatedDeltaNetBlock, Mamba2Block]
)
def test_recurrent_mixers_reject_invalid_state_shape_request(
    block_cls: type[nn.Module],
) -> None:
    block = block_cls(tiny_config())

    with pytest.raises(ValueError, match="batch_size"):
        block.recurrent_state_shape(0)


def test_model_uses_mamba2_backend_for_linear_mixer_layers() -> None:
    cfg = tiny_config()
    cfg.linear_mixer_backend = "mamba2"
    model = AlexandrosModel(cfg)

    linear_mixers = [
        block.mixer
        for block in model.layers
        if not cfg.is_attention_layer(block.layer_idx)
    ]

    assert linear_mixers
    assert all(isinstance(mixer, Mamba2Block) for mixer in linear_mixers)


def test_mamba2_backend_model_forward_and_cache_contract() -> None:
    cfg = tiny_config()
    cfg.linear_mixer_backend = "mamba2"
    model = AlexandrosModel(cfg).eval()
    input_ids = torch.tensor([[cfg.bos_token_id, 4, 5]])

    output = model(input_ids, use_cache=True)

    assert output.last_hidden_state.shape == (1, 3, cfg.hidden_size)
    assert output.past_ssm_states is not None
    for layer_idx, state in enumerate(output.past_ssm_states):
        if cfg.is_attention_layer(layer_idx):
            assert state is None
        else:
            assert state is not None
            assert state.shape == (1, cfg.hidden_size)
            assert state.requires_grad is False


def test_matrix_deltanet_state_contract_chunking_and_decode_equivalence() -> None:
    cfg = tiny_config()
    full_block = MatrixGatedDeltaNetBlock(cfg).eval()
    chunk_cfg = AlexandrosConfig.from_dict({**cfg.to_dict(), "deltanet_chunk_size": 2})
    chunked_block = MatrixGatedDeltaNetBlock(chunk_cfg).eval()
    chunked_block.load_state_dict(full_block.state_dict())
    x = torch.randn(2, 5, cfg.hidden_size)

    full_y, full_state = full_block(x)
    prefix_y, prefix_state = full_block(x[:, :3])
    suffix_y, suffix_state = full_block(x[:, 3:], state=prefix_state)
    chunked_y, chunked_state = chunked_block(x)

    decode_outputs = []
    decode_state = None
    for idx in range(x.size(1)):
        step_y, decode_state = full_block.decode_step(x[:, idx], state=decode_state)
        decode_outputs.append(step_y)
    decoded_y = torch.stack(decode_outputs, dim=1)

    assert full_block.recurrent_state_shape(batch_size=2) == (
        2,
        cfg.num_attention_heads,
        cfg.head_dim,
        cfg.head_dim,
    )
    assert full_state.shape == full_block.recurrent_state_shape(batch_size=2)
    assert full_state.requires_grad is False
    assert torch.allclose(torch.cat([prefix_y, suffix_y], dim=1), full_y, atol=1e-6)
    assert torch.allclose(suffix_state, full_state, atol=1e-6)
    assert torch.allclose(chunked_y, full_y, atol=1e-6)
    assert torch.allclose(chunked_state, full_state, atol=1e-6)
    assert torch.allclose(decoded_y, full_y, atol=1e-6)
    assert decode_state is not None
    assert torch.allclose(decode_state, full_state, atol=1e-6)


def test_matrix_deltanet_attention_mask_preserves_matrix_state() -> None:
    cfg = tiny_config()
    block = MatrixGatedDeltaNetBlock(cfg).eval()
    x = torch.randn(1, 3, cfg.hidden_size)
    mask_first_only = torch.tensor([[1, 0, 0]], dtype=torch.long)

    masked_y, masked_state = block(x, attention_mask=mask_first_only)
    first_y, first_state = block(x[:, :1])

    assert torch.allclose(masked_y[:, :1], first_y, atol=1e-6)
    assert torch.count_nonzero(masked_y[:, 1:]).item() == 0
    assert torch.allclose(masked_state, first_state, atol=1e-6)


@pytest.mark.parametrize(
    ("bad_state", "message"),
    [
        (torch.ones(1, 16), "shape"),
        (torch.ones(1, 4, 4, 4, dtype=torch.float64), "dtype"),
        (torch.ones(1, 4, 4, 4, dtype=torch.long), "floating-point"),
        (torch.full((1, 4, 4, 4), float("inf")), "finite"),
    ],
)
def test_matrix_deltanet_rejects_invalid_recurrent_state(
    bad_state: torch.Tensor,
    message: str,
) -> None:
    cfg = tiny_config()
    block = MatrixGatedDeltaNetBlock(cfg)
    x = torch.randn(1, 2, cfg.hidden_size)

    with pytest.raises(ValueError, match=message):
        block(x, state=bad_state)


def test_matrix_deltanet_rejects_invalid_decode_step_shape() -> None:
    block = MatrixGatedDeltaNetBlock(tiny_config())

    with pytest.raises(ValueError, match="decode_step"):
        block.decode_step(torch.randn(1, 1, block.hidden_size))


def test_model_uses_matrix_deltanet_backend_and_cache_contract() -> None:
    cfg = tiny_config()
    cfg.linear_mixer_backend = "matrix_deltanet"
    model = AlexandrosModel(cfg).eval()
    input_ids = torch.tensor([[cfg.bos_token_id, 4, 5]])

    output = model(input_ids, use_cache=True)
    next_output = model(
        torch.tensor([[6]]),
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
        past_key_values=output.past_key_values,
        past_ssm_states=output.past_ssm_states,
        use_cache=True,
    )

    linear_mixers = [
        block.mixer
        for block in model.layers
        if not cfg.is_attention_layer(block.layer_idx)
    ]
    assert linear_mixers
    assert all(isinstance(mixer, MatrixGatedDeltaNetBlock) for mixer in linear_mixers)
    assert output.past_ssm_states is not None
    assert next_output.past_ssm_states is not None
    for layer_idx, state in enumerate(output.past_ssm_states):
        if cfg.is_attention_layer(layer_idx):
            assert state is None
        else:
            assert state is not None
            assert state.shape == (
                1,
                cfg.num_attention_heads,
                cfg.head_dim,
                cfg.head_dim,
            )
            assert state.requires_grad is False
    assert next_output.last_hidden_state.shape == (1, 1, cfg.hidden_size)


def test_config_rejects_invalid_deltanet_controls() -> None:
    data = tiny_config().to_dict()
    data["deltanet_chunk_size"] = -1
    with pytest.raises(ValueError, match="deltanet_chunk_size"):
        AlexandrosConfig.from_dict(data)

    data = tiny_config().to_dict()
    data["deltanet_state_clip"] = 0.0
    with pytest.raises(ValueError, match="deltanet_state_clip"):
        AlexandrosConfig.from_dict(data)


def test_config_rejects_unknown_linear_mixer_backend() -> None:
    data = tiny_config().to_dict()
    data["linear_mixer_backend"] = "unknown"

    with pytest.raises(ValueError, match="linear_mixer_backend"):
        AlexandrosConfig.from_dict(data)
