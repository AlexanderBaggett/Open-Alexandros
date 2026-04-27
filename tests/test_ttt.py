from __future__ import annotations

import pytest
import torch

from alexandros import AlexandrosConfig, AlexandrosForCausalLM
from alexandros.ttt import (
    TTTMetaAdapter,
    TTTState,
    ttt_next_token_loss_from_logits,
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


def test_ttt_state_updates_and_applies_without_mutating_inputs() -> None:
    cfg = tiny_config()
    state = TTTState.from_config(cfg, torch.device("cpu"), torch.float32)
    hidden = torch.randn(2, 3, cfg.hidden_size)
    before_hidden = hidden.clone()

    state.prefill_update(hidden, lr=1e-3)
    adapted = state.apply(hidden, scale=0.5)

    assert state.steps == 1
    assert adapted.shape == hidden.shape
    assert torch.isfinite(adapted).all()
    assert torch.equal(hidden, before_hidden)


def test_ttt_state_reset_and_local_generator_are_request_scoped() -> None:
    cfg = tiny_config()
    hidden = torch.randn(2, 3, cfg.hidden_size)
    generator_a = torch.Generator().manual_seed(77)
    generator_b = torch.Generator().manual_seed(77)
    first = TTTState.from_config(cfg, torch.device("cpu"), torch.float32)
    second = TTTState.from_config(cfg, torch.device("cpu"), torch.float32)

    global_before = torch.get_rng_state()
    first.prefill_update(hidden, lr=1e-3, generator=generator_a)
    global_after = torch.get_rng_state()
    second.update(hidden, lr=1e-3, generator=generator_b)

    assert torch.equal(global_before, global_after)
    assert torch.allclose(first.fast_a, second.fast_a)
    assert torch.allclose(first.fast_b, second.fast_b)
    assert first.steps == 1

    first.reset()

    assert first.steps == 0
    assert torch.count_nonzero(first.fast_a).item() == 0
    assert torch.count_nonzero(first.fast_b).item() == 0


def test_ttt_apply_supports_request_local_gate() -> None:
    state = TTTState(16, 2, torch.device("cpu"), torch.float32)
    hidden = torch.randn(1, 3, 16)
    state.prefill_update(hidden, lr=0.1, generator=torch.Generator().manual_seed(3))

    ungated = state.apply(hidden)
    zero_gated = state.apply(hidden, gate=torch.zeros(1, 3))
    half_gated = state.apply(hidden, gate=torch.full((1, 3, 1), 0.5))

    assert torch.equal(zero_gated, hidden)
    torch.testing.assert_close(half_gated - hidden, (ungated - hidden) * 0.5)


def test_ttt_state_can_clone_learned_fast_weights_without_sharing() -> None:
    adapter = TTTMetaAdapter(16, 2, init_std=1e-3)
    state = adapter.request_state()

    state.fast_a.add_(1.0)
    state.fast_b.sub_(1.0)

    assert state.fast_a.shape == adapter.init_a.shape
    assert state.fast_b.shape == adapter.init_b.shape
    assert not torch.allclose(state.fast_a, adapter.init_a.detach())
    assert not torch.allclose(state.fast_b, adapter.init_b.detach())


def test_ttt_meta_adapter_inner_update_is_differentiable() -> None:
    torch.manual_seed(7)
    cfg = tiny_config()
    model = AlexandrosForCausalLM(cfg).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = TTTMetaAdapter.from_config(cfg, init_std=1e-3)
    input_ids = torch.randint(4, cfg.vocab_size, (2, 6))
    hidden = model.model(input_ids).last_hidden_state.detach()
    prefix_hidden = hidden[:, :3, :]
    prefix_ids = input_ids[:, :3]
    heldout_hidden = hidden[:, 3:, :]
    heldout_ids = input_ids[:, 3:]
    fast_a, fast_b = adapter.initial_fast_weights()

    update = adapter.inner_update(
        prefix_hidden,
        prefix_ids,
        model.lm_head,
        fast_a,
        fast_b,
        inner_lr=1e-2,
        create_graph=True,
    )
    adapted = adapter.apply(heldout_hidden, update.fast_a, update.fast_b)
    outer_loss = ttt_next_token_loss_from_logits(model.lm_head(adapted), heldout_ids)
    outer_loss.backward()

    assert torch.isfinite(update.loss)
    assert torch.isfinite(update.grad_norm)
    assert update.grad_norm.item() > 0.0
    assert not torch.allclose(update.fast_a.detach(), adapter.init_a.detach())
    assert not torch.allclose(update.fast_b.detach(), adapter.init_b.detach())
    assert adapter.init_a.grad is not None
    assert adapter.init_b.grad is not None
    assert torch.isfinite(adapter.init_a.grad).all()
    assert torch.isfinite(adapter.init_b.grad).all()
    assert adapter.init_a.grad.norm().item() > 0.0
    assert adapter.init_b.grad.norm().item() > 0.0


def test_ttt_many_updates_stay_finite_without_mutating_checkpoint_weights() -> None:
    torch.manual_seed(123)
    cfg = tiny_config()
    model = AlexandrosForCausalLM(cfg).eval()
    input_ids = torch.randint(4, cfg.vocab_size, (1, 12))
    before_params = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }
    with torch.no_grad():
        hidden = model.model(input_ids).last_hidden_state.detach()
    before_hidden = hidden.clone()
    state = TTTState.from_config(cfg, torch.device("cpu"), torch.float32)
    generator = torch.Generator().manual_seed(55)

    for step in range(64):
        chunk = torch.roll(hidden, shifts=step % hidden.size(1), dims=1)
        state.update(chunk, lr=0.1, decay=0.95, generator=generator)
    adapted = state.apply(hidden, scale=0.5)

    drift_ratio = (adapted - hidden).norm() / hidden.norm().clamp_min(1e-6)
    assert state.steps == 64
    assert torch.isfinite(state.fast_a).all()
    assert torch.isfinite(state.fast_b).all()
    assert torch.isfinite(adapted).all()
    assert drift_ratio.item() > 0.0
    assert drift_ratio.item() < 0.25
    assert torch.equal(hidden, before_hidden)
    for name, param in model.named_parameters():
        assert torch.equal(param, before_params[name])


def test_ttt_fast_weight_state_size_is_sequence_independent() -> None:
    state = TTTState(16, 2, torch.device("cpu"), torch.float32)
    fast_weight_numel = state.fast_a.numel() + state.fast_b.numel()
    generator = torch.Generator().manual_seed(101)

    for seq_len in (4, 16, 64):
        hidden = torch.randn(1, seq_len, 16)
        state.update(hidden, lr=1e-3, generator=generator)
        adapted_tail = state.apply(hidden[:, -4:, :])
        assert state.fast_a.numel() + state.fast_b.numel() == fast_weight_numel
        assert adapted_tail.shape == (1, min(4, seq_len), 16)
        assert torch.isfinite(adapted_tail).all()


def test_ttt_chunked_32k_prefill_acceptance() -> None:
    state = TTTState(8, 2, torch.device("cpu"), torch.float32)
    hidden = torch.randn(1, 32_768, 8)
    generator = torch.Generator().manual_seed(202)

    for start in range(0, hidden.size(1), 1024):
        state.update(
            hidden[:, start : start + 1024, :],
            lr=1e-3,
            decay=0.99,
            generator=generator,
        )
    adapted_tail = state.apply(hidden[:, -128:, :])

    assert state.steps == 32
    assert torch.isfinite(state.fast_a).all()
    assert torch.isfinite(state.fast_b).all()
    assert torch.isfinite(adapted_tail).all()
    assert state.fast_a.numel() + state.fast_b.numel() == 32


@pytest.mark.parametrize(
    ("hidden_size", "rank", "dtype", "message"),
    [
        (0, 1, torch.float32, "hidden_size"),
        (4, 0, torch.float32, "rank"),
        (4, 5, torch.float32, "rank"),
        (4, 1, torch.long, "dtype"),
    ],
)
def test_ttt_state_constructor_rejects_invalid_values(
    hidden_size: int,
    rank: int,
    dtype: torch.dtype,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TTTState(hidden_size, rank, torch.device("cpu"), dtype)


@pytest.mark.parametrize(
    ("bad_hidden", "message"),
    [
        (torch.ones(2, 16), "shape"),
        (torch.empty(0, 1, 16), "batch size"),
        (torch.empty(1, 0, 16), "sequence length"),
        (torch.ones(1, 2, 15), "hidden_size"),
        (torch.ones(1, 2, 16, dtype=torch.float64), "dtype"),
        (torch.full((1, 2, 16), float("nan")), "finite"),
    ],
)
def test_ttt_state_rejects_invalid_hidden_states(
    bad_hidden: torch.Tensor,
    message: str,
) -> None:
    state = TTTState(16, 2, torch.device("cpu"), torch.float32)

    with pytest.raises(ValueError, match=message):
        state.update(bad_hidden)
    with pytest.raises(ValueError, match=message):
        state.apply(bad_hidden)


def test_ttt_state_rejects_invalid_update_controls() -> None:
    state = TTTState(16, 2, torch.device("cpu"), torch.float32)
    hidden = torch.randn(1, 2, 16)

    with pytest.raises(ValueError, match="lr"):
        state.update(hidden, lr=0.0)
    with pytest.raises(ValueError, match="lr"):
        state.update(hidden, lr=float("nan"))
    with pytest.raises(ValueError, match="scale"):
        state.apply(hidden, scale=float("inf"))
    with pytest.raises(ValueError, match="decay"):
        state.prefill_update(hidden, decay=1.0)


@pytest.mark.parametrize(
    ("bad_gate", "message"),
    [
        (torch.ones(1, 2, dtype=torch.float64), "dtype"),
        (torch.ones(1, 2, 2), "gate must be a scalar"),
        (torch.full((1, 2), float("nan")), "finite"),
        (torch.full((1, 2), -0.1), r"\[0, 1\]"),
    ],
)
def test_ttt_state_rejects_invalid_apply_gates(
    bad_gate: torch.Tensor,
    message: str,
) -> None:
    state = TTTState(16, 2, torch.device("cpu"), torch.float32)
    hidden = torch.randn(1, 2, 16)

    with pytest.raises(ValueError, match=message):
        state.apply(hidden, gate=bad_gate)
