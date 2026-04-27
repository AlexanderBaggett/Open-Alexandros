from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from alexandros import AlexandrosConfig
from alexandros.training import (
    DistillationDataset,
    distillation_loss,
    iter_distillation_jsonl,
    objective_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def tiny_config() -> AlexandrosConfig:
    return AlexandrosConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=16,
        linear_attention_ratio=2,
        moe_num_experts=2,
        moe_num_shared_experts=1,
        moe_top_k=1,
        moe_expert_hidden_size=16,
        kv_lora_rank=4,
        latent_dim=8,
        latent_slots=2,
        diffusion_steps=3,
        mask_token_id=3,
        ttt_rank=2,
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_iter_distillation_jsonl_reads_token_and_logit_signals() -> None:
    cfg = tiny_config()
    path = ROOT / "tests" / "fixtures" / "distillation_tiny.jsonl"

    records = list(iter_distillation_jsonl(path, cfg))

    assert len(records) == 3
    assert records[0].input_ids == (1, 4, 5)
    assert records[0].teacher_token_ids == (4, 5, 2)
    assert records[0].teacher_logits is None
    assert records[1].teacher_token_ids is None
    assert records[1].teacher_logits is not None
    assert len(records[1].teacher_logits[0]) == cfg.vocab_size
    assert records[2].teacher_token_ids is not None
    assert records[2].teacher_logits is not None


def test_distillation_dataset_batches_mixed_teacher_signals_and_resume_state() -> None:
    cfg = tiny_config()
    path = ROOT / "tests" / "fixtures" / "distillation_tiny.jsonl"
    dataset = DistillationDataset.from_jsonl(path, cfg, seq_len=4)
    iterator = dataset.batch_iterator(
        batch_size=3,
        torch=torch,
        device=torch.device("cpu"),
        seed=11,
        shuffle=False,
    )

    batch = next(iterator)

    assert batch.input_ids.shape == (3, 4)
    assert torch.equal(batch.attention_mask[0], torch.tensor([1, 1, 1, 0]))
    assert batch.teacher_token_ids is not None
    assert batch.teacher_token_mask is not None
    assert batch.teacher_logits is not None
    assert batch.teacher_logits_mask is not None
    assert batch.teacher_token_ids.shape == (3, 4)
    assert batch.teacher_logits.shape == (3, 4, cfg.vocab_size)
    assert torch.equal(batch.teacher_token_mask[1], torch.zeros(4, dtype=torch.long))
    assert torch.equal(batch.teacher_logits_mask[0], torch.zeros(4, dtype=torch.long))
    assert torch.equal(batch.teacher_logits_mask[2], torch.tensor([1, 1, 1, 0]))

    iterator = dataset.batch_iterator(
        batch_size=1,
        torch=torch,
        device=torch.device("cpu"),
        seed=0,
        shuffle=False,
    )
    first = next(iterator)
    state = iterator.state_dict()
    expected = next(iterator)
    restored = dataset.batch_iterator(
        batch_size=1,
        torch=torch,
        device=torch.device("cpu"),
        seed=999,
        shuffle=True,
    )
    restored.load_state_dict(state)
    actual = next(restored)

    assert torch.equal(first.input_ids, torch.tensor([[1, 4, 5, cfg.pad_token_id]]))
    assert torch.equal(actual.input_ids, expected.input_ids)
    bad_state = dict(state)
    bad_state["cursor"] = True
    with pytest.raises(ValueError, match="cursor"):
        restored.load_state_dict(bad_state)


def test_distillation_jsonl_validation_rejects_malformed_records(tmp_path) -> None:
    cfg = tiny_config()
    path = tmp_path / "bad.jsonl"

    write_jsonl(path, [{"input_ids": [1, 4]}])
    with pytest.raises(ValueError, match="teacher_token_ids or teacher_logits"):
        list(iter_distillation_jsonl(path, cfg))

    write_jsonl(path, [{"input_ids": [1, True], "teacher_token_ids": [4]}])
    with pytest.raises(ValueError, match="token IDs must be integers"):
        list(iter_distillation_jsonl(path, cfg))

    write_jsonl(path, [{"input_ids": [1, 4], "teacher_token_ids": [cfg.vocab_size]}])
    with pytest.raises(ValueError, match="inside"):
        list(iter_distillation_jsonl(path, cfg))

    write_jsonl(path, [{"input_ids": [1, 4], "teacher_logits": [[0.0, 1.0]]}])
    with pytest.raises(ValueError, match="vocab_size"):
        list(iter_distillation_jsonl(path, cfg))

    write_jsonl(
        path,
        [{"input_ids": [1, 4], "teacher_logits": [[float("nan")] * cfg.vocab_size]}],
    )
    with pytest.raises(ValueError, match="finite"):
        list(iter_distillation_jsonl(path, cfg))


def test_distillation_loss_supports_tokens_and_logits() -> None:
    torch.manual_seed(123)
    cfg = tiny_config()
    path = ROOT / "tests" / "fixtures" / "distillation_tiny.jsonl"
    dataset = DistillationDataset.from_jsonl(path, cfg, seq_len=4)
    batch = next(
        dataset.batch_iterator(
            batch_size=3,
            torch=torch,
            device=torch.device("cpu"),
            shuffle=False,
        )
    )
    student_logits = torch.randn(3, 4, cfg.vocab_size, requires_grad=True)

    output = distillation_loss(
        student_logits,
        torch=torch,
        F=F,
        teacher_token_ids=batch.teacher_token_ids,
        teacher_token_mask=batch.teacher_token_mask,
        teacher_logits=batch.teacher_logits,
        teacher_logits_mask=batch.teacher_logits_mask,
        temperature=2.0,
    )

    assert output.token_loss is not None
    assert output.logit_loss is not None
    assert output.token_count == int(batch.teacher_token_mask.sum().item())
    assert output.logit_count == int(batch.teacher_logits_mask.sum().item())
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    assert (
        objective_contract("distillation").name == "teacher_token_or_logit_distillation"
    )


def test_distillation_loss_validates_shapes_and_weights() -> None:
    cfg = tiny_config()
    logits = torch.randn(1, 2, cfg.vocab_size)
    teacher_tokens = torch.tensor([[4, 5]], dtype=torch.long)
    token_mask = torch.tensor([[1, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="teacher_token_ids or teacher_logits"):
        distillation_loss(logits, torch=torch, F=F)
    with pytest.raises(ValueError, match="temperature"):
        distillation_loss(
            logits,
            torch=torch,
            F=F,
            teacher_token_ids=teacher_tokens,
            teacher_token_mask=token_mask,
            temperature=0.0,
        )
    with pytest.raises(ValueError, match="shape"):
        distillation_loss(
            logits,
            torch=torch,
            F=F,
            teacher_token_ids=torch.tensor([[4]], dtype=torch.long),
            teacher_token_mask=token_mask,
        )
    with pytest.raises(ValueError, match="teacher_token_mask"):
        distillation_loss(
            logits,
            torch=torch,
            F=F,
            teacher_token_ids=teacher_tokens,
            teacher_token_mask=torch.zeros_like(token_mask),
        )
    with pytest.raises(ValueError, match="loss weight"):
        distillation_loss(
            logits,
            torch=torch,
            F=F,
            teacher_token_ids=teacher_tokens,
            teacher_token_mask=token_mask,
            token_loss_weight=0.0,
        )
