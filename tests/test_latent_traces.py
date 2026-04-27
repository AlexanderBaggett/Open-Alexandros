from __future__ import annotations

import json

import pytest
import torch

from alexandros import AlexandrosConfig
from alexandros.training import (
    LatentTraceDataset,
    iter_latent_trace_jsonl,
    make_boolean_xor_latent_trace_records,
    make_modular_addition_latent_trace_records,
    summarize_latent_trace_record,
)


def tiny_config() -> AlexandrosConfig:
    return AlexandrosConfig(
        vocab_size=64,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        linear_attention_ratio=2,
        moe_num_experts=2,
        moe_num_shared_experts=1,
        moe_top_k=1,
        moe_expert_hidden_size=24,
        kv_lora_rank=8,
        latent_dim=8,
        latent_slots=2,
        diffusion_steps=3,
        mask_token_id=3,
    )


def write_jsonl(path, records) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_iter_latent_trace_jsonl_validates_token_contract(tmp_path) -> None:
    cfg = tiny_config()
    trace_path = tmp_path / "traces.jsonl"
    write_jsonl(
        trace_path,
        [
            {"input_ids": [4, 5, 6], "trace_ids": [7, 8], "target_ids": [9]},
            {"input_ids": [10], "trace_ids": [11, 12, 13]},
        ],
    )

    records = list(iter_latent_trace_jsonl(trace_path, cfg))

    assert records[0].input_ids == (4, 5, 6)
    assert records[0].trace_ids == (7, 8)
    assert records[0].target_ids == (9,)
    assert records[1].target_ids is None

    bad_path = tmp_path / "bad_traces.jsonl"
    write_jsonl(bad_path, [{"input_ids": [4, True], "trace_ids": [5]}])
    with pytest.raises(ValueError, match="token IDs must be integers"):
        list(iter_latent_trace_jsonl(bad_path, cfg))

    missing_path = tmp_path / "missing_trace.jsonl"
    write_jsonl(missing_path, [{"input_ids": [4, 5]}])
    with pytest.raises(ValueError, match="trace_ids"):
        list(iter_latent_trace_jsonl(missing_path, cfg))


def test_latent_trace_dataset_batches_masks_and_resume_state(tmp_path) -> None:
    cfg = tiny_config()
    trace_path = tmp_path / "traces.jsonl"
    write_jsonl(
        trace_path,
        [
            {"input_ids": [4, 5, 6], "trace_ids": [7, 8], "target_ids": [9, 10, 11]},
            {"input_ids": [12, 13, 14, 15, 16], "trace_ids": [17, 18, 19, 20]},
        ],
    )
    dataset = LatentTraceDataset.from_jsonl(
        trace_path,
        cfg,
        seq_len=4,
        trace_len=3,
    )
    iterator = dataset.batch_iterator(
        batch_size=2,
        torch=torch,
        device=torch.device("cpu"),
        seed=123,
        shuffle=False,
    )

    batch = next(iterator)

    assert batch.input_ids.shape == (2, 4)
    assert batch.trace_ids.shape == (2, 3)
    assert torch.equal(batch.input_attention_mask[0], torch.tensor([1, 1, 1, 0]))
    assert torch.equal(batch.input_ids[1], torch.tensor([12, 13, 14, 15]))
    assert torch.equal(batch.trace_attention_mask[0], torch.tensor([1, 1, 0]))
    assert batch.target_ids is not None
    assert batch.target_attention_mask is not None
    assert torch.equal(batch.target_ids[1], torch.tensor([17, 18, 19]))
    assert torch.equal(batch.target_attention_mask[1], torch.tensor([1, 1, 1]))

    iterator = dataset.batch_iterator(
        batch_size=1,
        torch=torch,
        device=torch.device("cpu"),
        seed=123,
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

    assert torch.equal(first.input_ids, torch.tensor([[4, 5, 6, cfg.pad_token_id]]))
    assert torch.equal(actual.input_ids, expected.input_ids)
    with pytest.raises(ValueError, match="cursor"):
        bad_state = dict(state)
        bad_state["cursor"] = True
        restored.load_state_dict(bad_state)


def test_synthetic_latent_toy_tasks_have_checkable_targets(tmp_path) -> None:
    cfg = tiny_config()

    modular = make_modular_addition_latent_trace_records(
        cfg,
        modulus=4,
        count=6,
        seed=1,
    )
    xor_records = make_boolean_xor_latent_trace_records(cfg, repetitions=2)

    assert len(modular) == 6
    assert len(xor_records) == 8
    value_offset = 8
    for record in modular:
        a = record.input_ids[2] - value_offset
        b = record.input_ids[3] - value_offset
        expected = value_offset + ((a + b) % 4)
        assert record.trace_ids[-1] == expected
        assert record.target_ids == (expected,)
    for record in xor_records:
        a = record.input_ids[2] - value_offset
        b = record.input_ids[3] - value_offset
        expected = value_offset + (a ^ b)
        assert record.trace_ids[-1] == expected
        assert record.target_ids == (expected,)

    trace_path = tmp_path / "toy_traces.jsonl"
    write_jsonl(
        trace_path,
        [
            {
                "input_ids": list(record.input_ids),
                "trace_ids": list(record.trace_ids),
                "target_ids": list(record.target_ids or ()),
            }
            for record in modular[:2] + xor_records[:2]
        ],
    )
    dataset = LatentTraceDataset.from_jsonl(trace_path, cfg, seq_len=5, trace_len=4)
    batch = next(
        dataset.batch_iterator(
            batch_size=4,
            torch=torch,
            device=torch.device("cpu"),
            shuffle=False,
        )
    )

    assert batch.input_ids.shape == (4, 5)
    assert batch.trace_ids.shape == (4, 4)
    assert batch.target_ids is not None
    assert batch.target_ids.shape == (4, 4)
    assert torch.equal(
        batch.target_attention_mask[:, 0], torch.ones(4, dtype=torch.long)
    )
    assert not batch.target_attention_mask[:, 1:].any()


def test_synthetic_latent_toy_tasks_validate_inputs() -> None:
    cfg = tiny_config()
    with pytest.raises(ValueError, match="modulus"):
        make_modular_addition_latent_trace_records(cfg, modulus=1)
    with pytest.raises(ValueError, match="repetitions"):
        make_boolean_xor_latent_trace_records(cfg, repetitions=0)
    small_cfg = tiny_config()
    small_cfg.vocab_size = 9
    with pytest.raises(ValueError, match="vocab_size"):
        make_modular_addition_latent_trace_records(small_cfg, modulus=4)


def test_latent_trace_decoded_summary_supports_vocabularies() -> None:
    cfg = tiny_config()
    record = make_modular_addition_latent_trace_records(
        cfg, modulus=4, count=1, seed=1
    )[0]
    vocabulary = {
        cfg.bos_token_id: "<bos>",
        4: "add_mod",
        6: "=",
        7: "answer",
        8: "0",
        9: "1",
        10: "2",
        11: "3",
    }

    summary = summarize_latent_trace_record(record, token_decoder=vocabulary)

    assert summary == {
        "input": "<bos> add_mod 0 1 answer",
        "trace": "0 1 = 1",
        "target": "1",
    }
    numeric = summarize_latent_trace_record(record, joiner=",")
    assert numeric["input"] == "1,4,8,9,7"
    callable_summary = summarize_latent_trace_record(
        record,
        token_decoder=lambda token_id: f"tok_{token_id}",
        joiner="|",
    )
    assert callable_summary["target"] == "tok_9"
    sequence_summary = summarize_latent_trace_record(record, token_decoder=["zero"])
    assert sequence_summary["input"].startswith("<1> <4>")


def test_latent_trace_decoded_summary_validates_inputs() -> None:
    cfg = tiny_config()
    record = make_boolean_xor_latent_trace_records(cfg)[0]
    with pytest.raises(ValueError, match="joiner"):
        summarize_latent_trace_record(record, joiner=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="token_decoder"):
        summarize_latent_trace_record(record, token_decoder=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="return strings"):
        summarize_latent_trace_record(record, token_decoder=lambda token_id: token_id)  # type: ignore[arg-type]
