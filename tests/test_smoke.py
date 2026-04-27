from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import alexandros.modeling_alexandros as modeling_alexandros
from alexandros import (
    AlexandrosConfig,
    AlexandrosForCausalLM,
    AlexandrosForDiffusionLM,
    AlexandrosModel,
)
from alexandros.bitlinear import (
    BitLinear,
    codes_to_ternary,
    export_packed_bitlinear_state,
    pack_ternary_codes,
    save_packed_bitlinear_state,
    ternary_codes_and_scales,
    ternary_to_codes,
    unpack_ternary_codes,
)
from alexandros.configuration_alexandros import load_config_file
from alexandros.diffusion import MaskedDiffusionScheduler
from alexandros.evaluation import (
    causal_lm_perplexity,
    count_attention_layers,
    estimate_cache_memory,
    estimate_flops,
    latent_reconstruction_metrics,
    masked_diffusion_reconstruction_accuracy,
    profile_model_runtime,
    recurrent_state_drift_probe,
    summarize_moe_stats,
    summarize_parameters,
    synthetic_copy_retrieval_probe,
    synthetic_lost_in_middle_probe,
    synthetic_modular_addition_probe,
    synthetic_needle_retrieval_probe,
    turboquant_reconstruction_metrics,
)
from alexandros.inference import GenerationRequest, reorder_generation_cache
from alexandros.kv_cache import TurboQuantKVCache, TurboQuantPacket
from alexandros.moe import MoEFeedForward
from alexandros.training import (
    IGNORE_INDEX,
    PackedTokenDataset,
    apply_trainability,
    iter_token_id_jsonl,
    make_block_diffusion_training_batch,
    make_diffusion_training_batch,
    objective_contract,
    objective_log_fields,
    pack_token_sequences,
    phase_checkpoint_metadata,
    standard_metric_names,
    validate_standard_metric_record,
)
from alexandros.ttt import TTTState

ROOT = Path(__file__).resolve().parents[1]
COMMON_SPEC = importlib.util.spec_from_file_location(
    "alexandros_script_common", ROOT / "scripts" / "_common.py"
)
assert COMMON_SPEC is not None and COMMON_SPEC.loader is not None
script_common = importlib.util.module_from_spec(COMMON_SPEC)
COMMON_SPEC.loader.exec_module(script_common)


def tiny_config(variant: str = "heavy") -> AlexandrosConfig:
    return AlexandrosConfig(
        variant=variant,
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
        enable_adaptive_depth=True,
        max_depth_iters=2,
        ttt_rank=4,
    )


def test_config_round_trip() -> None:
    cfg = tiny_config("lite")
    restored = AlexandrosConfig.from_dict(cfg.to_dict())
    assert restored == cfg
    assert cfg.mla_d_c == cfg.kv_lora_rank
    assert cfg.mla_d_r == 0
    assert cfg.mla_d_nope + cfg.mla_d_r == cfg.head_dim
    assert cfg.mla_value_head_dim == cfg.head_dim
    assert (
        cfg.standard_mha_elements_per_token
        == 2 * cfg.num_attention_heads * cfg.head_dim
    )
    assert cfg.mla_elements_per_token == cfg.mla_d_c + cfg.mla_d_r


def test_repository_configs_load_and_plan() -> None:
    config_paths = sorted((ROOT / "configs").glob("*.yaml"))
    assert config_paths
    seen = {path.name for path in config_paths}
    assert {"heavy_1b.yaml", "lite_1b.yaml"}.issubset(seen)
    for path in config_paths:
        cfg = load_config_file(path)
        assert cfg.attention_layers()
        report = estimate_cache_memory(cfg, batch_size=1, sequence_length=8)
        assert report.standard_kv_bits > 0
        flops = estimate_flops(cfg, batch_size=1, sequence_length=8)
        assert flops.prefill_flops > flops.decode_token_flops > 0
    assert load_config_file(ROOT / "configs" / "heavy_1b.yaml").variant == "heavy"
    assert load_config_file(ROOT / "configs" / "lite_1b.yaml").variant == "lite"


def test_script_common_metadata_and_jsonl(tmp_path) -> None:
    cfg = tiny_config()

    class Args:
        device = "cpu"
        dtype = "float32"
        seed = 123
        deterministic = True
        out_dir = str(tmp_path / "run")
        log_jsonl = ""

    out_dir = script_common.prepare_output_dir(Args)
    assert out_dir == tmp_path / "run"
    assert Args.log_jsonl.endswith("metrics.jsonl")
    metadata = script_common.run_metadata(Args, cfg, torch)
    assert metadata["config_hash"] == script_common.config_hash(cfg)
    assert metadata["deterministic"] is True
    script_common.write_jsonl(Args.log_jsonl, {"step": 0, "loss": 1.25})
    record = json.loads(Path(Args.log_jsonl).read_text(encoding="utf-8"))
    assert record == {"loss": 1.25, "step": 0}
    ar_contract = objective_contract("ar")
    assert ar_contract.ignore_index == IGNORE_INDEX
    assert (
        objective_log_fields("diffusion")["objective_name"]
        == "absorbing_masked_or_rao_blackwellized_token"
    )
    assert "loss" in standard_metric_names("ar")
    assert "prefill_chunk_loss" in standard_metric_names("ttt")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_standard_metric_record({"phase": "ar"}, "ar")
    with pytest.raises(ValueError, match="unknown objective phase"):
        objective_contract("unknown")
    with pytest.raises(ValueError, match="unknown training phase"):
        standard_metric_names("unknown")


def test_script_common_validation_helpers() -> None:
    parser = script_common.make_arg_parser("demo parser")
    parser.add_argument("--demo", default="x", help="demo value")
    assert parser.description == "demo parser"
    assert "default: x" in parser.format_help()

    class Args:
        grad_accum_steps = 1
        val_every = 2
        val_batches = 1
        lr = None
        warmup_steps = 0

    script_common.validate_training_args(Args)
    assert script_common.data_source_name(Args) == "synthetic_smoke"
    token_args = type("TokenArgs", (), {"token_ids_jsonl": "tokens.jsonl"})()
    assert script_common.data_source_name(token_args) == "token_ids_jsonl"
    heavy_args = type("HeavyArgs", (), {"lr": None})()
    script_common.resolve_training_hparams(heavy_args, tiny_config("heavy"))
    assert heavy_args.lr == pytest.approx(1e-3)
    lite_args = type("LiteArgs", (), {"lr": None})()
    script_common.resolve_training_hparams(lite_args, tiny_config("lite"))
    assert lite_args.lr == pytest.approx(3e-4)
    assert script_common.should_validate(Args, 0)
    assert not script_common.should_validate(Args, 1)

    events: list[tuple[str, float, int] | str] = []

    class FakeSummaryWriter:
        def __init__(self, log_dir: str) -> None:
            events.append(f"log_dir={log_dir}")

        def add_scalar(self, key: str, value: float, step: int) -> None:
            events.append((key, value, step))

        def flush(self) -> None:
            events.append("flush")

        def close(self) -> None:
            events.append("close")

    fake_tensorboard = type(sys)("torch.utils.tensorboard")
    fake_tensorboard.SummaryWriter = FakeSummaryWriter
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_tensorboard)
    try:
        tb_args = type("TbArgs", (), {"tensorboard_dir": "runs/tb"})()
        writer = script_common.make_tensorboard_writer(tb_args)
        script_common.write_tensorboard_scalars(
            writer,
            {
                "step": 3,
                "loss": 1.25,
                "finite": True,
                "not_scalar": [1, 2, 3],
                "nan": float("nan"),
            },
            step=3,
        )
        script_common.close_tensorboard_writer(writer)
    finally:
        monkeypatch.undo()
    assert "log_dir=runs/tb" in events
    assert ("loss", 1.25, 3) in events
    assert ("finite", 1.0, 3) in events
    assert not any(event[0] == "nan" for event in events if isinstance(event, tuple))
    assert "flush" in events
    assert "close" in events

    generator = script_common.seeded_generator(torch, 123, torch.device("cpu"))
    first = torch.randint(0, 100, (4,), generator=generator)
    generator = script_common.seeded_generator(torch, 123, torch.device("cpu"))
    second = torch.randint(0, 100, (4,), generator=generator)
    assert torch.equal(first, second)
    cfg = tiny_config()
    batch = script_common.sample_token_batch(
        cfg,
        batch_size=2,
        seq_len=3,
        device=torch.device("cpu"),
        torch=torch,
        force_bos=True,
    )
    assert batch.shape == (2, 3)
    assert batch[:, 0].eq(cfg.bos_token_id).all()
    with pytest.raises(ValueError, match="batch_size"):
        script_common.sample_token_batch(
            cfg,
            batch_size=0,
            seq_len=3,
            device=torch.device("cpu"),
            torch=torch,
        )
    with pytest.raises(ValueError, match="val_batches"):

        class BadArgs:
            grad_accum_steps = 1
            val_every = 1
            val_batches = 0

        script_common.validate_training_args(BadArgs)
    with pytest.raises(ValueError, match="warmup_steps"):

        class BadWarmupArgs:
            grad_accum_steps = 1
            val_every = 0
            val_batches = 1
            lr = 1e-3
            warmup_steps = -1

        script_common.validate_training_args(BadWarmupArgs)
    with pytest.raises(ValueError, match="steps"):

        class BadStepsArgs:
            steps = 0
            grad_accum_steps = 1
            val_every = 0
            val_batches = 1

        script_common.validate_training_args(BadStepsArgs)
    with pytest.raises(ValueError, match="batch_size"):

        class BadBatchArgs:
            batch_size = True
            grad_accum_steps = 1
            val_every = 0
            val_batches = 1

        script_common.validate_training_args(BadBatchArgs)
    with pytest.raises(ValueError, match="lr"):

        class BadLrArgs:
            grad_accum_steps = 1
            val_every = 0
            val_batches = 1
            lr = float("nan")

        script_common.validate_training_args(BadLrArgs)
    with pytest.raises(ValueError, match="grad_clip"):

        class BadGradClipArgs:
            grad_accum_steps = 1
            val_every = 0
            val_batches = 1
            grad_clip = float("inf")

        script_common.validate_training_args(BadGradClipArgs)


def test_script_common_training_precision_helpers() -> None:
    model = torch.nn.Linear(2, 2)
    prepared = script_common.prepare_training_model(
        model,
        torch.device("cpu"),
        torch.bfloat16,
        torch,
    )
    assert next(prepared.parameters()).dtype == torch.float32
    assert (
        script_common.make_grad_scaler(torch.device("cpu"), torch.bfloat16, torch)
        is None
    )
    context = script_common.autocast_context(torch.device("cpu"), torch.bfloat16, torch)
    with context():
        y = prepared(torch.randn(1, 2))
    assert torch.isfinite(y.float()).all()
    with pytest.raises(ValueError, match="float16 training"):
        script_common.prepare_training_model(
            torch.nn.Linear(2, 2),
            torch.device("cpu"),
            torch.float16,
            torch,
        )


def test_token_id_jsonl_dataset_packs_splits_and_batches(tmp_path) -> None:
    cfg = tiny_config()
    data_path = tmp_path / "tokens.jsonl"
    records = [
        {"input_ids": [4 + ((idx + offset) % 20) for offset in range(5)]}
        for idx in range(24)
    ]
    data_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    streamed = list(iter_token_id_jsonl(data_path))
    assert streamed[0] == records[0]["input_ids"]
    assert list(
        pack_token_sequences(
            [[4, 5], [6, 7, 8]], seq_len=3, eos_token_id=cfg.eos_token_id
        )
    ) == [
        [4, 5, cfg.eos_token_id],
        [6, 7, 8],
    ]

    train = PackedTokenDataset.from_jsonl(
        data_path,
        cfg,
        seq_len=4,
        validation_fraction=0.25,
        seed=11,
        prepend_bos=True,
    )
    validation = PackedTokenDataset.from_jsonl(
        data_path,
        cfg,
        seq_len=4,
        split="validation",
        validation_fraction=0.25,
        seed=11,
        prepend_bos=True,
    )
    assert train.chunks
    assert validation.chunks
    assert train.chunks != validation.chunks

    first_iter = train.batch_iterator(
        batch_size=2, torch=torch, device=torch.device("cpu"), seed=17
    )
    second_iter = train.batch_iterator(
        batch_size=2, torch=torch, device=torch.device("cpu"), seed=17
    )
    first = next(first_iter)
    second = next(second_iter)
    assert first.shape == (2, 4)
    assert torch.equal(first, second)
    saved_state = first_iter.state_dict()
    expected_next = next(first_iter)
    restored_iter = train.batch_iterator(
        batch_size=2, torch=torch, device=torch.device("cpu"), seed=999
    )
    restored_iter.load_state_dict(saved_state)
    assert torch.equal(next(restored_iter), expected_next)
    bad_state = dict(saved_state)
    bad_state["order"] = torch.zeros_like(saved_state["order"])
    with pytest.raises(ValueError, match="permutation"):
        restored_iter.load_state_dict(bad_state)
    bad_state = dict(saved_state)
    bad_state["order"] = saved_state["order"].float()
    with pytest.raises(ValueError, match="integer tensor"):
        restored_iter.load_state_dict(bad_state)
    bad_state = dict(saved_state)
    bad_state["cursor"] = 0.0
    with pytest.raises(ValueError, match="cursor"):
        restored_iter.load_state_dict(bad_state)
    bad_state = dict(saved_state)
    bad_state["seed"] = True
    with pytest.raises(ValueError, match="seed"):
        restored_iter.load_state_dict(bad_state)
    bad_state = dict(saved_state)
    bad_state["shuffle"] = "yes"
    with pytest.raises(ValueError, match="shuffle"):
        restored_iter.load_state_dict(bad_state)
    bad_state = dict(saved_state)
    bad_state["generator_state"] = saved_state["generator_state"].long()
    with pytest.raises(ValueError, match="uint8"):
        restored_iter.load_state_dict(bad_state)
    with pytest.raises(ValueError, match="batch_size"):
        train.batch_iterator(
            batch_size=True, torch=torch, device=torch.device("cpu"), seed=17
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        train.batch_iterator(
            batch_size=2, torch=torch, device=torch.device("cpu"), seed=True
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shuffle"):
        train.batch_iterator(
            batch_size=2,
            torch=torch,
            device=torch.device("cpu"),
            seed=17,
            shuffle=1,  # type: ignore[arg-type]
        )

    class Args:
        token_ids_jsonl = str(data_path)
        token_field = "input_ids"
        validation_fraction = 0.25
        seed = 11
        seq_len = 4
        batch_size = 2
        no_shuffle_data = True

    batch_iter = script_common.make_token_batch_iterator(
        Args,
        cfg,
        split="train",
        device=torch.device("cpu"),
        torch=torch,
        prepend_bos=True,
    )
    assert batch_iter is not None
    batch = script_common.next_token_batch(
        batch_iter,
        cfg,
        batch_size=2,
        seq_len=4,
        device=torch.device("cpu"),
        torch=torch,
    )
    assert batch.shape == (2, 4)


def test_checked_in_token_id_fixture_is_documented_and_usable() -> None:
    cfg = tiny_config()
    fixture_path = ROOT / "tests" / "fixtures" / "token_ids_tiny.jsonl"
    metadata_path = ROOT / "tests" / "fixtures" / "token_ids_tiny.metadata.json"
    readme_path = ROOT / "tests" / "fixtures" / "README.md"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    assert metadata["format"] == "open-alexandros-test-fixture-metadata"
    assert metadata["sha256"] == digest
    assert metadata["allowed_use"] == "Unit, integration, and CI smoke tests only."
    assert "Pretraining" in metadata["not_for"]
    assert digest in readme_path.read_text(encoding="utf-8")

    contract = metadata["vocab_contract"]
    assert cfg.vocab_size >= contract["vocab_size_minimum"]
    assert cfg.pad_token_id == contract["pad_token_id"]
    assert cfg.bos_token_id == contract["bos_token_id"]
    assert cfg.eos_token_id == contract["eos_token_id"]
    assert cfg.mask_token_id == contract["mask_token_id"]

    streamed = list(iter_token_id_jsonl(fixture_path))
    assert len(streamed) == 4
    assert streamed[0] == [cfg.bos_token_id, 4, 5, 6, 7, cfg.eos_token_id]
    dataset = PackedTokenDataset.from_jsonl(
        fixture_path,
        cfg,
        seq_len=4,
        append_eos=False,
    )
    assert dataset.chunks == (
        (1, 4, 5, 6),
        (7, 2, 1, 8),
        (9, 10, 11, 2),
        (1, 12, 13, 14),
        (15, 2, 1, 16),
        (17, 18, 19, 2),
    )


def test_token_id_jsonl_dataset_rejects_bad_tokens(tmp_path) -> None:
    cfg = tiny_config()
    data_path = tmp_path / "bad.jsonl"
    data_path.write_text(
        json.dumps({"input_ids": [4, cfg.vocab_size]}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="token IDs"):
        PackedTokenDataset.from_jsonl(data_path, cfg, seq_len=2)
    bool_path = tmp_path / "bool.jsonl"
    bool_path.write_text(json.dumps({"input_ids": [4, True]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integers"):
        PackedTokenDataset.from_jsonl(bool_path, cfg, seq_len=2)


def test_diffusion_training_batch_transform_is_reproducible() -> None:
    cfg = tiny_config()
    input_ids = torch.tensor(
        [
            [cfg.bos_token_id, 4, 5, 6, cfg.pad_token_id],
            [cfg.bos_token_id, 7, 8, 9, 10],
        ],
        dtype=torch.long,
    )
    torch.manual_seed(2025)
    before = torch.get_rng_state()
    first = make_diffusion_training_batch(
        input_ids,
        cfg,
        torch=torch,
        generator=torch.Generator().manual_seed(123),
    )
    after = torch.get_rng_state()
    second = make_diffusion_training_batch(
        input_ids,
        cfg,
        torch=torch,
        generator=torch.Generator().manual_seed(123),
    )
    assert torch.equal(before, after)
    assert torch.equal(first.noisy_input_ids, second.noisy_input_ids)
    assert torch.equal(first.mask, second.mask)
    assert torch.equal(first.timesteps, second.timesteps)
    assert first.masked_token_count.item() == first.mask.sum().item()
    assert 0.0 < first.masked_fraction <= 1.0
    assert first.labels[first.mask].ne(-100).all()
    assert first.labels[~first.mask].eq(-100).all()
    assert first.noisy_input_ids[first.mask].eq(cfg.mask_token_id).all()
    with pytest.raises(ValueError, match="at least one non-pad"):
        make_diffusion_training_batch(
            torch.full((1, 4), cfg.pad_token_id, dtype=torch.long),
            cfg,
            torch=torch,
            timesteps=torch.tensor([0]),
        )
    with pytest.raises(ValueError, match="integer tensor"):
        make_diffusion_training_batch(input_ids.float(), cfg, torch=torch)
    with pytest.raises(ValueError, match="batch size"):
        make_diffusion_training_batch(
            torch.empty((0, 4), dtype=torch.long), cfg, torch=torch
        )
    with pytest.raises(ValueError, match="sequence length"):
        make_diffusion_training_batch(
            torch.empty((1, 0), dtype=torch.long), cfg, torch=torch
        )
    with pytest.raises(ValueError, match="outside"):
        make_diffusion_training_batch(
            torch.tensor([[cfg.bos_token_id, cfg.vocab_size]], dtype=torch.long),
            cfg,
            torch=torch,
        )
    scheduler = MaskedDiffusionScheduler(cfg)
    with pytest.raises(ValueError, match="integer tensor"):
        scheduler.add_noise(input_ids.float(), timestep=0)
    with pytest.raises(ValueError, match="batch size"):
        scheduler.add_noise(torch.empty((0, 4), dtype=torch.long), timestep=0)
    with pytest.raises(ValueError, match="sequence length"):
        scheduler.add_noise(torch.empty((1, 0), dtype=torch.long), timestep=0)
    with pytest.raises(ValueError, match="outside"):
        scheduler.add_noise(
            torch.tensor([[cfg.bos_token_id, cfg.vocab_size]], dtype=torch.long),
            timestep=0,
        )
    with pytest.raises(ValueError, match="integer"):
        scheduler.mask_probability(torch.tensor([0.5]))
    with pytest.raises(ValueError, match="diffusion_steps"):
        make_diffusion_training_batch(
            input_ids,
            cfg,
            torch=torch,
            timesteps=torch.tensor([cfg.diffusion_steps, 0]),
        )
    with pytest.raises(ValueError, match="scalar"):
        make_diffusion_training_batch(
            input_ids,
            cfg,
            torch=torch,
            timesteps=torch.zeros(3, dtype=torch.long),
        )


def test_block_diffusion_training_batch_masks_contiguous_targets() -> None:
    cfg = tiny_config()
    input_ids = torch.tensor(
        [
            [cfg.bos_token_id, 4, 5, 6, 7, 8],
            [cfg.pad_token_id, cfg.pad_token_id, 9, 10, 11, 12],
        ],
        dtype=torch.long,
    )
    first = make_block_diffusion_training_batch(
        input_ids,
        cfg,
        torch=torch,
        block_size=3,
        generator=torch.Generator().manual_seed(321),
    )
    second = make_block_diffusion_training_batch(
        input_ids,
        cfg,
        torch=torch,
        block_size=3,
        generator=torch.Generator().manual_seed(321),
    )
    assert torch.equal(first.noisy_input_ids, second.noisy_input_ids)
    assert torch.equal(first.block_starts, second.block_starts)
    assert first.block_size == 3
    assert first.masked_token_count.item() == first.mask.sum().item()
    assert first.noisy_input_ids[first.mask].eq(cfg.mask_token_id).all()
    assert first.labels[first.mask].ne(-100).all()
    assert first.labels[~first.mask].eq(-100).all()
    for row, start in enumerate(first.block_starts.tolist()):
        outside = torch.ones(input_ids.size(1), dtype=torch.bool)
        outside[start : start + first.block_size] = False
        assert not first.mask[row, outside].any()
        assert first.mask[row, start : start + first.block_size].any()
    with pytest.raises(ValueError, match="integer tensor"):
        make_block_diffusion_training_batch(
            input_ids.float(), cfg, torch=torch, block_size=2
        )
    with pytest.raises(ValueError, match="sequence length"):
        make_block_diffusion_training_batch(
            torch.empty((1, 0), dtype=torch.long),
            cfg,
            torch=torch,
            block_size=1,
        )
    with pytest.raises(ValueError, match="outside"):
        make_block_diffusion_training_batch(
            torch.tensor([[cfg.bos_token_id, cfg.vocab_size]], dtype=torch.long),
            cfg,
            torch=torch,
            block_size=1,
        )
    with pytest.raises(ValueError, match="block_size"):
        make_block_diffusion_training_batch(input_ids, cfg, torch=torch, block_size=0)
    with pytest.raises(ValueError, match="at least one non-pad"):
        make_block_diffusion_training_batch(
            torch.full((1, 4), cfg.pad_token_id, dtype=torch.long),
            cfg,
            torch=torch,
            block_size=2,
        )
    with pytest.raises(ValueError, match="diffusion_steps"):
        make_block_diffusion_training_batch(
            input_ids,
            cfg,
            torch=torch,
            block_size=2,
            timesteps=torch.tensor([0, cfg.diffusion_steps]),
        )


def test_script_common_training_state_round_trip(tmp_path) -> None:
    cfg = tiny_config()
    model = AlexandrosForCausalLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = copy.deepcopy(model.state_dict())
    data_path = tmp_path / "tokens.jsonl"
    data_path.write_text(
        "\n".join(
            json.dumps(
                {"input_ids": [4 + ((idx + offset) % 20) for offset in range(5)]}
            )
            for idx in range(20)
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = PackedTokenDataset.from_jsonl(data_path, cfg, seq_len=4, seed=5)
    data_iter = dataset.batch_iterator(
        batch_size=2, torch=torch, device=torch.device("cpu"), seed=7
    )
    next(data_iter)

    class Args:
        device = "cpu"
        dtype = "float32"
        seed = 123
        deterministic = False

    checkpoint = tmp_path / "training_state.pt"
    script_common.save_training_state(
        checkpoint,
        model=model,
        optimizer=opt,
        config=cfg,
        args=Args,
        torch=torch,
        step=3,
        data_iterator=data_iter,
    )
    expected_batch_after_resume = next(data_iter)
    restored_data_iter = dataset.batch_iterator(
        batch_size=2,
        torch=torch,
        device=torch.device("cpu"),
        seed=999,
    )
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    next_step = script_common.load_training_state(
        checkpoint,
        model=model,
        optimizer=opt,
        config=cfg,
        torch=torch,
        map_location="cpu",
        data_iterator=restored_data_iter,
    )
    assert next_step == 4
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    assert torch.equal(next(restored_data_iter), expected_batch_after_resume)


def test_pretrain_validation_cli_writes_val_metrics(tmp_path) -> None:
    out_dir = tmp_path / "ar"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/pretrain.py",
            "--config",
            "configs/heavy_debug.yaml",
            "--steps",
            "1",
            "--out-dir",
            str(out_dir),
            "--val-every",
            "1",
            "--val-batches",
            "1",
            "--seed",
            "31",
            "--device",
            "cpu",
            "--dtype",
            "bfloat16",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "val_loss=" in result.stdout
    records = [
        json.loads(line)
        for line in (out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validate_standard_metric_record(records[-1], "ar")
    assert records[-1]["phase"] == "ar"
    assert records[-1]["val_loss"] > 0
    assert records[-1]["val_logit_norm"] > 0
    assert records[-1]["lr"] == pytest.approx(1e-3)
    assert records[-1]["amp_enabled"] is True
    assert records[-1]["grad_scaler_enabled"] is False


def test_pretrain_cli_reads_token_id_jsonl(tmp_path) -> None:
    out_dir = tmp_path / "ar_data"
    data_path = tmp_path / "tokens.jsonl"
    records = [
        {"input_ids": [4 + ((idx + offset) % 40) for offset in range(8)]}
        for idx in range(32)
    ]
    data_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/pretrain.py",
            "--config",
            "configs/heavy_debug.yaml",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--seq-len",
            "6",
            "--token-ids-jsonl",
            str(data_path),
            "--validation-fraction",
            "0.25",
            "--out-dir",
            str(out_dir),
            "--val-every",
            "1",
            "--val-batches",
            "1",
            "--seed",
            "41",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    records = [
        json.loads(line)
        for line in (out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validate_standard_metric_record(records[-1], "ar")
    assert records[-1]["phase"] == "ar"
    assert records[-1]["objective_name"] == "causal_next_token"
    assert records[-1]["objective_ignore_index"] == IGNORE_INDEX
    assert records[-1]["trainability_phase"] == "ar"
    assert records[-1]["trainability_scope"] == "phase_default"
    assert records[-1]["trainability_trainable_parameters"] > 0
    assert records[-1]["data_source"] == "token_ids_jsonl"
    assert records[-1]["val_loss"] > 0
    checkpoint_metadata = AlexandrosForCausalLM.load_checkpoint_metadata(
        out_dir / "checkpoint"
    )
    assert checkpoint_metadata is not None
    assert checkpoint_metadata["phase"] == "ar"
    assert checkpoint_metadata["enabled_modules"] == ["model", "lm_head"]


@pytest.mark.parametrize(
    ("script", "phase"),
    [
        ("scripts/train_diffusion.py", "diffusion"),
        ("scripts/train_latent.py", "latent"),
        ("scripts/meta_train_ttt.py", "ttt"),
    ],
)
def test_training_clis_accept_external_token_id_jsonl(
    tmp_path,
    script: str,
    phase: str,
) -> None:
    out_dir = tmp_path / phase
    data_path = tmp_path / f"{phase}_tokens.jsonl"
    records = [
        {"input_ids": [4 + ((idx + offset) % 40) for offset in range(8)]}
        for idx in range(32)
    ]
    data_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        script,
        "--config",
        "configs/heavy_debug.yaml",
        "--batch-size",
        "1",
        "--seq-len",
        "6",
        "--token-ids-jsonl",
        str(data_path),
        "--validation-fraction",
        "0.25",
        "--out-dir",
        str(out_dir),
        "--seed",
        "43",
        "--device",
        "cpu",
    ]
    if phase != "ttt":
        command.extend(["--steps", "1", "--val-every", "1", "--val-batches", "1"])
    if phase == "latent":
        command.extend(
            [
                "--lambda-kl",
                "0.02",
                "--lambda-rec",
                "0.5",
                "--latent-refinement-steps",
                "2",
            ]
        )
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    records = [
        json.loads(line)
        for line in (out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validate_standard_metric_record(records[-1], phase)
    assert records[-1]["phase"] == phase
    assert records[-1]["data_source"] == "token_ids_jsonl"
    assert records[-1]["loss" if phase != "ttt" else "pre_update_loss"] > 0
    assert records[-1]["objective_phase"] == phase
    assert records[-1]["trainability_phase"] == phase
    assert records[-1]["trainability_scope"] == "phase_default"
    expected_objectives = {
        "diffusion": "absorbing_masked_or_rao_blackwellized_token",
        "latent": "latent_reconstruction_kl",
        "ttt": "request_local_next_token_probe",
    }
    assert records[-1]["objective_name"] == expected_objectives[phase]
    if phase == "ttt":
        assert records[-1]["trainability_trainable_parameters"] == 0
        assert records[-1]["ttt_meta_trainable_parameters"] > 0
        assert Path(records[-1]["ttt_checkpoint_path"]).is_file()
    else:
        assert records[-1]["trainability_trainable_parameters"] > 0
    if phase == "diffusion":
        assert records[-1]["objective_ignore_index"] == IGNORE_INDEX
        assert (
            records[-1]["objective_normalization"]
            == "masked objective divides weighted masked-token CE by masked_token_count; "
            "rao_blackwellized objective divides weighted target-position CE by non-pad target count"
        )
        assert records[-1]["diffusion_objective"] == "masked"
        assert records[-1]["diffusion_loss_weighting"] == "uniform"
        assert records[-1]["moe_timestep_tracked_selections"] > 0
        assert len(records[-1]["moe_timestep_load_entropy"]) >= 1
        assert records[-1]["moe_noisy_step_load_entropy"] >= 0
        assert records[-1]["moe_polish_step_load_entropy"] >= 0
        assert records[-1]["moe_noisy_timestep_tracked_selections"] >= 0
        assert records[-1]["moe_polish_timestep_tracked_selections"] >= 0
    if phase == "latent":
        assert records[-1]["lambda_kl"] == pytest.approx(0.02)
        assert records[-1]["lambda_rec"] == pytest.approx(0.5)
        assert records[-1]["latent_refinement_steps"] == 2
        assert records[-1]["reconstruction_loss"] > 0
        assert records[-1]["vae_reconstruction_loss"] > 0
        assert records[-1]["refinement_reconstruction_loss"] > 0
        assert records[-1]["kl_loss"] >= 0
        assert records[-1]["val_reconstruction_loss"] > 0
        assert records[-1]["val_refinement_reconstruction_loss"] > 0
    if phase != "ttt":
        checkpoint_metadata = AlexandrosForDiffusionLM.load_checkpoint_metadata(
            out_dir / "checkpoint"
        )
        assert checkpoint_metadata is not None
        assert checkpoint_metadata["phase"] == phase
        assert checkpoint_metadata["phase_handoff_contract_version"] == 1


def test_train_latent_cli_rejects_invalid_objective_weights(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--config",
            "configs/heavy_debug.yaml",
            "--steps",
            "1",
            "--out-dir",
            str(tmp_path / "latent_bad"),
            "--lambda-kl",
            "-1",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "lambda_kl" in result.stderr

    bad_steps = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--config",
            "configs/heavy_debug.yaml",
            "--steps",
            "1",
            "--out-dir",
            str(tmp_path / "latent_bad_steps"),
            "--latent-refinement-steps",
            "0",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert bad_steps.returncode != 0
    assert "latent_refinement_steps" in bad_steps.stderr

    conflict_data = tmp_path / "tokens.jsonl"
    conflict_data.write_text(
        '{"input_ids":[4,5,6],"trace_ids":[7,8]}\n', encoding="utf-8"
    )
    conflict = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--config",
            "configs/heavy_debug.yaml",
            "--steps",
            "1",
            "--token-ids-jsonl",
            str(conflict_data),
            "--trace-jsonl",
            str(conflict_data),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert conflict.returncode != 0
    assert "trace_jsonl or token_ids_jsonl" in conflict.stderr


def test_train_latent_cli_reads_trace_jsonl(tmp_path) -> None:
    out_dir = tmp_path / "latent_trace"
    trace_path = tmp_path / "latent_traces.jsonl"
    records = [
        {
            "input_ids": [4 + ((idx + offset) % 32) for offset in range(7)],
            "trace_ids": [8 + ((idx + offset) % 28) for offset in range(5)],
            "target_ids": [12 + (idx % 20)],
        }
        for idx in range(32)
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--config",
            "configs/heavy_debug.yaml",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--seq-len",
            "6",
            "--trace-len",
            "5",
            "--trace-jsonl",
            str(trace_path),
            "--validation-fraction",
            "0.25",
            "--out-dir",
            str(out_dir),
            "--val-every",
            "1",
            "--val-batches",
            "1",
            "--seed",
            "47",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metric_records = [
        json.loads(line)
        for line in (out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metric_records[-1]["phase"] == "latent"
    assert metric_records[-1]["data_source"] == "latent_trace_jsonl"
    assert metric_records[-1]["loss"] > 0
    assert metric_records[-1]["val_reconstruction_loss"] > 0


def test_meta_train_ttt_cli_writes_pre_and_post_update_loss(tmp_path) -> None:
    out_dir = tmp_path / "ttt"
    subprocess.run(
        [
            sys.executable,
            "scripts/meta_train_ttt.py",
            "--config",
            "configs/heavy_debug.yaml",
            "--batch-size",
            "1",
            "--seq-len",
            "6",
            "--prefill-chunk-len",
            "3",
            "--out-dir",
            str(out_dir),
            "--seed",
            "37",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    records = [
        json.loads(line)
        for line in (out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["phase"] == "ttt"
    assert records[-1]["objective_name"] == "request_local_next_token_probe"
    validate_standard_metric_record(records[-1], "ttt")
    assert records[-1]["trainability_trainable_parameters"] == 0
    assert records[-1]["ttt_meta_trainable_parameters"] > 0
    assert records[-1]["pre_update_loss"] > 0
    assert records[-1]["post_update_loss"] > 0
    assert records[-1]["loss"] == pytest.approx(records[-1]["outer_loss"])
    assert records[-1]["inner_loss"] > 0
    assert records[-1]["grad_norm"] > 0
    assert records[-1]["ttt_inner_lr"] > 0
    assert records[-1]["ttt_steps"] == 1
    assert records[-1]["prefill_chunk_len"] == 3
    assert records[-1]["prefill_chunk_count"] == 1
    assert records[-1]["prefill_chunk_loss"] > 0
    assert records[-1]["ttt_inner_grad_norm"] > 0
    assert records[-1]["ttt_heldout_chunk_len"] == 3
    assert records[-1]["hidden_norm_before"] == pytest.approx(records[-1]["base_norm"])
    assert records[-1]["hidden_norm_after"] == pytest.approx(
        records[-1]["adapted_norm"]
    )
    assert records[-1]["hidden_delta_norm"] == pytest.approx(records[-1]["delta_norm"])
    checkpoint_path = Path(records[-1]["ttt_checkpoint_path"])
    assert checkpoint_path.is_file()
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    assert payload["format"] == "open-alexandros-ttt-meta-adapter"
    assert payload["request_local_state_saved"] is False
    assert payload["fast_weight_parameters"]["rank"] == tiny_config().ttt_rank
    assert "ttt_meta_adapter_state_dict" in payload
    assert not any(
        key.startswith("fast_") for key in payload["ttt_meta_adapter_state_dict"].keys()
    )


def test_meta_train_ttt_cli_rejects_invalid_prefill_chunk_len(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/meta_train_ttt.py",
            "--config",
            "configs/heavy_debug.yaml",
            "--batch-size",
            "1",
            "--seq-len",
            "6",
            "--prefill-chunk-len",
            "1",
            "--out-dir",
            str(tmp_path / "bad_ttt"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "prefill_chunk_len" in result.stderr


def test_eval_cli_compare_configs_writes_baseline_table(tmp_path) -> None:
    out_dir = tmp_path / "eval"
    subprocess.run(
        [
            sys.executable,
            "scripts/eval.py",
            "--config",
            "configs/heavy_debug.yaml",
            "--compare-config",
            "configs/lite_debug.yaml",
            "--out-dir",
            str(out_dir),
            "--seed",
            "23",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    records = [
        json.loads(line)
        for line in (out_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["config"] for record in records] == [
        "configs/heavy_debug.yaml",
        "configs/lite_debug.yaml",
    ]
    for record in records:
        assert set(record["generation_modes"]) == {
            "autoregressive",
            "block_diffusion",
            "latent_reasoning",
            "hybrid",
        }
        assert record["generation_modes"]["block_diffusion"]["new_tokens"] == 2
        assert record["latent_reconstruction_mse"] >= 0
        assert record["latent_vae_reconstruction_mse"] >= 0
        assert record["latent_refinement_reconstruction_mse"] >= 0
        assert record["latent_eval_steps"] == 1
        assert record["moe_timestep_tracked_selections"] >= 0
        assert len(record["moe_timestep_load_entropy"]) >= 1
        assert record["moe_noisy_step_load_entropy"] >= 0
        assert record["moe_polish_step_load_entropy"] >= 0
        assert record["lost_middle_worst_rank"] >= 1
        assert record["lost_middle_middle_rank"] >= 1
        assert len(record["lost_middle_positions"]) == 3
        assert len(record["lost_middle_target_ranks"]) == 3
        assert record["copy_target_rank"] >= 1
        assert 0.0 <= record["copy_target_probability"] <= 1.0
        assert record["recurrent_state_layers"] >= 0
        assert record["recurrent_state_max_norm"] >= 0
        assert record["recurrent_state_mean_update_norm"] >= 0
        assert record["recurrent_state_finite"] is True
        assert record["toy_reasoning_target_rank"] >= 1
        assert 0.0 <= record["toy_reasoning_target_probability"] <= 1.0
        assert record["toy_reasoning_prompt_token_ids"][0] == tiny_config().bos_token_id
    summary = (out_dir / "eval.md").read_text(encoding="utf-8")
    assert "Baseline Comparison" in summary
    assert "Generation Mode Smoke" in summary
    assert "Latent reconstruction MSE" in summary
    assert "MoE timestep load entropy" in summary
    assert "MoE noisy-step load entropy" in summary
    assert "Lost-middle worst rank" in summary
    assert "Copy target rank" in summary
    assert "Recurrent state max norm" in summary
    assert "Toy reasoning target rank" in summary
    assert "configs/lite_debug.yaml" in summary


def test_benchmark_cli_writes_comparison_outputs(tmp_path) -> None:
    out_dir = tmp_path / "benchmark"
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark.py",
            "--config",
            "configs/heavy_debug.yaml",
            "--compare-config",
            "configs/lite_debug.yaml",
            "--batch-size",
            "1",
            "--seq-len",
            "4",
            "--max-new-tokens",
            "1",
            "--out-dir",
            str(out_dir),
            "--seed",
            "67",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [
        json.loads(line)
        for line in (out_dir / "benchmark.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 2
    assert {row["variant"] for row in rows} == {"heavy", "lite"}
    assert all(row["ar_tokens_per_second"] >= 0 for row in rows)
    assert all(row["block_tokens_per_second_equivalent"] >= 0 for row in rows)
    assert all(row["parameter_bytes"] > 0 for row in rows)
    assert all(row["hidden_activation_bytes_estimate"] > 0 for row in rows)
    assert all(row["causal_loss"] > 0 for row in rows)
    assert all(row["diffusion_loss"] > 0 for row in rows)
    assert all(row["losses_finite"] is True for row in rows)
    assert all(
        row["fp16_cache_bits_baseline"] > row["mla_only_cache_bits"] for row in rows
    )
    assert all(row["mla_turboquant_cache_bits"] > 0 for row in rows)
    assert all(row["turboquant_reconstruction_mse"] >= 0 for row in rows)
    assert all(row["turboquant_reconstruction_max_abs_error"] >= 0 for row in rows)
    assert all(row["turboquant_cache_roundtrip_ms"] >= 0 for row in rows)
    assert all(row["turboquant_rotation_dim"] > 0 for row in rows)
    assert all(row["turboquant_dense_qr_rotation_ms"] >= 0 for row in rows)
    assert all(row["turboquant_structured_sign_permutation_ms"] >= 0 for row in rows)
    summary = (out_dir / "benchmark.md").read_text(encoding="utf-8")
    assert "Alexandros Benchmark" in summary
    assert "Act MB est" in summary
    assert "Causal loss" in summary
    assert "TQ mse" in summary
    assert "QR ms" in summary
    assert "Struct ms" in summary
    assert "configs/lite_debug.yaml" in summary


def test_config_validation_rejects_duplicate_special_tokens() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        AlexandrosConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            max_position_embeddings=16,
            moe_num_experts=1,
            moe_top_k=1,
            moe_expert_hidden_size=8,
            kv_lora_rank=4,
            latent_dim=4,
            latent_slots=1,
            diffusion_steps=1,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            mask_token_id=2,
            depth_lora_rank=1,
            ttt_rank=1,
        )


def test_config_validation_rejects_odd_attention_head_dim() -> None:
    with pytest.raises(ValueError, match="head_dim must be even"):
        AlexandrosConfig(
            vocab_size=16,
            hidden_size=6,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=16,
            moe_num_experts=1,
            moe_top_k=1,
            moe_expert_hidden_size=8,
            kv_lora_rank=4,
            latent_dim=4,
            latent_slots=1,
            diffusion_steps=1,
            depth_lora_rank=1,
            ttt_rank=1,
        )


def test_config_validation_rejects_bad_stability_knobs() -> None:
    for field, value, message in [
        ("rope_theta", float("nan"), "rope_theta"),
        ("rope_theta", 0.0, "rope_theta"),
        ("rms_norm_eps", float("inf"), "rms_norm_eps"),
        ("rms_norm_eps", 0.0, "rms_norm_eps"),
        ("dropout", float("nan"), "dropout"),
        ("dropout", 1.0, "dropout"),
        ("dropout", True, "dropout"),
        ("router_bias_update_rate", float("nan"), "router_bias_update_rate"),
        ("router_bias_update_rate", -1e-3, "router_bias_update_rate"),
        ("router_logit_clip", float("nan"), "router_logit_clip"),
        ("router_bias_clip", float("inf"), "router_bias_clip"),
        ("router_load_ema_decay", float("nan"), "router_load_ema_decay"),
        ("latent_update_clip", float("nan"), "latent_update_clip"),
        ("deltanet_state_clip", float("nan"), "deltanet_state_clip"),
        ("deltanet_state_clip", 0.0, "deltanet_state_clip"),
        ("act_threshold", float("nan"), "act_threshold"),
        ("act_threshold", 0.0, "act_threshold"),
        ("act_ponder_cost", float("nan"), "act_ponder_cost"),
        ("act_ponder_cost", -1.0, "act_ponder_cost"),
    ]:
        cfg = tiny_config()
        setattr(cfg, field, value)
        with pytest.raises(ValueError, match=message):
            AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="router_logit_clip"):
        cfg = tiny_config()
        cfg.router_logit_clip = 0.0
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="bitnet_activation_bits"):
        cfg = tiny_config("lite")
        cfg.bitnet_activation_bits = 0
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="latent_update_clip"):
        cfg = tiny_config()
        cfg.latent_update_clip = 0.0
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="deltanet_chunk_size"):
        cfg = tiny_config()
        cfg.deltanet_chunk_size = -1
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="router_bias_update_interval"):
        cfg = tiny_config()
        cfg.router_bias_update_interval = 0
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="router_load_ema_decay"):
        cfg = tiny_config()
        cfg.router_load_ema_decay = 1.0
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="diffusion_attention_mask_mode"):
        cfg = tiny_config()
        cfg.diffusion_attention_mask_mode = "full"
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="diffusion_objective"):
        cfg = tiny_config()
        cfg.diffusion_objective = "paperish"
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="diffusion_loss_weighting"):
        cfg = tiny_config()
        cfg.diffusion_loss_weighting = "paperish"
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="diffusion_rb_chunk_size"):
        cfg = tiny_config()
        cfg.diffusion_rb_chunk_size = -1
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="attention_backend"):
        cfg = tiny_config()
        cfg.attention_backend = "triton"
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="linear_mixer_backend"):
        cfg = tiny_config()
        cfg.linear_mixer_backend = "unknown"
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="moe_token_state_routing"):
        cfg = tiny_config()
        cfg.moe_token_state_routing = 1  # type: ignore[assignment]
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="moe_position_routing"):
        cfg = tiny_config()
        cfg.moe_position_routing = 1  # type: ignore[assignment]
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="moe_position_buckets"):
        cfg = tiny_config()
        cfg.moe_position_buckets = 0
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="vocab_size must be an integer"):
        cfg = tiny_config()
        cfg.vocab_size = True  # type: ignore[assignment]
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="moe_top_k must be an integer"):
        cfg = tiny_config()
        cfg.moe_top_k = 1.5  # type: ignore[assignment]
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="attention_layer_indices must be an integer"):
        cfg = tiny_config()
        cfg.attention_layer_indices = [True]  # type: ignore[list-item]
        AlexandrosConfig.from_dict(cfg.to_dict())


def test_explicit_attention_layer_schedule_round_trips() -> None:
    cfg = tiny_config()
    cfg.attention_layer_indices = [0, 2]
    restored = AlexandrosConfig.from_dict(cfg.to_dict())
    assert restored.attention_layers() == (0, 2)
    assert count_attention_layers(restored) == 2
    model = AlexandrosModel(restored)
    assert [layer.is_attention for layer in model.layers] == [True, False, True]
    with pytest.raises(ValueError, match="duplicates"):
        cfg = tiny_config()
        cfg.attention_layer_indices = [1, 1]
        AlexandrosConfig.from_dict(cfg.to_dict())
    with pytest.raises(ValueError, match="within"):
        cfg = tiny_config()
        cfg.attention_layer_indices = [cfg.num_hidden_layers]
        AlexandrosConfig.from_dict(cfg.to_dict())


def test_heavy_forward_and_generate_shape() -> None:
    cfg = tiny_config("heavy")
    model = AlexandrosForCausalLM(cfg)
    input_ids = torch.randint(4, cfg.vocab_size, (2, 8))
    out = model(input_ids, labels=input_ids, output_hidden_states=True)
    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert out.loss is not None
    generated = model.generate(input_ids[:, :3], max_new_tokens=2)
    assert generated.shape[0] == 2
    assert 3 <= generated.shape[1] <= 5
    assert torch.equal(generated[:, :3], input_ids[:, :3])
    unchanged = model.generate(input_ids[:, :3], max_new_tokens=0)
    assert torch.equal(unchanged, input_ids[:, :3])
    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(input_ids[:, :3], max_new_tokens=-1)
    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(input_ids[:, :3], max_new_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(input_ids[:, :3], max_new_tokens=1.5)  # type: ignore[arg-type]


def test_dropout_policy_is_train_only() -> None:
    cfg = tiny_config("heavy")
    cfg.dropout = 0.8
    model = AlexandrosForCausalLM(cfg)
    input_ids = torch.randint(4, cfg.vocab_size, (1, 8))
    model.eval()
    eval_a = model(input_ids).logits
    eval_b = model(input_ids).logits
    assert torch.allclose(eval_a, eval_b)
    model.train()
    torch.manual_seed(1)
    train_a = model(input_ids).logits
    torch.manual_seed(2)
    train_b = model(input_ids).logits
    assert not torch.allclose(train_a, train_b)


def test_cached_generation_matches_uncached_generation() -> None:
    torch.manual_seed(1234)
    cfg = tiny_config("heavy")
    model = AlexandrosForCausalLM(cfg).eval()
    input_ids = torch.tensor([[cfg.bos_token_id, 8, 9, 10]], dtype=torch.long)
    uncached = model.generate(input_ids, max_new_tokens=3, use_cache=False)
    cached = model.generate(input_ids, max_new_tokens=3, use_cache=True)
    assert torch.equal(cached, uncached)


def test_padded_batch_generation_matches_compact_rows() -> None:
    torch.manual_seed(4321)
    cfg = tiny_config("heavy")
    model = AlexandrosForCausalLM(cfg).eval()
    padded = torch.tensor(
        [
            [cfg.bos_token_id, 8, cfg.pad_token_id, cfg.pad_token_id],
            [cfg.bos_token_id, 9, 10, 11],
        ],
        dtype=torch.long,
    )
    row_outputs = [
        model.generate(padded[0:1, :2], max_new_tokens=2, use_cache=False).squeeze(0),
        model.generate(padded[1:2], max_new_tokens=2, use_cache=False).squeeze(0),
    ]
    expected = padded.new_full(
        (2, max(row.numel() for row in row_outputs)), cfg.pad_token_id
    )
    for row_idx, row in enumerate(row_outputs):
        expected[row_idx, : row.numel()] = row

    uncached = model.generate(padded, max_new_tokens=2, use_cache=False)
    cached = model.generate(padded, max_new_tokens=2, use_cache=True)

    assert torch.equal(uncached, expected)
    assert torch.equal(cached, expected)
    with pytest.raises(ValueError, match="left-padded"):
        model.generate(
            torch.tensor([[cfg.pad_token_id, cfg.bos_token_id, 8]], dtype=torch.long),
            max_new_tokens=1,
        )
    with pytest.raises(ValueError, match="right-padded"):
        model.generate(
            torch.tensor([[cfg.bos_token_id, cfg.pad_token_id, 8]], dtype=torch.long),
            max_new_tokens=1,
        )


def test_bidirectional_diffusion_attention_does_not_change_ar_causality() -> None:
    torch.manual_seed(123)
    cfg = tiny_config("heavy")
    cfg.num_hidden_layers = 1
    cfg.attention_layer_indices = [0]
    cfg.enable_adaptive_depth = False
    cfg.diffusion_attention_mask_mode = "bidirectional"
    model = AlexandrosForDiffusionLM(cfg).eval()
    input_a = torch.tensor([[cfg.bos_token_id, 8, 9, 10]], dtype=torch.long)
    input_b = torch.tensor([[cfg.bos_token_id, 8, 9, 11]], dtype=torch.long)

    with torch.no_grad():
        ar_a = model(input_a).logits[:, 0, :]
        ar_b = model(input_b).logits[:, 0, :]
        diff_a = model(input_a, diffusion_timestep=0).logits[:, 0, :]
        diff_b = model(input_b, diffusion_timestep=0).logits[:, 0, :]

    assert torch.allclose(ar_a, ar_b, atol=1e-6)
    assert not torch.allclose(diff_a, diff_b, atol=1e-6)


def test_diffusion_forward_rejects_cache_reuse() -> None:
    cfg = tiny_config("heavy")
    cfg.diffusion_attention_mask_mode = "bidirectional"
    model = AlexandrosForDiffusionLM(cfg)
    input_ids = torch.tensor([[cfg.bos_token_id, 8, 9]], dtype=torch.long)
    with pytest.raises(ValueError, match="use_cache"):
        model(input_ids, diffusion_timestep=0, use_cache=True)


def test_ar_generation_sampling_controls() -> None:
    cfg = tiny_config("heavy")
    model = AlexandrosForCausalLM(cfg).eval()
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    assert model._select_next_token(logits, do_sample=True, top_k=1).item() == 3
    with pytest.raises(ValueError, match="temperature"):
        model._select_next_token(logits, do_sample=True, temperature=0.0)
    with pytest.raises(ValueError, match="top_k"):
        model._select_next_token(logits, do_sample=True, top_k=0)
    with pytest.raises(ValueError, match="positive integer"):
        model._select_next_token(logits, do_sample=True, top_k=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        model._select_next_token(logits, do_sample=True, top_k=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="top_p"):
        model._select_next_token(logits, do_sample=True, top_p=1.5)
    with pytest.raises(ValueError, match="finite"):
        model._select_next_token(logits, do_sample=True, temperature=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        model._select_next_token(logits, do_sample=True, top_p=float("nan"))
    pad_high_logits = torch.tensor([[100.0, 1.0, 2.0, 3.0]])
    assert model._select_next_token(pad_high_logits).item() == 3
    with pytest.raises(ValueError, match="repetition_penalty"):
        model._apply_repetition_penalty(logits, torch.tensor([[1, 2]]), 0.0)
    with pytest.raises(ValueError, match="finite"):
        model._apply_repetition_penalty(logits, torch.tensor([[1, 2]]), float("nan"))
    penalized = model._apply_repetition_penalty(logits, torch.tensor([[3]]), 2.0)
    assert penalized[0, 3] < logits[0, 3]
    assert model._should_stop(torch.tensor([[2]]), cfg.eos_token_id, None)
    assert model._should_stop(torch.tensor([[8]]), None, [8])
    assert model._should_stop(
        torch.tensor([[9]]),
        None,
        None,
        generated=torch.tensor([[cfg.bos_token_id, 8, 9]]),
        stop_sequences=((8, 9),),
    )
    with pytest.raises(ValueError, match="empty"):
        model._normalize_stop_sequences([[]])
    input_ids = torch.tensor([[cfg.bos_token_id, 8, 9]], dtype=torch.long)
    with pytest.raises(ValueError, match="eos_token_id"):
        model.generate(input_ids, max_new_tokens=1, eos_token_id=cfg.vocab_size)
    with pytest.raises(ValueError, match="stop_token_ids"):
        model.generate(input_ids, max_new_tokens=1, stop_token_ids=[cfg.vocab_size])
    with pytest.raises(ValueError, match="stop_sequences"):
        model.generate(input_ids, max_new_tokens=1, stop_sequences=[[True]])  # type: ignore[list-item]
    generated = model.generate(
        input_ids,
        max_new_tokens=2,
        do_sample=True,
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    assert generated.shape == (1, 5)
    scripted_tokens = [torch.tensor([[8]]), torch.tensor([[9]])]

    def select_next_token(logits: torch.Tensor, **kwargs) -> torch.Tensor:
        return scripted_tokens.pop(0)

    model._select_next_token = select_next_token  # type: ignore[method-assign]
    stopped = model.generate(
        input_ids,
        max_new_tokens=4,
        eos_token_id=None,
        stop_sequences=[[8, 9]],
    )
    assert stopped.tolist()[0][-2:] == [8, 9]
    assert stopped.size(1) == input_ids.size(1) + 2


def test_cache_forward_accepts_current_token_attention_mask() -> None:
    cfg = tiny_config("heavy")
    model = AlexandrosForCausalLM(cfg).eval()
    prompt = torch.tensor([[cfg.bos_token_id, 8, 9]], dtype=torch.long)
    next_token = torch.tensor([[10]], dtype=torch.long)
    first = model(prompt, use_cache=True)
    second = model(
        next_token,
        past_key_values=first.past_key_values,
        past_ssm_states=first.past_ssm_states,
        use_cache=True,
    )
    assert second.logits.shape == (1, 1, cfg.vocab_size)
    assert second.past_key_values is not None
    cached_lengths = [
        cache["c_kv"].size(1) for cache in second.past_key_values if cache is not None
    ]
    assert cached_lengths and all(
        length == prompt.size(1) + 1 for length in cached_lengths
    )
    with pytest.raises(ValueError, match="right-padded"):
        model(
            next_token,
            attention_mask=torch.tensor([[True, False, True, True]]),
            past_key_values=first.past_key_values,
            past_ssm_states=first.past_ssm_states,
            use_cache=True,
        )
    assert first.past_key_values is not None
    bad_kv = list(first.past_key_values)
    attention_idx = next(idx for idx, cache in enumerate(bad_kv) if cache is not None)
    bad_kv[attention_idx] = {"c_kv": torch.zeros(2, 1, cfg.kv_lora_rank)}
    with pytest.raises(ValueError, match="past_key_values c_kv"):
        model(
            next_token,
            past_key_values=bad_kv,
            past_ssm_states=first.past_ssm_states,
            use_cache=True,
        )
    assert first.past_ssm_states is not None
    bad_ssm = list(first.past_ssm_states)
    ssm_idx = next(idx for idx, state in enumerate(bad_ssm) if state is not None)
    bad_ssm[ssm_idx] = torch.zeros(1, cfg.hidden_size + 1)
    with pytest.raises(ValueError, match="past_ssm_states"):
        model(
            next_token,
            past_key_values=first.past_key_values,
            past_ssm_states=bad_ssm,
            use_cache=True,
        )


def test_turboquant_attention_cache_can_decode_and_reorder() -> None:
    torch.manual_seed(41)
    cfg = tiny_config("heavy")
    cfg.turboquant_bits = 8
    cfg.use_turboquant_cache = True
    compressed = AlexandrosForCausalLM(cfg).eval()
    plain_cfg = AlexandrosConfig.from_dict(cfg.to_dict())
    plain_cfg.use_turboquant_cache = False
    plain = AlexandrosForCausalLM(plain_cfg).eval()
    plain.load_state_dict(compressed.state_dict())

    prompt = torch.tensor([[cfg.bos_token_id, 8, 9]], dtype=torch.long)
    next_token = torch.tensor([[10]], dtype=torch.long)
    first = compressed(prompt, use_cache=True)
    second = compressed(
        next_token,
        past_key_values=first.past_key_values,
        past_ssm_states=first.past_ssm_states,
        use_cache=True,
    )
    assert torch.isfinite(second.logits).all()
    assert second.past_key_values is not None
    packets = [
        cache["c_kv_packet"]
        for cache in second.past_key_values
        if cache is not None and "c_kv_packet" in cache
    ]
    assert packets
    assert all(isinstance(packet, TurboQuantPacket) for packet in packets)
    assert all(packet.q.dtype == torch.int8 for packet in packets)
    assert all(packet.q.size(1) == prompt.size(1) + 1 for packet in packets)
    assert compressed(past_key_values=None, input_ids=prompt).logits.shape == (
        1,
        3,
        cfg.vocab_size,
    )
    bad_packet = TurboQuantPacket(
        q=packets[0].q,
        scale=packets[0].scale.squeeze(-1),
        bits=packets[0].bits,
        original_dtype=packets[0].original_dtype,
        rotation_seed=packets[0].rotation_seed,
    )
    bad_cache = list(second.past_key_values)
    bad_cache_idx = next(
        idx
        for idx, cache in enumerate(bad_cache)
        if cache is not None and "c_kv_packet" in cache
    )
    bad_cache[bad_cache_idx] = {"c_kv_packet": bad_packet}
    with pytest.raises(ValueError, match="scale shape"):
        compressed(
            next_token,
            past_key_values=bad_cache,
            past_ssm_states=second.past_ssm_states,
            use_cache=True,
        )

    plain_first = plain(prompt, use_cache=True)
    plain_second = plain(
        next_token,
        past_key_values=plain_first.past_key_values,
        past_ssm_states=plain_first.past_ssm_states,
        use_cache=True,
    )
    max_diff = (second.logits - plain_second.logits).abs().max().item()
    assert max_diff < 0.5

    batch_prompt = torch.tensor(
        [
            [cfg.bos_token_id, 8, 9],
            [cfg.bos_token_id, 10, 11],
        ],
        dtype=torch.long,
    )
    batch_first = compressed(batch_prompt, use_cache=True)
    assert batch_first.past_key_values is not None
    reordered, _ = reorder_generation_cache(
        past_key_values=batch_first.past_key_values,
        past_ssm_states=batch_first.past_ssm_states,
        beam_idx=torch.tensor([1, 0], dtype=torch.long),
    )
    assert reordered is not None
    for old_cache, new_cache in zip(batch_first.past_key_values, reordered):
        if old_cache is not None and "c_kv_packet" in old_cache:
            assert new_cache is not None
            old_packet = old_cache["c_kv_packet"]
            new_packet = new_cache["c_kv_packet"]
            assert isinstance(old_packet, TurboQuantPacket)
            assert isinstance(new_packet, TurboQuantPacket)
            assert torch.equal(new_packet.q[0], old_packet.q[1])
            if old_packet.qjl_residual_norm is not None:
                assert new_packet.qjl_residual_norm is not None
                assert torch.equal(
                    new_packet.qjl_residual_norm[0], old_packet.qjl_residual_norm[1]
                )


def test_cache_reorder_utility_reorders_attention_and_ssm_state() -> None:
    cfg = tiny_config("heavy")
    model = AlexandrosForCausalLM(cfg).eval()
    prompt = torch.tensor(
        [
            [cfg.bos_token_id, 8, 9],
            [cfg.bos_token_id, 10, 11],
        ],
        dtype=torch.long,
    )
    first = model(prompt, use_cache=True)
    assert first.past_key_values is not None
    assert first.past_ssm_states is not None
    beam_idx = torch.tensor([1, 0], dtype=torch.long)
    reordered_kv, reordered_ssm = reorder_generation_cache(
        past_key_values=first.past_key_values,
        past_ssm_states=first.past_ssm_states,
        beam_idx=beam_idx,
    )
    assert reordered_kv is not None
    assert reordered_ssm is not None
    for old_cache, new_cache in zip(first.past_key_values, reordered_kv):
        if old_cache is not None:
            assert new_cache is not None
            assert torch.equal(new_cache["c_kv"][0], old_cache["c_kv"][1])
    for old_state, new_state in zip(first.past_ssm_states, reordered_ssm):
        if old_state is not None:
            assert new_state is not None
            assert torch.equal(new_state[0], old_state[1])
    with pytest.raises(ValueError, match="outside"):
        reorder_generation_cache(
            past_key_values=first.past_key_values,
            past_ssm_states=first.past_ssm_states,
            beam_idx=torch.tensor([0, 2], dtype=torch.long),
        )


def test_generation_request_schema_validates_and_runs() -> None:
    cfg = tiny_config("heavy")
    model = AlexandrosForDiffusionLM(cfg).eval()
    request = GenerationRequest(
        input_ids=[[cfg.bos_token_id, 8, 9]],
        max_new_tokens=2,
        mode="autoregressive",
        use_cache=True,
    )
    assert GenerationRequest.from_dict(request.to_dict()) == request
    output = model.generate(request.to_tensor(), **request.generate_kwargs())
    assert output.shape == (1, 5)
    diffusion_request = GenerationRequest(
        input_ids=[[cfg.bos_token_id, 8, 9]],
        max_new_tokens=2,
        mode="block_diffusion",
        do_sample=True,
        top_k=4,
        steps=1,
        block_size=1,
        confidence_schedule="all",
        remask_low_confidence=True,
    )
    assert diffusion_request.generate_kwargs()["steps"] == 1
    assert diffusion_request.generate_kwargs()["top_k"] == 4
    assert diffusion_request.generate_kwargs()["block_size"] == 1
    assert diffusion_request.generate_kwargs()["confidence_schedule"] == "all"
    assert diffusion_request.generate_kwargs()["remask_low_confidence"] is True
    with pytest.raises(ValueError, match="same length"):
        GenerationRequest(input_ids=[[1, 2], [1]], max_new_tokens=1)
    with pytest.raises(ValueError, match="integer"):
        GenerationRequest(input_ids=[[1, True]], max_new_tokens=1)
    with pytest.raises(ValueError, match="integer"):
        GenerationRequest(input_ids=[[1, "2"]], max_new_tokens=1)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="integer"):
        GenerationRequest(input_ids=[[1, 2]], stop_token_ids=[True])
    with pytest.raises(ValueError, match="integer"):
        GenerationRequest(input_ids=[[1, 2]], stop_sequences=[[1, "2"]])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerationRequest(input_ids=[[1, 2]], max_new_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerationRequest(input_ids=[[1, 2]], max_new_tokens=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="temperature"):
        GenerationRequest(input_ids=[[1, 2]], temperature=float("nan"))
    with pytest.raises(ValueError, match="top_k"):
        GenerationRequest(input_ids=[[1, 2]], top_k=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="top_p"):
        GenerationRequest(input_ids=[[1, 2]], top_p=float("nan"))
    with pytest.raises(ValueError, match="repetition_penalty"):
        GenerationRequest(input_ids=[[1, 2]], repetition_penalty=float("nan"))
    with pytest.raises(ValueError, match="steps"):
        GenerationRequest(input_ids=[[1, 2]], mode="block_diffusion", steps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="steps"):
        GenerationRequest(input_ids=[[1, 2]], mode="block_diffusion", steps=0)
    with pytest.raises(ValueError, match="block_size"):
        GenerationRequest(input_ids=[[1, 2]], mode="block_diffusion", block_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="confidence_schedule"):
        GenerationRequest(
            input_ids=[[1, 2]], mode="block_diffusion", confidence_schedule="unknown"
        )
    with pytest.raises(ValueError, match="remask_low_confidence"):
        GenerationRequest(
            input_ids=[[1, 2]],
            mode="block_diffusion",
            remask_low_confidence=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="latent_steps"):
        GenerationRequest(input_ids=[[1, 2]], mode="latent_reasoning", latent_steps=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="use_cache"):
        GenerationRequest(
            input_ids=[[1, 2]], max_new_tokens=1, mode="block_diffusion", use_cache=True
        )
    with pytest.raises(ValueError, match="sampling"):
        GenerationRequest(
            input_ids=[[1, 2]], max_new_tokens=1, mode="latent_reasoning", top_k=4
        )


def test_long_sequence_recurrent_state_norms_stay_finite() -> None:
    cfg = tiny_config("heavy")
    cfg.max_position_embeddings = 128
    model = AlexandrosForCausalLM(cfg).eval()
    input_ids = torch.randint(4, cfg.vocab_size, (1, 64))
    out = model(input_ids, use_cache=True)
    assert out.past_ssm_states is not None
    norms = [state.norm() for state in out.past_ssm_states if state is not None]
    assert norms
    assert all(torch.isfinite(norm) for norm in norms)
    assert max(norm.item() for norm in norms) < 1e4


def test_sequence_length_limit_is_enforced() -> None:
    cfg = tiny_config("heavy")
    cfg.max_position_embeddings = 4
    model = AlexandrosForCausalLM(cfg)
    exact = model(torch.randint(4, cfg.vocab_size, (1, 4)))
    assert exact.logits.shape == (1, 4, cfg.vocab_size)
    with pytest.raises(ValueError, match="exceeds max_position_embeddings"):
        model(torch.randint(4, cfg.vocab_size, (1, 5)))
    with pytest.raises(ValueError, match="exceeds max_position_embeddings"):
        model.generate(torch.randint(4, cfg.vocab_size, (1, 3)), max_new_tokens=2)
    padded = model.generate(
        torch.tensor([[cfg.bos_token_id, 8, cfg.pad_token_id]], dtype=torch.long),
        max_new_tokens=1,
    )
    assert padded.shape == (1, 3)
    assert padded[0, :2].tolist() == [cfg.bos_token_id, 8]


def test_empty_and_all_pad_inputs_are_rejected() -> None:
    cfg = tiny_config("heavy")
    model = AlexandrosForCausalLM(cfg)
    with pytest.raises(ValueError, match="sequence length must be > 0"):
        model(torch.empty((1, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="at least one non-pad token"):
        model(torch.full((1, 4), cfg.pad_token_id, dtype=torch.long))
    with pytest.raises(ValueError, match="outside"):
        model(torch.tensor([[cfg.bos_token_id, cfg.vocab_size]], dtype=torch.long))
    with pytest.raises(ValueError, match="integer tensor"):
        model(torch.tensor([[float(cfg.bos_token_id), 8.0]], dtype=torch.float32))


def test_bad_attention_masks_are_rejected() -> None:
    cfg = tiny_config("heavy")
    model = AlexandrosForCausalLM(cfg)
    input_ids = torch.tensor([[cfg.bos_token_id, 8, 9]], dtype=torch.long)
    with pytest.raises(ValueError, match="batch size"):
        model(input_ids, attention_mask=torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="length"):
        model(input_ids, attention_mask=torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="left-padded"):
        model(input_ids, attention_mask=torch.tensor([[False, True, True]]))
    with pytest.raises(ValueError, match="right-padded"):
        model(input_ids, attention_mask=torch.tensor([[True, False, True]]))
    with pytest.raises(ValueError, match="0/1"):
        model(input_ids, attention_mask=torch.tensor([[1, 2, 1]]))
    with pytest.raises(ValueError, match="finite"):
        model(input_ids, attention_mask=torch.tensor([[1.0, float("nan"), 1.0]]))


def test_right_padding_does_not_change_unpadded_prefix_logits() -> None:
    cfg = tiny_config("heavy")
    model = AlexandrosForCausalLM(cfg).eval()
    unpadded = torch.tensor([[cfg.bos_token_id, 8, 9]], dtype=torch.long)
    padded = torch.tensor(
        [
            [cfg.bos_token_id, 8, 9, cfg.pad_token_id],
            [cfg.bos_token_id, 10, 11, 12],
        ],
        dtype=torch.long,
    )
    mask = padded.ne(cfg.pad_token_id)
    with torch.no_grad():
        unpadded_logits = model(unpadded).logits
        padded_logits = model(padded, attention_mask=mask).logits
    assert torch.allclose(padded_logits[0, :3], unpadded_logits[0], atol=1e-5)
    assert torch.isfinite(padded_logits).all()


def test_bad_labels_are_rejected() -> None:
    cfg = tiny_config("heavy")
    model = AlexandrosForCausalLM(cfg)
    with pytest.raises(ValueError, match="same shape"):
        model(
            torch.randint(4, cfg.vocab_size, (1, 3)),
            labels=torch.randint(4, cfg.vocab_size, (1, 2)),
        )
    with pytest.raises(ValueError, match="sequence length >= 2"):
        ids = torch.randint(4, cfg.vocab_size, (1, 1))
        model(ids, labels=ids)
    with pytest.raises(ValueError, match="torch.long"):
        ids = torch.randint(4, cfg.vocab_size, (1, 3))
        model(ids, labels=ids.float())
    with pytest.raises(ValueError, match="non-ignored"):
        ids = torch.randint(4, cfg.vocab_size, (1, 3))
        model(ids, labels=torch.full_like(ids, -100))
    with pytest.raises(ValueError, match="outside"):
        ids = torch.randint(4, cfg.vocab_size, (1, 3))
        bad_labels = ids.clone()
        bad_labels[:, -1] = cfg.vocab_size
        model(ids, labels=bad_labels)


def test_lite_forward_uses_bitlinear() -> None:
    cfg = tiny_config("lite")
    cfg.bitnet_activation_bits = 5
    model = AlexandrosForCausalLM(cfg)
    assert any(isinstance(module, BitLinear) for module in model.modules())
    assert all(
        module.activation_bits == cfg.bitnet_activation_bits
        for module in model.modules()
        if isinstance(module, BitLinear)
    )
    input_ids = torch.randint(4, cfg.vocab_size, (1, 6))
    out = model(input_ids)
    assert torch.isfinite(out.logits).all()


@pytest.mark.parametrize("variant", ["heavy", "lite"])
def test_forward_backward_is_finite_for_heavy_and_lite(variant: str) -> None:
    cfg = tiny_config(variant)
    model = AlexandrosForCausalLM(cfg)
    input_ids = torch.randint(4, cfg.vocab_size, (1, 6))
    out = model(input_ids, labels=input_ids)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grad_norms = [
        param.grad.detach().norm()
        for param in model.parameters()
        if param.grad is not None
    ]
    assert grad_norms
    assert all(torch.isfinite(norm) for norm in grad_norms)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA mixed precision smoke requires CUDA"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_mixed_precision_forward_backward_smoke(dtype: torch.dtype) -> None:
    cfg = tiny_config("heavy")
    model = AlexandrosForCausalLM(cfg).cuda()
    input_ids = torch.randint(4, cfg.vocab_size, (1, 6), device="cuda")
    with torch.autocast(device_type="cuda", dtype=dtype):
        out = model(input_ids, labels=input_ids)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grad_norms = [
        param.grad.detach().float().norm()
        for param in model.parameters()
        if param.grad is not None
    ]
    assert grad_norms
    assert all(torch.isfinite(norm) for norm in grad_norms)


def test_lite_tiny_trains_for_several_steps_without_nans() -> None:
    torch.manual_seed(123)
    cfg = tiny_config("lite")
    cfg.enable_adaptive_depth = False
    model = AlexandrosForCausalLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(4, cfg.vocab_size, (1, 6))
    input_ids[:, 0] = cfg.bos_token_id
    losses: list[float] = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        out = model(input_ids, labels=input_ids)
        assert out.loss is not None
        assert torch.isfinite(out.loss)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        losses.append(float(out.loss.detach().item()))
    assert all(loss > 0 for loss in losses)


def test_heavy_tiny_overfits_toy_sequence() -> None:
    torch.manual_seed(0)
    cfg = AlexandrosConfig(
        variant="heavy",
        vocab_size=32,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        linear_attention_ratio=1,
        moe_num_experts=2,
        moe_num_shared_experts=0,
        moe_top_k=1,
        moe_expert_hidden_size=32,
        kv_lora_rank=8,
        latent_dim=12,
        latent_slots=2,
        diffusion_steps=2,
        mask_token_id=3,
        enable_adaptive_depth=False,
        depth_lora_rank=4,
        ttt_rank=4,
    )
    model = AlexandrosForCausalLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    input_ids = torch.tensor([[cfg.bos_token_id, 4, 5, 6, 7, 8]], dtype=torch.long)
    losses: list[float] = []
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        out = model(input_ids, labels=input_ids)
        assert out.loss is not None
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        losses.append(float(out.loss.detach().item()))
    assert losses[-1] < 0.2
    assert losses[-1] < losses[0] * 0.25


def test_parameter_report_tracks_active_moe_parameters() -> None:
    cfg = tiny_config("heavy")
    model = AlexandrosForCausalLM(cfg)
    report = summarize_parameters(model)
    assert report.total_parameters > 0
    assert report.trainable_parameters == report.total_parameters
    assert report.routed_expert_parameters > report.active_routed_expert_parameters
    assert report.active_parameters_per_token < report.total_parameters


def test_cache_memory_estimator_reports_compression() -> None:
    cfg = tiny_config("heavy")
    report = estimate_cache_memory(cfg, batch_size=2, sequence_length=16, dtype_bits=16)
    expected_standard_elements = 2 * cfg.num_attention_heads * cfg.head_dim
    expected_mla_elements = cfg.mla_d_c + cfg.mla_d_r
    assert report.standard_mha_elements_per_token == expected_standard_elements
    assert report.mla_elements_per_token == expected_mla_elements
    assert report.mla_compressed_elements_per_token == cfg.mla_d_c
    assert report.mla_rope_elements_per_token == cfg.mla_d_r
    assert report.attention_layers > 0
    assert report.standard_kv_bits == (
        2 * 16 * expected_standard_elements * 16 * report.attention_layers
    )
    assert report.mla_kv_bits == (
        2 * 16 * expected_mla_elements * 16 * report.attention_layers
    )
    assert report.mla_compression_ratio == pytest.approx(
        expected_standard_elements / expected_mla_elements
    )
    assert report.standard_kv_bits > report.mla_kv_bits
    assert report.mla_kv_bits > report.turboquant_mla_bits
    rope_cfg_data = cfg.to_dict()
    rope_cfg_data["mla_rope_dim"] = 2
    rope_cfg = AlexandrosConfig.from_dict(rope_cfg_data)
    rope_report = estimate_cache_memory(
        rope_cfg,
        batch_size=2,
        sequence_length=16,
        dtype_bits=16,
    )
    assert rope_report.mla_rope_elements_per_token == 2
    assert rope_report.mla_elements_per_token == rope_cfg.kv_lora_rank + 2
    assert rope_report.turboquant_mla_bits == (
        2
        * 16
        * (
            rope_cfg.kv_lora_rank * rope_cfg.turboquant_bits
            + 16
            + rope_cfg.mla_rope_dim * 16
        )
        * rope_report.attention_layers
    )
    flops = estimate_flops(cfg, batch_size=2, sequence_length=16)
    rope_flops = estimate_flops(rope_cfg, batch_size=2, sequence_length=16)
    assert flops.prefill_flops > flops.decode_token_flops > 0
    assert rope_flops.prefill_flops > rope_flops.decode_token_flops > 0
    assert flops.attention_layers + flops.linear_mixer_layers == cfg.num_hidden_layers
    with pytest.raises(ValueError, match="batch_size"):
        estimate_cache_memory(cfg, batch_size=0, sequence_length=16)
    with pytest.raises(ValueError, match="sequence_length"):
        estimate_flops(cfg, batch_size=1, sequence_length=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dtype_bits"):
        estimate_cache_memory(cfg, batch_size=1, sequence_length=16, dtype_bits=0)


def test_bitlinear_ternary_shadow_weight() -> None:
    layer = BitLinear(5, 3)
    q = layer.ternary_weight()
    scale = layer.weight.abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
    ternary = torch.round(q / scale).unique().tolist()
    assert set(float(x) for x in ternary).issubset({-1.0, 0.0, 1.0})
    y = layer(torch.randn(2, 5)).sum()
    y.backward()
    assert layer.weight.grad is not None


def test_bitlinear_packed_ternary_export_round_trip(tmp_path) -> None:
    layer = BitLinear(5, 3, bias=False)
    ternary, scale = ternary_codes_and_scales(layer.weight)
    codes = ternary_to_codes(ternary)
    packed, padding = pack_ternary_codes(codes)
    restored_codes = unpack_ternary_codes(
        packed, shape=tuple(codes.shape), padding=padding
    )
    assert torch.equal(restored_codes, codes.cpu())
    restored = codes_to_ternary(restored_codes).to(scale.dtype) * scale.cpu()
    assert torch.allclose(restored, layer.ternary_weight().detach().cpu())
    with pytest.raises(ValueError, match="ternary codes"):
        codes_to_ternary(torch.tensor([0, 1, 2, 3], dtype=torch.uint8))
    with pytest.raises(ValueError, match="reserved"):
        unpack_ternary_codes(
            torch.tensor([0b00000011], dtype=torch.uint8), shape=(1, 4)
        )

    cfg = tiny_config("lite")
    model = AlexandrosForCausalLM(cfg)
    state = export_packed_bitlinear_state(model)
    assert state["format"] == "alexandros-packed-bitlinear"
    assert state["layers"]
    first_layer = next(iter(state["layers"].values()))
    assert first_layer["encoding"] == "2bit_ternary_0_zero_1_pos_2_neg"
    assert first_layer["packed_weight"].dtype == torch.uint8
    assert first_layer["scale"].dtype == torch.float32

    export_path = tmp_path / "packed_bitlinear.pt"
    save_packed_bitlinear_state(model, export_path)
    loaded = torch.load(export_path, map_location="cpu", weights_only=False)
    assert loaded["format"] == "alexandros-packed-bitlinear"
    assert loaded["layers"].keys() == state["layers"].keys()


def test_moe_routing_and_bias_update() -> None:
    cfg = tiny_config()
    cfg.router_logit_clip = 0.1
    cfg.router_bias_clip = 0.05
    moe = MoEFeedForward(cfg)
    x = torch.randn(2, 5, cfg.hidden_size) * 1000.0
    y = moe(x, diffusion_timestep=torch.tensor([0, 2]))
    assert y.shape == x.shape
    assert moe.last_stats is not None
    assert moe.last_stats.expert_load_entropy.item() > 0
    assert moe.timestep_expert_count.sum() > 0
    assert int(moe.router_load_ema_steps.item()) == 1
    assert torch.isclose(moe.router_load_ema.sum(), torch.tensor(1.0))
    report = summarize_moe_stats(moe)
    assert report.layers_with_stats == 1
    assert report.timestep_tracked_selections > 0
    assert len(report.timestep_load_entropy) == cfg.diffusion_steps
    assert report.noisy_step_load_entropy >= 0
    assert report.polish_step_load_entropy >= 0
    assert report.noisy_timestep_tracked_selections > 0
    assert report.polish_timestep_tracked_selections > 0
    assert (
        moe.last_stats.router_probs.max()
        <= torch.sigmoid(torch.tensor(cfg.router_logit_clip)) + 1e-6
    )
    assert (
        moe.last_stats.router_probs.min()
        >= torch.sigmoid(torch.tensor(-cfg.router_logit_clip)) - 1e-6
    )
    old_bias = moe.router_bias.clone()
    overloaded = torch.tensor([1.0, 0.0, 0.0, 0.0])
    moe.update_router_bias(overloaded, rate=0.1)
    assert moe.router_bias[0] < old_bias[0]
    assert moe.router_bias.abs().max() <= cfg.router_bias_clip


def test_moe_tracks_load_ema_across_batches() -> None:
    cfg = tiny_config()
    cfg.moe_top_k = 1
    cfg.router_bias_clip = 10.0
    cfg.router_load_ema_decay = 0.5
    moe = MoEFeedForward(cfg)
    with torch.no_grad():
        moe.router.weight.zero_()  # type: ignore[attr-defined]
        moe.router_bias.zero_()
        moe.router_bias[0] = 5.0
    x = torch.randn(2, 3, cfg.hidden_size)
    moe(x)
    assert int(moe.router_load_ema_steps.item()) == 1
    assert torch.allclose(
        moe.router_load_ema,
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )

    with torch.no_grad():
        moe.router_bias.zero_()
        moe.router_bias[1] = 5.0
    moe(x)
    assert int(moe.router_load_ema_steps.item()) == 2
    assert torch.allclose(
        moe.router_load_ema,
        torch.tensor([0.5, 0.5, 0.0, 0.0]),
    )
    old_bias = moe.router_bias.clone()
    moe.update_router_bias(rate=0.1)
    assert moe.router_bias[0] < old_bias[0]
    assert moe.router_bias[1] < old_bias[1]
    assert moe.router_bias[2] > old_bias[2]


def test_router_bias_update_cadence() -> None:
    class CountingRouter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.config = tiny_config()
            self.config.router_bias_update_interval = 2
            self.calls = 0

        def update_router_bias(self) -> None:
            self.calls += 1

    args = type("Args", (), {"grad_clip": 0.0})()
    module = CountingRouter()
    opt = torch.optim.SGD(module.parameters(), lr=0.1)

    (module.weight * 1.0).backward()
    script_common.finish_optimization_step(module, opt, args, torch, step=0)
    assert module.calls == 0

    (module.weight * 1.0).backward()
    script_common.finish_optimization_step(module, opt, args, torch, step=1)
    assert module.calls == 1


def test_training_step_applies_warmup_learning_rate() -> None:
    layer = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(layer.parameters(), lr=0.01)
    args = type(
        "Args",
        (),
        {"grad_clip": 0.0, "lr": 0.01, "warmup_steps": 4},
    )()

    layer(torch.ones(1, 2)).sum().backward()
    metrics = script_common.finish_optimization_step(layer, opt, args, torch, step=0)
    assert metrics["lr"] == pytest.approx(0.0025)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0025)

    layer(torch.ones(1, 2)).sum().backward()
    metrics = script_common.finish_optimization_step(layer, opt, args, torch, step=3)
    assert metrics["lr"] == pytest.approx(0.01)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01)


def test_sparse_moe_matches_dense_reference() -> None:
    sparse_cfg = tiny_config()
    sparse_cfg.moe_sparse_dispatch = True
    dense_cfg = AlexandrosConfig.from_dict(sparse_cfg.to_dict())
    dense_cfg.moe_sparse_dispatch = False
    torch.manual_seed(9)
    sparse = MoEFeedForward(sparse_cfg)
    dense = MoEFeedForward(dense_cfg)
    dense.load_state_dict(sparse.state_dict())
    x = torch.randn(2, 4, sparse_cfg.hidden_size)
    with torch.no_grad():
        sparse_out = sparse(x)
        dense_out = dense(x)
    assert torch.allclose(sparse_out, dense_out, atol=1e-6)


def test_diffusion_timestep_bias_changes_moe_selection() -> None:
    cfg = tiny_config()
    cfg.moe_top_k = 1
    moe = MoEFeedForward(cfg)
    with torch.no_grad():
        moe.router.weight.zero_()  # type: ignore[attr-defined]
        moe.router_bias.zero_()
        moe.timestep_router_bias.weight.zero_()
        moe.timestep_router_bias.weight[0, 0] = 5.0
        moe.timestep_router_bias.weight[1, 1] = 5.0
    x = torch.randn(1, 3, cfg.hidden_size)
    moe(x, diffusion_timestep=0)
    assert moe.last_stats is not None
    step_zero = moe.last_stats.selected_experts.clone()
    moe(x, diffusion_timestep=1)
    assert moe.last_stats is not None
    step_one = moe.last_stats.selected_experts.clone()
    assert torch.equal(step_zero, torch.zeros_like(step_zero))
    assert torch.equal(step_one, torch.ones_like(step_one))
    assert moe.timestep_expert_count[:2].gt(0).all()


def test_diffusion_token_state_routing_reaches_model_moe_layers() -> None:
    cfg = tiny_config()
    cfg.moe_top_k = 1
    cfg.moe_num_shared_experts = 0
    cfg.moe_token_state_routing = True
    model = AlexandrosForDiffusionLM(cfg).eval()
    for layer in model.model.layers:
        with torch.no_grad():
            layer.moe.router.weight.zero_()
            layer.moe.timestep_router_bias.weight.zero_()
            layer.moe.token_state_router_bias.weight.zero_()
            layer.moe.token_state_router_bias.weight[0, 0] = 5.0
            layer.moe.token_state_router_bias.weight[1, 1] = 5.0
    input_ids = torch.tensor(
        [[cfg.bos_token_id, cfg.mask_token_id, 8]],
        dtype=torch.long,
    )

    model(input_ids, diffusion_timestep=0)

    stats = model.model.layers[0].moe.last_stats
    assert stats is not None
    assert stats.selected_experts[0, :, 0].tolist() == [0, 1, 0]
    assert "router_bias" not in dict(model.model.layers[0].moe.named_parameters())
    assert model.model.layers[0].moe.router_bias.requires_grad is False


def test_position_routing_reaches_model_moe_layers() -> None:
    cfg = tiny_config()
    cfg.moe_top_k = 1
    cfg.moe_num_shared_experts = 0
    cfg.moe_position_routing = True
    cfg.moe_position_buckets = 4
    cfg.max_position_embeddings = 8
    model = AlexandrosForDiffusionLM(cfg).eval()
    for layer in model.model.layers:
        with torch.no_grad():
            layer.moe.router.weight.zero_()
            layer.moe.timestep_router_bias.weight.zero_()
            layer.moe.token_state_router_bias.weight.zero_()
            layer.moe.position_router_bias.weight.zero_()
            layer.moe.position_router_bias.weight[0, 0] = 5.0
            layer.moe.position_router_bias.weight[1, 1] = 5.0
    input_ids = torch.tensor(
        [[cfg.bos_token_id, 7, cfg.mask_token_id, 8]],
        dtype=torch.long,
    )

    model(input_ids, diffusion_timestep=0)

    stats = model.model.layers[0].moe.last_stats
    assert stats is not None
    assert stats.selected_experts[0, :, 0].tolist() == [0, 0, 1, 1]
    assert "router_bias" not in dict(model.model.layers[0].moe.named_parameters())
    assert model.model.layers[0].moe.router_bias.requires_grad is False


@pytest.mark.parametrize("top_k", [1, 4])
def test_moe_top_k_edges(top_k: int) -> None:
    cfg = tiny_config()
    cfg.moe_top_k = top_k
    moe = MoEFeedForward(cfg)
    x = torch.randn(1, 3, cfg.hidden_size)
    y = moe(x)
    assert y.shape == x.shape
    assert moe.last_stats is not None
    assert moe.last_stats.selected_experts.shape[-1] == top_k


def test_diffusion_loss_and_generation() -> None:
    cfg = tiny_config()
    model = AlexandrosForDiffusionLM(cfg)
    input_ids = torch.randint(4, cfg.vocab_size, (2, 8))
    out = model.diffusion_loss(input_ids, timestep=torch.tensor([1, 2]))
    assert out.loss is not None
    noisy, mask = model.scheduler.add_noise(input_ids, timestep=cfg.diffusion_steps - 1)
    assert mask.all()
    assert noisy.eq(cfg.mask_token_id).all()
    with pytest.raises(ValueError, match="at least one non-pad"):
        model.diffusion_loss(torch.full((1, 4), cfg.pad_token_id, dtype=torch.long))
    generated = model.generate_block_diffusion(
        input_ids[:, :3], max_new_tokens=3, steps=2
    )
    assert generated.shape[0] == 2
    assert 3 <= generated.shape[1] <= 6
    assert torch.equal(generated[:, :3], input_ids[:, :3])
    assert not generated[:, 3:].eq(cfg.mask_token_id).any()
    chunked = model.generate_block_diffusion(
        input_ids[:, :3],
        max_new_tokens=5,
        steps=2,
        block_size=2,
        confidence_schedule="linear",
        remask_low_confidence=True,
    )
    assert chunked.shape[0] == 2
    assert 3 <= chunked.shape[1] <= 8
    assert torch.equal(chunked[:, :3], input_ids[:, :3])
    assert not chunked[:, 3:].eq(cfg.mask_token_id).any()
    prompt_with_mask_id = torch.tensor(
        [[cfg.bos_token_id, cfg.mask_token_id, 8]],
        dtype=torch.long,
    )
    preserved_prompt = model.generate_block_diffusion(
        prompt_with_mask_id,
        max_new_tokens=2,
        steps=2,
    )
    assert torch.equal(
        preserved_prompt[:, : prompt_with_mask_id.size(1)], prompt_with_mask_id
    )
    assert (
        not preserved_prompt[:, prompt_with_mask_id.size(1) :]
        .eq(cfg.mask_token_id)
        .any()
    )
    sampled = model.generate(
        input_ids[:, :3],
        max_new_tokens=3,
        mode="block_diffusion",
        steps=2,
        do_sample=True,
        temperature=0.8,
        top_k=5,
        top_p=0.9,
    )
    assert sampled.shape == (2, 6)
    assert not sampled[:, 3:].eq(cfg.mask_token_id).any()
    with pytest.raises(ValueError, match="temperature"):
        model.generate_block_diffusion(
            input_ids[:, :3],
            max_new_tokens=1,
            steps=1,
            do_sample=True,
            temperature=0.0,
        )
    with pytest.raises(ValueError, match="block_size"):
        model.generate_block_diffusion(
            input_ids[:, :3], max_new_tokens=1, block_size=True
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="confidence_schedule"):
        model.generate_block_diffusion(
            input_ids[:, :3],
            max_new_tokens=1,
            confidence_schedule="unknown",
        )
    with pytest.raises(ValueError, match="remask_low_confidence"):
        model.generate_block_diffusion(
            input_ids[:, :3],
            max_new_tokens=1,
            remask_low_confidence=1,  # type: ignore[arg-type]
        )


def test_diffusion_loss_weighting_changes_loss_explicitly() -> None:
    torch.manual_seed(123)
    cfg = tiny_config()
    uniform_cfg = AlexandrosConfig.from_dict(cfg.to_dict())
    weighted_cfg = AlexandrosConfig.from_dict(cfg.to_dict())
    weighted_cfg.diffusion_loss_weighting = "inverse_mask_prob"
    uniform = AlexandrosForDiffusionLM(uniform_cfg).eval()
    weighted = AlexandrosForDiffusionLM(weighted_cfg).eval()
    weighted.load_state_dict(uniform.state_dict())
    input_ids = torch.tensor(
        [
            [cfg.bos_token_id, 8, 9, 10],
            [cfg.bos_token_id, 11, 12, 13],
        ],
        dtype=torch.long,
    )
    timestep = torch.tensor([0, cfg.diffusion_steps - 1], dtype=torch.long)
    generator = torch.Generator().manual_seed(321)
    uniform_out = uniform.diffusion_loss(
        input_ids, timestep=timestep, generator=generator
    )
    generator = torch.Generator().manual_seed(321)
    weighted_out = weighted.diffusion_loss(
        input_ids, timestep=timestep, generator=generator
    )

    assert uniform_out.loss is not None
    assert weighted_out.loss is not None
    assert torch.isfinite(weighted_out.loss)
    assert not torch.allclose(uniform_out.loss, weighted_out.loss)


def test_block_diffusion_remasks_low_confidence_positions(monkeypatch) -> None:
    cfg = tiny_config()
    model = AlexandrosForDiffusionLM(cfg)
    prompt = torch.tensor([[cfg.bos_token_id, 4]], dtype=torch.long)
    seen_inputs: list[torch.Tensor] = []

    def fake_forward(input_ids: torch.Tensor, *args, **kwargs):
        seen_inputs.append(input_ids.detach().clone())
        logits = torch.full(
            (input_ids.size(0), input_ids.size(1), cfg.vocab_size),
            -10.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        block_start = prompt.size(1)
        for pos in range(block_start, input_ids.size(1)):
            relative = pos - block_start
            if relative % 2 == 0:
                logits[:, pos, 5] = 10.0
                logits[:, pos, 6] = -10.0
            else:
                logits[:, pos, 6] = 1.0
                logits[:, pos, 7] = 0.9
        return modeling_alexandros.AlexandrosCausalLMOutput(loss=None, logits=logits)

    monkeypatch.setattr(model, "forward", fake_forward)
    generated = model.generate_block_diffusion(
        prompt,
        max_new_tokens=4,
        steps=3,
        remask_low_confidence=True,
    )

    assert generated.shape == (1, 6)
    assert not generated[:, prompt.size(1) :].eq(cfg.mask_token_id).any()
    assert len(seen_inputs) >= 2
    second_step_block = seen_inputs[1][:, prompt.size(1) :]
    assert second_step_block[0, 0].item() == 5
    assert second_step_block[0, 1].item() == cfg.mask_token_id


def test_block_diffusion_stop_controls_postprocess(monkeypatch) -> None:
    cfg = tiny_config()
    model = AlexandrosForDiffusionLM(cfg)
    prompt = torch.tensor([[cfg.bos_token_id, 4]], dtype=torch.long)
    stop_id = 8

    def fake_forward(input_ids: torch.Tensor, *args, **kwargs):
        logits = torch.full(
            (input_ids.size(0), input_ids.size(1), cfg.vocab_size),
            -100.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[..., stop_id] = 100.0
        return modeling_alexandros.AlexandrosCausalLMOutput(loss=None, logits=logits)

    monkeypatch.setattr(model, "forward", fake_forward)
    generated = model.generate_block_diffusion(
        prompt,
        max_new_tokens=3,
        steps=1,
        eos_token_id=None,
        stop_token_ids=[stop_id],
    )
    assert generated.tolist() == [[cfg.bos_token_id, 4, stop_id]]


def test_diffusion_loss_local_generator_does_not_advance_global_rng() -> None:
    cfg = tiny_config()
    model = AlexandrosForDiffusionLM(cfg).eval()
    input_ids = torch.randint(4, cfg.vocab_size, (1, 6))
    torch.manual_seed(2024)
    before = torch.get_rng_state()
    generator = torch.Generator().manual_seed(99)
    out = model.diffusion_loss(input_ids, generator=generator)
    after = torch.get_rng_state()
    assert out.loss is not None
    assert torch.equal(before, after)


def test_evaluation_metrics_are_finite() -> None:
    cfg = tiny_config()
    model = AlexandrosForDiffusionLM(cfg)
    input_ids = torch.randint(4, cfg.vocab_size, (1, 6))
    perplexity = causal_lm_perplexity(model, input_ids)
    accuracy = masked_diffusion_reconstruction_accuracy(model, input_ids, timestep=1)
    latent = latent_reconstruction_metrics(model, input_ids, latent_steps=2)
    tq = turboquant_reconstruction_metrics(torch.randn(2, 4, cfg.kv_lora_rank), bits=4)
    lost = synthetic_lost_in_middle_probe(model, sequence_length=8)
    copy_report = synthetic_copy_retrieval_probe(model, sequence_length=8)
    drift = recurrent_state_drift_probe(model, sequence_length=8)
    reasoning = synthetic_modular_addition_probe(model, lhs=2, rhs=3, modulus=10)
    assert perplexity > 0
    assert 0.0 <= accuracy <= 1.0
    assert latent.reconstruction_mse >= 0
    assert latent.vae_reconstruction_mse >= 0
    assert latent.refinement_reconstruction_mse >= 0
    assert latent.kl_loss >= 0
    assert latent.latent_norm > 0
    assert latent.refined_latent_norm > 0
    assert latent.latent_steps == 2
    assert tq.mse >= 0
    assert tq.max_abs_error >= 0
    assert tq.compression_ratio > 1.0
    assert lost.needle_positions == (1, 4, 6)
    assert len(lost.target_ranks) == 3
    assert lost.worst_rank >= lost.middle_rank >= 1
    assert all(0.0 <= probability <= 1.0 for probability in lost.target_probabilities)
    assert copy_report.source_position == 2
    assert copy_report.query_position == 7
    assert 1 <= copy_report.target_rank <= cfg.vocab_size
    assert 0.0 <= copy_report.target_probability <= 1.0
    assert drift.sequence_length == 8
    assert drift.layers_with_state >= 0
    assert drift.max_state_norm >= 0
    assert drift.mean_state_norm >= 0
    assert drift.max_update_norm >= 0
    assert drift.mean_update_norm >= 0
    assert drift.finite is True
    assert reasoning.prompt_token_ids == (cfg.bos_token_id, 10, 14, 15, 11)
    assert reasoning.target_token_id == 17
    assert 1 <= reasoning.target_rank <= cfg.vocab_size
    assert 0.0 <= reasoning.target_probability <= 1.0
    with pytest.raises(ValueError, match="latent_steps"):
        latent_reconstruction_metrics(model, input_ids, latent_steps=0)
    with pytest.raises(ValueError, match="sequence_length"):
        synthetic_lost_in_middle_probe(model, sequence_length=4)
    with pytest.raises(ValueError, match="sequence_length"):
        synthetic_copy_retrieval_probe(model, sequence_length=3)
    with pytest.raises(ValueError, match="sequence_length"):
        recurrent_state_drift_probe(model, sequence_length=1)
    with pytest.raises(ValueError, match="max_position_embeddings"):
        synthetic_lost_in_middle_probe(
            model, sequence_length=cfg.max_position_embeddings + 1
        )
    with pytest.raises(ValueError, match="max_position_embeddings"):
        synthetic_copy_retrieval_probe(
            model, sequence_length=cfg.max_position_embeddings + 1
        )
    with pytest.raises(ValueError, match="max_position_embeddings"):
        recurrent_state_drift_probe(
            model, sequence_length=cfg.max_position_embeddings + 1
        )
    with pytest.raises(ValueError, match="modulus"):
        synthetic_modular_addition_probe(model, modulus=1)
    with pytest.raises(ValueError, match="base_token_id"):
        synthetic_modular_addition_probe(model, base_token_id=cfg.vocab_size)


def test_runtime_profile_reports_latency_and_memory() -> None:
    cfg = tiny_config()
    model = AlexandrosForDiffusionLM(cfg)
    input_ids = torch.randint(4, cfg.vocab_size, (1, 4))
    profile = profile_model_runtime(
        model,
        input_ids,
        max_new_tokens=1,
        mode="autoregressive",
        warmup=0,
        repeats=1,
    )
    assert profile.prefill_ms >= 0
    assert profile.generation_ms >= 0
    assert profile.generated_tokens == 1
    assert profile.parameter_bytes > 0
    assert profile.trainable_parameter_bytes > 0
    assert profile.peak_cuda_bytes >= 0
    with pytest.raises(ValueError, match="repeats"):
        profile_model_runtime(model, input_ids, repeats=0)
    causal_profile = profile_model_runtime(
        AlexandrosForCausalLM(cfg),
        input_ids,
        max_new_tokens=1,
    )
    assert causal_profile.generated_tokens == 1


def test_synthetic_needle_retrieval_probe_reports_rank() -> None:
    cfg = tiny_config()
    model = AlexandrosForDiffusionLM(cfg)
    report = synthetic_needle_retrieval_probe(
        model,
        sequence_length=16,
        needle_position=7,
        needle_token_id=8,
    )
    assert report.sequence_length == 16
    assert report.needle_position == 7
    assert report.needle_token_id == 8
    assert 1 <= report.target_rank <= cfg.vocab_size
    assert 0.0 <= report.target_probability <= 1.0
    assert 0 <= report.top_token_id < cfg.vocab_size
    with pytest.raises(ValueError, match="needle_position"):
        synthetic_needle_retrieval_probe(model, sequence_length=8, needle_position=7)


def test_generation_mode_kwargs_are_checked() -> None:
    cfg = tiny_config()
    model = AlexandrosForDiffusionLM(cfg)
    input_ids = torch.randint(4, cfg.vocab_size, (1, 4))
    hybrid = model.generate(
        input_ids,
        max_new_tokens=3,
        mode="hybrid",
        latent_steps=1,
        steps=1,
    )
    assert hybrid.shape[0] == 1
    assert input_ids.size(1) <= hybrid.shape[1] <= 7
    assert torch.equal(hybrid[:, : input_ids.size(1)], input_ids)
    with pytest.raises(ValueError, match="use_cache"):
        model.generate(
            input_ids, max_new_tokens=1, mode="block_diffusion", use_cache=True
        )
    uncached_diffusion = model.generate(
        input_ids,
        max_new_tokens=1,
        mode="block_diffusion",
        use_cache=False,
    )
    assert uncached_diffusion.shape == (1, 5)
    with pytest.raises(TypeError, match="unsupported hybrid"):
        model.generate(input_ids, max_new_tokens=1, mode="hybrid", temperature=0.8)
    with pytest.raises(ValueError, match="steps"):
        model.generate(input_ids, max_new_tokens=1, mode="block_diffusion", steps=0)
    with pytest.raises(ValueError, match="latent_steps"):
        model.generate(
            input_ids, max_new_tokens=1, mode="latent_reasoning", latent_steps=True
        )  # type: ignore[arg-type]


def test_diffusion_model_generation_handles_padded_batches_by_mode() -> None:
    torch.manual_seed(2026)
    cfg = tiny_config()
    model = AlexandrosForDiffusionLM(cfg).eval()
    padded = torch.tensor(
        [
            [cfg.bos_token_id, 8, cfg.pad_token_id],
            [cfg.bos_token_id, 9, 10],
        ],
        dtype=torch.long,
    )
    mode_kwargs = [
        ("autoregressive", {"use_cache": True}),
        ("block_diffusion", {"steps": 1, "block_size": 1}),
        ("latent_reasoning", {"latent_steps": 1}),
        ("hybrid", {"latent_steps": 1, "steps": 1, "block_size": 1}),
    ]
    for mode, kwargs in mode_kwargs:
        rows = [
            model.generate(
                padded[0:1, :2], max_new_tokens=2, mode=mode, **kwargs
            ).squeeze(0),
            model.generate(padded[1:2], max_new_tokens=2, mode=mode, **kwargs).squeeze(
                0
            ),
        ]
        expected = padded.new_full(
            (2, max(row.numel() for row in rows)), cfg.pad_token_id
        )
        for row_idx, row in enumerate(rows):
            expected[row_idx, : row.numel()] = row
        batched = model.generate(padded, max_new_tokens=2, mode=mode, **kwargs)
        assert torch.equal(batched, expected)


def test_latent_reasoning_shapes_and_sensitivity() -> None:
    cfg = tiny_config()
    cfg.latent_update_clip = 0.25
    model = AlexandrosForDiffusionLM(cfg)
    input_ids = torch.randint(4, cfg.vocab_size, (2, 8))
    hidden = model.model(input_ids).last_hidden_state
    vae = model.latent_vae(hidden)
    refined_one = model.latent_reasoner(vae.latents, steps=1)
    refined_two = model.latent_reasoner(vae.latents, steps=2)
    refined_long = model.latent_reasoner(vae.latents, steps=cfg.diffusion_steps + 5)
    with pytest.raises(ValueError, match="steps"):
        model.latent_reasoner(vae.latents, steps=0)
    with pytest.raises(ValueError, match="steps"):
        model.latent_reasoner(vae.latents, steps=1.5)  # type: ignore[arg-type]
    assert vae.reconstruction.shape == (2, cfg.latent_slots, cfg.hidden_size)
    assert torch.isfinite(vae.kl_loss)
    assert not torch.allclose(refined_one, refined_two)
    assert torch.isfinite(refined_long).all()
    delta_norm = (refined_two - vae.latents).norm(dim=-1).max()
    assert delta_norm <= cfg.latent_update_clip + 1e-5


def test_ttt_state_does_not_mutate_model() -> None:
    cfg = tiny_config()
    model = AlexandrosForCausalLM(cfg)
    before = copy.deepcopy(model.state_dict())
    input_ids = torch.randint(4, cfg.vocab_size, (2, 8))
    hidden = model.model(input_ids).last_hidden_state
    state = TTTState.from_config(cfg, hidden.device, hidden.dtype)
    state.update(hidden)
    adapted = model.model(input_ids, ttt_state=state).last_hidden_state
    assert torch.isfinite(adapted).all()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_ttt_state_rejected_for_diffusion_timestep() -> None:
    cfg = tiny_config()
    model = AlexandrosForCausalLM(cfg)
    input_ids = torch.randint(4, cfg.vocab_size, (1, 4))
    hidden = model.model(input_ids).last_hidden_state
    state = TTTState.from_config(cfg, hidden.device, hidden.dtype)
    with pytest.raises(ValueError, match="TTT state"):
        model(input_ids, diffusion_timestep=0, ttt_state=state)


def test_turboquant_round_trip() -> None:
    x = torch.randn(2, 4, 16)
    cache = TurboQuantKVCache(bits=4, use_qjl=True)
    packet = cache.compress(x)
    restored = cache.decompress(packet)
    assert restored.shape == x.shape
    assert torch.isfinite(restored).all()
    assert packet.rotation_seed == cache.seed
    assert packet.packet_format_version == 1
    assert packet.qjl_projection_seed == cache.seed
    assert packet.qjl_residual_norm is not None
    assert packet.qjl_residual_norm.shape == packet.scale.shape
    assert not hasattr(packet, "rotation")
    assert packet.estimated_bits < x.numel() * 32
    with pytest.raises(ValueError, match="bits"):
        TurboQuantKVCache(bits=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="floating-point"):
        cache.compress(torch.ones(1, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="bit range"):
        cache.decompress(
            TurboQuantPacket(
                q=torch.full_like(packet.q, 8),
                scale=packet.scale,
                bits=4,
                original_dtype=packet.original_dtype,
                rotation_seed=packet.rotation_seed,
            )
        )
    with pytest.raises(ValueError, match="qjl_sign"):
        cache.decompress(
            TurboQuantPacket(
                q=packet.q,
                scale=packet.scale,
                bits=packet.bits,
                original_dtype=packet.original_dtype,
                rotation_seed=packet.rotation_seed,
                qjl_sign=torch.full_like(packet.q, 2),
                qjl_projection_seed=packet.qjl_projection_seed,
                qjl_residual_norm=packet.qjl_residual_norm,
            )
        )


def test_save_load_round_trip(tmp_path) -> None:
    cfg = tiny_config()
    model = AlexandrosForDiffusionLM(cfg).eval()
    trainability = apply_trainability(model, phase="latent")
    checkpoint_metadata = phase_checkpoint_metadata("latent", trainability)
    tokenizer_metadata = {
        "tokenizer_class": "ToyTokenIdTokenizer",
        "source": "unit-test-fixture",
        "vocab_size": cfg.vocab_size,
        "special_tokens": {
            "pad_token_id": cfg.pad_token_id,
            "bos_token_id": cfg.bos_token_id,
            "eos_token_id": cfg.eos_token_id,
            "mask_token_id": cfg.mask_token_id,
        },
    }
    model.save_pretrained(
        tmp_path,
        tokenizer_metadata=tokenizer_metadata,
        checkpoint_metadata=checkpoint_metadata,
    )
    extra = json.loads((tmp_path / "alexandros_extra.json").read_text(encoding="utf-8"))
    assert extra["checkpoint_format_version"] == 1
    assert extra["model_class"] == "AlexandrosForDiffusionLM"
    assert extra["tokenizer_metadata"] == "tokenizer_metadata.json"
    assert extra["checkpoint_metadata"] == "checkpoint_metadata.json"
    hf_compatibility = extra["hf_compatibility"]
    assert hf_compatibility["transformers_required"] is False
    assert hf_compatibility["config_class"] == "AlexandrosConfig"
    assert hf_compatibility["model_class"] == "AlexandrosForDiffusionLM"
    assert hf_compatibility["auto_config_registered"] is False
    assert hf_compatibility["auto_model_registered"] is False
    assert hf_compatibility["requires_trust_remote_code"] is False
    checkpoint_payload = json.loads(
        (tmp_path / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    assert checkpoint_payload["format"] == "open-alexandros-checkpoint-metadata"
    assert checkpoint_payload["checkpoint"]["phase"] == "latent"
    assert (
        checkpoint_payload["checkpoint"]["objective"]["name"]
        == "latent_reconstruction_kl"
    )
    assert (
        AlexandrosForDiffusionLM.load_checkpoint_metadata(tmp_path)
        == checkpoint_metadata
    )
    tokenizer_payload = json.loads(
        (tmp_path / "tokenizer_metadata.json").read_text(encoding="utf-8")
    )
    assert tokenizer_payload["format"] == "open-alexandros-tokenizer-metadata"
    assert tokenizer_payload["config"]["vocab_size"] == cfg.vocab_size
    assert (
        AlexandrosForDiffusionLM.load_tokenizer_metadata(tmp_path) == tokenizer_metadata
    )
    generation_config = json.loads(
        (tmp_path / "generation_config.json").read_text(encoding="utf-8")
    )
    assert generation_config["bos_token_id"] == cfg.bos_token_id
    assert generation_config["eos_token_id"] == cfg.eos_token_id
    assert generation_config["max_length"] == cfg.max_position_embeddings
    assert generation_config["block_diffusion_steps"] == cfg.diffusion_steps
    assert generation_config["block_diffusion_confidence_schedule"] == "median"
    assert generation_config["block_diffusion_remask_low_confidence"] is False
    restored = AlexandrosForDiffusionLM.from_pretrained(tmp_path).eval()
    input_ids = torch.randint(4, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        a = model(input_ids).logits
        b = restored(input_ids).logits
    assert torch.allclose(a, b, atol=1e-6)


def test_save_pretrained_rejects_incompatible_tokenizer_metadata(tmp_path) -> None:
    cfg = tiny_config()
    model = AlexandrosForCausalLM(cfg)
    with pytest.raises(ValueError, match="vocab_size"):
        model.save_pretrained(
            tmp_path / "bad_vocab",
            tokenizer_metadata={"vocab_size": cfg.vocab_size + 1},
        )
    assert not (tmp_path / "bad_vocab").exists()
    with pytest.raises(ValueError, match="bos_token_id"):
        model.save_pretrained(
            tmp_path / "bad_bos",
            tokenizer_metadata={
                "vocab_size": cfg.vocab_size,
                "special_tokens": {"bos_token_id": cfg.bos_token_id + 1},
            },
        )
    with pytest.raises(ValueError, match="JSON serializable"):
        model.save_pretrained(
            tmp_path / "bad_json",
            tokenizer_metadata={"tokenizer": object()},  # type: ignore[dict-item]
        )
    assert not (tmp_path / "bad_json").exists()
    with pytest.raises(ValueError, match="JSON serializable"):
        model.save_pretrained(
            tmp_path / "bad_nan",
            tokenizer_metadata={"score": float("nan")},
        )
    with pytest.raises(ValueError, match="checkpoint_metadata"):
        model.save_pretrained(
            tmp_path / "bad_checkpoint_metadata",
            checkpoint_metadata={"score": float("nan")},
        )
    assert not (tmp_path / "bad_checkpoint_metadata").exists()
    assert AlexandrosForCausalLM.load_tokenizer_metadata(tmp_path / "missing") is None
    assert AlexandrosForCausalLM.load_checkpoint_metadata(tmp_path / "missing") is None


def test_phase_handoff_checkpoint_loading_paths(tmp_path) -> None:
    cfg = tiny_config()
    input_ids = torch.randint(4, cfg.vocab_size, (1, 5))

    ar = AlexandrosForCausalLM(cfg).eval()
    ar_dir = tmp_path / "ar"
    ar_trainability = apply_trainability(ar, phase="ar")
    ar.save_pretrained(
        ar_dir,
        checkpoint_metadata=phase_checkpoint_metadata("ar", ar_trainability),
    )
    loaded_diffusion = AlexandrosForDiffusionLM.from_pretrained(ar_dir).eval()
    with torch.no_grad():
        ar_logits = ar(input_ids).logits
        handoff_logits = loaded_diffusion(input_ids).logits
    assert torch.allclose(ar_logits, handoff_logits, atol=1e-6)
    diffusion_loss = loaded_diffusion.diffusion_loss(
        input_ids,
        timestep=torch.tensor([cfg.diffusion_steps - 1]),
    )
    assert diffusion_loss.loss is not None
    assert torch.isfinite(diffusion_loss.loss)

    diffusion = AlexandrosForDiffusionLM(cfg).eval()
    diffusion_dir = tmp_path / "diffusion"
    diffusion_trainability = apply_trainability(diffusion, phase="diffusion")
    diffusion.save_pretrained(
        diffusion_dir,
        checkpoint_metadata=phase_checkpoint_metadata(
            "diffusion", diffusion_trainability
        ),
    )
    loaded_causal = AlexandrosForCausalLM.from_pretrained(diffusion_dir).eval()
    with torch.no_grad():
        expected = diffusion(input_ids).logits
        actual = loaded_causal(input_ids).logits
    assert torch.allclose(expected, actual, atol=1e-6)

    latent = AlexandrosForDiffusionLM.from_pretrained(diffusion_dir).eval()
    latent_dir = tmp_path / "latent"
    latent_trainability = apply_trainability(latent, phase="latent")
    latent.save_pretrained(
        latent_dir,
        checkpoint_metadata=phase_checkpoint_metadata("latent", latent_trainability),
    )
    ttt_model = AlexandrosForCausalLM.from_pretrained(latent_dir).eval()
    with torch.no_grad():
        hidden = ttt_model.model(input_ids).last_hidden_state
    state = TTTState.from_config(cfg, hidden.device, hidden.dtype)
    state.update(hidden, lr=0.1)
    adapted = state.apply(hidden)
    assert torch.isfinite(adapted).all()
    assert (adapted - hidden).abs().max().item() > 0.0


def test_safe_serialization_optional_round_trip(tmp_path, monkeypatch) -> None:
    class FakeSafeTensorsTorch:
        @staticmethod
        def save_file(state, path) -> None:
            torch.save(state, path)

        @staticmethod
        def load_file(path, device: str = "cpu"):
            try:
                return torch.load(path, map_location=device, weights_only=True)
            except TypeError:
                return torch.load(path, map_location=device)

    monkeypatch.setattr(
        modeling_alexandros,
        "_load_safetensors_torch",
        lambda: FakeSafeTensorsTorch,
    )
    cfg = tiny_config()
    model = AlexandrosForDiffusionLM(cfg).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)
    assert not (tmp_path / "pytorch_model.bin").exists()
    assert (tmp_path / "model.safetensors").is_file()
    extra = json.loads((tmp_path / "alexandros_extra.json").read_text(encoding="utf-8"))
    assert extra["weights"] == "model.safetensors"
    assert extra["safe_serialization"] is True
    restored = AlexandrosForDiffusionLM.from_pretrained(tmp_path).eval()
    input_ids = torch.randint(4, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        a = model(input_ids).logits
        b = restored(input_ids).logits
    assert torch.allclose(a, b, atol=1e-6)


def test_safe_serialization_requires_optional_dependency(tmp_path, monkeypatch) -> None:
    def missing_safetensors() -> None:
        raise ImportError(
            "safe_serialization=True requires the optional 'safetensors' package"
        )

    monkeypatch.setattr(
        modeling_alexandros, "_load_safetensors_torch", missing_safetensors
    )
    model = AlexandrosForDiffusionLM(tiny_config())
    with pytest.raises(ImportError, match="safetensors"):
        model.save_pretrained(tmp_path, safe_serialization=True)
