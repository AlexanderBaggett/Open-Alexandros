from __future__ import annotations

import json
import time
from pathlib import Path

from _deps import require_runtime_dependencies

require_runtime_dependencies()

import torch

from _common import (
    add_runtime_args,
    load_config,
    make_arg_parser,
    prepare_model,
    prepare_output_dir,
    run_metadata,
    seeded_generator,
    setup_torch_runtime,
)
from alexandros import AlexandrosForDiffusionLM, GenerationMode
from alexandros.evaluation import (
    estimate_cache_memory,
    estimate_flops,
    profile_model_runtime,
    summarize_parameters,
)
from alexandros.kv_cache import TurboQuantKVCache


def _dtype_bits(dtype) -> int:
    if dtype == torch.float32:
        return 32
    if dtype in {torch.float16, torch.bfloat16}:
        return 16
    return 32


def _hidden_activation_bytes(
    config, *, batch_size: int, sequence_length: int, dtype
) -> int:
    return batch_size * sequence_length * config.hidden_size * (_dtype_bits(dtype) // 8)


def _loss_stability_probe(model, input_ids, *, generator) -> tuple[float, float]:
    with torch.no_grad():
        causal = model(input_ids, labels=input_ids)
        diffusion = model.diffusion_loss(input_ids, generator=generator)
    if causal.loss is None or diffusion.loss is None:
        raise ValueError("loss stability probe expected causal and diffusion losses")
    if not torch.isfinite(causal.loss.detach()) or not torch.isfinite(
        diffusion.loss.detach()
    ):
        raise FloatingPointError(
            "non-finite loss encountered during benchmark stability probe"
        )
    return (
        float(causal.loss.detach().cpu().item()),
        float(diffusion.loss.detach().cpu().item()),
    )


def _sync_if_needed(device, torch) -> None:
    if getattr(device, "type", str(device).split(":", 1)[0]) == "cuda":
        torch.cuda.synchronize(device)


def _cache_roundtrip_profile(
    config, args, device, dtype, torch
) -> tuple[float, float, float]:
    generator = seeded_generator(torch, int(args.seed) + 8_000_001, device)
    sample = torch.randn(
        args.batch_size,
        min(args.seq_len, config.max_position_embeddings),
        config.kv_lora_rank,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    cache = TurboQuantKVCache(bits=config.turboquant_bits, use_qjl=config.use_qjl)
    for _ in range(args.warmup):
        packet = cache.compress(sample)
        cache.decompress(packet)
    _sync_if_needed(device, torch)
    total = 0.0
    restored = sample
    for _ in range(args.repeats):
        _sync_if_needed(device, torch)
        start = time.perf_counter()
        packet = cache.compress(sample)
        restored = cache.decompress(packet)
        _sync_if_needed(device, torch)
        total += time.perf_counter() - start
    error = (restored.float() - sample.float()).abs()
    return (
        float(error.pow(2).mean().item()),
        float(error.max().item()),
        (total / args.repeats) * 1000.0,
    )


def _structured_sign_permutation(sample, *, seed: int, torch):
    dim = sample.size(-1)
    generator = seeded_generator(torch, seed, sample.device)
    permutation = torch.randperm(dim, device=sample.device, generator=generator)
    signs = torch.randint(
        0,
        2,
        (dim,),
        device=sample.device,
        generator=generator,
        dtype=torch.int8,
    )
    signs = signs.to(dtype=sample.dtype).mul(2).sub(1)
    return sample.index_select(-1, permutation) * signs


def _rotation_candidate_profile(
    config, args, device, dtype, torch
) -> tuple[int, float, float]:
    generator = seeded_generator(torch, int(args.seed) + 8_500_001, device)
    sample = torch.randn(
        args.batch_size,
        min(args.seq_len, config.max_position_embeddings),
        config.kv_lora_rank,
        device=device,
        dtype=dtype,
        generator=generator,
    ).float()
    dim = sample.size(-1)

    for idx in range(args.warmup):
        cache = TurboQuantKVCache(
            bits=config.turboquant_bits, seed=int(args.seed) + idx
        )
        rotation = cache._rotation(dim, sample.device, sample.dtype)
        _ = sample @ rotation
        _ = _structured_sign_permutation(
            sample, seed=int(args.seed) + 1000 + idx, torch=torch
        )

    dense_total = 0.0
    structured_total = 0.0
    for idx in range(args.repeats):
        _sync_if_needed(device, torch)
        start = time.perf_counter()
        cache = TurboQuantKVCache(
            bits=config.turboquant_bits, seed=int(args.seed) + 10_000 + idx
        )
        rotation = cache._rotation(dim, sample.device, sample.dtype)
        _ = sample @ rotation
        _sync_if_needed(device, torch)
        dense_total += time.perf_counter() - start

        _sync_if_needed(device, torch)
        start = time.perf_counter()
        _ = _structured_sign_permutation(
            sample, seed=int(args.seed) + 20_000 + idx, torch=torch
        )
        _sync_if_needed(device, torch)
        structured_total += time.perf_counter() - start

    return (
        dim,
        (dense_total / args.repeats) * 1000.0,
        (structured_total / args.repeats) * 1000.0,
    )


def benchmark_config(config_path: str, args, device, dtype, torch) -> dict:
    config = load_config(config_path)
    model = prepare_model(AlexandrosForDiffusionLM(config), device, dtype, torch).eval()
    seq_len = min(args.seq_len, config.max_position_embeddings)
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    input_ids = torch.full(
        (args.batch_size, seq_len),
        4,
        dtype=torch.long,
        device=device,
    )
    input_ids[:, 0] = config.bos_token_id

    params = summarize_parameters(model)
    cache = estimate_cache_memory(
        config,
        batch_size=args.batch_size,
        sequence_length=seq_len,
        dtype_bits=16 if dtype != torch.float32 else 32,
    )
    flops = estimate_flops(config, batch_size=args.batch_size, sequence_length=seq_len)
    tq_mse, tq_max_error, tq_roundtrip_ms = _cache_roundtrip_profile(
        config,
        args,
        device,
        dtype,
        torch,
    )
    rotation_dim, dense_qr_ms, structured_rotation_ms = _rotation_candidate_profile(
        config,
        args,
        device,
        dtype,
        torch,
    )
    loss_generator = seeded_generator(torch, int(args.seed) + 7_000_001, device)
    causal_loss, diffusion_loss = _loss_stability_probe(
        model,
        input_ids,
        generator=loss_generator,
    )
    ar = profile_model_runtime(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        mode=GenerationMode.AUTOREGRESSIVE,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    block = profile_model_runtime(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        mode=GenerationMode.BLOCK_DIFFUSION,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    return {
        **run_metadata(args, config, torch),
        "config": config_path,
        "variant": config.variant,
        "batch_size": args.batch_size,
        "sequence_length": seq_len,
        "max_new_tokens": args.max_new_tokens,
        "total_parameters": params.total_parameters,
        "trainable_parameters": params.trainable_parameters,
        "active_parameters_per_token": params.active_parameters_per_token,
        "parameter_bytes": ar.parameter_bytes,
        "trainable_parameter_bytes": ar.trainable_parameter_bytes,
        "hidden_activation_bytes_estimate": _hidden_activation_bytes(
            config,
            batch_size=args.batch_size,
            sequence_length=seq_len,
            dtype=dtype,
        ),
        "causal_loss": round(causal_loss, 6),
        "diffusion_loss": round(diffusion_loss, 6),
        "losses_finite": True,
        "standard_kv_bits": cache.standard_kv_bits,
        "mla_kv_bits": cache.mla_kv_bits,
        "turboquant_mla_bits": cache.turboquant_mla_bits,
        "fp16_cache_bits_baseline": cache.standard_kv_bits,
        "mla_only_cache_bits": cache.mla_kv_bits,
        "mla_turboquant_cache_bits": cache.turboquant_mla_bits,
        "mla_cache_compression_ratio": round(cache.mla_compression_ratio, 4),
        "turboquant_cache_compression_ratio": round(
            cache.turboquant_compression_ratio, 4
        ),
        "turboquant_reconstruction_mse": round(tq_mse, 8),
        "turboquant_reconstruction_max_abs_error": round(tq_max_error, 6),
        "turboquant_cache_roundtrip_ms": round(tq_roundtrip_ms, 4),
        "turboquant_rotation_dim": rotation_dim,
        "turboquant_dense_qr_rotation_ms": round(dense_qr_ms, 4),
        "turboquant_structured_sign_permutation_ms": round(structured_rotation_ms, 4),
        "prefill_flops": flops.prefill_flops,
        "decode_token_flops": flops.decode_token_flops,
        "ar_prefill_ms": round(ar.prefill_ms, 4),
        "ar_generation_ms": round(ar.generation_ms, 4),
        "ar_tokens_per_second": round(ar.generation_tokens_per_second, 4),
        "block_prefill_ms": round(block.prefill_ms, 4),
        "block_generation_ms": round(block.generation_ms, 4),
        "block_tokens_per_second_equivalent": round(
            block.generation_tokens_per_second, 4
        ),
        "peak_cuda_bytes": max(ar.peak_cuda_bytes, block.peak_cuda_bytes),
    }


def write_jsonl(path: str, rows: list[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_markdown(path: str, rows: list[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Alexandros Benchmark",
        "",
        "| Config | Variant | Params | Active/token | Param MB | Act MB est | AR tok/s | Block tok/s eq | Causal loss | Diff loss | MLA cache | Turbo cache | TQ mse | TQ ms | QR ms | Struct ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['config']}` | "
            f"{row['variant']} | "
            f"{row['total_parameters']} | "
            f"{row['active_parameters_per_token']} | "
            f"{row['parameter_bytes'] / 1_000_000:.3f} | "
            f"{row['hidden_activation_bytes_estimate'] / 1_000_000:.3f} | "
            f"{row['ar_tokens_per_second']} | "
            f"{row['block_tokens_per_second_equivalent']} | "
            f"{row['causal_loss']} | "
            f"{row['diffusion_loss']} | "
            f"{row['mla_cache_compression_ratio']} | "
            f"{row['turboquant_cache_compression_ratio']} | "
            f"{row['turboquant_reconstruction_mse']} | "
            f"{row['turboquant_cache_roundtrip_ms']} | "
            f"{row['turboquant_dense_qr_rotation_ms']} | "
            f"{row['turboquant_structured_sign_permutation_ms']} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = make_arg_parser("Run local Alexandros runtime and memory benchmarks.")
    parser.add_argument("--config", default="configs/heavy_debug.yaml")
    parser.add_argument("--compare-config", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out-jsonl", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--out-dir", default="")
    add_runtime_args(parser)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if args.repeats <= 0:
        raise ValueError("repeats must be > 0")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")

    device, dtype = setup_torch_runtime(args, torch)
    out_dir = prepare_output_dir(args)
    if out_dir is not None:
        args.out_jsonl = args.out_jsonl or str(out_dir / "benchmark.jsonl")
        args.out_md = args.out_md or str(out_dir / "benchmark.md")
    rows = [
        benchmark_config(config_path, args, device, dtype, torch)
        for config_path in [args.config, *args.compare_config]
    ]
    if args.out_jsonl:
        write_jsonl(args.out_jsonl, rows)
    if args.out_md:
        write_markdown(args.out_md, rows)
    print(rows[0] if len(rows) == 1 else rows)


if __name__ == "__main__":
    main()
