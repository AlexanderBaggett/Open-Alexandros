from __future__ import annotations

import json
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
    setup_torch_runtime,
)
from alexandros import AlexandrosForDiffusionLM, GenerationMode
from alexandros.evaluation import (
    CodeBenchmarkSuite,
    HumanEvalStyleTask,
    adaptive_depth_toy_benchmark,
    causal_lm_perplexity,
    code_benchmark_plan,
    estimate_cache_memory,
    estimate_flops,
    humaneval_style_harness_report,
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


def generation_mode_smoke(model, input_ids, torch) -> dict[str, dict[str, object]]:
    mode_kwargs = {
        GenerationMode.AUTOREGRESSIVE: {},
        GenerationMode.BLOCK_DIFFUSION: {"steps": 2, "block_size": 1},
        GenerationMode.LATENT_REASONING: {"latent_steps": 1},
        GenerationMode.HYBRID: {"latent_steps": 1, "steps": 2, "block_size": 1},
    }
    report = {}
    with torch.no_grad():
        for mode, kwargs in mode_kwargs.items():
            output = model.generate(input_ids, max_new_tokens=2, mode=mode, **kwargs)
            continuation = output[:, input_ids.size(1) :]
            report[mode.value] = {
                "shape": list(output.shape),
                "new_tokens": int(max(output.size(1) - input_ids.size(1), 0)),
                "continuation_sum": int(continuation.sum().item())
                if continuation.numel()
                else 0,
            }
    return report


def evaluate_config(config_path: str, args, device, dtype, torch) -> dict:
    config = load_config(config_path)
    model = prepare_model(AlexandrosForDiffusionLM(config), device, dtype, torch).eval()
    input_ids = torch.tensor(
        [[config.bos_token_id, 4, 5, 6]], dtype=torch.long, device=device
    )
    generation_modes = generation_mode_smoke(model, input_ids, torch)
    perplexity = causal_lm_perplexity(model, input_ids)
    diffusion_accuracy = masked_diffusion_reconstruction_accuracy(model, input_ids)
    latent = latent_reconstruction_metrics(model, input_ids, latent_steps=1)
    params = summarize_parameters(model)
    moe = summarize_moe_stats(model)
    cache = estimate_cache_memory(config, sequence_length=input_ids.size(1))
    flops = estimate_flops(config, sequence_length=input_ids.size(1))
    runtime = profile_model_runtime(
        model,
        input_ids,
        max_new_tokens=2,
        mode=GenerationMode.AUTOREGRESSIVE,
        warmup=0,
        repeats=1,
    )
    needle = synthetic_needle_retrieval_probe(
        model,
        sequence_length=min(config.max_position_embeddings, 32),
    )
    lost_middle = synthetic_lost_in_middle_probe(
        model,
        sequence_length=min(config.max_position_embeddings, 32),
    )
    copy_retrieval = synthetic_copy_retrieval_probe(
        model,
        sequence_length=min(config.max_position_embeddings, 32),
    )
    drift = recurrent_state_drift_probe(
        model,
        sequence_length=min(config.max_position_embeddings, 32),
    )
    reasoning = synthetic_modular_addition_probe(model)
    adaptive_depth = (
        adaptive_depth_toy_benchmark(model) if config.enable_adaptive_depth else None
    )
    toy_code_task = HumanEvalStyleTask(
        task_id="ci/add_mod",
        prompt="def add_mod(a: int, b: int, m: int) -> int:\n    ",
        entry_point="add_mod",
        tests=("assert add_mod(2, 5, 7) == 0",),
    )
    humaneval_plan = humaneval_style_harness_report([toy_code_task])
    swe_plan = code_benchmark_plan(CodeBenchmarkSuite.SWE_BENCH)
    terminal_plan = code_benchmark_plan(CodeBenchmarkSuite.TERMINAL_BENCH)
    tq = turboquant_reconstruction_metrics(
        torch.randn(
            2, input_ids.size(1), config.kv_lora_rank, device=device, dtype=dtype
        ),
        bits=config.turboquant_bits,
        use_qjl=config.use_qjl,
    )
    return {
        "suite": args.suite,
        "config": config_path,
        **run_metadata(args, config, torch),
        "ar_shape": generation_modes[GenerationMode.AUTOREGRESSIVE.value]["shape"],
        "diffusion_shape": generation_modes[GenerationMode.BLOCK_DIFFUSION.value][
            "shape"
        ],
        "generation_modes": generation_modes,
        "perplexity": round(perplexity, 4),
        "diffusion_reconstruction_accuracy": round(diffusion_accuracy, 4),
        "latent_reconstruction_mse": round(latent.reconstruction_mse, 8),
        "latent_vae_reconstruction_mse": round(latent.vae_reconstruction_mse, 8),
        "latent_refinement_reconstruction_mse": round(
            latent.refinement_reconstruction_mse, 8
        ),
        "latent_kl_loss": round(latent.kl_loss, 8),
        "latent_norm": round(latent.latent_norm, 6),
        "latent_refined_norm": round(latent.refined_latent_norm, 6),
        "latent_update_norm": round(latent.latent_update_norm, 6),
        "latent_reconstruction_norm": round(latent.reconstruction_norm, 6),
        "latent_refined_reconstruction_norm": round(
            latent.refined_reconstruction_norm, 6
        ),
        "latent_eval_steps": latent.latent_steps,
        "total_parameters": params.total_parameters,
        "active_parameters_per_token": params.active_parameters_per_token,
        "moe_layers_with_stats": moe.layers_with_stats,
        "moe_mean_load_entropy": round(moe.mean_load_entropy, 4),
        "moe_timestep_tracked_selections": moe.timestep_tracked_selections,
        "moe_timestep_load_entropy": [
            round(value, 4) for value in moe.timestep_load_entropy
        ],
        "moe_noisy_step_load_entropy": round(moe.noisy_step_load_entropy, 4),
        "moe_polish_step_load_entropy": round(moe.polish_step_load_entropy, 4),
        "moe_noisy_timestep_tracked_selections": moe.noisy_timestep_tracked_selections,
        "moe_polish_timestep_tracked_selections": moe.polish_timestep_tracked_selections,
        "mla_cache_compression_ratio": round(cache.mla_compression_ratio, 4),
        "turboquant_cache_compression_ratio": round(
            cache.turboquant_compression_ratio, 4
        ),
        "prefill_flops": flops.prefill_flops,
        "decode_token_flops": flops.decode_token_flops,
        "prefill_ms": round(runtime.prefill_ms, 4),
        "generation_ms": round(runtime.generation_ms, 4),
        "generation_tokens_per_second": round(runtime.generation_tokens_per_second, 4),
        "parameter_bytes": runtime.parameter_bytes,
        "trainable_parameter_bytes": runtime.trainable_parameter_bytes,
        "peak_cuda_bytes": runtime.peak_cuda_bytes,
        "needle_sequence_length": needle.sequence_length,
        "needle_position": needle.needle_position,
        "needle_token_id": needle.needle_token_id,
        "needle_target_rank": needle.target_rank,
        "needle_target_probability": round(needle.target_probability, 8),
        "needle_top_token_id": needle.top_token_id,
        "lost_middle_sequence_length": lost_middle.sequence_length,
        "lost_middle_positions": list(lost_middle.needle_positions),
        "lost_middle_target_ranks": list(lost_middle.target_ranks),
        "lost_middle_target_probabilities": [
            round(probability, 8) for probability in lost_middle.target_probabilities
        ],
        "lost_middle_worst_rank": lost_middle.worst_rank,
        "lost_middle_middle_rank": lost_middle.middle_rank,
        "copy_sequence_length": copy_retrieval.sequence_length,
        "copy_source_position": copy_retrieval.source_position,
        "copy_query_position": copy_retrieval.query_position,
        "copy_token_id": copy_retrieval.copy_token_id,
        "copy_target_rank": copy_retrieval.target_rank,
        "copy_target_probability": round(copy_retrieval.target_probability, 8),
        "copy_top_token_id": copy_retrieval.top_token_id,
        "recurrent_state_sequence_length": drift.sequence_length,
        "recurrent_state_layers": drift.layers_with_state,
        "recurrent_state_max_norm": round(drift.max_state_norm, 8),
        "recurrent_state_mean_norm": round(drift.mean_state_norm, 8),
        "recurrent_state_max_update_norm": round(drift.max_update_norm, 8),
        "recurrent_state_mean_update_norm": round(drift.mean_update_norm, 8),
        "recurrent_state_finite": drift.finite,
        "toy_reasoning_prompt_token_ids": list(reasoning.prompt_token_ids),
        "toy_reasoning_lhs": reasoning.lhs,
        "toy_reasoning_rhs": reasoning.rhs,
        "toy_reasoning_modulus": reasoning.modulus,
        "toy_reasoning_target_token_id": reasoning.target_token_id,
        "toy_reasoning_target_rank": reasoning.target_rank,
        "toy_reasoning_target_probability": round(reasoning.target_probability, 8),
        "toy_reasoning_top_token_id": reasoning.top_token_id,
        "adaptive_depth_toy_target_rank": adaptive_depth.target_rank
        if adaptive_depth is not None
        else None,
        "adaptive_depth_toy_target_probability": round(
            adaptive_depth.target_probability, 8
        )
        if adaptive_depth is not None
        else None,
        "adaptive_depth_average_loop_count": round(adaptive_depth.average_loop_count, 4)
        if adaptive_depth is not None
        else None,
        "adaptive_depth_ponder_cost": round(adaptive_depth.ponder_cost, 8)
        if adaptive_depth is not None
        else None,
        "adaptive_depth_toy_elapsed_ms": round(adaptive_depth.elapsed_ms, 4)
        if adaptive_depth is not None
        else None,
        "humaneval_style_task_count": humaneval_plan.task_count,
        "humaneval_style_runnable": humaneval_plan.runnable,
        "humaneval_style_missing_requirements": list(
            humaneval_plan.missing_requirements
        ),
        "swe_bench_runnable": swe_plan.runnable,
        "swe_bench_missing_requirements": list(swe_plan.missing_requirements),
        "terminal_bench_runnable": terminal_plan.runnable,
        "terminal_bench_missing_requirements": list(terminal_plan.missing_requirements),
        "turboquant_reconstruction_mse": round(tq.mse, 8),
        "turboquant_reconstruction_max_abs_error": round(tq.max_abs_error, 6),
    }


def main() -> None:
    parser = make_arg_parser("Run Alexandros smoke evaluation suites.")
    parser.add_argument("--config", default="configs/heavy_tiny.yaml")
    parser.add_argument("--compare-config", action="append", default=[])
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--out-jsonl", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--out-dir", default="")
    add_runtime_args(parser)
    args = parser.parse_args()

    device, dtype = setup_torch_runtime(args, torch)
    out_dir = prepare_output_dir(args)
    if out_dir is not None:
        args.out_jsonl = args.out_jsonl or str(out_dir / "eval.jsonl")
        args.out_md = args.out_md or str(out_dir / "eval.md")
    results = [
        evaluate_config(config_path, args, device, dtype, torch)
        for config_path in [args.config, *args.compare_config]
    ]
    if args.out_jsonl:
        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
    if args.out_md:
        out_path = Path(args.out_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = results[0]
        lines = [
            "# Alexandros Eval Summary",
            "",
            f"- Suite: `{result['suite']}`",
            f"- Config: `{result['config']}`",
            f"- AR shape: `{result['ar_shape']}`",
            f"- Diffusion shape: `{result['diffusion_shape']}`",
            f"- Perplexity: `{result['perplexity']}`",
            f"- Diffusion reconstruction accuracy: `{result['diffusion_reconstruction_accuracy']}`",
            f"- Latent reconstruction MSE: `{result['latent_reconstruction_mse']}`",
            f"- Latent KL loss: `{result['latent_kl_loss']}`",
            f"- Latent update norm: `{result['latent_update_norm']}`",
            f"- Total parameters: `{result['total_parameters']}`",
            f"- Active parameters/token: `{result['active_parameters_per_token']}`",
            f"- MoE layers with stats: `{result['moe_layers_with_stats']}`",
            f"- MoE mean load entropy: `{result['moe_mean_load_entropy']}`",
            f"- MoE timestep load entropy: `{result['moe_timestep_load_entropy']}`",
            f"- MoE noisy-step load entropy: `{result['moe_noisy_step_load_entropy']}`",
            f"- MoE polish-step load entropy: `{result['moe_polish_step_load_entropy']}`",
            f"- MLA cache compression: `{result['mla_cache_compression_ratio']}`",
            f"- TurboQuant cache compression: `{result['turboquant_cache_compression_ratio']}`",
            f"- TurboQuant reconstruction MSE: `{result['turboquant_reconstruction_mse']}`",
            f"- Prefill FLOPs: `{result['prefill_flops']}`",
            f"- Decode-token FLOPs: `{result['decode_token_flops']}`",
            f"- Prefill latency ms: `{result['prefill_ms']}`",
            f"- Generation latency ms: `{result['generation_ms']}`",
            f"- Generation tokens/sec: `{result['generation_tokens_per_second']}`",
            f"- Parameter bytes: `{result['parameter_bytes']}`",
            f"- Peak CUDA bytes: `{result['peak_cuda_bytes']}`",
            f"- Needle target rank: `{result['needle_target_rank']}`",
            f"- Needle target probability: `{result['needle_target_probability']}`",
            f"- Lost-middle worst rank: `{result['lost_middle_worst_rank']}`",
            f"- Lost-middle middle rank: `{result['lost_middle_middle_rank']}`",
            f"- Copy target rank: `{result['copy_target_rank']}`",
            f"- Copy target probability: `{result['copy_target_probability']}`",
            f"- Recurrent state layers: `{result['recurrent_state_layers']}`",
            f"- Recurrent state max norm: `{result['recurrent_state_max_norm']}`",
            f"- Recurrent state mean update norm: `{result['recurrent_state_mean_update_norm']}`",
            f"- Recurrent state finite: `{result['recurrent_state_finite']}`",
            f"- Toy reasoning target rank: `{result['toy_reasoning_target_rank']}`",
            f"- Toy reasoning target probability: `{result['toy_reasoning_target_probability']}`",
            f"- Adaptive-depth average loop count: `{result['adaptive_depth_average_loop_count']}`",
            f"- Adaptive-depth toy elapsed ms: `{result['adaptive_depth_toy_elapsed_ms']}`",
            f"- HumanEval-style runnable: `{result['humaneval_style_runnable']}`",
            f"- HumanEval-style missing requirements: `{result['humaneval_style_missing_requirements']}`",
            f"- SWE-Bench runnable: `{result['swe_bench_runnable']}`",
            f"- Terminal-Bench runnable: `{result['terminal_bench_runnable']}`",
            "",
            "## Generation Mode Smoke",
            "",
            "| Mode | Shape | New tokens | Continuation checksum |",
            "| --- | --- | ---: | ---: |",
        ]
        for mode, mode_result in result["generation_modes"].items():
            lines.append(
                "| "
                f"`{mode}` | "
                f"`{mode_result['shape']}` | "
                f"{mode_result['new_tokens']} | "
                f"{mode_result['continuation_sum']} |"
            )
        lines.extend(
            [
                "",
            ]
        )
        if len(results) > 1:
            lines.extend(
                [
                    "## Baseline Comparison",
                    "",
                    "| Config | Params | Active/token | PPL | Diff acc | Gen tok/s | Needle rank |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in results:
                lines.append(
                    "| "
                    f"`{row['config']}` | "
                    f"{row['total_parameters']} | "
                    f"{row['active_parameters_per_token']} | "
                    f"{row['perplexity']} | "
                    f"{row['diffusion_reconstruction_accuracy']} | "
                    f"{row['generation_tokens_per_second']} | "
                    f"{row['needle_target_rank']} |"
                )
            lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")
    print(results[0] if len(results) == 1 else results)


if __name__ == "__main__":
    main()
