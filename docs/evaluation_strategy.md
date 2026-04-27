# Evaluation Strategy

Alexandros evaluation is currently a mechanism and reproducibility layer. Tiny
metrics can detect regressions, broken routing, cache drift, and invalid
training plumbing, but they are not quality claims.

## Required Run Metadata

Every benchmark or external comparison report should include:

- config path and config hash;
- git commit when available;
- Python, PyTorch, and direct dependency versions;
- hardware name when available, CPU/GPU count, device, and dtype;
- tokenizer or token-id fixture contract;
- command used to produce the result;
- whether the run used synthetic fixtures, user-supplied data, or an external
  benchmark.

Repository scripts already record most of this in JSONL through
`run_metadata(...)`; reports should preserve those fields.

## Ablation Strategy

Use the existing config and `--compare-config` mechanisms where possible. Each
ablation should change one axis at a time, publish the changed config fields,
and report runtime/memory alongside task metrics.

| Ablation | Primary Mechanism | Required Notes |
| --- | --- | --- |
| Dense FFN vs MoE | Compare configs with `moe_num_experts=1` or sparse routing disabled against routed MoE configs. | Report active parameters per token and expert entropy. |
| Standard attention vs MLA | Compare current MLA path against a future standard-attention config or module once implemented. | Mark as future until the standard-attention path exists. |
| Full attention ratio variants | Change `linear_attention_ratio` or explicit `attention_layer_indices`. | Report attention-layer count and sequence length. |
| Gated DeltaNet vs attention-only | Use explicit attention layer indices for all-attention runs. | Note that all-attention diffusion can be bidirectional; Gated DeltaNet remains directional in v1. |
| Heavy vs Lite at matched active parameters | Compare Heavy/Lite configs with similar active parameter counts. | Report BitLinear activation bits and packed export status. |
| AR vs block diffusion vs latent reasoning | Use `scripts/eval.py` generation-mode smoke outputs and phase-specific losses. | Do not compare as final quality without a real tokenizer and held-out dataset. |
| TTT on/off for long context | Compare request-local `TTTState` probes with no-TTT runs. | Report fast-weight rank, update LR, chunk length, and verify checkpoint weights are unchanged. |
| TurboQuant on/off | Compare `use_turboquant_cache` configs and cache reconstruction metrics. | Report compression ratio, reconstruction error, and compressed-vs-uncompressed logit drift. |

## External Baseline Policy

Small open baselines may be used for local sanity checks only when their license
and terms allow the comparison. Suggested categories are:

- tiny public causal LMs for smoke perplexity/runtime comparisons;
- small open code models only for harness plumbing, not Alexandros quality
  claims;
- deterministic toy baselines for synthetic modular arithmetic, retrieval, and
  copy tasks.

Fair-tokenizer caveats are mandatory:

- If tokenizers differ, do not compare raw perplexity as a model-quality claim.
- Report tokenization method, vocabulary size, sequence length, and any prompt
  formatting for each model.
- Prefer token-rank, exact-match toy diagnostics, latency, and memory metrics
  for cross-tokenizer smoke comparisons.
- Any published claim must separate Alexandros mechanism tests from downstream
  trained-checkpoint evaluations.

## Reasoning And Code Benchmarks

The repository includes tiny deterministic reasoning probes for CI mechanics.
It also includes a non-executing
[code benchmark harness placeholder](code_benchmark_harness.md) that validates
HumanEval-style task metadata and reports missing prerequisites for
HumanEval-style, SWE-Bench, and Terminal-Bench suites.
For adaptive-depth experiments, `adaptive_depth_toy_benchmark(...)` reports the
toy modular-addition rank/probability alongside average loop count, ponder cost,
and elapsed milliseconds. Treat this as a cost-vs-mechanics diagnostic, not as
evidence of real reasoning ability.

The placeholder intentionally does not execute generated code, load benchmark
datasets, compute pass@k, or claim quality. Full HumanEval-style, SWE-Bench, or
Terminal-Bench harnesses should be made runnable only after a tokenizer, trained
checkpoint, execution sandbox, dataset/license review, and release policy exist.
Until then, code-benchmark fields in reports are readiness diagnostics and
future scale targets, not benchmark results.
