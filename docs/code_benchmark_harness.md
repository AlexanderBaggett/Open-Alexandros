# Code Benchmark Harness Placeholder

Alexandros does not claim runnable HumanEval, SWE-Bench, or Terminal-Bench
quality in v1. The repository exposes explicit planning helpers so evaluation
reports can say why those suites are not ready instead of silently omitting
them.

## Current Public API

`alexandros.evaluation.code_benchmarks` provides:

- `CodeBenchmarkSuite`: `humaneval_style`, `swe_bench`, and `terminal_bench`.
- `CodeBenchmarkPrerequisites`: tokenizer, trained checkpoint, execution
  sandbox, release policy, dataset-license review, and tool-environment gates.
- `code_benchmark_plan(...)`: returns runnable status and missing requirements.
- `ensure_code_benchmark_ready(...)`: raises `CodeBenchmarkNotReadyError` if a
  suite is requested before the required gates are met.
- `HumanEvalStyleTask`: metadata container for prompt, entry point, and tests.
- `validate_humaneval_style_tasks(...)`: validates non-empty tasks and duplicate
  IDs without executing code.
- `humaneval_style_harness_report(...)`: summarizes task count plus readiness
  status.

`scripts/eval.py` records HumanEval-style, SWE-Bench, and Terminal-Bench
readiness fields in its JSONL and Markdown outputs. By default these are not
runnable because the repo intentionally has no production tokenizer, trained
checkpoint, benchmark dataset review, or execution sandbox.

## HumanEval-Style Boundary

The v1 placeholder validates task metadata only. It does not:

- execute generated code;
- load HumanEval data;
- score pass@k;
- claim model quality.

A real harness must add a reviewed sandbox, timeout/resource controls, prompt
format, tokenizer contract, pass@k computation, and dataset/license record.

## SWE-Bench And Terminal-Bench Boundary

SWE-Bench and Terminal-Bench-style evaluations are future coding-agent scale
work. Before implementation, they need:

- isolated repository checkout handling;
- controlled terminal execution;
- network policy;
- patch/test-result capture;
- task-level environment metadata;
- safety and misuse review for generated commands.

Until those pieces exist, Alexandros reports these suites as planned evaluation
targets rather than runnable benchmarks.
