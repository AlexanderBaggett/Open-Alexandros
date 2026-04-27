# Hardware Targets

Alexandros v1 is a correctness-first PyTorch reference implementation. Hardware
targets describe what the repository should keep working, not performance
claims.

## CPU-Only Minimum Viable Path

- Required for CI and local contributor validation.
- Expected workflows:
  - Import package modules.
  - Load debug configs.
  - Run `python -m ruff check src scripts tests examples`.
  - Run `python -m ruff format --check src scripts tests examples`.
  - Run `python -m compileall -q src scripts tests examples`.
  - Run the pytest suite.
  - Run tiny CLI smoke commands for generation, evaluation, and short training.
- Precision: `float32` by default; CPU `bfloat16` autocast may be used by smoke
  trainers when PyTorch supports it.
- Non-goals: speed, long-context stress tests, realistic training throughput,
  or native packed BitLinear inference.

## Single Consumer GPU Path

- Intended for local experiments with debug/tiny configs and small custom
  datasets.
- Expected workflows:
  - AR, diffusion, latent, and TTT smoke trainers.
  - Mixed precision with `--dtype float16` or `--dtype bfloat16`.
  - Benchmark script comparisons across small configs.
  - Optional safetensors checkpoint export.
- Constraints:
  - No requirement for Flash Attention, Triton, vLLM, or custom CUDA kernels.
  - Long-context tests should scale down when memory is insufficient.
  - Lite still uses reference `BitLinear`; packed runtime kernels are future
    serving work.

## Single Data-Center GPU Path

- Intended for larger local research runs once downstream users provide data.
- Expected workflows:
  - Larger configs when memory allows.
  - BF16/FP16 training experiments using standard PyTorch optimizers.
  - Profiling MLA, TurboQuant, MoE, and diffusion variants with repository
    benchmark scripts.
- Constraints:
  - Reference modules must remain correct without optimized kernels.
  - Any optional kernel path must preserve eager/CPU fallbacks and parity tests.
  - Published experiment reports should include config hash, dtype, device,
    hardware name, dependency versions, and peak memory when available.

## Multi-GPU Training Path

- Not a v1 supported implementation path.
- Required before claiming support:
  - `torchrun` entrypoint.
  - DDP/FSDP strategy.
  - MoE expert-parallel dispatch contract.
  - Save/resume tests with deterministic seed handling.
  - Checkpoint compatibility with single-process `from_pretrained`.
- Expert-parallel work must document dispatch tensor shapes and all-to-all
  requirements before code lands.

## Expected Unsupported Paths

- Internet-scale data collection or scraping.
- Production distributed training claims.
- vLLM, Text Generation Inference, bitnet.cpp, or packed BitLinear serving
  claims.
- Custom CUDA/Triton kernels without CPU/eager parity tests.
- Hardware-specific claims that do not include reproducible benchmark commands,
  dependency versions, dtype, and model config.
