# Open-Alexandros

Alexandros is a research-oriented PyTorch prototype for a hybrid language
model family with two flavors:

- **Heavy**: BF16/FP16 dense linear layers for maximum quality.
- **Lite**: the same architecture with BitNet b1.58-style ternary
  `BitLinear` layers for efficient experimentation.

The repository is intentionally structured as an implementation handoff for
LLMs and human engineers. The design docs embed the core research assumptions
so an implementer does not need to rediscover the papers before working.

## What Is Included

- A research matrix mapping each architectural breakthrough to concrete
  Alexandros implementation requirements.
- A master design document with public APIs, training mechanisms, acceptance
  criteria, and known v1 simplifications.
- A runnable PyTorch package with:
  - MLA-style compressed KV attention.
  - Gated DeltaNet-inspired linear sequence blocks.
  - DeepSeek-style shared+routed MoE FFNs.
  - Heavy and Lite variants via a variant-aware linear factory.
  - Masked/block diffusion generation.
  - Latent reasoning modules.
  - TTT-E2E-inspired request-local fast weights.
  - TurboQuant-style reference KV compression.

## Quick Start

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e .[dev]
# Optional safe checkpoint serialization:
# pip install -e .[serialization]
# Optional Hugging Face wrapper integration:
# pip install -e .[hf]
python scripts/generate.py --config configs/heavy_tiny.yaml --prompt-ids 1,2,3
python scripts/generate.py --config configs/heavy_debug.yaml --prompt-ids 1,2,3 --use-cache
python scripts/generate.py --config configs/heavy_debug.yaml --prompt-ids 1,2,3 --do-sample --temperature 0.8 --top-k 5 --top-p 0.9
python scripts/generate.py --config configs/heavy_debug.yaml --mode block_diffusion --prompt-ids 1,2,3 --max-new-tokens 3 --do-sample --temperature 0.8 --top-k 5
python scripts/generate.py --config configs/heavy_debug.yaml --prompt-ids 1,2,3 --stop-sequences 8,9
python scripts/pretrain.py --config configs/heavy_debug.yaml --steps 1 --grad-accum-steps 2 --out-dir runs/smoke_ar
python scripts/pretrain.py --config configs/heavy_debug.yaml --steps 1 --val-every 1 --val-batches 1 --out-dir runs/smoke_ar
python scripts/pretrain.py --config configs/heavy_debug.yaml --steps 1 --out-dir runs/smoke_ar --resume runs/smoke_ar/training_state.pt
python scripts/benchmark.py --config configs/heavy_debug.yaml --compare-config configs/lite_debug.yaml --out-dir runs/benchmark
python scripts/export_hf.py --config configs/heavy_debug.yaml --out runs/export_debug
python examples/minimal_model.py
python -m ruff check src scripts tests examples
python -m ruff format --check src scripts tests examples
python -m pytest
```

Use Python 3.10-3.13. The project was validated locally with Python 3.11 and
CPU Torch. Torch is intentionally not installed as a default dependency because
the correct wheel depends on your CPU/GPU target.

The tiny configs are for smoke tests and architecture validation. They are not
intended to produce useful language quality.
The `heavy_1b.yaml` and `lite_1b.yaml` files are research-scale planning
targets. Use the estimator/eval tools before attempting to instantiate them on
limited hardware.

Smoke trainers accept `--out-dir` for a local artifact folder. When provided,
they write `metrics.jsonl` with run metadata such as config hash, git commit,
Python/Torch versions, device, dtype, seed, and deterministic-mode status. The
AR, diffusion, and latent smoke trainers also write `training_state.pt` for
trusted local resume and a `checkpoint/` directory for portable model loading.
Use `--val-every N` to append tiny deterministic validation metrics to every
Nth training step.

## Training Data Stance

Alexandros does not currently include a production tokenizer, curated
pretraining corpus, or official training curriculum. The goal of this repo is
to publish a research-grounded architecture with correct, inspectable training
mechanisms: loss computation, optimizer steps, resume, validation metrics,
checkpoint export, and evaluation hooks.

Real tokenizer choice, corpus curation, large-scale pretraining, distillation,
instruction tuning, and preference tuning are intentionally left to downstream
trainers and the broader AI community. The repo may include tiny synthetic or
permissively licensed sample fixtures for CI only; those fixtures must not be
presented as quality training data. Internet scraping and web-scale data
collection are out of scope for this repository.

## Key Documents

- [Master design](docs/alexandros_master_design.md)
- [Research matrix](docs/research_matrix.md)
- [Architecture decisions](docs/architecture_decisions.md)
- [Architecture compatibility contracts](docs/architecture_contracts.md)
- [API reference](docs/api_reference.md)
- [Checkpoint format](docs/checkpoint_format.md)
- [Code benchmark harness placeholder](docs/code_benchmark_harness.md)
- [Latent trace format](docs/latent_trace_format.md)
- [Tokenizer contract](docs/tokenizer_contract.md)
- [Test fixtures](docs/test_fixtures.md)
- [Training objective contracts](docs/training_objective_contracts.md)
- [Training guide](docs/training_guide.md)
- [Community training curriculum](docs/community_training_curriculum.md)
- [Data policy](docs/data_policy.md)
- [Distributed MoE planning](docs/distributed_moe.md)
- [Evaluation strategy](docs/evaluation_strategy.md)
- [Hardware targets](docs/hardware_targets.md)
- [Optimized kernel roadmap](docs/optimized_kernels.md)
- [Safety evaluation plan](docs/safety_evaluation.md)
- [Serving integration plan](docs/serving_integration.md)
- [Post-training scope](docs/post_training_scope.md)
- [Roadmap and implementation gaps](Roadmap.md)
- [Contributing](docs/contributing.md)
- [Release checklist](docs/release_checklist.md)
- [Dependency strategy](docs/dependency_strategy.md)
- [Examples](examples/)

## Status

This is a reference prototype. It favors clarity, testability, and clean module
boundaries over optimized kernels. Triton, vLLM, distributed training, and
bitnet.cpp serving integrations are deliberately left as later production
paths.
