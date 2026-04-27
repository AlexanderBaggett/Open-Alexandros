# Dependency Strategy

Open-Alexandros uses PyTorch, but PyTorch wheels are hardware-specific. The
project therefore does not install `torch` as a default dependency.

## Recommended CPU Setup

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e .[dev]
```

## GPU Setup

Install the PyTorch wheel that matches your CUDA/ROCm target using the official
PyTorch instructions, then run:

```bash
pip install -e .[dev]
```

## Optional Extras

- `.[dev]`: pytest, PyYAML, and Ruff for tests, YAML configs, formatting,
  import sorting, and linting.
- `.[runtime]`: generic `torch>=2.1` for environments where the default PyPI
  wheel is known to be appropriate.
- `.[serialization]`: safetensors for safe checkpoint serialization.
- `.[tracking]`: TensorBoard for optional `--tensorboard-dir` scalar logging.
- `.[hf]`: Transformers for the optional Hugging Face config/model wrappers and
  AutoClass registration helper.

## Transformers Compatibility

`transformers` is not a default v1 dependency. Alexandros writes Hugging
Face-style checkpoint filenames (`config.json`, `generation_config.json`, and
model weight files) and records the compatibility decision in
`alexandros_extra.json`.

Installing `.[hf]` enables optional wrappers:

- `AlexandrosHFConfig`
- `AlexandrosHFModel`
- `AlexandrosHFForCausalLM`
- `AlexandrosHFForDiffusionLM`
- `register_alexandros_with_transformers()`

These wrappers are an integration bridge, not a replacement for the reference
`save_pretrained`/`from_pretrained` path. The reference save/load path must keep
working without installing `transformers`.

## Version Bounds

`pyproject.toml` uses lower and upper bounds for direct dependencies:

- `numpy>=1.26,<3`
- `pytest>=7,<9`
- `pyyaml>=6,<7`
- `ruff>=0.5,<1`
- `torch>=2.1,<3` in the optional `runtime` extra
- `safetensors>=0.4,<1` in the optional `serialization` extra
- `tensorboard>=2.15,<3` in the optional `tracking` extra
- `transformers>=4.40,<5` in the optional `hf` extra

These are compatibility constraints, not a full lockfile. Reproducible training
runs should record exact installed versions in `metrics.jsonl` and, for serious
experiments, publish a generated lockfile or container image alongside the run
artifacts.

Optional CUDA, Triton, Flash Attention, vLLM, `mamba-ssm`, and native serving
dependencies are intentionally excluded until those integrations exist. The v1
Mamba-2 option is a local PyTorch diagonal SSM reference backend. Any future
native extension must document compiler, CUDA/ROCm, Python, and platform
requirements before becoming a supported install path.

## Why Not Default Torch?

Installing `torch` directly from the default PyPI index may pull a large wheel
stack that is wrong for CPU-only validation or for a specific accelerator. The
runtime dependency check in CLI scripts points users to an explicit PyTorch
install step instead.
