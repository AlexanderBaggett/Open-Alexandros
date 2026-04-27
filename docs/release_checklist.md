# Alexandros Release Checklist

Use this checklist before publishing checkpoints, configs, or evaluation
results.

## Checkpoint Integrity

For each released artifact:

```bash
sha256sum config.json generation_config.json alexandros_extra.json
sha256sum pytorch_model.bin  # or model.safetensors
```

Publish the checksums next to the checkpoint. For signed releases, sign the
checksum file rather than each large artifact:

```bash
sha256sum * > SHA256SUMS
gpg --detach-sign --armor SHA256SUMS
```

Prefer `model.safetensors` for untrusted distribution. `pytorch_model.bin` is
supported for local compatibility, but it relies on PyTorch serialization and
should be treated as trusted input.

## Required Files

- `config.json`
- `generation_config.json`
- `alexandros_extra.json`
- `model.safetensors` preferred, or `pytorch_model.bin`
- Model card
- Dataset card or explicit statement that no dataset is bundled
- Evaluation report
- Known limitations and misuse statement

For the initial architecture release, prefer the explicit no-dataset statement
unless a tiny CI/sample fixture is actually bundled. Do not imply that smoke
fixtures are pretraining data or that Alexandros publishes an official trained
base model.

## Validation

Run:

```bash
python -m ruff check src scripts tests examples
python -m ruff format --check src scripts tests examples
python -m pytest -q
python -m compileall -q src scripts tests examples
python scripts/eval.py --config configs/heavy_debug.yaml --out-dir runs/release_eval_smoke
```

Record hardware, Python version, Torch version, config hash, and git commit in
the evaluation report.

## Training Data Scope

Before publishing a checkpoint, state one of:

- no training data is bundled and no useful model-quality checkpoint is being
  released;
- only tiny synthetic/permissively licensed fixtures are bundled for CI; or
- a downstream trainer supplied data, tokenizer, and licenses, with separate
  dataset/model cards.

If a contributor publishes a distilled checkpoint, the release notes must name
the teacher source, cite the teacher license/terms, and state whether generated
targets, logits, or both were used.
