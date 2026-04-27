# Contributing To Open-Alexandros

This repository is a research prototype. Prefer small, verifiable changes over
large rewrites.

## Development Setup

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e .[dev]
```

Optional safetensors support:

```bash
pip install safetensors
```

## Test Commands

Run these before handing off a code change:

```bash
python -m ruff check src scripts tests examples
python -m ruff format --check src scripts tests examples
python -m pytest -q
python -m compileall -q src scripts tests examples
python scripts/generate.py --config configs/heavy_debug.yaml --prompt-ids 1,2,3 --max-new-tokens 2 --use-cache
python scripts/eval.py --config configs/heavy_debug.yaml --out-dir runs/eval_smoke
```

For trainer changes, also run at least one tiny training command:

```bash
python scripts/pretrain.py --config configs/heavy_debug.yaml --steps 1 --val-every 1 --out-dir runs/smoke_ar
```

## Coding Style

- Ruff is the formatter, import sorter, and lint gate.
- Keep reference implementations readable and CPU-testable.
- Prefer existing helpers and config patterns before adding abstractions.
- Add tests for normal behavior and one failure or edge case.
- Document any approximation that remains intentionally reference-only.
- Do not commit generated checkpoints, local runs, caches, or datasets.

## Updating The Roadmap

Update `Roadmap.md` in the same change when a task becomes complete. A task is
complete only when the implementation, tests, docs, and smoke path are present
where relevant.

If work exposes a new gap, add it to the roadmap rather than hiding it in a code
comment.
