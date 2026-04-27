# Alexandros Examples

Run these from the repository root after installing the local package
dependencies.

```bash
python examples/minimal_model.py
python examples/tiny_ar_training.py
python examples/diffusion_generation.py
python examples/inspect_bitlinear.py
python examples/save_load_roundtrip.py
```

Expected behavior:

- `minimal_model.py` prints the logits shape for a tiny causal model.
- `tiny_ar_training.py` runs two optimizer steps and prints loss values.
- `diffusion_generation.py` runs block diffusion generation on a tiny prompt.
- `inspect_bitlinear.py` prints the ternary weight levels.
- `save_load_roundtrip.py` saves a temporary checkpoint and verifies logits
  match after loading.

These scripts use debug/tiny settings and are intended for API smoke checks, not
quality evaluation.
