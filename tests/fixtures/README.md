# Test Fixtures

These files are tiny, synthetic fixtures for correctness tests only. They are
not training data, evaluation data, benchmark data, or evidence of model
quality.

## `token_ids_tiny.jsonl`

- Source: synthetic token IDs hand-written for Open-Alexandros tests.
- License: MIT License, same as the repository.
- Allowed use: unit, integration, and CI smoke tests.
- Disallowed use: pretraining, quality claims, benchmark comparisons, or model
  release claims.
- SHA-256:
  `273d948c45eedf7c2a958d74a164214307d5efc4a04b94ab23159319ebe73eb8`

The fixture assumes the standard tiny-token contract used by tests:
`pad=0`, `bos=1`, `eos=2`, `mask=3`, and regular tokens starting at `4`.

## `latent_traces_tiny.jsonl`

- Source: synthetic token IDs hand-written for Open-Alexandros tests.
- License: CC0-1.0.
- Allowed use: latent trace ingestion and CI smoke tests.
- Disallowed use: pretraining, quality claims, reasoning benchmark claims, or
  model release claims.
- SHA-256:
  `9ea2a3f2f76dbc258fd24b3113176ae74db518bcb3aae25a872d2d04d0adff83`

The fixture uses already-tokenized `input_ids`, `trace_ids`, and optional
`target_ids`. It exists only to prove that user-supplied visible traces can flow
through the parser and latent trainer.

## `distillation_tiny.jsonl`

- Source: synthetic token IDs and synthetic logits hand-written for
  Open-Alexandros tests.
- License: CC0-1.0.
- Allowed use: distillation ingestion and loss-mechanism CI smoke tests only.
- Disallowed use: pretraining, teacher-model quality claims, benchmark claims,
  or model release claims.
- SHA-256:
  `ce6616d2863991fa892b08e1fcefd4fa9ab608cf9181ae1f746966b2cda3f084`

The fixture uses `input_ids` plus `teacher_token_ids` and/or `teacher_logits`
with `vocab_size=16`. It contains no outputs from a real teacher model.
