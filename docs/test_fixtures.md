# Test Fixtures

Alexandros includes only tiny synthetic fixtures for correctness tests. These
fixtures prove that data-loading and training mechanisms accept externally
prepared token IDs; they are not model-training data and must not be used for
quality claims.

## `tests/fixtures/token_ids_tiny.jsonl`

- Source: synthetic token IDs hand-written for Open-Alexandros tests.
- License: MIT License, same as the repository.
- Allowed use: unit, integration, and CI smoke tests only.
- Disallowed use: pretraining, quality claims, benchmark comparisons, or model
  release claims.
- Format: JSONL objects with an `input_ids` list.
- SHA-256:
  `273d948c45eedf7c2a958d74a164214307d5efc4a04b94ab23159319ebe73eb8`

The fixture follows the standard tiny-token contract used by tests:
`pad=0`, `bos=1`, `eos=2`, `mask=3`, and regular tokens starting at `4`.
It is already tokenized and intentionally does not imply any production
tokenizer choice.

## `tests/fixtures/latent_traces_tiny.jsonl`

- Source: synthetic token IDs hand-written for Open-Alexandros tests.
- License: CC0-1.0.
- Allowed use: latent trace ingestion and CI smoke tests only.
- Disallowed use: pretraining, quality claims, reasoning benchmark claims, or
  model release claims.
- Format: JSONL objects with `input_ids`, `trace_ids`, and optional
  `target_ids` lists.
- SHA-256:
  `9ea2a3f2f76dbc258fd24b3113176ae74db518bcb3aae25a872d2d04d0adff83`

This fixture verifies mechanics only: parser validation, right-padding masks,
resumeable trace batching, and `train_latent.py --trace-jsonl`. It is not a
reasoning dataset.

## `tests/fixtures/distillation_tiny.jsonl`

- Source: synthetic token IDs and synthetic logits hand-written for
  Open-Alexandros tests.
- License: CC0-1.0.
- Allowed use: distillation ingestion and loss-mechanism CI smoke tests only.
- Disallowed use: pretraining, teacher-model quality claims, benchmark claims,
  or model release claims.
- Format: JSONL objects with `input_ids` plus `teacher_token_ids` and/or
  `teacher_logits`.
- SHA-256:
  `ce6616d2863991fa892b08e1fcefd4fa9ab608cf9181ae1f746966b2cda3f084`

This fixture verifies mechanism support for user-supplied teacher targets and
teacher logits. It contains no outputs from a real teacher model and does not
imply any teacher choice for Alexandros.

## Synthetic Latent Toy Helpers

`make_modular_addition_latent_trace_records(...)` and
`make_boolean_xor_latent_trace_records(...)` generate small semantic trace
records in memory for tests. They use task markers `4` and `5`, an equals marker
`6`, an answer marker `7`, and value tokens starting at `8`. Their purpose is to
prove latent-trace ingestion can carry checkable math/logical targets without
bundling real training data.
