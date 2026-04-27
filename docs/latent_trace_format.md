# Latent Trace Input Format

Alexandros does not ship or curate a visible-reasoning trace corpus. The latent
trace interface exists so downstream trainers can supply already-tokenized trace
examples and verify that the latent VAE/refinement mechanism can train against
them.

## JSONL Record

Each line must be a JSON object:

```json
{"input_ids":[4,5,6],"trace_ids":[7,8,9],"target_ids":[10]}
```

- `input_ids`: required prompt/problem token IDs.
- `trace_ids`: required visible reasoning trace token IDs. In v1, the latent
  trainer encodes these tokens with the current backbone and reconstructs their
  pooled hidden representation.
- `target_ids`: optional final-answer token IDs, validated and batched for
  future supervised extensions. The v1 latent reconstruction objective does not
  consume them.

Field names can be changed with `--trace-input-field`, `--trace-field`, and
`--trace-target-field`.

## Batching Semantics

- All IDs must be non-boolean integers inside `[0, vocab_size)`.
- Empty `input_ids` or `trace_ids` are rejected.
- Records are right-padded with `pad_token_id` to `--seq-len` and
  `--trace-len`.
- Overlong fields are truncated on the right in the reference loader.
- The trainer passes attention masks to the backbone and uses masked pooling so
  padding does not contribute to the reconstruction target.
- Dataset split and iterator state use the same deterministic machinery as the
  token-ID smoke loaders.

## Trainer Usage

```bash
python scripts/train_latent.py \
  --config configs/heavy_debug.yaml \
  --steps 1 \
  --batch-size 1 \
  --seq-len 64 \
  --trace-len 64 \
  --trace-jsonl path/to/traces.jsonl \
  --validation-fraction 0.05 \
  --val-every 100
```

Do not pass `--token-ids-jsonl` and `--trace-jsonl` together. `--trace-jsonl`
selects the latent-trace objective input; plain `--token-ids-jsonl` remains the
hidden-state reconstruction smoke path.

## Synthetic Toy Trace Helpers

`alexandros.training.make_modular_addition_latent_trace_records(...)` and
`alexandros.training.make_boolean_xor_latent_trace_records(...)` create tiny
already-tokenized records for CI and mechanism tests. They are deliberately
simple math/logical tasks with checkable `target_ids`; they are not a reasoning
corpus and should not be used for quality claims.

The helper vocabulary is:

- `4`: modular-addition task marker.
- `5`: XOR task marker.
- `6`: equals/derivation marker.
- `7`: answer marker.
- `8+`: integer value tokens.

## Debug Summaries

`alexandros.training.summarize_latent_trace_record(...)` renders a
`LatentTraceRecord` into `input`, `trace`, and optional `target` strings for
research debugging. Pass a token decoder as a callable, `dict[int, str]`, or
token string sequence when a tokenizer vocabulary is available. Without a
decoder, the helper returns numeric token IDs as strings; missing mapping entries
fall back to `<token_id>`.

## Privacy And Audit Caveats

Visible traces may contain private, sensitive, or copyrighted material. Users
supplying traces are responsible for tokenization, consent, license review,
retention policy, and release constraints.

Latent slots are internal activations, not faithful explanations. Reconstructing
or summarizing trace-like targets can help debug the training mechanism, but it
must not be represented as proof that hidden latent reasoning is transparent or
complete.
