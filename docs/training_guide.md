# Alexandros Training Guide

This guide describes the current smoke-training path and training-mechanism
handoffs. The scripts are correctness and API exercises, not production
pretraining loops.

## Scope

Alexandros does not currently ship a production tokenizer, a curated
pretraining corpus, or an official training curriculum. The repo's training
responsibility is narrower: prove that the model can train, resume, validate,
log metrics, and export checkpoints using synthetic data, tiny sample fixtures,
or user-supplied token-id JSONL.

Large-scale corpus selection, dataset filtering, tokenizer training, and
post-training are intentionally deferred to downstream trainers and the AI
community. This repository should not add internet scraping or web-scale data
collection scripts unless the project scope is explicitly changed.
See [Post-training scope](post_training_scope.md) for the base-model boundary
and future SFT/preference/tool-use gates.
See [Data policy](data_policy.md) and
[Safety evaluation plan](safety_evaluation.md) for downstream dataset and
user-facing release requirements.
See [Community training curriculum](community_training_curriculum.md) for a
non-official outline downstream trainers can adapt.

The only checked-in data fixtures are tiny synthetic token-id and latent-trace
JSONL files for CI mechanism tests. See [Test fixtures](test_fixtures.md).
They are not useful training data and must not be cited as evidence of model
quality.

Optional distillation is supported as a mechanism by accepting user-provided
teacher token targets or dense teacher logits through
`DistillationDataset.from_jsonl(...)` and `distillation_loss(...)`. Teacher
model choice is not part of the initial release. License verification, terms
compliance, generated-output rights, and redistribution rights belong to the
downstream trainer before any distilled checkpoint is published.

## AR Phase

Use `scripts/pretrain.py` to validate causal LM training, optimizer updates,
MoE router-bias updates, checkpoint save/resume, JSONL logging, and optional
validation.

Recommended smoke command:

```bash
python scripts/pretrain.py \
  --config configs/heavy_debug.yaml \
  --steps 2 \
  --grad-accum-steps 2 \
  --val-every 1 \
  --out-dir runs/smoke_ar
```

Resume:

```bash
python scripts/pretrain.py \
  --config configs/heavy_debug.yaml \
  --steps 1 \
  --resume runs/smoke_ar/training_state.pt \
  --out-dir runs/smoke_ar
```

The AR mechanism trains embeddings, hybrid backbone, MoE experts, and LM head.
For real training, downstream trainers should supply packed tokenizer data and
a held-out validation split.

The smoke trainers choose a variant-aware default learning rate when `--lr` is
omitted: `1e-3` for Heavy and `3e-4` for Lite. Override it explicitly for real
runs. Use `--warmup-steps N` to linearly warm up from zero to the selected LR
over the first `N` optimizer steps. Each JSONL metric record includes the
effective `lr`. Runs without `--token-ids-jsonl` are logged with
`data_source="synthetic_smoke"` so synthetic batches cannot be mistaken for a
real corpus. Every smoke trainer also emits `objective_*` metric fields; see
[Training objective contracts](training_objective_contracts.md).
`alexandros.training.standard_metric_names(phase)` returns the required JSONL
field names for each smoke phase, and
`validate_standard_metric_record(record, phase)` can be used by downstream
wrappers to fail fast when a trainer or experiment tracker drops required
fields.

Smoke trainers apply explicit phase trainability controls before creating their
optimizer:

- AR `phase_default`: all causal model parameters.
- Diffusion `phase_default`: backbone plus LM head; latent VAE/reasoner modules
  are frozen because this phase does not use them.
- Latent `phase_default`: backbone plus latent VAE and latent reasoner; LM head
  is frozen because the latent objective does not use token logits.
- TTT `phase_default`: no checkpoint model parameters are trainable; only the
  request-local `TTTState` is updated.

Use `--trainable-scope` to override the smoke default with `all`,
`backbone_only`, `head_only`, `latent_only`, or `none`. Metric logs include
`trainability_*` fields.

## Precision Policy

AR, diffusion, and latent smoke trainers keep FP32 master weights during
training. When `--dtype bfloat16` or `--dtype float16` is selected, forward and
validation passes run under `torch.autocast`; optimizer state and checkpointed
weights remain FP32. CUDA float16 additionally uses `GradScaler`, and the scaler
state is saved in `training_state.pt`. CPU bfloat16 uses autocast without a
scaler. CPU float16 training is rejected because the reference path does not
reliably support it.

Lite keeps embeddings, normalization layers, timestep embeddings, and router
bias state full precision. `BitLinear` keeps full-precision shadow weights even
though its forward path uses ternary weights.

Initialization is explicit across Heavy and Lite. Projection weights use a
small normal initializer, residual-output projections are depth-scaled, pad
embeddings start at zero, normalization weights start at one, and normalization
biases start at zero. MoE load-balancing buffers and optional timestep,
token-state, and position router biases start neutral at zero so routing offsets
are learned rather than accidentally inherited from framework defaults.

## Token-ID JSONL Data

The smoke trainers can read pre-tokenized JSONL via `--token-ids-jsonl`. Each
line may be either a bare list of integer token IDs or an object with an
`input_ids` field:

```json
{"input_ids": [1, 42, 43, 2]}
```

Use `--token-field` to choose a different object field. The loader validates
IDs against `vocab_size`, appends `eos_token_id`, packs records into fixed
`--seq-len` chunks, and repeats deterministic shuffled batches. AR and TTT
training prepend `bos_token_id`; diffusion and latent training use the chunks
as-is. `--validation-fraction 0.1` creates a deterministic record-level
validation split for `--val-every` runs. Train iterators save their cursor,
shuffled order, and generator state in `training_state.pt`, so `--resume`
continues from the same packed-data position. The tokenizer itself is supplied
by the trainer; this path is intentionally for already-tokenized data.
See [Tokenizer contract](tokenizer_contract.md) for the optional checkpoint
metadata format. Passing `tokenizer_metadata` to `save_pretrained` records
user-supplied tokenizer details without making Alexandros responsible for
training or loading that tokenizer.

## Diffusion Phase

Use `scripts/train_diffusion.py` to validate masked-token loss, timestep-aware
MoE routing, validation loss, and expert-use metrics.

```bash
python scripts/train_diffusion.py \
  --config configs/heavy_debug.yaml \
  --steps 2 \
  --val-every 1 \
  --out-dir runs/smoke_diffusion
```

The current objective masks absorbing tokens with a simple timestep schedule
and normalizes weighted loss by masked-token count. The default
`diffusion_loss_weighting="uniform"` preserves the v1 unweighted objective.
`mask_prob` and `inverse_mask_prob` are available as explicit research knobs
for timestep-weighted smoke experiments. `diffusion_objective="rao_blackwellized"`
expands each sampled noised background over all non-pad target positions and
averages the weighted target losses; use `diffusion_rb_chunk_size` to cap the
expanded batch size. These mechanisms are correctness references, not
performance-optimized diffusion kernels.

Diffusion metrics include `moe_timestep_tracked_selections` for routed expert
selections observed with timestep conditioning. `moe_timestep_load_entropy`
reports per-denoising-step expert balance, while
`moe_noisy_step_load_entropy` and `moe_polish_step_load_entropy` split those
diagnostics by high-mask-probability and low-mask-probability timesteps.

Block diffusion generation supports `steps`, optional `block_size` chunking,
`confidence_schedule` values of `median`, `linear`, or `all`, and optional
`remask_low_confidence` for rewriting low-confidence generated block positions
back to the absorbing mask before later denoising steps. These controls are
reference sampler mechanics, not a claim of generation quality.

For standalone data tests or future training loops,
`make_diffusion_training_batch(...)` creates reproducible absorbing-mask
training batches with sampled timesteps, noisy inputs, `-100` labels outside
masked positions, and masked-token counts for loss normalization.
`make_block_diffusion_training_batch(...)` creates contiguous masked-block
targets for block diffusion experiments while keeping pad tokens ignored.

## Latent Phase

Use `scripts/train_latent.py` to validate the latent VAE and reconstruction
path.

```bash
python scripts/train_latent.py \
  --config configs/heavy_debug.yaml \
  --steps 2 \
  --val-every 1 \
  --out-dir runs/smoke_latent
```

By default, the current target reconstructs a pooled hidden-state proxy. When
users have already-tokenized visible reasoning traces, pass `--trace-jsonl` to
train against trace-derived hidden targets instead:

```bash
python scripts/train_latent.py \
  --config configs/heavy_debug.yaml \
  --steps 2 \
  --trace-jsonl path/to/traces.jsonl \
  --seq-len 64 \
  --trace-len 64 \
  --validation-fraction 0.05 \
  --val-every 100 \
  --out-dir runs/trace_latent
```

The trace format is documented in
[`docs/latent_trace_format.md`](latent_trace_format.md). The repo bundles only
synthetic/toy trace fixtures for CI; real trace data, task-level reasoning
evaluation, consent, and licensing are downstream responsibilities. The smoke
trainer exercises both `LatentThoughtVAE` reconstruction and
`LatentDiffusionReasoner` bidirectional slot-attention refinement/decode
mechanics.
Set `latent_adaptive_threshold > 0` in the config to allow the latent reasoner
to stop early when the latest bounded refinement update is small. The default
`0.0` keeps fixed-step refinement for reproducibility.

The latent smoke objective is:

```text
loss = lambda_rec * mse(reconstruction, pooled_hidden_target)
     + lambda_kl * kl_loss
```

Use `--lambda-rec` and `--lambda-kl` to make those mechanism-test weights
explicit. Both must be finite and non-negative, and at least one must be
positive. `reconstruction_loss` is the mean of the direct VAE reconstruction
loss and the latent-refinement reconstruction loss. Use
`--latent-refinement-steps` to control the number of reasoner update steps. The
metrics log records `reconstruction_loss`, `vae_reconstruction_loss`,
`refinement_reconstruction_loss`, `kl_loss`, `lambda_rec`, `lambda_kl`, and
`latent_refinement_steps`. `data_source` is `latent_trace_jsonl` when the trace
loader is used.

## TTT Phase

Use `scripts/meta_train_ttt.py` to run the reference TTT-E2E-style mechanism:
split long contexts into prefix and heldout chunks, unroll differentiable
fast-weight updates on prefix chunks, and optimize the learned low-rank
`TTTMetaAdapter` initialization on the heldout outer loss.

```bash
python scripts/meta_train_ttt.py \
  --config configs/heavy_debug.yaml \
  --prefill-chunk-len 16 \
  --out-dir runs/smoke_ttt
```

By default `phase_default` freezes the base checkpoint and trains only the
separate `TTTMetaAdapter` learned fast-weight initialization. Explicit
`--trainable-scope all`, `backbone_only`, or `head_only` lets researchers update
checkpoint parameters through the outer loss. Long-context training data should
be user supplied or synthetic for tests.

The reference `TTTState` exposes explicit request-local operations:
`reset()` clears fast weights, `prefill_update(...)` updates them from hidden
states using an optional local RNG generator, and `apply(...)` mixes the adapted
residual back into hidden states with an optional gate. `update(...)` remains an
alias for compatibility. These fast weights are never part of model
`state_dict()` or portable checkpoints.
Use `--prefill-chunk-len` to split contexts; the final valid chunk is held out
for the outer loss. The metrics include `inner_loss`, `outer_loss`,
`ttt_inner_lr`, `prefill_chunk_len`, `prefill_chunk_count`,
`ttt_heldout_chunk_len`, `pre_update_loss`, and `post_update_loss`. Learned TTT
components are saved to `checkpoint/ttt_meta_adapter.pt`; request-local
`TTTState.fast_a`/`fast_b` tensors are not saved.

## Lite Cautions

Lite replaces linear projections with `BitLinear` and keeps trainable
full-precision shadow weights. Embeddings, normalization, timestep embeddings,
and router-bias state remain full precision in the reference implementation.

Use smaller learning rates, gradient clipping, and warmup when changing Lite
training. The current default is `3e-4` with `--grad-clip 1.0`; for unstable
runs start with `--warmup-steps 100` or longer and reduce LR before changing
the architecture. The current smoke tests cover forward/backward finiteness and
a few tiny optimizer steps, not quality convergence.

Calibration evidence for the default smoke settings is intentionally tiny and
CPU-local. On `configs/heavy_debug.yaml`, a 3-step deterministic AR smoke run
with the default Heavy LR `1e-3` stayed finite with losses `4.1666`, `4.1710`,
`4.1638` and gradient norms `3.2920`, `1.9114`, `1.9757`. On
`configs/lite_debug.yaml`, the same run with the default Lite LR `3e-4` stayed
finite with losses `4.1671`, `4.1734`, `4.1586` and gradient norms `3.1502`,
`1.9195`, `2.2410`. A Lite run with `--warmup-steps 2` produced effective LRs
`0.00015`, `0.0003`, `0.0003` and remained finite. These observations justify
the smoke defaults only; downstream training should recalibrate LR, warmup, and
clipping on real data and hardware.

Use `scripts/export_hf.py --config configs/lite_debug.yaml --out runs/lite_export
--export-packed-bitnet` to write a `packed_bitlinear.pt` reference export. The
file stores 2-bit ternary codes, row scales, optional bias, and activation-bit
metadata for future serving converters; it is not a bitnet.cpp runtime file.

## Artifact Layout

When `--out-dir` is provided:

- `metrics.jsonl`: per-step metrics and run metadata.
- Optional TensorBoard events when `--tensorboard-dir` is provided. Install
  `open-alexandros[tracking]` or another compatible TensorBoard package before
  using this argument.
- `training_state.pt`: trusted local resume state for AR, diffusion, and latent
  smoke trainers, including token-data or latent-trace iterator state when
  present.
- `checkpoint/`: portable model checkpoint with `config.json`,
  `generation_config.json`, `alexandros_extra.json`, and either
  `pytorch_model.bin` or optional `model.safetensors`. Smoke trainers also
  write `checkpoint_metadata.json` with the phase, objective contract,
  trainability report, enabled/disabled module lists, cross-phase migration
  notes, and metadata stating that v1 checkpoints use HF-style files without a
  `transformers` dependency.

Portable phase handoff uses the strict migration rules documented in
[Checkpoint format](checkpoint_format.md): AR checkpoints can initialize a
diffusion model with fresh latent modules, and diffusion/latent checkpoints can
initialize a causal model while dropping latent-only parameters. Other missing
or unexpected state-dict keys remain errors.
