# Alexandros API Reference

This reference covers the public interfaces in the current PyTorch prototype.
It is intentionally concise; paper-faithful replacements and optimized kernels
remain tracked in `Roadmap.md`.

## Config Fields

`AlexandrosConfig` is a dataclass that can be round-tripped through
`to_dict()`, `from_dict()`, `save_pretrained()`, and `from_pretrained()`.

Core sizing:

- `variant`: `heavy` or `lite`.
- `vocab_size`, `hidden_size`, `intermediate_size`, `num_hidden_layers`.
- `num_attention_heads`, `max_position_embeddings`.
- `linear_attention_ratio`: number of Gated DeltaNet-style layers per attention
  layer in the reference schedule.
- `linear_mixer_backend`: `gated_deltanet` by default,
  `matrix_deltanet` for the per-head matrix-state Gated DeltaNet backend, or
  `mamba2` for the local Mamba-2/SSD-style diagonal SSM backend.
- `deltanet_chunk_size`: optional causal chunk size for
  `matrix_deltanet` prefill. `0` runs one full sequential pass.
- `deltanet_state_clip`: positive clamp bound for matrix DeltaNet state updates.
- `attention_layer_indices`: optional explicit zero-based attention layer
  indices. When set, it overrides `linear_attention_ratio`; use it for custom
  early, middle, late, or irregular attention schedules.

MoE:

- `moe_num_experts`, `moe_num_shared_experts`, `moe_top_k`.
- `moe_expert_hidden_size`.
- `moe_sparse_dispatch`: when true, the reference path runs only selected routed
  experts and scatters outputs back to token positions.
- `moe_token_state_routing`: when true, diffusion forwards add a learned
  masked/unmasked token-state router bias. Unmasked tokens use state `0`; tokens
  equal to `mask_token_id` use state `1`.
- `moe_position_routing`: when true, model forwards add a learned router bias
  based on absolute token position buckets.
- `moe_position_buckets`: number of buckets used by position-aware MoE routing.
- `router_bias_update_rate`.
- `router_bias_update_interval`: optimizer-step cadence for non-gradient router
  bias updates.
- `router_load_ema_decay`: smoothing factor for the multi-batch expert-load EMA
  used by router bias updates. The EMA is non-persistent runtime state; only the
  router bias itself is checkpointed.
- `router_logit_clip`: clamps router logits before sigmoid routing.
- `router_bias_clip`: clamps the persistent load-balancing bias after updates.

MLA and cache:

- `kv_lora_rank`: compressed latent KV rank used by `MLAAttention`.
- `mla_rope_dim`: optional separate RoPE key-cache dimension. The default `0`
  preserves the compact v1 cache; positive even values below `head_dim` enable
  paper-faithful `k_rope` cache entries.
- Derived MLA cache dimensions: `mla_d_c = kv_lora_rank`,
  `mla_d_r = mla_rope_dim`, `mla_d_nope = head_dim - mla_d_r`,
  `mla_value_head_dim = head_dim`,
  `standard_mha_elements_per_token = 2 * num_attention_heads * head_dim`, and
  `mla_elements_per_token = mla_d_c + mla_d_r`.
- `attention_backend`: `eager` is the reference matmul/softmax path. `sdpa`
  opts MLA attention layers into PyTorch
  `torch.nn.functional.scaled_dot_product_attention` while keeping the same
  masks and cache semantics. `flash` constrains SDPA to PyTorch's Flash
  Attention backend and raises clearly when CUDA/kernel support is unavailable.
- `turboquant_bits`, `use_qjl`: reference TurboQuant cache settings.
- `use_turboquant_cache`: stores AR MLA `c_kv` cache entries as TurboQuant
  packets and decompresses them on attention read. When `mla_rope_dim > 0`,
  `k_rope` stays as an uncompressed positional tensor. It is off by default.

Diffusion and latent reasoning:

- `diffusion_steps`, `mask_token_id`.
- `diffusion_objective`: `masked` by default, or `rao_blackwellized` for an
  exact target-position expansion over the sampled noised background.
- `diffusion_attention_mask_mode`: `causal` keeps diffusion forwards on the
  same left-to-right MLA mask as AR. `bidirectional` removes the causal mask
  from MLA attention layers only when `diffusion_timestep` is supplied; recurrent
  Gated DeltaNet layers remain directional. Cache reuse is rejected for
  diffusion forwards in either mode.
- `diffusion_loss_weighting`: masked-token objective weighting. `uniform`
  preserves the v1 unweighted objective; `mask_prob` and `inverse_mask_prob`
  are explicit research knobs for timestep-weighted smoke experiments.
- `diffusion_rb_chunk_size`: optional chunk size for the Rao-Blackwellized
  objective expansion. `0` evaluates all target-position variants in one batch.
- `latent_dim`, `latent_slots`.
- `latent_update_clip`: maximum per-iteration latent-refinement update norm.
  The reasoner uses bidirectional latent-slot self-attention plus a compact
  timestep-conditioned feed-forward update.
- `latent_adaptive_threshold`: optional latent-refinement halting threshold.
  `0.0` disables adaptive halting. Positive values stop the refinement loop
  after at least one iteration when the latest bounded update norm is at or
  below the threshold.

Adaptive depth and TTT:

- `prelude_layers`, `recurrent_layers`, `coda_layers`: optional
  OpenMythos-inspired staged stack counts. The default `0/0/0` keeps the
  uniform one-pass stack. When any count is nonzero, the counts must sum to
  `num_hidden_layers` and `recurrent_layers` must be positive.
- `recurrent_depth`: number of times to repeat the recurrent middle segment in
  the staged stack. Cache reuse is rejected when this is greater than `1`
  because repeated attention-cache semantics need a separate contract.
- `use_loop_index_embeddings`: when true, adds learned loop-index embeddings
  before each recurrent segment pass in non-diffusion forwards. Diffusion
  forwards disable loop-index embeddings until bidirectional denoising semantics
  are separately validated.
- `enable_adaptive_depth`, `max_depth_iters`, `act_threshold`,
  `act_ponder_cost`, `depth_lora_rank`.
- `depth_lora_ranks`: optional per-loop override for adaptive-depth low-rank
  adapter ranks. When provided, its length must match `max_depth_iters`; when
  omitted, every loop uses `depth_lora_rank`.
- `ttt_rank`.

Precision and tokens:

- `dropout`, `rms_norm_eps`, `rope_theta`.
- `bitnet_activation_bits`: Lite-only activation quantization width for
  `BitLinear`.
- `pad_token_id`, `bos_token_id`, `eos_token_id`, `mask_token_id`.
- `tie_word_embeddings`: supported for Heavy; Lite keeps the LM head as a
  `BitLinear` layer.

Lite uses `BitLinear` for linear projections and the untied LM head. Token
embeddings, normalization layers, timestep embeddings, and persistent router
biases remain full precision in the reference implementation.

## Model Classes

`AlexandrosModel`

- Embedding layer, hybrid backbone, optional adaptive-depth loop, final norm.
- Returns `AlexandrosModelOutput`.

`AlexandrosForCausalLM`

- Adds an LM head and causal loss.
- Supports greedy/sampling AR generation.
- Supports cached AR generation with MLA cache and SSM recurrent state.

`AlexandrosForDiffusionLM`

- Extends the causal model with diffusion and latent-reasoning helpers.
- Supports `diffusion_loss`, block diffusion generation, latent reasoning
  generation, and hybrid generation.

Optional Hugging Face wrappers in `alexandros.hf_compat`

- `transformers_available()`: returns whether the optional dependency is
  installed.
- `AlexandrosHFConfig`: `PretrainedConfig` wrapper with
  `from_alexandros_config(...)` and `to_alexandros_config()`.
- `AlexandrosHFModel`, `AlexandrosHFForCausalLM`, and
  `AlexandrosHFForDiffusionLM`: `PreTrainedModel` wrappers around the reference
  modules.
- `register_alexandros_with_transformers()`: registers `AutoConfig`,
  `AutoModel`, and `AutoModelForCausalLM` entries when `transformers` is
  installed. Without `.[hf]`, these symbols raise a helpful `ImportError`.

`TTTState`

- Request-local fast-weight state created with `TTTState.from_config(...)`.
- `from_fast_weights(fast_a, fast_b)` clones learned or computed fast weights
  into an isolated request-local state.
- `reset()` clears fast weights and step count for request isolation.
- `prefill_update(hidden_states, lr=..., generator=...)` updates fast weights
  from prefill hidden states without mutating model parameters. `update(...)`
  remains a backward-compatible alias.
- `apply(hidden_states, scale=..., gate=...)` mixes the adapted residual into
  hidden states. `gate` may be a scalar, `[batch, sequence]`, or
  `[batch, sequence, 1]` tensor with values in `[0, 1]`.

`TTTMetaAdapter`

- Trainable low-rank `phi_0` initialization for TTT-E2E-style adaptation,
  created with `TTTMetaAdapter.from_config(...)`.
- `initial_fast_weights(...)` returns the learned `init_a`/`init_b` tensors for
  differentiable inner-loop unrolls.
- `inner_update(hidden, input_ids, lm_head, fast_a, fast_b, inner_lr=...)`
  computes prefix next-token loss and returns updated request-local fast
  weights.
- `request_state(...)` clones learned initialization into a `TTTState` for
  inference without sharing tensors.

## Evaluation Helpers

- `summarize_parameters(model)`: total, trainable, routed expert, and active
  per-token parameter counts.
- `estimate_cache_memory(config)`: standard KV, MLA latent KV, and TurboQuant
  reference cache estimates, including the exact per-token element counts used
  for compression-ratio accounting.
- `estimate_flops(config)`: rough prefill and decode-token FLOP planning
  estimates.
- `summarize_moe_stats(model)`: load entropy, min/max expert load, diffusion
  timestep expert-selection counts, per-timestep load entropy, and noisy-step
  vs polish-step entropy splits from recent forwards.
- `causal_lm_perplexity(model, input_ids)`: tiny/local causal perplexity helper.
- `masked_diffusion_reconstruction_accuracy(model, input_ids)`: masked-token
  reconstruction accuracy for `AlexandrosForDiffusionLM`.
- `latent_reconstruction_metrics(model, input_ids)`: latent VAE/refinement
  reconstruction MSE, KL, latent norms, and update norm for audit diagnostics.
- `turboquant_reconstruction_metrics(x)`: reference cache compression error and
  compression ratio.
- `profile_model_runtime(model, input_ids)`: tiny smoke profiler for prefill
  latency, generation latency, parameter bytes, and CUDA peak memory when
  available.
- `synthetic_needle_retrieval_probe(model)`: token-level long-context diagnostic
  that reports the target token rank and probability at the query position.
- `synthetic_lost_in_middle_probe(model)`: runs early, middle, and late needle
  probes and reports relative rank/probability degradation.
- `synthetic_copy_retrieval_probe(model)`: synthetic copy-token retrieval
  diagnostic for long-context plumbing.
- `recurrent_state_drift_probe(model)`: token-by-token cached SSM diagnostic
  that reports recurrent state norms, update norms, finite-status, and how many
  layers returned recurrent state.
- `synthetic_modular_addition_probe(model)`: deterministic toy reasoning
  harness that reports the target answer-token rank and probability for a
  small modular-addition prompt. It is a mechanics check, not a quality claim.
- `adaptive_depth_toy_benchmark(model)`: runs the modular-addition probe on a
  model with `enable_adaptive_depth=True` and reports target rank/probability
  together with average ACT loop count, ponder cost, and elapsed milliseconds.
  This is a cost/diagnostic surface, not a reasoning-quality claim.
- `code_benchmark_plan(...)`, `ensure_code_benchmark_ready(...)`,
  `HumanEvalStyleTask`, and `humaneval_style_harness_report(...)`: non-executing
  code-benchmark readiness helpers for HumanEval-style, SWE-Bench, and
  Terminal-Bench planning. They validate metadata and missing prerequisites but
  do not execute generated code or compute benchmark scores.

`scripts/benchmark.py` wraps these helpers for local config comparisons. It
reports AR tokens/sec, block-diffusion tokens/sec equivalent, parameter bytes,
estimated hidden-activation bytes, tiny causal/diffusion loss stability,
active parameters, cache-memory estimates, TurboQuant reconstruction error,
TurboQuant cache round-trip latency, dense-QR rotation timing compared with a
structured sign/permutation candidate, and reference FLOP estimates as JSONL
and Markdown.
See [Hardware targets](hardware_targets.md) for supported validation paths and
required hardware/precision reporting before publishing benchmark claims.
See [Evaluation strategy](evaluation_strategy.md) for ablation axes, external
baseline caveats, and cross-tokenizer reporting rules.
See [Optimized kernels](optimized_kernels.md),
[Serving integration](serving_integration.md), and
[Distributed MoE planning](distributed_moe.md) for future acceleration and
deployment contracts.

`scripts/eval.py --compare-config <path>` appends additional config runs to the
JSONL output, records a smoke comparison across autoregressive, block
diffusion, latent reasoning, and hybrid generation modes, logs synthetic
long-context diagnostics, and adds a Markdown baseline comparison table for
Heavy/Lite or other tiny local comparisons.

## CLI Runtime Helpers

Scripts use the shared `scripts/_common.py::make_arg_parser(...)` factory so
help output includes default values consistently. Smoke training scripts also
share runtime and artifact controls:

- `--device`, `--dtype`, `--seed`, and `--deterministic`.
- `--dtype` uses FP32 master weights plus autocast in AR, diffusion, and latent
  trainers; CUDA float16 also saves/restores `GradScaler` state.
- `--grad-accum-steps` for tiny gradient accumulation tests.
- `--val-every` and `--val-batches` for tiny deterministic validation passes
  in AR, diffusion, and latent smoke trainers.
- `--out-dir` for a local artifact directory.
- `--resume` to continue from a trusted local `training_state.pt` written by
  a smoke trainer with the same config hash.
- `--tensorboard-dir` for optional TensorBoard scalar logging. TensorBoard
  remains an optional dependency and is required only when this argument is
  used.
- `--token-ids-jsonl`, `--token-field`, `--validation-fraction`, and
  `--no-shuffle-data` for pre-tokenized JSONL training data.
- `--trace-jsonl`, `--trace-len`, and trace field-name overrides on
  `train_latent.py` for user-supplied visible reasoning trace token IDs.
- `--prefill-chunk-len` on `meta_train_ttt.py` for request-local TTT prefill
  update chunks.
- `--trainable-scope` for phase-aware trainability controls:
  `phase_default`, `all`, `backbone_only`, `head_only`, `latent_only`, or
  `none`.
- `--log-jsonl` for explicit metric log placement. If omitted while
  `--out-dir` is set, trainers write `metrics.jsonl` in that directory.

JSONL records include config hash, git commit when available, Python/Torch
versions, platform, device, dtype, seed, phase, step, and task-specific metrics.
They also include `objective_*` fields from
`alexandros.training.objective_contract(phase)`, including objective name,
inputs, targets, normalization, ignore-index behavior, and smoke stopping
criteria.
Training scripts include `trainability_*` fields from the applied trainability
report, including scope and trainable/frozen parameter counts.
When validation is enabled, the same step record includes `val_loss` and
phase-specific validation norms.
When `--out-dir` is set, AR, diffusion, and latent smoke trainers also write a
versioned `training_state.pt` containing model state, optimizer state, RNG
state, optional token/trace-data iterator state, optional GradScaler state, and
run metadata, plus a `checkpoint/` directory in the reference model format.
The TTT smoke script writes `pre_update_loss`, `post_update_loss`, fast-weight
step count, and hidden-state norm diagnostics.
Saved model checkpoints include `generation_config.json` with conservative AR
generation defaults and special token IDs.
`save_pretrained(..., safe_serialization=True)` writes `model.safetensors` when
the optional `safetensors` package is installed; otherwise the default remains
`pytorch_model.bin`.
`alexandros_extra.json` records the current Hugging Face compatibility decision:
version `1` checkpoints use HF-style filenames and generation metadata but do
not require `transformers`, do not claim AutoClass registration, and do not
require `trust_remote_code`.
`save_pretrained(..., tokenizer_metadata={...})` writes optional
`tokenizer_metadata.json` after validating any declared `vocab_size` and special
token IDs against `AlexandrosConfig`. `load_tokenizer_metadata(checkpoint_dir)`
returns the user-supplied metadata object or `None`; it does not instantiate a
tokenizer dependency.
`save_pretrained(..., checkpoint_metadata={...})` writes optional
`checkpoint_metadata.json`; `load_checkpoint_metadata(checkpoint_dir)` returns
the inner checkpoint metadata object or `None`.

`alexandros.training` exposes lightweight data helpers:

- `objective_contract(phase)` and `objective_log_fields(phase)` for the AR,
  diffusion, latent, TTT, and optional distillation objective contracts.
- `standard_metric_names(phase)` and `validate_standard_metric_record(...)` for
  required JSONL metric fields across AR, diffusion, latent, and TTT smoke
  trainers.
- `apply_trainability(model, phase=..., scope=...)` and
  `trainable_parameters(model)` for phase-aware optimizer filtering.
- `PackedTokenDataset.from_jsonl(...)` for token-id JSONL packing and splitting.
- `PackedTokenBatchIterator` with `state_dict()` and `load_state_dict()`.
- `iter_latent_trace_jsonl(...)` and `LatentTraceDataset.from_jsonl(...)` for
  user-supplied visible reasoning trace token IDs. See
  [`docs/latent_trace_format.md`](latent_trace_format.md).
- `iter_distillation_jsonl(...)` and `DistillationDataset.from_jsonl(...)` for
  user-supplied teacher token targets and/or dense teacher logits. These helpers
  validate mechanism shape and token contracts only; teacher selection and
  license compliance remain downstream responsibilities.
- `distillation_loss(...)` for masked teacher-token cross entropy and
  temperature-scaled teacher-logit KL.
- `summarize_latent_trace_record(...)` for tokenizer-agnostic decoded summaries
  of latent trace records during research debugging.
- `make_modular_addition_latent_trace_records(...)` and
  `make_boolean_xor_latent_trace_records(...)` for tiny semantic trace fixtures
  used by CI/mechanism tests only.
- `make_diffusion_training_batch(...)` for absorbing-mask corruption, labels,
  timesteps, and masked-token loss accounting.
- `make_block_diffusion_training_batch(...)` for contiguous masked-block
  targets used by block diffusion experiments.

## Model Outputs

`AlexandrosModelOutput`

- `last_hidden_state`: `[batch, sequence, hidden_size]`.
- `past_key_values`: per-layer attention cache entries or `None`.
- `past_ssm_states`: per-layer recurrent states or `None`.
  `GatedDeltaNetBlock` and `Mamba2Block` states have shape
  `[batch, hidden_size]`; `MatrixGatedDeltaNetBlock` states have shape
  `[batch, heads, value_dim, key_dim]`. States match the mixer hidden-state
  dtype/device and are detached before reuse.
- `hidden_states`: optional tuple of intermediate hidden states.
- `moe_stats`: per-layer routing stats.
- `halting`: optional adaptive-depth halt probabilities.

When adaptive depth is enabled, `model.adaptive_depth.last_stats` records the
latest ACT diagnostics: `halting_sum`, `remainder`, `n_updates`,
`ponder_cost`, and `average_loop_count`.
`model.last_stack_stats` records whether the staged recurrent stack ran, how
many recurrent loops were executed, how many layer executions occurred, and
whether loop-index embeddings were applied.
`model.latent_reasoner.last_stats` records the latest latent refinement
diagnostics: requested steps, steps run, whether adaptive halting fired, the
last update norm, and the configured adaptive threshold.

`AlexandrosCausalLMOutput`

- `loss`: optional scalar loss.
- `logits`: `[batch, sequence, vocab_size]`.
- `past_key_values`, `past_ssm_states`, `hidden_states`, `moe_stats`.

## Generation Modes

`GenerationMode.AUTOREGRESSIVE`

- Left-to-right decode with optional cache reuse.
- Sampling controls: `do_sample`, `temperature`, `top_k`, `top_p`,
  `repetition_penalty`.
- Stop controls: `eos_token_id`, `stop_token_ids`, and `stop_sequences`.
- `pad_token_id` is suppressed during token selection; padding is reserved for
  prompt masks and rectangular batched outputs.
- Right-padded mixed-length prompt batches are supported by a reference
  fallback that compacts each row, generates independently, and pads returned
  rows to a rectangle. Left padding and interior padding remain rejected.

`GenerationMode.BLOCK_DIFFUSION`

- Appends a masked block and iteratively fills mask tokens.
- Only the appended block is denoised; prompt tokens are preserved even when a
  prompt token ID equals `mask_token_id`.
- Uses `diffusion_attention_mask_mode` for MLA layers during denoising. The
  default is causal; `bidirectional` is useful for all-attention denoising
  experiments, while hybrid stacks still include directional Gated DeltaNet
  layers.
- Cache reuse is rejected because block denoising is not a left-to-right cache
  problem.
- Supports `do_sample`, `temperature`, `top_k`, and `top_p` for masked-token
  selection. `pad_token_id` and `mask_token_id` are suppressed for committed
  block tokens.
- Supports `steps`, optional `block_size` chunking, `confidence_schedule`
  (`median`, `linear`, or `all`), and `remask_low_confidence`.
- Supports `eos_token_id`, `stop_token_ids`, and `stop_sequences` as a
  post-processing step. Batched outputs remain rectangular; rows that stop
  early are padded after the stop point and common trailing all-pad columns are
  trimmed.
- Streaming is not implemented for block diffusion because token commitments can
  occur out of left-to-right order.

`GenerationMode.LATENT_REASONING`

- Refines latent thought slots before decoding the first continuation token,
  then falls back to AR generation.

`GenerationMode.HYBRID`

- Runs one latent-reasoning step, then block diffusion for the remaining
  requested tokens.
- Streaming is not implemented for hybrid generation for the same reason as
  block diffusion.

## Cache Packet Formats

MLA cache entries are dictionaries:

- `{"c_kv": tensor}` where `c_kv` has shape
  `[batch, cached_sequence, kv_lora_rank]`.
- With `use_turboquant_cache=True`, attention cache entries use
  `{"c_kv_packet": TurboQuantPacket}` instead.
- With `mla_rope_dim > 0`, cache entries also include `{"k_rope": tensor}` with
  shape `[batch, cached_sequence, mla_rope_dim]`.

SSM cache entries are tensors:

- Shape `[batch, hidden_size]` for elementwise Gated DeltaNet and Mamba-2
  layers.
- Shape `[batch, heads, value_dim, key_dim]` for matrix DeltaNet layers.

TurboQuant packets are represented by `TurboQuantPacket`:

- `q`: quantized int8 values.
- `scale`: per-vector scalar quantization scale.
- `bits`: quantization bit width.
- `original_dtype`: dtype restored by `decompress`.
- `rotation_seed`: seed used to reconstruct the deterministic shared rotation.
- `qjl_sign`: optional residual sign sketch when `use_qjl=True`.
- `packet_format_version`: currently `1`; newer versions are rejected until a
  migration path exists.
- `qjl_projection_seed`: optional residual-sketch projection seed when
  `use_qjl=True`.
- `qjl_residual_norm`: optional per-vector residual norm with the same shape as
  `scale` when `use_qjl=True`.

`TurboQuantKVCache.estimate_attention_scores(query, packet, use_qjl=...)`
estimates `[batch, query_length, key_length]` dot-product scores directly from
the quantized rotated key packet. When `use_qjl=True`, it adds the QJL residual
sign/norm sketch before scoring. This is a reference cache-level estimator; the
current MLA attention path still decompresses `c_kv` before applying learned
key/value projections.

Lite packed BitLinear exports are written by
`scripts/export_hf.py --export-packed-bitnet` to `packed_bitlinear.pt`:

- Top-level format: `alexandros-packed-bitlinear`, version `1`.
- Per-layer format: `alexandros-packed-bitlinear-layer`, version `1`.
- `encoding`: `2bit_ternary_0_zero_1_pos_2_neg`.
- `packed_weight`: uint8 tensor storing four ternary codes per byte.
- `weight_shape`: original `[out_features, in_features]` shape.
- `padding`: unused packed codes in the final byte.
- `scale`: FP32 absmean scale per output row.
- `bias`: optional FP32 bias.
- `activation_bits`: activation quantization width used by the source layer.

This is a reference interchange artifact, not a bitnet.cpp ABI guarantee.

`alexandros.inference` exposes cache helpers for future beam-search decoders:

- `GenerationRequest`: serializable, validated request object for CLI/server
  wrappers.
- `generate_from_request(model, request, device=None)`: validates dict
  payloads as `GenerationRequest`, builds the input tensor, and calls
  `model.generate(...)`.
- `reorder_past_key_values(past_key_values, beam_idx)`.
- `reorder_past_ssm_states(past_ssm_states, beam_idx)`.
- `reorder_generation_cache(...)`.
