# Alexandros Roadmap

This file is the task tracker for all known outstanding work. Anything marked
partial, reference-only, stubbed, or unvalidated should remain here until it is
completed and tested.

## Task Completion Standard

A roadmap task is complete only when a new contributor can verify it without
guesswork:

- [ ] The implementation is present in code, not only described in docs.
- [ ] The relevant design/research note is linked or summarized locally.
- [ ] Unit tests cover normal behavior and at least one failure/edge case.
- [ ] Integration or smoke tests prove the feature works with a tiny config.
- [ ] Public APIs, config fields, and scripts are documented when changed.
- [ ] Any approximation is either removed or explicitly kept as a documented
  fallback.

## Environment And Validation

- [x] Static Python compilation passes with `python -m compileall`.
- [x] Install project dependencies in a clean environment.
  - [x] Confirm `torch` import works.
  - [x] Confirm `pytest` import works.
  - [x] Validate a local Python 3.11 CPU-only `.venv`.
  - [x] Decide the default Torch install story; direct PyPI Torch may pull a
    large CUDA stack, while CPU validation used the PyTorch CPU index.
  - [x] Run `pip install -e .[dev]` successfully with the chosen default index.
- [x] Run the existing smoke test suite.
  - [x] Run `python -m pytest`.
  - [x] Fix runtime failures in `tests/test_smoke.py`.
  - [x] Add regression tests for bugs found during first PyTorch run.
- [x] Add continuous integration.
  - [x] Test Python 3.10, 3.11, and the local target Python version.
  - [x] Run compile, unit tests, and CLI smoke commands.
  - [x] Cache dependencies without committing generated artifacts.

## Research Documentation

- [x] Create `docs/research_matrix.md` with source URLs and implementation
  translations.
- [x] Create `docs/alexandros_master_design.md` with architecture and public
  APIs.
- [x] Expand each research entry into implementation-grade notes.
  - [x] Add equations or pseudocode for DeepSeekMoE normalized sigmoid routing
    and auxiliary-loss-free bias updates.
  - [x] Add equations or pseudocode for Gated DeltaNet/Mamba-2 state updates.
  - [x] Add MLA tensor-shape details, including RoPE/no-RoPE channel split.
  - [x] Add MDLM/LLaDA masking objective and sampling schedule details.
  - [x] Add diffusion-MoE timestep-aware routing and diagnostics notes.
  - [x] Add latent-reasoning VAE/refinement/audit notes.
  - [x] Add BitNet ternary projection, activation quantization, and packed
    export notes.
  - [x] Add TTT-E2E training-time and inference-time algorithms.
  - [x] Add TurboQuant rotation, scalar quantization, and QJL details.
  - [x] Add OpenMythos recurrent-depth, ACT, and depth-wise LoRA notes.
- [x] Add architecture decision records.
  - [x] Explain why v1 uses reference implementations.
  - [x] Record which paper-faithful behavior is deferred.
  - [x] Record acceptance criteria for replacing each reference module.

## Packaging And Public API

- [x] Add `pyproject.toml`.
- [x] Add importable `alexandros` package.
- [x] Add `AlexandrosConfig`, `AlexandrosModel`, `AlexandrosForCausalLM`,
  `AlexandrosForDiffusionLM`, and `GenerationMode`.
- [x] Harden package metadata.
  - [x] Add license metadata.
  - [x] Add authors/maintainers.
  - [x] Add package classifiers.
  - [x] Decide whether the package name should be `open-alexandros` or
    `alexandros`.
- [x] Improve Hugging Face compatibility.
  - [x] Decide whether to depend on `transformers`.
    V1 keeps `transformers` out of the dependency set, writes HF-style
    checkpoint filenames, records the decision in `alexandros_extra.json`, and
    offers `transformers` through the optional `.[hf]` extra only.
  - [x] Add optional `PretrainedConfig`/`PreTrainedModel` subclasses.
  - [x] Add `AutoConfig`, `AutoModel`, and `AutoModelForCausalLM` registration
    path.
  - [x] Add `generation_config.json` support.
- [x] Improve checkpoint format.
  - [x] Add `safetensors` support.
  - [x] Preserve backward compatibility with `pytorch_model.bin`.
  - [x] Save user-supplied tokenizer metadata when available.
  - [x] Add checkpoint versioning metadata.
  - [x] Add migration notes.

## Configs And Model Sizing

- [x] Add Heavy/Lite tiny configs.
- [x] Add Heavy/Lite base configs.
- [x] Add config validation beyond shape checks.
  - [x] Validate special token IDs are inside vocabulary.
  - [x] Validate `mask_token_id`, `bos_token_id`, `eos_token_id`, and
    `pad_token_id` are distinct where required.
  - [x] Validate diffusion and latent settings are positive.
  - [x] Validate TTT and adaptive-depth ranks are sensible for hidden size.
  - [x] Validate TurboQuant bit range.
  - [x] Validate router clipping and BitLinear activation-bit settings.
  - [x] Reject boolean or non-integer values for integer config fields.
- [x] Add scale planning tools.
  - [x] Parameter count estimator.
  - [x] Active parameter count estimator for MoE.
  - [x] KV cache memory estimator for MLA and TurboQuant.
  - [x] FLOP estimate for Heavy vs Lite tiny/base/large profiles.
  - [x] Reject invalid estimator dimensions before producing nonsensical
    memory/FLOP reports.
- [x] Add more configs.
  - [x] `heavy_debug.yaml` for CPU-only testing.
  - [x] `lite_debug.yaml` for CPU-only testing.
  - [x] `heavy_1b.yaml` research-scale target.
  - [x] `lite_1b.yaml` research-scale target.

## Architecture Compatibility Contracts

- [x] Define which architectural features are compatible in each mode.
  - [x] AR mode: causal masks, cache reuse, SSM recurrent state, optional TTT.
  - [x] Block diffusion mode: bidirectional or block-local denoising masks,
    timestep routing, no assumption of left-to-right cache reuse.
  - [x] Latent reasoning mode: latent slots/refinement before token decode.
  - [x] Hybrid mode: explicit order of latent refinement, block diffusion
    continuation, and cache/state handling.
- [x] Resolve causal-vs-bidirectional backbone policy.
  - [x] Decide whether diffusion uses the same causal backbone, a bidirectional
    attention mask, or a separate denoising head/backbone.
  - [x] Add config fields for `attention_mask_mode` or equivalent.
  - [x] Test AR logits remain causal while diffusion can denoise masked spans.
  - [x] Reject diffusion cache reuse at runtime.
  - [x] Document that `bidirectional` applies to MLA attention layers only;
    recurrent mixer layers remain directional in the reference hybrid backbone.
- [x] Define TTT boundaries.
  - [x] Specify that TTT applies only to causal/SSM paths unless explicitly
    extended.
  - [x] Prevent request-local TTT state from being used in incompatible
    diffusion training paths.
  - [x] Prevent request-local TTT state from being used in incompatible
    diffusion forward paths.
  - [x] Add runtime assertions or clear warnings for unsupported generation
    combinations.
- [x] Define cache semantics by mode.
  - [x] MLA cache for AR decode.
  - [x] SSM state cache for recurrent decode.
  - [x] TurboQuant cache only where cache reuse is mathematically valid.
  - [x] Reject cached generation in diffusion/latent modes until proven safe.

## Initialization, Numerical Stability, And Precision

- [x] Define initialization policy.
  - [x] Embeddings and LM head initialization.
  - [x] Attention and MLA projection scaling.
  - [x] MoE expert and router initialization.
  - [x] BitLinear shadow-weight initialization.
  - [x] Latent VAE and diffusion timestep embedding initialization.
- [x] Define dtype policy.
  - [x] FP32 master weights vs BF16/FP16 compute for Heavy.
  - [x] Full-precision modules in Lite, such as embeddings/norms/router biases.
  - [x] Autocast behavior in training scripts.
  - [x] CPU fallback behavior.
- [x] Add stability guards.
  - [x] Gradient clipping.
  - [x] Activation/gradient norm logging.
  - [x] NaN/Inf loss checks in smoke training scripts.
  - [x] Router logit clipping or bias clipping.
  - [x] Latent refinement norm controls.
- [x] Add numerical regression tests.
  - [x] Forward/backward finite checks for Heavy and Lite.
  - [x] Mixed precision smoke tests when CUDA is available.
  - [x] Long sequence stability tests for recurrent state norms.

## Training Interfaces, Tokenization, And Sample Data

Scope: Alexandros should publish trainable model code and industry-standard
training interfaces. This repo will not gather or curate an internet-scale
pretraining corpus. Tokenizers, corpora, and full training recipes are expected
to be supplied by downstream trainers or the AI community.

- [x] Make tokenizer integration pluggable rather than project-owned.
  - [x] Define the tokenizer contract: `vocab_size`, `pad/bos/eos/mask`
    token IDs, encode/decode shape expectations, and save/load metadata.
  - [x] Support loading user-provided tokenizer metadata into checkpoints when
    available.
  - [x] Document how to use SentencePiece, tiktoken-style BPE, or Hugging Face
    tokenizers without choosing one as the Alexandros default.
  - [x] Add a tiny toy tokenizer or fixed token-id fixture for tests only.
- [x] Build text dataset pipeline.
  - [x] Add streaming dataset loader.
  - [x] Add token packing and sequence chunking.
  - [x] Add deterministic shuffling and seed control.
  - [x] Add train/validation split handling.
- [x] Build diffusion dataset transforms.
  - [x] Add absorbing-mask corruption with reproducible timesteps.
  - [x] Add block diffusion targets.
  - [x] Add masked-position loss accounting.
- [x] Add sample data fixtures only for correctness tests.
  - [x] Use tiny synthetic or permissively licensed open sample text snippets.
  - [x] Keep fixtures small enough for CI and exclude any claim of model
    quality.
  - [x] Record source, license, checksum, and allowed use for any bundled
    sample.
  - [x] Add dataset-card notes for bundled sample fixtures only.
- [x] Define optional latent trace input interface without bundling trace data.
  - [x] Specify accepted visible reasoning trace format.
  - [x] Add synthetic trace-to-latent examples for CI only.
  - [x] Document privacy/audit caveats for user-supplied traces.

## Training Mechanism Contracts And Phase Handoff

Scope: prove that the architecture can be trained, resumed, evaluated, and
exported. Do not attempt to define or own a full data curriculum.

- [x] Define objective contracts for each trainer.
  - [x] AR next-token loss inputs, labels, ignore-index behavior, and stopping
    criteria for smoke runs.
  - [x] Diffusion masked-token objective inputs, timestep schedule, and loss
    normalization.
  - [x] Latent VAE/refinement losses for synthetic or user-supplied hidden/trace
    examples.
  - [x] TTT prefill/update objective for synthetic or user-supplied long-context
    examples.
- [x] Define module trainability and freeze/unfreeze controls.
  - [x] Which modules can train during AR mechanism tests.
  - [x] Which modules can train during diffusion mechanism tests.
  - [x] Which modules can train during latent mechanism tests.
  - [x] Which modules can train during TTT mechanism tests.
- [x] Define checkpoint handoff contracts.
  - [x] Required checkpoint fields after each phase.
  - [x] Versioned metadata for enabled/disabled modules.
  - [x] Migration path when adding new modules to old checkpoints.
- [x] Add phase-handoff smoke tests.
  - [x] One-step AR to diffusion handoff.
  - [x] One-step diffusion to latent handoff.
  - [x] One-step latent to TTT handoff.

## Core Backbone

- [x] Implement `AlexandrosBlock` with RMSNorm, sequence mixer, residuals, and
  MoE FFN.
- [x] Implement configurable hybrid schedule using `linear_attention_ratio`.
- [x] Make layer scheduling fully configurable.
  - [x] Allow explicit attention layer indices.
  - [x] Allow 3:1, 4:1, early/middle/late, and custom schedules.
  - [x] Test schedule serialization and layer construction.
- [x] Add real recurrent state caching.
  - [x] Return and reuse `GatedDeltaNetBlock` states during generation.
  - [x] Distinguish attention cache and SSM state cache in public outputs.
  - [x] Add cache reorder support for future beam search.
- [x] Add dropout policy.
  - [x] Apply config dropout in attention, MoE, and residual paths.
  - [x] Ensure deterministic eval behavior.

## Gated DeltaNet / Mamba-Style Mixer

- [x] Add runnable `GatedDeltaNetBlock` reference recurrence.
- [x] Add a paper-faithful matrix-state Gated DeltaNet option.
  - [x] Use the [research matrix](docs/research_matrix.md) Gated DeltaNet
    recurrence: per-head matrix state, `q/k/v` projections, normalized keys,
    write-strength `beta_t`, retention gate `alpha_t`, delta residual
    `v_t - S_{t-1} k_t`, and bounded state updates.
  - [x] Define and document exact recurrent state tensor shape, dtype/device, and
    cache semantics. The elementwise and Mamba-2 reference states are
    `[batch, hidden_size]`; `MatrixGatedDeltaNetBlock` uses
    `[batch, heads, value_dim, key_dim]`.
  - [x] Implement matrix-valued DeltaNet/Gated DeltaNet update.
  - [x] Add causal chunked training path through `deltanet_chunk_size`.
  - [x] Implement a reference chunked prefill recurrence for long prefill.
    Optimized associative/SSD scan kernels remain future performance work.
  - [x] Add recurrent decode path through `decode_step(...)` and model cache
    reuse.
  - [x] Verify chunked and recurrent decode outputs match on small sequences.
- [x] Add optional Mamba-2 backend.
  - [x] Decide dependency strategy for `mamba-ssm` or a local minimal kernel:
    v1 uses a local PyTorch diagonal SSM reference backend; optimized native
    `mamba-ssm` integration remains future work.
  - [x] Implement the research-matrix backend contract:
    `forward(x, state, attention_mask) -> (y, next_state)` with an
    `s_t = A_t * s_{t-1} + B_t * x_t` style state update.
  - [x] Add config switch between `gated_deltanet`, `matrix_deltanet`, and
    `mamba2`.
  - [x] Add tests that all mixers satisfy their documented shape/cache contract.
- [x] Add long-context diagnostics.
  - [x] Synthetic copy/retrieval tests.
  - [x] Lost-in-the-middle style tests.
  - [x] State norm, update norm, and drift monitoring.

## MLA Attention

- [x] Add reference `MLAAttention` with compressed `c_kv` cache.
- [x] Implement paper-faithful MLA projection structure.
  - [x] Add config fields or derived dimensions for `d_c`, `d_nope`, `d_R`,
    and value-head dimension, with validation that concatenated query/key
    dimensions match attention scoring requirements.
    `mla_rope_dim` enables a separate `d_R`; the default remains `0` for
    backward-compatible tiny configs.
  - [x] Split query/key channels into RoPE and no-RoPE components.
  - [x] Store compressed KV latent plus separate positional key component.
  - [x] Cache exactly `cache.c_kv: [B, total_T, d_c]` and
    `cache.k_rope: [B, total_T, d_R]`.
  - [x] Reconstruct no-RoPE K/V from latent.
  - [x] Apply RoPE only to `q_rope` and `k_rope`; keep `c_kv` position-free.
  - [x] Ensure TurboQuant, when enabled, compresses `c_kv` only until a
    separate positional-cache compressor is specified.
  - [x] Add tests for cache-size formulas.
    - [x] Verify `standard_mha_elements_per_token = 2 * heads * head_dim`.
    - [x] Verify `mla_elements_per_token = d_c + d_R`.
    - [x] Verify compression ratio is computed from those two quantities.
  - [x] Add cached-vs-uncached logit tests for the paper-faithful cache layout.
- [x] Use attention cache in generation.
  - [x] Update AR generation to pass `past_key_values`.
  - [x] Extend attention masks correctly when cache is present.
  - [x] Test cached generation equals uncached generation for greedy decode.
- [x] Add optimized attention path.
  - [x] Evaluate `torch.nn.functional.scaled_dot_product_attention`.
  - [x] Add optional SDPA path with eager-equivalence tests.
  - [x] Add optional Flash Attention path.
  - [x] Keep reference path for CPU tests.

## Mixture Of Experts

- [x] Add shared+routed `MoEFeedForward`.
- [x] Add normalized sigmoid top-k routing.
- [x] Add router bias update method.
- [x] Add diffusion timestep router bias.
- [x] Make MoE sparse instead of dense-over-all-experts.
  - [x] Gather tokens by expert.
  - [x] Run only selected experts.
  - [x] Scatter outputs back to token positions.
  - [x] Preserve shared expert behavior.
  - [x] Add correctness test comparing dense reference and sparse path.
- [x] Improve auxiliary-loss-free load balancing.
  - [x] Track load over multiple batches.
  - [x] Update router bias during training loop.
  - [x] Add config for update cadence.
  - [x] Add config for router logit and bias clipping.
  - [x] Add monitoring metrics for expert load entropy.
- [x] Add research-matrix MoE compliance tests.
  - [x] Prove load-balancing bias affects top-k expert selection only, not
    normalized routed-expert mixture weights.
  - [x] Prove routed weights are normalized sigmoid probabilities over selected
    experts, not softmax logits.
  - [x] Prove shared experts remain always-on and are combined separately from
    routed expert weights.
- [x] Add distributed MoE planning.
  - [x] Define expert parallelism interface.
  - [x] Specify dispatch tensor contract as
    `[num_selected_tokens_for_expert, hidden_size]` per expert.
  - [x] Define all-to-all dispatch and combine tensor shapes.
  - [x] Define weighted scatter/combine semantics back to
    `[batch, sequence, hidden_size]`.
  - [x] Document all-to-all requirements in `docs/distributed_moe.md`.
  - [x] Keep non-distributed dense/sparse fallback tests.

## Lite BitNet Path

- [x] Add `BitLinear` reference QAT layer.
- [x] Route Lite model linears through `BitLinear`.
- [x] Audit all linear projections.
  - [x] Confirm every intended linear uses `make_linear`.
  - [x] Decide whether embeddings and LM head should remain full precision.
  - [x] Document any intentionally full-precision modules.
- [x] Improve BitNet training recipe.
  - [x] Add Lite-specific optimizer defaults.
  - [x] Add warmup and gradient clipping guidance.
  - [x] Add NaN/Inf guards in training scripts.
  - [x] Add activation quantization bit-width config.
  - [x] Add research-matrix compliance checks for row-wise absmean weight
    scaling, ternary thresholding, and per-token absmax activation scaling.
  - [x] Add explicit STE gradient test showing gradients reach the
    full-precision shadow weight.
  - [x] Document modules that must remain full precision: embeddings, norms,
    scalar router/load buffers, and non-linear state buffers.
  - [x] Calibrate Lite optimizer and warmup defaults from actual training runs.
    - [x] Run Heavy/Lite tiny stability comparisons with the benchmark script.
    - [x] Record recommended LR, warmup, and clipping settings in the training
      guide.
- [x] Add packed ternary export.
  - [x] Define packed weight format.
  - [x] Export ternary signs and scales.
  - [x] Add compatibility notes for bitnet.cpp-style serving.
- [x] Benchmark Lite vs Heavy.
  - [x] Parameter memory.
  - [x] Activation memory.
  - [x] CPU latency.
  - [x] Tiny-model loss stability.

## Diffusion Language Modeling

- [x] Add `MaskedDiffusionScheduler`.
- [x] Add masked-position diffusion loss.
- [x] Add greedy block diffusion generation.
- [x] Implement a stronger MDLM/LLaDA objective.
  - [x] Add timestep sampling distribution.
  - [x] Choose and document the weighting schedule `w(t)` and default masked
    objective simplification.
  - [x] Add optional weighted masked objective for explicit research schedules.
  - [x] Add optional Rao-Blackwellized objective with tests that compare it
    against the current masked-token normalized objective.
  - [x] Add loss normalization by masked-token count for the default objective
    and non-pad target count for the Rao-Blackwellized objective.
  - [x] Add tests for all-mask and no-mask edge cases.
- [x] Add fully bidirectional or block-local denoising mixer support if needed.
  V1 keeps bidirectional denoising on MLA attention layers and documents that
  Gated DeltaNet remains directional; all-attention configs can exercise
  bidirectional masked-span denoising.
  - [x] Decide whether to add a bidirectional Gated DeltaNet variant,
    block-local mixer, or separate denoising backbone.
  - [x] Test masked-span logits can depend on both left and right context across
    every mixer type used in diffusion mode.
  - [x] Document cache and recurrent-state semantics for the chosen denoising
    mixer.
- [x] Improve block diffusion sampler.
  - [x] Add remasking policy.
    - [x] Keep high-confidence positions and remask low-confidence generated
      block positions after each denoising step.
  - [x] Add temperature/top-k/top-p controls.
  - [x] Add confidence schedules.
    - [x] Implement a schedule function that maps denoising step and
      confidence distribution to a commit threshold.
  - [x] Add variable block size.
  - [x] Compare generation modes in smoke eval.
- [x] Integrate diffusion with generation API.
  - [x] Add generation config.
  - [x] Add stop token handling.
  - [x] Add streaming strategy or document why it is unsupported.

## Diffusion Plus MoE

- [x] Add timestep-aware router bias embedding.
- [x] Track expert usage by diffusion timestep.
  - [x] Store per-timestep load histograms.
  - [x] Emit metrics from eval reporting.
  - [x] Emit per-timestep expert metrics from diffusion training loop.
  - [x] Add tests that timestep routing changes selection.
- [x] Add time-aware routing options from diffusion MoE papers.
  - [x] Per-layer timestep conditioning.
  - [x] Position-aware routing.
  - [x] Add optional masked/unmasked token-state embedding to router features.
  - [x] Noisy-step vs polish-step expert diagnostics.
    - [x] Report per-step expert-load entropy.
    - [x] Split metrics into high-mask-probability noisy steps and low-mask
      polish steps.
  - [x] Keep load-balancing bias non-gradient even when timestep/token-state/
    position router features are enabled.

## Latent Reasoning

- [x] Add `LatentThoughtVAE`.
- [x] Add `LatentDiffusionReasoner`.
- [x] Add latent reasoning generation path.
- [x] Replace the simple MLP reasoner with a stronger latent transformer.
  - [x] Preserve the research-matrix latent contract:
    `hidden -> mu/logvar -> sampled slots -> refinement -> hidden decode`.
  - [x] Add attention or learned pooling before `mu/logvar`, or explicitly
    retain mean pooling as the documented v1 simplification.
  - [x] Add bidirectional latent attention over latent slots.
  - [x] Add timestep/noise conditioning.
  - [x] Add self-correction or adaptive compute loop.
  - [x] Test that latent steps produce stable, non-divergent updates.
- [x] Support user-supplied latent reasoning examples.
  - [x] Implement trace ingestion plumbing against the documented optional trace
    format.
  - [x] Add synthetic/toy trace fixtures for CI only.
  - [x] Add reconstruction or latent-target loss with configurable `lambda_kl`
    and `lambda_rec`.
  - [x] Add math/logical toy tasks that validate mechanics, not final model
    quality.
- [x] Improve auditability.
  - [x] Add latent reconstruction diagnostics.
  - [x] Add latent reconstruction metrics to eval reports.
  - [x] Add optional decoded summaries for research debugging.
  - [x] Document limitations of hidden reasoning.

## Adaptive Depth / OpenMythos-Inspired Loop

- [x] Add optional `AdaptiveDepthController`.
- [x] Make ACT halting more faithful.
  - [x] Add ponder cost.
  - [x] Add remainder trick tests.
  - [x] Implement ACT bookkeeping from the research matrix:
    `halting_sum`, `remainder`, `n_updates`, and weighted accumulated output.
  - [x] Add token-wise early-exit masking.
  - [x] Report average loop count.
- [x] Add prelude/recurrent/coda architecture option.
  - [x] Keep current uniform stack as default.
  - [x] Add config fields for prelude/recurrent/coda layer counts.
  - [x] Decide how recurrent loops interact with diffusion and latent modules.
  - [x] Define loop-index embedding mode behavior: shared by non-diffusion
    forwards, including latent/hybrid prompt encoding, and disabled in
    diffusion forwards until denoising semantics are tested.
- [x] Add depth-wise LoRA controls.
  - [x] Make LoRA rank configurable per loop.
  - [x] Implement per-loop low-rank adapters `delta_W_k = A_k @ B_k` on the
    adaptive-depth transition projection.
  - [x] Decide whether to extend depth-wise LoRA to staged backbone recurrent
    projections after those projection ownership boundaries are explicit.
  - [x] Add loop-index embeddings tests.
  - [x] Benchmark adaptive-depth cost vs quality on toy tasks.

## TTT-E2E-Inspired Long-Context Adaptation

- [x] Add request-local `TTTState`.
- [x] Ensure tests are planned for no base-weight mutation.
- [x] Replace the current sketch with a trained update rule.
  - [x] Define fast-weight parameters and update objective.
    - [x] Specify which parameters are updateable fast weights and which remain
      frozen checkpoint weights.
    - [x] Add `TTTMetaAdapter.init_a` and `init_b` as learned low-rank `phi_0`
      parameters.
    - [x] Use prefix-chunk next-token loss for differentiable inner updates and
      heldout/future chunk next-token loss for the outer objective.
    - [x] Keep base checkpoint parameters frozen by default; allow explicit
      trainability scopes to update checkpoint parameters through the outer
      loss when a researcher chooses that path.
  - [x] Add prefill-time next-token update loop.
    - [x] Split long context into chunks, compute next-token loss per prefix
      chunk, and update request-local fast weights before generation.
  - [x] Add gated base/adapted mixture.
  - [x] Add reset/isolation semantics per request.
    - [x] Expose explicit `reset`, `prefill_update`, and `apply` operations.
    - [x] Assert request-local fast state is never saved into model
      checkpoints.
- [x] Add meta-training mechanism script.
  - [x] Replace `scripts/meta_train_ttt.py` smoke demo with a mechanism-complete
    script that accepts user-provided or synthetic long-context chunks.
  - [x] Implement research-matrix outer loop: unroll inner fast-weight updates
    over prefix chunks, compute outer loss on heldout/future chunks, and update
    checkpoint/meta-initialization parameters through the unroll.
  - [x] Add synthetic long-context batches for CI only.
  - [x] Track pre-update and post-update loss.
  - [x] Save learned TTT components in checkpoints.
- [x] Add long-context acceptance tests.
  - [x] Add a 32K-token chunked prefill mechanism test for request-local fast
    weights.
  - [x] Add bounded fast-state-size coverage as the CI-safe proxy for bounded
    latency.
  - [x] Drift tests after many updates.
- [ ] Add downstream retrieval-quality benchmark once trained checkpoints or
  user-provided TTT training data are available.

## TurboQuant KV Cache

- [x] Add reference `TurboQuantKVCache` compressor/decompressor.
- [x] Store optional QJL residual signs.
- [x] Integrate TurboQuant with MLA attention cache.
  - [x] Compress `c_kv` cache online.
  - [x] Decompress on attention read in reference path.
  - [x] Compare compressed and uncompressed logits.
  - [x] Add config switch for cache compression.
  - [x] Keep shape, reorder, and compressed-vs-uncompressed logit tests.
- [x] Implement QJL correction in attention scoring.
  - [x] Define exact inner-product correction path from the
    [research matrix](docs/research_matrix.md): residual, projection seed,
    residual norms, sign bits, and correction term.
  - [x] Add packet metadata needed by future kernels.
    - [x] Store projection seed and residual norm if the chosen QJL estimator
      requires them.
    - [x] Version packet metadata so old TurboQuant packets fail cleanly.
  - [x] Implement attention-score estimator using quantized rotated vectors plus
    QJL residual correction.
  - [x] Validate quality on synthetic attention comparisons.
    - [x] Compare full-cache dot products, scalar-quantized dot products, and
      QJL-corrected dot products.
- [x] Add memory and speed benchmarks.
  - [x] FP16 cache baseline.
  - [x] MLA-only cache.
  - [x] MLA plus TurboQuant cache.
  - [x] Report compression ratio and reconstruction error.
  - [x] Benchmark dense QR rotation against any structured random transform
    replacement before changing the reference implementation.

## Training Scripts

- [x] Add smoke `scripts/pretrain.py`.
- [x] Add smoke `scripts/train_diffusion.py`.
- [x] Add smoke `scripts/train_latent.py`.
- [x] Add smoke `scripts/meta_train_ttt.py`.
- [ ] Replace smoke scripts with mechanism-complete training loops.
  - [x] Configurable dataset loading.
  - [x] Gradient accumulation.
  - [x] Mixed precision.
  - [x] Checkpoint save/resume.
  - [x] Validation loop.
  - [x] Metric logging.
  - [x] Router-bias update calls.
  - [x] Shared LR, warmup, and gradient clipping arguments for smoke trainers.
  - [x] Loss, gradient norm, and activation/logit norm logging in smoke
    trainers.
  - [x] Reject invalid step counts, batch sizes, LR, and gradient clipping
    values before training setup.
  - [x] Prove trainers accept externally supplied token-id JSONL or tensor
    batches without requiring Alexandros-owned datasets.
  - [x] Keep synthetic/sample-data modes explicitly labeled as smoke tests.
- [ ] Add distributed training support.
  - [ ] `torchrun` entrypoint.
  - [ ] DDP/FSDP strategy.
  - [ ] Expert-parallel planning hooks.
  - [ ] Seed and checkpoint consistency tests.
- [x] Add reproducibility controls.
  - [x] Lockfile or documented dependency pinning strategy.
  - [x] Record git commit, config hash, dependency versions, and hardware.
  - [x] Save RNG state in smoke training checkpoints.
  - [x] Save dataloader state when a dataloader exists.
  - [x] Add deterministic mode for tiny tests.
  - [x] Add exact rerun instructions for every example.
- [x] Add experiment tracking.
  - [x] Local JSONL logs.
  - [x] Optional Weights & Biases or TensorBoard integration.
    Added optional TensorBoard scalar logging behind `--tensorboard-dir`; it is
    dependency-free unless the argument is used.
  - [x] Standard metric names across AR, diffusion, latent, and TTT training.
  - [x] Artifact layout for checkpoints, configs, logs, and eval reports.

## Inference And Generation

- [x] Add greedy AR generation.
- [x] Add block diffusion generation.
- [x] Add latent reasoning generation.
- [x] Add production-grade AR generation controls.
  - [x] Temperature.
  - [x] Top-k.
  - [x] Top-p.
  - [x] Repetition penalty.
  - [x] Single-token stop IDs.
  - [x] Multi-token stop sequences.
  - [x] Max sequence length enforcement.
- [x] Add cached generation.
  - [x] Reuse MLA cache.
  - [x] Reuse SSM states.
  - [x] Support batch continuation with different prompt lengths.
- [x] Populate `src/alexandros/inference`.
  - [x] Move reusable generation helpers out of model class if needed.
    Added `GenerationRequest`, cache reorder helpers, and
    `generate_from_request(...)`; core mode internals stay on the model until a
    larger serving integration needs a separate decoder object.
  - [x] Add cache utilities.
  - [x] Add CLI/server-ready request schema.

## Evaluation

- [x] Add smoke `scripts/eval.py`.
- [x] Populate `src/alexandros/evaluation`.
  - [x] Parameter and active-parameter reporting.
  - [x] Perplexity evaluation.
  - [x] Masked diffusion reconstruction accuracy.
  - [x] Long-context retrieval/needle tests.
  - [x] Expert load balance metrics.
  - [x] TurboQuant/MLA cache memory estimates.
  - [x] Reference FLOP estimates.
  - [x] TurboQuant reconstruction/cache metrics.
  - [x] Latency and memory profiler.
- [x] Add reasoning/code benchmarks.
  - [x] Tiny deterministic math tasks for CI.
  - [x] HumanEval-style harness placeholder.
  - [x] SWE-Bench/Terminal-Bench notes for future coding-agent scale.
  - [x] Keep code-benchmark helpers non-executing until tokenizer,
    trained-checkpoint, sandbox, dataset/license, and release-policy gates are
    satisfied.
- [x] Add result reporting.
  - [x] JSONL metrics output.
  - [x] Markdown summary output.
  - [x] Baseline comparison table for Heavy vs Lite.
- [x] Add ablation strategy.
  - [x] Dense FFN vs MoE.
  - [x] Standard attention vs MLA.
  - [x] Full attention ratio variants.
  - [x] Gated DeltaNet vs attention-only.
  - [x] Heavy vs Lite at matched active parameters.
  - [x] AR vs block diffusion vs latent reasoning.
  - [x] TTT on/off for long context.
  - [x] TurboQuant on/off for cache quality and memory.
- [x] Add external baseline policy.
  - [x] Pick small open baselines for tiny/local comparisons.
  - [x] Define fair-tokenizer caveats.
  - [x] Track hardware and precision for every benchmark.

## Tests

- [x] Add `tests/test_smoke.py`.
- [x] Run and harden existing tests after installing dependencies.
- [x] Add focused unit tests per module.
  - [x] `test_attention_mla.py`.
  - [x] `test_moe.py`.
  - [x] `test_bitlinear.py`.
  - [x] `test_diffusion.py`.
  - [x] `test_latent_reasoning.py`.
  - [x] `test_ttt.py`.
  - [x] `test_kv_cache.py`.
  - [x] `test_generation.py`.
- [x] Add integration tests.
  - [x] Heavy tiny overfits a toy sequence.
  - [x] Lite tiny trains for several steps without NaNs.
  - [x] Save/load preserves logits in eval mode.
  - [x] Smoke training checkpoint resume.
  - [x] Smoke training validation metrics.
  - [x] Cached and uncached generation match.
  - [x] Diffusion fills mask tokens consistently.
- [x] Add property/edge tests.
  - [x] Empty or all-pad input handling.
  - [x] Diffusion all-mask and no-mask/all-pad edge cases.
  - [x] Diffusion masking can use a local generator without advancing global
    RNG state.
  - [x] Diffusion training timesteps reject non-integer, out-of-range, and
    non-broadcastable values before masking.
  - [x] Diffusion training transforms reject empty, non-integer, and
    out-of-vocabulary token tensors before masking.
  - [x] Block diffusion preserves prompt tokens equal to `mask_token_id`.
  - [x] Zero-token generation returns the prompt unchanged.
  - [x] Config validation rejects duplicate special token IDs.
  - [x] Latent refinement remains finite when requested steps exceed configured
    diffusion steps.
  - [x] Odd attention head dimensions are rejected because RoPE requires even
    pairs.
  - [x] Sequence length over max position is rejected.
  - [x] Sequence length exactly at max position.
  - [x] `moe_top_k=1` and `moe_top_k=num_experts`.
  - [x] Batch size one and mixed prompt lengths.
  - [x] Invalid generation kwargs are rejected by mode.
  - [x] AR sampling control validation.
  - [x] Right-padded generation prompts use compact per-row continuation, while
    left padding and interior padding are rejected.
  - [x] Out-of-vocabulary input token IDs are rejected.
  - [x] Attention masks reject non-0/1 values, non-finite values, and interior
    holes instead of silently treating them as valid context.
  - [x] Cached forwards reject full attention masks with interior padding holes.
  - [x] Cached forwards validate past MLA cache and SSM state shapes before
    tensor operations.
  - [x] Out-of-vocabulary causal labels are rejected before loss computation.
  - [x] Non-integer input tensors and non-`torch.long` labels are rejected with
    clear errors.
  - [x] Malformed or all-ignored causal labels are rejected.
  - [x] Token-id JSONL rejects boolean values as invalid token IDs.
  - [x] Packed token iterator resume state validates order permutations,
    integer metadata, generator-state dtype, seed type, and shuffle type before
    restoring.
  - [x] Cache reorder rejects out-of-range beam indices.
  - [x] Cache reorder rejects non-tensor and non-integer beam indices.
  - [x] Ternary BitLinear code helpers reject non-integer, boolean, and
    reserved code values before packing or decoding.
  - [x] Lite activation quantization rejects invalid one-bit settings that
    would produce non-finite scales.
  - [x] TurboQuant cache packets validate bit width, scale shape, value ranges,
    and QJL sign metadata before decompression or cached decode.
  - [x] TurboQuant compression rejects non-floating, empty-last-dimension, and
    non-finite tensors before quantization.
  - [x] TTT state rejects invalid constructor values, incompatible hidden-state
    tensors, and non-finite update controls.
  - [x] Latent VAE and reasoner reject empty, malformed, non-floating, and
    non-finite hidden/latent tensors before refinement.
  - [x] MLA attention rejects malformed hidden states, masks, and cache tensors
    before attention scoring.
  - [x] Config stability knobs reject NaN/Inf, boolean, and out-of-range float
    values before model construction.
  - [x] Generation request schema rejects non-integer token payloads instead of
    coercing them with `int(...)`.
  - [x] Generation sampling controls reject NaN/non-finite values and non-integer
    `top_k` settings.
  - [x] Generation length, diffusion step, and latent step controls reject
    boolean, non-integer, negative, or zero values as appropriate.
  - [x] Generation stop controls reject non-integer and out-of-vocabulary token
    IDs.
  - [x] MoE diffusion timestep routing rejects boolean, non-integer,
    out-of-range, empty, and non-broadcastable timestep inputs.

## CLI And Developer Experience

- [x] Add basic CLI scripts.
- [x] Improve CLI ergonomics.
  - [x] Shared argument parser.
  - [x] Device selection.
  - [x] Dtype selection.
  - [x] Seed control.
  - [x] Output directory handling.
  - [x] Helpful error when `torch` is not installed.
- [x] Add examples.
  - [x] Minimal model instantiation.
  - [x] Tiny AR training.
  - [x] Tiny diffusion generation.
  - [x] Lite BitLinear inspection.
  - [x] Save/load round trip.
- [x] Add contributor docs.
  - [x] Development setup.
  - [x] Test commands.
  - [x] Coding style.
  - [x] How to update this roadmap.

## Documentation

- [x] Update `README.md` with quick start and document links.
- [x] Add API reference.
  - [x] Config fields.
  - [x] Model outputs.
  - [x] Generation modes.
  - [x] Cache packet formats.
- [x] Add training guide.
  - [x] AR phase.
  - [x] Diffusion phase.
  - [x] Latent phase.
  - [x] TTT phase.
  - [x] Lite-specific training cautions.
  - [x] Document training-data scope: user-supplied corpora/tokenizers,
    synthetic or permissively licensed sample fixtures only, and no internet
    data gathering.
- [x] Fold reference-only implementation gaps into this roadmap.
  - [x] Convert every completion bar from the former gap guide into task items.
  - [x] Remove the standalone `docs/implementation_gaps.md` file.
  - [x] Update docs and README references to point at this roadmap instead.

## Serving And Optimization

- [x] Add optimized kernel roadmap artifacts.
  - [x] Triton MLA attention plan.
  - [x] SDPA attention plan with CPU fallback.
  - [x] Sparse MoE dispatch plan.
  - [x] Packed BitLinear plan.
  - [x] TurboQuant attention read plan.
  - [x] Document acceptance criteria in `docs/optimized_kernels.md`.
- [x] Add serving integration plan.
  - [x] vLLM compatibility investigation.
  - [x] Text Generation Inference compatibility investigation.
  - [x] bitnet.cpp export investigation.
  - [x] CPU-only Lite inference path.
  - [x] Packed BitLinear runtime serving path.
  - [x] Sparse MoE runtime serving path.
  - [x] Document compatibility gates in `docs/serving_integration.md`.
- [x] Add benchmark targets.
  - [x] Tokens/sec for AR.
  - [x] Tokens/sec equivalent for block diffusion.
  - [x] Memory by sequence length.
  - [x] Active parameter count by variant.
- [x] Add hardware target matrix.
  - [x] CPU-only minimum viable path.
  - [x] Single consumer GPU path.
  - [x] Single data-center GPU path.
  - [x] Multi-GPU training path.
  - [x] Expected unsupported hardware paths.

## Security, Safety, And Release Governance

- [x] Harden serialization security.
  - [x] Prefer `safetensors` for untrusted checkpoints.
  - [x] Document `torch.load` trust assumptions while it remains supported.
  - [x] Add checksum/signature guidance for released checkpoints.
- [x] Add model release artifacts.
  - [x] Model card template.
  - [x] Dataset card template.
  - [x] Evaluation report template.
  - [x] Known limitations and misuse statement.
- [x] Add supply-chain hygiene.
  - [x] Pin or constrain dependencies.
  - [x] Audit optional CUDA/kernel dependencies.
  - [x] Document build requirements for any native extensions.
- [x] Add safety evaluation plan if instruction-tuned models are released.
  - [x] Basic harmful request eval.
  - [x] Privacy leakage eval.
  - [x] Memorization/copyright probe plan.
  - [x] Red-team notes and release gates.

## Deferred Community Training And Data Work

These items are intentionally deprioritized for the initial publication. The
Alexandros repo should make training mechanisms correct and interoperable, but
large-scale data acquisition, full training curricula, and post-training are
expected to be handled by downstream trainers or the broader AI community.

- [x] Choose or train a production tokenizer.
  V1 chooses downstream-supplied tokenizers and does not publish an official
  tokenizer or tokenizer trainer.
  - [x] Decide between SentencePiece, tiktoken-style BPE, Hugging Face
    tokenizers, or a tokenizer supplied by a downstream trainer.
  - [x] Reserve special tokens for mask and latent/reasoning markers.
  - [x] Add tokenizer training script only if the project later chooses to
    publish an official tokenizer.
- [x] Curate or document large-scale training datasets.
  V1 does not curate a corpus; the repo now documents downstream data-policy
  requirements and keeps only tiny CI fixtures.
  - [x] Track dataset licenses and allowed use.
  - [x] Add PII filtering policy.
  - [x] Add benchmark contamination/decontamination checks.
  - [x] Add dataset cards for any bundled or recommended datasets.
  - [x] Document copyright and redistribution assumptions.
  - [x] Do not add internet scraping or web-scale data collection scripts unless
    project scope is explicitly changed.
- [x] Define a full training curriculum for community runs.
  - [x] AR pretraining objective and stopping criteria.
  - [x] Diffusion objective warm-start or train-from-scratch choice.
  - [x] Latent reasoning curriculum from visible traces to latent-only traces.
  - [x] TTT meta-training stage and prerequisites.
  - [x] Optional instruction tuning and preference tuning stages.
- [x] Optional teacher-model distillation path.
  - [x] Support user-provided teacher logits or generated token targets.
    Added `iter_distillation_jsonl`, `DistillationDataset`, batch iteration,
    and `distillation_loss` for masked teacher-token CE and teacher-logit KL.
  - [x] Require contributors to verify teacher model license, terms, and
    redistribution constraints before publishing distilled checkpoints.
  - [x] Add tiny synthetic distillation fixture for tests only.
  - [x] Document that teacher selection is not part of the initial release.
- [x] Decide whether Alexandros includes post-training in scope.
  - [x] If out of scope, document base-model-only boundaries.
  - [x] If in scope later, add SFT data format and training script.
- [x] Add instruction tuning plan only after a dataset/tokenizer decision.
  - [x] Chat template/tokenizer integration.
  - [x] Supervised fine-tuning loop.
  - [x] Instruction-following evaluation.
- [x] Add preference/alignment plan only after post-training scope is accepted.
  - [x] DPO/IPO/ORPO or RL preference method selection.
  - [x] Safety and refusal dataset policy.
  - [x] Evaluation for over-refusal and unsafe compliance.
- [x] Add tool-use/agentic training plan if coding-agent use is desired.
  - [x] Tool call schema.
  - [x] Environment feedback loop.
  - [x] Verifiable coding task dataset.
  - [x] Terminal/SWE-style evaluation.

## Repository Hygiene

- [x] Add `.gitignore`.
  - [x] Ignore `__pycache__`.
  - [x] Ignore checkpoints.
  - [x] Ignore logs and local datasets.
  - [x] Ignore build artifacts.
- [ ] Decide fate of `AI model Idea Alexandros.md`.
  - [ ] Keep as source artifact and commit intentionally.
  - [ ] Or move into `docs/`.
  - [ ] Or ignore if it should remain local only.
- [x] Add formatting/linting.
  - [x] Choose Ruff.
  - [x] Add formatter config.
  - [x] Add import sorting.
  - [x] Add CI lint step.
  - [x] Document local lint/format commands for contributors and releases.
