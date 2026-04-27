# Alexandros Architecture Decisions

This document records the major implementation decisions behind the current
prototype. The goal is to prevent future work from confusing a deliberate
reference implementation with an accidental omission.

## ADR-001: Keep V1 As A Reference PyTorch Prototype

**Decision:** V1 prioritizes readable PyTorch modules, CPU smoke tests, and
small-model correctness over optimized kernels or distributed training.

**Rationale:** Alexandros combines several experimental ideas at once:
hybrid linear-attention/attention blocks, MLA-style cache compression, sparse
MoE, diffusion language modeling, latent reasoning, BitNet-style ternary
linears, TTT fast weights, and TurboQuant cache compression. A clear reference
implementation gives new contributors a stable behavioral target before they
replace individual modules with faster or more paper-faithful versions.

**Consequences:**

- CPU tests remain first-class.
- Optimized CUDA/Triton/vLLM/bitnet.cpp paths must preserve the public API and
  tiny-config behavior.
- Performance claims are not accepted until benchmark scripts measure them.

## ADR-002: Paper-Faithful Work Is Tracked Explicitly

**Decision:** Modules may start as approximations only when the approximation is
documented and tracked in `Roadmap.md`.

**Deferred paper-faithful behavior includes:**

- Optimized associative/SSD kernels for matrix DeltaNet and native Mamba-2.
  A readable matrix-state DeltaNet reference backend exists behind
  `linear_mixer_backend="matrix_deltanet"`.
- Lower-rank DeepSeek-style query compression for MLA. Separate
  RoPE/no-RoPE channels and positional `k_rope` cache support exist behind
  `mla_rope_dim`.
- Fully paper-faithful continuous-time MDLM/LLaDA weighting, learned remasking,
  and confidence schedules beyond the current masked/Rao-Blackwellized
  reference objectives and explicit research weighting knobs.
- Optimized TTT-E2E kernels and large-scale long-context quality validation.
  The reference learned low-rank meta-adapter exists, but it is intentionally a
  readable PyTorch mechanism rather than a production kernel.
- Fused QJL-aware TurboQuant attention reads inside MLA kernels; the reference
  cache-level QJL score estimator exists, but the MLA hot path still
  decompresses before learned projections.
- Distributed expert parallelism for MoE.

**Consequences:**

- A roadmap checkbox is not complete if it only changes documentation.
- A paper-faithful replacement must keep the reference path or an equivalent
  CPU-testable fallback unless the project explicitly drops CPU support.

## ADR-003: One Config Drives Heavy And Lite

**Decision:** Heavy and Lite use the same `AlexandrosConfig` and model classes.
The `variant` field selects full-precision linears or BitNet-style `BitLinear`
through the shared `make_linear` factory.

**Rationale:** Keeping one architecture family makes tests and checkpoints
easier to compare. It also lets Lite inherit architectural fixes without a
parallel model stack.

**Consequences:**

- New linear projections must use `make_linear` unless intentionally
  full-precision.
- Embeddings, normalization layers, timestep embeddings, and router biases stay
  full precision in Lite unless a future ADR changes that policy.
- Lite-specific serving/export work belongs behind explicit export paths, not
  inside the training-time `BitLinear` shadow-weight implementation.

## ADR-004: Cache Semantics Are Mode-Specific

**Decision:** Cache reuse is supported for autoregressive generation only.
Diffusion and latent generation reject cached generation until their masking and
state semantics are proven safe.

**Rationale:** AR cache reuse is causally well-defined. Block diffusion and
latent refinement can revise multiple token positions, so blindly reusing KV or
SSM state risks incorrect logits.

**Consequences:**

- MLA and SSM caches are valid for AR decode.
- TurboQuant cache compression is only applied to AR MLA cache packets.
- `diffusion_attention_mask_mode` may make MLA attention bidirectional during
  denoising, but this does not make recurrent Gated DeltaNet layers
  bidirectional.
- Any future diffusion cache must include tests that prove masked-span
  denoising remains correct.

## ADR-005: Router Bias Updates Are Non-Gradient Training State

**Decision:** MoE load balancing uses a persistent router-bias buffer updated
outside backprop from a non-persistent expert-load EMA.

**Rationale:** This follows the auxiliary-loss-free load-balancing direction
without adding router losses to the main objective. The EMA is runtime training
state and should not make ordinary forwards mutate checkpoint contents.

**Consequences:**

- `router_bias` is checkpointed.
- `router_load_ema` and `router_load_ema_steps` are not checkpointed.
- Training scripts control update cadence through
  `router_bias_update_interval`.

## ADR-006: Initialization Is Explicit And Neutral Where State Learns Offsets

**Decision:** Alexandros uses explicit reference initialization instead of
relying on PyTorch module defaults. Trainable projection weights use a small
normal initializer with residual-output projections scaled by model depth.
Embeddings use the same small normal initializer, with the pad-token row set to
zero. Norm weights start at one and norm biases start at zero.

MoE load-balancing buffers and timestep, token-state, and position router-bias
embeddings start at zero. This keeps initial expert selection driven by the base
router rather than arbitrary routing offsets; routing features can still be
learned or set explicitly in tests. Latent diffusion timestep embeddings use the
normal initializer because they are feature conditioning vectors, not routing
offsets.

**Consequences:**

- Heavy `nn.Linear` weights and Lite `BitLinear` shadow weights follow the same
  reference initializer.
- Router bias, router EMA buffers, and timestep expert counters reset to zero.
- Tied Heavy LM heads share the already-initialized token embedding weights.
- Future optimized kernels must preserve these initial checkpoint semantics or
  document a migration.

## ADR-007: Depth-Wise LoRA Is Scoped To Adaptive Depth In V1

**Decision:** V1 implements per-loop low-rank adapters inside
`AdaptiveDepthController` and exposes optional `depth_lora_ranks` for per-loop
rank control. It does not inject LoRA adapters into staged backbone recurrent
layers.

**Rationale:** The adaptive-depth transition projection is a clear, isolated
owner for loop-specific residual adapters. The staged prelude/recurrent/coda
stack reuses ordinary Alexandros blocks whose projections also serve attention,
SSM, MoE, and Lite BitLinear behavior. Adding per-loop adapters there would
require explicit projection ownership and cache semantics before it is safe.

**Consequences:**

- `depth_lora_rank` remains the default rank for every adaptive-depth loop.
- `depth_lora_ranks` can override ranks per ACT loop.
- Staged backbone recurrent projections remain unchanged in v1.
- Future backbone depth-wise LoRA must specify which projections are adapted,
  how adapters interact with cache reuse, and how Lite `BitLinear` adapters are
  represented.

## Acceptance Criteria For Replacing Reference Modules

A replacement for any reference module must satisfy all of these before the
roadmap item can be checked off:

- Preserve public config fields or provide a documented migration.
- Pass existing tiny-config unit and integration tests.
- Add at least one test that would fail on the previous approximation.
- Document tensor shapes, cache/state semantics, and unsupported modes.
- Include a CPU fallback or update the hardware support matrix and CI policy.
- Demonstrate finite forward/backward behavior and deterministic eval behavior
  for Heavy and Lite when applicable.
