# Alexandros Architecture Research Matrix

This document turns the research assumptions behind Alexandros into concrete
implementation instructions. Implementation agents should treat this file as a
local research dossier and should not need to browse before making changes.

## Source Index

- DeepSeekMoE: https://arxiv.org/abs/2401.06066
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- Jamba: https://arxiv.org/abs/2403.19887
- Mamba-2 / Structured State Space Duality: https://arxiv.org/abs/2405.21060
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- Kimi Linear: https://arxiv.org/abs/2510.26692
- Qwen3-Coder-Next: https://arxiv.org/abs/2603.00729
- MDLM: https://arxiv.org/abs/2406.07524
- LLaDA: https://arxiv.org/abs/2502.09992
- Mercury 2: https://www.inceptionlabs.ai/blog/introducing-mercury-2
- DiT-MoE: https://arxiv.org/abs/2407.11633
- Diff-MoE: https://openreview.net/forum?id=JCUsWrwkKw
- Efficient Training of Diffusion MoE: https://arxiv.org/abs/2512.01252
- COCONUT: https://arxiv.org/abs/2412.06769
- LaDiR: https://arxiv.org/abs/2510.04573
- SpiralThinker: https://arxiv.org/abs/2511.08983
- BitNet b1.58: https://arxiv.org/abs/2402.17764
- BitNet b1.58 2B4T: https://arxiv.org/abs/2504.12285
- TerDiT: https://arxiv.org/abs/2405.14854
- Microsoft bitnet.cpp: https://github.com/microsoft/BitNet
- TTT-E2E: https://arxiv.org/abs/2512.23675
- TurboQuant: https://arxiv.org/abs/2504.19874
- Google TurboQuant post: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- OpenMythos: https://github.com/kyegomez/OpenMythos
- OpenMythos class docs: https://github.com/kyegomez/OpenMythos/blob/main/docs/open_mythos.md

## Fine-Grained MoE

**Sources:** DeepSeekMoE, DeepSeek-V3.

**Core idea:** Scale total parameters without activating all parameters per
token. Use many small routed experts, a small set of always-on shared experts,
top-k routing, and load-balancing mechanisms that avoid degrading the main loss.

**Alexandros adaptation:** `MoEFeedForward` uses one or more shared SwiGLU
experts plus normalized sigmoid top-k routed experts. Router bias is a
non-gradient buffer updated from a multi-batch expert-load EMA at a configurable
optimizer-step cadence, matching the auxiliary-loss-free spirit of DeepSeek-V3.

**V1 simplification:** The default path uses readable PyTorch sparse dispatch
instead of fused expert-parallel kernels. A dense reference path remains
available for correctness comparison.

**Risks:** Expert collapse, unstable routing, and excessive memory use when
expert count grows.

**Tests:** Assert top-k routing shape, shared expert contribution, router-bias
updates away from overloaded experts, EMA accumulation across batches, cadence
control, and nonzero load across experts.

### Implementation Notes

Use sigmoid scores for routing and normalize only across the selected experts.
The load-balancing bias is used for top-k selection, not for the final mixture
weights.

```text
Input:
  h                 token hidden state, shape [B, T, d_model]
  W_router          router matrix, shape [d_model, E]
  b_balance         non-gradient router bias, shape [E]
  shared_experts    always-on experts S_j
  routed_experts    sparse experts E_i
  k                 routed top-k count

Scores:
  logits[b,t,e] = clip(h[b,t] @ W_router[:,e], -router_logit_clip, router_logit_clip)
  probs[b,t,e]  = sigmoid(logits[b,t,e])

Selection:
  select_score[b,t,e] = probs[b,t,e] + b_balance[e]
  K[b,t] = top_k_indices(select_score[b,t], k)

Normalized sigmoid routing:
  denom[b,t] = sum_{e in K[b,t]} probs[b,t,e] + eps
  gate[b,t,e] = probs[b,t,e] / denom[b,t] for e in K[b,t]

Output:
  y_shared[b,t] = mean_j S_j(h[b,t])
  y_routed[b,t] = sum_{e in K[b,t]} gate[b,t,e] * E_e(h[b,t])
  y[b,t] = y_shared[b,t] + y_routed[b,t]
```

Bias-update rule for auxiliary-loss-free balancing:

```text
per_batch_load[e] = selected_count[e] / sum_i selected_count[i]
load_ema[e] = decay * load_ema[e] + (1 - decay) * per_batch_load[e]
target[e] = 1 / E
b_balance[e] = clip(
    b_balance[e] + router_bias_update_rate * (target[e] - load_ema[e]),
    -router_bias_clip,
    router_bias_clip,
)
```

Implementation constraints:

- Do not backpropagate through `b_balance`.
- Use the bias only to decide which experts are selected.
- Keep a dense reference path until sparse dispatch has shape, value, and load
  tests.
- In a future expert-parallel path, define dispatch tensors as
  `[num_selected_tokens_for_expert, d_model]` and combine by scattering weighted
  outputs back to `[B, T, d_model]`.

## Hybrid Long-Context Backbone

**Sources:** Jamba, Mamba-2, Gated DeltaNet, Kimi Linear, Qwen3-Coder-Next.

**Core idea:** Use mostly linear-time recurrent/SSM/linear-attention blocks for
long context, with occasional full attention layers for precise recall.

**Alexandros adaptation:** `AlexandrosBlock` chooses `MLAAttention` on scheduled
attention layers and a configurable linear mixer on the remaining layers. The
default tiny/base schedule uses three linear blocks per one attention block.
`linear_mixer_backend="gated_deltanet"` uses the compact elementwise
DeltaNet-inspired reference recurrence, `linear_mixer_backend="matrix_deltanet"`
uses a per-head matrix-state Gated DeltaNet recurrence, and
`linear_mixer_backend="mamba2"` uses a local diagonal SSM backend.

**V1 simplification:** `MatrixGatedDeltaNetBlock` implements the paper-style
matrix recurrence in readable PyTorch, including causal chunked prefill, but not
an optimized associative/SSD scan kernel. `GatedDeltaNetBlock` remains the
compact elementwise fallback. `Mamba2Block` is a minimal Mamba-2/SSD-style
diagonal recurrence rather than the fused selective-scan implementation.

**Risks:** Linear states can forget details; attention layers can dominate
memory at long context.

**Tests:** Shape invariants, deterministic recurrence state shape, generation
with mixed block types, and synthetic long-context retrieval in later stages.

### Implementation Notes

All linear mixers keep this public interface:

```text
forward(x, state=None, attention_mask=None) -> (y, next_state)

x:          [B, T, d_model]
y:          [B, T, d_model]
state:      backend-owned recurrent state, same dtype/device as x
next_state: backend-owned recurrent state, detached for cached inference
```

Current state shapes:

```text
GatedDeltaNetBlock:       [batch, hidden_size]
Mamba2Block:              [batch, hidden_size]
MatrixGatedDeltaNetBlock: [batch, heads, value_dim, key_dim]
```

All backends keep explicit shape/device/dtype validation and
cached-vs-chunked equivalence tests.

Current dependency decision: Alexandros does not depend on `mamba-ssm` in v1.
The optional Mamba-2 backend is implemented locally as `Mamba2Block`, using
diagonal elementwise retention and write gates so CPU tests can validate the
contract without native kernels.

The paper-faithful matrix backend uses a fast-weight state per head:

```text
For token t and head h:
  q_t, k_t, v_t = projections(x_t)
  beta_t        = sigmoid(beta_proj(x_t))       # write strength
  alpha_t       = sigmoid(alpha_proj(x_t))      # retention gate
  k_t           = normalize(k_t)

  # Delta rule: write the residual between target value and current read.
  read_t        = S_{t-1} @ k_t
  delta_t       = v_t - read_t
  S_t           = alpha_t * S_{t-1} + beta_t * outer(delta_t, k_t)
  y_t           = S_t @ q_t
```

Equivalent implementations sometimes write the gated update as:

```text
S_t = S_{t-1} @ (alpha_t * (I - beta_t * outer(k_t, k_t)))
      + beta_t * outer(v_t, k_t)
```

Mamba-2/SSD-compatible backend contract:

```text
Continuous/state-space intuition:
  s_t = A_t * s_{t-1} + B_t * x_t
  y_t = C_t * s_t + D * x_t

Chunked SSD form for i in a chunk:
  y_i = sum_{j <= i} C_i * (prod_{m=j+1..i} A_m) * B_j * x_j
```

`MatrixGatedDeltaNetBlock` uses `deltanet_chunk_size` to slice long prefill into
causal chunks while carrying the exact recurrent state across chunk boundaries.
That proves chunk/cache semantics in the reference implementation; an optimized
associative SSD scan can replace the inner loop later if it preserves the same
tests.

Acceptance criteria before replacing the reference block:

- Prefill and recurrent one-token decode match within tolerance on short
  sequences.
- `state` never contains gradients during cached inference.
- The backend handles masked/padded rows consistently with the model attention
  mask policy.
- Long-context diagnostics track state norm, update norm, drift, synthetic
  copy retrieval, and early/middle/late needle-rank behavior.

## Multi-Head Latent Attention

**Source:** DeepSeek-V3 / DeepSeek-V2 MLA.

**Core idea:** Cache a compressed KV latent instead of full key/value tensors,
then reconstruct K/V during attention reads. Store positional/RoPE information
separately.

**Alexandros adaptation:** `MLAAttention` projects hidden states to a compact
`c_kv`, reconstructs no-RoPE keys and values through learned up-projections,
and can store a separate `k_rope` positional cache when `mla_rope_dim > 0`.

**V1 simplification:** The implementation reconstructs dense K/V in PyTorch
before scaled dot-product attention. Query compression is represented by a
single full query projection that is split into `q_nope` and `q_rope` channels;
a lower-rank query down/up path remains optional future work.

**Risks:** Reconstruction cost can offset memory savings without fused kernels.

**Tests:** Cache stores rank-sized tensors, output shape matches input, and cache
memory is smaller than full KV for configured ranks.

### Implementation Notes

Use these symbols:

```text
B       batch size
T       current sequence length
H       number of attention heads
Dh      per-head content dimension
d_model hidden size = H * Dh
d_c     compressed KV latent rank
d_R     per-token RoPE key dimension, shared or broadcastable across heads
d_nope  no-position query/key dimension
```

Paper-faithful MLA cache layout:

```text
c_kv      = x @ W_DKV                      # [B, T, d_c]
k_rope    = x @ W_KR                       # [B, T, d_R]

q_down    = x @ W_DQ                       # optional compressed query latent
q_nope    = q_down @ W_UQ_nope             # [B, T, H, d_nope]
q_rope    = q_down @ W_UQ_rope             # [B, T, H, d_R]

k_nope    = c_kv @ W_UK_nope               # [B, T, H, d_nope]
v         = c_kv @ W_UV                    # [B, T, H, Dv]

q         = concat(q_nope, RoPE(q_rope))    # [B, T, H, d_nope + d_R]
k         = concat(k_nope, RoPE(k_rope))    # [B, T, H, d_nope + d_R]
```

Cache exactly:

```text
cache.c_kv   shape [B, total_T, d_c]
cache.k_rope shape [B, total_T, d_R]
```

The default reference configuration keeps `mla_rope_dim = 0`, stores only
`c_kv`, and applies RoPE to the reconstructed full key for backward-compatible
tiny configs. Setting `mla_rope_dim > 0` activates the paper-faithful split:
`q_nope/k_nope` are content channels, `q_rope/k_rope` receive RoPE, and `c_kv`
remains position-independent.

Cache-size formula for tests:

```text
standard_mha_elements_per_token = 2 * H * Dh
mla_elements_per_token          = d_c + d_R
compression_ratio               = standard_mha_elements_per_token / (d_c + d_R)
```

The config exposes these as derived properties:
`mla_d_c = kv_lora_rank`, `mla_d_r = mla_rope_dim`, and
`mla_d_nope = head_dim - mla_d_r`. `mla_rope_dim` must be even and smaller than
`head_dim`.

Acceptance criteria:

- Cached and uncached logits match for greedy AR decode.
- Cache tensors have the exact shapes above when `mla_rope_dim > 0`.
- TurboQuant, if enabled, wraps `c_kv` only; `k_rope` remains uncompressed until
  a separate proven positional-cache compressor exists.

## Diffusion Language Modeling

**Sources:** MDLM, LLaDA, Mercury/Mercury 2.

**Core idea:** Replace purely left-to-right decoding with masked or blockwise
parallel denoising. Generate multiple tokens per step and refine them over a
small number of iterations.

**Alexandros adaptation:** `MaskedDiffusionScheduler` masks tokens using an
absorbing mask token. `AlexandrosForDiffusionLM.generate_block_diffusion`
appends a masked block, predicts appended masked tokens in parallel, commits
high-confidence predictions, and iterates. `diffusion_attention_mask_mode`
lets diffusion forwards keep causal MLA attention or use bidirectional MLA
attention for denoising experiments. The sampler supports greedy or sampled
token selection while preserving prompt tokens.

**V1 simplification:** The sampler uses greedy confidence commits instead of a
learned noise schedule or advanced remasking policy. Bidirectional masking is
limited to MLA attention layers; Gated DeltaNet layers remain directional.

**Risks:** Training/inference mismatch, weak open-ended generation, and poor
calibration of token confidence.

**Tests:** Masking ratio behavior, denoising returns non-mask tokens, and
diffusion loss only trains masked positions when labels are supplied.

### Implementation Notes

Masked diffusion treats `[MASK]` as an absorbing noised state. For a token
sequence `x0` and timestep `t`, construct `x_t` by independently replacing
non-special tokens with `mask_token_id`.

```text
Discrete schedule:
  t in {0, ..., S - 1}
  mask_prob(t) = (t + 1) / S

Forward corruption:
  m_i ~ Bernoulli(mask_prob(t)) for each non-pad token i
  x_t[i] = MASK if m_i else x0[i]

Training objective:
  w(t) = 1 by default
  L = sum_{i:m_i=1} w(t_i) * CE(model(x_t, t)[i], x0[i]) / max(sum_i m_i, 1)
```

Alexandros exposes explicit research weighting knobs:

```text
uniform:           w(t) = 1
mask_prob:         w(t) = mask_prob(t)
inverse_mask_prob: w(t) = 1 / max(mask_prob(t), eps)
```

These schedules are mechanism tests, not a claim of paper-faithful MDLM/LLaDA
weighting. Alexandros also exposes `diffusion_objective="rao_blackwellized"`.
For a sampled noised background `x_t`, it expands every non-pad target position
`i`, forces `x_t[i] = MASK`, computes the CE for that target, and averages the
weighted losses by non-pad target count. `diffusion_rb_chunk_size` bounds the
expanded batch size without changing the objective value.

```text
Rao-Blackwellized target-position objective:
  V = {i | x0[i] != PAD}
  for i in V:
      x_t^i = x_t
      x_t^i[i] = MASK
      L_i = w(t_i) * CE(model(x_t^i, t_i)[i], x0[i])
  L_rb = sum_i L_i / |V|
```

Block diffusion sampling:

```text
generated = concat(prompt, [MASK] * block_size)
for step in reversed(range(num_steps)):
    logits = model(generated, diffusion_timestep=step)
    logits[..., MASK] = -inf
    candidate_tokens = sample_or_argmax(logits)
    confidence = prob(candidate_tokens)

    masked_positions = positions in generated block still equal to MASK
    if step == 0:
        commit = masked_positions
    else:
        threshold = schedule(step, confidence[masked_positions])
        commit = masked_positions & (confidence >= threshold)

    generated[commit] = candidate_tokens[commit]
```

Future remasking policy:

```text
after each step:
  keep top confidence fraction according to schedule
  remask low-confidence generated block positions
```

Implemented v1 schedules:

- `median`: preserve the original reference behavior by committing positions
  with confidence above the current median.
- `linear`: commit a linearly increasing fraction of positions as denoising
  approaches the final step.
- `all`: commit every candidate position at every step, useful for debugging
  and deterministic smoke tests.

`remask_low_confidence=True` rewrites low-confidence generated block positions
back to `mask_token_id` after each non-final denoising step. `block_size` may
split a continuation into multiple masked chunks while preserving the original
prompt and previously completed chunks.

AR fallback:

```text
if mode == autoregressive:
  use causal mask, cached MLA/SSM state, and next-token loss/generation
else:
  reject cache reuse unless a bidirectional/recurrent denoising contract exists
```

## Diffusion Plus MoE

**Sources:** DiT-MoE, Diff-MoE, Efficient Training of Diffusion MoE.

**Core idea:** Expert choice in diffusion depends on denoising step and token
position; early/noisy steps and late/polish steps benefit from different experts.

**Alexandros adaptation:** `MoEFeedForward` accepts `diffusion_timestep` and adds
a learned timestep router bias before sigmoid top-k selection. When
`moe_token_state_routing` is enabled, diffusion forwards also derive a
masked/unmasked token state from `input_ids == mask_token_id` and add a learned
token-state router bias. When `moe_position_routing` is enabled, model forwards
derive absolute token positions and add a learned position-bucket router bias.

**V1 simplification:** The timestep signal is scalar-per-layer, and token state
is binary masked/unmasked. Position is bucketed by `max_position_embeddings`
rather than using a full space/time adaptive router.

**Risks:** Timestep routing can worsen load imbalance or overfit to a fixed
number of diffusion steps.

**Tests:** Router choices change when timestep, token state, or position bucket
changes, and expert statistics are reported per forward pass.

### Implementation Notes

Diffusion MoE routing should expose denoising time to the router without
changing the expert interface.

```text
Input:
  h[b,t]                 token hidden state
  denoise_step[b,t]      integer step in [0, S)
  position[t]            optional sequence position
  base_probs[b,t,e]      sigmoid router scores from hidden state

V1 timestep-aware routing:
  step_bias[b,t,e] = timestep_embedding[denoise_step[b,t], e]
  state_bias[b,t,e] = token_state_embedding[is_masked[b,t], e]
  pos_bias[b,t,e] = position_embedding[position_bucket[t], e]
  select_score[b,t,e] = base_probs[b,t,e] + balance_bias[e]
                      + step_bias[b,t,e] + state_bias[b,t,e] + pos_bias[b,t,e]
  selected = top_k(select_score[b,t], k)
  gate = normalize(base_probs[b,t,selected])
```

Track expert usage by denoising phase:

```text
for each forward:
  for each selected expert e at denoise step s:
      timestep_expert_load[s,e] += 1
      timestep_expert_count[s] += 1

diagnostics:
  noisy_step_load  = rows for high mask-probability steps
  polish_step_load = rows for low mask-probability steps
  per_step_entropy = entropy(timestep_expert_load[s] / timestep_expert_count[s])
```

Time/token-state/position-aware router:

```text
router_features = h
                + W_step[denoise_step]
                + W_mask_state[is_masked]
                + W_pos[position_bucket]
logits = router(router_features)
```

Acceptance criteria:

- `diffusion_timestep` must validate shape and range before routing.
- Optional masked/unmasked token-state routing must validate broadcastable 0/1
  state tensors; model diffusion forwards derive this state from
  `input_ids == mask_token_id`.
- Optional position routing must validate broadcastable integer positions in
  `[0, max_position_embeddings)` and map them into configured buckets.
- Timestep, token-state, and position bias may affect expert selection, but
  load-balancing bias still remains non-gradient and auxiliary-loss-free.
- Expert-use reports must include per-step load counts, not only aggregate load.

## Latent Reasoning

**Sources:** COCONUT, LaDiR, SpiralThinker.

**Core idea:** Move reasoning into continuous latent vectors and refine those
vectors before decoding, reducing visible chain-of-thought tokens while allowing
iterative internal computation.

**Alexandros adaptation:** `LatentThoughtVAE` compresses hidden states into
latent thought slots. `LatentDiffusionReasoner` iteratively refines those slots
with bidirectional slot self-attention, timestep embeddings, a compact
feed-forward update, and optional small-update adaptive halting through
`latent_adaptive_threshold`. `AdaptiveDepthController` optionally repeats
backbone hidden-state refinement with ACT-style halting.

**V1 simplification:** The VAE keeps mean pooling as the visible-state
compression path, and the reasoner is a compact latent-slot transformer rather
than a full bidirectional diffusion transformer with learned self-correction.
Adaptive halting is heuristic and update-norm based rather than a learned
reasoning-quality controller.

**Risks:** Latents may become uninterpretable placeholders without improving
reasoning; hidden reasoning is harder to audit.

**Tests:** Encode/decode shape, KL term finite, changing latent steps changes
outputs, bidirectional slot attention changes a slot from other-slot context,
adaptive latent halting stops on zero/small updates, and ACT halting remains
bounded.

### Implementation Notes

Latent reasoning has three separable pieces: visible-state compression,
iterative latent refinement, and optional reconstruction/audit loss.

VAE compression:

```text
Input hidden states H: [B, T, d_model]
pool = mean_or_attention_pool(H)                         # [B, d_model]
mu = pool @ W_mu                                         # [B, d_latent]
logvar = clip(pool @ W_logvar, min_logvar, max_logvar)   # [B, d_latent]
eps ~ Normal(0, I) during training, eps = 0 in eval
z = mu + exp(0.5 * logvar) * eps                         # [B, d_latent]
latents = repeat(z, latent_slots)                        # [B, L, d_latent]
reconstruction = decoder(latents)                        # [B, L, d_model]
kl = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
```

Iterative latent refinement:

```text
h_0 = latents
for step s in 0..S-1:
    t = timestep_embedding[min(s, max_timestep)]
    a_s = bidirectional_slot_attention(norm(h_s + t))
    f_s = feed_forward(norm(h_s + t + a_s))
    u_s = a_s + f_s
    u_s = clip_norm(u_s, latent_update_clip)
    h_{s+1} = h_s + u_s / S
    if latent_adaptive_threshold > 0 and norm(u_s / S) <= latent_adaptive_threshold:
        break
```

COCONUT-style latent thought slots:

```text
visible prefix -> hidden states -> latent slots
latent slots are fed back into the decoder/head as continuous context
visible rationale tokens are optional training supervision, not required at inference
```

LaDiR/SpiralThinker-style future extension:

```text
latent transformer:
  learned self-correction loop with bounded number of iterations
  richer noise/objective conditioning
  learned trace/answer supervision adapters

optional reconstruction:
  trace_tokens -> encoder -> latent_target
  latent_pred -> decoder -> reconstructed_trace_summary
  loss = task_loss + lambda_kl * kl + lambda_rec * reconstruction_loss
```

Acceptance criteria:

- Latent update norm is bounded per step.
- More refinement steps should be able to change outputs, but must remain
  finite.
- Any hidden-reasoning feature must document audit limitations and avoid
  claiming visibility into the exact latent reasoning process.

## Lite Ternary Model

**Sources:** BitNet b1.58, BitNet b1.58 2B4T, TerDiT, Microsoft bitnet.cpp.

**Core idea:** Train models natively with ternary weights `{-1, 0, +1}` and
activation quantization rather than post-training quantizing dense checkpoints.

**Alexandros adaptation:** `BitLinear` stores full-precision trainable weights
but uses straight-through ternary weights during forward passes. It applies
absmean weight scaling and per-token absmax activation quantization. The
reference export packs four ternary weight codes per byte using
`0=zero, 1=+1, 2=-1`, and stores one FP32 absmean scale per output row plus
optional FP32 bias.

**V1 simplification:** The reference layer uses PyTorch dense matmul with
straight-through quantization. The packed export is a converter-friendly
artifact, not the Microsoft bitnet.cpp binary ABI and not an optimized kernel.

**Risks:** Lite training may require warmup, smaller learning rates, and careful
normalization to avoid instability.

**Tests:** Ternary projected weights contain only `-1, 0, +1`, gradients reach
the full-precision shadow weights, packed codes round-trip to the ternary
weights, and Lite tiny forwards without NaNs.

### Implementation Notes

BitNet b1.58-style reference layer:

```text
Shadow weight:
  W_fp                 trainable full-precision parameter [out, in]

Ternary projection:
  alpha[o] = mean_i(abs(W_fp[o,i])) + eps
  W_norm[o,i] = W_fp[o,i] / alpha[o]

  W_ternary[o,i] =
      +1 if W_norm[o,i] >  threshold
       0 if abs(W_norm[o,i]) <= threshold
      -1 if W_norm[o,i] < -threshold

  W_q = alpha * W_ternary

Straight-through estimator:
  W_forward = W_fp + stop_gradient(W_q - W_fp)
```

Activation quantization:

```text
For each token vector x[b,t]:
  qmax = 2^(activation_bits - 1) - 1
  beta[b,t] = max_i(abs(x[b,t,i])) / qmax + eps
  x_int = clamp(round(x / beta), -qmax, qmax)
  x_q = x_int * beta
  x_forward = x + stop_gradient(x_q - x)
```

Linear output:

```text
y = x_forward @ W_forward.T + bias
```

Packed export:

```text
code(0)  = 0
code(+1) = 1
code(-1) = 2
pack four 2-bit codes per uint8:
  byte = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
store:
  packed_weight bytes
  padding count in [0, 3]
  weight_shape
  per-output-row alpha
  optional bias
  activation_bits
```

Training cautions:

- Do not replace embeddings, norms, or scalar router/load buffers with
  `BitLinear`.
- Keep optimizer state on the shadow weights.
- Lite defaults may need lower LR or longer warmup than Heavy.
- `activation_bits` must be at least 2; one-bit signed activation quantization
  makes `qmax = 0` and is not a valid b1.58 activation path.

## Long-Context Adaptation

**Source:** TTT-E2E.

**Core idea:** Treat long context as test-time training data and adapt a
request-local memory path while preserving base knowledge.

**Alexandros adaptation:** `TTTMetaAdapter` owns learned low-rank `phi_0`
initialization weights. Meta-training clones them into differentiable
request-local fast weights, unrolls prefix-chunk next-token updates, and applies
the adapted residual path to heldout/future chunks. `TTTState` remains the
inference/request-local state container.

**V1 simplification:** The reference path trains the low-rank TTT
meta-initialization and can optionally update checkpoint parameters through the
outer loss, but it does not implement optimized associative updates or
production long-context kernels.

**Risks:** Request-local updates can drift or leak between requests if not
carefully isolated.

**Tests:** Applying TTT changes outputs while all model parameters remain
byte-identical by default; the meta-training script saves only learned
`TTTMetaAdapter` components and never saves request-local fast state.

### Implementation Notes

TTT-E2E has two loops: a training-time meta-learning loop and an inference-time
request-local update loop. Alexandros implements a reference low-rank version of
both loops.

Training-time meta-learning target:

```text
Given a long sequence split into chunks c_1...c_n:
  theta       = checkpoint parameters
  phi_0       = TTTMetaAdapter.init_a/init_b learned fast-weight parameters

  for chunk j in prefix chunks:
      loss_inner_j = next_token_loss(theta, phi_{j-1}, c_j)
      phi_j = phi_{j-1} - inner_lr * grad_phi(loss_inner_j)

  loss_outer = next_token_loss(theta, phi_n, heldout/future chunk)
  update phi_0, and optionally theta when --trainable-scope enables checkpoint
  parameters, through the unrolled inner updates
```

Inference-time update loop:

```text
For each request:
  phi = clone_or_zero_fast_state(theta)   # request-local only

  for context chunk c_j before generation:
      loss_j = next_token_loss(theta, phi, c_j)
      phi = stop_checkpoint_mutation(phi - inner_lr * grad_phi(loss_j))

  decode next tokens with model(theta, phi)
  discard phi when the request ends
```

Reference meta-training loop:

```text
base_hidden = model.backbone(long_context_tokens)
prefix_chunks, heldout_chunk = split(base_hidden, tokens, prefill_chunk_len)
fast_a, fast_b = ttt_meta.initial_fast_weights()

for hidden_j, token_ids_j in prefix_chunks:
    update = ttt_meta.inner_update(
        hidden_j, token_ids_j, lm_head, fast_a, fast_b, inner_lr
    )
    fast_a, fast_b = update.fast_a, update.fast_b

outer_loss = next_token_loss(
    lm_head(ttt_meta.apply(heldout_hidden, fast_a, fast_b)),
    heldout_tokens,
)
outer_loss.backward()
optimizer.step()
```

Alexandros constraints:

- Never mutate checkpoint weights during request adaptation.
- Never share `TTTState` across requests.
- Use `TTTState.reset()` at request boundaries and `prefill_update(...)` for
  request-local fast-weight updates.
- Use `TTTMetaAdapter.request_state()` when a learned `phi_0` initialization
  should seed request-local inference state without sharing tensors.
- Prefer a local `torch.Generator` for request adaptation when deterministic
  replay matters; the reference update accepts `generator=...`.
- Use `apply(..., gate=...)` for bounded base/adapted mixing; gates are scalar
  or token-wise values in `[0, 1]`.
- Reject TTT in diffusion forwards until a denoising-compatible objective is
  specified.
- Save only learned TTT modules in checkpoints, not request-local fast state.

Acceptance criteria:

- `scripts/meta_train_ttt.py` saves `checkpoint/ttt_meta_adapter.pt` containing
  learned `init_a`/`init_b` and optimizer state.
- CI covers differentiable inner updates, no base-weight mutation, many-update
  drift, bounded fast-state size, and a 32K-token chunked prefill mechanism
  check.
- Retrieval quality remains a downstream benchmark because the repository does
  not ship training data or trained checkpoints.

## TurboQuant-Style KV Cache Compression

**Sources:** TurboQuant paper and Google Research post.

**Core idea:** Compress cache vectors online using random rotation plus scalar
quantization, with optional QJL residual sign correction for inner-product
quality.

**Alexandros adaptation:** `TurboQuantKVCache` provides a reference compressor
with deterministic random orthogonal rotations and optional residual sign
metadata. `MLAAttention` can store compressed AR `c_kv` cache packets behind
`use_turboquant_cache` and decompress them on attention reads. Packets store a
small rotation seed, while the compressor reconstructs and reuses the rotation
matrix locally.

**V1 simplification:** QJL signs are consumed by a reference cache-level
attention-score estimator, but the main MLA path still decompresses `c_kv`
before learned key/value projections. A fused MLA/TurboQuant kernel remains
future work.

**Risks:** Quality and speed claims require optimized kernels and benchmark
reproduction.

**Tests:** Round-trip shape, finite error, deterministic compression,
compressed-cache decode, compressed-vs-uncompressed tiny-logit comparison, and
smaller payload estimates than FP16/FP32 cache.

### Implementation Notes

Reference scalar-quantized rotation path:

```text
Input vector x in R^d
R = random_orthogonal_matrix(d, seed)
z = x @ R

qmax = 2^(bits - 1) - 1
scale = max(abs(z)) / qmax
q = clamp(round(z / scale), -qmax, qmax)
z_hat = q * scale
x_hat = z_hat @ R.T
```

The paper target uses random rotations to make coordinates near-independent and
then applies scalar quantization. Production kernels may replace the dense QR
matrix with a structured random transform such as random signs plus
Hadamard-like rotations.
`scripts/benchmark.py` reports dense QR rotation timing next to a structured
sign/permutation candidate before any reference compressor replacement is made.

QJL residual correction target:

```text
residual r = z - z_hat
P          = structured sign transform
sign_bits  = sign(r * P)                  # 1-bit residual metadata

For inner products:
  <query, key> = <query @ R, key @ R> because R is orthogonal
  estimate scores with:
      <query_rotated, key_z_hat + residual_estimate>

  residual_estimate =
      (sign_bits * P) * residual_norm / sqrt(nonzero_sign_count)
```

Alexandros stores QJL metadata (`qjl_sign`, `qjl_projection_seed`, and
`qjl_residual_norm`) and `TurboQuantKVCache.estimate_attention_scores(...)`
uses it in a reference dot-product estimator. The estimator is intentionally
cache-local: it scores full query vectors against compressed key vectors. The
MLA module still decompresses `c_kv` on the hot path because its learned
key/value projections happen after the compressed latent cache. Until a fused
kernel exists:

- compressed decode uses `x_hat = (q * scale) @ R.T`;
- QJL signs must validate shape and values `{-1, 0, +1}`;
- residual norms must validate shape, finiteness, and non-negativity;
- packet format versions above the implemented version must fail cleanly;
- metrics should report reconstruction error and cache byte estimates; and
- attention-score quality tests should compare full-cache dot products,
  scalar-quantized dot products, and QJL-corrected dot products on synthetic
  vectors.

## OpenMythos Inspiration

**Sources:** kyegomez/OpenMythos and its class docs.

**Core idea:** Recurrent-depth models can spend adaptive compute on harder
tokens through a prelude/recurrent/coda structure, loop-index embeddings,
depth-wise LoRA, and ACT halting.

**Alexandros adaptation:** Alexandros remains a hybrid diffusion/SSM/MoE model,
but includes an optional prelude/recurrent/coda staged layer stack, optional
loop-index embeddings for non-diffusion recurrent-stack forwards, an optional
`AdaptiveDepthController` for refinement loops, and OpenMythos-style docs,
configs, scripts, and tests.

**V1 simplification:** The uniform stack remains the default. The
prelude/recurrent/coda path is opt-in, repeats the configured middle segment a
fixed number of times, and rejects cache reuse when `recurrent_depth > 1`
instead of defining repeated attention-cache semantics. Loop-index embeddings
are disabled for diffusion forwards until denoising-specific semantics are
tested. Depth-wise LoRA is implemented in the adaptive-depth controller as
per-loop low-rank adapters; applying those adapters directly inside the staged
backbone recurrent segment remains future work.

**Risks:** Recurrence can destabilize training if not bounded.

**Tests:** Halting probability is bounded, max iterations are respected, the
controller preserves tensor shapes, staged stacks preserve output shapes, cache
reuse is rejected for repeated recurrent stacks, and loop-index embeddings are
disabled for diffusion forwards.

### Implementation Notes

OpenMythos is not a dependency. Use it as a design pattern for optional
recurrent-depth control.

Prelude/recurrent/coda option:

```text
hidden = embeddings(input_ids)

for layer in prelude_layers:
    hidden = layer(hidden)

for loop_idx in 0..recurrent_depth-1:
    if use_loop_index_embeddings and not diffusion_forward:
        hidden = hidden + loop_index_embedding[loop_idx]
    for layer in recurrent_layers:
        hidden = layer(hidden)

for layer in coda_layers:
    hidden = layer(hidden)
```

Cache boundary:

```text
if recurrent_depth > 1 and (use_cache or past_cache_present):
    reject("cache reuse is not supported for repeated staged stacks")
```

ACT-style halting bookkeeping:

```text
halting_sum_i = 0
remainder_i = 0
n_updates_i = 0
output_i = 0

for loop k:
    p_i = sigmoid(halting_head(hidden_i))
    still_running_i = halting_sum_i < act_threshold
    new_halted_i = still_running_i and halting_sum_i + p_i >= act_threshold

    update_weight_i =
        p_i                         if still_running and not new_halted
        act_threshold - halting_sum_i if new_halted
        0                           otherwise

    output_i += update_weight_i * hidden_i
    halting_sum_i += update_weight_i
    n_updates_i += still_running_i
```

Depth-wise LoRA option:

```text
for recurrent loop k:
  rank_k = depth_lora_ranks[k] if provided else depth_lora_rank
  delta_W_k = A_k @ B_k
  adaptive_update_k(x) = base_transition(x) + x @ delta_W_k
```

Acceptance criteria:

- `max_depth_iters` is a hard upper bound.
- Average loop count is reported through `AdaptiveDepthController.last_stats`.
- Token-wise early exit masks halted tokens out of later ACT accumulator updates
  without changing tensor shapes.
- `act_ponder_cost` is a non-negative config knob; the reference controller
  records `ponder_cost = act_ponder_cost * mean(n_updates)` as a diagnostic.
- Recurrent loops must define their interaction with diffusion timestep
  conditioning before being enabled for diffusion modes.
