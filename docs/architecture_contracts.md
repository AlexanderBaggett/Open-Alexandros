# Alexandros Architecture Compatibility Contracts

This document defines which architectural features are valid together in the
current reference prototype. It is intentionally conservative: unsupported
combinations must fail clearly or remain absent from public APIs until tests
prove they are mathematically valid.

## Autoregressive Mode

Supported:

- Causal MLA attention.
- MLA `c_kv` cache reuse.
- Linear-mixer recurrent state reuse. `GatedDeltaNetBlock` and `Mamba2Block`
  states have shape `[batch, hidden_size]`; `MatrixGatedDeltaNetBlock` states
  have shape `[batch, heads, value_dim, key_dim]`. All states use the same
  dtype/device as the mixer hidden states, and returned states are detached
  before cache reuse.
- Optional prelude/recurrent/coda staged stack. The default uniform stack is
  unchanged. When `recurrent_depth > 1`, cache reuse is rejected because the
  repeated middle segment has no paper-faithful attention-cache contract yet.
- Optional TurboQuant compression of AR MLA cache packets.
- Optional request-local `TTTState` after the backbone forward.
- AR sampling controls, stop IDs, and stop sequences.

Unsupported:

- Left-padded current inputs in the reference causal cache path.
- Variable prompt-length batch continuation. Batch rows must share the same
  current prompt length until a dedicated continuation mask is implemented.

## Block Diffusion Mode

Supported:

- Absorbing-mask corruption with timestep conditioning.
- Masked-token loss normalization by masked-token count.
- Configurable diffusion MLA mask policy through
  `diffusion_attention_mask_mode`.
  - `causal`: default; diffusion forwards share the AR causal MLA mask.
  - `bidirectional`: MLA attention layers can see the full unmasked sequence
    during forwards with `diffusion_timestep`.
- Timestep-aware MoE router bias and per-timestep expert accounting.
- Block diffusion sampling with temperature, top-k, top-p, and stop-token
  handling.

Unsupported:

- Cache reuse. Diffusion may revise multiple positions, so AR KV/SSM cache
  semantics are not valid.
- Request-local `TTTState`. The current TTT sketch is causal/adaptive-state
  oriented and is not defined for denoising updates.
- Fully bidirectional Gated DeltaNet/SSM layers. The bidirectional diffusion
  mask applies to MLA attention layers; recurrent mixer layers remain
  directional in the reference hybrid backbone.
- Loop-index embeddings inside diffusion forwards. They are disabled until
  denoising-specific recurrent-depth semantics are tested.

## Latent Reasoning Mode

Supported:

- Latent slot encoding through `LatentThoughtVAE`.
- Iterative latent refinement through `LatentDiffusionReasoner`, including
  bidirectional self-attention across latent slots.
- Optional adaptive halting in `LatentDiffusionReasoner` when
  `latent_adaptive_threshold > 0`. The loop always runs at least one iteration,
  never mutates checkpoint weights, and records diagnostics in `last_stats`.
- Optional adaptive-depth refinement after the visible-token backbone.
- Decode through the regular LM head after latent refinement.

Unsupported:

- Cache reuse across latent refinement steps.
- Treating latent slots as auditable hidden chain-of-thought. The current
  latents are research activations, not faithful explanations.

## Hybrid Mode

The current hybrid order is:

1. Encode prompt tokens with the causal backbone.
2. Run latent refinement.
3. Decode through the LM head.
4. Use block diffusion for the remaining requested continuation tokens.

Unsupported:

- Mixing diffusion cache reuse with latent refinement.
- Applying request-local TTT inside diffusion denoising.
- Streaming block diffusion. Parallel denoising does not expose a stable
  token-by-token stream yet.

## Cache Semantics

- `past_key_values` are only for AR attention layers.
- `past_ssm_states` are only for AR recurrent mixer layers.
- `use_turboquant_cache=True` changes AR MLA cache storage from `c_kv` tensors
  to `TurboQuantPacket` entries under `c_kv_packet`. When `mla_rope_dim > 0`,
  cache entries also include uncompressed `k_rope` tensors.
- Cache memory estimates must use
  `standard_mha_elements_per_token = 2 * heads * head_dim` and
  `mla_elements_per_token = mla_d_c + mla_d_r`, where
  `mla_d_r = mla_rope_dim`.
- Diffusion, latent, and hybrid paths must reject or avoid cache arguments until
  a mode-specific correctness test exists.

## TTT Boundaries

`TTTState` is request-local and must never mutate checkpoint weights. The
learned `TTTMetaAdapter` can be trained by `scripts/meta_train_ttt.py`, but its
weights are checkpointed separately from per-request fast state. In V1 TTT is
valid only for non-diffusion causal forwards. Any future extension to diffusion
or latent training must define:

- Fast-weight reset/isolation rules per request.
- The update objective.
- Where adapted state is applied in the forward pass.
- Tests proving base checkpoint weights and unrelated requests are unchanged.
