# Alexandros Master Design

Alexandros is a research-grounded language model prototype that combines sparse
expert capacity, long-context recurrent processing, latent reasoning, and
diffusion-style generation. It ships in two flavors: Heavy for quality and Lite
for BitNet-style efficiency.

## Goals

- Provide a runnable PyTorch prototype that validates the architecture at tiny
  scale.
- Keep public interfaces close to Hugging Face conventions: config classes,
  `save_pretrained`, `from_pretrained`, and model-specific heads.
- Embed enough research context for another LLM or engineer to continue without
  browsing papers.
- Publish a trainable architecture and industry-standard training mechanisms so
  downstream trainers can choose their own tokenizer, data, scale, and
  curriculum.
- Keep optimized serving paths out of v1 while preserving interfaces where
  optimized kernels can later plug in.

## Non-Goals For V1

- Frontier-scale distributed training.
- Triton/vLLM fused kernels.
- Packed ternary inference.
- Full reproduction of every cited paper.
- A production tokenizer, curated pretraining corpus, internet data gathering,
  or official large-scale training run.
- Official instruction tuning, preference tuning, or tool-use training data.

## Model Family

### Variants

`variant="heavy"` uses ordinary dense `torch.nn.Linear` modules and is intended
for quality-focused experiments.

`variant="lite"` replaces model linear projections with `BitLinear`, which uses
native BitNet b1.58-style ternary weights during forward passes while preserving
full-precision shadow weights for training.

### Public Classes

- `AlexandrosConfig`: serializable model configuration.
- `AlexandrosModel`: embedding layer, hybrid backbone, final norm, and optional
  hidden-state, attention-cache, and SSM-state outputs.
- `AlexandrosForCausalLM`: autoregressive training and greedy generation.
- `AlexandrosForDiffusionLM`: masked/block diffusion and latent reasoning entry
  points.
- `GenerationMode`: enum with `autoregressive`, `block_diffusion`,
  `latent_reasoning`, and `hybrid`.

### Backbone

Each layer is an `AlexandrosBlock`:

1. RMSNorm.
2. Sequence mixer:
   - `MLAAttention` on scheduled attention layers.
   - `GatedDeltaNetBlock` on linear-time layers by default.
   - `MatrixGatedDeltaNetBlock` when
     `linear_mixer_backend="matrix_deltanet"`.
   - `Mamba2Block` when `linear_mixer_backend="mamba2"`.
3. Residual connection.
4. RMSNorm.
5. `MoEFeedForward`.
6. Residual connection.

The default schedule is three linear mixer layers followed by one MLA layer.
This follows the Jamba/Qwen/Kimi family of hybrid long-context designs while
keeping the v1 code compact. The matrix DeltaNet option is a paper-style
per-head fast-weight reference backend with causal chunking. The Mamba-2 option
is a local diagonal SSM reference backend, not an optimized `mamba-ssm` kernel.

`config.dropout` is applied to MLA attention probabilities and residual branch
outputs from the sequence mixer and MoE FFN. Dropout is train-only; eval mode is
deterministic.

### Attention

`MLAAttention` compresses hidden states into a rank-sized KV latent `c_kv`, then
reconstructs keys and values for scaled dot-product attention. Its cache stores
the compressed latent rather than full K/V tensors. When `mla_rope_dim > 0`,
the cache also stores a separate positional `k_rope` tensor so the compressed
latent remains position-free.

Cache-size accounting uses derived config terms:
`standard_mha_elements_per_token = 2 * heads * head_dim` and
`mla_elements_per_token = mla_d_c + mla_d_r`. `mla_d_c = kv_lora_rank`,
`mla_d_r = mla_rope_dim`, and `mla_d_nope = head_dim - mla_d_r`; the default
`mla_rope_dim = 0` keeps older tiny configs compact.

In AR forwards, MLA always uses a causal mask. In diffusion forwards,
`diffusion_attention_mask_mode` chooses whether MLA remains causal or becomes
bidirectional for the current denoising pass. Cache reuse is rejected for
diffusion forwards because masked positions can be revised together.

### MoE

`MoEFeedForward` contains:

- Shared always-active SwiGLU experts.
- Fine-grained routed SwiGLU experts.
- Normalized sigmoid top-k routing.
- Non-gradient router-bias load balancing using a non-persistent multi-batch
  expert-load EMA and configurable update cadence.
- Optional diffusion timestep, masked/unmasked token-state, and position-bucket
  routing biases.

### Diffusion

`MaskedDiffusionScheduler` creates absorbing-mask training examples.
`AlexandrosForDiffusionLM.generate_block_diffusion` appends a block of mask
tokens, repeatedly predicts all appended masked positions in parallel, and
commits token predictions over a small number of steps. The reference sampler
supports greedy or sampled token selection, while prompt tokens are always
preserved. Bidirectional diffusion is implemented as an MLA attention-mask mode;
Gated DeltaNet layers remain directional in the reference hybrid backbone.

### Latent Reasoning

`LatentThoughtVAE` compresses hidden states into latent slots. The
`LatentDiffusionReasoner` refines those slots iteratively with bidirectional
slot self-attention, timestep conditioning, and a clipped feed-forward update.
The optional `AdaptiveDepthController` adds OpenMythos-inspired repeated latent
computation with ACT-style halting.

### Long Context

`TTTState` is request-local. It may be created during prefill, updated from
hidden states with `prefill_update(...)`, reset between requests, and applied
as a gated low-rank fast-weight residual. It must never mutate model checkpoint
parameters.

### KV Compression

`TurboQuantKVCache` is a reference compressor for cache tensors. It uses
deterministic random orthogonal rotations and scalar quantization. Optional QJL
residual signs/norms feed a cache-level attention-score estimator for synthetic
dot-product tests. When `use_turboquant_cache=True`, AR MLA cache writes store
compressed `c_kv` packets and attention reads decompress them in the reference
path before learned key/value projections. Packets carry a rotation seed rather
than the full rotation matrix; the compressor reconstructs the shared rotation
when needed. `scripts/benchmark.py` times this dense QR rotation path against a
structured sign/permutation candidate before any serving-oriented rotation
replacement is considered.
If `mla_rope_dim > 0`, TurboQuant compresses only `c_kv`; `k_rope` remains an
ordinary floating-point cache tensor until a positional-cache compressor is
specified and tested.

## Training And Data Scope

Alexandros is intended to be published as a reference architecture and training
mechanism, not as a project-owned corpus or official trained checkpoint. The
repo should prove that the model can train, resume, validate, export, and report
metrics using synthetic data, tiny permissively licensed samples, or
user-supplied token-id JSONL. It must not include web-scale scraping or imply
that bundled smoke fixtures are suitable for model quality.

Downstream trainers are expected to choose:

- tokenizer family and vocabulary;
- training corpus and licenses;
- AR/diffusion/latent/TTT curriculum;
- teacher model, if distillation is attempted;
- post-training and safety datasets.

Optional distillation is supported as a mechanism interface through
user-supplied teacher token targets or dense teacher logits, but any teacher
model must be user supplied and license-checked by the trainer before publishing
derived checkpoints.

## Training Mechanism Phases

1. **AR mechanism validation:** train `AlexandrosForCausalLM` on synthetic,
   sample, or user-supplied token IDs to validate loss, optimizer updates,
   checkpointing, and resume.
2. **Diffusion mechanism validation:** train masked token reconstruction with
   timestep conditioning and timestep-aware MoE routing.
3. **Latent mechanism validation:** train latent VAE reconstruction and
   iterative latent refinement on hidden states or optional user-supplied trace
   examples following `docs/latent_trace_format.md`.
4. **TTT mechanism validation:** train or simulate fast-weight update rules on
   synthetic or user-supplied long-context chunks.
5. **Export:** save config, model weights, and Alexandros metadata.

## Acceptance Criteria

- Heavy tiny and Lite tiny instantiate and run forward passes.
- AR generation returns tokens with expected shape.
- Block diffusion generation fills mask tokens.
- Latent reasoning path changes output when refinement steps change.
- TTT state changes hidden activations without changing model parameters.
- TurboQuant round-trips tensors with finite reconstruction error.
- TurboQuant-compressed AR MLA cache decodes with finite logits and remains near
  the uncompressed cache path on tiny configs.
- Synthetic long-context probes report finite needle, lost-in-the-middle, copy,
  and recurrent-state drift diagnostics on tiny configs.
- Cached AR generation matches uncached greedy generation on tiny configs.
- `save_pretrained` and `from_pretrained` round-trip a tiny model.

## Known V1 Simplifications

- MoE uses a clear PyTorch sparse dispatch path, not fused expert-parallel
  kernels.
- The default Gated DeltaNet backend uses an elementwise recurrent state for
  compact CPU tests. `matrix_deltanet` provides the paper-style per-head matrix
  recurrence, but still uses a readable PyTorch loop rather than an optimized
  associative scan kernel.
- MLA reconstructs dense K/V before attention.
- Diffusion sampling uses simple confidence commits rather than learned
  remasking/confidence schedules.
- TTT uses a reference low-rank meta-adapter and differentiable inner-loop
  unroll, but not optimized production long-context kernels.
- TurboQuant QJL metadata is consumed by a reference cache-level dot-product
  estimator, but not by the MLA hot path or optimized kernels.

## File Map

```text
src/alexandros/configuration_alexandros.py  # config serialization
src/alexandros/modeling_alexandros.py       # public model classes
src/alexandros/attention_mla.py             # MLA attention
src/alexandros/ssm_gated_deltanet.py        # compact recurrent mixer
src/alexandros/ssm_matrix_deltanet.py       # matrix-state DeltaNet mixer
src/alexandros/moe.py                       # shared+routed MoE
src/alexandros/bitlinear.py                 # Lite ternary linear layer
src/alexandros/diffusion.py                 # masked/block diffusion utilities
src/alexandros/latent_reasoning.py          # VAE and latent refinement
src/alexandros/adaptive_depth.py            # ACT loop controller
src/alexandros/ttt.py                       # request-local fast weights
src/alexandros/kv_cache.py                  # TurboQuant reference cache
```

## Testing Strategy

Unit tests cover tensor shapes, routing behavior, quantization, latent modules,
TTT isolation, cache compression, and save/load. Integration smoke tests check
generation and tiny training paths. Performance benchmarks are intentionally
separate from correctness tests.
