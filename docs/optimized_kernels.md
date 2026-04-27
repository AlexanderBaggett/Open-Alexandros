# Optimized Kernel Roadmap

Optimized kernels are future work. The reference PyTorch implementation remains
the correctness oracle for every accelerated path.

## Shared Acceptance Criteria

Every optimized path must:

- be optional and disabled by default;
- preserve eager CPU/PyTorch fallback behavior;
- match reference outputs within documented tolerances;
- reject unsupported shapes instead of silently changing semantics;
- include benchmark commands with hardware, dtype, config hash, and dependency
  versions;
- keep checkpoint formats compatible with `from_pretrained`.

## Matrix DeltaNet Scan Plan

Current status:

- `linear_mixer_backend="matrix_deltanet"` implements the per-head matrix
  Gated DeltaNet recurrence in PyTorch.
- `deltanet_chunk_size` provides exact causal chunking for prefill and cached
  decode tests.

Future work:

- replace the Python chunk loop with an associative/SSD-style scan kernel;
- benchmark long-prefill throughput against the reference recurrence;
- preserve chunked-vs-full and cached-decode parity tests.

## Triton MLA Attention Plan

Target:

- fused MLA projection/cache read where possible;
- paper-faithful `c_kv` plus optional `k_rope` layout;
- causal and bidirectional MLA masks;
- optional TurboQuant cache read after the attention-score estimator is defined.

Required tests:

- eager-vs-kernel logits on tiny configs;
- cached-vs-uncached generation parity;
- right-padding and invalid-mask rejection;
- TurboQuant disabled/enabled parity when supported.

## SDPA Attention Plan

Current status:

- `attention_backend="sdpa"` exists as an optional PyTorch SDPA path.
- `attention_backend="flash"` exists as an explicit PyTorch Flash Attention
  request through `torch.nn.attention.sdpa_kernel`; it raises on CPU or
  unsupported CUDA shapes instead of silently changing semantics.
- Eager-equivalence tests cover full and cached attention.
- CPU fallback remains eager-compatible.

Future work:

- broader dtype/device coverage;
- benchmark reporting for SDPA/Flash vs eager across sequence lengths.

## Sparse MoE Dispatch Plan

Target:

- sparse dispatch for top-k routed experts without computing every expert;
- explicit weighted combine back to `[batch, sequence, hidden_size]`;
- compatibility with shared experts and timestep-aware routing diagnostics.

Required tests:

- sparse-vs-dense MoE parity with copied weights;
- expert load accounting unchanged;
- invalid dispatch indices rejected;
- eventual distributed all-to-all contract if expert parallelism is added.

## Packed BitLinear Plan

Target:

- runtime path that consumes `packed_bitlinear.pt` or an equivalent packed
  checkpoint representation;
- ternary weights with row scales;
- activation scaling consistent with reference `BitLinear`.

Required tests:

- packed-vs-reference logits;
- save/load round trip;
- CPU-only Lite inference path;
- clear distinction from Microsoft `bitnet.cpp` ABI unless a converter exists.

## TurboQuant Attention Read Plan

Target:

- port the reference `TurboQuantKVCache.estimate_attention_scores(...)`
  estimator into a fused attention read path;
- support QJL residual correction with the documented sign/norm estimator;
- no full decompression on the hot path.

Required tests:

- full-cache dot products vs scalar-quantized estimator;
- QJL-corrected estimator improves or bounds error on synthetic comparisons;
- compressed-vs-uncompressed generation drift stays within documented limits.
