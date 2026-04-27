# Serving Integration Plan

Alexandros v1 does not ship a production serving backend. This plan documents
what must be true before claiming compatibility with common inference systems.

## Current CPU-Only Lite Path

Supported today:

- instantiate Lite configs with reference `BitLinear`;
- run generation through PyTorch;
- export reference checkpoints with `save_pretrained`;
- export `packed_bitlinear.pt` as an interchange artifact.

Not supported today:

- packed BitLinear runtime kernels;
- bitnet.cpp-compatible binaries;
- sparse MoE serving kernels;
- vLLM or Text Generation Inference adapters.

## vLLM Compatibility Investigation

Before claiming vLLM support, define:

- model config registration strategy;
- attention/cache interface for MLA `c_kv` and optional `k_rope`;
- recurrent state handling for elementwise, matrix DeltaNet, and Mamba-2
  mixer layers;
- MoE routing and sparse expert dispatch path;
- Lite BitLinear kernel strategy;
- supported generation modes. Block diffusion and latent/hybrid generation may
  require custom scheduler APIs beyond standard AR serving.

Required tests:

- reference PyTorch generation parity on tiny checkpoints;
- cache reorder and batching behavior;
- unsupported mode errors.

## Text Generation Inference Compatibility Investigation

Before claiming TGI support, define:

- custom model loading path;
- tokenizer metadata handling;
- AR cache representation;
- streaming support boundaries;
- unsupported diffusion/latent mode behavior.

Required tests:

- load reference checkpoint;
- generate AR continuations matching PyTorch greedy decode;
- reject unsupported modes clearly.

## bitnet.cpp Export Investigation

The current `packed_bitlinear.pt` is not a bitnet.cpp ABI. A real converter must
define:

- exact tensor layout and endian behavior;
- row-scale representation;
- activation quantization contract;
- metadata needed by bitnet.cpp;
- validation against reference Lite logits.

## Packed BitLinear Runtime Serving Path

Required before support:

- packed weight loader;
- CPU and/or GPU kernel;
- fallback to reference `BitLinear`;
- packed-vs-reference parity tests;
- benchmark reporting.

## Sparse MoE Runtime Serving Path

Required before support:

- sparse dispatch kernel or backend integration;
- shared+routed expert combine semantics;
- batching behavior;
- expert-load diagnostics;
- parity with dense reference dispatch.
