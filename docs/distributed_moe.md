# Distributed MoE Planning

Alexandros v1 keeps MoE execution local and reference-oriented. Distributed
expert parallelism is future work and must preserve the local `MoEFeedForward`
contract.

## Expert Parallelism Interface

Future expert-parallel implementations should expose:

```text
dispatch(hidden_states, selected_experts, expert_weights)
combine(expert_outputs, combine_indices, expert_weights)
```

Inputs:

- `hidden_states`: `[batch, sequence, hidden_size]`
- `selected_experts`: `[batch, sequence, top_k]`
- `expert_weights`: `[batch, sequence, top_k]`

Outputs:

- combined routed result: `[batch, sequence, hidden_size]`
- diagnostics matching local expert-load accounting.

Shared experts remain always-active and can run locally or be replicated.

## Dispatch Tensor Contract

Flatten token routes before all-to-all:

```text
route_tokens      [num_routes, hidden_size]
route_experts     [num_routes]
route_weights     [num_routes]
route_batch_index [num_routes]
route_seq_index   [num_routes]
route_topk_index  [num_routes]
```

`num_routes = batch * sequence * top_k` before optional expert-local filtering.
Routes must preserve enough index metadata to scatter results back to the
original `[batch, sequence, top_k]` layout.

## All-To-All Requirements

A distributed implementation must define:

- expert-to-rank assignment;
- route sorting/grouping by destination rank;
- all-to-all send and receive tensor shapes;
- padding policy for uneven route counts;
- deterministic behavior when seeds and inputs match;
- failure behavior when a rank owns no routes for a batch.

## Weighted Scatter/Combine Semantics

Each routed expert output is multiplied by its normalized router weight before
accumulating back into `[batch, sequence, hidden_size]`.

Required parity checks:

- local dense routed experts vs sparse local dispatch;
- sparse local dispatch vs distributed dispatch on one process group;
- expert load stats unchanged by dispatch backend;
- gradients flow to selected expert parameters and router parameters.

## Documentation Gate

Do not land distributed MoE code without:

- shape diagrams for all communication tensors;
- device/dtype support table;
- seed and checkpoint consistency tests;
- clear fallback to local reference MoE.
