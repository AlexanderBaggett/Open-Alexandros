# Training Objective Contracts

Alexandros smoke trainers are mechanism tests, not full training recipes. Each
trainer emits `objective_*` fields in `metrics.jsonl` so a downstream trainer
can verify the objective that was exercised.

These contracts are also available in code via
`alexandros.training.objective_contract(phase)` and
`alexandros.training.objective_log_fields(phase)`.

## Shared Constant

- `IGNORE_INDEX = -100` for token positions excluded from cross entropy.

## AR

- Phase: `ar`
- Objective: `causal_next_token`
- Inputs: `input_ids[batch, sequence]`
- Targets: `labels=input_ids` shifted left by one token inside
  `AlexandrosForCausalLM.forward`.
- Ignore behavior: `IGNORE_INDEX` labels are skipped after the shift.
- Normalization: mean cross entropy over non-ignored next-token targets.
- Smoke stopping criteria: `--steps` optimizer steps.
- Data behavior: AR token-id JSONL prepends BOS and appends EOS during packing.

## Diffusion

- Phase: `diffusion`
- Objective: `absorbing_masked_or_rao_blackwellized_token`
- Inputs: absorbing-mask-corrupted `noisy_input_ids` plus `diffusion_timestep`.
- Targets: original `input_ids` at masked positions for the default objective;
  all non-pad target positions for `diffusion_objective="rao_blackwellized"`.
- Ignore behavior: unmasked positions use `IGNORE_INDEX` in the default masked
  objective.
- Normalization: default masked objective divides weighted CE by
  `masked_token_count`; Rao-Blackwellized objective divides weighted
  target-position CE by non-pad target count.
- Smoke stopping criteria: `--steps` optimizer steps.
- V1 scope: masked-token objective with `uniform` default weighting,
  `mask_prob`/`inverse_mask_prob` research knobs, and optional
  Rao-Blackwellized target-position expansion.

## Latent

- Phase: `latent`
- Objective: `latent_reconstruction_kl`
- Inputs: backbone hidden states pooled into latent VAE slots. With
  `--trace-jsonl`, inputs come from user-supplied already-tokenized
  `input_ids`.
- Targets: pooled hidden-state reconstruction proxy. With `--trace-jsonl`, the
  reconstruction target is a masked pool of the backbone hidden states produced
  from `trace_ids`.
- Normalization:

```text
loss = lambda_rec * mean(
         mse(vae_reconstruction, pooled_hidden_target),
         mse(refined_latent_reconstruction, pooled_hidden_target)
       )
     + lambda_kl * kl_loss
```

- Smoke stopping criteria: `--steps` optimizer steps.
- V1 scope: validates VAE and latent-refinement mechanics only. Visible
  reasoning trace data is user-supplied; bundled trace fixtures are synthetic
  CI-only examples.

## TTT

- Phase: `ttt`
- Objective: `request_local_next_token_probe`
- Inputs: long-context hidden chunks, learned `TTTMetaAdapter` `phi_0` fast
  weights, and request-local inner-loop fast-weight updates.
- Targets: next tokens on prefix chunks for inner updates and heldout/future
  chunks for the outer loss.
- Normalization: mean cross entropy over next-token targets per chunk.
- Smoke stopping criteria: `--steps` outer optimizer steps.
- V1 scope: base checkpoint parameters are frozen by default while
  `TTTMetaAdapter.init_a`/`init_b` train through the unrolled inner updates.
  Explicit trainability scopes may update checkpoint parameters too. Learned
  TTT components are saved in `checkpoint/ttt_meta_adapter.pt`; request-local
  `TTTState` fast tensors are never saved into model checkpoints.

## Distillation

- Phase: `distillation`
- Objective: `teacher_token_or_logit_distillation`
- Inputs: `input_ids` plus user-supplied `teacher_token_ids` and/or
  `teacher_logits`.
- Targets: generated teacher token targets, dense teacher probability
  distributions, or both.
- Normalization: masked mean teacher-token cross entropy plus
  temperature-scaled masked mean KL over teacher logits.
- Smoke stopping criteria: downstream trainer defined; the repository provides
  mechanism helpers and tiny fixtures only.
- V1 scope: Alexandros does not choose a teacher model. Contributors must verify
  teacher model license, terms, generated-output rights, and redistribution
  constraints before publishing distilled checkpoints.
