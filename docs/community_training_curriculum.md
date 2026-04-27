# Community Training Curriculum Outline

Alexandros does not publish an official full-scale training curriculum in v1.
This outline documents the sequence downstream trainers should decide and
report if they train a checkpoint.

## Phase 1: AR Pretraining

Decisions to publish:

- tokenizer and vocabulary;
- corpus list, licenses, filtering, deduplication, and PII policy;
- sequence length schedule;
- optimizer, LR schedule, batch size, precision, and hardware;
- stopping criteria, validation split, and loss curves;
- checkpoint cadence and resume policy.

Mechanism already covered by the repo:

- causal LM loss;
- gradient accumulation;
- mixed precision;
- validation loop;
- metrics JSONL;
- checkpoint save/resume;
- tokenizer metadata attachment.

## Phase 2: Diffusion Objective

Downstream trainers should decide whether diffusion is:

- warm-started from an AR checkpoint;
- trained jointly;
- trained from scratch;
- used only as a generation head or auxiliary objective.

Report:

- masking schedule;
- `diffusion_loss_weighting`;
- denoising mask mode;
- block size and remasking policy;
- expert-load behavior by timestep.

## Phase 3: Latent Reasoning

Decide whether latent reasoning uses:

- visible trace token IDs;
- synthetic/task traces;
- latent-only reconstruction after trace warmup;
- no trace data, using only hidden-state reconstruction.

Report:

- trace data source and license;
- `lambda_kl`, `lambda_rec`, and latent refinement steps;
- whether latent modules are trained alone or with the backbone;
- reconstruction metrics and downstream task deltas.

## Phase 4: TTT Meta-Training

The v1 repo includes request-local TTT mechanism probes, not a learned
meta-training implementation. A full TTT curriculum should define:

- fast-weight parameters;
- inner update objective;
- outer heldout/future chunk objective;
- unroll length;
- checkpoint format for any learned TTT components;
- long-context acceptance tests.

## Phase 5: Optional Distillation

Distillation may use user-provided teacher token targets or dense teacher
logits. Downstream trainers must verify:

- teacher model license and terms;
- generated-output rights;
- redistribution constraints;
- temperature and loss weights;
- whether teacher outputs are stored, regenerated, or discarded.

## Phase 6: Optional Post-Training

Instruction tuning, preference optimization, and tool-use training are out of
scope for v1. If a downstream release adds them, follow
`docs/post_training_scope.md` and `docs/safety_evaluation.md`.
