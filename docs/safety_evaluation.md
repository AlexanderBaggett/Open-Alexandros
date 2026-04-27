# Safety Evaluation Plan

Alexandros v1 is a base-model architecture prototype, not an instruction-tuned
assistant. This plan becomes a release gate only if downstream work publishes an
instruction-tuned, preference-tuned, tool-using, or otherwise user-facing
checkpoint.

## Basic Harmful Request Evaluation

Instruction-tuned releases should evaluate:

- direct harmful requests;
- transformed harmful requests, such as summarization or translation of unsafe
  instructions;
- cyber, bio, chemical, self-harm, fraud, and weapons categories appropriate to
  the release context;
- benign refusal controls to detect over-refusal.

Reports should include prompts, scoring rubric, model version, decoding
settings, and reviewer process where disclosure is safe.

## Privacy Leakage Evaluation

Before release, evaluate:

- extraction of emails, phone numbers, addresses, credentials, and secrets;
- memorized training snippets when a downstream corpus is known;
- canary strings or synthetic secrets if the trainer inserted them;
- behavior on prompts asking for private data about named individuals.

Any release should document training-data privacy filtering and residual risk.

## Memorization And Copyright Probes

Evaluate:

- long verbatim continuation of known copyrighted or benchmark text;
- near-duplicate training examples when the downstream trainer can audit data;
- refusal or transformation behavior when asked to reproduce protected text;
- benchmark contamination risk.

Results should distinguish base-model memorization risk from instruction-tuned
policy behavior.

## Red-Team Notes And Release Gates

A user-facing release should have:

- documented red-team scope and dates;
- severity categories and examples;
- mitigations or accepted risks;
- final go/no-go owner;
- model card and evaluation report updates.

If these gates are not complete, release the checkpoint as research-only or do
not release it.
