# Post-Training Scope

Alexandros v1 is published as a base-model architecture and training-mechanism
prototype. It does not include official instruction tuning, preference
optimization, tool-use training, or agentic training.

## Base-Model Boundary

The repository owns:

- architecture modules and reference forward passes;
- AR, diffusion, latent, TTT, and optional distillation mechanism helpers;
- smoke trainers that prove optimizer, resume, validation, metric logging, and
  checkpoint export work;
- tokenizer, fixture, checkpoint, and evaluation contracts.

Downstream trainers own:

- production tokenizer choice;
- corpus selection and licenses;
- large-scale pretraining curriculum;
- teacher-model selection and distillation rights;
- instruction/chat data;
- preference/alignment data;
- safety and post-training release gates.

## Instruction Tuning

Instruction tuning is out of scope until a downstream project chooses a
tokenizer, chat template, dataset, and release policy. A future SFT path should
define:

- chat template and special tokens;
- JSONL data schema;
- supervised loss mask semantics;
- trainability scope;
- evaluation for instruction following and hallucination;
- safety and license gates before publishing an instruction-tuned checkpoint.

## Preference And Alignment Training

Preference training is out of scope until post-training is explicitly accepted.
A future plan should choose a method such as DPO, IPO, ORPO, or RL-based
optimization, then define:

- preference data format and allowed sources;
- refusal/safety data policy;
- over-refusal and unsafe-compliance evaluation;
- privacy and memorization probes;
- release gates for aligned checkpoints.

## Tool-Use And Agentic Training

Tool-use training is out of scope until a coding-agent release goal exists. A
future plan should define:

- tool call schema;
- environment feedback loop;
- verifiable coding-task datasets;
- terminal/SWE-style evaluation harness;
- sandbox and safety controls.
