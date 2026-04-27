# Alexandros Checkpoint Format

Alexandros reference checkpoints currently use format version `1`.

## Files

- `config.json`: serialized `AlexandrosConfig`.
- `pytorch_model.bin`: PyTorch `state_dict` weights.
- `model.safetensors`: optional safetensors weights when
  `safe_serialization=True`.
- `generation_config.json`: Hugging Face-style generation defaults for local
  export and downstream tooling.
- `tokenizer_metadata.json`: optional user-supplied tokenizer metadata. It is
  written only when `save_pretrained(..., tokenizer_metadata=...)` is called.
- `checkpoint_metadata.json`: optional versioned training/handoff metadata. It
  is written by smoke trainers and by direct
  `save_pretrained(..., checkpoint_metadata=...)` calls.
- `alexandros_extra.json`: Alexandros-specific metadata.

Smoke training runs can also write a separate `training_state.pt` file next to
the model checkpoint. This is a local resume artifact, not the portable model
checkpoint format.

## Version 1 Metadata

```json
{
  "checkpoint_format_version": 1,
  "format": "open-alexandros-reference",
  "model_class": "AlexandrosForDiffusionLM",
  "weights": "pytorch_model.bin",
  "safe_serialization": false,
  "preferred_future_weights": "model.safetensors",
  "hf_compatibility": {
    "transformers_required": false,
    "config_class": "AlexandrosConfig",
    "model_class": "AlexandrosForDiffusionLM",
    "auto_config_registered": false,
    "auto_model_registered": false,
    "requires_trust_remote_code": false
  },
  "tokenizer_metadata": null,
  "checkpoint_metadata": null
}
```

The `hf_compatibility` block is informational. Version `1` checkpoints use
Hugging Face-style filenames but do not require `transformers`. The optional
`alexandros.hf_compat` bridge provides `PretrainedConfig`/`PreTrainedModel`
wrappers and AutoClass registration when `.[hf]` is installed, but the
reference checkpoint format still does not require `trust_remote_code`.

When tokenizer metadata is supplied, `tokenizer_metadata` is
`"tokenizer_metadata.json"` and the referenced file uses:

```json
{
  "format": "open-alexandros-tokenizer-metadata",
  "format_version": 1,
  "tokenizer": {},
  "config": {
    "vocab_size": 32000,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "mask_token_id": 3
  }
}
```

The `tokenizer` object is user supplied. If it declares `vocab_size` or special
token IDs, `save_pretrained` validates them against `config.json`.

When checkpoint metadata is supplied, `checkpoint_metadata` is
`"checkpoint_metadata.json"` and the referenced file uses:

```json
{
  "format": "open-alexandros-checkpoint-metadata",
  "format_version": 1,
  "model_class": "AlexandrosForDiffusionLM",
  "checkpoint": {
    "phase_handoff_contract_version": 1,
    "phase": "diffusion",
    "objective": {},
    "trainability": {},
    "enabled_modules": ["model", "lm_head"],
    "disabled_modules": ["latent_vae", "latent_reasoner"],
    "migration": {}
  }
}
```

The smoke trainers write this metadata so downstream tooling can see which
training mechanism produced a checkpoint, which modules were trainable, and how
cross-phase handoff should treat modules that are absent or extra.

## Phase Handoff Loading

`from_pretrained()` is strict for same-class loads. Cross-class handoff is
allowed only for the known Alexandros phase transitions:

- `AlexandrosForCausalLM` checkpoint loaded as `AlexandrosForDiffusionLM`:
  backbone and LM-head weights are loaded, while missing `latent_vae.*` and
  `latent_reasoner.*` parameters are freshly initialized by the target class.
- `AlexandrosForDiffusionLM` checkpoint loaded as `AlexandrosForCausalLM`:
  backbone and LM-head weights are loaded, while extra `latent_vae.*` and
  `latent_reasoner.*` parameters are ignored.

Any other missing or unexpected state-dict key remains an error.

## Compatibility Policy

- Version `1` checkpoints must remain loadable through `from_pretrained`.
- `pytorch_model.bin` is kept as the default for compatibility with the current
  reference implementation.
- `model.safetensors` is supported when the optional `safetensors` package is
  installed and `safe_serialization=True` is requested.
- `transformers` is not required to read or write version `1` checkpoints.
  Optional Hugging Face wrappers live in `alexandros.hf_compat`; native
  Transformers checkpoints should still be validated separately from reference
  Alexandros checkpoints.
- `torch.load(..., weights_only=True)` is preferred when the installed PyTorch
  version supports it.
- Future checkpoint versions must update `checkpoint_format_version` and document
  any migration steps in this file.
- Safetensors support is additive. Existing `pytorch_model.bin` checkpoints
  should continue to load unless a deliberate breaking migration is documented.
- Tokenizer metadata is optional and additive. `from_pretrained` loads model
  weights without instantiating a tokenizer; use `load_tokenizer_metadata(...)`
  to read the metadata object when present.
- Checkpoint metadata is optional and additive. Use
  `load_checkpoint_metadata(...)` to inspect phase handoff metadata when
  present.

## Smoke Training State

`training_state.pt` currently uses format version `1` and is written with
`torch.save` by the AR, diffusion, and latent smoke trainers when `--out-dir` is
provided.

Expected fields:

- `format`: `open-alexandros-training-state`.
- `format_version`: integer training-state format version.
- `step`: last completed optimization step.
- `config_hash`: hash of the serialized `AlexandrosConfig`; resume rejects
  mismatches.
- `model_state_dict`: model weights at the saved step.
- `optimizer_state_dict`: optimizer state for local resume.
- `torch_rng_state`: CPU RNG state.
- `cuda_rng_state_all`: CUDA RNG states when CUDA is available, otherwise
  `null`.
- `data_state`: optional state for the training token-data or latent-trace
  iterator, currently used by `--token-ids-jsonl` and `--trace-jsonl` runs to
  preserve cursor, shuffled order, and data-generator state across resume.
- `grad_scaler_state_dict`: optional CUDA float16 GradScaler state for mixed
  precision resume.
- `metadata`: run metadata such as git commit, dependency versions, device,
  dtype, seed, deterministic flag, and hardware summary.

This artifact is for trusted local continuation only. It uses general PyTorch
pickle loading because optimizer and RNG states are part of the resume contract.
Portable model loading should use `from_pretrained()` on the `checkpoint/`
directory instead.
