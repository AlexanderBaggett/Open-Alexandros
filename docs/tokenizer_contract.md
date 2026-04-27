# Tokenizer Contract

Alexandros does not choose or train a production tokenizer. Downstream trainers
must supply token IDs that match `AlexandrosConfig` and may attach tokenizer
metadata to saved checkpoints for reproducibility.

## Required Compatibility

Any tokenizer used with Alexandros must satisfy:

- `vocab_size` equals `config.vocab_size`.
- `pad_token_id`, `bos_token_id`, `eos_token_id`, and `mask_token_id` match the
  config when those IDs are declared in metadata.
- IDs `0..3` are reserved by the default configs for pad, BOS, EOS, and mask.
  Downstream tokenizers may choose different IDs only by changing the matching
  `AlexandrosConfig` fields and preserving distinct special-token IDs.
- Any latent/reasoning marker tokens used by downstream curricula must be
  reserved by that tokenizer and documented in tokenizer metadata; Alexandros
  v1 does not reserve official latent-marker IDs beyond `mask_token_id`.
- Encoded training examples are integer token IDs in `[0, vocab_size)`.
- Packed training batches have shape `[batch, sequence]`.
- `pad_token_id` is used only for right-padding in the reference causal path.

## V1 Decision

The initial publication uses a downstream-supplied tokenizer. Alexandros does
not choose between SentencePiece, tiktoken-style BPE, Hugging Face tokenizers,
or a custom tokenizer, and does not provide a tokenizer training script.

## Metadata Format

`save_pretrained(..., tokenizer_metadata=...)` writes
`tokenizer_metadata.json` next to the model checkpoint. The user-supplied
metadata is wrapped with Alexandros format/version fields and a copy of the
config token IDs:

```json
{
  "format": "open-alexandros-tokenizer-metadata",
  "format_version": 1,
  "tokenizer": {
    "tokenizer_class": "ExampleTokenizer",
    "source": "user-supplied",
    "vocab_size": 32000,
    "special_tokens": {
      "pad_token_id": 0,
      "bos_token_id": 1,
      "eos_token_id": 2,
      "mask_token_id": 3
    }
  },
  "config": {
    "vocab_size": 32000,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "mask_token_id": 3
  }
}
```

The metadata may include tokenizer file names, checksums, source repository,
license, normalization settings, or training notes. It must remain JSON
serializable. If it declares `vocab_size` or special token IDs, they must match
the model config.

## Loading

Use:

```python
metadata = AlexandrosForCausalLM.load_tokenizer_metadata(checkpoint_dir)
```

The method returns the user-supplied `tokenizer` object or `None` when no
metadata file is present. It does not instantiate a tokenizer dependency.

## Non-Goals

- No tokenizer training script is part of the initial release.
- No SentencePiece, tiktoken, or Hugging Face tokenizer is selected as the
  official Alexandros tokenizer.
- No internet-scale text collection or tokenizer corpus is bundled.
