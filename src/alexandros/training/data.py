from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alexandros.configuration_alexandros import AlexandrosConfig
from alexandros.diffusion import MaskedDiffusionScheduler
from alexandros.training.objectives import IGNORE_INDEX


def iter_token_id_jsonl(
    path: str | Path, *, field: str = "input_ids"
) -> Iterator[list[int]]:
    """Stream token-id sequences from JSONL.

    Each line may be either a JSON object containing `field` or a bare list of
    integer token IDs. Blank lines are ignored.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_number}: {exc.msg}"
                ) from exc
            token_ids = payload.get(field) if isinstance(payload, dict) else payload
            if not isinstance(token_ids, list):
                raise ValueError(
                    f"line {line_number} must contain a list field named {field!r}"
                )
            if not token_ids:
                raise ValueError(f"line {line_number} has an empty token sequence")
            if any(
                not isinstance(token_id, int) or isinstance(token_id, bool)
                for token_id in token_ids
            ):
                raise ValueError(f"line {line_number} token IDs must be integers")
            yield token_ids


def _json_token_ids(
    payload: dict[str, Any],
    field: str,
    *,
    line_number: int,
    required: bool,
) -> tuple[int, ...] | None:
    if field not in payload:
        if required:
            raise ValueError(
                f"line {line_number} must contain a list field named {field!r}"
            )
        return None
    token_ids = payload[field]
    if not isinstance(token_ids, list):
        raise ValueError(
            f"line {line_number} field {field!r} must be a list of token IDs"
        )
    if not token_ids:
        raise ValueError(f"line {line_number} field {field!r} cannot be empty")
    if any(
        not isinstance(token_id, int) or isinstance(token_id, bool)
        for token_id in token_ids
    ):
        raise ValueError(
            f"line {line_number} field {field!r} token IDs must be integers"
        )
    return tuple(token_ids)


def validate_token_ids(token_ids: Sequence[int], config: AlexandrosConfig) -> None:
    for token_id in token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise ValueError("token IDs must be integers")
        if token_id < 0 or token_id >= config.vocab_size:
            raise ValueError("token IDs must be inside [0, vocab_size)")


@dataclass(frozen=True)
class LatentTraceRecord:
    input_ids: tuple[int, ...]
    trace_ids: tuple[int, ...]
    target_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class DistillationRecord:
    input_ids: tuple[int, ...]
    teacher_token_ids: tuple[int, ...] | None = None
    teacher_logits: tuple[tuple[float, ...], ...] | None = None


TokenDecoder = Callable[[int], str] | Mapping[int, str] | Sequence[str] | None


def _decode_token_id(token_id: int, token_decoder: TokenDecoder) -> str:
    if not isinstance(token_id, int) or isinstance(token_id, bool):
        raise ValueError("latent trace token IDs must be integers")
    if token_decoder is None:
        return str(token_id)
    if callable(token_decoder):
        decoded = token_decoder(token_id)
    elif isinstance(token_decoder, Mapping):
        decoded = token_decoder.get(token_id, f"<{token_id}>")
    elif isinstance(token_decoder, Sequence) and not isinstance(
        token_decoder, str | bytes
    ):
        decoded = (
            token_decoder[token_id]
            if 0 <= token_id < len(token_decoder)
            else f"<{token_id}>"
        )
    else:
        raise ValueError("token_decoder must be a callable, mapping, sequence, or None")
    if not isinstance(decoded, str):
        raise ValueError("token_decoder must return strings")
    return decoded


def _decode_token_ids(
    token_ids: Sequence[int], token_decoder: TokenDecoder, joiner: str
) -> str:
    return joiner.join(
        _decode_token_id(token_id, token_decoder) for token_id in token_ids
    )


def summarize_latent_trace_record(
    record: LatentTraceRecord,
    *,
    token_decoder: TokenDecoder = None,
    joiner: str = " ",
) -> dict[str, str | None]:
    """Return a lightweight decoded summary for latent-trace debugging.

    This helper is intentionally tokenizer-agnostic. Downstream users can pass a
    callable, mapping, or sequence that maps token IDs to display strings. Missing
    mapping/sequence entries fall back to `<token_id>`.
    """

    if not isinstance(record, LatentTraceRecord):
        raise ValueError("record must be a LatentTraceRecord")
    if not isinstance(joiner, str):
        raise ValueError("joiner must be a string")
    target = (
        None
        if record.target_ids is None
        else _decode_token_ids(record.target_ids, token_decoder, joiner)
    )
    return {
        "input": _decode_token_ids(record.input_ids, token_decoder, joiner),
        "trace": _decode_token_ids(record.trace_ids, token_decoder, joiner),
        "target": target,
    }


_SYNTHETIC_MOD_ADD_TOKEN = 4
_SYNTHETIC_XOR_TOKEN = 5
_SYNTHETIC_EQUALS_TOKEN = 6
_SYNTHETIC_ANSWER_TOKEN = 7
_SYNTHETIC_VALUE_OFFSET = 8


def _validate_synthetic_latent_task_args(
    config: AlexandrosConfig,
    *,
    value_count: int,
    count: int | None = None,
) -> None:
    if not isinstance(value_count, int) or isinstance(value_count, bool):
        raise ValueError("value_count must be an integer")
    if value_count <= 0:
        raise ValueError("value_count must be > 0")
    if count is not None and (
        not isinstance(count, int) or isinstance(count, bool) or count <= 0
    ):
        raise ValueError("count must be a positive integer when provided")
    required_vocab = _SYNTHETIC_VALUE_OFFSET + value_count
    if config.vocab_size < required_vocab:
        raise ValueError(
            "vocab_size is too small for synthetic latent trace task; "
            f"requires at least {required_vocab}"
        )


def _synthetic_value_token(value: int) -> int:
    return _SYNTHETIC_VALUE_OFFSET + value


def make_modular_addition_latent_trace_records(
    config: AlexandrosConfig,
    *,
    modulus: int = 4,
    count: int | None = None,
    seed: int = 0,
) -> tuple[LatentTraceRecord, ...]:
    """Create deterministic token-ID traces for `(a + b) mod modulus`.

    These records are CI fixtures for latent-trace mechanics, not training data
    for model quality. Token IDs use a tiny synthetic vocabulary:
    `4=mod_add`, `6=equals`, `7=answer`, and values start at `8`.
    """

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(modulus, int) or isinstance(modulus, bool):
        raise ValueError("modulus must be an integer")
    if modulus < 2:
        raise ValueError("modulus must be >= 2")
    _validate_synthetic_latent_task_args(config, value_count=modulus, count=count)
    total = modulus * modulus if count is None else count
    records: list[LatentTraceRecord] = []
    for idx in range(total):
        pair_idx = (idx + seed) % (modulus * modulus)
        a = pair_idx // modulus
        b = pair_idx % modulus
        result = (a + b) % modulus
        a_token = _synthetic_value_token(a)
        b_token = _synthetic_value_token(b)
        result_token = _synthetic_value_token(result)
        records.append(
            LatentTraceRecord(
                input_ids=(
                    config.bos_token_id,
                    _SYNTHETIC_MOD_ADD_TOKEN,
                    a_token,
                    b_token,
                    _SYNTHETIC_ANSWER_TOKEN,
                ),
                trace_ids=(a_token, b_token, _SYNTHETIC_EQUALS_TOKEN, result_token),
                target_ids=(result_token,),
            )
        )
    return tuple(records)


def make_boolean_xor_latent_trace_records(
    config: AlexandrosConfig,
    *,
    repetitions: int = 1,
) -> tuple[LatentTraceRecord, ...]:
    """Create deterministic token-ID traces for boolean XOR examples."""

    if not isinstance(repetitions, int) or isinstance(repetitions, bool):
        raise ValueError("repetitions must be an integer")
    if repetitions <= 0:
        raise ValueError("repetitions must be > 0")
    _validate_synthetic_latent_task_args(config, value_count=2, count=repetitions)
    records: list[LatentTraceRecord] = []
    for _ in range(repetitions):
        for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
            result = a ^ b
            a_token = _synthetic_value_token(a)
            b_token = _synthetic_value_token(b)
            result_token = _synthetic_value_token(result)
            records.append(
                LatentTraceRecord(
                    input_ids=(
                        config.bos_token_id,
                        _SYNTHETIC_XOR_TOKEN,
                        a_token,
                        b_token,
                        _SYNTHETIC_ANSWER_TOKEN,
                    ),
                    trace_ids=(a_token, b_token, _SYNTHETIC_EQUALS_TOKEN, result_token),
                    target_ids=(result_token,),
                )
            )
    return tuple(records)


def iter_latent_trace_jsonl(
    path: str | Path,
    config: AlexandrosConfig,
    *,
    input_field: str = "input_ids",
    trace_field: str = "trace_ids",
    target_field: str = "target_ids",
) -> Iterator[LatentTraceRecord]:
    """Stream user-supplied visible-reasoning trace records from token-ID JSONL."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            input_ids = _json_token_ids(
                payload, input_field, line_number=line_number, required=True
            )
            trace_ids = _json_token_ids(
                payload, trace_field, line_number=line_number, required=True
            )
            target_ids = _json_token_ids(
                payload, target_field, line_number=line_number, required=False
            )
            assert input_ids is not None
            assert trace_ids is not None
            validate_token_ids(input_ids, config)
            validate_token_ids(trace_ids, config)
            if target_ids is not None:
                validate_token_ids(target_ids, config)
            yield LatentTraceRecord(
                input_ids=input_ids, trace_ids=trace_ids, target_ids=target_ids
            )


def _json_teacher_logits(
    payload: dict[str, Any],
    field: str,
    *,
    line_number: int,
    config: AlexandrosConfig,
) -> tuple[tuple[float, ...], ...] | None:
    if field not in payload:
        return None
    logits = payload[field]
    if not isinstance(logits, list):
        raise ValueError(
            f"line {line_number} field {field!r} must be a list of logit rows"
        )
    if not logits:
        raise ValueError(f"line {line_number} field {field!r} cannot be empty")
    rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(logits):
        if not isinstance(row, list):
            raise ValueError(
                f"line {line_number} field {field!r} row {row_index} must be a list"
            )
        if len(row) != config.vocab_size:
            raise ValueError(
                f"line {line_number} field {field!r} row {row_index} must have vocab_size entries"
            )
        values: list[float] = []
        for value in row:
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(
                    f"line {line_number} field {field!r} logits must be numbers"
                )
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(
                    f"line {line_number} field {field!r} logits must be finite"
                )
            values.append(value)
        rows.append(tuple(values))
    return tuple(rows)


def iter_distillation_jsonl(
    path: str | Path,
    config: AlexandrosConfig,
    *,
    input_field: str = "input_ids",
    teacher_token_field: str = "teacher_token_ids",
    teacher_logits_field: str = "teacher_logits",
) -> Iterator[DistillationRecord]:
    """Stream user-supplied distillation records from token-ID/logit JSONL.

    Each record must include `input_ids` and at least one teacher signal:
    generated token targets in `teacher_token_ids` or dense per-token logits in
    `teacher_logits`. The helper validates structure only; teacher model
    selection and license checks are downstream responsibilities.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            input_ids = _json_token_ids(
                payload,
                input_field,
                line_number=line_number,
                required=True,
            )
            teacher_token_ids = _json_token_ids(
                payload,
                teacher_token_field,
                line_number=line_number,
                required=False,
            )
            teacher_logits = _json_teacher_logits(
                payload,
                teacher_logits_field,
                line_number=line_number,
                config=config,
            )
            assert input_ids is not None
            if teacher_token_ids is None and teacher_logits is None:
                raise ValueError(
                    f"line {line_number} must contain teacher_token_ids or teacher_logits"
                )
            validate_token_ids(input_ids, config)
            if teacher_token_ids is not None:
                validate_token_ids(teacher_token_ids, config)
            yield DistillationRecord(
                input_ids=input_ids,
                teacher_token_ids=teacher_token_ids,
                teacher_logits=teacher_logits,
            )


@dataclass(frozen=True)
class DiffusionTrainingBatch:
    input_ids: Any
    noisy_input_ids: Any
    labels: Any
    mask: Any
    timesteps: Any
    masked_token_count: Any

    @property
    def masked_fraction(self) -> float:
        total = max(int(self.mask.numel()), 1)
        return float(self.masked_token_count.item()) / total


@dataclass(frozen=True)
class BlockDiffusionTrainingBatch(DiffusionTrainingBatch):
    block_starts: Any
    block_size: int


@dataclass(frozen=True)
class LatentTraceBatch:
    input_ids: Any
    input_attention_mask: Any
    trace_ids: Any
    trace_attention_mask: Any
    target_ids: Any | None = None
    target_attention_mask: Any | None = None


def _validate_training_input_ids(input_ids, config: AlexandrosConfig, *, torch) -> None:
    if not torch.is_tensor(input_ids):
        raise ValueError("input_ids must be a torch tensor")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.dtype not in {torch.int32, torch.int64}:
        raise ValueError("input_ids must be an integer tensor of token IDs")
    if input_ids.size(0) == 0:
        raise ValueError("input_ids batch size must be > 0")
    if input_ids.size(1) == 0:
        raise ValueError("input_ids sequence length must be > 0")
    if input_ids.min().item() < 0 or input_ids.max().item() >= config.vocab_size:
        raise ValueError("input_ids contain token IDs outside [0, vocab_size)")


def make_diffusion_training_batch(
    input_ids,
    config: AlexandrosConfig,
    *,
    torch,
    scheduler: MaskedDiffusionScheduler | None = None,
    timesteps=None,
    generator=None,
) -> DiffusionTrainingBatch:
    _validate_training_input_ids(input_ids, config, torch=torch)
    scheduler = MaskedDiffusionScheduler(config) if scheduler is None else scheduler
    if timesteps is None:
        timesteps = torch.randint(
            0,
            config.diffusion_steps,
            (input_ids.size(0),),
            device=input_ids.device,
            generator=generator,
        )
    noisy, mask = scheduler.add_noise(input_ids, timesteps, generator=generator)
    masked_count = mask.sum()
    if masked_count.item() == 0:
        raise ValueError("diffusion batch requires at least one non-pad token to mask")
    labels = input_ids.masked_fill(~mask, IGNORE_INDEX)
    return DiffusionTrainingBatch(
        input_ids=input_ids,
        noisy_input_ids=noisy,
        labels=labels,
        mask=mask,
        timesteps=timesteps,
        masked_token_count=masked_count,
    )


def make_block_diffusion_training_batch(
    input_ids,
    config: AlexandrosConfig,
    *,
    torch,
    block_size: int,
    timesteps=None,
    generator=None,
) -> BlockDiffusionTrainingBatch:
    _validate_training_input_ids(input_ids, config, torch=torch)
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    seq_len = input_ids.size(1)
    if block_size > seq_len:
        raise ValueError("block_size cannot exceed sequence length")
    if timesteps is None:
        timesteps = torch.randint(
            0,
            config.diffusion_steps,
            (input_ids.size(0),),
            device=input_ids.device,
            generator=generator,
        )
    else:
        timesteps = MaskedDiffusionScheduler(config)._normalize_timestep_for_input(
            timesteps,
            input_ids,
        )
    starts = torch.randint(
        0,
        seq_len - block_size + 1,
        (input_ids.size(0),),
        device=input_ids.device,
        generator=generator,
    )
    valid = input_ids.ne(config.pad_token_id)
    if not valid.any(dim=1).all():
        raise ValueError(
            "block diffusion batch requires at least one non-pad token per row"
        )
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row in range(input_ids.size(0)):
        start = int(starts[row].item())
        end = start + block_size
        row_mask = valid[row, start:end]
        if not row_mask.any():
            first_valid = int(valid[row].nonzero(as_tuple=False)[0].item())
            start = min(max(first_valid - block_size + 1, 0), seq_len - block_size)
            end = start + block_size
            starts[row] = start
            row_mask = valid[row, start:end]
        mask[row, start:end] = row_mask
    masked_count = mask.sum()
    if masked_count.item() == 0:
        raise ValueError("block diffusion batch requires at least one token to mask")
    noisy = input_ids.masked_fill(mask, config.mask_token_id)
    labels = input_ids.masked_fill(~mask, IGNORE_INDEX)
    return BlockDiffusionTrainingBatch(
        input_ids=input_ids,
        noisy_input_ids=noisy,
        labels=labels,
        mask=mask,
        timesteps=timesteps,
        masked_token_count=masked_count,
        block_starts=starts,
        block_size=block_size,
    )


def _stable_fraction(index: int, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{index}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64)


def record_in_split(
    index: int,
    *,
    split: str,
    validation_fraction: float,
    seed: int,
) -> bool:
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must satisfy 0 <= fraction < 1")
    is_validation = (
        validation_fraction > 0 and _stable_fraction(index, seed) < validation_fraction
    )
    return is_validation if split == "validation" else not is_validation


def pack_token_sequences(
    sequences: Iterable[Sequence[int]],
    *,
    seq_len: int,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> Iterator[list[int]]:
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    buffer: list[int] = []
    for sequence in sequences:
        record = list(sequence)
        if bos_token_id is not None:
            record.insert(0, bos_token_id)
        if eos_token_id is not None:
            record.append(eos_token_id)
        buffer.extend(record)
        while len(buffer) >= seq_len:
            yield buffer[:seq_len]
            del buffer[:seq_len]


def _pad_or_truncate_token_ids(
    token_ids: Sequence[int],
    *,
    length: int,
    pad_token_id: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ValueError("trace dataset lengths must be positive integers")
    trimmed = tuple(token_ids[:length])
    mask = (1,) * len(trimmed)
    pad_count = length - len(trimmed)
    if pad_count > 0:
        trimmed = trimmed + (pad_token_id,) * pad_count
        mask = mask + (0,) * pad_count
    return trimmed, mask


def _pad_or_truncate_logits(
    logits: Sequence[Sequence[float]],
    *,
    length: int,
    vocab_size: int,
) -> tuple[tuple[tuple[float, ...], ...], tuple[int, ...]]:
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ValueError("distillation dataset lengths must be positive integers")
    trimmed = tuple(tuple(float(value) for value in row) for row in logits[:length])
    for row in trimmed:
        if len(row) != vocab_size:
            raise ValueError("teacher logit rows must have vocab_size entries")
        if any(not math.isfinite(value) for value in row):
            raise ValueError("teacher logits must be finite")
    mask = (1,) * len(trimmed)
    pad_count = length - len(trimmed)
    if pad_count > 0:
        zero_row = (0.0,) * vocab_size
        trimmed = trimmed + (zero_row,) * pad_count
        mask = mask + (0,) * pad_count
    return trimmed, mask


@dataclass(frozen=True)
class PackedTokenDataset:
    chunks: tuple[tuple[int, ...], ...]

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        config: AlexandrosConfig,
        *,
        seq_len: int,
        field: str = "input_ids",
        split: str = "train",
        validation_fraction: float = 0.0,
        seed: int = 0,
        prepend_bos: bool = False,
        append_eos: bool = True,
    ) -> "PackedTokenDataset":
        def selected_sequences() -> Iterator[list[int]]:
            for index, token_ids in enumerate(iter_token_id_jsonl(path, field=field)):
                validate_token_ids(token_ids, config)
                if record_in_split(
                    index,
                    split=split,
                    validation_fraction=validation_fraction,
                    seed=seed,
                ):
                    yield token_ids

        chunks = tuple(
            tuple(chunk)
            for chunk in pack_token_sequences(
                selected_sequences(),
                seq_len=seq_len,
                bos_token_id=config.bos_token_id if prepend_bos else None,
                eos_token_id=config.eos_token_id if append_eos else None,
            )
        )
        if not chunks:
            raise ValueError(
                "token-id dataset produced no packed chunks; add more tokens, "
                "lower seq_len, or adjust validation_fraction"
            )
        return cls(chunks=chunks)

    def batch_iterator(
        self,
        *,
        batch_size: int,
        torch,
        device,
        seed: int = 0,
        shuffle: bool = True,
    ) -> "PackedTokenBatchIterator":
        return PackedTokenBatchIterator(
            self,
            batch_size=batch_size,
            torch=torch,
            device=device,
            seed=seed,
            shuffle=shuffle,
        )


class PackedTokenBatchIterator:
    def __init__(
        self,
        dataset: PackedTokenDataset,
        *,
        batch_size: int,
        torch,
        device,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean")
        self.dataset = dataset
        self.batch_size = batch_size
        self.torch = torch
        self.device = device
        self.seed = seed
        self.shuffle = shuffle
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        self.order = torch.arange(len(self.dataset.chunks))
        self.cursor = 0

    def __iter__(self) -> "PackedTokenBatchIterator":
        return self

    def __next__(self):
        rows = []
        for _ in range(self.batch_size):
            if self.cursor == 0 and self.shuffle and len(self.order) > 1:
                self.order = self.order[
                    self.torch.randperm(len(self.order), generator=self.generator)
                ]
            rows.append(self.dataset.chunks[int(self.order[self.cursor].item())])
            self.cursor += 1
            if self.cursor >= len(self.order):
                self.cursor = 0
        return self.torch.tensor(rows, dtype=self.torch.long, device=self.device)

    def state_dict(self) -> dict[str, object]:
        return {
            "format": "packed-token-batch-iterator",
            "batch_size": self.batch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "cursor": self.cursor,
            "order": self.order.clone(),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("format") != "packed-token-batch-iterator":
            raise ValueError("unsupported packed token iterator state")
        batch_size = state.get("batch_size")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError("packed token iterator batch_size must be an integer")
        if batch_size != self.batch_size:
            raise ValueError("packed token iterator batch_size does not match")
        order = state.get("order")
        generator_state = state.get("generator_state")
        if (
            not self.torch.is_tensor(order)
            or order.ndim != 1
            or order.numel() != len(self.dataset.chunks)
        ):
            raise ValueError("packed token iterator order does not match dataset")
        if order.dtype not in {self.torch.int32, self.torch.int64}:
            raise ValueError("packed token iterator order must be an integer tensor")
        if (
            not self.torch.is_tensor(generator_state)
            or generator_state.dtype != self.torch.uint8
        ):
            raise ValueError(
                "packed token iterator generator_state must be a uint8 tensor"
            )
        order = order.to(dtype=self.torch.long).cpu().reshape(-1)
        expected_order = self.torch.arange(
            len(self.dataset.chunks), dtype=self.torch.long
        )
        if not self.torch.equal(self.torch.sort(order).values, expected_order):
            raise ValueError("packed token iterator order must be a permutation")
        cursor = state.get("cursor")
        if not isinstance(cursor, int) or isinstance(cursor, bool):
            raise ValueError("packed token iterator cursor must be an integer")
        if cursor < 0 or cursor >= len(self.dataset.chunks):
            raise ValueError("packed token iterator cursor is out of range")
        seed = state.get("seed", self.seed)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("packed token iterator seed must be an integer")
        shuffle = state.get("shuffle", self.shuffle)
        if not isinstance(shuffle, bool):
            raise ValueError("packed token iterator shuffle must be a boolean")
        self.seed = seed
        self.shuffle = shuffle
        self.cursor = cursor
        self.order = order
        self.generator.set_state(generator_state.cpu())


@dataclass(frozen=True)
class DistillationExample:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    teacher_token_ids: tuple[int, ...] | None
    teacher_token_mask: tuple[int, ...] | None
    teacher_logits: tuple[tuple[float, ...], ...] | None
    teacher_logits_mask: tuple[int, ...] | None


@dataclass(frozen=True)
class DistillationBatch:
    input_ids: Any
    attention_mask: Any
    teacher_token_ids: Any | None = None
    teacher_token_mask: Any | None = None
    teacher_logits: Any | None = None
    teacher_logits_mask: Any | None = None


@dataclass(frozen=True)
class DistillationDataset:
    examples: tuple[DistillationExample, ...]

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        config: AlexandrosConfig,
        *,
        seq_len: int,
        input_field: str = "input_ids",
        teacher_token_field: str = "teacher_token_ids",
        teacher_logits_field: str = "teacher_logits",
        split: str = "train",
        validation_fraction: float = 0.0,
        seed: int = 0,
    ) -> "DistillationDataset":
        if split == "validation" and validation_fraction <= 0.0:
            raise ValueError(
                "validation_fraction must be > 0 when validating distillation data"
            )
        examples: list[DistillationExample] = []
        for index, record in enumerate(
            iter_distillation_jsonl(
                path,
                config,
                input_field=input_field,
                teacher_token_field=teacher_token_field,
                teacher_logits_field=teacher_logits_field,
            )
        ):
            if not record_in_split(
                index,
                split=split,
                validation_fraction=validation_fraction,
                seed=seed,
            ):
                continue
            input_ids, attention_mask = _pad_or_truncate_token_ids(
                record.input_ids,
                length=seq_len,
                pad_token_id=config.pad_token_id,
            )
            teacher_token_ids = None
            teacher_token_mask = None
            if record.teacher_token_ids is not None:
                teacher_token_ids, teacher_token_mask = _pad_or_truncate_token_ids(
                    record.teacher_token_ids,
                    length=seq_len,
                    pad_token_id=config.pad_token_id,
                )
            teacher_logits = None
            teacher_logits_mask = None
            if record.teacher_logits is not None:
                teacher_logits, teacher_logits_mask = _pad_or_truncate_logits(
                    record.teacher_logits,
                    length=seq_len,
                    vocab_size=config.vocab_size,
                )
            examples.append(
                DistillationExample(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    teacher_token_ids=teacher_token_ids,
                    teacher_token_mask=teacher_token_mask,
                    teacher_logits=teacher_logits,
                    teacher_logits_mask=teacher_logits_mask,
                )
            )
        if not examples:
            raise ValueError(
                "distillation dataset produced no examples; add records or adjust validation_fraction"
            )
        return cls(examples=tuple(examples))

    def batch_iterator(
        self,
        *,
        batch_size: int,
        torch,
        device,
        seed: int = 0,
        shuffle: bool = True,
    ) -> "DistillationBatchIterator":
        return DistillationBatchIterator(
            self,
            batch_size=batch_size,
            torch=torch,
            device=device,
            seed=seed,
            shuffle=shuffle,
        )


class DistillationBatchIterator:
    def __init__(
        self,
        dataset: DistillationDataset,
        *,
        batch_size: int,
        torch,
        device,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean")
        self.dataset = dataset
        self.batch_size = batch_size
        self.torch = torch
        self.device = device
        self.seed = seed
        self.shuffle = shuffle
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        self.order = torch.arange(len(self.dataset.examples))
        self.cursor = 0

    def __iter__(self) -> "DistillationBatchIterator":
        return self

    def __next__(self) -> DistillationBatch:
        examples = []
        for _ in range(self.batch_size):
            if self.cursor == 0 and self.shuffle and len(self.order) > 1:
                self.order = self.order[
                    self.torch.randperm(len(self.order), generator=self.generator)
                ]
            examples.append(self.dataset.examples[int(self.order[self.cursor].item())])
            self.cursor += 1
            if self.cursor >= len(self.order):
                self.cursor = 0

        teacher_token_ids = None
        teacher_token_mask = None
        if any(example.teacher_token_ids is not None for example in examples):
            first_with_tokens = next(
                example for example in examples if example.teacher_token_ids is not None
            )
            token_pad = (0,) * len(first_with_tokens.teacher_token_ids)
            mask_pad = (0,) * len(first_with_tokens.teacher_token_ids)
            teacher_token_ids = self.torch.tensor(
                [
                    example.teacher_token_ids
                    if example.teacher_token_ids is not None
                    else token_pad
                    for example in examples
                ],
                dtype=self.torch.long,
                device=self.device,
            )
            teacher_token_mask = self.torch.tensor(
                [
                    example.teacher_token_mask
                    if example.teacher_token_mask is not None
                    else mask_pad
                    for example in examples
                ],
                dtype=self.torch.long,
                device=self.device,
            )

        teacher_logits = None
        teacher_logits_mask = None
        if any(example.teacher_logits is not None for example in examples):
            first_with_logits = next(
                example for example in examples if example.teacher_logits is not None
            )
            seq_len = len(first_with_logits.teacher_logits)
            vocab_size = len(first_with_logits.teacher_logits[0])
            logits_pad = tuple((0.0,) * vocab_size for _ in range(seq_len))
            mask_pad = (0,) * seq_len
            teacher_logits = self.torch.tensor(
                [
                    example.teacher_logits
                    if example.teacher_logits is not None
                    else logits_pad
                    for example in examples
                ],
                dtype=self.torch.float32,
                device=self.device,
            )
            teacher_logits_mask = self.torch.tensor(
                [
                    example.teacher_logits_mask
                    if example.teacher_logits_mask is not None
                    else mask_pad
                    for example in examples
                ],
                dtype=self.torch.long,
                device=self.device,
            )

        return DistillationBatch(
            input_ids=self.torch.tensor(
                [example.input_ids for example in examples],
                dtype=self.torch.long,
                device=self.device,
            ),
            attention_mask=self.torch.tensor(
                [example.attention_mask for example in examples],
                dtype=self.torch.long,
                device=self.device,
            ),
            teacher_token_ids=teacher_token_ids,
            teacher_token_mask=teacher_token_mask,
            teacher_logits=teacher_logits,
            teacher_logits_mask=teacher_logits_mask,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "format": "distillation-batch-iterator",
            "batch_size": self.batch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "cursor": self.cursor,
            "order": self.order.clone(),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("format") != "distillation-batch-iterator":
            raise ValueError("unsupported distillation iterator state")
        batch_size = state.get("batch_size")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError("distillation iterator batch_size must be an integer")
        if batch_size != self.batch_size:
            raise ValueError("distillation iterator batch_size does not match")
        order = state.get("order")
        generator_state = state.get("generator_state")
        if (
            not self.torch.is_tensor(order)
            or order.ndim != 1
            or order.numel() != len(self.dataset.examples)
        ):
            raise ValueError("distillation iterator order does not match dataset")
        if order.dtype not in {self.torch.int32, self.torch.int64}:
            raise ValueError("distillation iterator order must be an integer tensor")
        if (
            not self.torch.is_tensor(generator_state)
            or generator_state.dtype != self.torch.uint8
        ):
            raise ValueError(
                "distillation iterator generator_state must be a uint8 tensor"
            )
        order = order.to(dtype=self.torch.long).cpu().reshape(-1)
        expected_order = self.torch.arange(
            len(self.dataset.examples), dtype=self.torch.long
        )
        if not self.torch.equal(self.torch.sort(order).values, expected_order):
            raise ValueError("distillation iterator order must be a permutation")
        cursor = state.get("cursor")
        if not isinstance(cursor, int) or isinstance(cursor, bool):
            raise ValueError("distillation iterator cursor must be an integer")
        if cursor < 0 or cursor >= len(self.dataset.examples):
            raise ValueError("distillation iterator cursor is out of range")
        seed = state.get("seed", self.seed)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("distillation iterator seed must be an integer")
        shuffle = state.get("shuffle", self.shuffle)
        if not isinstance(shuffle, bool):
            raise ValueError("distillation iterator shuffle must be a boolean")
        self.seed = seed
        self.shuffle = shuffle
        self.cursor = cursor
        self.order = order
        self.generator.set_state(generator_state.cpu())


@dataclass(frozen=True)
class LatentTraceExample:
    input_ids: tuple[int, ...]
    input_attention_mask: tuple[int, ...]
    trace_ids: tuple[int, ...]
    trace_attention_mask: tuple[int, ...]
    target_ids: tuple[int, ...] | None
    target_attention_mask: tuple[int, ...] | None


@dataclass(frozen=True)
class LatentTraceDataset:
    examples: tuple[LatentTraceExample, ...]

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        config: AlexandrosConfig,
        *,
        seq_len: int,
        trace_len: int,
        input_field: str = "input_ids",
        trace_field: str = "trace_ids",
        target_field: str = "target_ids",
        split: str = "train",
        validation_fraction: float = 0.0,
        seed: int = 0,
    ) -> "LatentTraceDataset":
        if split == "validation" and validation_fraction <= 0.0:
            raise ValueError(
                "validation_fraction must be > 0 when validating latent trace data"
            )
        examples: list[LatentTraceExample] = []
        for index, record in enumerate(
            iter_latent_trace_jsonl(
                path,
                config,
                input_field=input_field,
                trace_field=trace_field,
                target_field=target_field,
            )
        ):
            if not record_in_split(
                index,
                split=split,
                validation_fraction=validation_fraction,
                seed=seed,
            ):
                continue
            input_ids, input_mask = _pad_or_truncate_token_ids(
                record.input_ids,
                length=seq_len,
                pad_token_id=config.pad_token_id,
            )
            trace_ids, trace_mask = _pad_or_truncate_token_ids(
                record.trace_ids,
                length=trace_len,
                pad_token_id=config.pad_token_id,
            )
            target_ids = None
            target_mask = None
            if record.target_ids is not None:
                target_ids, target_mask = _pad_or_truncate_token_ids(
                    record.target_ids,
                    length=trace_len,
                    pad_token_id=config.pad_token_id,
                )
            examples.append(
                LatentTraceExample(
                    input_ids=input_ids,
                    input_attention_mask=input_mask,
                    trace_ids=trace_ids,
                    trace_attention_mask=trace_mask,
                    target_ids=target_ids,
                    target_attention_mask=target_mask,
                )
            )
        if not examples:
            raise ValueError(
                "latent trace dataset produced no examples; add records or adjust validation_fraction"
            )
        return cls(examples=tuple(examples))

    def batch_iterator(
        self,
        *,
        batch_size: int,
        torch,
        device,
        seed: int = 0,
        shuffle: bool = True,
    ) -> "LatentTraceBatchIterator":
        return LatentTraceBatchIterator(
            self,
            batch_size=batch_size,
            torch=torch,
            device=device,
            seed=seed,
            shuffle=shuffle,
        )


class LatentTraceBatchIterator:
    def __init__(
        self,
        dataset: LatentTraceDataset,
        *,
        batch_size: int,
        torch,
        device,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean")
        self.dataset = dataset
        self.batch_size = batch_size
        self.torch = torch
        self.device = device
        self.seed = seed
        self.shuffle = shuffle
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        self.order = torch.arange(len(self.dataset.examples))
        self.cursor = 0

    def __iter__(self) -> "LatentTraceBatchIterator":
        return self

    def __next__(self) -> LatentTraceBatch:
        examples = []
        for _ in range(self.batch_size):
            if self.cursor == 0 and self.shuffle and len(self.order) > 1:
                self.order = self.order[
                    self.torch.randperm(len(self.order), generator=self.generator)
                ]
            examples.append(self.dataset.examples[int(self.order[self.cursor].item())])
            self.cursor += 1
            if self.cursor >= len(self.order):
                self.cursor = 0

        target_ids = None
        target_attention_mask = None
        if any(example.target_ids is not None for example in examples):
            target_ids = self.torch.tensor(
                [
                    example.target_ids
                    if example.target_ids is not None
                    else example.trace_ids
                    for example in examples
                ],
                dtype=self.torch.long,
                device=self.device,
            )
            target_attention_mask = self.torch.tensor(
                [
                    example.target_attention_mask
                    if example.target_attention_mask is not None
                    else example.trace_attention_mask
                    for example in examples
                ],
                dtype=self.torch.long,
                device=self.device,
            )

        return LatentTraceBatch(
            input_ids=self.torch.tensor(
                [example.input_ids for example in examples],
                dtype=self.torch.long,
                device=self.device,
            ),
            input_attention_mask=self.torch.tensor(
                [example.input_attention_mask for example in examples],
                dtype=self.torch.long,
                device=self.device,
            ),
            trace_ids=self.torch.tensor(
                [example.trace_ids for example in examples],
                dtype=self.torch.long,
                device=self.device,
            ),
            trace_attention_mask=self.torch.tensor(
                [example.trace_attention_mask for example in examples],
                dtype=self.torch.long,
                device=self.device,
            ),
            target_ids=target_ids,
            target_attention_mask=target_attention_mask,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "format": "latent-trace-batch-iterator",
            "batch_size": self.batch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "cursor": self.cursor,
            "order": self.order.clone(),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("format") != "latent-trace-batch-iterator":
            raise ValueError("unsupported latent trace iterator state")
        batch_size = state.get("batch_size")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError("latent trace iterator batch_size must be an integer")
        if batch_size != self.batch_size:
            raise ValueError("latent trace iterator batch_size does not match")
        order = state.get("order")
        generator_state = state.get("generator_state")
        if (
            not self.torch.is_tensor(order)
            or order.ndim != 1
            or order.numel() != len(self.dataset.examples)
        ):
            raise ValueError("latent trace iterator order does not match dataset")
        if order.dtype not in {self.torch.int32, self.torch.int64}:
            raise ValueError("latent trace iterator order must be an integer tensor")
        if (
            not self.torch.is_tensor(generator_state)
            or generator_state.dtype != self.torch.uint8
        ):
            raise ValueError(
                "latent trace iterator generator_state must be a uint8 tensor"
            )
        order = order.to(dtype=self.torch.long).cpu().reshape(-1)
        expected_order = self.torch.arange(
            len(self.dataset.examples), dtype=self.torch.long
        )
        if not self.torch.equal(self.torch.sort(order).values, expected_order):
            raise ValueError("latent trace iterator order must be a permutation")
        cursor = state.get("cursor")
        if not isinstance(cursor, int) or isinstance(cursor, bool):
            raise ValueError("latent trace iterator cursor must be an integer")
        if cursor < 0 or cursor >= len(self.dataset.examples):
            raise ValueError("latent trace iterator cursor is out of range")
        seed = state.get("seed", self.seed)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("latent trace iterator seed must be an integer")
        shuffle = state.get("shuffle", self.shuffle)
        if not isinstance(shuffle, bool):
            raise ValueError("latent trace iterator shuffle must be a boolean")
        self.seed = seed
        self.shuffle = shuffle
        self.cursor = cursor
        self.order = order
        self.generator.set_state(generator_state.cpu())
