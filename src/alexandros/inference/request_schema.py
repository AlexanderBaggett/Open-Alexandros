from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from alexandros.modeling_alexandros import GenerationMode


def _validate_nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class GenerationRequest:
    """Serializable generation request for CLIs and simple serving wrappers."""

    input_ids: list[list[int]]
    max_new_tokens: int = 16
    mode: GenerationMode | str = GenerationMode.AUTOREGRESSIVE
    use_cache: bool = False
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    stop_token_ids: list[int] | None = None
    stop_sequences: list[list[int]] | None = None
    steps: int | None = None
    block_size: int | None = None
    confidence_schedule: str = "median"
    remask_low_confidence: bool = False
    latent_steps: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", GenerationMode(self.mode))
        self._validate_input_ids()
        _validate_nonnegative_int("max_new_tokens", self.max_new_tokens)
        if not math.isfinite(float(self.temperature)) or self.temperature <= 0:
            raise ValueError("temperature must be finite and > 0")
        if self.top_k is not None:
            if (
                not isinstance(self.top_k, int)
                or isinstance(self.top_k, bool)
                or self.top_k <= 0
            ):
                raise ValueError("top_k must be a positive integer when provided")
        if self.top_p is not None and (
            not math.isfinite(float(self.top_p)) or not 0.0 < self.top_p <= 1.0
        ):
            raise ValueError(
                "top_p must be finite and satisfy 0 < top_p <= 1 when provided"
            )
        if (
            not math.isfinite(float(self.repetition_penalty))
            or self.repetition_penalty <= 0
        ):
            raise ValueError("repetition_penalty must be finite and > 0")
        if self.steps is not None:
            _validate_positive_int("steps", self.steps)
        if self.block_size is not None:
            _validate_positive_int("block_size", self.block_size)
        if self.confidence_schedule not in {"median", "linear", "all"}:
            raise ValueError("confidence_schedule must be one of: median, linear, all")
        if not isinstance(self.remask_low_confidence, bool):
            raise ValueError("remask_low_confidence must be a boolean")
        if self.latent_steps is not None:
            _validate_positive_int("latent_steps", self.latent_steps)
        self._validate_stop_controls()
        if self.mode != GenerationMode.AUTOREGRESSIVE and self.use_cache:
            raise ValueError("use_cache is only valid for autoregressive requests")
        if self.mode in {GenerationMode.LATENT_REASONING, GenerationMode.HYBRID}:
            if self.do_sample or self.top_k is not None or self.top_p is not None:
                raise ValueError(
                    "sampling controls are only valid for autoregressive or block_diffusion requests"
                )

    def _validate_input_ids(self) -> None:
        if not self.input_ids:
            raise ValueError("input_ids must contain at least one row")
        width = len(self.input_ids[0])
        if width == 0:
            raise ValueError("input_ids rows must be non-empty")
        for row in self.input_ids:
            if len(row) != width:
                raise ValueError("input_ids rows must have the same length")
            if any(
                not isinstance(token, int) or isinstance(token, bool) for token in row
            ):
                raise ValueError("input_ids must be integer token IDs")
            if any(token < 0 for token in row):
                raise ValueError("input_ids must be non-negative token IDs")

    def _validate_stop_controls(self) -> None:
        if self.stop_token_ids is not None:
            if any(
                not isinstance(token, int) or isinstance(token, bool)
                for token in self.stop_token_ids
            ):
                raise ValueError("stop_token_ids must be integer token IDs")
            if any(token < 0 for token in self.stop_token_ids):
                raise ValueError("stop_token_ids must be non-negative")
        if self.stop_sequences is not None:
            for sequence in self.stop_sequences:
                if not sequence:
                    raise ValueError("stop_sequences cannot contain empty sequences")
                if any(
                    not isinstance(token, int) or isinstance(token, bool)
                    for token in sequence
                ):
                    raise ValueError("stop_sequences must contain integer token IDs")
                if any(token < 0 for token in sequence):
                    raise ValueError("stop_sequences must be non-negative")

    def to_tensor(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.tensor(self.input_ids, dtype=torch.long, device=device)

    def generate_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "mode": self.mode,
        }
        if self.mode == GenerationMode.AUTOREGRESSIVE:
            kwargs.update(
                {
                    "use_cache": self.use_cache,
                    "do_sample": self.do_sample,
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                    "repetition_penalty": self.repetition_penalty,
                    "stop_token_ids": self.stop_token_ids,
                    "stop_sequences": self.stop_sequences,
                }
            )
        elif self.mode == GenerationMode.BLOCK_DIFFUSION:
            kwargs.update(
                {
                    "do_sample": self.do_sample,
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                    "stop_token_ids": self.stop_token_ids,
                    "stop_sequences": self.stop_sequences,
                }
            )
            if self.steps is not None:
                kwargs["steps"] = self.steps
            if self.block_size is not None:
                kwargs["block_size"] = self.block_size
            kwargs["confidence_schedule"] = self.confidence_schedule
            kwargs["remask_low_confidence"] = self.remask_low_confidence
        elif self.mode == GenerationMode.LATENT_REASONING:
            if self.latent_steps is not None:
                kwargs["latent_steps"] = self.latent_steps
        else:
            if self.latent_steps is not None:
                kwargs["latent_steps"] = self.latent_steps
            if self.steps is not None:
                kwargs["steps"] = self.steps
            if self.block_size is not None:
                kwargs["block_size"] = self.block_size
            kwargs["confidence_schedule"] = self.confidence_schedule
            kwargs["remask_low_confidence"] = self.remask_low_confidence
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationRequest":
        return cls(**data)


def generate_from_request(
    model: Any,
    request: GenerationRequest | dict[str, Any],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Run a validated generation request against an Alexandros model."""

    if not isinstance(request, GenerationRequest):
        request = GenerationRequest.from_dict(request)
    input_ids = request.to_tensor(device=device)
    return model.generate(input_ids, **request.generate_kwargs())
