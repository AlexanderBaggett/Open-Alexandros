from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from torch import nn

TRAINABLE_SCOPES = (
    "phase_default",
    "all",
    "backbone_only",
    "head_only",
    "latent_only",
    "none",
)


@dataclass(frozen=True)
class TrainabilityReport:
    phase: str
    scope: str
    trainable_parameters: int
    frozen_parameters: int
    trainable_tensors: int
    frozen_tensors: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

    def to_log_fields(self) -> dict[str, int | str]:
        return {f"trainability_{key}": value for key, value in self.to_dict().items()}


def _default_scope_for_phase(phase: str) -> str:
    if phase == "ar":
        return "all"
    if phase == "diffusion":
        return "diffusion_default"
    if phase == "latent":
        return "latent_default"
    if phase == "ttt":
        return "none"
    raise ValueError(f"unknown training phase: {phase}")


def _enabled_prefixes(phase: str, scope: str) -> tuple[str, ...] | None:
    actual_scope = (
        _default_scope_for_phase(phase) if scope == "phase_default" else scope
    )
    if actual_scope == "all":
        return None
    if actual_scope == "none":
        return ()
    if actual_scope == "backbone_only":
        return ("model",)
    if actual_scope == "head_only":
        return ("lm_head",)
    if actual_scope == "latent_only":
        return ("latent_vae", "latent_reasoner")
    if actual_scope == "diffusion_default":
        return ("model", "lm_head")
    if actual_scope == "latent_default":
        return ("model", "latent_vae", "latent_reasoner")
    raise ValueError(f"unknown trainable scope: {scope}")


def _matches_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def apply_trainability(
    model: nn.Module,
    *,
    phase: str,
    scope: str = "phase_default",
) -> TrainabilityReport:
    if scope not in TRAINABLE_SCOPES:
        raise ValueError(f"trainable scope must be one of {TRAINABLE_SCOPES}")
    prefixes = _enabled_prefixes(phase, scope)
    trainable_parameters = 0
    frozen_parameters = 0
    trainable_tensors = 0
    frozen_tensors = 0
    for name, parameter in model.named_parameters():
        enabled = True if prefixes is None else _matches_prefix(name, prefixes)
        parameter.requires_grad_(enabled)
        count = parameter.numel()
        if enabled:
            trainable_parameters += count
            trainable_tensors += 1
        else:
            frozen_parameters += count
            frozen_tensors += 1
    return TrainabilityReport(
        phase=phase,
        scope=scope,
        trainable_parameters=trainable_parameters,
        frozen_parameters=frozen_parameters,
        trainable_tensors=trainable_tensors,
        frozen_tensors=frozen_tensors,
    )


def trainable_parameters(model: nn.Module):
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
