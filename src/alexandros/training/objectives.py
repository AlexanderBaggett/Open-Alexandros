from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

IGNORE_INDEX = -100


@dataclass(frozen=True)
class ObjectiveContract:
    phase: str
    name: str
    inputs: str
    targets: str
    normalization: str
    ignore_index: int | None
    stopping_criteria: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_log_fields(self) -> dict[str, Any]:
        return {f"objective_{key}": value for key, value in self.to_dict().items()}


@dataclass(frozen=True)
class DistillationLossOutput:
    loss: Any
    token_loss: Any | None
    logit_loss: Any | None
    token_count: int
    logit_count: int


OBJECTIVE_CONTRACTS: dict[str, ObjectiveContract] = {
    "ar": ObjectiveContract(
        phase="ar",
        name="causal_next_token",
        inputs="input_ids[batch,sequence]",
        targets="labels=input_ids shifted left by one token",
        normalization="mean cross entropy over non-ignored next-token targets",
        ignore_index=IGNORE_INDEX,
        stopping_criteria="smoke run stops after --steps optimizer steps",
        notes="AR token-id JSONL prepends BOS and appends EOS during packing.",
    ),
    "diffusion": ObjectiveContract(
        phase="diffusion",
        name="absorbing_masked_or_rao_blackwellized_token",
        inputs="noisy_input_ids from absorbing-mask corruption plus diffusion_timestep",
        targets="original input_ids at masked positions only",
        normalization=(
            "masked objective divides weighted masked-token CE by masked_token_count; "
            "rao_blackwellized objective divides weighted target-position CE by non-pad target count"
        ),
        ignore_index=IGNORE_INDEX,
        stopping_criteria="smoke run stops after --steps optimizer steps",
        notes=(
            "Default objective is masked. rao_blackwellized expands each sampled "
            "noised background over all non-pad target positions. Weighting defaults "
            "to uniform; mask_prob and inverse_mask_prob are explicit research knobs."
        ),
    ),
    "latent": ObjectiveContract(
        phase="latent",
        name="latent_reconstruction_kl",
        inputs="backbone hidden states pooled into latent VAE slots",
        targets="pooled hidden-state reconstruction proxy for VAE and refined latent decoders",
        normalization="lambda_rec * mean(VAE MSE, refined-latent MSE) plus lambda_kl * KL",
        ignore_index=None,
        stopping_criteria="smoke run stops after --steps optimizer steps",
        notes="Visible reasoning traces are not bundled; this validates VAE and refinement mechanics only.",
    ),
    "ttt": ObjectiveContract(
        phase="ttt",
        name="request_local_next_token_probe",
        inputs=(
            "long-context hidden chunks, learned TTT phi_0 fast weights, and "
            "request-local inner-loop fast-weight updates"
        ),
        targets="next tokens on prefix chunks for inner updates and heldout/future chunks for outer loss",
        normalization="mean cross entropy over next-token targets per chunk",
        ignore_index=None,
        stopping_criteria="smoke run stops after --steps outer optimizer steps",
        notes=(
            "The learned TTTMetaAdapter is checkpointed separately; request-local "
            "fast state is never saved into model checkpoints."
        ),
    ),
    "distillation": ObjectiveContract(
        phase="distillation",
        name="teacher_token_or_logit_distillation",
        inputs="input_ids plus user-supplied teacher_token_ids and/or teacher_logits",
        targets="teacher generated token targets and/or dense teacher probability distribution",
        normalization="masked mean token CE plus temperature-scaled masked mean KL",
        ignore_index=None,
        stopping_criteria="downstream trainer defined; repository provides mechanism helpers only",
        notes=(
            "Teacher choice, license checks, terms compliance, and redistribution "
            "constraints are downstream responsibilities."
        ),
    ),
}


def objective_contract(phase: str) -> ObjectiveContract:
    try:
        return OBJECTIVE_CONTRACTS[phase]
    except KeyError as exc:
        raise ValueError(f"unknown objective phase: {phase}") from exc


def objective_log_fields(phase: str) -> dict[str, Any]:
    return objective_contract(phase).to_log_fields()


def _validate_student_logits(student_logits: Any) -> tuple[int, int, int]:
    if (
        not hasattr(student_logits, "is_floating_point")
        or not student_logits.is_floating_point()
    ):
        raise ValueError("student_logits must be a floating-point tensor")
    if student_logits.ndim != 3:
        raise ValueError("student_logits must have shape [batch, sequence, vocab_size]")
    batch, sequence, vocab_size = student_logits.shape
    if batch <= 0 or sequence <= 0 or vocab_size <= 0:
        raise ValueError("student_logits dimensions must be non-empty")
    return batch, sequence, vocab_size


def _validate_mask(
    mask: Any, *, name: str, batch: int, sequence: int, torch: Any
) -> Any:
    if mask is None:
        raise ValueError(f"{name} is required")
    if not torch.is_tensor(mask) or mask.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, sequence]")
    if mask.shape != (batch, sequence):
        raise ValueError(f"{name} shape must match student_logits")
    valid_values = mask.eq(0) | mask.eq(1)
    if not valid_values.all():
        raise ValueError(f"{name} must contain only 0/1 or bool values")
    mask = mask.to(dtype=torch.bool, device=mask.device)
    if not mask.any():
        raise ValueError(f"{name} must select at least one token")
    return mask


def _validate_loss_weight(name: str, value: float) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return value


def distillation_loss(
    student_logits: Any,
    *,
    torch: Any,
    F: Any,
    teacher_token_ids: Any | None = None,
    teacher_token_mask: Any | None = None,
    teacher_logits: Any | None = None,
    teacher_logits_mask: Any | None = None,
    temperature: float = 1.0,
    token_loss_weight: float = 1.0,
    logit_loss_weight: float = 1.0,
) -> DistillationLossOutput:
    """Compute a masked distillation loss from user-supplied teacher signals."""

    batch, sequence, vocab_size = _validate_student_logits(student_logits)
    if not torch.isfinite(student_logits).all():
        raise ValueError("student_logits must be finite")
    if not isinstance(temperature, int | float) or isinstance(temperature, bool):
        raise ValueError("temperature must be finite and > 0")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and > 0")
    token_loss_weight = _validate_loss_weight("token_loss_weight", token_loss_weight)
    logit_loss_weight = _validate_loss_weight("logit_loss_weight", logit_loss_weight)
    if teacher_token_ids is None and teacher_logits is None:
        raise ValueError(
            "distillation_loss requires teacher_token_ids or teacher_logits"
        )

    token_loss = None
    logit_loss = None
    token_count = 0
    logit_count = 0
    total = student_logits.new_tensor(0.0)
    used_weight = 0.0

    if teacher_token_ids is not None:
        if not torch.is_tensor(teacher_token_ids) or teacher_token_ids.ndim != 2:
            raise ValueError("teacher_token_ids must have shape [batch, sequence]")
        if teacher_token_ids.shape != (batch, sequence):
            raise ValueError("teacher_token_ids shape must match student_logits")
        if teacher_token_ids.dtype not in {torch.int32, torch.int64}:
            raise ValueError("teacher_token_ids must be an integer tensor")
        if (
            teacher_token_ids.min().item() < 0
            or teacher_token_ids.max().item() >= vocab_size
        ):
            raise ValueError("teacher_token_ids must be inside [0, vocab_size)")
        token_mask = _validate_mask(
            teacher_token_mask,
            name="teacher_token_mask",
            batch=batch,
            sequence=sequence,
            torch=torch,
        ).to(device=student_logits.device)
        per_token = F.cross_entropy(
            student_logits.float().reshape(-1, vocab_size),
            teacher_token_ids.to(
                device=student_logits.device, dtype=torch.long
            ).reshape(-1),
            reduction="none",
        ).view(batch, sequence)
        token_loss = (
            per_token * token_mask.float()
        ).sum() / token_mask.sum().clamp_min(1)
        token_count = int(token_mask.sum().item())
        if token_loss_weight > 0.0:
            total = total + token_loss_weight * token_loss.to(
                dtype=student_logits.dtype
            )
            used_weight += token_loss_weight

    if teacher_logits is not None:
        if (
            not torch.is_tensor(teacher_logits)
            or not teacher_logits.is_floating_point()
        ):
            raise ValueError("teacher_logits must be a floating-point tensor")
        if teacher_logits.shape != (batch, sequence, vocab_size):
            raise ValueError("teacher_logits shape must match student_logits")
        if not torch.isfinite(teacher_logits).all():
            raise ValueError("teacher_logits must be finite")
        logit_mask = _validate_mask(
            teacher_logits_mask,
            name="teacher_logits_mask",
            batch=batch,
            sequence=sequence,
            torch=torch,
        ).to(device=student_logits.device)
        student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
        teacher_probs = F.softmax(
            teacher_logits.to(device=student_logits.device).float() / temperature,
            dim=-1,
        )
        per_token_kl = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1) * (temperature * temperature)
        logit_loss = (
            per_token_kl * logit_mask.float()
        ).sum() / logit_mask.sum().clamp_min(1)
        logit_count = int(logit_mask.sum().item())
        if logit_loss_weight > 0.0:
            total = total + logit_loss_weight * logit_loss.to(
                dtype=student_logits.dtype
            )
            used_weight += logit_loss_weight

    if used_weight == 0.0:
        raise ValueError("at least one distillation loss weight must be > 0")
    return DistillationLossOutput(
        loss=total,
        token_loss=token_loss,
        logit_loss=logit_loss,
        token_count=token_count,
        logit_count=logit_count,
    )
