from __future__ import annotations

from typing import Any

COMMON_TRAINING_METRIC_KEYS: tuple[str, ...] = (
    "phase",
    "step",
    "objective_phase",
    "objective_name",
    "objective_inputs",
    "objective_targets",
    "objective_normalization",
    "trainability_phase",
    "trainability_scope",
    "trainability_trainable_parameters",
    "trainability_frozen_parameters",
    "data_source",
)

PHASE_TRAINING_METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "ar": (
        "loss",
        "grad_norm",
        "lr",
        "logit_norm",
        "grad_accum_steps",
        "amp_enabled",
        "grad_scaler_enabled",
    ),
    "diffusion": (
        "loss",
        "grad_norm",
        "lr",
        "logit_norm",
        "diffusion_objective",
        "diffusion_loss_weighting",
        "moe_mean_load_entropy",
        "moe_timestep_tracked_selections",
        "moe_timestep_load_entropy",
        "moe_noisy_step_load_entropy",
        "moe_polish_step_load_entropy",
        "grad_accum_steps",
        "amp_enabled",
        "grad_scaler_enabled",
    ),
    "latent": (
        "loss",
        "reconstruction_loss",
        "vae_reconstruction_loss",
        "refinement_reconstruction_loss",
        "kl_loss",
        "grad_norm",
        "lr",
        "latent_norm",
        "latent_update_norm",
        "reconstruction_norm",
        "latent_refinement_steps",
        "lambda_kl",
        "lambda_rec",
        "grad_accum_steps",
        "amp_enabled",
        "grad_scaler_enabled",
    ),
    "ttt": (
        "loss",
        "outer_loss",
        "inner_loss",
        "grad_norm",
        "lr",
        "ttt_inner_lr",
        "pre_update_loss",
        "post_update_loss",
        "ttt_steps",
        "prefill_chunk_len",
        "prefill_chunk_count",
        "prefill_chunk_loss",
        "ttt_inner_grad_norm",
        "ttt_heldout_chunk_len",
        "ttt_meta_trainable_parameters",
        "ttt_checkpoint_path",
        "hidden_norm_before",
        "hidden_norm_after",
        "hidden_delta_norm",
        "grad_accum_steps",
        "amp_enabled",
        "grad_scaler_enabled",
    ),
}


def standard_metric_names(
    phase: str | None = None,
) -> tuple[str, ...] | dict[str, tuple[str, ...]]:
    """Return the required JSONL metric names for smoke training logs."""

    if phase is None:
        return {
            name: (*COMMON_TRAINING_METRIC_KEYS, *keys)
            for name, keys in PHASE_TRAINING_METRIC_KEYS.items()
        }
    if phase not in PHASE_TRAINING_METRIC_KEYS:
        raise ValueError(f"unknown training phase: {phase}")
    return (*COMMON_TRAINING_METRIC_KEYS, *PHASE_TRAINING_METRIC_KEYS[phase])


def missing_standard_metric_names(
    record: dict[str, Any], phase: str | None = None
) -> tuple[str, ...]:
    actual_phase = record.get("phase") if phase is None else phase
    if not isinstance(actual_phase, str):
        raise ValueError("metric record phase must be a string")
    expected = standard_metric_names(actual_phase)
    assert isinstance(expected, tuple)
    return tuple(key for key in expected if key not in record)


def validate_standard_metric_record(
    record: dict[str, Any], phase: str | None = None
) -> None:
    missing = missing_standard_metric_names(record, phase)
    if missing:
        raise ValueError(
            f"metric record is missing required fields: {', '.join(missing)}"
        )
