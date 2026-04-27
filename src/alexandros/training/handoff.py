from __future__ import annotations

from typing import Any

from alexandros.training.objectives import objective_contract
from alexandros.training.trainability import TrainabilityReport

PHASE_MODULE_CONTRACTS: dict[str, dict[str, list[str]]] = {
    "ar": {
        "enabled_modules": ["model", "lm_head"],
        "disabled_modules": ["latent_vae", "latent_reasoner"],
    },
    "diffusion": {
        "enabled_modules": ["model", "lm_head"],
        "disabled_modules": ["latent_vae", "latent_reasoner"],
    },
    "latent": {
        "enabled_modules": ["model", "latent_vae", "latent_reasoner"],
        "disabled_modules": ["lm_head"],
    },
    "ttt": {
        "enabled_modules": [],
        "disabled_modules": ["model", "lm_head", "latent_vae", "latent_reasoner"],
    },
}


def phase_module_contract(phase: str) -> dict[str, list[str]]:
    try:
        contract = PHASE_MODULE_CONTRACTS[phase]
    except KeyError as exc:
        raise ValueError(f"unknown handoff phase: {phase}") from exc
    return {
        "enabled_modules": list(contract["enabled_modules"]),
        "disabled_modules": list(contract["disabled_modules"]),
    }


def phase_checkpoint_metadata(
    phase: str,
    trainability: TrainabilityReport,
) -> dict[str, Any]:
    modules = phase_module_contract(phase)
    return {
        "phase_handoff_contract_version": 1,
        "phase": phase,
        "objective": objective_contract(phase).to_dict(),
        "trainability": trainability.to_dict(),
        **modules,
        "migration": {
            "AlexandrosForCausalLM->AlexandrosForDiffusionLM": (
                "initialize missing latent_vae and latent_reasoner parameters"
            ),
            "AlexandrosForDiffusionLM->AlexandrosForCausalLM": (
                "ignore extra latent_vae and latent_reasoner parameters"
            ),
        },
    }
