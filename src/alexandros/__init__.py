"""Open-Alexandros research prototype."""

from alexandros.configuration_alexandros import AlexandrosConfig

__all__ = [
    "AlexandrosConfig",
    "AlexandrosModel",
    "AlexandrosForCausalLM",
    "AlexandrosForDiffusionLM",
    "AlexandrosHFConfig",
    "AlexandrosHFModel",
    "AlexandrosHFForCausalLM",
    "AlexandrosHFForDiffusionLM",
    "GenerationMode",
    "register_alexandros_with_transformers",
    "transformers_available",
]


def __getattr__(name: str):
    if name in {
        "AlexandrosModel",
        "AlexandrosForCausalLM",
        "AlexandrosForDiffusionLM",
        "GenerationMode",
    }:
        from alexandros import modeling_alexandros

        value = getattr(modeling_alexandros, name)
        globals()[name] = value
        return value
    if name in {
        "AlexandrosHFConfig",
        "AlexandrosHFModel",
        "AlexandrosHFForCausalLM",
        "AlexandrosHFForDiffusionLM",
        "register_alexandros_with_transformers",
        "transformers_available",
    }:
        from alexandros import hf_compat

        value = getattr(hf_compat, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'alexandros' has no attribute {name!r}")
