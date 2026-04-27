from __future__ import annotations

import importlib
from dataclasses import fields
from typing import Any

import torch
from torch import nn

from alexandros.configuration_alexandros import AlexandrosConfig
from alexandros.modeling_alexandros import (
    AlexandrosForCausalLM,
    AlexandrosForDiffusionLM,
    AlexandrosModel,
)

_INSTALL_HINT = "Install the optional Hugging Face extra with `pip install -e .[hf]`."


def transformers_available() -> bool:
    return importlib.util.find_spec("transformers") is not None


def _require_transformers() -> Any:
    try:
        return importlib.import_module("transformers")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Hugging Face compatibility requires the optional `transformers` "
            f"dependency. {_INSTALL_HINT}"
        ) from exc


def _alexandros_config_field_names() -> set[str]:
    return {field.name for field in fields(AlexandrosConfig)}


def _to_alexandros_config(config: Any) -> AlexandrosConfig:
    if isinstance(config, AlexandrosConfig):
        return config
    if hasattr(config, "to_alexandros_config"):
        return config.to_alexandros_config()
    raise TypeError("config must be an AlexandrosConfig or AlexandrosHFConfig")


if transformers_available():
    _transformers = _require_transformers()
    PretrainedConfig = _transformers.PretrainedConfig
    PreTrainedModel = _transformers.PreTrainedModel
    AutoConfig = _transformers.AutoConfig
    AutoModel = _transformers.AutoModel
    AutoModelForCausalLM = _transformers.AutoModelForCausalLM
    modeling_outputs = importlib.import_module("transformers.modeling_outputs")
    BaseModelOutputWithPast = modeling_outputs.BaseModelOutputWithPast
    CausalLMOutputWithPast = modeling_outputs.CausalLMOutputWithPast

    class AlexandrosHFConfig(PretrainedConfig):
        """Optional Hugging Face config wrapper for AlexandrosConfig."""

        model_type = "alexandros"

        def __init__(self, **kwargs: Any) -> None:
            field_names = _alexandros_config_field_names()
            config_kwargs = {
                key: kwargs.pop(key) for key in list(kwargs) if key in field_names
            }
            alexandros_config = AlexandrosConfig.from_dict(config_kwargs)
            super().__init__(
                bos_token_id=alexandros_config.bos_token_id,
                eos_token_id=alexandros_config.eos_token_id,
                pad_token_id=alexandros_config.pad_token_id,
                tie_word_embeddings=alexandros_config.tie_word_embeddings,
                **kwargs,
            )
            for key, value in alexandros_config.to_dict().items():
                setattr(self, key, value)

        @classmethod
        def from_alexandros_config(
            cls, config: AlexandrosConfig
        ) -> "AlexandrosHFConfig":
            return cls(**config.to_dict())

        def to_alexandros_config(self) -> AlexandrosConfig:
            field_names = _alexandros_config_field_names()
            return AlexandrosConfig.from_dict(
                {key: getattr(self, key) for key in field_names if hasattr(self, key)}
            )

    class AlexandrosHFModel(PreTrainedModel):
        """HF-native base model wrapper around the reference AlexandrosModel."""

        config_class = AlexandrosHFConfig
        base_model_prefix = "alexandros"
        supports_gradient_checkpointing = False

        def __init__(self, config: AlexandrosHFConfig | AlexandrosConfig) -> None:
            hf_config = (
                config
                if isinstance(config, AlexandrosHFConfig)
                else AlexandrosHFConfig.from_alexandros_config(config)
            )
            super().__init__(hf_config)
            self.alexandros = AlexandrosModel(hf_config.to_alexandros_config())

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            past_key_values: list[dict[str, torch.Tensor] | None] | None = None,
            use_cache: bool | None = None,
            output_hidden_states: bool | None = None,
            **kwargs: Any,
        ) -> Any:
            if input_ids is None:
                raise ValueError("input_ids is required")
            if kwargs:
                raise TypeError(
                    f"unsupported AlexandrosHFModel kwargs: {sorted(kwargs)}"
                )
            output = self.alexandros(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=bool(use_cache),
                output_hidden_states=bool(output_hidden_states),
            )
            return BaseModelOutputWithPast(
                last_hidden_state=output.last_hidden_state,
                past_key_values=output.past_key_values,
                hidden_states=output.hidden_states,
            )

    class AlexandrosHFForCausalLM(PreTrainedModel):
        """HF-native causal LM wrapper around AlexandrosForCausalLM."""

        config_class = AlexandrosHFConfig
        base_model_prefix = "alexandros"
        supports_gradient_checkpointing = False

        def __init__(self, config: AlexandrosHFConfig | AlexandrosConfig) -> None:
            hf_config = (
                config
                if isinstance(config, AlexandrosHFConfig)
                else AlexandrosHFConfig.from_alexandros_config(config)
            )
            super().__init__(hf_config)
            self.alexandros = AlexandrosForCausalLM(hf_config.to_alexandros_config())

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            past_key_values: list[dict[str, torch.Tensor] | None] | None = None,
            use_cache: bool | None = None,
            output_hidden_states: bool | None = None,
            **kwargs: Any,
        ) -> Any:
            if input_ids is None:
                raise ValueError("input_ids is required")
            if kwargs:
                raise TypeError(
                    f"unsupported AlexandrosHFForCausalLM kwargs: {sorted(kwargs)}"
                )
            output = self.alexandros(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=bool(use_cache),
                output_hidden_states=bool(output_hidden_states),
            )
            return CausalLMOutputWithPast(
                loss=output.loss,
                logits=output.logits,
                past_key_values=output.past_key_values,
                hidden_states=output.hidden_states,
            )

        def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return self.alexandros.generate(input_ids, **kwargs)

    class AlexandrosHFForDiffusionLM(PreTrainedModel):
        """HF-native wrapper for Alexandros diffusion/latent generation helpers."""

        config_class = AlexandrosHFConfig
        base_model_prefix = "alexandros"
        supports_gradient_checkpointing = False

        def __init__(self, config: AlexandrosHFConfig | AlexandrosConfig) -> None:
            hf_config = (
                config
                if isinstance(config, AlexandrosHFConfig)
                else AlexandrosHFConfig.from_alexandros_config(config)
            )
            super().__init__(hf_config)
            self.alexandros = AlexandrosForDiffusionLM(hf_config.to_alexandros_config())

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            past_key_values: list[dict[str, torch.Tensor] | None] | None = None,
            use_cache: bool | None = None,
            output_hidden_states: bool | None = None,
            **kwargs: Any,
        ) -> Any:
            if input_ids is None:
                raise ValueError("input_ids is required")
            if kwargs:
                raise TypeError(
                    f"unsupported AlexandrosHFForDiffusionLM kwargs: {sorted(kwargs)}"
                )
            output = self.alexandros(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=bool(use_cache),
                output_hidden_states=bool(output_hidden_states),
            )
            return CausalLMOutputWithPast(
                loss=output.loss,
                logits=output.logits,
                past_key_values=output.past_key_values,
                hidden_states=output.hidden_states,
            )

        def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return self.alexandros.generate(input_ids, **kwargs)

else:

    class AlexandrosHFConfig:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_transformers()

    class AlexandrosHFModel(nn.Module):  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_transformers()
            super().__init__()

    class AlexandrosHFForCausalLM(nn.Module):  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_transformers()
            super().__init__()

    class AlexandrosHFForDiffusionLM(nn.Module):  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_transformers()
            super().__init__()


def _register(auto_class: Any, *args: Any) -> None:
    try:
        auto_class.register(*args, exist_ok=True)
    except TypeError:
        try:
            auto_class.register(*args)
        except ValueError as exc:
            if "already" not in str(exc).lower():
                raise


def register_alexandros_with_transformers() -> None:
    """Register Alexandros optional HF classes with Transformers AutoClasses."""

    transformers = _require_transformers()
    _register(
        transformers.AutoConfig, AlexandrosHFConfig.model_type, AlexandrosHFConfig
    )
    _register(transformers.AutoModel, AlexandrosHFConfig, AlexandrosHFModel)
    _register(
        transformers.AutoModelForCausalLM,
        AlexandrosHFConfig,
        AlexandrosHFForCausalLM,
    )


__all__ = [
    "AlexandrosHFConfig",
    "AlexandrosHFModel",
    "AlexandrosHFForCausalLM",
    "AlexandrosHFForDiffusionLM",
    "register_alexandros_with_transformers",
    "transformers_available",
]
