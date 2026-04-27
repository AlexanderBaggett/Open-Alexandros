from __future__ import annotations

import torch
from torch import nn

from alexandros.bitlinear import make_linear
from alexandros.configuration_alexandros import AlexandrosConfig
from alexandros.initialization import initialize_linear, residual_projection_std


class GatedDeltaNetBlock(nn.Module):
    """Runnable Gated DeltaNet-inspired recurrent sequence mixer.

    This is an elementwise reference recurrence. It preserves the design intent:
    linear-time state updates with gates controlling memory retention and delta
    writes, while avoiding custom kernels in v1.
    """

    def __init__(self, config: AlexandrosConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        linear_kwargs = {
            "variant": config.variant,
            "activation_bits": config.bitnet_activation_bits,
        }
        self.q_proj = make_linear(
            config.hidden_size, config.hidden_size, bias=False, **linear_kwargs
        )
        self.k_proj = make_linear(
            config.hidden_size, config.hidden_size, bias=False, **linear_kwargs
        )
        self.v_proj = make_linear(
            config.hidden_size, config.hidden_size, bias=False, **linear_kwargs
        )
        self.alpha_proj = make_linear(
            config.hidden_size, config.hidden_size, bias=True, **linear_kwargs
        )
        self.beta_proj = make_linear(
            config.hidden_size, config.hidden_size, bias=True, **linear_kwargs
        )
        self.out_proj = make_linear(
            config.hidden_size, config.hidden_size, bias=False, **linear_kwargs
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_linear(self.q_proj)
        initialize_linear(self.k_proj)
        initialize_linear(self.v_proj)
        initialize_linear(self.alpha_proj)
        initialize_linear(self.beta_proj)
        initialize_linear(self.out_proj, std=residual_projection_std(self.config))

    def recurrent_state_shape(self, batch_size: int) -> tuple[int, int]:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        return (batch_size, self.hidden_size)

    def _validate_hidden_states(self, x: torch.Tensor) -> None:
        if not torch.is_tensor(x) or not x.is_floating_point():
            raise ValueError(
                "GatedDeltaNet hidden states must be a floating-point tensor"
            )
        if x.ndim != 3:
            raise ValueError(
                "GatedDeltaNet hidden states must have shape [batch, sequence, hidden_size]"
            )
        if x.size(0) == 0:
            raise ValueError("GatedDeltaNet hidden states batch size must be > 0")
        if x.size(1) == 0:
            raise ValueError("GatedDeltaNet hidden states sequence length must be > 0")
        if x.size(-1) != self.hidden_size:
            raise ValueError(
                "GatedDeltaNet hidden states last dimension must match hidden_size"
            )
        if not torch.isfinite(x).all():
            raise ValueError(
                "GatedDeltaNet hidden states must contain only finite values"
            )

    def _validate_or_initialize_state(
        self,
        state: torch.Tensor | None,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if state is None:
            return torch.zeros(
                self.recurrent_state_shape(batch),
                device=device,
                dtype=dtype,
            )
        if not torch.is_tensor(state) or not state.is_floating_point():
            raise ValueError(
                "GatedDeltaNet recurrent state must be a floating-point tensor"
            )
        if state.shape != self.recurrent_state_shape(batch):
            raise ValueError(
                "GatedDeltaNet recurrent state must have shape [batch, hidden_size]"
            )
        if state.device != device:
            raise ValueError(
                "GatedDeltaNet recurrent state device must match hidden states"
            )
        if state.dtype != dtype:
            raise ValueError(
                "GatedDeltaNet recurrent state dtype must match hidden states"
            )
        if not torch.isfinite(state).all():
            raise ValueError(
                "GatedDeltaNet recurrent state must contain only finite values"
            )
        return state

    def _current_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        *,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None:
            return torch.ones(batch, seq_len, device=device, dtype=torch.bool)
        if not torch.is_tensor(attention_mask):
            raise ValueError("attention_mask must be a tensor")
        if attention_mask.ndim != 2 or attention_mask.size(0) != batch:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if attention_mask.size(1) < seq_len:
            raise ValueError(
                "attention_mask length cannot be shorter than current sequence"
            )
        if (
            torch.is_floating_point(attention_mask)
            and not torch.isfinite(attention_mask).all()
        ):
            raise ValueError("attention_mask must contain only finite 0/1 values")
        valid_values = attention_mask.eq(0) | attention_mask.eq(1)
        if not valid_values.all():
            raise ValueError("attention_mask must contain only 0/1 or bool values")
        return attention_mask[:, -seq_len:].to(device=device, dtype=torch.bool)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_hidden_states(x)
        batch, seq_len, hidden = x.shape
        state = self._validate_or_initialize_state(
            state,
            batch=batch,
            device=x.device,
            dtype=x.dtype,
        )
        current_mask = self._current_attention_mask(
            attention_mask,
            batch=batch,
            seq_len=seq_len,
            device=x.device,
        )
        q = torch.tanh(self.q_proj(x))
        k = torch.tanh(self.k_proj(x))
        v = self.v_proj(x)
        alpha = torch.sigmoid(self.alpha_proj(x))
        beta = torch.sigmoid(self.beta_proj(x))

        outputs = []
        for t in range(seq_len):
            prediction = state * k[:, t]
            delta = v[:, t] - prediction
            candidate_state = alpha[:, t] * state + beta[:, t] * delta
            valid = current_mask[:, t].unsqueeze(-1)
            state = torch.where(valid, candidate_state, state)
            outputs.append(torch.where(valid, q[:, t] * state, torch.zeros_like(state)))
        y = torch.stack(outputs, dim=1)
        return self.out_proj(y), state.detach()
