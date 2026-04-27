from __future__ import annotations

import torch
from torch import nn

from alexandros.bitlinear import make_linear
from alexandros.configuration_alexandros import AlexandrosConfig
from alexandros.initialization import initialize_linear, residual_projection_std


class Mamba2Block(nn.Module):
    """Minimal Mamba-2/SSD-style diagonal state-space sequence mixer.

    This is a local, CPU-testable reference backend that preserves the recurrent
    contract used by Alexandros linear-time layers. It is intentionally a small
    diagonal SSM approximation, not the optimized Mamba-2 selective-scan kernel.
    """

    def __init__(self, config: AlexandrosConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        linear_kwargs = {
            "variant": config.variant,
            "activation_bits": config.bitnet_activation_bits,
        }
        self.in_proj = make_linear(
            config.hidden_size, config.hidden_size, bias=False, **linear_kwargs
        )
        self.a_proj = make_linear(
            config.hidden_size, config.hidden_size, bias=True, **linear_kwargs
        )
        self.b_proj = make_linear(
            config.hidden_size, config.hidden_size, bias=True, **linear_kwargs
        )
        self.c_proj = make_linear(
            config.hidden_size, config.hidden_size, bias=True, **linear_kwargs
        )
        self.d_proj = make_linear(
            config.hidden_size, config.hidden_size, bias=False, **linear_kwargs
        )
        self.out_proj = make_linear(
            config.hidden_size, config.hidden_size, bias=False, **linear_kwargs
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_linear(self.in_proj)
        initialize_linear(self.a_proj)
        initialize_linear(self.b_proj)
        initialize_linear(self.c_proj)
        initialize_linear(self.d_proj)
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
            raise ValueError("Mamba2 hidden states must be a floating-point tensor")
        if x.ndim != 3:
            raise ValueError(
                "Mamba2 hidden states must have shape [batch, sequence, hidden_size]"
            )
        if x.size(0) == 0:
            raise ValueError("Mamba2 hidden states batch size must be > 0")
        if x.size(1) == 0:
            raise ValueError("Mamba2 hidden states sequence length must be > 0")
        if x.size(-1) != self.hidden_size:
            raise ValueError(
                "Mamba2 hidden states last dimension must match hidden_size"
            )
        if not torch.isfinite(x).all():
            raise ValueError("Mamba2 hidden states must contain only finite values")

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
            raise ValueError("Mamba2 recurrent state must be a floating-point tensor")
        if state.shape != self.recurrent_state_shape(batch):
            raise ValueError(
                "Mamba2 recurrent state must have shape [batch, hidden_size]"
            )
        if state.device != device:
            raise ValueError("Mamba2 recurrent state device must match hidden states")
        if state.dtype != dtype:
            raise ValueError("Mamba2 recurrent state dtype must match hidden states")
        if not torch.isfinite(state).all():
            raise ValueError("Mamba2 recurrent state must contain only finite values")
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
        batch, seq_len, _ = x.shape
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

        content = torch.tanh(self.in_proj(x))
        retention = torch.sigmoid(self.a_proj(x))
        write_gate = torch.sigmoid(self.b_proj(x))
        read_gate = torch.sigmoid(self.c_proj(x))
        skip = self.d_proj(x)

        outputs = []
        for t in range(seq_len):
            a_t = retention[:, t]
            b_x_t = (1.0 - a_t) * write_gate[:, t] * content[:, t]
            candidate_state = a_t * state + b_x_t
            valid = current_mask[:, t].unsqueeze(-1)
            state = torch.where(valid, candidate_state, state)
            y_t = read_gate[:, t] * state + skip[:, t]
            outputs.append(torch.where(valid, y_t, torch.zeros_like(y_t)))
        y = torch.stack(outputs, dim=1)
        return self.out_proj(y), state.detach()
