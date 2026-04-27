from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from alexandros.bitlinear import make_linear
from alexandros.configuration_alexandros import AlexandrosConfig
from alexandros.initialization import initialize_linear, residual_projection_std


class MatrixGatedDeltaNetBlock(nn.Module):
    """Per-head matrix-state Gated DeltaNet reference mixer.

    This backend implements the research-matrix recurrence:

    S_t = alpha_t * S_{t-1} + beta_t * outer(v_t - S_{t-1} k_t, k_t)
    y_t = S_t q_t

    It is a readable PyTorch reference path. `deltanet_chunk_size` slices long
    prefill into causal chunks while preserving the exact recurrent state
    transition; optimized associative scans can replace the chunk loop later.
    """

    def __init__(self, config: AlexandrosConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.chunk_size = config.deltanet_chunk_size
        self.state_clip = config.deltanet_state_clip
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
            config.hidden_size, config.num_attention_heads, bias=True, **linear_kwargs
        )
        self.beta_proj = make_linear(
            config.hidden_size, config.num_attention_heads, bias=True, **linear_kwargs
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

    def recurrent_state_shape(self, batch_size: int) -> tuple[int, int, int, int]:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        return (batch_size, self.num_heads, self.head_dim, self.head_dim)

    def _validate_hidden_states(self, x: torch.Tensor) -> None:
        if not torch.is_tensor(x) or not x.is_floating_point():
            raise ValueError(
                "MatrixGatedDeltaNet hidden states must be a floating-point tensor"
            )
        if x.ndim != 3:
            raise ValueError(
                "MatrixGatedDeltaNet hidden states must have shape "
                "[batch, sequence, hidden_size]"
            )
        if x.size(0) == 0:
            raise ValueError("MatrixGatedDeltaNet hidden states batch size must be > 0")
        if x.size(1) == 0:
            raise ValueError(
                "MatrixGatedDeltaNet hidden states sequence length must be > 0"
            )
        if x.size(-1) != self.hidden_size:
            raise ValueError(
                "MatrixGatedDeltaNet hidden states last dimension must match hidden_size"
            )
        if not torch.isfinite(x).all():
            raise ValueError(
                "MatrixGatedDeltaNet hidden states must contain only finite values"
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
                "MatrixGatedDeltaNet recurrent state must be a floating-point tensor"
            )
        if state.shape != self.recurrent_state_shape(batch):
            raise ValueError(
                "MatrixGatedDeltaNet recurrent state must have shape "
                "[batch, heads, value_dim, key_dim]"
            )
        if state.device != device:
            raise ValueError(
                "MatrixGatedDeltaNet recurrent state device must match hidden states"
            )
        if state.dtype != dtype:
            raise ValueError(
                "MatrixGatedDeltaNet recurrent state dtype must match hidden states"
            )
        if not torch.isfinite(state).all():
            raise ValueError(
                "MatrixGatedDeltaNet recurrent state must contain only finite values"
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

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _project_inputs(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = F.normalize(self._shape_heads(self.q_proj(x)), dim=-1, eps=1e-6)
        k = F.normalize(self._shape_heads(self.k_proj(x)), dim=-1, eps=1e-6)
        v = torch.tanh(self._shape_heads(self.v_proj(x)))
        alpha = torch.sigmoid(self.alpha_proj(x)).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        return q, k, v, alpha, beta

    def _run_recurrence(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        mask: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        seq_len = q.size(2)
        for t in range(seq_len):
            k_t = k[:, :, t]
            q_t = q[:, :, t]
            v_t = v[:, :, t]
            read = torch.einsum("bhvk,bhk->bhv", state, k_t)
            delta = v_t - read
            write = torch.einsum("bhv,bhk->bhvk", delta, k_t)
            candidate_state = (
                alpha[:, :, t].unsqueeze(-1).unsqueeze(-1) * state
                + beta[:, :, t].unsqueeze(-1).unsqueeze(-1) * write
            ).clamp(-self.state_clip, self.state_clip)
            valid_state = mask[:, t].view(mask.size(0), 1, 1, 1)
            state = torch.where(valid_state, candidate_state, state)
            y_t = torch.einsum("bhvk,bhk->bhv", state, q_t)
            valid_output = mask[:, t].view(mask.size(0), 1, 1)
            outputs.append(torch.where(valid_output, y_t, torch.zeros_like(y_t)))
        return torch.stack(outputs, dim=2), state

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
        q, k, v, alpha, beta = self._project_inputs(x)

        outputs = []
        if self.chunk_size > 0:
            ranges = range(0, seq_len, self.chunk_size)
        else:
            ranges = range(0, seq_len, seq_len)
        for start in ranges:
            end = min(start + (self.chunk_size or seq_len), seq_len)
            chunk_y, state = self._run_recurrence(
                q[:, :, start:end],
                k[:, :, start:end],
                v[:, :, start:end],
                alpha[:, :, start:end],
                beta[:, :, start:end],
                current_mask[:, start:end],
                state,
            )
            outputs.append(chunk_y)
        y = torch.cat(outputs, dim=2).transpose(1, 2).contiguous()
        y = y.view(batch, seq_len, self.hidden_size)
        return self.out_proj(y), state.detach()

    def decode_step(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(x) or x.ndim != 2:
            raise ValueError(
                "decode_step hidden states must have shape [batch, hidden_size]"
            )
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(1)
        y, next_state = self.forward(
            x.unsqueeze(1),
            state=state,
            attention_mask=attention_mask,
        )
        return y[:, 0], next_state
