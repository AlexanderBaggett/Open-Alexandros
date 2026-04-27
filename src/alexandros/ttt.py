from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from alexandros.configuration_alexandros import AlexandrosConfig


def _validate_float_dtype(dtype: torch.dtype) -> None:
    try:
        is_float_dtype = torch.empty((), dtype=dtype).is_floating_point()
    except TypeError:
        is_float_dtype = False
    if not is_float_dtype:
        raise ValueError("dtype must be a floating-point torch dtype")


def _validate_rank(hidden_size: int, rank: int) -> None:
    if (
        not isinstance(hidden_size, int)
        or isinstance(hidden_size, bool)
        or hidden_size <= 0
    ):
        raise ValueError("hidden_size must be a positive integer")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if rank > hidden_size:
        raise ValueError("rank cannot exceed hidden_size")


def _validate_hidden_states(
    hidden_states: torch.Tensor,
    *,
    hidden_size: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    if not torch.is_tensor(hidden_states) or not hidden_states.is_floating_point():
        raise ValueError("hidden_states must be a floating-point tensor")
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden_size]")
    if hidden_states.size(0) == 0:
        raise ValueError("hidden_states batch size must be > 0")
    if hidden_states.size(1) == 0:
        raise ValueError("hidden_states sequence length must be > 0")
    if hidden_states.size(-1) != hidden_size:
        raise ValueError("hidden_states last dimension must match TTT hidden_size")
    if device is not None and hidden_states.device != device:
        raise ValueError("hidden_states device must match TTT state device")
    if dtype is not None and hidden_states.dtype != dtype:
        raise ValueError("hidden_states dtype must match TTT state dtype")
    if not torch.isfinite(hidden_states).all():
        raise ValueError("hidden_states must contain only finite values")


def _validate_fast_weights(
    fast_a: torch.Tensor,
    fast_b: torch.Tensor,
    *,
    hidden_size: int,
    rank: int,
) -> None:
    if not torch.is_tensor(fast_a) or not torch.is_tensor(fast_b):
        raise ValueError("fast weights must be tensors")
    if not fast_a.is_floating_point() or not fast_b.is_floating_point():
        raise ValueError("fast weights must be floating-point tensors")
    if tuple(fast_a.shape) != (hidden_size, rank):
        raise ValueError("fast_a must have shape [hidden_size, rank]")
    if tuple(fast_b.shape) != (rank, hidden_size):
        raise ValueError("fast_b must have shape [rank, hidden_size]")
    if fast_a.device != fast_b.device:
        raise ValueError("fast weights must be on the same device")
    if fast_a.dtype != fast_b.dtype:
        raise ValueError("fast weights must have the same dtype")
    if not torch.isfinite(fast_a).all() or not torch.isfinite(fast_b).all():
        raise ValueError("fast weights must contain only finite values")


def _validate_gate(
    gate: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if not torch.is_tensor(gate) or not gate.is_floating_point():
        raise ValueError("gate must be a floating-point tensor")
    if gate.device != hidden_states.device:
        raise ValueError("gate device must match hidden_states device")
    if gate.dtype != hidden_states.dtype:
        raise ValueError("gate dtype must match hidden_states dtype")
    if not torch.isfinite(gate).all():
        raise ValueError("gate must contain only finite values")
    if gate.ndim == 0:
        gate = gate.reshape(1, 1, 1)
    elif gate.shape == hidden_states.shape[:-1]:
        gate = gate.unsqueeze(-1)
    elif gate.shape != (*hidden_states.shape[:-1], 1):
        raise ValueError(
            "gate must be a scalar, [batch, sequence], or [batch, sequence, 1]"
        )
    if gate.lt(0).any() or gate.gt(1).any():
        raise ValueError("gate values must be in [0, 1]")
    return gate


def ttt_next_token_loss_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Next-token loss used by the TTT inner and outer objectives."""

    if not torch.is_tensor(logits) or not logits.is_floating_point():
        raise ValueError("logits must be a floating-point tensor")
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocab_size]")
    if not torch.is_tensor(input_ids) or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be a torch.long tensor")
    if input_ids.shape != logits.shape[:2]:
        raise ValueError("input_ids shape must match logits batch and sequence")
    if logits.size(1) < 2:
        raise ValueError("next-token TTT loss requires chunk length >= 2")
    if logits.size(-1) <= 0:
        raise ValueError("logits vocab dimension must be non-empty")
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
    )
    if not torch.isfinite(loss.detach()):
        raise FloatingPointError(f"non-finite TTT loss encountered: {loss.item()}")
    return loss


@dataclass(frozen=True)
class TTTInnerUpdateOutput:
    """Result of one differentiable TTT-E2E-style inner update."""

    fast_a: torch.Tensor
    fast_b: torch.Tensor
    loss: torch.Tensor
    grad_norm: torch.Tensor


class TTTState:
    """Request-local low-rank fast weights inspired by TTT-E2E."""

    def __init__(
        self, hidden_size: int, rank: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        _validate_rank(hidden_size, rank)
        _validate_float_dtype(dtype)
        self.hidden_size = hidden_size
        self.rank = rank
        self.fast_a = torch.zeros(hidden_size, rank, device=device, dtype=dtype)
        self.fast_b = torch.zeros(rank, hidden_size, device=device, dtype=dtype)
        self.steps = 0

    @classmethod
    def from_config(
        cls,
        config: AlexandrosConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "TTTState":
        return cls(config.hidden_size, config.ttt_rank, device, dtype)

    @classmethod
    def from_fast_weights(
        cls,
        fast_a: torch.Tensor,
        fast_b: torch.Tensor,
        *,
        detach: bool = True,
    ) -> "TTTState":
        if not torch.is_tensor(fast_a) or not torch.is_tensor(fast_b):
            raise ValueError("fast weights must be tensors")
        if fast_a.ndim != 2 or fast_b.ndim != 2:
            raise ValueError("fast weights must be rank-2 tensors")
        hidden_size, rank = fast_a.shape
        _validate_rank(int(hidden_size), int(rank))
        _validate_fast_weights(
            fast_a,
            fast_b,
            hidden_size=int(hidden_size),
            rank=int(rank),
        )
        state = cls(int(hidden_size), int(rank), fast_a.device, fast_a.dtype)
        with torch.no_grad():
            source_a = fast_a.detach() if detach else fast_a
            source_b = fast_b.detach() if detach else fast_b
            state.fast_a.copy_(source_a)
            state.fast_b.copy_(source_b)
        return state

    def _validate_hidden_states(self, hidden_states: torch.Tensor) -> None:
        _validate_hidden_states(
            hidden_states,
            hidden_size=self.hidden_size,
            device=self.fast_a.device,
            dtype=self.fast_a.dtype,
        )

    @torch.no_grad()
    def reset(self) -> "TTTState":
        self.fast_a.zero_()
        self.fast_b.zero_()
        self.steps = 0
        return self

    def _validate_gate(
        self,
        gate: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return _validate_gate(gate, hidden_states)

    @torch.no_grad()
    def prefill_update(
        self,
        hidden_states: torch.Tensor,
        lr: float = 1e-3,
        *,
        generator: torch.Generator | None = None,
        decay: float = 0.99,
    ) -> "TTTState":
        self._validate_hidden_states(hidden_states)
        if not math.isfinite(float(lr)) or lr <= 0:
            raise ValueError("lr must be finite and > 0")
        if not math.isfinite(float(decay)) or not 0.0 <= decay < 1.0:
            raise ValueError("decay must be finite and satisfy 0 <= decay < 1")
        pooled = hidden_states.detach().mean(dim=(0, 1))
        if pooled.norm() == 0:
            return self
        direction = pooled / pooled.norm().clamp_min(1e-6)
        basis = torch.randn(
            self.hidden_size,
            self.rank,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            generator=generator,
        )
        basis = torch.linalg.qr(basis, mode="reduced").Q
        self.fast_a.mul_(decay).add_(basis * lr)
        self.fast_b.mul_(decay).add_(
            basis.transpose(0, 1) * direction.unsqueeze(0) * lr
        )
        self.steps += 1
        return self

    @torch.no_grad()
    def update(
        self,
        hidden_states: torch.Tensor,
        lr: float = 1e-3,
        *,
        generator: torch.Generator | None = None,
        decay: float = 0.99,
    ) -> "TTTState":
        return self.prefill_update(
            hidden_states, lr=lr, generator=generator, decay=decay
        )

    def apply(
        self,
        hidden_states: torch.Tensor,
        scale: float = 1.0,
        *,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_hidden_states(hidden_states)
        if not math.isfinite(float(scale)):
            raise ValueError("scale must be finite")
        adapted = hidden_states @ self.fast_a @ self.fast_b
        if gate is not None:
            adapted = adapted * self._validate_gate(gate, hidden_states)
        return hidden_states + adapted * scale


class TTTMetaAdapter(nn.Module):
    """Trainable low-rank TTT fast-weight initialization and update rule.

    The adapter owns learned ``phi_0`` fast-weight initialization parameters.
    Inner updates operate on request-local tensors derived from these
    parameters, so inference adaptation never mutates checkpoint weights.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        *,
        init_std: float = 1e-3,
        adapter_scale: float = 1.0,
    ) -> None:
        super().__init__()
        _validate_rank(hidden_size, rank)
        if not math.isfinite(float(init_std)) or init_std < 0.0:
            raise ValueError("init_std must be finite and >= 0")
        if not math.isfinite(float(adapter_scale)):
            raise ValueError("adapter_scale must be finite")
        self.hidden_size = hidden_size
        self.rank = rank
        self.adapter_scale = float(adapter_scale)
        self.init_a = nn.Parameter(torch.empty(hidden_size, rank))
        self.init_b = nn.Parameter(torch.empty(rank, hidden_size))
        nn.init.normal_(self.init_a, mean=0.0, std=float(init_std))
        nn.init.normal_(self.init_b, mean=0.0, std=float(init_std))

    @classmethod
    def from_config(
        cls,
        config: AlexandrosConfig,
        *,
        init_std: float = 1e-3,
        adapter_scale: float = 1.0,
    ) -> "TTTMetaAdapter":
        return cls(
            config.hidden_size,
            config.ttt_rank,
            init_std=init_std,
            adapter_scale=adapter_scale,
        )

    def initial_fast_weights(
        self,
        *,
        detach: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fast_a: torch.Tensor = self.init_a
        fast_b: torch.Tensor = self.init_b
        if detach:
            fast_a = fast_a.detach().clone()
            fast_b = fast_b.detach().clone()
        if device is not None or dtype is not None:
            fast_a = fast_a.to(device=device, dtype=dtype)
            fast_b = fast_b.to(device=device, dtype=dtype)
        return fast_a, fast_b

    def request_state(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> TTTState:
        """Clone learned initialization into an isolated request-local state."""

        fast_a, fast_b = self.initial_fast_weights(
            detach=True,
            device=device,
            dtype=dtype,
        )
        return TTTState.from_fast_weights(fast_a, fast_b, detach=True)

    def apply(
        self,
        hidden_states: torch.Tensor,
        fast_a: torch.Tensor | None = None,
        fast_b: torch.Tensor | None = None,
        *,
        scale: float | None = None,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if fast_a is None or fast_b is None:
            if fast_a is not None or fast_b is not None:
                raise ValueError("fast_a and fast_b must be provided together")
            fast_a, fast_b = self.initial_fast_weights()
        _validate_hidden_states(hidden_states, hidden_size=self.hidden_size)
        _validate_fast_weights(
            fast_a,
            fast_b,
            hidden_size=self.hidden_size,
            rank=self.rank,
        )
        if hidden_states.device != fast_a.device:
            raise ValueError("hidden_states device must match fast weights device")
        if hidden_states.dtype != fast_a.dtype:
            raise ValueError("hidden_states dtype must match fast weights dtype")
        actual_scale = self.adapter_scale if scale is None else float(scale)
        if not math.isfinite(actual_scale):
            raise ValueError("scale must be finite")
        adapted = hidden_states @ fast_a @ fast_b
        if gate is not None:
            adapted = adapted * _validate_gate(gate, hidden_states)
        return hidden_states + adapted * actual_scale

    def inner_update(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        lm_head: nn.Module,
        fast_a: torch.Tensor,
        fast_b: torch.Tensor,
        *,
        inner_lr: float,
        create_graph: bool = True,
    ) -> TTTInnerUpdateOutput:
        if not math.isfinite(float(inner_lr)) or inner_lr <= 0.0:
            raise ValueError("inner_lr must be finite and > 0")
        adapted = self.apply(hidden_states, fast_a, fast_b)
        logits = lm_head(adapted)
        loss = ttt_next_token_loss_from_logits(logits, input_ids)
        grad_a, grad_b = torch.autograd.grad(
            loss,
            (fast_a, fast_b),
            create_graph=create_graph,
            retain_graph=create_graph,
        )
        grad_norm = torch.linalg.vector_norm(
            torch.stack((grad_a.detach().norm(), grad_b.detach().norm()))
        )
        updated_a = fast_a - float(inner_lr) * grad_a
        updated_b = fast_b - float(inner_lr) * grad_b
        return TTTInnerUpdateOutput(
            fast_a=updated_a,
            fast_b=updated_b,
            loss=loss,
            grad_norm=grad_norm,
        )
