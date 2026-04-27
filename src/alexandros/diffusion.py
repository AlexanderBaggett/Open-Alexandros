from __future__ import annotations

import torch

from alexandros.configuration_alexandros import AlexandrosConfig

_INPUT_ID_DTYPES = {torch.int32, torch.int64}


class MaskedDiffusionScheduler:
    """Absorbing-mask scheduler for masked/block diffusion language modeling."""

    def __init__(self, config: AlexandrosConfig) -> None:
        self.config = config

    def _as_timestep_tensor(
        self,
        timestep: int | torch.Tensor,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if isinstance(timestep, bool):
            raise ValueError("timesteps must be integer values")
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=device)
        elif device is not None:
            timestep = timestep.to(device=device)
        if timestep.dtype == torch.bool:
            raise ValueError("timesteps must be integer values")
        if timestep.numel() == 0:
            raise ValueError("timesteps cannot be empty")
        if torch.is_floating_point(timestep):
            if not torch.isfinite(timestep).all():
                raise ValueError("timesteps must be finite integer values")
            if not torch.equal(timestep, timestep.round()):
                raise ValueError("timesteps must be integer values")
        timestep = timestep.to(dtype=torch.long)
        if (
            timestep.min().item() < 0
            or timestep.max().item() >= self.config.diffusion_steps
        ):
            raise ValueError("timesteps must be in [0, diffusion_steps)")
        return timestep

    def _normalize_timestep_for_input(
        self,
        timestep: int | torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        timestep = self._as_timestep_tensor(timestep, device=input_ids.device)
        batch, seq_len = input_ids.shape
        if timestep.ndim == 0:
            return timestep
        if timestep.ndim == 1 and timestep.size(0) in {1, batch}:
            return timestep
        if (
            timestep.ndim == 2
            and timestep.size(0) in {1, batch}
            and timestep.size(1) in {1, seq_len}
        ):
            return timestep
        raise ValueError(
            "timesteps must be scalar, [batch], [1], [batch, sequence], "
            "or broadcastable [1, 1]/[batch, 1]"
        )

    def _validate_input_ids(self, input_ids: torch.Tensor) -> None:
        if not torch.is_tensor(input_ids):
            raise ValueError("input_ids must be a torch tensor")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in _INPUT_ID_DTYPES:
            raise ValueError("input_ids must be an integer tensor of token IDs")
        if input_ids.size(0) == 0:
            raise ValueError("input_ids batch size must be > 0")
        if input_ids.size(1) == 0:
            raise ValueError("input_ids sequence length must be > 0")
        if (
            input_ids.min().item() < 0
            or input_ids.max().item() >= self.config.vocab_size
        ):
            raise ValueError("input_ids contain token IDs outside [0, vocab_size)")

    def mask_probability(self, timestep: int | torch.Tensor) -> torch.Tensor:
        timestep = self._as_timestep_tensor(timestep).to(dtype=torch.float32)
        denom = max(float(self.config.diffusion_steps), 1.0)
        return ((timestep + 1.0) / denom).clamp(0.0, 1.0)

    def timestep_grid(
        self,
        timestep: int | torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_input_ids(input_ids)
        timestep = self._normalize_timestep_for_input(timestep, input_ids)
        batch, seq_len = input_ids.shape
        if timestep.ndim == 0:
            return timestep.view(1, 1).expand(batch, seq_len)
        if timestep.ndim == 1:
            if timestep.size(0) == batch:
                return timestep[:, None].expand(batch, seq_len)
            return timestep.view(1, 1).expand(batch, seq_len)
        return timestep.expand(batch, seq_len)

    def loss_weight(
        self,
        timestep: int | torch.Tensor,
        *,
        scheme: str | None = None,
    ) -> torch.Tensor:
        scheme = self.config.diffusion_loss_weighting if scheme is None else scheme
        if scheme not in {"uniform", "mask_prob", "inverse_mask_prob"}:
            raise ValueError(
                "diffusion loss weighting scheme must be uniform, mask_prob, or inverse_mask_prob"
            )
        prob = self.mask_probability(timestep)
        if scheme == "uniform":
            return torch.ones_like(prob)
        if scheme == "mask_prob":
            return prob
        return prob.clamp_min(1e-6).reciprocal()

    def add_noise(
        self,
        input_ids: torch.Tensor,
        timestep: int | torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_input_ids(input_ids)
        timestep = self._normalize_timestep_for_input(timestep, input_ids)
        prob = self.mask_probability(timestep).to(device=input_ids.device)
        while prob.ndim < input_ids.ndim:
            prob = prob.unsqueeze(-1)
        mask = (
            torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
            < prob
        )
        special = input_ids.eq(self.config.pad_token_id)
        mask = mask & ~special
        flat_mask = mask.view(mask.size(0), -1)
        flat_special = special.view(special.size(0), -1)
        for row in range(flat_mask.size(0)):
            if not flat_mask[row].any():
                candidates = (~flat_special[row]).nonzero(as_tuple=False).flatten()
                if candidates.numel() > 0:
                    flat_mask[row, candidates[0]] = True
        noisy = input_ids.masked_fill(mask, self.config.mask_token_id)
        return noisy, mask
