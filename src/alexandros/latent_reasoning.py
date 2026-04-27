from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from alexandros.bitlinear import make_linear
from alexandros.configuration_alexandros import AlexandrosConfig
from alexandros.initialization import (
    initialize_embedding,
    initialize_linear,
    initialize_norm,
)


@dataclass
class LatentVAEOutput:
    latents: torch.Tensor
    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    kl_loss: torch.Tensor


class LatentThoughtVAE(nn.Module):
    """Compact VAE for continuous thought slots."""

    def __init__(self, config: AlexandrosConfig) -> None:
        super().__init__()
        self.config = config
        linear_kwargs = {
            "variant": config.variant,
            "activation_bits": config.bitnet_activation_bits,
        }
        self.to_mu = make_linear(config.hidden_size, config.latent_dim, **linear_kwargs)
        self.to_logvar = make_linear(
            config.hidden_size, config.latent_dim, **linear_kwargs
        )
        self.decoder = make_linear(
            config.latent_dim, config.hidden_size, **linear_kwargs
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_linear(self.to_mu)
        initialize_linear(self.to_logvar)
        initialize_linear(self.decoder)

    def _validate_hidden_states(self, hidden_states: torch.Tensor) -> None:
        if not torch.is_tensor(hidden_states) or not hidden_states.is_floating_point():
            raise ValueError("hidden_states must be a floating-point tensor")
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, hidden_size]"
            )
        if hidden_states.size(0) == 0:
            raise ValueError("hidden_states batch size must be > 0")
        if hidden_states.size(1) == 0:
            raise ValueError("hidden_states sequence length must be > 0")
        if hidden_states.size(-1) != self.config.hidden_size:
            raise ValueError("hidden_states last dimension must match hidden_size")
        if not torch.isfinite(hidden_states).all():
            raise ValueError("hidden_states must contain only finite values")

    def forward(self, hidden_states: torch.Tensor) -> LatentVAEOutput:
        self._validate_hidden_states(hidden_states)
        pooled = hidden_states.mean(dim=1)
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled).clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) if self.training else torch.zeros_like(std)
        z = mu + eps * std
        latents = z[:, None, :].expand(-1, self.config.latent_slots, -1).contiguous()
        reconstruction = self.decoder(latents)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return LatentVAEOutput(latents, reconstruction, mu, logvar, kl)


class LatentDiffusionReasoner(nn.Module):
    """Iterative latent refinement module with bidirectional slot attention."""

    def __init__(self, config: AlexandrosConfig) -> None:
        super().__init__()
        self.config = config
        self.num_latent_heads = self._select_latent_heads(
            config.latent_dim,
            config.num_attention_heads,
        )
        self.latent_head_dim = config.latent_dim // self.num_latent_heads
        self.attn_scale = 1.0 / math.sqrt(self.latent_head_dim)
        self.timestep_embed = nn.Embedding(
            max(config.diffusion_steps + 1, 2), config.latent_dim
        )
        self.last_stats: dict[str, int | float | bool] = {}
        self.attn_norm = nn.LayerNorm(config.latent_dim)
        self.ffn_norm = nn.LayerNorm(config.latent_dim)
        linear_kwargs = {
            "variant": config.variant,
            "activation_bits": config.bitnet_activation_bits,
        }
        self.q_proj = make_linear(
            config.latent_dim, config.latent_dim, bias=False, **linear_kwargs
        )
        self.k_proj = make_linear(
            config.latent_dim, config.latent_dim, bias=False, **linear_kwargs
        )
        self.v_proj = make_linear(
            config.latent_dim, config.latent_dim, bias=False, **linear_kwargs
        )
        self.attn_out_proj = make_linear(
            config.latent_dim,
            config.latent_dim,
            bias=False,
            **linear_kwargs,
        )
        self.in_proj = make_linear(
            config.latent_dim, config.latent_dim * 2, **linear_kwargs
        )
        self.out_proj = make_linear(
            config.latent_dim * 2, config.latent_dim, **linear_kwargs
        )
        self.to_hidden = make_linear(
            config.latent_dim, config.hidden_size, **linear_kwargs
        )
        self.reset_parameters()

    @staticmethod
    def _select_latent_heads(latent_dim: int, preferred_heads: int) -> int:
        heads = min(preferred_heads, latent_dim)
        while heads > 1 and latent_dim % heads != 0:
            heads -= 1
        return heads

    def reset_parameters(self) -> None:
        initialize_embedding(self.timestep_embed)
        initialize_norm(self.attn_norm)
        initialize_norm(self.ffn_norm)
        initialize_linear(self.q_proj)
        initialize_linear(self.k_proj)
        initialize_linear(self.v_proj)
        initialize_linear(self.attn_out_proj)
        initialize_linear(self.in_proj)
        initialize_linear(self.out_proj)
        initialize_linear(self.to_hidden)

    def _validate_latents(self, latents: torch.Tensor) -> None:
        if not torch.is_tensor(latents) or not latents.is_floating_point():
            raise ValueError("latents must be a floating-point tensor")
        if latents.ndim != 3:
            raise ValueError(
                "latents must have shape [batch, latent_slots, latent_dim]"
            )
        if latents.size(0) == 0:
            raise ValueError("latents batch size must be > 0")
        if latents.size(1) == 0:
            raise ValueError("latents must contain at least one slot")
        if latents.size(-1) != self.config.latent_dim:
            raise ValueError("latents last dimension must match latent_dim")
        if not torch.isfinite(latents).all():
            raise ValueError("latents must contain only finite values")

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, slots, _ = x.shape
        return x.view(
            batch,
            slots,
            self.num_latent_heads,
            self.latent_head_dim,
        ).transpose(1, 2)

    def _slot_attention(self, x: torch.Tensor) -> torch.Tensor:
        q = self._shape_heads(self.q_proj(x))
        k = self._shape_heads(self.k_proj(x))
        v = self._shape_heads(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.attn_scale
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous()
        out = out.view(x.size(0), x.size(1), self.config.latent_dim)
        return self.attn_out_proj(out)

    def forward(self, latents: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        self._validate_latents(latents)
        steps = self.config.diffusion_steps if steps is None else steps
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        h = latents
        steps_run = 0
        last_update_norm = 0.0
        halted = False
        for step in range(steps):
            step_id = min(step, self.timestep_embed.num_embeddings - 1)
            t = torch.full((h.size(0),), step_id, dtype=torch.long, device=h.device)
            t_emb = self.timestep_embed(t).unsqueeze(1)
            attn_update = self._slot_attention(self.attn_norm(h + t_emb))
            ffn_input = self.ffn_norm(h + t_emb + attn_update)
            ffn_update = self.out_proj(F.silu(self.in_proj(ffn_input)))
            update = attn_update + ffn_update
            update_norm = update.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            update = update * (self.config.latent_update_clip / update_norm).clamp(
                max=1.0
            )
            step_update = update / max(steps, 1)
            h = h + step_update
            steps_run = step + 1
            last_update_norm = float(
                step_update.detach().float().norm(dim=-1).max().cpu().item()
            )
            if (
                self.config.latent_adaptive_threshold > 0.0
                and last_update_norm <= self.config.latent_adaptive_threshold
            ):
                halted = True
                break
        self.last_stats = {
            "steps_requested": steps,
            "steps_run": steps_run,
            "halted": halted,
            "last_update_norm": last_update_norm,
            "adaptive_threshold": float(self.config.latent_adaptive_threshold),
        }
        return h

    def decode_to_hidden(self, latents: torch.Tensor) -> torch.Tensor:
        self._validate_latents(latents)
        return self.to_hidden(latents)
