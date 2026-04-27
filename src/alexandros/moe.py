from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from alexandros.bitlinear import make_linear
from alexandros.blocks import SwiGLUExpert
from alexandros.configuration_alexandros import AlexandrosConfig
from alexandros.initialization import initialize_linear

_TIMESTEP_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


@dataclass
class MoEStats:
    router_probs: torch.Tensor
    selected_experts: torch.Tensor
    routed_weights: torch.Tensor
    expert_load: torch.Tensor

    @property
    def expert_load_entropy(self) -> torch.Tensor:
        probs = self.expert_load.clamp_min(1e-12)
        return -(probs * probs.log()).sum()


class MoEFeedForward(nn.Module):
    """Fine-grained shared+routed MoE FFN with sigmoid top-k routing."""

    def __init__(self, config: AlexandrosConfig) -> None:
        super().__init__()
        self.config = config
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.router = make_linear(
            config.hidden_size,
            self.num_experts,
            bias=False,
            variant=config.variant,
            activation_bits=config.bitnet_activation_bits,
        )
        self.routed_experts = nn.ModuleList(
            [
                SwiGLUExpert(config, config.moe_expert_hidden_size)
                for _ in range(self.num_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                SwiGLUExpert(config, config.moe_expert_hidden_size * self.top_k)
                for _ in range(config.moe_num_shared_experts)
            ]
        )
        self.timestep_router_bias = nn.Embedding(
            max(config.diffusion_steps + 1, 2), self.num_experts
        )
        self.token_state_router_bias = nn.Embedding(2, self.num_experts)
        self.position_router_bias = nn.Embedding(
            config.moe_position_buckets, self.num_experts
        )
        self.register_buffer(
            "router_bias", torch.zeros(self.num_experts), persistent=True
        )
        self.register_buffer(
            "router_load_ema", torch.zeros(self.num_experts), persistent=False
        )
        self.register_buffer(
            "router_load_ema_steps",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "timestep_expert_load",
            torch.zeros(max(config.diffusion_steps + 1, 2), self.num_experts),
            persistent=False,
        )
        self.register_buffer(
            "timestep_expert_count",
            torch.zeros(max(config.diffusion_steps + 1, 2)),
            persistent=False,
        )
        self.last_stats: MoEStats | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_linear(self.router)
        for expert in self.routed_experts:
            expert.reset_parameters()
        for expert in self.shared_experts:
            expert.reset_parameters()
        with torch.no_grad():
            self.timestep_router_bias.weight.zero_()
            self.token_state_router_bias.weight.zero_()
            self.position_router_bias.weight.zero_()
            self.router_bias.zero_()
            self.router_load_ema.zero_()
            self.router_load_ema_steps.zero_()
            self.timestep_expert_load.zero_()
            self.timestep_expert_count.zero_()
        self.last_stats = None

    def _as_timestep_tensor(
        self,
        diffusion_timestep: int | torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(diffusion_timestep, bool):
            raise ValueError("diffusion_timestep must be an integer value")
        if not torch.is_tensor(diffusion_timestep):
            diffusion_timestep = torch.tensor(diffusion_timestep, device=device)
        else:
            diffusion_timestep = diffusion_timestep.to(device=device)
        if diffusion_timestep.dtype == torch.bool:
            raise ValueError("diffusion_timestep must be an integer value")
        if diffusion_timestep.numel() == 0:
            raise ValueError("diffusion_timestep cannot be empty")
        if torch.is_floating_point(diffusion_timestep):
            if not torch.isfinite(diffusion_timestep).all():
                raise ValueError("diffusion_timestep must be finite integer values")
            if not torch.equal(diffusion_timestep, diffusion_timestep.round()):
                raise ValueError("diffusion_timestep must be integer values")
        elif diffusion_timestep.dtype not in _TIMESTEP_DTYPES:
            raise ValueError("diffusion_timestep must be integer values")
        timestep = diffusion_timestep.to(dtype=torch.long)
        if (
            timestep.min().item() < 0
            or timestep.max().item() >= self.config.diffusion_steps
        ):
            raise ValueError("diffusion_timestep must be in [0, diffusion_steps)")
        return timestep

    def _normalize_timestep(
        self,
        diffusion_timestep: int | torch.Tensor,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        timestep = self._as_timestep_tensor(diffusion_timestep, device)
        if timestep.ndim == 0:
            return timestep.view(1, 1).expand(batch, seq_len)
        if timestep.ndim == 1:
            if timestep.size(0) == batch:
                return timestep[:, None].expand(batch, seq_len)
            if timestep.size(0) == 1:
                return timestep.view(1, 1).expand(batch, seq_len)
        if timestep.ndim == 2:
            if timestep.size(0) in {1, batch} and timestep.size(1) in {1, seq_len}:
                return timestep.expand(batch, seq_len)
        raise ValueError(
            "diffusion_timestep must be scalar, [batch], [1], [batch, sequence], "
            "or broadcastable [1, 1]/[batch, 1]"
        )

    def _normalize_token_state(
        self,
        diffusion_token_state: torch.Tensor,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(diffusion_token_state):
            raise ValueError("diffusion_token_state must be a tensor")
        token_state = diffusion_token_state.to(device=device)
        if token_state.dtype == torch.bool:
            token_state = token_state.to(dtype=torch.long)
        elif torch.is_floating_point(token_state):
            if not torch.isfinite(token_state).all():
                raise ValueError("diffusion_token_state must contain finite 0/1 values")
            if not torch.equal(token_state, token_state.round()):
                raise ValueError(
                    "diffusion_token_state must contain integer 0/1 values"
                )
            token_state = token_state.to(dtype=torch.long)
        elif token_state.dtype in _TIMESTEP_DTYPES:
            token_state = token_state.to(dtype=torch.long)
        else:
            raise ValueError("diffusion_token_state must contain integer 0/1 values")
        if token_state.numel() == 0:
            raise ValueError("diffusion_token_state cannot be empty")
        if token_state.min().item() < 0 or token_state.max().item() > 1:
            raise ValueError("diffusion_token_state must contain only 0/1 values")
        if token_state.ndim == 0:
            return token_state.view(1, 1).expand(batch, seq_len)
        if token_state.ndim == 1:
            if token_state.size(0) == batch:
                return token_state[:, None].expand(batch, seq_len)
            if token_state.size(0) == 1:
                return token_state.view(1, 1).expand(batch, seq_len)
        if token_state.ndim == 2:
            if token_state.size(0) in {1, batch} and token_state.size(1) in {
                1,
                seq_len,
            }:
                return token_state.expand(batch, seq_len)
        raise ValueError(
            "diffusion_token_state must be scalar, [batch], [1], [batch, sequence], "
            "or broadcastable [1, 1]/[batch, 1]"
        )

    def _normalize_position_ids(
        self,
        position_ids: torch.Tensor,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(position_ids):
            raise ValueError("position_ids must be a tensor")
        positions = position_ids.to(device=device)
        if positions.dtype == torch.bool:
            raise ValueError("position_ids must contain integer positions")
        if torch.is_floating_point(positions):
            if not torch.isfinite(positions).all():
                raise ValueError("position_ids must contain finite integer positions")
            if not torch.equal(positions, positions.round()):
                raise ValueError("position_ids must contain integer positions")
            positions = positions.to(dtype=torch.long)
        elif positions.dtype in _TIMESTEP_DTYPES:
            positions = positions.to(dtype=torch.long)
        else:
            raise ValueError("position_ids must contain integer positions")
        if positions.numel() == 0:
            raise ValueError("position_ids cannot be empty")
        if (
            positions.min().item() < 0
            or positions.max().item() >= self.config.max_position_embeddings
        ):
            raise ValueError("position_ids must be in [0, max_position_embeddings)")
        if positions.ndim == 0:
            return positions.view(1, 1).expand(batch, seq_len)
        if positions.ndim == 1:
            if positions.size(0) == seq_len:
                return positions.view(1, seq_len).expand(batch, seq_len)
            if positions.size(0) == batch:
                return positions[:, None].expand(batch, seq_len)
            if positions.size(0) == 1:
                return positions.view(1, 1).expand(batch, seq_len)
        if positions.ndim == 2:
            if positions.size(0) in {1, batch} and positions.size(1) in {1, seq_len}:
                return positions.expand(batch, seq_len)
        raise ValueError(
            "position_ids must be scalar, [sequence], [batch], [1], "
            "[batch, sequence], or broadcastable [1, 1]/[batch, 1]"
        )

    def _position_buckets(self, position_ids: torch.Tensor) -> torch.Tensor:
        buckets = torch.div(
            position_ids * self.config.moe_position_buckets,
            self.config.max_position_embeddings,
            rounding_mode="floor",
        )
        return buckets.clamp(max=self.config.moe_position_buckets - 1)

    def _route_dense(
        self,
        x: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        expert_outputs = torch.stack(
            [expert(x) for expert in self.routed_experts], dim=-2
        )
        gather_index = top_indices.unsqueeze(-1).expand(*top_indices.shape, x.size(-1))
        selected_outputs = torch.gather(expert_outputs, dim=-2, index=gather_index)
        return (selected_outputs * top_weights.unsqueeze(-1)).sum(dim=-2)

    def _route_sparse(
        self,
        x: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        hidden = x.size(-1)
        flat_x = x.reshape(-1, hidden)
        flat_indices = top_indices.reshape(-1, self.top_k)
        flat_weights = top_weights.reshape(-1, self.top_k)
        routed = torch.zeros_like(flat_x)
        for expert_idx, expert in enumerate(self.routed_experts):
            selected = flat_indices.eq(expert_idx)
            token_mask = selected.any(dim=-1)
            if not token_mask.any():
                continue
            weights = (flat_weights * selected.to(flat_weights.dtype)).sum(dim=-1)
            routed[token_mask] = routed[token_mask] + (
                expert(flat_x[token_mask]) * weights[token_mask].unsqueeze(-1)
            )
        return routed.view_as(x)

    @torch.no_grad()
    def _track_timestep_load(
        self,
        top_indices: torch.Tensor,
        timestep: torch.Tensor,
    ) -> None:
        flat_timestep = timestep.reshape(-1)
        flat_indices = top_indices.reshape(flat_timestep.numel(), self.top_k)
        for step in flat_timestep.unique():
            step_idx = int(step.item())
            mask = flat_timestep.eq(step_idx)
            counts = torch.bincount(
                flat_indices[mask].reshape(-1),
                minlength=self.num_experts,
            ).to(self.timestep_expert_load.dtype)
            self.timestep_expert_load[step_idx].add_(counts)
            self.timestep_expert_count[step_idx].add_(counts.sum())

    @torch.no_grad()
    def _track_router_load(self, expert_load: torch.Tensor) -> None:
        load = expert_load.to(
            device=self.router_load_ema.device, dtype=self.router_load_ema.dtype
        )
        if int(self.router_load_ema_steps.item()) == 0:
            self.router_load_ema.copy_(load)
        else:
            decay = self.config.router_load_ema_decay
            self.router_load_ema.mul_(decay).add_(load, alpha=1.0 - decay)
        self.router_load_ema_steps.add_(1)

    def forward(
        self,
        x: torch.Tensor,
        diffusion_timestep: int | torch.Tensor | None = None,
        diffusion_token_state: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        router_bias = self.router_bias.clamp(
            -self.config.router_bias_clip,
            self.config.router_bias_clip,
        )
        logits = self.router(x)
        timestep_for_stats = None
        if diffusion_timestep is not None:
            timestep_for_stats = self._normalize_timestep(
                diffusion_timestep,
                batch,
                seq_len,
                x.device,
            )
            logits = logits + self.timestep_router_bias(timestep_for_stats)
        if self.config.moe_token_state_routing and diffusion_token_state is not None:
            token_state = self._normalize_token_state(
                diffusion_token_state,
                batch,
                seq_len,
                x.device,
            )
            logits = logits + self.token_state_router_bias(token_state)
        if self.config.moe_position_routing and position_ids is not None:
            normalized_positions = self._normalize_position_ids(
                position_ids,
                batch,
                seq_len,
                x.device,
            )
            position_buckets = self._position_buckets(normalized_positions)
            logits = logits + self.position_router_bias(position_buckets)
        logits = logits.clamp(
            -self.config.router_logit_clip,
            self.config.router_logit_clip,
        )
        selection_logits = (logits + router_bias).clamp(
            -self.config.router_logit_clip,
            self.config.router_logit_clip,
        )

        probs = torch.sigmoid(logits)
        selection_probs = torch.sigmoid(selection_logits)
        _, top_indices = torch.topk(selection_probs, self.top_k, dim=-1)
        top_values = torch.gather(probs, dim=-1, index=top_indices)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        if self.config.moe_sparse_dispatch:
            routed = self._route_sparse(x, top_indices, top_weights)
        else:
            routed = self._route_dense(x, top_indices, top_weights)

        if self.shared_experts:
            shared = torch.stack(
                [expert(x) for expert in self.shared_experts], dim=0
            ).mean(dim=0)
            output = routed + shared
        else:
            output = routed

        load = torch.bincount(top_indices.reshape(-1), minlength=self.num_experts).to(
            device=x.device, dtype=torch.float32
        )
        load = load / load.sum().clamp_min(1.0)
        self.last_stats = MoEStats(
            router_probs=probs.detach(),
            selected_experts=top_indices.detach(),
            routed_weights=top_weights.detach(),
            expert_load=load.detach(),
        )
        self._track_router_load(load.detach())
        if timestep_for_stats is not None:
            self._track_timestep_load(top_indices.detach(), timestep_for_stats.detach())
        return output

    @torch.no_grad()
    def update_router_bias(
        self, expert_load: torch.Tensor | None = None, rate: float | None = None
    ) -> None:
        if expert_load is None:
            if int(self.router_load_ema_steps.item()) > 0:
                expert_load = self.router_load_ema
            elif self.last_stats is not None:
                expert_load = self.last_stats.expert_load
            else:
                return
        rate = self.config.router_bias_update_rate if rate is None else rate
        expert_load = expert_load.to(
            device=self.router_bias.device, dtype=self.router_bias.dtype
        )
        target = torch.full_like(expert_load, 1.0 / self.num_experts)
        self.router_bias.add_(rate * (target - expert_load))
        self.router_bias.clamp_(
            -self.config.router_bias_clip, self.config.router_bias_clip
        )
