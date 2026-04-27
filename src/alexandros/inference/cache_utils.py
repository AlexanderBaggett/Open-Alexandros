from __future__ import annotations

import torch

from alexandros.kv_cache import TurboQuantPacket

AttentionCache = dict[str, torch.Tensor | TurboQuantPacket]
_BEAM_INDEX_DTYPES = {torch.int32, torch.int64}


def _reorder_tensor(value: torch.Tensor, beam_idx: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(beam_idx):
        raise ValueError("beam_idx must be a tensor")
    if beam_idx.ndim != 1:
        raise ValueError("beam_idx must be a 1D tensor")
    if beam_idx.dtype not in _BEAM_INDEX_DTYPES:
        raise ValueError("beam_idx must be an integer tensor")
    if value.size(0) != beam_idx.numel():
        raise ValueError("cache batch dimension must match beam_idx length")
    index = beam_idx.to(device=value.device, dtype=torch.long)
    if index.numel() > 0 and (
        index.min().item() < 0 or index.max().item() >= value.size(0)
    ):
        raise ValueError("beam_idx contains indices outside the cache batch dimension")
    return value.index_select(0, index)


def _reorder_cache_value(
    value: torch.Tensor | TurboQuantPacket,
    beam_idx: torch.Tensor,
) -> torch.Tensor | TurboQuantPacket:
    if torch.is_tensor(value):
        return _reorder_tensor(value, beam_idx)
    if isinstance(value, TurboQuantPacket):
        return TurboQuantPacket(
            q=_reorder_tensor(value.q, beam_idx),
            scale=_reorder_tensor(value.scale, beam_idx),
            bits=value.bits,
            original_dtype=value.original_dtype,
            rotation_seed=value.rotation_seed,
            qjl_sign=None
            if value.qjl_sign is None
            else _reorder_tensor(value.qjl_sign, beam_idx),
            packet_format_version=value.packet_format_version,
            qjl_projection_seed=value.qjl_projection_seed,
            qjl_residual_norm=None
            if value.qjl_residual_norm is None
            else _reorder_tensor(value.qjl_residual_norm, beam_idx),
        )
    raise TypeError("unsupported attention cache value")


def reorder_past_key_values(
    past_key_values: list[AttentionCache | None] | None,
    beam_idx: torch.Tensor,
) -> list[AttentionCache | None] | None:
    if past_key_values is None:
        return None
    reordered = []
    for layer_cache in past_key_values:
        if layer_cache is None:
            reordered.append(None)
            continue
        reordered.append(
            {
                key: _reorder_cache_value(value, beam_idx)
                for key, value in layer_cache.items()
            }
        )
    return reordered


def reorder_past_ssm_states(
    past_ssm_states: list[torch.Tensor | None] | None,
    beam_idx: torch.Tensor,
) -> list[torch.Tensor | None] | None:
    if past_ssm_states is None:
        return None
    return [
        None if state is None else _reorder_tensor(state, beam_idx)
        for state in past_ssm_states
    ]


def reorder_generation_cache(
    *,
    past_key_values: list[AttentionCache | None] | None,
    past_ssm_states: list[torch.Tensor | None] | None,
    beam_idx: torch.Tensor,
) -> tuple[
    list[AttentionCache | None] | None,
    list[torch.Tensor | None] | None,
]:
    """Reorder Alexandros AR caches for future beam-search style decoders."""

    return (
        reorder_past_key_values(past_key_values, beam_idx),
        reorder_past_ssm_states(past_ssm_states, beam_idx),
    )
