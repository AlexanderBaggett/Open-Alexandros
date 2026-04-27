from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TurboQuantPacket:
    q: torch.Tensor
    scale: torch.Tensor
    bits: int
    original_dtype: torch.dtype
    rotation_seed: int
    qjl_sign: torch.Tensor | None = None
    packet_format_version: int = 1
    qjl_projection_seed: int | None = None
    qjl_residual_norm: torch.Tensor | None = None

    @property
    def estimated_bits(self) -> int:
        seed_bits = 32
        qjl_bits = 0
        if self.qjl_sign is not None:
            qjl_bits += self.qjl_sign.numel()
        if self.qjl_projection_seed is not None:
            qjl_bits += 32
        if self.qjl_residual_norm is not None:
            qjl_bits += self.qjl_residual_norm.numel() * 16
        return int(
            self.q.numel() * self.bits + self.scale.numel() * 16 + seed_bits + qjl_bits
        )


def _is_float_dtype(dtype: Any) -> bool:
    try:
        return torch.empty((), dtype=dtype).is_floating_point()
    except TypeError:
        return False


def validate_turboquant_packet(
    packet: TurboQuantPacket,
    *,
    expected_batch: int | None = None,
    expected_last_dim: int | None = None,
) -> None:
    if not isinstance(packet.packet_format_version, int) or isinstance(
        packet.packet_format_version, bool
    ):
        raise ValueError("TurboQuant packet format version must be an integer")
    if packet.packet_format_version != 1:
        raise ValueError("unsupported TurboQuant packet format version")
    if not isinstance(packet.bits, int) or isinstance(packet.bits, bool):
        raise ValueError("TurboQuant packet bits must be an integer")
    if packet.bits < 2 or packet.bits > 8:
        raise ValueError("TurboQuant packet bits must be between 2 and 8")
    if not isinstance(packet.rotation_seed, int) or isinstance(
        packet.rotation_seed, bool
    ):
        raise ValueError("TurboQuant packet rotation_seed must be an integer")
    if not _is_float_dtype(packet.original_dtype):
        raise ValueError(
            "TurboQuant packet original_dtype must be a floating-point torch dtype"
        )
    if not torch.is_tensor(packet.q) or packet.q.dtype != torch.int8:
        raise ValueError("TurboQuant packet q must be an int8 tensor")
    if packet.q.ndim == 0 or packet.q.size(-1) == 0:
        raise ValueError("TurboQuant packet q must have a non-empty last dimension")
    if expected_batch is not None and packet.q.size(0) != expected_batch:
        raise ValueError("TurboQuant packet q batch dimension does not match")
    if expected_last_dim is not None and packet.q.size(-1) != expected_last_dim:
        raise ValueError("TurboQuant packet q last dimension does not match")
    if not torch.is_tensor(packet.scale) or not packet.scale.is_floating_point():
        raise ValueError("TurboQuant packet scale must be a floating-point tensor")
    expected_scale_shape = packet.q.shape[:-1] + (1,)
    if tuple(packet.scale.shape) != tuple(expected_scale_shape):
        raise ValueError(
            "TurboQuant packet scale shape must match q with a final singleton dimension"
        )
    if not torch.isfinite(packet.scale).all() or packet.scale.le(0).any():
        raise ValueError("TurboQuant packet scale values must be finite and positive")
    qmax = 2 ** (packet.bits - 1) - 1
    if packet.q.numel() and packet.q.to(torch.int16).abs().max().item() > qmax:
        raise ValueError("TurboQuant packet q values exceed the packet bit range")
    if packet.qjl_sign is not None:
        if not torch.is_tensor(packet.qjl_sign) or packet.qjl_sign.dtype != torch.int8:
            raise ValueError("TurboQuant packet qjl_sign must be an int8 tensor")
        if tuple(packet.qjl_sign.shape) != tuple(packet.q.shape):
            raise ValueError("TurboQuant packet qjl_sign shape must match q")
        if packet.qjl_sign.numel() and packet.qjl_sign.abs().max().item() > 1:
            raise ValueError("TurboQuant packet qjl_sign values must be -1, 0, or 1")
        if (
            packet.qjl_projection_seed is None
            or not isinstance(packet.qjl_projection_seed, int)
            or isinstance(packet.qjl_projection_seed, bool)
        ):
            raise ValueError("TurboQuant packet qjl_projection_seed must be an integer")
        if (
            not torch.is_tensor(packet.qjl_residual_norm)
            or not packet.qjl_residual_norm.is_floating_point()
        ):
            raise ValueError(
                "TurboQuant packet qjl_residual_norm must be a floating-point tensor"
            )
        if tuple(packet.qjl_residual_norm.shape) != tuple(expected_scale_shape):
            raise ValueError(
                "TurboQuant packet qjl_residual_norm shape must match q with a final singleton dimension"
            )
        if (
            not torch.isfinite(packet.qjl_residual_norm).all()
            or packet.qjl_residual_norm.lt(0).any()
        ):
            raise ValueError(
                "TurboQuant packet qjl_residual_norm values must be finite and non-negative"
            )
    elif packet.qjl_projection_seed is not None or packet.qjl_residual_norm is not None:
        raise ValueError("TurboQuant packet QJL metadata requires qjl_sign")


class TurboQuantKVCache:
    """Reference TurboQuant-style online vector quantizer."""

    def __init__(self, bits: int = 4, use_qjl: bool = False, seed: int = 17) -> None:
        if not isinstance(bits, int) or isinstance(bits, bool):
            raise ValueError("bits must be an integer")
        if bits < 2 or bits > 8:
            raise ValueError("bits must be between 2 and 8 for the reference quantizer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        self.bits = bits
        self.use_qjl = use_qjl
        self.seed = seed
        self._rotations: dict[
            tuple[int, torch.device, torch.dtype, int], torch.Tensor
        ] = {}

    def _rotation(
        self,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: int | None = None,
    ) -> torch.Tensor:
        seed = self.seed if seed is None else seed
        key = (dim, device, dtype, seed)
        if key not in self._rotations:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + dim)
            matrix = torch.randn(
                dim, dim, generator=generator, device=device, dtype=dtype
            )
            q, _ = torch.linalg.qr(matrix)
            self._rotations[key] = q
        return self._rotations[key]

    def _qjl_projection(
        self,
        shape: torch.Size | tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
    ) -> torch.Tensor:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + int(torch.tensor(tuple(shape)).sum().item()))
        bits = torch.randint(
            0,
            2,
            tuple(shape),
            generator=generator,
            device=device,
            dtype=torch.int8,
        )
        return (bits.to(dtype) * 2.0) - 1.0

    def compress(self, x: torch.Tensor) -> TurboQuantPacket:
        if not torch.is_tensor(x) or not x.is_floating_point():
            raise ValueError(
                "TurboQuant compression input must be a floating-point tensor"
            )
        if x.ndim == 0 or x.size(-1) == 0:
            raise ValueError(
                "TurboQuant compression input must have a non-empty last dimension"
            )
        if not torch.isfinite(x).all():
            raise ValueError(
                "TurboQuant compression input must contain only finite values"
            )
        original_dtype = x.dtype
        work = x.float()
        rotation = self._rotation(work.size(-1), work.device, work.dtype)
        rotated = work @ rotation
        qmax = float(2 ** (self.bits - 1) - 1)
        scale = rotated.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / qmax
        q = torch.round(rotated / scale).clamp(-qmax, qmax).to(torch.int8)
        qjl_sign = None
        qjl_residual_norm = None
        qjl_projection_seed = None
        if self.use_qjl:
            residual = rotated - q.to(work.dtype) * scale
            projection = self._qjl_projection(
                residual.shape,
                residual.device,
                residual.dtype,
                self.seed,
            )
            qjl_sign = (residual * projection).sign().to(torch.int8)
            qjl_residual_norm = residual.norm(dim=-1, keepdim=True).to(scale.dtype)
            qjl_projection_seed = self.seed
        return TurboQuantPacket(
            q=q,
            scale=scale,
            bits=self.bits,
            original_dtype=original_dtype,
            rotation_seed=self.seed,
            qjl_sign=qjl_sign,
            packet_format_version=1,
            qjl_projection_seed=qjl_projection_seed,
            qjl_residual_norm=qjl_residual_norm,
        )

    def decompress(self, packet: TurboQuantPacket) -> torch.Tensor:
        validate_turboquant_packet(packet)
        rotated = packet.q.to(packet.scale.dtype) * packet.scale
        rotation = self._rotation(
            rotated.size(-1),
            rotated.device,
            rotated.dtype,
            packet.rotation_seed,
        )
        x = rotated @ rotation.transpose(-1, -2)
        return x.to(packet.original_dtype)

    def _qjl_residual_estimate(self, packet: TurboQuantPacket) -> torch.Tensor:
        validate_turboquant_packet(packet)
        if (
            packet.qjl_sign is None
            or packet.qjl_projection_seed is None
            or packet.qjl_residual_norm is None
        ):
            raise ValueError("QJL residual estimate requires QJL packet metadata")
        dtype = packet.scale.dtype
        projection = self._qjl_projection(
            packet.qjl_sign.shape,
            packet.qjl_sign.device,
            dtype,
            packet.qjl_projection_seed,
        )
        signed_direction = packet.qjl_sign.to(dtype) * projection
        nonzero = signed_direction.ne(0).sum(dim=-1, keepdim=True).clamp_min(1)
        return signed_direction * packet.qjl_residual_norm / nonzero.to(dtype).sqrt()

    def _rotated_values(
        self,
        packet: TurboQuantPacket,
        *,
        use_qjl: bool = False,
    ) -> torch.Tensor:
        validate_turboquant_packet(packet)
        rotated = packet.q.to(packet.scale.dtype) * packet.scale
        if use_qjl:
            rotated = rotated + self._qjl_residual_estimate(packet)
        return rotated

    def estimate_attention_scores(
        self,
        query: torch.Tensor,
        packet: TurboQuantPacket,
        *,
        use_qjl: bool | None = None,
    ) -> torch.Tensor:
        """Estimate `query @ key.T` from a compressed key packet.

        `query` must have shape `[batch, query_length, dim]`, while `packet`
        stores key vectors with shape `[batch, key_length, dim]`. The estimator
        rotates query vectors with the packet rotation seed, multiplies against
        scalar-dequantized key vectors, and optionally adds the QJL residual
        sketch before computing the dot products.
        """

        validate_turboquant_packet(packet)
        use_qjl = self.use_qjl if use_qjl is None else use_qjl
        if not torch.is_tensor(query) or not query.is_floating_point():
            raise ValueError("query must be a floating-point tensor")
        if query.ndim != 3:
            raise ValueError("query must have shape [batch, query_length, dim]")
        if query.size(0) != packet.q.size(0):
            raise ValueError("query batch dimension must match packet")
        if query.size(-1) != packet.q.size(-1):
            raise ValueError("query last dimension must match packet")
        if not torch.isfinite(query).all():
            raise ValueError("query must contain only finite values")
        if use_qjl and packet.qjl_sign is None:
            raise ValueError("use_qjl=True requires QJL packet metadata")
        query_work = query.float()
        rotation = self._rotation(
            query_work.size(-1),
            query_work.device,
            query_work.dtype,
            packet.rotation_seed,
        )
        query_rotated = query_work @ rotation
        key_rotated = self._rotated_values(packet, use_qjl=use_qjl).to(query_work.dtype)
        return torch.matmul(query_rotated, key_rotated.transpose(-1, -2))
