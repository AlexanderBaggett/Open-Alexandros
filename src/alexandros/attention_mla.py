from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from alexandros.bitlinear import make_linear
from alexandros.configuration_alexandros import AlexandrosConfig
from alexandros.initialization import initialize_linear, residual_projection_std
from alexandros.kv_cache import TurboQuantKVCache, TurboQuantPacket


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    dim = x.size(-1)
    if dim < 2:
        return x
    device = x.device
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=x.dtype) / dim)
    )
    angles = positions.to(dtype=x.dtype).unsqueeze(-1) * freqs
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1)
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1)
    cos = cos.view(1, 1, cos.size(0), cos.size(1))
    sin = sin.view(1, 1, sin.size(0), sin.size(1))
    return (x * cos) + (_rotate_half(x) * sin)


class MLAAttention(nn.Module):
    """Reference Multi-Head Latent Attention with compressed KV cache."""

    def __init__(self, config: AlexandrosConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.nope_dim = config.mla_d_nope
        self.rope_dim = config.mla_d_r
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = config.dropout
        self.q_proj = make_linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            variant=config.variant,
            activation_bits=config.bitnet_activation_bits,
        )
        self.kv_down_proj = make_linear(
            config.hidden_size,
            config.kv_lora_rank,
            bias=False,
            variant=config.variant,
            activation_bits=config.bitnet_activation_bits,
        )
        self.k_up_proj = make_linear(
            config.kv_lora_rank,
            config.num_attention_heads * config.mla_d_nope,
            bias=False,
            variant=config.variant,
            activation_bits=config.bitnet_activation_bits,
        )
        self.k_rope_proj = (
            make_linear(
                config.hidden_size,
                config.mla_d_r,
                bias=False,
                variant=config.variant,
                activation_bits=config.bitnet_activation_bits,
            )
            if config.mla_d_r > 0
            else None
        )
        self.v_up_proj = make_linear(
            config.kv_lora_rank,
            config.hidden_size,
            bias=False,
            variant=config.variant,
            activation_bits=config.bitnet_activation_bits,
        )
        self.o_proj = make_linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            variant=config.variant,
            activation_bits=config.bitnet_activation_bits,
        )
        self.cache_compressor = (
            TurboQuantKVCache(bits=config.turboquant_bits, use_qjl=config.use_qjl)
            if config.use_turboquant_cache
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_linear(self.q_proj)
        initialize_linear(self.kv_down_proj)
        initialize_linear(self.k_up_proj)
        if self.k_rope_proj is not None:
            initialize_linear(self.k_rope_proj)
        initialize_linear(self.v_up_proj)
        initialize_linear(self.o_proj, std=residual_projection_std(self.config))

    def _shape(self, x: torch.Tensor, head_dim: int | None = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        head_dim = self.head_dim if head_dim is None else head_dim
        return x.view(batch, seq, self.num_heads, head_dim).transpose(1, 2)

    def _validate_hidden_states(self, x: torch.Tensor) -> None:
        if not torch.is_tensor(x) or not x.is_floating_point():
            raise ValueError("MLA hidden states must be a floating-point tensor")
        if x.ndim != 3:
            raise ValueError(
                "MLA hidden states must have shape [batch, sequence, hidden_size]"
            )
        if x.size(0) == 0:
            raise ValueError("MLA hidden states batch size must be > 0")
        if x.size(1) == 0:
            raise ValueError("MLA hidden states sequence length must be > 0")
        if x.size(-1) != self.hidden_size:
            raise ValueError("MLA hidden states last dimension must match hidden_size")
        if not torch.isfinite(x).all():
            raise ValueError("MLA hidden states must contain only finite values")

    def _validate_attention_mask(
        self,
        attention_mask: torch.Tensor,
        *,
        batch: int,
        seq_len: int,
        kv_len: int,
    ) -> None:
        if not torch.is_tensor(attention_mask):
            raise ValueError("attention_mask must be a tensor")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if attention_mask.size(0) != batch:
            raise ValueError("attention_mask batch size must match hidden states")
        if attention_mask.size(1) not in {seq_len, kv_len}:
            raise ValueError(
                "attention_mask length must match current sequence length "
                "or full key/value length when using cache"
            )
        if (
            torch.is_floating_point(attention_mask)
            and not torch.isfinite(attention_mask).all()
        ):
            raise ValueError("attention_mask must contain only finite 0/1 values")
        valid_values = attention_mask.eq(0) | attention_mask.eq(1)
        if not valid_values.all():
            raise ValueError("attention_mask must contain only 0/1 or bool values")
        if not attention_mask.to(dtype=torch.bool).any(dim=1).all():
            raise ValueError(
                "attention_mask must contain at least one non-pad token per row"
            )

    def _validate_cached_c_kv(
        self,
        c_kv: torch.Tensor,
        *,
        batch: int,
        device: torch.device,
    ) -> None:
        if c_kv.ndim != 3:
            raise ValueError(
                "MLA c_kv cache must have shape [batch, sequence, kv_lora_rank]"
            )
        if c_kv.size(0) != batch:
            raise ValueError("MLA c_kv cache batch size must match hidden states")
        if c_kv.size(1) == 0:
            raise ValueError("MLA c_kv cache sequence length must be > 0")
        if c_kv.size(-1) != self.config.kv_lora_rank:
            raise ValueError("MLA c_kv cache last dimension must match kv_lora_rank")
        if c_kv.device != device:
            raise ValueError("MLA c_kv cache device must match hidden states")
        if not c_kv.is_floating_point():
            raise ValueError("MLA c_kv cache must be a floating-point tensor")
        if not torch.isfinite(c_kv).all():
            raise ValueError("MLA c_kv cache must contain only finite values")

    def _validate_cached_k_rope(
        self,
        k_rope: torch.Tensor,
        *,
        batch: int,
        sequence_length: int,
        device: torch.device,
    ) -> None:
        if k_rope.ndim != 3:
            raise ValueError(
                "MLA k_rope cache must have shape [batch, sequence, mla_rope_dim]"
            )
        if k_rope.size(0) != batch:
            raise ValueError("MLA k_rope cache batch size must match hidden states")
        if k_rope.size(1) != sequence_length:
            raise ValueError("MLA k_rope cache sequence length must match c_kv")
        if k_rope.size(-1) != self.config.mla_d_r:
            raise ValueError("MLA k_rope cache last dimension must match mla_rope_dim")
        if k_rope.device != device:
            raise ValueError("MLA k_rope cache device must match hidden states")
        if not k_rope.is_floating_point():
            raise ValueError("MLA k_rope cache must be a floating-point tensor")
        if not torch.isfinite(k_rope).all():
            raise ValueError("MLA k_rope cache must contain only finite values")

    def _read_cached_c_kv(
        self,
        past_key_value: dict[str, torch.Tensor | TurboQuantPacket] | None,
    ) -> torch.Tensor | None:
        if past_key_value is None:
            return None
        if "c_kv" in past_key_value:
            value = past_key_value["c_kv"]
            if not torch.is_tensor(value):
                raise TypeError("c_kv cache entry must be a tensor")
            return value
        if "c_kv_packet" in past_key_value:
            if self.cache_compressor is None:
                raise ValueError(
                    "received TurboQuant cache but use_turboquant_cache is disabled"
                )
            packet = past_key_value["c_kv_packet"]
            if not isinstance(packet, TurboQuantPacket):
                raise TypeError("c_kv_packet cache entry must be a TurboQuantPacket")
            return self.cache_compressor.decompress(packet)
        raise ValueError("MLA cache entries must contain c_kv or c_kv_packet")

    def _read_cached_k_rope(
        self,
        past_key_value: dict[str, torch.Tensor | TurboQuantPacket] | None,
    ) -> torch.Tensor | None:
        if past_key_value is None or self.config.mla_d_r == 0:
            return None
        value = past_key_value.get("k_rope")
        if value is None:
            raise ValueError(
                "MLA cache must include k_rope when mla_rope_dim is enabled"
            )
        if not torch.is_tensor(value):
            raise TypeError("k_rope cache entry must be a tensor")
        return value

    def _write_cache(
        self,
        c_kv: torch.Tensor,
        k_rope: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | TurboQuantPacket]:
        c_kv = c_kv.detach()
        cache: dict[str, torch.Tensor | TurboQuantPacket]
        if self.cache_compressor is None:
            cache = {"c_kv": c_kv}
        else:
            cache = {"c_kv_packet": self.cache_compressor.compress(c_kv)}
        if self.config.mla_d_r > 0:
            if k_rope is None:
                raise ValueError("k_rope is required when mla_rope_dim is enabled")
            cache["k_rope"] = k_rope.detach()
        return cache

    def _attention_allowed_mask(
        self,
        attention_mask: torch.Tensor | None,
        *,
        batch: int,
        seq_len: int,
        kv_len: int,
        past_len: int,
        is_causal: bool,
        device: torch.device,
    ) -> torch.Tensor | None:
        allowed = None
        if is_causal:
            allowed = torch.ones(seq_len, kv_len, device=device, dtype=torch.bool).tril(
                diagonal=past_len
            )
            allowed = allowed.view(1, 1, seq_len, kv_len)
        if attention_mask is None:
            return allowed
        self._validate_attention_mask(
            attention_mask,
            batch=batch,
            seq_len=seq_len,
            kv_len=kv_len,
        )
        if attention_mask.size(-1) == seq_len and past_len > 0:
            prefix = torch.ones(
                attention_mask.size(0),
                past_len,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([prefix, attention_mask], dim=-1)
        key_allowed = attention_mask[:, None, None, :].to(
            device=device, dtype=torch.bool
        )
        if allowed is None:
            return key_allowed
        return allowed & key_allowed

    def _eager_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        allowed: torch.Tensor | None,
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if allowed is not None:
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=self.dropout, training=self.training)
        return torch.matmul(probs, v)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        allowed: torch.Tensor | None,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        allowed: torch.Tensor | None,
    ) -> torch.Tensor:
        if q.device.type != "cuda":
            raise RuntimeError(
                "attention_backend='flash' requires CUDA; use 'eager' or 'sdpa' "
                "for CPU validation"
            )
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except ImportError as exc:
            raise RuntimeError(
                "attention_backend='flash' requires a PyTorch build with "
                "torch.nn.attention.sdpa_kernel"
            ) from exc
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=allowed,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                )
        except RuntimeError as exc:
            raise RuntimeError(
                "Flash attention backend failed for this device, dtype, mask, or "
                "head dimension; use attention_backend='sdpa' or 'eager' as the "
                "reference fallback"
            ) from exc

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: dict[str, torch.Tensor | TurboQuantPacket] | None = None,
        use_cache: bool = False,
        is_causal: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | TurboQuantPacket] | None]:
        if not is_causal and (past_key_value is not None or use_cache):
            raise ValueError("non-causal MLA attention does not support cache reuse")
        self._validate_hidden_states(x)
        batch, seq_len, _ = x.shape
        c_kv_new = self.kv_down_proj(x)
        k_rope_new = self.k_rope_proj(x) if self.k_rope_proj is not None else None
        cached_c_kv = self._read_cached_c_kv(past_key_value)
        cached_k_rope = self._read_cached_k_rope(past_key_value)
        if cached_c_kv is not None:
            self._validate_cached_c_kv(cached_c_kv, batch=batch, device=x.device)
            cached_c_kv = cached_c_kv.to(device=x.device, dtype=c_kv_new.dtype)
            c_kv = torch.cat([cached_c_kv, c_kv_new], dim=1)
            past_len = cached_c_kv.size(1)
            if self.config.mla_d_r > 0:
                assert k_rope_new is not None
                assert cached_k_rope is not None
                self._validate_cached_k_rope(
                    cached_k_rope,
                    batch=batch,
                    sequence_length=past_len,
                    device=x.device,
                )
                cached_k_rope = cached_k_rope.to(
                    device=x.device,
                    dtype=k_rope_new.dtype,
                )
                k_rope = torch.cat([cached_k_rope, k_rope_new], dim=1)
            else:
                k_rope = None
        else:
            c_kv = c_kv_new
            k_rope = k_rope_new
            past_len = 0

        q = self._shape(self.q_proj(x))
        if self.config.mla_d_r > 0:
            q_nope, q_rope = torch.split(
                q,
                [self.nope_dim, self.rope_dim],
                dim=-1,
            )
            k_nope = self._shape(self.k_up_proj(c_kv), self.nope_dim)
        else:
            q_nope = None
            q_rope = None
            k_nope = None
            k = self._shape(self.k_up_proj(c_kv))
        v = self._shape(self.v_up_proj(c_kv))

        q_pos = torch.arange(past_len, past_len + seq_len, device=x.device)
        k_pos = torch.arange(0, c_kv.size(1), device=x.device)
        if self.config.mla_d_r > 0:
            assert q_nope is not None
            assert q_rope is not None
            assert k_nope is not None
            assert k_rope is not None
            q_rope = apply_rope(q_rope, q_pos, self.config.rope_theta)
            k_rope_heads = apply_rope(
                k_rope.unsqueeze(1),
                k_pos,
                self.config.rope_theta,
            ).expand(-1, self.num_heads, -1, -1)
            q = torch.cat([q_nope, q_rope], dim=-1)
            k = torch.cat([k_nope, k_rope_heads], dim=-1)
        else:
            q = apply_rope(q, q_pos, self.config.rope_theta)
            k = apply_rope(k, k_pos, self.config.rope_theta)

        kv_len = c_kv.size(1)
        allowed = self._attention_allowed_mask(
            attention_mask,
            batch=batch,
            seq_len=seq_len,
            kv_len=kv_len,
            past_len=past_len,
            is_causal=is_causal,
            device=x.device,
        )
        if self.config.attention_backend == "sdpa":
            out = self._sdpa_attention(q, k, v, allowed)
        elif self.config.attention_backend == "flash":
            out = self._flash_attention(q, k, v, allowed)
        else:
            out = self._eager_attention(q, k, v, allowed)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch, seq_len, self.hidden_size)
        cache = self._write_cache(c_kv, k_rope) if use_cache else None
        return self.o_proj(out), cache
