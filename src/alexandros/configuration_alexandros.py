from __future__ import annotations

import ast
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


def _validate_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _validate_finite_number(name: str, value: Any) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass
class AlexandrosConfig:
    """Serializable configuration for the Alexandros model family.

    The class intentionally mirrors the subset of Hugging Face config behavior
    needed by this prototype without requiring transformers as a dependency.
    """

    model_type: str = "alexandros"
    variant: str = "heavy"
    vocab_size: int = 32000
    hidden_size: int = 512
    intermediate_size: int = 1408
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 4096
    linear_attention_ratio: int = 3
    linear_mixer_backend: str = "gated_deltanet"
    deltanet_chunk_size: int = 0
    deltanet_state_clip: float = 5.0
    attention_layer_indices: list[int] | None = None
    moe_num_experts: int = 8
    moe_num_shared_experts: int = 1
    moe_top_k: int = 2
    moe_sparse_dispatch: bool = True
    moe_token_state_routing: bool = False
    moe_position_routing: bool = False
    moe_position_buckets: int = 32
    moe_expert_hidden_size: int = 256
    kv_lora_rank: int = 128
    mla_rope_dim: int = 0
    latent_dim: int = 256
    latent_slots: int = 8
    latent_update_clip: float = 5.0
    latent_adaptive_threshold: float = 0.0
    diffusion_steps: int = 8
    diffusion_attention_mask_mode: str = "causal"
    diffusion_objective: str = "masked"
    diffusion_loss_weighting: str = "uniform"
    diffusion_rb_chunk_size: int = 0
    mask_token_id: int = 3
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    attention_backend: str = "eager"
    router_bias_update_rate: float = 1e-3
    router_bias_update_interval: int = 1
    router_load_ema_decay: float = 0.9
    router_logit_clip: float = 30.0
    router_bias_clip: float = 5.0
    bitnet_activation_bits: int = 8
    enable_adaptive_depth: bool = False
    prelude_layers: int = 0
    recurrent_layers: int = 0
    coda_layers: int = 0
    recurrent_depth: int = 1
    use_loop_index_embeddings: bool = False
    max_depth_iters: int = 2
    act_threshold: float = 0.99
    act_ponder_cost: float = 0.0
    depth_lora_rank: int = 8
    depth_lora_ranks: list[int] | None = None
    ttt_rank: int = 8
    turboquant_bits: int = 4
    use_qjl: bool = False
    use_turboquant_cache: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.variant not in {"heavy", "lite"}:
            raise ValueError("variant must be 'heavy' or 'lite'")
        if self.diffusion_attention_mask_mode not in {"causal", "bidirectional"}:
            raise ValueError(
                "diffusion_attention_mask_mode must be 'causal' or 'bidirectional'"
            )
        if self.diffusion_objective not in {"masked", "rao_blackwellized"}:
            raise ValueError(
                "diffusion_objective must be 'masked' or 'rao_blackwellized'"
            )
        if self.diffusion_loss_weighting not in {
            "uniform",
            "mask_prob",
            "inverse_mask_prob",
        }:
            raise ValueError(
                "diffusion_loss_weighting must be 'uniform', 'mask_prob', or 'inverse_mask_prob'"
            )
        if self.attention_backend not in {"eager", "sdpa", "flash"}:
            raise ValueError("attention_backend must be 'eager', 'sdpa', or 'flash'")
        if self.linear_mixer_backend not in {
            "gated_deltanet",
            "matrix_deltanet",
            "mamba2",
        }:
            raise ValueError(
                "linear_mixer_backend must be 'gated_deltanet', "
                "'matrix_deltanet', or 'mamba2'"
            )
        if not isinstance(self.moe_token_state_routing, bool):
            raise ValueError("moe_token_state_routing must be a boolean")
        if not isinstance(self.moe_position_routing, bool):
            raise ValueError("moe_position_routing must be a boolean")
        if not isinstance(self.use_loop_index_embeddings, bool):
            raise ValueError("use_loop_index_embeddings must be a boolean")
        positive_ints = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "linear_attention_ratio": self.linear_attention_ratio,
            "moe_num_experts": self.moe_num_experts,
            "moe_top_k": self.moe_top_k,
            "moe_position_buckets": self.moe_position_buckets,
            "moe_expert_hidden_size": self.moe_expert_hidden_size,
            "kv_lora_rank": self.kv_lora_rank,
            "latent_dim": self.latent_dim,
            "latent_slots": self.latent_slots,
            "diffusion_steps": self.diffusion_steps,
            "recurrent_depth": self.recurrent_depth,
            "max_depth_iters": self.max_depth_iters,
            "depth_lora_rank": self.depth_lora_rank,
            "ttt_rank": self.ttt_rank,
            "bitnet_activation_bits": self.bitnet_activation_bits,
            "router_bias_update_interval": self.router_bias_update_interval,
        }
        for name, value in positive_ints.items():
            _validate_int(name, value)
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.bitnet_activation_bits < 2:
            raise ValueError("bitnet_activation_bits must be >= 2")
        nonnegative_ints = {
            "prelude_layers": self.prelude_layers,
            "recurrent_layers": self.recurrent_layers,
            "coda_layers": self.coda_layers,
            "deltanet_chunk_size": self.deltanet_chunk_size,
            "diffusion_rb_chunk_size": self.diffusion_rb_chunk_size,
        }
        for name, value in nonnegative_ints.items():
            _validate_int(name, value)
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        staged_count = self.prelude_layers + self.recurrent_layers + self.coda_layers
        if staged_count != 0:
            if staged_count != self.num_hidden_layers:
                raise ValueError(
                    "prelude_layers + recurrent_layers + coda_layers must equal "
                    "num_hidden_layers when using staged recurrent stack"
                )
            if self.recurrent_layers <= 0:
                raise ValueError(
                    "recurrent_layers must be > 0 when using staged recurrent stack"
                )
        rope_theta = _validate_finite_number("rope_theta", self.rope_theta)
        if rope_theta <= 0:
            raise ValueError("rope_theta must be > 0")
        rms_norm_eps = _validate_finite_number("rms_norm_eps", self.rms_norm_eps)
        if rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be > 0")
        dropout = _validate_finite_number("dropout", self.dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        router_bias_update_rate = _validate_finite_number(
            "router_bias_update_rate",
            self.router_bias_update_rate,
        )
        if router_bias_update_rate < 0:
            raise ValueError("router_bias_update_rate must be >= 0")
        router_logit_clip = _validate_finite_number(
            "router_logit_clip", self.router_logit_clip
        )
        if router_logit_clip <= 0:
            raise ValueError("router_logit_clip must be > 0")
        router_bias_clip = _validate_finite_number(
            "router_bias_clip", self.router_bias_clip
        )
        if router_bias_clip <= 0:
            raise ValueError("router_bias_clip must be > 0")
        router_load_ema_decay = _validate_finite_number(
            "router_load_ema_decay",
            self.router_load_ema_decay,
        )
        if not 0.0 <= router_load_ema_decay < 1.0:
            raise ValueError("router_load_ema_decay must satisfy 0 <= decay < 1")
        latent_update_clip = _validate_finite_number(
            "latent_update_clip",
            self.latent_update_clip,
        )
        if latent_update_clip <= 0:
            raise ValueError("latent_update_clip must be > 0")
        latent_adaptive_threshold = _validate_finite_number(
            "latent_adaptive_threshold",
            self.latent_adaptive_threshold,
        )
        if latent_adaptive_threshold < 0:
            raise ValueError("latent_adaptive_threshold must be >= 0")
        deltanet_state_clip = _validate_finite_number(
            "deltanet_state_clip",
            self.deltanet_state_clip,
        )
        if deltanet_state_clip <= 0:
            raise ValueError("deltanet_state_clip must be > 0")
        act_threshold = _validate_finite_number("act_threshold", self.act_threshold)
        if not 0.0 < act_threshold <= 1.0:
            raise ValueError("act_threshold must satisfy 0 < act_threshold <= 1")
        act_ponder_cost = _validate_finite_number(
            "act_ponder_cost", self.act_ponder_cost
        )
        if act_ponder_cost < 0:
            raise ValueError("act_ponder_cost must be >= 0")
        _validate_int("moe_num_shared_experts", self.moe_num_shared_experts)
        if self.moe_num_shared_experts < 0:
            raise ValueError("moe_num_shared_experts must be >= 0")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("attention head_dim must be even for RoPE")
        _validate_int("mla_rope_dim", self.mla_rope_dim)
        if self.mla_rope_dim < 0:
            raise ValueError("mla_rope_dim must be >= 0")
        if self.mla_rope_dim >= self.head_dim:
            raise ValueError("mla_rope_dim must be smaller than head_dim")
        if self.mla_rope_dim % 2 != 0:
            raise ValueError("mla_rope_dim must be even")
        if self.mla_d_r < 0 or self.mla_d_nope < 0:
            raise ValueError("MLA RoPE and no-RoPE dimensions must be non-negative")
        if self.mla_d_nope + self.mla_d_r != self.head_dim:
            raise ValueError("MLA no-RoPE and RoPE dimensions must sum to head_dim")
        if self.mla_value_head_dim != self.head_dim:
            raise ValueError(
                "MLA value-head dimension must match head_dim in the reference implementation"
            )
        if self.mla_elements_per_token <= 0:
            raise ValueError("MLA cache elements per token must be > 0")
        if self.kv_lora_rank > self.hidden_size:
            raise ValueError(
                "kv_lora_rank cannot exceed hidden_size in the reference MLA"
            )
        if self.depth_lora_rank > self.hidden_size:
            raise ValueError("depth_lora_rank cannot exceed hidden_size")
        if self.depth_lora_ranks is not None:
            if not isinstance(self.depth_lora_ranks, list):
                raise ValueError("depth_lora_ranks must be a list of integers")
            if len(self.depth_lora_ranks) != self.max_depth_iters:
                raise ValueError("depth_lora_ranks length must match max_depth_iters")
            for idx, rank in enumerate(self.depth_lora_ranks):
                _validate_int(f"depth_lora_ranks[{idx}]", rank)
                if rank <= 0:
                    raise ValueError("depth_lora_ranks values must be > 0")
                if rank > self.hidden_size:
                    raise ValueError(
                        "depth_lora_ranks values cannot exceed hidden_size"
                    )
        if self.ttt_rank > self.hidden_size:
            raise ValueError("ttt_rank cannot exceed hidden_size")
        if self.moe_top_k > self.moe_num_experts:
            raise ValueError("moe_top_k cannot exceed moe_num_experts")
        if self.attention_layer_indices is not None:
            if not self.attention_layer_indices:
                raise ValueError(
                    "attention_layer_indices cannot be empty when provided"
                )
            for idx in self.attention_layer_indices:
                _validate_int("attention_layer_indices", idx)
            if any(
                idx < 0 or idx >= self.num_hidden_layers
                for idx in self.attention_layer_indices
            ):
                raise ValueError(
                    "attention_layer_indices must be within [0, num_hidden_layers)"
                )
            if len(set(self.attention_layer_indices)) != len(
                self.attention_layer_indices
            ):
                raise ValueError("attention_layer_indices cannot contain duplicates")
        if self.turboquant_bits < 2 or self.turboquant_bits > 8:
            raise ValueError("turboquant_bits must be between 2 and 8")
        token_ids = {
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "mask_token_id": self.mask_token_id,
        }
        for name, token_id in token_ids.items():
            _validate_int(name, token_id)
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"{name} must be in [0, vocab_size)")
        if len(set(token_ids.values())) != len(token_ids):
            raise ValueError("pad/bos/eos/mask token IDs must be distinct")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def mla_d_c(self) -> int:
        """Compressed KV latent dimension (`d_c`) for MLA cache accounting."""

        return self.kv_lora_rank

    @property
    def mla_d_r(self) -> int:
        """Separate RoPE key dimension (`d_R`) for MLA cache accounting."""

        return self.mla_rope_dim

    @property
    def mla_d_nope(self) -> int:
        """No-RoPE query/key dimension per head for reference attention."""

        return self.head_dim - self.mla_d_r

    @property
    def mla_value_head_dim(self) -> int:
        return self.head_dim

    @property
    def standard_mha_elements_per_token(self) -> int:
        return 2 * self.num_attention_heads * self.head_dim

    @property
    def mla_elements_per_token(self) -> int:
        return self.mla_d_c + self.mla_d_r

    def is_attention_layer(self, layer_idx: int) -> bool:
        if layer_idx < 0 or layer_idx >= self.num_hidden_layers:
            raise ValueError("layer_idx must be within [0, num_hidden_layers)")
        if self.attention_layer_indices is not None:
            return layer_idx in set(self.attention_layer_indices)
        cycle = self.linear_attention_ratio + 1
        return ((layer_idx + 1) % cycle == 0) or (
            layer_idx == self.num_hidden_layers - 1
        )

    def attention_layers(self) -> tuple[int, ...]:
        return tuple(
            idx for idx in range(self.num_hidden_layers) if self.is_attention_layer(idx)
        )

    @property
    def uses_recurrent_stack(self) -> bool:
        return (self.prelude_layers + self.recurrent_layers + self.coda_layers) > 0

    @property
    def recurrent_layer_range(self) -> tuple[int, int]:
        if not self.uses_recurrent_stack:
            return (0, 0)
        start = self.prelude_layers
        return (start, start + self.recurrent_layers)

    def depth_lora_rank_for_loop(self, loop_idx: int) -> int:
        _validate_int("loop_idx", loop_idx)
        if loop_idx < 0 or loop_idx >= self.max_depth_iters:
            raise ValueError("loop_idx must be within [0, max_depth_iters)")
        if self.depth_lora_ranks is None:
            return self.depth_lora_rank
        return self.depth_lora_ranks[loop_idx]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlexandrosConfig":
        allowed = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in allowed}
        return cls(**filtered)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def save_pretrained(self, save_directory: str | Path) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(self.to_json_string(), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "AlexandrosConfig":
        config_path = Path(path)
        if config_path.is_dir():
            config_path = config_path / "config.json"
        return cls.from_dict(json.loads(config_path.read_text(encoding="utf-8")))


def load_config_file(path: str | Path) -> AlexandrosConfig:
    """Load JSON or simple YAML config files.

    PyYAML is optional. The repository YAML files use a JSON-compatible scalar
    subset, so this function also includes a small fallback parser.
    """

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return AlexandrosConfig.from_dict(json.loads(text))
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data: dict[str, Any] = {}
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if value.lower() in {"true", "false"}:
                parsed: Any = value.lower() == "true"
            elif value.startswith("[") or value.startswith("("):
                parsed = ast.literal_eval(value)
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value.strip("\"'")
            data[key.strip()] = parsed
    return AlexandrosConfig.from_dict(data)
