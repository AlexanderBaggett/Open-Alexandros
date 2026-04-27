from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alexandros.configuration_alexandros import (  # noqa: E402
    AlexandrosConfig,
    load_config_file,
)
from alexandros.training import (  # noqa: E402
    TRAINABLE_SCOPES,
    LatentTraceDataset,
    PackedTokenDataset,
)


def make_arg_parser(description: str | None = None) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


def load_config(path: str | None) -> AlexandrosConfig:
    if path is None:
        return AlexandrosConfig()
    return load_config_file(path)


def add_runtime_args(parser: Any) -> None:
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")


def add_training_args(parser: Any, *, default_lr: float | None = None) -> None:
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--val-batches", type=int, default=1)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--log-jsonl", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--token-ids-jsonl", default="")
    parser.add_argument("--token-field", default="input_ids")
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--no-shuffle-data", action="store_true")
    parser.add_argument(
        "--tensorboard-dir",
        default="",
        help="Optional TensorBoard log directory; requires the tensorboard package.",
    )


def add_trainability_args(parser: Any) -> None:
    parser.add_argument(
        "--trainable-scope",
        choices=TRAINABLE_SCOPES,
        default="phase_default",
        help=(
            "Select which parameter groups may update in the smoke trainer. "
            "'phase_default' uses the documented default for the trainer phase."
        ),
    )


def setup_torch_runtime(args: Any, torch: Any) -> tuple[Any, Any]:
    torch.manual_seed(args.seed)
    if getattr(args, "deterministic", False):
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    return device, dtype


def prepare_model(model: Any, device: Any, dtype: Any, torch: Any) -> Any:
    model = model.to(device=device)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    return model


def prepare_training_model(model: Any, device: Any, dtype: Any, torch: Any) -> Any:
    if str(device).startswith("cpu") and dtype == torch.float16:
        raise ValueError(
            "float16 training is not supported on CPU; use bfloat16 or float32"
        )
    return model.to(device=device)


def autocast_context(device: Any, dtype: Any, torch: Any):
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    enabled = dtype != torch.float32

    @contextmanager
    def context() -> Iterator[None]:
        if not enabled:
            yield
            return
        with torch.autocast(device_type=device_type, dtype=dtype):
            yield

    return context


def make_grad_scaler(device: Any, dtype: Any, torch: Any) -> Any | None:
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if device_type != "cuda" or dtype != torch.float16:
        return None
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        try:
            return amp.GradScaler("cuda")
        except TypeError:
            return amp.GradScaler(enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def prepare_output_dir(args: Any) -> Path | None:
    out_dir = getattr(args, "out_dir", "")
    if not out_dir:
        return None
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    if not getattr(args, "log_jsonl", ""):
        args.log_jsonl = str(path / "metrics.jsonl")
    return path


def validate_training_args(args: Any) -> None:
    lr = getattr(args, "lr", None)
    if lr is not None and (not math.isfinite(float(lr)) or lr <= 0):
        raise ValueError("lr must be finite and > 0")
    if hasattr(args, "steps"):
        _validate_int_arg(args.steps, "steps", min_value=1)
    if hasattr(args, "batch_size"):
        _validate_int_arg(args.batch_size, "batch_size", min_value=1)
    if hasattr(args, "seq_len"):
        _validate_int_arg(args.seq_len, "seq_len", min_value=1)
    _validate_int_arg(getattr(args, "warmup_steps", 0), "warmup_steps", min_value=0)
    _validate_int_arg(
        getattr(args, "grad_accum_steps", 1), "grad_accum_steps", min_value=1
    )
    _validate_int_arg(getattr(args, "val_every", 0), "val_every", min_value=0)
    _validate_int_arg(getattr(args, "val_batches", 1), "val_batches", min_value=1)
    grad_clip = getattr(args, "grad_clip", 1.0)
    if not math.isfinite(float(grad_clip)) or grad_clip < 0:
        raise ValueError("grad_clip must be finite and >= 0")
    validation_fraction = float(getattr(args, "validation_fraction", 0.0))
    if not math.isfinite(validation_fraction) or not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must satisfy 0 <= fraction < 1")


def _validate_int_arg(value: Any, name: str, *, min_value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value < min_value:
        comparator = ">=" if min_value == 0 else ">"
        shown_min = min_value if min_value == 0 else min_value - 1
        raise ValueError(f"{name} must be {comparator} {shown_min}")


def should_validate(args: Any, step: int) -> bool:
    val_every = int(getattr(args, "val_every", 0))
    return val_every > 0 and step % val_every == 0


def seeded_generator(torch: Any, seed: int, device: Any) -> Any:
    if str(device).startswith("cuda"):
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def default_learning_rate(config: AlexandrosConfig) -> float:
    return 3e-4 if config.variant == "lite" else 1e-3


def resolve_training_hparams(args: Any, config: AlexandrosConfig) -> None:
    if getattr(args, "lr", None) is None:
        args.lr = default_learning_rate(config)


def data_source_name(args: Any) -> str:
    if getattr(args, "trace_jsonl", ""):
        return "latent_trace_jsonl"
    return (
        "token_ids_jsonl" if getattr(args, "token_ids_jsonl", "") else "synthetic_smoke"
    )


def make_token_batch_iterator(
    args: Any,
    config: AlexandrosConfig,
    *,
    split: str,
    device: Any,
    torch: Any,
    prepend_bos: bool = False,
) -> Iterator[Any] | None:
    path = getattr(args, "token_ids_jsonl", "")
    if not path:
        return None
    if (
        split == "validation"
        and float(getattr(args, "validation_fraction", 0.0)) <= 0.0
    ):
        raise ValueError(
            "validation_fraction must be > 0 when validating token_ids_jsonl data"
        )
    dataset = PackedTokenDataset.from_jsonl(
        path,
        config,
        seq_len=int(args.seq_len),
        field=getattr(args, "token_field", "input_ids"),
        split=split,
        validation_fraction=float(getattr(args, "validation_fraction", 0.0)),
        seed=int(getattr(args, "seed", 0)),
        prepend_bos=prepend_bos,
        append_eos=True,
    )
    seed_offset = 17 if split == "train" else 1_000_017
    return dataset.batch_iterator(
        batch_size=int(args.batch_size),
        torch=torch,
        device=device,
        seed=int(getattr(args, "seed", 0)) + seed_offset,
        shuffle=split == "train" and not bool(getattr(args, "no_shuffle_data", False)),
    )


def next_token_batch(
    batch_iterator: Iterator[Any] | None,
    config: AlexandrosConfig,
    *,
    batch_size: int,
    seq_len: int,
    device: Any,
    torch: Any,
    generator: Any = None,
    force_bos: bool = False,
) -> Any:
    if batch_iterator is not None:
        return next(batch_iterator)
    return sample_token_batch(
        config,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        torch=torch,
        generator=generator,
        force_bos=force_bos,
    )


def make_latent_trace_batch_iterator(
    args: Any,
    config: AlexandrosConfig,
    *,
    split: str,
    device: Any,
    torch: Any,
) -> Iterator[Any] | None:
    path = getattr(args, "trace_jsonl", "")
    if not path:
        return None
    trace_len = getattr(args, "trace_len", None)
    if trace_len is None:
        trace_len = int(args.seq_len)
    dataset = LatentTraceDataset.from_jsonl(
        path,
        config,
        seq_len=int(args.seq_len),
        trace_len=int(trace_len),
        input_field=getattr(args, "trace_input_field", "input_ids"),
        trace_field=getattr(args, "trace_field", "trace_ids"),
        target_field=getattr(args, "trace_target_field", "target_ids"),
        split=split,
        validation_fraction=float(getattr(args, "validation_fraction", 0.0)),
        seed=int(getattr(args, "seed", 0)),
    )
    seed_offset = 31 if split == "train" else 1_000_031
    return dataset.batch_iterator(
        batch_size=int(args.batch_size),
        torch=torch,
        device=device,
        seed=int(getattr(args, "seed", 0)) + seed_offset,
        shuffle=split == "train" and not bool(getattr(args, "no_shuffle_data", False)),
    )


def sample_token_batch(
    config: AlexandrosConfig,
    *,
    batch_size: int,
    seq_len: int,
    device: Any,
    torch: Any,
    generator: Any = None,
    force_bos: bool = False,
) -> Any:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if config.vocab_size <= 4:
        raise ValueError("synthetic token batches require vocab_size > 4")
    input_ids = torch.randint(
        4,
        config.vocab_size,
        (batch_size, seq_len),
        device=device,
        generator=generator,
    )
    if force_bos:
        input_ids[:, 0] = config.bos_token_id
    return input_ids


@contextmanager
def evaluating(model: Any, torch: Any) -> Iterator[None]:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def config_hash(config: AlexandrosConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def run_metadata(args: Any, config: AlexandrosConfig, torch: Any) -> dict[str, Any]:
    dependency_versions = {"torch": torch.__version__}
    try:
        import numpy  # type: ignore

        dependency_versions["numpy"] = numpy.__version__
    except Exception:
        dependency_versions["numpy"] = "unavailable"
    try:
        import yaml  # type: ignore

        dependency_versions["pyyaml"] = yaml.__version__
    except Exception:
        dependency_versions["pyyaml"] = "unavailable"
    return {
        "config_hash": config_hash(config),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "dependency_versions": dependency_versions,
        "device": str(getattr(args, "device", "cpu")),
        "dtype": str(getattr(args, "dtype", "float32")),
        "amp_enabled": bool(getattr(args, "amp_enabled", False)),
        "grad_scaler_enabled": bool(getattr(args, "grad_scaler_enabled", False)),
        "seed": int(getattr(args, "seed", 0)),
        "deterministic": bool(getattr(args, "deterministic", False)),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def write_jsonl(path: str | Path | None, record: dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def make_tensorboard_writer(args: Any) -> Any | None:
    log_dir = getattr(args, "tensorboard_dir", "")
    if not log_dir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except Exception as exc:
        raise ImportError(
            "--tensorboard-dir requires the optional 'tensorboard' package"
        ) from exc
    return SummaryWriter(log_dir=str(log_dir))


def write_tensorboard_scalars(
    writer: Any | None, record: dict[str, Any], *, step: int
) -> None:
    if writer is None:
        return
    for key, value in record.items():
        if isinstance(value, bool):
            scalar = float(value)
        elif isinstance(value, int | float) and not isinstance(value, bool):
            scalar = float(value)
        else:
            continue
        if math.isfinite(scalar):
            writer.add_scalar(key, scalar, step)
    flush = getattr(writer, "flush", None)
    if flush is not None:
        flush()


def close_tensorboard_writer(writer: Any | None) -> None:
    if writer is None:
        return
    close = getattr(writer, "close", None)
    if close is not None:
        close()


def _iterator_state(data_iterator: Any | None) -> dict[str, Any] | None:
    if data_iterator is None:
        return None
    state_dict = getattr(data_iterator, "state_dict", None)
    if state_dict is None:
        return None
    return state_dict()


def _load_iterator_state(
    data_iterator: Any | None, state: dict[str, Any] | None
) -> None:
    if data_iterator is None or state is None:
        return
    load_state_dict = getattr(data_iterator, "load_state_dict", None)
    if load_state_dict is None:
        raise ValueError(
            "training state contains data_state but iterator cannot restore it"
        )
    load_state_dict(state)


def save_training_state(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    config: AlexandrosConfig,
    args: Any,
    torch: Any,
    step: int,
    data_iterator: Any | None = None,
    grad_scaler: Any | None = None,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "open-alexandros-training-state",
        "format_version": 1,
        "step": step,
        "config_hash": config_hash(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
        "data_state": _iterator_state(data_iterator),
        "grad_scaler_state_dict": None
        if grad_scaler is None
        else grad_scaler.state_dict(),
        "metadata": run_metadata(args, config, torch),
    }
    torch.save(state, out_path)


def load_training_state(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    config: AlexandrosConfig,
    torch: Any,
    map_location: Any = "cpu",
    data_iterator: Any | None = None,
    grad_scaler: Any | None = None,
) -> int:
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=map_location)
    if state.get("format") != "open-alexandros-training-state":
        raise ValueError("unsupported training checkpoint format")
    if state.get("format_version", 0) > 1:
        raise ValueError("unsupported training checkpoint version")
    if state.get("config_hash") != config_hash(config):
        raise ValueError(
            "training checkpoint config hash does not match current config"
        )
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    rng_state = state.get("torch_rng_state")
    if rng_state is not None:
        torch.set_rng_state(rng_state.cpu())
    cuda_rng_state_all = state.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state_all)
    _load_iterator_state(data_iterator, state.get("data_state"))
    scaler_state = state.get("grad_scaler_state_dict")
    if grad_scaler is not None and scaler_state:
        grad_scaler.load_state_dict(scaler_state)
    return int(state.get("step", -1)) + 1


def _grad_norm(model: Any, args: Any, torch: Any) -> Any:
    parameters = [param for param in model.parameters() if param.grad is not None]
    if getattr(args, "grad_clip", 0.0) and args.grad_clip > 0:
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                args.grad_clip,
                error_if_nonfinite=True,
            )
        except TypeError:
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite gradient norm encountered: {grad_norm.item()}"
                )
    elif parameters:
        grad_norm = torch.linalg.vector_norm(
            torch.stack([param.grad.detach().norm() for param in parameters])
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"non-finite gradient norm encountered: {grad_norm.item()}"
            )
    else:
        grad_norm = torch.tensor(0.0)
    return grad_norm


def backward_loss(
    loss: Any, args: Any, torch: Any, *, grad_scaler: Any | None = None
) -> None:
    if not torch.isfinite(loss.detach()):
        raise FloatingPointError(f"non-finite loss encountered: {loss.item()}")
    accum_steps = int(getattr(args, "grad_accum_steps", 1))
    if accum_steps <= 0:
        raise ValueError("grad_accum_steps must be > 0")
    scaled_loss = loss / accum_steps
    if grad_scaler is not None:
        grad_scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()


def _router_bias_update_due(module: Any, step: int | None) -> bool:
    if step is None:
        return True
    config = getattr(module, "config", None)
    interval = int(getattr(config, "router_bias_update_interval", 1))
    if interval <= 0:
        raise ValueError("router_bias_update_interval must be > 0")
    return (step + 1) % interval == 0


def _scheduled_learning_rate(optimizer: Any, args: Any, step: int | None) -> float:
    base_lr = getattr(args, "lr", None)
    if base_lr is None:
        base_lr = optimizer.param_groups[0].get("lr", 0.0)
    base_lr = float(base_lr)
    warmup_steps = int(getattr(args, "warmup_steps", 0))
    if step is not None and warmup_steps > 0:
        lr = base_lr * min(1.0, float(step + 1) / float(warmup_steps))
    else:
        lr = base_lr
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def finish_optimization_step(
    model: Any,
    optimizer: Any,
    args: Any,
    torch: Any,
    *,
    step: int | None = None,
    grad_scaler: Any | None = None,
) -> dict[str, float]:
    if grad_scaler is not None:
        grad_scaler.unscale_(optimizer)
    grad_norm = _grad_norm(model, args, torch)
    lr = _scheduled_learning_rate(optimizer, args, step)
    if grad_scaler is not None:
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        optimizer.step()
    for module in model.modules():
        update_router_bias = getattr(module, "update_router_bias", None)
        if update_router_bias is not None and _router_bias_update_due(module, step):
            update_router_bias()
    optimizer.zero_grad(set_to_none=True)
    return {"grad_norm": float(grad_norm.detach().cpu().item()), "lr": lr}


def optimization_step(
    loss: Any,
    model: Any,
    optimizer: Any,
    args: Any,
    torch: Any,
    *,
    step: int | None = None,
    grad_scaler: Any | None = None,
) -> dict[str, float]:
    backward_loss(loss, args, torch, grad_scaler=grad_scaler)
    return finish_optimization_step(
        model, optimizer, args, torch, step=step, grad_scaler=grad_scaler
    )
