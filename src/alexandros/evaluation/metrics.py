from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from alexandros.kv_cache import TurboQuantKVCache


@dataclass(frozen=True)
class TurboQuantReconstructionReport:
    mse: float
    max_abs_error: float
    estimated_bits: int
    uncompressed_bits: int

    @property
    def compression_ratio(self) -> float:
        return self.uncompressed_bits / max(self.estimated_bits, 1)


@dataclass(frozen=True)
class RuntimeProfile:
    prefill_ms: float
    generation_ms: float
    generated_tokens: int
    parameter_bytes: int
    trainable_parameter_bytes: int
    peak_cuda_bytes: int

    @property
    def generation_tokens_per_second(self) -> float:
        if self.generation_ms <= 0 or self.generated_tokens <= 0:
            return 0.0
        return self.generated_tokens / (self.generation_ms / 1000.0)


@dataclass(frozen=True)
class NeedleRetrievalReport:
    sequence_length: int
    needle_position: int
    needle_token_id: int
    target_rank: int
    target_probability: float
    top_token_id: int


@dataclass(frozen=True)
class LostInMiddleReport:
    sequence_length: int
    needle_positions: tuple[int, ...]
    target_ranks: tuple[int, ...]
    target_probabilities: tuple[float, ...]
    worst_rank: int
    middle_rank: int


@dataclass(frozen=True)
class CopyRetrievalReport:
    sequence_length: int
    source_position: int
    query_position: int
    copy_token_id: int
    target_rank: int
    target_probability: float
    top_token_id: int


@dataclass(frozen=True)
class RecurrentStateDriftReport:
    sequence_length: int
    layers_with_state: int
    max_state_norm: float
    mean_state_norm: float
    max_update_norm: float
    mean_update_norm: float
    finite: bool


@dataclass(frozen=True)
class ToyReasoningProbeReport:
    prompt_token_ids: tuple[int, ...]
    lhs: int
    rhs: int
    modulus: int
    target_token_id: int
    target_rank: int
    target_probability: float
    top_token_id: int


@dataclass(frozen=True)
class LatentReconstructionReport:
    reconstruction_mse: float
    vae_reconstruction_mse: float
    refinement_reconstruction_mse: float
    kl_loss: float
    latent_norm: float
    refined_latent_norm: float
    latent_update_norm: float
    reconstruction_norm: float
    refined_reconstruction_norm: float
    latent_steps: int


@dataclass(frozen=True)
class AdaptiveDepthToyBenchmarkReport:
    target_rank: int
    target_probability: float
    top_token_id: int
    average_loop_count: float
    ponder_cost: float
    elapsed_ms: float


def causal_lm_perplexity(model: Any, input_ids: torch.Tensor) -> float:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(input_ids, labels=input_ids)
        if output.loss is None:
            raise ValueError("model did not return a causal LM loss")
        return math.exp(float(output.loss.detach().cpu().item()))
    finally:
        model.train(was_training)


def masked_diffusion_reconstruction_accuracy(
    model: Any,
    input_ids: torch.Tensor,
    timestep: int | torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> float:
    if not hasattr(model, "scheduler"):
        raise TypeError("masked diffusion accuracy requires AlexandrosForDiffusionLM")
    timestep = model.config.diffusion_steps - 1 if timestep is None else timestep
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            noisy, mask = model.scheduler.add_noise(
                input_ids, timestep, generator=generator
            )
            output = model(noisy, diffusion_timestep=timestep)
            predictions = output.logits.argmax(dim=-1)
        if not mask.any():
            return 0.0
        return float(predictions[mask].eq(input_ids[mask]).float().mean().item())
    finally:
        model.train(was_training)


def latent_reconstruction_metrics(
    model: Any,
    input_ids: torch.Tensor,
    *,
    latent_steps: int = 1,
) -> LatentReconstructionReport:
    """Report latent VAE/refinement reconstruction diagnostics.

    The target is the same smoke-training proxy used by `train_latent.py`: the
    pooled backbone hidden state expanded across latent slots. These diagnostics
    are for mechanism auditability, not for judging reasoning quality.
    """

    if not hasattr(model, "latent_vae") or not hasattr(model, "latent_reasoner"):
        raise TypeError(
            "latent reconstruction metrics require AlexandrosForDiffusionLM"
        )
    if (
        not isinstance(latent_steps, int)
        or isinstance(latent_steps, bool)
        or latent_steps <= 0
    ):
        raise ValueError("latent_steps must be a positive integer")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            hidden = model.model(input_ids).last_hidden_state
            vae = model.latent_vae(hidden)
            target = hidden.mean(dim=1, keepdim=True).expand_as(vae.reconstruction)
            refined = model.latent_reasoner(vae.latents, steps=latent_steps)
            refined_reconstruction = model.latent_reasoner.decode_to_hidden(refined)
            vae_error = (vae.reconstruction.float() - target.float()).pow(2).mean()
            refined_error = (
                (refined_reconstruction.float() - target.float()).pow(2).mean()
            )
            reconstruction_mse = 0.5 * (vae_error + refined_error)
            latent_update = refined.float() - vae.latents.float()
        return LatentReconstructionReport(
            reconstruction_mse=float(reconstruction_mse.item()),
            vae_reconstruction_mse=float(vae_error.item()),
            refinement_reconstruction_mse=float(refined_error.item()),
            kl_loss=float(vae.kl_loss.detach().float().item()),
            latent_norm=float(vae.latents.detach().float().norm().item()),
            refined_latent_norm=float(refined.detach().float().norm().item()),
            latent_update_norm=float(latent_update.norm().item()),
            reconstruction_norm=float(
                vae.reconstruction.detach().float().norm().item()
            ),
            refined_reconstruction_norm=float(
                refined_reconstruction.detach().float().norm().item()
            ),
            latent_steps=latent_steps,
        )
    finally:
        model.train(was_training)


def turboquant_reconstruction_metrics(
    x: torch.Tensor,
    *,
    bits: int = 4,
    use_qjl: bool = False,
) -> TurboQuantReconstructionReport:
    cache = TurboQuantKVCache(bits=bits, use_qjl=use_qjl)
    packet = cache.compress(x)
    restored = cache.decompress(packet)
    error = (restored.float() - x.float()).abs()
    return TurboQuantReconstructionReport(
        mse=float(error.pow(2).mean().item()),
        max_abs_error=float(error.max().item()),
        estimated_bits=packet.estimated_bits,
        uncompressed_bits=x.numel() * torch.finfo(x.dtype).bits,
    )


def _parameter_bytes(model: Any, *, trainable_only: bool = False) -> int:
    total = 0
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        total += param.numel() * param.element_size()
    return total


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _generate_for_profile(
    model: Any, input_ids: torch.Tensor, *, max_new_tokens: int, mode: Any
) -> torch.Tensor:
    if hasattr(model, "scheduler"):
        return model.generate(input_ids, max_new_tokens=max_new_tokens, mode=mode)
    return model.generate(input_ids, max_new_tokens=max_new_tokens)


def profile_model_runtime(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 2,
    mode: Any = "autoregressive",
    warmup: int = 0,
    repeats: int = 1,
) -> RuntimeProfile:
    """Measure tiny local runtime and model-parameter memory.

    This is a smoke profiler, not a benchmark harness. It avoids process-level
    CPU memory sampling so it remains portable in CI-like environments.
    """

    if repeats <= 0:
        raise ValueError("repeats must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    was_training = model.training
    model.eval()
    device = input_ids.device
    peak_cuda_bytes = 0
    try:
        with torch.no_grad():
            for _ in range(warmup):
                model(input_ids)
                _generate_for_profile(
                    model, input_ids, max_new_tokens=max_new_tokens, mode=mode
                )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            prefill_total = 0.0
            generation_total = 0.0
            generated_tokens = 0
            for _ in range(repeats):
                _sync_if_needed(device)
                start = time.perf_counter()
                model(input_ids)
                _sync_if_needed(device)
                prefill_total += time.perf_counter() - start

                _sync_if_needed(device)
                start = time.perf_counter()
                output = _generate_for_profile(
                    model,
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    mode=mode,
                )
                _sync_if_needed(device)
                generation_total += time.perf_counter() - start
                generated_tokens += max(
                    output.size(1) - input_ids.size(1), 0
                ) * output.size(0)
            if device.type == "cuda":
                peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
        return RuntimeProfile(
            prefill_ms=(prefill_total / repeats) * 1000.0,
            generation_ms=(generation_total / repeats) * 1000.0,
            generated_tokens=generated_tokens // repeats,
            parameter_bytes=_parameter_bytes(model),
            trainable_parameter_bytes=_parameter_bytes(model, trainable_only=True),
            peak_cuda_bytes=peak_cuda_bytes,
        )
    finally:
        model.train(was_training)


def synthetic_needle_retrieval_probe(
    model: Any,
    *,
    sequence_length: int | None = None,
    needle_position: int | None = None,
    needle_token_id: int | None = None,
    filler_token_id: int = 4,
    query_token_id: int | None = None,
) -> NeedleRetrievalReport:
    """Run a token-level needle diagnostic on an untrained or trained model.

    The probe places a target token inside a synthetic context and reports the
    target token's rank at the final position. It is a diagnostic surface, not a
    pass/fail quality benchmark.
    """

    config = model.config
    sequence_length = (
        min(config.max_position_embeddings, 32)
        if sequence_length is None
        else sequence_length
    )
    if sequence_length < 3:
        raise ValueError("sequence_length must be >= 3")
    if sequence_length > config.max_position_embeddings:
        raise ValueError("sequence_length cannot exceed max_position_embeddings")
    needle_position = (
        sequence_length // 2 if needle_position is None else needle_position
    )
    if needle_position <= 0 or needle_position >= sequence_length - 1:
        raise ValueError(
            "needle_position must be inside the context and before the query"
        )

    needle_token_id = (
        min(8, config.vocab_size - 1) if needle_token_id is None else needle_token_id
    )
    query_token_id = (
        min(9, config.vocab_size - 1) if query_token_id is None else query_token_id
    )
    for name, token_id in {
        "needle_token_id": needle_token_id,
        "filler_token_id": filler_token_id,
        "query_token_id": query_token_id,
    }.items():
        if token_id < 0 or token_id >= config.vocab_size:
            raise ValueError(f"{name} must be in [0, vocab_size)")

    device = next(model.parameters()).device
    input_ids = torch.full(
        (1, sequence_length),
        filler_token_id,
        dtype=torch.long,
        device=device,
    )
    input_ids[:, 0] = config.bos_token_id
    input_ids[:, needle_position] = needle_token_id
    input_ids[:, -1] = query_token_id

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
            probs = logits.softmax(dim=-1)
        target_logit = logits[0, needle_token_id]
        target_rank = int(logits[0].gt(target_logit).sum().item()) + 1
        top_token_id = int(logits.argmax(dim=-1).item())
        return NeedleRetrievalReport(
            sequence_length=sequence_length,
            needle_position=needle_position,
            needle_token_id=needle_token_id,
            target_rank=target_rank,
            target_probability=float(probs[0, needle_token_id].item()),
            top_token_id=top_token_id,
        )
    finally:
        model.train(was_training)


def synthetic_lost_in_middle_probe(
    model: Any,
    *,
    sequence_length: int | None = None,
    needle_token_id: int | None = None,
) -> LostInMiddleReport:
    """Run early/middle/late synthetic retrieval probes.

    This is a diagnostic surface for comparing positions, not a quality claim.
    """

    config = model.config
    sequence_length = (
        min(config.max_position_embeddings, 32)
        if sequence_length is None
        else sequence_length
    )
    if sequence_length < 5:
        raise ValueError("sequence_length must be >= 5")
    if sequence_length > config.max_position_embeddings:
        raise ValueError("sequence_length cannot exceed max_position_embeddings")
    positions = (
        1,
        sequence_length // 2,
        sequence_length - 2,
    )
    reports = tuple(
        synthetic_needle_retrieval_probe(
            model,
            sequence_length=sequence_length,
            needle_position=position,
            needle_token_id=needle_token_id,
        )
        for position in positions
    )
    ranks = tuple(report.target_rank for report in reports)
    probabilities = tuple(report.target_probability for report in reports)
    return LostInMiddleReport(
        sequence_length=sequence_length,
        needle_positions=positions,
        target_ranks=ranks,
        target_probabilities=probabilities,
        worst_rank=max(ranks),
        middle_rank=ranks[1],
    )


def synthetic_copy_retrieval_probe(
    model: Any,
    *,
    sequence_length: int | None = None,
    source_position: int | None = None,
    copy_token_id: int | None = None,
    filler_token_id: int = 4,
    query_token_id: int | None = None,
) -> CopyRetrievalReport:
    """Place a copy token in context and rank it at a query position."""

    config = model.config
    sequence_length = (
        min(config.max_position_embeddings, 32)
        if sequence_length is None
        else sequence_length
    )
    if sequence_length < 4:
        raise ValueError("sequence_length must be >= 4")
    if sequence_length > config.max_position_embeddings:
        raise ValueError("sequence_length cannot exceed max_position_embeddings")
    source_position = (
        sequence_length // 3 if source_position is None else source_position
    )
    query_position = sequence_length - 1
    if source_position <= 0 or source_position >= query_position:
        raise ValueError(
            "source_position must be inside the context and before the query"
        )
    copy_token_id = (
        min(10, config.vocab_size - 1) if copy_token_id is None else copy_token_id
    )
    query_token_id = (
        min(11, config.vocab_size - 1) if query_token_id is None else query_token_id
    )
    for name, token_id in {
        "copy_token_id": copy_token_id,
        "filler_token_id": filler_token_id,
        "query_token_id": query_token_id,
    }.items():
        if token_id < 0 or token_id >= config.vocab_size:
            raise ValueError(f"{name} must be in [0, vocab_size)")

    device = next(model.parameters()).device
    input_ids = torch.full(
        (1, sequence_length),
        filler_token_id,
        dtype=torch.long,
        device=device,
    )
    input_ids[:, 0] = config.bos_token_id
    input_ids[:, source_position] = copy_token_id
    input_ids[:, query_position] = query_token_id

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(input_ids).logits[:, query_position, :]
            probs = logits.softmax(dim=-1)
        target_logit = logits[0, copy_token_id]
        target_rank = int(logits[0].gt(target_logit).sum().item()) + 1
        top_token_id = int(logits.argmax(dim=-1).item())
        return CopyRetrievalReport(
            sequence_length=sequence_length,
            source_position=source_position,
            query_position=query_position,
            copy_token_id=copy_token_id,
            target_rank=target_rank,
            target_probability=float(probs[0, copy_token_id].item()),
            top_token_id=top_token_id,
        )
    finally:
        model.train(was_training)


def recurrent_state_drift_probe(
    model: Any,
    *,
    sequence_length: int | None = None,
    filler_token_id: int = 4,
) -> RecurrentStateDriftReport:
    """Track recurrent SSM state norms during cached token-by-token prefill."""

    config = model.config
    sequence_length = (
        min(config.max_position_embeddings, 32)
        if sequence_length is None
        else sequence_length
    )
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")
    if sequence_length > config.max_position_embeddings:
        raise ValueError("sequence_length cannot exceed max_position_embeddings")
    if filler_token_id < 0 or filler_token_id >= config.vocab_size:
        raise ValueError("filler_token_id must be in [0, vocab_size)")
    device = next(model.parameters()).device
    input_ids = torch.full(
        (1, sequence_length),
        filler_token_id,
        dtype=torch.long,
        device=device,
    )
    input_ids[:, 0] = config.bos_token_id
    past_key_values = None
    past_ssm_states = None
    previous_states: list[torch.Tensor | None] | None = None
    state_norms: list[torch.Tensor] = []
    update_norms: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for idx in range(sequence_length):
                output = model(
                    input_ids[:, idx : idx + 1],
                    use_cache=True,
                    past_key_values=past_key_values,
                    past_ssm_states=past_ssm_states,
                )
                current_states = output.past_ssm_states or []
                for layer_idx, state in enumerate(current_states):
                    if state is None:
                        continue
                    state_norms.append(state.detach().float().norm())
                    if (
                        previous_states is not None
                        and previous_states[layer_idx] is not None
                    ):
                        update_norms.append(
                            (
                                state.detach().float()
                                - previous_states[layer_idx].detach().float()
                            ).norm()
                        )
                previous_states = current_states
                past_key_values = output.past_key_values
                past_ssm_states = output.past_ssm_states
        if state_norms:
            state_stack = torch.stack(state_norms)
            max_state_norm = float(state_stack.max().item())
            mean_state_norm = float(state_stack.mean().item())
            finite_states = bool(torch.isfinite(state_stack).all().item())
        else:
            max_state_norm = 0.0
            mean_state_norm = 0.0
            finite_states = True
        if update_norms:
            update_stack = torch.stack(update_norms)
            max_update_norm = float(update_stack.max().item())
            mean_update_norm = float(update_stack.mean().item())
            finite_updates = bool(torch.isfinite(update_stack).all().item())
        else:
            max_update_norm = 0.0
            mean_update_norm = 0.0
            finite_updates = True
        return RecurrentStateDriftReport(
            sequence_length=sequence_length,
            layers_with_state=sum(
                state is not None for state in (past_ssm_states or [])
            ),
            max_state_norm=max_state_norm,
            mean_state_norm=mean_state_norm,
            max_update_norm=max_update_norm,
            mean_update_norm=mean_update_norm,
            finite=finite_states and finite_updates,
        )
    finally:
        model.train(was_training)


def synthetic_modular_addition_probe(
    model: Any,
    *,
    lhs: int = 2,
    rhs: int = 3,
    modulus: int = 10,
    base_token_id: int = 12,
    operator_token_id: int = 10,
    equals_token_id: int = 11,
) -> ToyReasoningProbeReport:
    """Rank a toy modular-addition answer token.

    The probe is a deterministic mechanics check for evaluation harnesses. It
    does not imply that an untrained Alexandros checkpoint can do arithmetic.
    """

    for name, value in {"lhs": lhs, "rhs": rhs, "modulus": modulus}.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if modulus <= 1:
        raise ValueError("modulus must be > 1")
    config = model.config
    required_token_ids = {
        "base_token_id": base_token_id,
        "operator_token_id": operator_token_id,
        "equals_token_id": equals_token_id,
    }
    for name, token_id in required_token_ids.items():
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise ValueError(f"{name} must be an integer")
        if token_id < 0 or token_id >= config.vocab_size:
            raise ValueError(f"{name} must be in [0, vocab_size)")
    if base_token_id + modulus > config.vocab_size:
        raise ValueError("base_token_id + modulus must be <= vocab_size")
    lhs_token_id = base_token_id + (lhs % modulus)
    rhs_token_id = base_token_id + (rhs % modulus)
    target_token_id = base_token_id + ((lhs + rhs) % modulus)
    prompt = (
        config.bos_token_id,
        operator_token_id,
        lhs_token_id,
        rhs_token_id,
        equals_token_id,
    )
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
            probs = logits.softmax(dim=-1)
        target_logit = logits[0, target_token_id]
        target_rank = int(logits[0].gt(target_logit).sum().item()) + 1
        top_token_id = int(logits.argmax(dim=-1).item())
        return ToyReasoningProbeReport(
            prompt_token_ids=prompt,
            lhs=lhs,
            rhs=rhs,
            modulus=modulus,
            target_token_id=target_token_id,
            target_rank=target_rank,
            target_probability=float(probs[0, target_token_id].item()),
            top_token_id=top_token_id,
        )
    finally:
        model.train(was_training)


def adaptive_depth_toy_benchmark(model: Any) -> AdaptiveDepthToyBenchmarkReport:
    """Measure adaptive-depth cost on the toy modular-addition probe.

    The report is a mechanics diagnostic: target rank/probability are toy
    quality surfaces, while loop count, ponder cost, and elapsed time are cost
    surfaces. It is not evidence of real reasoning quality.
    """

    backbone = getattr(model, "model", None)
    controller = getattr(backbone, "adaptive_depth", None)
    if controller is None:
        raise TypeError("adaptive depth benchmark requires enable_adaptive_depth=True")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            _sync_if_needed(device)
            start = time.perf_counter()
            reasoning = synthetic_modular_addition_probe(model)
            _sync_if_needed(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        stats = controller.last_stats
        if stats is None:
            raise ValueError("adaptive depth controller did not record stats")
        return AdaptiveDepthToyBenchmarkReport(
            target_rank=reasoning.target_rank,
            target_probability=reasoning.target_probability,
            top_token_id=reasoning.top_token_id,
            average_loop_count=stats.average_loop_count,
            ponder_cost=float(stats.ponder_cost.detach().cpu().item()),
            elapsed_ms=elapsed_ms,
        )
    finally:
        model.train(was_training)
