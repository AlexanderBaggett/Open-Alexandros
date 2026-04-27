from __future__ import annotations

import math

from _deps import require_runtime_dependencies

require_runtime_dependencies()

import torch
from torch.nn import functional as F

from _common import (
    add_runtime_args,
    add_trainability_args,
    add_training_args,
    autocast_context,
    backward_loss,
    close_tensorboard_writer,
    data_source_name,
    evaluating,
    finish_optimization_step,
    load_config,
    load_training_state,
    make_arg_parser,
    make_grad_scaler,
    make_latent_trace_batch_iterator,
    make_tensorboard_writer,
    make_token_batch_iterator,
    next_token_batch,
    prepare_output_dir,
    prepare_training_model,
    resolve_training_hparams,
    run_metadata,
    save_training_state,
    seeded_generator,
    setup_torch_runtime,
    should_validate,
    validate_training_args,
    write_jsonl,
    write_tensorboard_scalars,
)
from alexandros import AlexandrosForDiffusionLM
from alexandros.training import (
    apply_trainability,
    objective_log_fields,
    phase_checkpoint_metadata,
    trainable_parameters,
)


def validate_latent_objective_args(args) -> None:
    for name in ("lambda_kl", "lambda_rec"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")
    if float(args.lambda_kl) == 0.0 and float(args.lambda_rec) == 0.0:
        raise ValueError("at least one of lambda_kl or lambda_rec must be > 0")
    if (
        not isinstance(args.latent_refinement_steps, int)
        or isinstance(args.latent_refinement_steps, bool)
        or args.latent_refinement_steps <= 0
    ):
        raise ValueError("latent_refinement_steps must be a positive integer")
    if getattr(args, "trace_jsonl", "") and getattr(args, "token_ids_jsonl", ""):
        raise ValueError(
            "use either trace_jsonl or token_ids_jsonl for latent training, not both"
        )
    trace_len = getattr(args, "trace_len", None)
    if trace_len is not None and (
        not isinstance(trace_len, int) or isinstance(trace_len, bool) or trace_len <= 0
    ):
        raise ValueError("trace_len must be a positive integer")


def _masked_mean(hidden_states, attention_mask):
    hidden_states = hidden_states.detach()
    if attention_mask is None:
        return hidden_states.mean(dim=1, keepdim=True)
    mask = attention_mask.to(
        device=hidden_states.device, dtype=hidden_states.dtype
    ).unsqueeze(-1)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1, keepdim=True) / denom


def latent_loss(
    model,
    input_ids,
    torch,
    F,
    *,
    attention_mask=None,
    trace_ids=None,
    trace_attention_mask=None,
    lambda_kl: float = 0.01,
    lambda_rec: float = 1.0,
    latent_refinement_steps: int = 1,
):
    hidden = model.model(input_ids, attention_mask=attention_mask).last_hidden_state
    vae = model.latent_vae(hidden)
    if trace_ids is None:
        pooled_target = _masked_mean(hidden, attention_mask)
    else:
        with torch.no_grad():
            trace_hidden = model.model(
                trace_ids,
                attention_mask=trace_attention_mask,
            ).last_hidden_state
        pooled_target = _masked_mean(trace_hidden, trace_attention_mask)
    recon_target = pooled_target.to(dtype=vae.reconstruction.dtype).expand_as(
        vae.reconstruction
    )
    vae_reconstruction_loss = F.mse_loss(vae.reconstruction, recon_target)
    refined = model.latent_reasoner(vae.latents, steps=latent_refinement_steps)
    refined_reconstruction = model.latent_reasoner.decode_to_hidden(refined)
    refinement_reconstruction_loss = F.mse_loss(refined_reconstruction, recon_target)
    reconstruction_loss = 0.5 * (
        vae_reconstruction_loss + refinement_reconstruction_loss
    )
    loss = lambda_rec * reconstruction_loss + lambda_kl * vae.kl_loss
    return (
        loss,
        vae,
        refined,
        reconstruction_loss,
        vae_reconstruction_loss,
        refinement_reconstruction_loss,
    )


def next_latent_batch(
    trace_batch_iter,
    token_batch_iter,
    config,
    args,
    device,
    torch,
    *,
    generator=None,
):
    if trace_batch_iter is not None:
        return next(trace_batch_iter)
    return next_token_batch(
        token_batch_iter,
        config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
        torch=torch,
        generator=generator,
    )


def run_validation(
    model,
    config,
    args,
    device,
    torch,
    F,
    step: int,
    batch_iter=None,
    trace_batch_iter=None,
    precision_context=None,
) -> dict[str, float]:
    if precision_context is None:
        precision_context = autocast_context(device, torch.float32, torch)
    loss_total = 0.0
    kl_loss_total = 0.0
    reconstruction_loss_total = 0.0
    vae_reconstruction_loss_total = 0.0
    refinement_reconstruction_loss_total = 0.0
    latent_norm_total = 0.0
    latent_update_norm_total = 0.0
    recon_norm_total = 0.0
    generator = seeded_generator(torch, int(args.seed) + 3_000_003 + step, device)
    with evaluating(model, torch):
        for _ in range(args.val_batches):
            batch = next_latent_batch(
                trace_batch_iter,
                batch_iter,
                config,
                args,
                device=device,
                torch=torch,
                generator=generator,
            )
            if hasattr(batch, "input_ids"):
                input_ids = batch.input_ids
                attention_mask = batch.input_attention_mask
                trace_ids = batch.trace_ids
                trace_attention_mask = batch.trace_attention_mask
            else:
                input_ids = batch
                attention_mask = None
                trace_ids = None
                trace_attention_mask = None
            with precision_context():
                (
                    loss,
                    vae,
                    refined,
                    reconstruction_loss,
                    vae_reconstruction_loss,
                    refinement_reconstruction_loss,
                ) = latent_loss(
                    model,
                    input_ids,
                    torch,
                    F,
                    attention_mask=attention_mask,
                    trace_ids=trace_ids,
                    trace_attention_mask=trace_attention_mask,
                    lambda_kl=args.lambda_kl,
                    lambda_rec=args.lambda_rec,
                    latent_refinement_steps=args.latent_refinement_steps,
                )
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(
                    f"non-finite validation loss encountered: {loss.item()}"
                )
            loss_total += loss.detach().item()
            kl_loss_total += vae.kl_loss.detach().item()
            reconstruction_loss_total += reconstruction_loss.detach().item()
            vae_reconstruction_loss_total += vae_reconstruction_loss.detach().item()
            refinement_reconstruction_loss_total += (
                refinement_reconstruction_loss.detach().item()
            )
            latent_norm_total += vae.latents.detach().norm().item()
            recon_norm_total += vae.reconstruction.detach().norm().item()
            latent_update_norm_total += (
                (refined.detach() - vae.latents.detach()).norm().item()
            )
    return {
        "val_loss": loss_total / args.val_batches,
        "val_reconstruction_loss": reconstruction_loss_total / args.val_batches,
        "val_vae_reconstruction_loss": vae_reconstruction_loss_total / args.val_batches,
        "val_refinement_reconstruction_loss": refinement_reconstruction_loss_total
        / args.val_batches,
        "val_kl_loss": kl_loss_total / args.val_batches,
        "val_latent_norm": latent_norm_total / args.val_batches,
        "val_recon_norm": recon_norm_total / args.val_batches,
        "val_reconstruction_norm": recon_norm_total / args.val_batches,
        "val_latent_update_norm": latent_update_norm_total / args.val_batches,
    }


def main() -> None:
    parser = make_arg_parser("Run a smoke latent reasoning training loop.")
    parser.add_argument("--config", default="configs/heavy_tiny.yaml")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--lambda-kl", type=float, default=0.01)
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--latent-refinement-steps", type=int, default=1)
    parser.add_argument("--trace-jsonl", default="")
    parser.add_argument("--trace-len", type=int, default=None)
    parser.add_argument("--trace-input-field", default="input_ids")
    parser.add_argument("--trace-field", default="trace_ids")
    parser.add_argument("--trace-target-field", default="target_ids")
    add_runtime_args(parser)
    add_training_args(parser)
    add_trainability_args(parser)
    args = parser.parse_args()
    validate_training_args(args)
    validate_latent_objective_args(args)

    config = load_config(args.config)
    resolve_training_hparams(args, config)
    device, dtype = setup_torch_runtime(args, torch)
    precision_context = autocast_context(device, dtype, torch)
    grad_scaler = make_grad_scaler(device, dtype, torch)
    args.amp_enabled = dtype != torch.float32
    args.grad_scaler_enabled = grad_scaler is not None
    out_dir = prepare_output_dir(args)
    tb_writer = make_tensorboard_writer(args)
    model = prepare_training_model(
        AlexandrosForDiffusionLM(config), device, dtype, torch
    )
    trainability = apply_trainability(model, phase="latent", scope=args.trainable_scope)
    opt_params = list(trainable_parameters(model))
    if not opt_params:
        raise ValueError("latent training requires at least one trainable parameter")
    opt = torch.optim.AdamW(opt_params, lr=args.lr)
    opt.zero_grad(set_to_none=True)
    train_trace_iter = make_latent_trace_batch_iterator(
        args,
        config,
        split="train",
        device=device,
        torch=torch,
    )
    train_batch_iter = make_token_batch_iterator(
        args,
        config,
        split="train",
        device=device,
        torch=torch,
    )
    val_trace_iter = (
        make_latent_trace_batch_iterator(
            args,
            config,
            split="validation",
            device=device,
            torch=torch,
        )
        if args.trace_jsonl and args.val_every
        else None
    )
    val_batch_iter = (
        make_token_batch_iterator(
            args,
            config,
            split="validation",
            device=device,
            torch=torch,
        )
        if args.token_ids_jsonl and args.val_every
        else None
    )
    start_step = 0
    if args.resume:
        start_step = load_training_state(
            args.resume,
            model=model,
            optimizer=opt,
            config=config,
            torch=torch,
            map_location=device,
            data_iterator=train_trace_iter or train_batch_iter,
            grad_scaler=grad_scaler,
        )
    metadata = run_metadata(args, config, torch)
    objective_fields = objective_log_fields("latent")
    checkpoint_metadata = phase_checkpoint_metadata("latent", trainability)
    final_step = start_step - 1
    for step in range(start_step, start_step + args.steps):
        loss_total = 0.0
        kl_loss_total = 0.0
        reconstruction_loss_total = 0.0
        vae_reconstruction_loss_total = 0.0
        refinement_reconstruction_loss_total = 0.0
        latent_norm_total = 0.0
        latent_update_norm_total = 0.0
        recon_norm_total = 0.0
        for _ in range(args.grad_accum_steps):
            batch = next_latent_batch(
                train_trace_iter,
                train_batch_iter,
                config,
                args,
                device=device,
                torch=torch,
            )
            if hasattr(batch, "input_ids"):
                input_ids = batch.input_ids
                attention_mask = batch.input_attention_mask
                trace_ids = batch.trace_ids
                trace_attention_mask = batch.trace_attention_mask
            else:
                input_ids = batch
                attention_mask = None
                trace_ids = None
                trace_attention_mask = None
            with precision_context():
                (
                    loss,
                    vae,
                    refined,
                    reconstruction_loss,
                    vae_reconstruction_loss,
                    refinement_reconstruction_loss,
                ) = latent_loss(
                    model,
                    input_ids,
                    torch,
                    F,
                    attention_mask=attention_mask,
                    trace_ids=trace_ids,
                    trace_attention_mask=trace_attention_mask,
                    lambda_kl=args.lambda_kl,
                    lambda_rec=args.lambda_rec,
                    latent_refinement_steps=args.latent_refinement_steps,
                )
            loss_total += loss.detach().item()
            kl_loss_total += vae.kl_loss.detach().item()
            reconstruction_loss_total += reconstruction_loss.detach().item()
            vae_reconstruction_loss_total += vae_reconstruction_loss.detach().item()
            refinement_reconstruction_loss_total += (
                refinement_reconstruction_loss.detach().item()
            )
            latent_norm_total += vae.latents.detach().norm().item()
            latent_update_norm_total += (
                (refined.detach() - vae.latents.detach()).norm().item()
            )
            recon_norm_total += vae.reconstruction.detach().norm().item()
            backward_loss(loss, args, torch, grad_scaler=grad_scaler)
        metrics = finish_optimization_step(
            model, opt, args, torch, step=step, grad_scaler=grad_scaler
        )
        loss_value = loss_total / args.grad_accum_steps
        kl_loss_value = kl_loss_total / args.grad_accum_steps
        reconstruction_loss_value = reconstruction_loss_total / args.grad_accum_steps
        vae_reconstruction_loss_value = (
            vae_reconstruction_loss_total / args.grad_accum_steps
        )
        refinement_reconstruction_loss_value = (
            refinement_reconstruction_loss_total / args.grad_accum_steps
        )
        latent_norm = latent_norm_total / args.grad_accum_steps
        latent_update_norm = latent_update_norm_total / args.grad_accum_steps
        recon_norm = recon_norm_total / args.grad_accum_steps
        val_metrics = (
            run_validation(
                model,
                config,
                args,
                device,
                torch,
                F,
                step,
                val_batch_iter,
                val_trace_iter,
                precision_context,
            )
            if should_validate(args, step)
            else {}
        )
        val_text = f" val_loss={val_metrics['val_loss']:.4f}" if val_metrics else ""
        print(
            f"latent_step={step} loss={loss_value:.4f} "
            f"grad_norm={metrics['grad_norm']:.4f} lr={metrics['lr']:.6g} "
            f"latent_norm={latent_norm:.4f} recon_norm={recon_norm:.4f}"
            f"{val_text}"
        )
        metrics_record = {
            **metadata,
            **objective_fields,
            **trainability.to_log_fields(),
            "phase": "latent",
            "step": step,
            "loss": loss_value,
            "reconstruction_loss": reconstruction_loss_value,
            "vae_reconstruction_loss": vae_reconstruction_loss_value,
            "refinement_reconstruction_loss": refinement_reconstruction_loss_value,
            "kl_loss": kl_loss_value,
            "lambda_kl": args.lambda_kl,
            "lambda_rec": args.lambda_rec,
            "latent_refinement_steps": args.latent_refinement_steps,
            "grad_norm": metrics["grad_norm"],
            "lr": metrics["lr"],
            "latent_norm": latent_norm,
            "latent_update_norm": latent_update_norm,
            "recon_norm": recon_norm,
            "reconstruction_norm": recon_norm,
            "grad_accum_steps": args.grad_accum_steps,
            "data_source": data_source_name(args),
            "amp_enabled": args.amp_enabled,
            "grad_scaler_enabled": args.grad_scaler_enabled,
            **val_metrics,
        }
        write_jsonl(args.log_jsonl, metrics_record)
        write_tensorboard_scalars(tb_writer, metrics_record, step=step)
        final_step = step
        if out_dir is not None:
            save_training_state(
                out_dir / "training_state.pt",
                model=model,
                optimizer=opt,
                config=config,
                args=args,
                torch=torch,
                step=step,
                data_iterator=train_trace_iter or train_batch_iter,
                grad_scaler=grad_scaler,
            )
    if out_dir is not None:
        model.save_pretrained(
            out_dir / "checkpoint", checkpoint_metadata=checkpoint_metadata
        )
        if final_step < start_step:
            save_training_state(
                out_dir / "training_state.pt",
                model=model,
                optimizer=opt,
                config=config,
                args=args,
                torch=torch,
                step=final_step,
                data_iterator=train_trace_iter or train_batch_iter,
                grad_scaler=grad_scaler,
            )
    close_tensorboard_writer(tb_writer)


if __name__ == "__main__":
    main()
