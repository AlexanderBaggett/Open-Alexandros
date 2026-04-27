from __future__ import annotations

from _deps import require_runtime_dependencies

require_runtime_dependencies()

import torch

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
from alexandros.evaluation import summarize_moe_stats
from alexandros.training import (
    apply_trainability,
    objective_log_fields,
    phase_checkpoint_metadata,
    trainable_parameters,
)


def run_validation(
    model,
    config,
    args,
    device,
    torch,
    step: int,
    batch_iter=None,
    precision_context=None,
) -> dict[str, float]:
    if precision_context is None:
        precision_context = autocast_context(device, torch.float32, torch)
    loss_total = 0.0
    logit_norm_total = 0.0
    generator = seeded_generator(torch, int(args.seed) + 2_000_003 + step, device)
    with evaluating(model, torch):
        for _ in range(args.val_batches):
            input_ids = next_token_batch(
                batch_iter,
                config,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                device=device,
                torch=torch,
                generator=generator,
            )
            with precision_context():
                out = model.diffusion_loss(input_ids, generator=generator)
            if out.loss is None:
                raise ValueError("validation diffusion forward did not return a loss")
            if not torch.isfinite(out.loss.detach()):
                raise FloatingPointError(
                    f"non-finite validation loss encountered: {out.loss.item()}"
                )
            loss_total += out.loss.detach().item()
            logit_norm_total += out.logits.detach().norm().item()
    return {
        "val_loss": loss_total / args.val_batches,
        "val_logit_norm": logit_norm_total / args.val_batches,
    }


def main() -> None:
    parser = make_arg_parser("Run a smoke masked diffusion training loop.")
    parser.add_argument("--config", default="configs/heavy_tiny.yaml")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    add_runtime_args(parser)
    add_training_args(parser)
    add_trainability_args(parser)
    args = parser.parse_args()
    validate_training_args(args)

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
    trainability = apply_trainability(
        model, phase="diffusion", scope=args.trainable_scope
    )
    opt_params = list(trainable_parameters(model))
    if not opt_params:
        raise ValueError("diffusion training requires at least one trainable parameter")
    opt = torch.optim.AdamW(opt_params, lr=args.lr)
    opt.zero_grad(set_to_none=True)
    train_batch_iter = make_token_batch_iterator(
        args,
        config,
        split="train",
        device=device,
        torch=torch,
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
            data_iterator=train_batch_iter,
            grad_scaler=grad_scaler,
        )
    metadata = run_metadata(args, config, torch)
    objective_fields = objective_log_fields("diffusion")
    checkpoint_metadata = phase_checkpoint_metadata("diffusion", trainability)
    final_step = start_step - 1
    for step in range(start_step, start_step + args.steps):
        loss_total = 0.0
        logit_norm_total = 0.0
        for _ in range(args.grad_accum_steps):
            input_ids = next_token_batch(
                train_batch_iter,
                config,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                device=device,
                torch=torch,
            )
            with precision_context():
                out = model.diffusion_loss(input_ids)
            assert out.loss is not None
            loss_total += out.loss.detach().item()
            logit_norm_total += out.logits.detach().norm().item()
            backward_loss(out.loss, args, torch, grad_scaler=grad_scaler)
        moe = summarize_moe_stats(model)
        metrics = finish_optimization_step(
            model, opt, args, torch, step=step, grad_scaler=grad_scaler
        )
        loss_value = loss_total / args.grad_accum_steps
        logit_norm = logit_norm_total / args.grad_accum_steps
        val_metrics = (
            run_validation(
                model,
                config,
                args,
                device,
                torch,
                step,
                val_batch_iter,
                precision_context,
            )
            if should_validate(args, step)
            else {}
        )
        val_text = f" val_loss={val_metrics['val_loss']:.4f}" if val_metrics else ""
        print(
            f"diffusion_step={step} loss={loss_value:.4f} "
            f"grad_norm={metrics['grad_norm']:.4f} lr={metrics['lr']:.6g} "
            f"logit_norm={logit_norm:.4f} "
            f"objective={config.diffusion_objective} "
            f"weighting={config.diffusion_loss_weighting} "
            f"moe_entropy={moe.mean_load_entropy:.4f} "
            f"moe_noisy_entropy={moe.noisy_step_load_entropy:.4f} "
            f"moe_polish_entropy={moe.polish_step_load_entropy:.4f} "
            f"moe_timestep_selections={moe.timestep_tracked_selections}"
            f"{val_text}"
        )
        metrics_record = {
            **metadata,
            **objective_fields,
            **trainability.to_log_fields(),
            "phase": "diffusion",
            "step": step,
            "loss": loss_value,
            "grad_norm": metrics["grad_norm"],
            "lr": metrics["lr"],
            "logit_norm": logit_norm,
            "diffusion_objective": config.diffusion_objective,
            "diffusion_loss_weighting": config.diffusion_loss_weighting,
            "moe_mean_load_entropy": moe.mean_load_entropy,
            "moe_timestep_tracked_selections": moe.timestep_tracked_selections,
            "moe_timestep_load_entropy": list(moe.timestep_load_entropy),
            "moe_noisy_step_load_entropy": moe.noisy_step_load_entropy,
            "moe_polish_step_load_entropy": moe.polish_step_load_entropy,
            "moe_noisy_timestep_tracked_selections": moe.noisy_timestep_tracked_selections,
            "moe_polish_timestep_tracked_selections": moe.polish_timestep_tracked_selections,
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
                data_iterator=train_batch_iter,
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
                data_iterator=train_batch_iter,
                grad_scaler=grad_scaler,
            )
    close_tensorboard_writer(tb_writer)


if __name__ == "__main__":
    main()
