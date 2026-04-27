from __future__ import annotations

import math
from pathlib import Path

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
    finish_optimization_step,
    load_config,
    load_training_state,
    make_arg_parser,
    make_tensorboard_writer,
    make_token_batch_iterator,
    next_token_batch,
    prepare_output_dir,
    prepare_training_model,
    resolve_training_hparams,
    run_metadata,
    save_training_state,
    setup_torch_runtime,
    validate_training_args,
    write_jsonl,
    write_tensorboard_scalars,
)
from alexandros import AlexandrosForCausalLM
from alexandros.training import (
    apply_trainability,
    objective_log_fields,
    phase_checkpoint_metadata,
    trainable_parameters,
)
from alexandros.ttt import TTTMetaAdapter, ttt_next_token_loss_from_logits


def next_token_loss_value(model: AlexandrosForCausalLM, hidden, input_ids) -> float:
    logits = model.lm_head(hidden)
    loss = ttt_next_token_loss_from_logits(logits, input_ids)
    return float(loss.detach().cpu().item())


def validate_ttt_args(args) -> None:
    chunk_len = int(getattr(args, "prefill_chunk_len", 0))
    if chunk_len < 0:
        raise ValueError("prefill_chunk_len must be >= 0")
    if chunk_len == 1:
        raise ValueError("prefill_chunk_len must be 0 or >= 2")
    inner_lr = getattr(args, "inner_lr", None)
    if inner_lr is not None and (not math.isfinite(float(inner_lr)) or inner_lr <= 0):
        raise ValueError("inner_lr must be finite and > 0")
    if not math.isfinite(float(args.ttt_init_std)) or args.ttt_init_std < 0.0:
        raise ValueError("ttt_init_std must be finite and >= 0")
    if not math.isfinite(float(args.ttt_adapter_scale)):
        raise ValueError("ttt_adapter_scale must be finite")


def effective_chunk_len(seq_len: int, requested_chunk_len: int) -> int:
    if requested_chunk_len > 0:
        return requested_chunk_len
    return max(2, seq_len // 2)


def iter_prefill_chunks(hidden, input_ids, chunk_len: int):
    seq_len = hidden.size(1)
    for start in range(0, seq_len, chunk_len):
        end = min(start + chunk_len, seq_len)
        if end - start < 2:
            continue
        yield hidden[:, start:end, :], input_ids[:, start:end]


def split_meta_chunks(hidden, input_ids, chunk_len: int):
    chunks = list(iter_prefill_chunks(hidden, input_ids, chunk_len))
    if len(chunks) < 2:
        raise ValueError(
            "TTT meta-training requires at least two chunks with length >= 2; "
            "increase seq_len or reduce prefill_chunk_len"
        )
    return chunks[:-1], chunks[-1]


def ttt_outer_loss(model, ttt_meta, hidden, input_ids, fast_a, fast_b):
    adapted = ttt_meta.apply(hidden, fast_a, fast_b)
    logits = model.lm_head(adapted)
    return ttt_next_token_loss_from_logits(logits, input_ids), adapted


def count_trainable(module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def save_ttt_meta_checkpoint(
    path: Path,
    *,
    ttt_meta: TTTMetaAdapter,
    optimizer,
    config,
    args,
    torch,
    step: int,
    trainability,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "open-alexandros-ttt-meta-adapter",
        "format_version": 1,
        "step": step,
        "config": config.to_dict(),
        "config_hash": run_metadata(args, config, torch)["config_hash"],
        "ttt_meta_adapter_state_dict": ttt_meta.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "trainability": trainability.to_dict(),
        "fast_weight_parameters": {
            "init_a": list(ttt_meta.init_a.shape),
            "init_b": list(ttt_meta.init_b.shape),
            "rank": ttt_meta.rank,
            "hidden_size": ttt_meta.hidden_size,
            "adapter_scale": ttt_meta.adapter_scale,
        },
        "request_local_state_saved": False,
        "metadata": run_metadata(args, config, torch),
    }
    torch.save(payload, path)


def main() -> None:
    parser = make_arg_parser("Run TTT-E2E-style meta-training for fast weights.")
    parser.add_argument("--config", default="configs/heavy_tiny.yaml")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument(
        "--prefill-chunk-len",
        type=int,
        default=0,
        help=(
            "Chunk length for TTT inner updates; 0 splits each sequence roughly "
            "in half for prefix/heldout meta-training."
        ),
    )
    parser.add_argument(
        "--inner-lr",
        type=float,
        default=None,
        help="Inner-loop fast-weight learning rate; defaults to --lr.",
    )
    parser.add_argument(
        "--ttt-init-std",
        type=float,
        default=1e-3,
        help="Normal initialization stddev for learned TTT phi_0 fast weights.",
    )
    parser.add_argument(
        "--ttt-adapter-scale",
        type=float,
        default=1.0,
        help="Residual scale applied by the learned TTT adapter.",
    )
    add_runtime_args(parser)
    add_training_args(parser)
    add_trainability_args(parser)
    args = parser.parse_args()
    validate_training_args(args)
    validate_ttt_args(args)
    if args.seq_len < 4:
        raise ValueError("seq_len must be >= 4 for TTT prefix/heldout chunks")

    config = load_config(args.config)
    resolve_training_hparams(args, config)
    if args.inner_lr is None:
        args.inner_lr = args.lr
    validate_ttt_args(args)
    device, dtype = setup_torch_runtime(args, torch)
    precision_context = autocast_context(device, dtype, torch)
    args.amp_enabled = dtype != torch.float32
    args.grad_scaler_enabled = False
    out_dir = prepare_output_dir(args)
    tb_writer = make_tensorboard_writer(args)
    model = prepare_training_model(AlexandrosForCausalLM(config), device, dtype, torch)
    trainability = apply_trainability(model, phase="ttt", scope=args.trainable_scope)
    model.train(trainability.trainable_parameters > 0)
    ttt_meta = TTTMetaAdapter.from_config(
        config,
        init_std=args.ttt_init_std,
        adapter_scale=args.ttt_adapter_scale,
    ).to(device=device)
    training_module = torch.nn.ModuleDict(
        {"model": model, "ttt_meta_adapter": ttt_meta}
    )
    opt_params = [*trainable_parameters(model), *ttt_meta.parameters()]
    if not opt_params:
        raise ValueError("TTT meta-training requires trainable TTT parameters")
    opt = torch.optim.AdamW(opt_params, lr=args.lr)
    opt.zero_grad(set_to_none=True)
    batch_iter = make_token_batch_iterator(
        args,
        config,
        split="train",
        device=device,
        torch=torch,
        prepend_bos=True,
    )
    start_step = 0
    if args.resume:
        start_step = load_training_state(
            args.resume,
            model=training_module,
            optimizer=opt,
            config=config,
            torch=torch,
            map_location=device,
            data_iterator=batch_iter,
        )
    metadata = run_metadata(args, config, torch)
    objective_fields = objective_log_fields("ttt")
    checkpoint_dir = out_dir / "checkpoint" if out_dir is not None else None
    ttt_checkpoint_path = (
        checkpoint_dir / "ttt_meta_adapter.pt" if checkpoint_dir is not None else None
    )
    actual_chunk_len = effective_chunk_len(
        int(args.seq_len), int(args.prefill_chunk_len)
    )
    final_step = start_step - 1

    for step in range(start_step, start_step + args.steps):
        loss_total = 0.0
        pre_update_loss_total = 0.0
        post_update_loss_total = 0.0
        prefill_chunk_loss_total = 0.0
        inner_grad_norm_total = 0.0
        hidden_norm_before_total = 0.0
        hidden_norm_after_total = 0.0
        hidden_delta_norm_total = 0.0
        ttt_steps = 0
        prefill_chunk_count = 0
        heldout_chunk_len = 0
        for _ in range(args.grad_accum_steps):
            input_ids = next_token_batch(
                batch_iter,
                config,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                device=device,
                torch=torch,
                force_bos=True,
            )
            with precision_context():
                if trainability.trainable_parameters == 0:
                    with torch.no_grad():
                        base = model.model(input_ids).last_hidden_state
                    base = base.detach()
                else:
                    base = model.model(input_ids).last_hidden_state
                prefix_chunks, heldout_chunk = split_meta_chunks(
                    base,
                    input_ids,
                    actual_chunk_len,
                )
                heldout_hidden, heldout_ids = heldout_chunk
                heldout_chunk_len = heldout_hidden.size(1)
                with torch.no_grad():
                    pre_update_loss_total += next_token_loss_value(
                        model,
                        heldout_hidden.detach(),
                        heldout_ids,
                    )
                fast_a, fast_b = ttt_meta.initial_fast_weights(
                    device=base.device,
                    dtype=base.dtype,
                )
                inner_loss_total = 0.0
                inner_grad_norm = 0.0
                for chunk_hidden, chunk_ids in prefix_chunks:
                    update = ttt_meta.inner_update(
                        chunk_hidden,
                        chunk_ids,
                        model.lm_head,
                        fast_a,
                        fast_b,
                        inner_lr=args.inner_lr,
                        create_graph=True,
                    )
                    fast_a = update.fast_a
                    fast_b = update.fast_b
                    inner_loss_total += update.loss.detach().item()
                    inner_grad_norm += update.grad_norm.detach().item()
                outer_loss, adapted_heldout = ttt_outer_loss(
                    model,
                    ttt_meta,
                    heldout_hidden,
                    heldout_ids,
                    fast_a,
                    fast_b,
                )
            backward_loss(outer_loss, args, torch)
            loss_total += outer_loss.detach().item()
            post_update_loss_total += outer_loss.detach().item()
            prefill_chunk_loss_total += inner_loss_total / len(prefix_chunks)
            inner_grad_norm_total += inner_grad_norm / len(prefix_chunks)
            with torch.no_grad():
                adapted_full = ttt_meta.apply(
                    base.detach(),
                    fast_a.detach(),
                    fast_b.detach(),
                )
                hidden_norm_before_total += base.detach().norm().item()
                hidden_norm_after_total += adapted_full.norm().item()
                hidden_delta_norm_total += (adapted_full - base.detach()).norm().item()
                if not torch.isfinite(adapted_heldout.detach()).all():
                    raise FloatingPointError("non-finite TTT adapted hidden states")
            ttt_steps = len(prefix_chunks)
            prefill_chunk_count = len(prefix_chunks)

        metrics = finish_optimization_step(
            training_module,
            opt,
            args,
            torch,
            step=step,
        )
        divisor = float(args.grad_accum_steps)
        loss_value = loss_total / divisor
        pre_update_loss = pre_update_loss_total / divisor
        post_update_loss = post_update_loss_total / divisor
        prefill_chunk_loss = prefill_chunk_loss_total / divisor
        inner_grad_norm = inner_grad_norm_total / divisor
        hidden_norm_before = hidden_norm_before_total / divisor
        hidden_norm_after = hidden_norm_after_total / divisor
        hidden_delta_norm = hidden_delta_norm_total / divisor
        print(
            f"ttt_step={step} outer_loss={loss_value:.4f} "
            f"pre_loss={pre_update_loss:.4f} post_loss={post_update_loss:.4f} "
            f"grad_norm={metrics['grad_norm']:.4f} lr={metrics['lr']:.6g}"
        )
        metrics_record = {
            **metadata,
            **objective_fields,
            **trainability.to_log_fields(),
            "phase": "ttt",
            "step": step,
            "loss": loss_value,
            "outer_loss": loss_value,
            "inner_loss": prefill_chunk_loss,
            "grad_norm": metrics["grad_norm"],
            "lr": metrics["lr"],
            "ttt_inner_lr": args.inner_lr,
            "ttt_steps": ttt_steps,
            "prefill_chunk_len": actual_chunk_len,
            "prefill_chunk_count": prefill_chunk_count,
            "prefill_chunk_loss": prefill_chunk_loss,
            "ttt_inner_grad_norm": inner_grad_norm,
            "ttt_heldout_chunk_len": heldout_chunk_len,
            "ttt_meta_trainable_parameters": count_trainable(ttt_meta),
            "ttt_checkpoint_path": ""
            if ttt_checkpoint_path is None
            else str(ttt_checkpoint_path),
            "pre_update_loss": pre_update_loss,
            "post_update_loss": post_update_loss,
            "hidden_norm_before": hidden_norm_before,
            "hidden_norm_after": hidden_norm_after,
            "hidden_delta_norm": hidden_delta_norm,
            "base_norm": hidden_norm_before,
            "adapted_norm": hidden_norm_after,
            "delta_norm": hidden_delta_norm,
            "grad_accum_steps": args.grad_accum_steps,
            "data_source": data_source_name(args),
            "amp_enabled": args.amp_enabled,
            "grad_scaler_enabled": args.grad_scaler_enabled,
        }
        write_jsonl(args.log_jsonl, metrics_record)
        write_tensorboard_scalars(tb_writer, metrics_record, step=step)
        final_step = step
        if out_dir is not None:
            save_training_state(
                out_dir / "training_state.pt",
                model=training_module,
                optimizer=opt,
                config=config,
                args=args,
                torch=torch,
                step=step,
                data_iterator=batch_iter,
            )

    if checkpoint_dir is not None:
        model.save_pretrained(
            checkpoint_dir,
            checkpoint_metadata=phase_checkpoint_metadata("ttt", trainability),
        )
        assert ttt_checkpoint_path is not None
        save_ttt_meta_checkpoint(
            ttt_checkpoint_path,
            ttt_meta=ttt_meta,
            optimizer=opt,
            config=config,
            args=args,
            torch=torch,
            step=final_step,
            trainability=trainability,
        )
        if final_step < start_step:
            save_training_state(
                out_dir / "training_state.pt",
                model=training_module,
                optimizer=opt,
                config=config,
                args=args,
                torch=torch,
                step=final_step,
                data_iterator=batch_iter,
            )
    close_tensorboard_writer(tb_writer)


if __name__ == "__main__":
    main()
