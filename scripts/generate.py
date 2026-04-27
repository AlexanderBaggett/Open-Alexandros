from __future__ import annotations

from _deps import require_runtime_dependencies

require_runtime_dependencies()

import torch

from _common import (
    add_runtime_args,
    load_config,
    make_arg_parser,
    prepare_model,
    setup_torch_runtime,
)
from alexandros import AlexandrosForDiffusionLM, GenerationMode
from alexandros.inference import GenerationRequest, generate_from_request


def parse_ids(text: str) -> list[int]:
    ids = [int(part.strip()) for part in text.split(",") if part.strip()]
    return ids


def parse_optional_ids(text: str | None) -> list[int] | None:
    if text is None or not text.strip():
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_optional_sequences(text: str | None) -> list[list[int]] | None:
    if text is None or not text.strip():
        return None
    sequences = []
    for raw_sequence in text.split(";"):
        tokens = [int(part.strip()) for part in raw_sequence.split(",") if part.strip()]
        if tokens:
            sequences.append(tokens)
    return sequences or None


def main() -> None:
    parser = make_arg_parser("Generate token IDs from an Alexandros checkpoint config.")
    parser.add_argument("--config", default="configs/heavy_tiny.yaml")
    parser.add_argument("--prompt-ids", default="1,2,3")
    parser.add_argument("--mode", default=GenerationMode.AUTOREGRESSIVE.value)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--confidence-schedule",
        choices=("median", "linear", "all"),
        default="median",
    )
    parser.add_argument("--remask-low-confidence", action="store_true")
    parser.add_argument("--latent-steps", type=int, default=None)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop-token-ids", default=None)
    parser.add_argument("--stop-sequences", default=None)
    add_runtime_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    device, dtype = setup_torch_runtime(args, torch)
    model = prepare_model(AlexandrosForDiffusionLM(config), device, dtype, torch).eval()
    try:
        request = GenerationRequest(
            input_ids=[parse_ids(args.prompt_ids)],
            max_new_tokens=args.max_new_tokens,
            mode=args.mode,
            use_cache=args.use_cache,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_token_ids=parse_optional_ids(args.stop_token_ids),
            stop_sequences=parse_optional_sequences(args.stop_sequences),
            steps=args.steps,
            block_size=args.block_size,
            confidence_schedule=args.confidence_schedule,
            remask_low_confidence=args.remask_low_confidence,
            latent_steps=args.latent_steps,
        )
    except ValueError as exc:
        parser.error(str(exc))
    with torch.no_grad():
        output = generate_from_request(model, request, device=device)
    print(output.tolist())


if __name__ == "__main__":
    main()
