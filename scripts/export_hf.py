from __future__ import annotations

from _deps import require_runtime_dependencies

require_runtime_dependencies()

from _common import load_config, make_arg_parser
from alexandros import AlexandrosForDiffusionLM
from alexandros.bitlinear import save_packed_bitlinear_state


def main() -> None:
    parser = make_arg_parser("Export an Alexandros checkpoint in the reference format.")
    parser.add_argument("--config", default="configs/heavy_tiny.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--safe-serialization", action="store_true")
    parser.add_argument(
        "--export-packed-bitnet",
        action="store_true",
        help="also write packed_bitlinear.pt for Lite BitLinear layers",
    )
    args = parser.parse_args()

    model = AlexandrosForDiffusionLM(load_config(args.config))
    model.save_pretrained(args.out, safe_serialization=args.safe_serialization)
    if args.export_packed_bitnet:
        save_packed_bitlinear_state(model, f"{args.out}/packed_bitlinear.pt")
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()
