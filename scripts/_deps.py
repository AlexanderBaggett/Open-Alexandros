from __future__ import annotations

import importlib.util
import sys


def require_runtime_dependencies() -> None:
    missing = [
        package for package in ("torch",) if importlib.util.find_spec(package) is None
    ]
    if not missing:
        return
    names = ", ".join(missing)
    print(
        f"Missing runtime dependencies: {names}. Install PyTorch for your "
        "hardware first, for example CPU-only with "
        "`pip install --index-url https://download.pytorch.org/whl/cpu torch`, "
        "then install this project with `pip install -e .[dev]`.",
        file=sys.stderr,
    )
    raise SystemExit(1)
