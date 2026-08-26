"""Configuration, device, and random seed utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the project's flat YAML configuration without an extra package."""
    config: dict[str, Any] = {}
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"invalid config line {line_number}: {raw_line}")
        key, raw_value = (part.strip() for part in line.split(":", 1))
        lowered = raw_value.lower()
        if lowered in {"true", "false"}:
            value: Any = lowered == "true"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value.strip("\"'")
        config[key] = value
    if not config:
        raise ValueError("configuration is empty")
    return config


def seed_everything(seed: int) -> None:
    """Seed the random number generators used by the project."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    """Prefer CUDA or Apple Metal when available, otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
