"""Recoverable PyTorch checkpoint IO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(
    path: str | Path, model: nn.Module, device: torch.device, *,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if expected_fingerprint is not None:
        actual = payload.get("metadata", {}).get("experiment_fingerprint")
        if actual != expected_fingerprint:
            raise ValueError("Checkpoint fingerprint does not match this experiment; use a new output directory.")
    model.load_state_dict(payload["model_state"])
    return payload


# Design note: checkpoint selection is driven by validation Macro-F1 and freezes
# the exact parameters subsequently evaluated once on the held-out test runs.
