"""Train-only fixed scaling over unique (run, time) observations.

Fitting deliberately bypasses __getitem__: synthetic corruption is applied in
physical units later. Overlapping sampled windows must not count a time point
more than once. Validation/test datasets are rejected by the fitting API.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
import math
from typing import Any, Iterator, Mapping

import torch
from torch import Tensor
from torch.utils.data import Subset

from .tep_dataset import TEPWindowDataset


def _selection(dataset: Any) -> tuple[TEPWindowDataset, list[int]]:
    if isinstance(dataset, Subset):
        base, indices = dataset.dataset, [int(i) for i in dataset.indices]
    else:
        base, indices = dataset, list(range(len(dataset)))
    if not isinstance(base, TEPWindowDataset):
        raise TypeError("Expected a TEPWindowDataset or one direct Subset of it.")
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("Selected windows must be non-empty and unique.")
    if min(indices) < 0 or max(indices) >= len(base):
        raise IndexError("Selected window index outside the dataset.")
    return base, indices


def selection_manifest(dataset: Any) -> dict[str, Any]:
    """Describe selected windows without loading observations or fault labels."""
    base, indices = _selection(dataset)
    windows = []
    for index in indices:
        run_index = bisect_right(base._cumulative, index)
        preceding = base._cumulative[run_index - 1] if run_index else 0
        start = (index - preceding) * base.stride
        windows.append({
            "run_uid": base.runs[run_index].run_uid,
            "start": start,
            "stop": start + base.window_length,
        })
    payload = {
        "split": base.split, "window_length": base.window_length,
        "stride": base.stride, "windows": windows,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {**payload, "sha256": digest, "num_windows": len(windows)}


def _selected_observations(dataset: Any) -> Iterator[tuple[str, Tensor, Tensor]]:
    base, indices = _selection(dataset)
    intervals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for index in indices:
        run_index = bisect_right(base._cumulative, index)
        preceding = base._cumulative[run_index - 1] if run_index else 0
        start = (index - preceding) * base.stride
        intervals[run_index].append((start, start + base.window_length))
    for run_index in sorted(intervals):
        run = base.runs[run_index]
        selected = torch.zeros(run.length, dtype=torch.bool)
        for start, stop in intervals[run_index]:
            selected[start:stop] = True
        times = selected.nonzero(as_tuple=False).flatten()
        values, _ = base._run_tensors(run_index)
        yield run.run_uid, times, values[times].detach().cpu()


def observation_fingerprint(dataset: Any) -> str:
    """Hash only the selected clean observations; no learned statistics."""
    digest = hashlib.sha256()
    for run_uid, times, values in _selected_observations(dataset):
        digest.update(run_uid.encode("utf-8") + b"\0")
        digest.update(times.numpy().astype("<i8", copy=False).tobytes())
        mask = torch.isfinite(values)
        digest.update(mask.numpy().tobytes())
        safe = torch.where(mask, values, torch.zeros_like(values))
        digest.update(safe.numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


class FixedVariableStandardizer:
    """CPU dataset transform with a portable JSON/checkpoint representation."""

    def __init__(
        self, mean: Tensor, scale: Tensor, count: Tensor, epsilon: float,
        selection_sha256: str,
    ) -> None:
        self.mean = torch.as_tensor(mean, dtype=torch.float64).detach().cpu().clone()
        self.scale = torch.as_tensor(scale, dtype=torch.float64).detach().cpu().clone()
        self.count = torch.as_tensor(count, dtype=torch.long).detach().cpu().clone()
        self.epsilon = float(epsilon)
        self.selection_sha256 = str(selection_sha256)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Scaling epsilon must be finite and positive.")
        if self.mean.shape != (52,) or self.scale.shape != (52,) or self.count.shape != (52,):
            raise ValueError("Fixed scaling requires exactly 52 variables.")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all():
            raise ValueError("Scaler parameters must be finite.")
        if (self.scale < self.epsilon).any() or (self.count <= 0).any():
            raise ValueError("Each variable needs valid training observations and a positive scale.")

    @classmethod
    def fit(cls, dataset: Any, epsilon: float = 1e-6) -> "FixedVariableStandardizer":
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
            raise ValueError("Scaling epsilon must be finite and positive.")
        base, _ = _selection(dataset)
        if base.split != "train" or any(run.source_split != "train" for run in base.runs):
            raise ValueError("Scaler fitting is restricted to the training partition.")
        count = torch.zeros(52, dtype=torch.float64)
        mean = torch.zeros(52, dtype=torch.float64)
        m2 = torch.zeros(52, dtype=torch.float64)
        for _, _, raw in _selected_observations(dataset):
            x = raw.to(torch.float64)
            valid = torch.isfinite(x)
            n = valid.sum(dim=0).to(torch.float64)
            safe = torch.where(valid, x, torch.zeros_like(x))
            batch_mean = safe.sum(dim=0) / n.clamp_min(1)
            centered = torch.where(valid, x - batch_mean, torch.zeros_like(x))
            batch_m2 = centered.square().sum(dim=0)
            total = count + n
            delta = batch_mean - mean
            m2 = m2 + batch_m2 + delta.square() * count * n / total.clamp_min(1)
            mean = mean + delta * n / total.clamp_min(1)
            count = total
        std = (m2 / count.clamp_min(1)).clamp_min(0).sqrt()
        return cls(mean, std.clamp_min(float(epsilon)), count.long(), epsilon,
                   selection_manifest(dataset)["sha256"])

    def transform(self, window: Tensor, mask: Tensor) -> Tensor:
        if window.ndim != 2 or window.shape[0] != 52 or mask.shape != window.shape:
            raise ValueError("Scaling expects matching [52, time] observations and mask.")
        valid = mask.bool() & torch.isfinite(window)
        mean = self.mean.to(device=window.device, dtype=window.dtype)[:, None]
        scale = self.scale.to(device=window.device, dtype=window.dtype)[:, None]
        # Fill before arithmetic so NaN/Inf never enters the transformed branch.
        values = (torch.where(valid, window, mean) - mean) / scale
        return torch.where(valid, values, torch.zeros_like(values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "fixed_variable_scaler.v1", "fit_split": "train",
            "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "count": self.count.tolist(), "epsilon": self.epsilon,
            "selection_sha256": self.selection_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FixedVariableStandardizer":
        if payload.get("schema_version") != "fixed_variable_scaler.v1" or payload.get("fit_split") != "train":
            raise ValueError("Unsupported scaler format or fitting partition.")
        return cls(payload["mean"], payload["scale"], payload["count"],
                   float(payload["epsilon"]), str(payload["selection_sha256"]))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()
