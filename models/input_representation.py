"""Small causal temporal encoder shared by all measured/manipulated variables."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SharedVariableTCN(nn.Module):
    def __init__(self, channels: int = 32, kernel_size: int = 3,
                 dilations: Sequence[int] = (1, 2)) -> None:
        super().__init__()
        if channels <= 0 or kernel_size <= 0 or not dilations or any(int(d) <= 0 for d in dilations):
            raise ValueError("Temporal dimensions and dilations must be positive.")
        self.output_dim = int(channels)
        self.left_padding = [(int(kernel_size) - 1) * int(d) for d in dilations]
        self.convolutions = nn.ModuleList([
            nn.Conv1d(3 if position == 0 else channels, channels, kernel_size,
                      dilation=int(dilation), padding=0)
            for position, dilation in enumerate(dilations)
        ])

    @staticmethod
    def inputs(window: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if window.ndim != 2 or mask.shape != window.shape:
            raise ValueError("Temporal encoder expects [nodes, time] windows and masks.")
        valid = mask.bool() & torch.isfinite(window)
        clean = torch.where(valid, window, torch.zeros_like(window))
        difference = torch.zeros_like(clean)
        difference[:, 1:] = torch.where(
            valid[:, 1:] & valid[:, :-1], clean[:, 1:] - clean[:, :-1],
            torch.zeros_like(clean[:, 1:]),
        )
        return torch.stack((clean, valid.to(clean.dtype), difference), dim=1), valid

    def forward(self, window: Tensor, mask: Tensor) -> Tensor:
        values, valid = self.inputs(window, mask)
        for convolution, padding in zip(self.convolutions, self.left_padding):
            values = F.relu(convolution(F.pad(values, (padding, 0))))
        # Summarize the whole observed window; every convolution is left-causal.
        weights = valid[:, None, :].to(values.dtype)
        return (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1)
