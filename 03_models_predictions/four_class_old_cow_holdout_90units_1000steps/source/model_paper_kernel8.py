# -*- coding: utf-8 -*-
"""Independent kernel-8 CCNN used by the R5 GPU full-budget rerun.

This is a *paper-inspired native-rate* variant, not a strict reproduction of
Arablouei et al. (2024): the local datasets retain their native 25 Hz/128 and
10 Hz/51 windows and the task has three classes.

Every convolution has four output filters, kernel size eight, and no bias.
Only the two branch-entry CCNNs are linearly constrained, matching the paper's
description.  They use DCT initialisation and must be projected after every
optimizer step through :meth:`PaperKernel8CCNN.project_constraints`.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn as nn

__all__ = [
    "dct_basis",
    "ZeroSumConv1d",
    "PaperKernel8CCNN",
]


def dct_basis(n_filters: int = 4, kernel_size: int = 8) -> torch.Tensor:
    """Return unit-norm DCT-II rows 1..n_filters (the DC row is omitted)."""
    if n_filters < 1 or n_filters >= kernel_size:
        raise ValueError("n_filters must be in [1, kernel_size - 1]")
    n = torch.arange(kernel_size, dtype=torch.float64)
    rows = []
    for order in range(1, n_filters + 1):
        row = torch.cos(math.pi * (n + 0.5) * order / kernel_size)
        rows.append(row / row.norm())
    return torch.stack(rows).to(torch.float32)


class ZeroSumConv1d(nn.Conv1d):
    """DCT-initialised Conv1d with a zero-sum constraint per output filter."""

    def __init__(self, in_channels: int) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=4,
            kernel_size=8,
            bias=False,
        )
        self.reset_dct()

    @torch.no_grad()
    def reset_dct(self) -> None:
        basis = dct_basis(4, 8)
        weights = basis[:, None, :].repeat(1, self.in_channels, 1)
        weights = weights / math.sqrt(self.in_channels)
        self.weight.copy_(weights)
        self.project()

    @torch.no_grad()
    def project(self) -> None:
        """Subtract each output filter's own mean across channels and time."""
        self.weight.sub_(self.weight.mean(dim=(1, 2), keepdim=True))

    @torch.no_grad()
    def max_abs_filter_sum(self) -> float:
        sums = self.weight.sum(dim=(1, 2)).abs()
        return float(sums.max().item())


class PaperKernel8CCNN(nn.Module):
    """Two-branch, three-class CCNN with length-adaptive aggregation."""

    variant_name = "paper-inspired native-rate CCNN full-budget rerun"
    architecture_name = "paper_kernel8_native_rate"
    n_classes = 3
    n_filters = 4
    n_features = 8
    hidden_width = 6
    min_length = 29  # 8 + 3 * (8 - 1)

    def __init__(self, in_channels: int = 3, n_classes: int = 3) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("R5 contract requires exactly 3 acceleration channels")
        if n_classes != 3:
            raise ValueError("R5 contract requires exactly 3 classes")

        self.in_channels = in_channels

        # Linear branch: constrained CCNN -> temporal median.
        self.ccnn_linear = ZeroSumConv1d(in_channels)

        # Non-linear branch: constrained CCNN -> 3 x (Conv + tanh).
        self.ccnn_nonlinear = ZeroSumConv1d(in_channels)
        self.nonlinear_convs = nn.ModuleList(
            [
                nn.Conv1d(4, 4, kernel_size=8, bias=False),
                nn.Conv1d(4, 4, kernel_size=8, bias=False),
                nn.Conv1d(4, 4, kernel_size=8, bias=False),
            ]
        )

        self.feature_bn = nn.BatchNorm1d(self.n_features)
        self.classifier = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_width),
            nn.ReLU(),
            nn.Linear(self.hidden_width, self.n_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 3:
            raise ValueError(f"expected (B, 3, L), got {tuple(x.shape)}")
        if x.shape[-1] < self.min_length:
            raise ValueError(
                f"window length {x.shape[-1]} is below minimum {self.min_length}"
            )

        # torch.median(dim=...) returns indices and its CUDA backward is not
        # accepted by PyTorch's deterministic-algorithm gate.  Sorting and
        # selecting the lower middle element is mathematically identical to
        # torch.median for both odd and even lengths, while retaining an exact
        # temporal-median aggregation and a deterministic CUDA backward path.
        linear_sequence = self.ccnn_linear(x)
        middle = (linear_sequence.shape[-1] - 1) // 2
        linear = linear_sequence.sort(dim=-1).values[..., middle]

        nonlinear = self.ccnn_nonlinear(x)
        for conv in self.nonlinear_convs:
            nonlinear = torch.tanh(conv(nonlinear))
        nonlinear = nonlinear.abs().mean(dim=-1)

        return torch.cat((linear, nonlinear), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.classifier(self.feature_bn(features))

    def constrained_layers(self) -> Iterator[ZeroSumConv1d]:
        for module in self.modules():
            if isinstance(module, ZeroSumConv1d):
                yield module

    @torch.no_grad()
    def project_constraints(self) -> None:
        for layer in self.constrained_layers():
            layer.project()

    @torch.no_grad()
    def constraint_max_abs_filter_sum(self) -> float:
        return max(layer.max_abs_filter_sum() for layer in self.constrained_layers())

    def architecture_manifest(self) -> dict:
        convolutions = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                convolutions.append(
                    {
                        "name": name,
                        "in_channels": int(module.in_channels),
                        "out_channels": int(module.out_channels),
                        "kernel_size": int(module.kernel_size[0]),
                        "bias": module.bias is not None,
                        "zero_sum_constrained": isinstance(module, ZeroSumConv1d),
                    }
                )
        return {
            "architecture_name": self.architecture_name,
            "input_channels": self.in_channels,
            "n_classes": self.n_classes,
            "n_features": self.n_features,
            "hidden_width": self.hidden_width,
            "min_input_length": self.min_length,
            "linear_aggregation": "temporal_median",
            "nonlinear_aggregation": "temporal_mean_absolute",
            "convolutions": convolutions,
            "constrained_convolution_names": [
                name
                for name, module in self.named_modules()
                if isinstance(module, ZeroSumConv1d)
            ],
        }
