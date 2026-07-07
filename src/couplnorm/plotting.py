"""Matplotlib helpers for the demo notebooks.

Thin convenience wrappers only; matplotlib is an optional dependency (install
with ``pip install couplnorm[dev]``). Each function accepts an optional ``ax``
so plots compose into subplot grids.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from couplnorm.coupling import coupling_from_samples

__all__ = [
    "plot_coupling_bars",
    "plot_covariance_heatmap",
    "plot_running_trajectory",
    "plot_training",
]


def _require_mpl():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(
            "Plotting requires matplotlib. Install with: pip install couplnorm[dev]"
        ) from exc
    return plt


def plot_coupling_bars(
    named_samples: dict[str, Tensor],
    ax=None,
    real_fft: bool = True,
    title: str = "Coupling C across distributions",
):
    """Bar chart of C for each named ``(B, N)`` sample tensor."""
    plt = _require_mpl()
    names, values = [], []
    for name, phi in named_samples.items():
        names.append(name)
        values.append(coupling_from_samples(phi, real_fft=real_fft).item())
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(names, values, color="#3b6ea5")
    ax.set_ylabel("C")
    ax.set_ylim(0, max(values + [0.1]) * 1.15)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    return ax


def plot_covariance_heatmap(cov: Tensor, ax=None, title: str = "Spectral-energy covariance"):
    """Heatmap of a spectral-energy covariance matrix."""
    plt = _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cov.detach().cpu().numpy(), cmap="magma")
    ax.set_xlabel("mode k'")
    ax.set_ylabel("mode k")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_running_trajectory(
    trajectory: Sequence[float],
    batch_C: Optional[float] = None,
    ax=None,
    title: str = "Running C converges to batch C",
):
    """Line plot of a running-C trajectory with an optional batch-C reference."""
    plt = _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(list(trajectory), label="running C")
    if batch_C is not None:
        ax.axhline(batch_C, color="red", linestyle="--", label=f"batch C = {batch_C:.3f}")
    ax.set_xlabel("chunk index")
    ax.set_ylabel("running C")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_training(
    losses: Sequence[float],
    c_steps: Optional[Sequence[int]] = None,
    c_values: Optional[Sequence[float]] = None,
    target_C: Optional[float] = None,
):
    """Two-panel training summary: loss curve and generator-C trajectory."""
    plt = _require_mpl()
    ncols = 2 if c_values is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4))
    ax0 = axes[0] if ncols == 2 else axes
    ax0.plot(list(losses))
    ax0.set_yscale("log")
    ax0.set_xlabel("step")
    ax0.set_ylabel("loss")
    ax0.set_title("Training loss")
    if ncols == 2:
        ax1 = axes[1]
        ax1.plot(list(c_steps), list(c_values), marker="o", label="generator C")
        if target_C is not None:
            ax1.axhline(target_C, color="red", linestyle="--", label=f"target C = {target_C:.3f}")
        ax1.set_xlabel("step")
        ax1.set_ylabel("C")
        ax1.set_title("Generator C trajectory")
        ax1.legend()
    fig.tight_layout()
    return fig
