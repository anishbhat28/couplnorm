"""Masked Autoregressive Flow (MAF) baseline.

A compact MADE-based MAF: a stack of autoregressive Gaussian transforms with a
fixed permutation between blocks and a standard-normal base density. Unlike the
Gaussian and Fourier baselines, a MAF can represent higher-order structure, so
it is the model you would actually train to drive C toward a target.

Kept intentionally small (this is plumbing, not the contribution). Trained by
maximum likelihood; sampled by inverting each block sequentially.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["MAF"]


class _MaskedLinear(nn.Linear):
    """Linear layer with a fixed binary mask on the weight matrix."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: Tensor) -> None:
        self.mask.copy_(mask)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return F.linear(x, self.weight * self.mask, self.bias)


class _MADE(nn.Module):
    """Masked autoencoder producing (mu, alpha) with the autoregressive property.

    Output dimension ``d`` depends only on inputs ``0..d-1`` (in the current
    variable order), so ``mu_d`` and ``alpha_d`` are functions of ``x_{<d}``.
    """

    def __init__(self, D: int, hidden: int = 64, n_hidden: int = 1):
        super().__init__()
        self.D = D
        sizes = [D] + [hidden] * n_hidden + [2 * D]
        self.net = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.net.append(_MaskedLinear(sizes[i], sizes[i + 1]))

        # Degree assignment (deterministic): inputs 0..D-1, hidden units cycle
        # through 0..D-2, outputs 0..D-1.
        m_in = torch.arange(D)
        degrees = [m_in]
        for _ in range(n_hidden):
            degrees.append(torch.arange(hidden) % max(D - 1, 1))
        m_out = torch.arange(D)

        # Hidden connectivity: non-strict >=.
        for i, layer in enumerate(self.net[:-1]):
            prev, cur = degrees[i], degrees[i + 1]
            mask = (cur.unsqueeze(1) >= prev.unsqueeze(0)).float()
            layer.set_mask(mask)
        # Output connectivity: strict > (no self/future leakage), duplicated for
        # the mu and alpha halves.
        out_mask = (m_out.unsqueeze(1) > degrees[-1].unsqueeze(0)).float()
        self.net[-1].set_mask(torch.cat([out_mask, out_mask], dim=0))

    def forward(self, x: Tensor):
        h = x
        for layer in self.net[:-1]:
            h = F.relu(layer(h))
        out = self.net[-1](h)
        mu, alpha = out.chunk(2, dim=-1)
        alpha = torch.tanh(alpha)  # keep the log-scale bounded for stability
        return mu, alpha


class MAF(nn.Module):
    """Masked autoregressive flow over ``N``-dimensional position-space fields.

    Parameters
    ----------
    N : int
        Field dimension.
    hidden : int
        Hidden width of each MADE.
    n_blocks : int
        Number of autoregressive blocks (with permutations between them).
    n_hidden : int
        Hidden layers per MADE.
    """

    def __init__(self, N: int, hidden: int = 64, n_blocks: int = 4, n_hidden: int = 1):
        super().__init__()
        self.N = int(N)
        self.blocks = nn.ModuleList(
            [_MADE(N, hidden, n_hidden) for _ in range(n_blocks)]
        )
        # Fixed permutations (reverse ordering is a standard, cheap choice that
        # guarantees every variable eventually conditions on every other).
        for i in range(n_blocks):
            perm = torch.arange(N - 1, -1, -1) if i % 2 == 0 else torch.arange(N)
            self.register_buffer(f"perm_{i}", perm)
        self.register_buffer("data_mean", torch.zeros(N))
        self.register_buffer("data_std", torch.ones(N))

    def _perm(self, i: int) -> Tensor:
        return getattr(self, f"perm_{i}")

    def log_prob(self, x: Tensor) -> Tensor:
        """Exact log-density of the flow (up to the constant standardization term)."""
        y = (x - self.data_mean) / self.data_std
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for i, made in enumerate(self.blocks):
            y = y[:, self._perm(i)]
            mu, alpha = made(y)
            y = (y - mu) * torch.exp(-alpha)
            log_det = log_det - alpha.sum(dim=-1)
        base = -0.5 * (y.pow(2) + math.log(2 * math.pi)).sum(dim=-1)
        return base + log_det

    def fit(
        self,
        data: Tensor,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 256,
        verbose: bool = False,
    ) -> "MAF":
        if data.dim() != 2 or data.shape[1] != self.N:
            raise ValueError(
                f"Expected data of shape (B, {self.N}); got {tuple(data.shape)}"
            )
        data = data.to(self.data_mean)
        with torch.no_grad():
            self.data_mean = data.mean(dim=0)
            self.data_std = data.std(dim=0).clamp_min(1e-6)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        n = data.shape[0]
        for epoch in range(epochs):
            perm = torch.randperm(n, device=data.device)
            total = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                loss = -self.log_prob(data[idx]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * idx.numel()
            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print(f"[MAF] epoch {epoch:4d}  nll={total / n:.4f}")
        return self

    @torch.no_grad()
    def sample(self, n: int) -> Tensor:
        """Draw ``n`` samples by inverting each block sequentially."""
        y = torch.randn(n, self.N, device=self.data_mean.device, dtype=self.data_mean.dtype)
        for i in reversed(range(len(self.blocks))):
            made = self.blocks[i]
            t = torch.zeros_like(y)
            for d in range(self.N):
                mu, alpha = made(t)
                t[:, d] = y[:, d] * torch.exp(alpha[:, d]) + mu[:, d]
            inv = torch.argsort(self._perm(i))
            y = t[:, inv]
        return y * self.data_std + self.data_mean

    def extra_repr(self) -> str:
        return f"N={self.N}, n_blocks={len(self.blocks)}"
