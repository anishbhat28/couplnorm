"""Low-rank Gaussian (PCA) baseline."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["PCAModel"]


class PCAModel(nn.Module):
    """Low-rank Gaussian: mean + top-``k`` principal components + isotropic noise.

    Fits a rank-``k`` approximation of the data covariance and adds isotropic
    residual noise equal to the mean of the discarded eigenvalues, so the model
    is a proper full-rank Gaussian that under-represents off-principal
    directions. Weaker than :class:`FullGaussian`; useful for showing that
    truncating structure lowers coupling fidelity.

    Parameters
    ----------
    N : int
        Field dimension.
    n_components : int
        Number of principal components to retain.
    """

    def __init__(self, N: int, n_components: int = 4):
        super().__init__()
        if n_components < 1 or n_components > N:
            raise ValueError(f"n_components must be in [1, {N}]; got {n_components}")
        self.N = int(N)
        self.k = int(n_components)
        self.register_buffer("mean", torch.zeros(N))
        self.register_buffer("components", torch.zeros(N, self.k))  # scaled by sqrt(eigval)
        self.register_buffer("noise_std", torch.zeros(()))
        self._fitted = False

    @torch.no_grad()
    def fit(self, data: Tensor) -> "PCAModel":
        if data.dim() != 2 or data.shape[1] != self.N:
            raise ValueError(
                f"Expected data of shape (B, {self.N}); got {tuple(data.shape)}"
            )
        data = data.to(self.mean)
        self.mean = data.mean(dim=0)
        centered = data - self.mean
        cov = centered.T @ centered / (data.shape[0] - 1)
        # eigh returns ascending eigenvalues; take the top k.
        evals, evecs = torch.linalg.eigh(cov)
        evals = evals.clamp_min(0.0)
        top_vals = evals[-self.k:]
        top_vecs = evecs[:, -self.k:]
        self.components = top_vecs * top_vals.sqrt().unsqueeze(0)
        residual = evals[: self.N - self.k]
        self.noise_std = residual.mean().clamp_min(0.0).sqrt() if residual.numel() else torch.zeros(())
        self._fitted = True
        return self

    @torch.no_grad()
    def sample(self, n: int) -> Tensor:
        if not self._fitted:
            raise RuntimeError("Call fit(data) before sample(n).")
        z = torch.randn(n, self.k, device=self.mean.device, dtype=self.mean.dtype)
        x = self.mean + z @ self.components.T
        if self.noise_std > 0:
            x = x + self.noise_std * torch.randn_like(x)
        return x

    def extra_repr(self) -> str:
        return f"N={self.N}, n_components={self.k}, fitted={self._fitted}"
