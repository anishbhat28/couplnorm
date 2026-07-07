"""Full multivariate Gaussian baseline."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["FullGaussian"]


class FullGaussian(nn.Module):
    """Position-space multivariate Gaussian fit to the data covariance.

    Matches the mean and full covariance of the training fields exactly (up to
    finite-sample error), then samples via the Cholesky factor. Being Gaussian,
    it reproduces all second-order structure but no genuine 4th-order coupling:
    its spectral energies inherit only the coupling implied by the covariance.

    Parameters
    ----------
    N : int
        Number of lattice sites (field dimension).
    jitter : float
        Diagonal loading added before the Cholesky factorization for numerical
        stability with near-singular covariances.
    """

    def __init__(self, N: int, jitter: float = 1e-5):
        super().__init__()
        self.N = int(N)
        self.jitter = float(jitter)
        self.register_buffer("mean", torch.zeros(N))
        self.register_buffer("cov", torch.eye(N))
        self.register_buffer("chol", torch.eye(N))
        self._fitted = False

    @torch.no_grad()
    def fit(self, data: Tensor) -> "FullGaussian":
        if data.dim() != 2 or data.shape[1] != self.N:
            raise ValueError(
                f"Expected data of shape (B, {self.N}); got {tuple(data.shape)}"
            )
        data = data.to(self.mean)
        self.mean = data.mean(dim=0)
        centered = data - self.mean
        cov = centered.T @ centered / (data.shape[0] - 1)
        cov = cov + self.jitter * torch.eye(self.N, device=cov.device, dtype=cov.dtype)
        self.cov = cov
        self.chol = torch.linalg.cholesky(cov)
        self._fitted = True
        return self

    @torch.no_grad()
    def sample(self, n: int) -> Tensor:
        if not self._fitted:
            raise RuntimeError("Call fit(data) before sample(n).")
        z = torch.randn(n, self.N, device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + z @ self.chol.T

    def extra_repr(self) -> str:
        return f"N={self.N}, fitted={self._fitted}"
