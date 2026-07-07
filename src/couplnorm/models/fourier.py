"""Independent-mode Fourier baseline (the marginal-independence model)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["FourierModel"]


class FourierModel(nn.Module):
    """Models each Fourier mode as an independent Gaussian.

    Fits a per-mode complex Gaussian to ``rfft(data)`` (independent real and
    imaginary parts, no cross-mode correlation) and samples by drawing each mode
    independently and inverting the transform. Because the modes are independent
    by construction, the spectral-energy covariance is diagonal and C is near
    zero regardless of the target distribution.

    This is the "wrong by construction" baseline: a generative model that
    assumes marginal independence of modes cannot reproduce joint 4th-order
    coupling. It is the whole reason the C diagnostic is interesting.

    Parameters
    ----------
    N : int
        Field dimension (length of the real position-space signal).
    """

    def __init__(self, N: int):
        super().__init__()
        self.N = int(N)
        self.M = N // 2 + 1
        self.register_buffer("mean_real", torch.zeros(self.M))
        self.register_buffer("mean_imag", torch.zeros(self.M))
        self.register_buffer("std_real", torch.ones(self.M))
        self.register_buffer("std_imag", torch.ones(self.M))
        self._fitted = False

    @torch.no_grad()
    def fit(self, data: Tensor) -> "FourierModel":
        if data.dim() != 2 or data.shape[1] != self.N:
            raise ValueError(
                f"Expected data of shape (B, {self.N}); got {tuple(data.shape)}"
            )
        data = data.to(self.mean_real)
        x_hat = torch.fft.rfft(data, dim=-1, norm="ortho")
        self.mean_real = x_hat.real.mean(dim=0)
        self.mean_imag = x_hat.imag.mean(dim=0)
        self.std_real = x_hat.real.std(dim=0).clamp_min(1e-8)
        self.std_imag = x_hat.imag.std(dim=0).clamp_min(1e-8)
        # The DC mode (and Nyquist mode when N is even) is purely real for a
        # real signal; force its imaginary part to zero so irfft stays real.
        self.mean_imag[0] = 0.0
        self.std_imag[0] = 0.0
        if self.N % 2 == 0:
            self.mean_imag[-1] = 0.0
            self.std_imag[-1] = 0.0
        self._fitted = True
        return self

    @torch.no_grad()
    def sample(self, n: int) -> Tensor:
        if not self._fitted:
            raise RuntimeError("Call fit(data) before sample(n).")
        dev, dt = self.mean_real.device, self.mean_real.dtype
        re = self.mean_real + self.std_real * torch.randn(n, self.M, device=dev, dtype=dt)
        im = self.mean_imag + self.std_imag * torch.randn(n, self.M, device=dev, dtype=dt)
        x_hat = torch.complex(re, im)
        return torch.fft.irfft(x_hat, n=self.N, dim=-1, norm="ortho")

    def extra_repr(self) -> str:
        return f"N={self.N}, M={self.M}, fitted={self._fitted}"
