"""Normalized off-diagonal coupling diagnostic and loss for evaluating
generative models on Fourier-decomposable data.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["CouplingMetric", "CouplingLoss", "coupling_from_samples"]


def _spectral_energies(
    x: Tensor,
    input_space: str = "position",
    fft_norm: str = "ortho",
    real_fft: bool = True,
) -> Tensor:
    if input_space == "position":
        if real_fft:
            x_hat = torch.fft.rfft(x, dim=-1, norm=fft_norm)
        else:
            x_hat = torch.fft.fft(x, dim=-1, norm=fft_norm)
        return x_hat.real.pow(2) + x_hat.imag.pow(2)
    elif input_space == "spectral":
        return x
    else:
        raise ValueError(
            f"input_space must be 'position' or 'spectral', got {input_space!r}"
        )


def _batch_covariance(E: Tensor, unbiased: bool = True) -> Tensor:
    if E.dim() != 2:
        raise ValueError(f"Expected 2D tensor (B, M), got shape {tuple(E.shape)}")
    B = E.shape[0]
    if B < 2 and unbiased:
        raise ValueError(
            f"Need B >= 2 for unbiased covariance; got B={B}. "
            "Set unbiased=False or use mode='running' for streaming."
        )
    mean = E.mean(dim=0, keepdim=True)
    centered = E - mean
    denom = (B - 1) if unbiased else B
    return centered.T @ centered / denom


def _coupling_from_cov(cov: Tensor, eps: float = 1e-12) -> Tensor:
    diag_vec = torch.diagonal(cov)
    off_diag = cov - torch.diag_embed(diag_vec)
    num = torch.linalg.matrix_norm(off_diag, ord="fro")
    den = torch.linalg.matrix_norm(cov, ord="fro")
    return num / (den + eps)


def coupling_from_samples(
    x: Tensor,
    input_space: str = "position",
    fft_norm: str = "ortho",
    real_fft: bool = True,
    eps: float = 1e-12,
    unbiased: bool = True,
) -> Tensor:
    if x.dim() == 1:
        raise ValueError(
            "coupling_from_samples requires batched input (B, N). "
            "For streaming, use CouplingMetric(mode='running')."
        )
    E = _spectral_energies(x, input_space=input_space, fft_norm=fft_norm, real_fft=real_fft)
    cov = _batch_covariance(E, unbiased=unbiased)
    return _coupling_from_cov(cov, eps)


class CouplingMetric(nn.Module):
    def __init__(
        self,
        input_space: str = "position",
        mode: str = "batch",
        momentum: float = 0.1,
        eps: float = 1e-12,
        fft_norm: str = "ortho",
        real_fft: bool = True,
        unbiased: bool = True,
        n_modes: Optional[int] = None,
    ):
        super().__init__()
        if input_space not in ("position", "spectral"):
            raise ValueError(
                f"input_space must be 'position' or 'spectral'; got {input_space!r}"
            )
        if mode not in ("batch", "running"):
            raise ValueError(f"mode must be 'batch' or 'running'; got {mode!r}")
        if not (0.0 < momentum <= 1.0):
            raise ValueError(f"momentum must be in (0, 1]; got {momentum}")

        self.input_space = input_space
        self.mode = mode
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.fft_norm = fft_norm
        self.real_fft = bool(real_fft)
        self.unbiased = bool(unbiased)

        if mode == "running":
            if n_modes is not None:
                self.register_buffer("running_mean", torch.zeros(n_modes))
                self.register_buffer("running_second", torch.zeros(n_modes, n_modes))
            else:
                self.register_buffer("running_mean", torch.empty(0))
                self.register_buffer("running_second", torch.empty(0))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def _maybe_init_buffers(self, M: int, device, dtype) -> None:
        if self.running_mean.numel() == 0:
            self.running_mean = torch.zeros(M, device=device, dtype=dtype)
            self.running_second = torch.zeros(M, M, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(f"Expected 1D or 2D input, got shape {tuple(x.shape)}")
        E = _spectral_energies(x, self.input_space, self.fft_norm, self.real_fft)
        B, M = E.shape

        if self.mode == "batch":
            cov = _batch_covariance(E, unbiased=self.unbiased)
            return _coupling_from_cov(cov, self.eps)

        self._maybe_init_buffers(M, E.device, E.dtype)
        if self.running_mean.shape[0] != M:
            raise ValueError(
                "Number of modes changed: running buffers have "
                f"N={self.running_mean.shape[0]} but got N={M}."
            )

        with torch.no_grad():
            batch_mean = E.mean(dim=0)
            batch_second = (E.T @ E) / B
            m = self.momentum
            self.running_mean.mul_(1.0 - m).add_(batch_mean, alpha=m)
            self.running_second.mul_(1.0 - m).add_(batch_second, alpha=m)
            self.num_batches_tracked.add_(1)
        return self.compute()

    def compute(self) -> Tensor:
        if self.mode != "running":
            raise RuntimeError("compute() is only valid in running mode.")
        if self.running_mean.numel() == 0 or self.num_batches_tracked.item() == 0:
            raise RuntimeError("No batches tracked yet; call forward() at least once.")
        mu = self.running_mean
        M = self.running_second
        cov = M - torch.outer(mu, mu)
        return _coupling_from_cov(cov, self.eps)

    def covariance(self) -> Tensor:
        if self.mode != "running":
            raise RuntimeError(
                "covariance() requires mode='running'. For one-off batches, "
                "use coupling_from_samples or recompute manually."
            )
        mu = self.running_mean
        M = self.running_second
        return M - torch.outer(mu, mu)

    def reset(self) -> None:
        if self.mode != "running":
            return
        if self.running_mean.numel() > 0:
            self.running_mean.zero_()
            self.running_second.zero_()
        self.num_batches_tracked.zero_()

    def extra_repr(self) -> str:
        return (
            f"input_space={self.input_space!r}, mode={self.mode!r}, "
            f"momentum={self.momentum}, eps={self.eps}, "
            f"fft_norm={self.fft_norm!r}, real_fft={self.real_fft}"
        )


class CouplingLoss(nn.Module):
    def __init__(
        self,
        target_type: str = "matrix",
        target: Optional[Union[Tensor, float]] = None,
        input_space: str = "position",
        eps: float = 1e-12,
        fft_norm: str = "ortho",
        real_fft: bool = True,
        unbiased: bool = True,
    ):
        super().__init__()
        if target_type not in ("matrix", "scalar", "regularize"):
            raise ValueError(
                "target_type must be 'matrix', 'scalar', or 'regularize'; "
                f"got {target_type!r}"
            )
        if input_space not in ("position", "spectral"):
            raise ValueError(
                f"input_space must be 'position' or 'spectral'; got {input_space!r}"
            )

        if target_type == "matrix":
            if target is None:
                raise ValueError(
                    "target_type='matrix' requires a (M, M) target covariance."
                )
            tgt = torch.as_tensor(target, dtype=torch.float32)
            if tgt.dim() != 2 or tgt.shape[0] != tgt.shape[1]:
                raise ValueError(
                    f"matrix target must be (M, M); got shape {tuple(tgt.shape)}"
                )
            self.register_buffer("target_cov", tgt.detach().clone())
            self.register_buffer(
                "target_cov_frob",
                torch.linalg.matrix_norm(tgt, ord="fro").detach().clone(),
            )
        elif target_type == "scalar":
            if target is None:
                raise ValueError("target_type='scalar' requires a target C value.")
            self.register_buffer(
                "target_C", torch.as_tensor(float(target), dtype=torch.float32)
            )

        self.target_type = target_type
        self.input_space = input_space
        self.eps = float(eps)
        self.fft_norm = fft_norm
        self.real_fft = bool(real_fft)
        self.unbiased = bool(unbiased)

    @classmethod
    def from_data(
        cls,
        data: Tensor,
        target_type: str = "matrix",
        input_space: str = "position",
        fft_norm: str = "ortho",
        real_fft: bool = True,
        unbiased: bool = True,
        eps: float = 1e-12,
    ) -> "CouplingLoss":
        if target_type not in ("matrix", "scalar"):
            raise ValueError(f"from_data does not support target_type={target_type!r}")
        with torch.no_grad():
            E = _spectral_energies(
                data, input_space=input_space, fft_norm=fft_norm, real_fft=real_fft
            )
            cov = _batch_covariance(E, unbiased=unbiased)
            if target_type == "matrix":
                return cls(
                    target_type="matrix", target=cov, input_space=input_space, eps=eps,
                    fft_norm=fft_norm, real_fft=real_fft, unbiased=unbiased,
                )
            else:
                C = _coupling_from_cov(cov, eps).item()
                return cls(
                    target_type="scalar", target=C, input_space=input_space, eps=eps,
                    fft_norm=fft_norm, real_fft=real_fft, unbiased=unbiased,
                )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        E = _spectral_energies(x, self.input_space, self.fft_norm, self.real_fft)
        cov = _batch_covariance(E, unbiased=self.unbiased)
        if self.target_type == "matrix":
            diff = cov - self.target_cov
            num = torch.linalg.matrix_norm(diff, ord="fro").pow(2)
            den = self.target_cov_frob.pow(2) + self.eps
            return num / den
        elif self.target_type == "scalar":
            C = _coupling_from_cov(cov, self.eps)
            return (C - self.target_C).pow(2)
        else:
            return _coupling_from_cov(cov, self.eps)

    def extra_repr(self) -> str:
        return (
            f"target_type={self.target_type!r}, "
            f"input_space={self.input_space!r}, fft_norm={self.fft_norm!r}"
        )
