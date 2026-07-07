"""Reference sampler for the 1D periodic phi^4 scalar field.

This module is plumbing: it exists so users can install couplnorm and
reproduce the paper plots without writing their own MCMC. The intellectual
core of the library is ``coupling.py``; this file just feeds it data.

Lattice action (a = 1, periodic boundary conditions)::

    S(phi) = sum_x [ (1/2) (phi_{x+1} - phi_x)^2
                     + (1/2) m2 * phi_x^2
                     + lam * phi_x^4 ]

At ``lam = 0`` the action is quadratic and the Fourier basis diagonalizes it,
so distinct modes are independent Gaussians. The per-mode variance under the
unitary ("ortho") FFT is then exactly ``1 / (m2 + 4 sin^2(pi k / N))``; see
:func:`free_theory_mode_variance`. Turning on ``lam > 0`` couples the modes
through the quartic vertex, which is exactly the structure the C diagnostic
picks up.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

__all__ = ["Phi4Sampler", "free_theory_mode_variance"]


def free_theory_mode_variance(
    N: int,
    m2: float = 1.0,
    real_fft: bool = True,
    device=None,
    dtype=torch.float32,
) -> Tensor:
    """Analytical spectral-energy expectation for the free (lam=0) theory.

    Returns E[|phi_tilde_k|^2] = 1 / (m2 + 4 sin^2(pi k / N)) under the unitary
    FFT convention (``norm="ortho"``). With ``real_fft=True`` only the
    ``N//2 + 1`` unique modes are returned, matching :func:`Phi4Sampler.sample`
    followed by ``torch.fft.rfft(..., norm="ortho")``.

    Parameters
    ----------
    N : int
        Number of lattice sites.
    m2 : float
        Mass-squared parameter of the action.
    real_fft : bool
        If True, return the ``N//2 + 1`` rfft modes; else all ``N`` modes.
    """
    k = torch.arange(N // 2 + 1 if real_fft else N, device=device, dtype=dtype)
    lattice_momentum = 4.0 * torch.sin(math.pi * k / N).pow(2)
    return 1.0 / (m2 + lattice_momentum)


class Phi4Sampler:
    """Checkerboard Metropolis-Hastings sampler for the 1D phi^4 field.

    Each call to :meth:`sample` runs ``n`` independent Markov chains in
    parallel (fully vectorized over chains and lattice sites), thermalizes
    them, and returns their final configurations. Because the 1D nearest-
    neighbor action couples only adjacent sites, even-indexed sites are
    conditionally independent given odd-indexed sites (and vice versa), so a
    whole sublattice can be proposed and accepted in one vectorized step.

    Parameters
    ----------
    N : int
        Number of lattice sites per configuration.
    m2 : float
        Mass-squared parameter of the action.
    lam : float
        Quartic coupling. ``lam = 0`` recovers the free theory.
    step : float
        Half-width of the uniform Metropolis proposal. Tune for ~40-60%
        acceptance; the default is reasonable for ``m2 ~ 1``.
    n_therm : int
        Number of thermalization sweeps before a configuration is returned.
    device, dtype :
        Passed through to the field tensors.
    """

    def __init__(
        self,
        N: int = 32,
        m2: float = 1.0,
        lam: float = 0.0,
        step: float = 1.0,
        n_therm: int = 1000,
        device=None,
        dtype=torch.float32,
    ):
        if N < 2:
            raise ValueError(f"N must be >= 2; got {N}")
        if N % 2 != 0:
            raise ValueError(
                f"N must be even for checkerboard updates; got {N}"
            )
        if step <= 0:
            raise ValueError(f"step must be > 0; got {step}")
        self.N = int(N)
        self.m2 = float(m2)
        self.lam = float(lam)
        self.step = float(step)
        self.n_therm = int(n_therm)
        self.device = device
        self.dtype = dtype
        # Cached acceptance rate from the most recent run, for diagnostics.
        self.last_acceptance: Optional[float] = None
        parity = torch.arange(self.N) % 2
        self._even = (parity == 0)
        self._odd = (parity == 1)

    def _color_update(self, phi: Tensor, color_mask: Tensor) -> Tensor:
        """Propose and accept/reject on one sublattice, in place-safe fashion."""
        left = phi.roll(1, dims=-1)   # phi_{x-1}
        right = phi.roll(-1, dims=-1)  # phi_{x+1}
        delta = (torch.rand_like(phi) * 2.0 - 1.0) * self.step
        phi_new = phi + delta

        d_kin = 0.5 * (
            (phi_new - right).pow(2) - (phi - right).pow(2)
            + (phi_new - left).pow(2) - (phi - left).pow(2)
        )
        d_mass = 0.5 * self.m2 * (phi_new.pow(2) - phi.pow(2))
        d_quartic = self.lam * (phi_new.pow(4) - phi.pow(4))
        dS = d_kin + d_mass + d_quartic

        accept = torch.rand_like(phi) < torch.exp(-dS)
        do_update = accept & color_mask
        self._accepted += (do_update & color_mask).sum()
        self._proposed += color_mask.sum() * phi.shape[0]
        return torch.where(do_update, phi_new, phi)

    def sweep(self, phi: Tensor) -> Tensor:
        """One full checkerboard sweep (even sublattice, then odd)."""
        mask_even = self._even.to(phi.device)
        mask_odd = self._odd.to(phi.device)
        phi = self._color_update(phi, mask_even)
        phi = self._color_update(phi, mask_odd)
        return phi

    @torch.no_grad()
    def sample(self, n: int, seed: Optional[int] = None) -> Tensor:
        """Draw ``n`` thermalized configurations, shape ``(n, N)``.

        Runs ``n`` parallel chains from a random start and returns each chain's
        configuration after ``n_therm`` sweeps.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1; got {n}")
        if seed is not None:
            torch.manual_seed(seed)
        phi = torch.randn(n, self.N, device=self.device, dtype=self.dtype)
        self._accepted = torch.zeros((), device=phi.device)
        self._proposed = torch.zeros((), device=phi.device)
        for _ in range(self.n_therm):
            phi = self.sweep(phi)
        denom = self._proposed.clamp_min(1.0)
        self.last_acceptance = float((self._accepted / denom).item())
        return phi

    def __repr__(self) -> str:
        return (
            f"Phi4Sampler(N={self.N}, m2={self.m2}, lam={self.lam}, "
            f"step={self.step}, n_therm={self.n_therm})"
        )
