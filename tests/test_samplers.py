"""Tests for the phi^4 reference sampler."""
from __future__ import annotations

import pytest
import torch

from couplnorm import coupling_from_samples
from couplnorm.samplers import Phi4Sampler, free_theory_mode_variance


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


class TestFreeTheory:
    def test_mode_variance_matches_analytical(self):
        """At lam=0 the empirical per-mode energy matches 1/(m2 + 4 sin^2)."""
        N, m2 = 32, 1.0
        sampler = Phi4Sampler(N=N, m2=m2, lam=0.0, step=1.0, n_therm=1500)
        phi = sampler.sample(6000, seed=0)

        x_hat = torch.fft.rfft(phi, dim=-1, norm="ortho")
        E = x_hat.real.pow(2) + x_hat.imag.pow(2)
        empirical = E.mean(dim=0)
        analytical = free_theory_mode_variance(N, m2=m2, real_fft=True)

        rel_err = ((empirical - analytical).abs() / analytical).mean().item()
        assert rel_err < 0.10, f"mean relative error {rel_err:.3f} too large"

    def test_free_theory_has_low_coupling(self):
        """Independent modes => C near the finite-sample noise floor."""
        sampler = Phi4Sampler(N=32, m2=1.0, lam=0.0, step=1.0, n_therm=1500)
        phi = sampler.sample(6000, seed=1)
        C = coupling_from_samples(phi).item()
        assert C < 0.12, f"free-theory C={C:.3f} unexpectedly high"


class TestInteracting:
    def test_coupling_increases_with_lambda(self):
        """Turning on the quartic coupling raises C above the free theory."""
        free = Phi4Sampler(N=32, m2=1.0, lam=0.0, step=1.0, n_therm=1200)
        inter = Phi4Sampler(N=32, m2=1.0, lam=1.0, step=0.7, n_therm=1200)
        C_free = coupling_from_samples(free.sample(5000, seed=2)).item()
        C_inter = coupling_from_samples(inter.sample(5000, seed=3)).item()
        assert C_inter > C_free


class TestMechanics:
    def test_sample_shape_and_finite(self):
        sampler = Phi4Sampler(N=16, n_therm=100)
        phi = sampler.sample(64)
        assert phi.shape == (64, 16)
        assert torch.isfinite(phi).all()

    def test_reproducible_with_seed(self):
        sampler = Phi4Sampler(N=16, lam=0.5, n_therm=100)
        a = sampler.sample(32, seed=7)
        b = sampler.sample(32, seed=7)
        assert torch.allclose(a, b)

    def test_acceptance_rate_recorded(self):
        sampler = Phi4Sampler(N=16, n_therm=200)
        sampler.sample(128, seed=0)
        assert sampler.last_acceptance is not None
        assert 0.0 < sampler.last_acceptance <= 1.0

    def test_odd_N_rejected(self):
        with pytest.raises(ValueError, match="even"):
            Phi4Sampler(N=17)
