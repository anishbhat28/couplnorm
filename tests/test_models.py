"""Tests for the baseline generative models."""
from __future__ import annotations

import pytest
import torch

from couplnorm import coupling_from_samples
from couplnorm.models import MAF, FourierModel, FullGaussian, PCAModel


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def make_coupled_field(B: int, N: int, coupling_strength: float = 1.5) -> torch.Tensor:
    """Global multiplicative latent ties all mode amplitudes -> high C."""
    z = 1.0 + coupling_strength * torch.randn(B, 1)
    eta = torch.randn(B, N)
    return z * eta


class TestFullGaussian:
    def test_sample_shape_and_finite(self):
        data = make_coupled_field(2000, 16)
        model = FullGaussian(16).fit(data)
        s = model.sample(500)
        assert s.shape == (500, 16)
        assert torch.isfinite(s).all()

    def test_reproduces_covariance(self):
        data = make_coupled_field(8000, 12)
        model = FullGaussian(12).fit(data)
        samples = model.sample(20000)
        cov_data = torch.cov(data.T)
        cov_samp = torch.cov(samples.T)
        rel = torch.linalg.matrix_norm(cov_samp - cov_data) / torch.linalg.matrix_norm(cov_data)
        assert rel.item() < 0.1

    def test_sample_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            FullGaussian(8).sample(4)


class TestFourierModel:
    def test_independent_modes_give_low_C(self):
        """The marginal-independence baseline erases coupling: C -> ~0."""
        data = make_coupled_field(8000, 32, coupling_strength=2.0)
        C_data = coupling_from_samples(data).item()
        model = FourierModel(32).fit(data)
        C_model = coupling_from_samples(model.sample(8000)).item()
        assert C_data > 0.3
        assert C_model < 0.10
        assert C_model < C_data

    def test_sample_is_real_and_right_shape(self):
        data = make_coupled_field(1000, 16)
        model = FourierModel(16).fit(data)
        s = model.sample(200)
        assert s.shape == (200, 16)
        assert s.dtype == torch.float32
        assert torch.isfinite(s).all()


class TestPCAModel:
    def test_sample_shape_and_finite(self):
        data = make_coupled_field(2000, 16)
        model = PCAModel(16, n_components=4).fit(data)
        s = model.sample(300)
        assert s.shape == (300, 16)
        assert torch.isfinite(s).all()

    def test_bad_n_components(self):
        with pytest.raises(ValueError):
            PCAModel(8, n_components=20)


class TestMAF:
    def test_sample_shape_and_finite(self):
        model = MAF(8, hidden=32, n_blocks=2)
        s = model.sample(16)
        assert s.shape == (16, 8)
        assert torch.isfinite(s).all()

    def test_fit_reduces_nll(self):
        torch.manual_seed(0)
        data = make_coupled_field(1024, 8, coupling_strength=1.0)
        model = MAF(8, hidden=32, n_blocks=2)
        with torch.no_grad():
            nll_start = -model.log_prob(data).mean().item()
        model.fit(data, epochs=40, lr=1e-3, batch_size=256)
        with torch.no_grad():
            nll_end = -model.log_prob(data).mean().item()
        assert nll_end < nll_start
