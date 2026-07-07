"""Tests for CouplingMetric and CouplingLoss."""
from __future__ import annotations

import math

import pytest
import torch

from couplnorm import CouplingLoss, CouplingMetric, coupling_from_samples


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(0)


def make_independent_gaussian(B, N, var=None):
    if var is None:
        var = torch.ones(N) * 0.5
    return torch.randn(B, N) * torch.sqrt(var)


def make_coupled_field(B, N, coupling_strength=1.0):
    z = 1.0 + coupling_strength * torch.randn(B, 1)
    eta = torch.randn(B, N)
    return z * eta


class TestCouplingMetricBatch:
    def test_independent_modes_gives_low_C(self):
        phi = make_independent_gaussian(8000, 32)
        C = CouplingMetric(input_space="position", mode="batch")(phi).item()
        assert 0.0 <= C < 0.10

    def test_coupled_modes_gives_high_C(self):
        phi = make_coupled_field(4000, 32, coupling_strength=2.0)
        assert CouplingMetric()(phi).item() > 0.3

    def test_C_is_scalar(self):
        C = CouplingMetric()(torch.randn(100, 16))
        assert C.dim() == 0 and torch.isfinite(C)

    def test_C_in_unit_interval(self):
        for _ in range(5):
            C = CouplingMetric()(torch.randn(200, 16)).item()
            assert -1e-6 <= C <= 1.0 + 1e-6

    def test_position_and_spectral_consistent(self):
        phi = torch.randn(500, 16)
        phi_hat = torch.fft.rfft(phi, dim=-1, norm="ortho")
        E = phi_hat.real.pow(2) + phi_hat.imag.pow(2)
        C_pos = CouplingMetric(input_space="position")(phi).item()
        C_spec = CouplingMetric(input_space="spectral")(E).item()
        assert math.isclose(C_pos, C_spec, rel_tol=1e-6, abs_tol=1e-8)

    def test_functional_matches_module(self):
        phi = torch.randn(500, 16)
        assert math.isclose(
            CouplingMetric()(phi).item(), coupling_from_samples(phi).item(),
            rel_tol=1e-6, abs_tol=1e-8,
        )

    def test_full_fft_inflates_C_for_real_field(self):
        torch.manual_seed(0)
        N = 32
        phi = torch.randn(10000, N)
        C_full = CouplingMetric(real_fft=False)(phi).item()
        C_unique = CouplingMetric(real_fft=True)(phi).item()
        analytical_floor = math.sqrt((N - 2) / (2 * N - 2))
        assert abs(C_full - analytical_floor) < 0.05
        assert C_unique < 0.10


class TestCouplingMetricRunning:
    def test_running_converges_to_batch(self):
        B_total, N, n_chunks = 8000, 16, 80
        M = N // 2 + 1
        phi = make_coupled_field(B_total, N, coupling_strength=1.5)
        C_batch = CouplingMetric(mode="batch")(phi).item()
        metric = CouplingMetric(mode="running", momentum=0.02, n_modes=M)
        chunk = B_total // n_chunks
        for i in range(n_chunks):
            metric(phi[i * chunk : (i + 1) * chunk])
        assert abs(metric.compute().item() - C_batch) < 0.05

    def test_reset_clears_state(self):
        M = 16 // 2 + 1
        metric = CouplingMetric(mode="running", n_modes=M)
        metric(torch.randn(100, 16))
        assert metric.num_batches_tracked.item() == 1
        metric.reset()
        assert metric.num_batches_tracked.item() == 0
        with pytest.raises(RuntimeError):
            metric.compute()

    def test_lazy_buffer_allocation(self):
        metric = CouplingMetric(mode="running")
        metric(torch.randn(100, 24))
        M = 24 // 2 + 1
        assert metric.running_mean.shape == (M,)
        assert metric.running_second.shape == (M, M)

    def test_running_rejects_mode_change(self):
        M = 16 // 2 + 1
        metric = CouplingMetric(mode="running", n_modes=M)
        metric(torch.randn(50, 16))
        with pytest.raises(ValueError, match="Number of modes changed"):
            metric(torch.randn(50, 32))

    def test_compute_before_forward_raises(self):
        with pytest.raises(RuntimeError):
            CouplingMetric(mode="running", n_modes=9).compute()

    def test_covariance_only_in_running(self):
        with pytest.raises(RuntimeError):
            CouplingMetric(mode="batch").covariance()
        M = 16 // 2 + 1
        metric = CouplingMetric(mode="running", n_modes=M)
        metric(torch.randn(200, 16))
        cov = metric.covariance()
        assert cov.shape == (M, M)
        assert torch.all(torch.diagonal(cov) > 0)


class TestCouplingMetricErrors:
    def test_bad_input_space(self):
        with pytest.raises(ValueError, match="input_space"):
            CouplingMetric(input_space="bogus")

    def test_bad_mode(self):
        with pytest.raises(ValueError, match="mode"):
            CouplingMetric(mode="bogus")

    def test_bad_momentum(self):
        with pytest.raises(ValueError, match="momentum"):
            CouplingMetric(mode="running", momentum=0.0)
        with pytest.raises(ValueError, match="momentum"):
            CouplingMetric(mode="running", momentum=-0.1)

    def test_single_sample_batch_mode(self):
        with pytest.raises(ValueError, match="B >= 2"):
            CouplingMetric(mode="batch", unbiased=True)(torch.randn(1, 16))

    def test_3d_input_rejected(self):
        with pytest.raises(ValueError, match="1D or 2D"):
            CouplingMetric()(torch.randn(2, 4, 8))


class TestGradients:
    def test_gradient_flows_through_metric(self):
        phi = torch.randn(200, 16, requires_grad=True)
        CouplingMetric()(phi).backward()
        assert phi.grad is not None and phi.grad.shape == phi.shape
        assert torch.isfinite(phi.grad).all() and phi.grad.abs().sum() > 0

    def test_gradient_flows_through_loss(self):
        M = 16 // 2 + 1
        loss_fn = CouplingLoss(target_type="matrix", target=torch.eye(M) * 2.0)
        phi = torch.randn(200, 16, requires_grad=True)
        loss_fn(phi).backward()
        assert phi.grad is not None and torch.isfinite(phi.grad).all()


class TestCouplingLossMatrix:
    def test_loss_at_target_is_small(self):
        torch.manual_seed(42)
        data = make_coupled_field(8000, 16, coupling_strength=1.0)
        loss_fn = CouplingLoss.from_data(data, target_type="matrix")
        torch.manual_seed(43)
        same = make_coupled_field(8000, 16, coupling_strength=1.0)
        assert 0.0 < loss_fn(same).item() < 0.05

    def test_loss_grows_with_distribution_mismatch(self):
        torch.manual_seed(42)
        data = make_coupled_field(8000, 16, coupling_strength=2.0)
        loss_fn = CouplingLoss.from_data(data, target_type="matrix")
        torch.manual_seed(99)
        L_indep = loss_fn(make_independent_gaussian(8000, 16)).item()
        L_coupled = loss_fn(make_coupled_field(8000, 16, coupling_strength=2.0)).item()
        assert L_indep > 5 * L_coupled


class TestCouplingLossScalar:
    def test_scalar_loss_zero_at_target(self):
        torch.manual_seed(0)
        data = make_coupled_field(8000, 16, coupling_strength=1.5)
        C_target = coupling_from_samples(data).item()
        loss_fn = CouplingLoss(target_type="scalar", target=C_target)
        assert loss_fn(data).item() < 1e-4

    def test_scalar_loss_positive_off_target(self):
        loss_fn = CouplingLoss(target_type="scalar", target=0.5)
        assert loss_fn(torch.randn(4000, 32)).item() > 0.15


class TestCouplingLossRegularize:
    def test_regularize_returns_C(self):
        torch.manual_seed(7)
        phi = make_coupled_field(4000, 16, coupling_strength=1.5)
        L = CouplingLoss(target_type="regularize")(phi).item()
        C = coupling_from_samples(phi).item()
        assert math.isclose(L, C, rel_tol=1e-6, abs_tol=1e-8)


class TestOptimization:
    def test_loss_decreases_under_optimization(self):
        torch.manual_seed(0)
        target_data = make_coupled_field(8000, 16, coupling_strength=1.0)
        loss_fn = CouplingLoss.from_data(target_data, target_type="matrix")
        G = torch.nn.Linear(8, 16, bias=False)
        opt = torch.optim.Adam(G.parameters(), lr=1e-2)
        losses = []
        for _ in range(200):
            L = loss_fn(G(torch.randn(256, 8)))
            opt.zero_grad(); L.backward(); opt.step()
            losses.append(L.item())
        assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5


class TestCouplingLossErrors:
    def test_matrix_requires_target(self):
        with pytest.raises(ValueError, match="matrix.*requires"):
            CouplingLoss(target_type="matrix")

    def test_scalar_requires_target(self):
        with pytest.raises(ValueError, match="scalar.*requires"):
            CouplingLoss(target_type="scalar")

    def test_matrix_bad_shape(self):
        with pytest.raises(ValueError, match="must be"):
            CouplingLoss(target_type="matrix", target=torch.zeros(4, 5))

    def test_bad_target_type(self):
        with pytest.raises(ValueError, match="target_type"):
            CouplingLoss(target_type="bogus")

    def test_from_data_rejects_regularize(self):
        with pytest.raises(ValueError, match="from_data"):
            CouplingLoss.from_data(torch.randn(100, 16), target_type="regularize")
