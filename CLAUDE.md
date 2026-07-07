# couplnorm — instructions for Claude Code

`couplnorm` is a small, focused PyTorch library implementing the normalized
off-diagonal coupling diagnostic **C**, a 4th-order spectral statistic. It is
exposed as an evaluation metric (`CouplingMetric`), a differentiable training
loss (`CouplingLoss`), and a one-line functional helper
(`coupling_from_samples`).

## The one hard rule

`src/couplnorm/coupling.py` and `tests/test_coupling.py` are **hand-authored by
the maintainer** and are the intellectual core of the library. Do **not**
rewrite, refactor, or "improve" them without an explicit request. If a change is
needed there, propose a diff and explain the reasoning; do not apply it silently.

Everything else — `samplers.py`, `models/`, `plotting.py`, `test_samplers.py`,
`test_models.py`, packaging, notebooks, CI — is plumbing and is fine to edit.

## Conventions that must not drift

- **FFT convention:** default `real_fft=True` uses `torch.fft.rfft` and keeps
  only the `N//2 + 1` unique modes of a real field. Using the full FFT on a real
  field inflates C by a conjugate-symmetry floor of `sqrt((N-2)/(2N-2)) ≈ 0.7`.
  Never change the default. The regression test in `test_coupling.py` guards it.
- **FFT normalization:** `norm="ortho"` (unitary). C is a ratio, so this cancels,
  but ortho keeps intermediate spectral quantities physically interpretable and
  makes the free-theory mode-variance formula in `samplers.py` exact.
- **Covariance layout:** `(batch, features)` — the deep-learning convention, the
  opposite of `torch.cov`. Helpers take `(B, M)` spectral energies.
- **Public API:** only `CouplingMetric`, `CouplingLoss`, and
  `coupling_from_samples` are exported from the top-level package. Everything
  prefixed with `_` is private.

## Layout

```
src/couplnorm/
  coupling.py     # hand-authored — the metric, loss, and helpers
  samplers.py     # phi^4 Metropolis-Hastings reference sampler
  models/         # baseline generative models (Fourier, PCA, Gaussian, MAF)
  plotting.py     # matplotlib helpers for the demo notebooks
tests/            # test_coupling.py is hand-authored; the rest are plumbing
notebooks/        # runnable demos
```

## Dev workflow

```bash
pip install -e .[dev]
pytest -q
```

Baseline models and the sampler all follow a common interface: construct, then
`.fit(data)` where applicable, then `.sample(n)` returning `(n, N)` real fields
in position space.
