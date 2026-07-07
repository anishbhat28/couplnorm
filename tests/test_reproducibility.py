"""Reproducibility test: pin the library's C to your paper's own MCMC.

This test is **skipped by default**. To enable it, drop a reference blob at
``tests/data/phi4_reference.pt`` (or point ``COUPLNORM_MCMC_REFERENCE`` at one).
The blob pins couplnorm's output to a C value computed by an independent source
(your original paper code), so any future refactor that changes the number is
caught immediately.

Expected blob format (a dict saved with ``torch.save``)::

    {
        "samples": Tensor of shape (B, N),   # position-space field configs
        "C_reference": float,                # C from your original/paper code
        "real_fft": True,                    # optional, default True
        "rel_tol": 0.02,                     # optional, default 0.02
        "meta": {"lambda": ..., "N": ...},   # optional, free-form
    }

See ``tests/data/README.md`` for how to build one.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from couplnorm import coupling_from_samples

_ENV_PATH = os.environ.get("COUPLNORM_MCMC_REFERENCE")
_DEFAULT_PATH = Path(__file__).parent / "data" / "phi4_reference.pt"
_REFERENCE = Path(_ENV_PATH) if _ENV_PATH else _DEFAULT_PATH


@pytest.mark.skipif(
    not _REFERENCE.exists(),
    reason=(
        f"No reference blob at {_REFERENCE}. Drop one there (see "
        "tests/data/README.md) or set COUPLNORM_MCMC_REFERENCE to enable."
    ),
)
def test_matches_reference_mcmc():
    blob = torch.load(_REFERENCE, weights_only=False)
    samples = blob["samples"]
    if not torch.is_tensor(samples):
        samples = torch.as_tensor(samples)
    samples = samples.to(torch.float32)
    if samples.dim() != 2:
        raise ValueError(
            f"reference 'samples' must be (B, N); got {tuple(samples.shape)}"
        )

    C_ref = float(blob["C_reference"])
    real_fft = bool(blob.get("real_fft", True))
    rel_tol = float(blob.get("rel_tol", 0.02))

    C_new = coupling_from_samples(samples, real_fft=real_fft).item()
    rel_err = abs(C_new - C_ref) / max(abs(C_ref), 1e-6)
    assert rel_err < rel_tol, (
        f"couplnorm C={C_new:.6f} disagrees with reference C={C_ref:.6f} "
        f"(relative error {rel_err:.4f} >= tol {rel_tol})"
    )
