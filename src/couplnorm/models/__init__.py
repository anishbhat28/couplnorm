"""Baseline generative models for couplnorm demos and benchmarks.

Every model is an ``nn.Module`` with a common interface:

- ``fit(data)`` estimates parameters from a ``(B, N)`` batch of position-space
  field configurations (a no-op / trainer depending on the model).
- ``sample(n)`` returns ``(n, N)`` real position-space configurations.

The models span the spectrum of joint-coupling fidelity:

- :class:`FourierModel` assumes marginal independence of Fourier modes, so its
  samples have C near zero regardless of the target. It is the "wrong by
  construction" baseline that motivates the C diagnostic.
- :class:`FullGaussian` matches the full second-order (covariance) structure of
  the data but, being Gaussian, cannot reproduce genuine 4th-order coupling.
- :class:`PCAModel` is a low-rank Gaussian; even weaker than FullGaussian.
- :class:`MAF` is a masked autoregressive flow that can, in principle, capture
  higher-order structure and drive C toward the target.
"""
from couplnorm.models.fourier import FourierModel
from couplnorm.models.gaussian import FullGaussian
from couplnorm.models.maf import MAF
from couplnorm.models.pca import PCAModel

__all__ = ["FourierModel", "FullGaussian", "PCAModel", "MAF"]
