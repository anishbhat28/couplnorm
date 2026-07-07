"""couplnorm: normalized 4th-order coupling metrics and losses for
generative model evaluation on Fourier-decomposable data.
"""
from couplnorm.coupling import (
    CouplingLoss,
    CouplingMetric,
    coupling_from_samples,
)

__all__ = ["CouplingMetric", "CouplingLoss", "coupling_from_samples"]
__version__ = "0.1.0"
