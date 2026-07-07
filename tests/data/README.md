# Reference data for the reproducibility test

`tests/test_reproducibility.py` is skipped until a reference blob exists here at
`phi4_reference.pt` (or at the path in `COUPLNORM_MCMC_REFERENCE`). The blob is
**not** committed — sample tensors are large and gitignored (`*.pt`). This
directory only tracks the format doc.

## Why

It pins couplnorm's `C` to a value produced by an **independent** source — your
original φ⁴ paper code. If a future refactor of `coupling.py` (say, an accidental
`fft` ↔ `rfft` swap) changes the number, this test fails and tells you.

## Build the blob

Run this once, using your paper's own MCMC and its own C implementation:

```python
import torch

# 1. Load ~5000 field configurations at a single (lambda, N) from your paper.
samples = load_my_phi4_samples()          # -> Tensor (B, N), position space

# 2. Compute C with your ORIGINAL paper code (not couplnorm), so the two are
#    genuinely independent.
C_reference = my_paper_coupling(samples)  # -> float

torch.save(
    {
        "samples": samples.to(torch.float32),
        "C_reference": float(C_reference),
        "real_fft": True,        # must match the convention your paper used
        "rel_tol": 0.02,         # 1-2% agreement is the target
        "meta": {"lambda": 0.5, "N": samples.shape[1]},
    },
    "tests/data/phi4_reference.pt",
)
```

Then `pytest tests/test_reproducibility.py -v` runs the check.

## Convention check

`real_fft` **must** match what your paper used. couplnorm defaults to `rfft`
(unique modes). If your paper used the full FFT on a real field, its reported C
includes the ~0.7 conjugate-symmetry floor — set `real_fft=False` here to
compare like-for-like, and note the discrepancy in your paper's errata/README.
