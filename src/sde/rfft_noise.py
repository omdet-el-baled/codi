"""Noise helpers for one-sided FFT representations of real-valued signals.

For an orthonormal rFFT representation of a real signal, interior frequency bins
are proper complex Gaussian variables, whereas the DC bin and (for even n_fft)
the Nyquist bin are real-valued degrees of freedom.  Using an unconstrained
complex Gaussian at those endpoints creates targets that a physically
constrained score network cannot represent.
"""
from __future__ import annotations

import math
import torch


def _has_nyquist(n_bins: int, n_fft: int | None) -> bool:
    """Return whether the last rFFT bin is a Nyquist bin."""
    if n_fft is None:
        return False
    n_fft = int(n_fft)
    return n_fft % 2 == 0 and int(n_bins) == n_fft // 2 + 1


def standard_rfft_noise_like(
    x: torch.Tensor,
    *,
    num_sources: int,
    n_fft: int | None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw unit-variance Gaussian noise in the real-signal rFFT subspace.

    The expected input layout is ``[B, K*F]`` where ``K=num_sources`` and
    ``F`` is the number of non-redundant rFFT bins.  For complex tensors:

    * interior bins follow CN(0, 1), i.e. real and imaginary parts each have
      variance 1/2;
    * DC is real N(0, 1);
    * Nyquist is real N(0, 1) when ``n_fft`` is even.

    This matches the degrees of freedom of the rFFT of a real, unit-variance
    Gaussian signal under orthonormal FFT normalization.
    """
    if not torch.is_complex(x):
        return torch.randn(
            x.shape, dtype=x.dtype, device=x.device, generator=generator
        )

    if x.ndim != 2:
        raise ValueError(f"Expected flattened tensor [B, K*F], got {tuple(x.shape)}")

    B, D = x.shape
    K = int(num_sources)
    if K < 1 or D % K != 0:
        raise ValueError(f"Cannot reshape D={D} into K={K} source blocks")
    F = D // K

    rdtype = x.real.dtype
    real = torch.randn(x.shape, dtype=rdtype, device=x.device, generator=generator)
    imag = torch.randn(x.shape, dtype=rdtype, device=x.device, generator=generator)
    z = torch.complex(real, imag) / math.sqrt(2.0)
    z = z.view(B, K, F)

    # DC is a single real degree of freedom with unit variance.
    dc = torch.randn((B, K), dtype=rdtype, device=x.device, generator=generator)
    z[..., 0] = dc.to(z.dtype)

    # For even-length real FFTs, Nyquist is also purely real with unit variance.
    if _has_nyquist(F, n_fft):
        nyq = torch.randn((B, K), dtype=rdtype, device=x.device, generator=generator)
        z[..., -1] = nyq.to(z.dtype)

    return z.reshape_as(x)


def project_rfft_endpoints(
    x: torch.Tensor,
    *,
    num_sources: int,
    n_fft: int | None,
) -> torch.Tensor:
    """Project DC/Nyquist values onto the real rFFT subspace.

    This is useful after adding deterministic shifts (e.g. the terminal
    mismatch correction), where tiny imaginary endpoint components can arise
    numerically even though the physical quantity should be real.
    """
    if not torch.is_complex(x):
        return x
    if x.ndim != 2:
        raise ValueError(f"Expected flattened tensor [B, K*F], got {tuple(x.shape)}")

    B, D = x.shape
    K = int(num_sources)
    if K < 1 or D % K != 0:
        raise ValueError(f"Cannot reshape D={D} into K={K} source blocks")
    F = D // K

    out = x.view(B, K, F).clone()
    out[..., 0] = out[..., 0].real.to(out.dtype)
    if _has_nyquist(F, n_fft):
        out[..., -1] = out[..., -1].real.to(out.dtype)
    return out.reshape_as(x)
