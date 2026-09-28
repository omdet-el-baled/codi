from __future__ import annotations

import torch

from src.sde.rfft_noise import standard_rfft_noise_like


def test_even_rfft_dc_and_nyquist_are_real():
    B, K, n_fft = 4, 2, 16
    F = n_fft // 2 + 1
    ref = torch.zeros(B, K * F, dtype=torch.complex64)
    z = standard_rfft_noise_like(ref, num_sources=K, n_fft=n_fft)
    z = z.view(B, K, F)
    assert torch.count_nonzero(z[..., 0].imag) == 0
    assert torch.count_nonzero(z[..., -1].imag) == 0
    assert z[..., 1:-1].imag.abs().mean() > 0


def test_odd_rfft_only_dc_is_forced_real():
    B, K, n_fft = 4, 2, 15
    F = n_fft // 2 + 1
    ref = torch.zeros(B, K * F, dtype=torch.complex64)
    z = standard_rfft_noise_like(ref, num_sources=K, n_fft=n_fft)
    z = z.view(B, K, F)
    assert torch.count_nonzero(z[..., 0].imag) == 0
    assert z[..., -1].imag.abs().mean() > 0
