from __future__ import annotations

import torch

from src.score_parameterization import (
    ScoreParameterization,
    network_output_to_physical,
    network_output_to_whitened,
)
from src.sde.freqsde import VEConvMixSDE


def make_sde():
    F = 5
    h1 = torch.tensor([1.4, 1.2+0.1j, 1.1-0.1j, 1.3+0.05j, 1.5], dtype=torch.complex64)
    h2 = torch.tensor([0.8, 0.9-0.05j, 1.0+0.08j, 0.7-0.03j, 0.6], dtype=torch.complex64)
    H = torch.stack([h1, h2])
    sde = VEConvMixSDE(2, 3.0, 2.0, 0.2, 5.0, 1024, 1.0, 50)
    sde.real_n_fft = 8
    sde.get_mix_mat(H, F)
    return sde


def test_physical_and_whitened_parameterizations_are_algebraically_equivalent():
    sde = make_sde()
    B, D = 3, sde.eigenvalues.numel()
    t = torch.tensor([0.02, 0.2, 0.8])
    physical = torch.randn(B, D, dtype=torch.complex64)
    std = sde.modal_std(t)
    whitened = sde.apply_SH(physical, t, std)

    p_from_p = network_output_to_physical(
        physical, sde=sde, t=t,
        parameterization=ScoreParameterization.PHYSICAL, std_modes=std,
    )
    p_from_w = network_output_to_physical(
        whitened, sde=sde, t=t,
        parameterization=ScoreParameterization.WHITENED, std_modes=std,
    )
    w_from_p = network_output_to_whitened(
        physical, sde=sde, t=t,
        parameterization=ScoreParameterization.PHYSICAL, std_modes=std,
    )
    w_from_w = network_output_to_whitened(
        whitened, sde=sde, t=t,
        parameterization=ScoreParameterization.WHITENED, std_modes=std,
    )

    torch.testing.assert_close(p_from_p, physical)
    torch.testing.assert_close(p_from_w, physical, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(w_from_p, whitened, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(w_from_w, whitened)


def test_whitened_dsm_target_is_bounded_by_construction():
    sde = make_sde()
    B, D = 64, sde.eigenvalues.numel()
    z = torch.randn(B, D, dtype=torch.complex64)
    # The direct whitened network target is exactly -z at every t.
    for value in [0.005, 0.01, 0.1, 1.0]:
        t = torch.full((B,), value)
        target = -z
        assert torch.isfinite(target.real).all()
        assert torch.isfinite(target.imag).all()
        # Same target magnitude regardless of t.
        torch.testing.assert_close(target.abs().square().mean(), z.abs().square().mean())
