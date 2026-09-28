from __future__ import annotations

import math

import pytest
import torch

from src.sde.freqsde import VEConvMixSDE


def make_sde(*, K=2, F=5, alpha=3.0, beta=2.0, cov_steps=512):
    torch.manual_seed(0)
    # Real filters imply Hermitian-compatible rFFT responses. Keep H0 away from 0.
    h = torch.randn(K, F, dtype=torch.complex64) * 0.08
    h[:, 0] = h[:, 0].real + 1.0
    if F > 1:
        h[0] = h[0] + (1.5 + 0j)
    sde = VEConvMixSDE(
        feature_dim=K,
        drift_decay_rate=alpha,
        zero_mode_decay_rate=beta,
        sigma_min=0.2,
        sigma_max=5.0,
        steps=cov_steps,
        terminal_time=1.0,
        num_reverse_steps=50,
        drift_mode="mixing",
        store_full_matrix=True,
    )
    sde.real_n_fft = 2 * (F - 1)
    sde.get_mix_mat(h, F)
    return sde, h


def explicit_A(h):
    K, F = h.shape
    row = torch.cat([torch.diag(h[k]) for k in range(K)], dim=1)
    return row.repeat(K, 1)


def test_eigendecomposition_reconstructs_mixing_operator():
    sde, h = make_sde()
    A = explicit_A(h)
    A_hat = (
        sde.eigenvector_matrix
        @ torch.diag(sde.eigenvalues)
        @ sde.inv_eigenvector_matrix
    )
    torch.testing.assert_close(A_hat, A, rtol=2e-5, atol=2e-5)


def test_instantaneous_drift_is_derivative_of_integrated_drift():
    sde, _ = make_sde(alpha=4.0, beta=3.0)
    t = torch.tensor([0.13, 0.47, 0.83], dtype=torch.float32)
    eps = 2e-4
    num = (
        sde._eigval_Ft_integral(t + eps)
        - sde._eigval_Ft_integral(t - eps)
    ) / (2 * eps)
    ana = sde._eigval_Ft(t)
    torch.testing.assert_close(num, ana, rtol=4e-3, atol=4e-3)


def test_structured_exp_matches_dense_exp_operator():
    sde, _ = make_sde()
    B, D = 3, sde.eigenvalues.numel()
    x = torch.randn(B, D, dtype=torch.complex64)
    t = torch.tensor([0.05, 0.4, 1.0])
    structured = sde.apply_exp_Ft(x, t)
    dense = torch.einsum("bij,bj->bi", sde.get_exp_Ft_integral(t), x)
    torch.testing.assert_close(structured, dense, rtol=2e-5, atol=2e-5)


def test_S_and_SH_structured_ops_match_dense_factor():
    sde, _ = make_sde(cov_steps=1024)
    B, D = 2, sde.eigenvalues.numel()
    t = torch.tensor([0.05, 0.7])
    z = torch.randn(B, D, dtype=torch.complex64)
    x = torch.randn(B, D, dtype=torch.complex64)

    std = sde.modal_std(t)
    dense_S = sde.eigenvector_matrix.unsqueeze(0) * std.unsqueeze(1)

    torch.testing.assert_close(
        sde.apply_S(z, t, std_modes=std),
        torch.einsum("bij,bj->bi", dense_S, z),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        sde.apply_SH(x, t, std_modes=std),
        torch.einsum("bij,bj->bi", dense_S.conj().transpose(1, 2), x),
        rtol=2e-5,
        atol=2e-5,
    )


def test_solve_S_and_solve_SH_are_inverses():
    sde, _ = make_sde(cov_steps=1024)
    B, D = 2, sde.eigenvalues.numel()
    t = torch.tensor([0.03, 0.6])
    z = torch.randn(B, D, dtype=torch.complex64)
    x = torch.randn(B, D, dtype=torch.complex64)
    std = sde.modal_std(t)

    torch.testing.assert_close(
        sde.solve_S(sde.apply_S(z, t, std), t, std),
        z,
        rtol=3e-5,
        atol=3e-5,
    )
    torch.testing.assert_close(
        sde.solve_SH(sde.apply_SH(x, t, std), t, std),
        x,
        rtol=4e-5,
        atol=4e-5,
    )


def test_diffusion_covariance_action_matches_dense_BBH():
    sde, _ = make_sde()
    B, D = 3, sde.eigenvalues.numel()
    x = torch.randn(B, D, dtype=torch.complex64)
    y = torch.zeros(B, D // 2, dtype=torch.complex64)
    t = torch.tensor([0.02, 0.3, 0.9])
    _, Bmat = sde.sde(x, y, t)
    score = torch.randn(B, D, dtype=torch.complex64)
    dense = torch.einsum(
        "bij,bj->bi",
        Bmat,
        torch.einsum("bij,bj->bi", Bmat.conj().transpose(1, 2), score),
    )
    structured = sde.apply_diffusion_covariance(score, t)
    torch.testing.assert_close(structured, dense, rtol=3e-5, atol=3e-5)


def test_modal_variance_satisfies_scalar_lyapunov_ode():
    # Numerical derivative of q=|s|^2 should satisfy q'=2 Re(f)q+g^2.
    sde, _ = make_sde(cov_steps=2048)
    t0 = 0.35
    h = 2e-3
    tm = torch.tensor([t0 - h])
    tp = torch.tensor([t0 + h])
    t = torch.tensor([t0])
    qm = sde.modal_std(tm).abs().square()[0]
    qp = sde.modal_std(tp).abs().square()[0]
    q = sde.modal_std(t).abs().square()[0]
    qdot_num = (qp - qm) / (2 * h)
    f = sde._eigval_Ft(t)[0].real
    g2 = (
        2.0
        * sde.logsig
        * sde.sigma_min**2
        * (sde.ratiosig ** (2.0 * t0 / sde.T))
    )
    rhs = 2.0 * f * q + g2
    torch.testing.assert_close(qdot_num, rhs, rtol=2e-2, atol=2e-2)
