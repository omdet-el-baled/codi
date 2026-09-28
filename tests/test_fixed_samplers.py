from __future__ import annotations

import math
import torch

from src.sde.fixed_samplers import _time_grid, sample_fixed


class LinearODE:
    """Forward dx/dt = a x, B=0; used to test reverse numerical integration."""
    def __init__(self, a=1.0, T=1.0):
        self.a = float(a)
        self.T = float(T)
        self.feature_dim = 1
        self.real_n_fft = None

    def apply_Ft(self, x, t):
        return self.a * x

    def apply_diffusion_covariance(self, score, t):
        return torch.zeros_like(score)

    def sde(self, x, y, t):
        B, D = x.shape
        return self.a * x, torch.zeros(B, D, D, dtype=x.dtype, device=x.device)

    def prior_sampling(self, shape, y):
        raise RuntimeError("x_T is supplied explicitly in this test")


def zero_score(x, y, t):
    return torch.zeros_like(x)


def test_time_grid_has_exact_endpoint_and_interval_count():
    for steps in [1, 5, 10, 50, 100]:
        g = _time_grid(1.0, 0.005, steps, torch.device("cpu"), torch.float64)
        assert g.numel() == steps + 1
        assert float(g[0]) == 1.0
        assert float(g[-1]) == 0.005
        assert torch.all(g[:-1] > g[1:])


def test_euler_converges_with_more_steps_for_exact_smooth_vector_field():
    ode = LinearODE(a=1.0)
    eps = 0.005
    x0 = torch.tensor([[1.25]], dtype=torch.complex64)
    xT = x0 * math.exp(1.0)
    target = x0 * math.exp(eps)
    y = torch.zeros_like(x0)

    errors = []
    for n in [5, 10, 20, 50, 100, 200]:
        out = sample_fixed(
            ode, zero_score, y,
            method="euler", steps=n, eps=eps, x_T=xT,
        )
        errors.append(float((out.sample - target).abs().max()))
        assert out.nfe == n
        assert abs(out.final_time - eps) < 1e-7

    assert all(b < a for a, b in zip(errors, errors[1:])), errors


def test_heun_and_rk4_end_exactly_at_eps():
    ode = LinearODE(a=0.7)
    eps = 0.005
    x = torch.ones(2, 3, dtype=torch.complex64)
    y = torch.zeros_like(x)
    for method, nfe_factor in [("heun", 2), ("rk4", 4)]:
        out = sample_fixed(
            ode, zero_score, y,
            method=method, steps=7, eps=eps, x_T=x,
        )
        assert abs(out.final_time - eps) < 1e-7
        assert out.nfe == 7 * nfe_factor
