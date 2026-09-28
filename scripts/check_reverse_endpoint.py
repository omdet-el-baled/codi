#!/usr/bin/env python3
"""Numerical sanity check: fixed reverse solvers must stop exactly at eps.

Uses a constant ODE dx/dt = 1. Starting from x(T)=0, exact backward
integration to eps gives x(eps)=eps-T, independent of the number of steps.
"""
import torch
from src.sde.fixed_samplers import sample_fixed


class DummySDE:
    T = 1.0
    feature_dim = 1
    real_n_fft = None

    def copy(self):
        return self

    def prior_sampling(self, shape, y):
        return torch.zeros_like(y)

    def sde(self, x, y, t):
        B, D = x.shape
        drift = torch.ones_like(x)
        diffusion = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
        return drift, diffusion


def zero_score(x, y, t):
    return torch.zeros_like(x)


def main():
    eps = 0.005
    y = torch.zeros(2, 3, dtype=torch.complex64)
    target = eps - 1.0
    for method in ("euler", "heun", "rk4"):
        for steps in (5, 10, 20, 50, 100, 500):
            out = sample_fixed(
                DummySDE(), zero_score, y, method=method, steps=steps, eps=eps
            )
            err = float((out.sample.real - target).abs().max())
            print(
                f"{method:5s} steps={steps:3d} final_time={out.final_time:.9f} "
                f"endpoint_error={err:.3e} nfe={out.nfe}"
            )
            assert abs(out.final_time - eps) < 1e-7
            assert err < 2e-5
    print("PASS: all deterministic fixed-step reverse samplers stop at t_epsilon.")


if __name__ == "__main__":
    main()
