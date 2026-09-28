"""Fixed-step reverse samplers with explicit NFE accounting.

These samplers keep the project's original SDE/score interface but make solver
choice explicit.  Deterministic methods integrate the probability-flow ODE;
``em`` integrates the reverse SDE; ``pc`` adds Langevin corrector steps before
an Euler-Maruyama predictor.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch

from .rfft_noise import standard_rfft_noise_like


@dataclass
class SamplerResult:
    sample: torch.Tensor
    nfe: int
    steps: int
    method: str
    final_time: float


def _noise_like(x, sde, generator=None):
    return standard_rfft_noise_like(
        x,
        num_sources=int(getattr(sde, "feature_dim", 1)),
        n_fft=getattr(sde, "real_n_fft", None),
        generator=generator,
    )


def _time_grid(T, eps, steps, device, dtype, schedule="linear"):
    """Return exactly ``steps`` reverse intervals from ``T`` to ``eps``.

    The returned tensor has ``steps + 1`` points.  The last point is forced to
    exactly ``eps`` so no fixed-step sampler can silently integrate past the
    configured training cutoff ``t_epsilon``.
    """
    T = float(T)
    eps = float(eps)
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not (0.0 < eps < T):
        raise ValueError(f"Need 0 < eps < T, got eps={eps}, T={T}.")

    if schedule == "linear":
        grid = torch.linspace(T, eps, steps + 1, device=device, dtype=dtype)
    elif schedule == "quadratic":
        grid = torch.linspace(
            math.sqrt(T), math.sqrt(eps), steps + 1, device=device, dtype=dtype
        ).square()
    elif schedule == "log":
        grid = torch.logspace(
            math.log10(T), math.log10(eps), steps + 1, device=device, dtype=dtype
        )
    else:
        raise ValueError("schedule must be linear, quadratic, or log")

    # Make the integration interval explicit and exact.
    grid[0] = torch.as_tensor(T, device=device, dtype=dtype)
    grid[-1] = torch.as_tensor(eps, device=device, dtype=dtype)
    if not torch.all(grid[:-1] > grid[1:]):
        raise RuntimeError("Reverse time grid must be strictly decreasing.")
    return grid


def _score_flat(score_fn, x, y, t):
    s = score_fn(x, y, t)
    return s.reshape(x.shape[0], -1)


def _reverse_drift(sde, score_fn, x, y, t, probability_flow: bool):
    """Reverse drift with a structured fast path for the mixing SDE."""
    score = _score_flat(score_fn, x, y, t)

    if hasattr(sde, "apply_Ft") and hasattr(sde, "apply_diffusion_covariance"):
        forward_drift = sde.apply_Ft(x, t)
        covariance_score = sde.apply_diffusion_covariance(score, t)
        B = None if probability_flow else sde.sde(x, y, t)[1]
    else:
        forward_drift, B = sde.sde(x, y, t)
        tmp = torch.einsum("bij,bj->bi", B.conj().transpose(1, 2), score)
        covariance_score = torch.einsum("bij,bj->bi", B, tmp)

    coefficient = 0.5 if probability_flow else 1.0
    return forward_drift - coefficient * covariance_score, B


def _ode_drift(sde, score_fn, x, y, t):
    return _reverse_drift(sde, score_fn, x, y, t, probability_flow=True)[0]


@torch.no_grad()
def sample_fixed(
    sde,
    score_fn,
    y,
    *,
    method="euler",
    steps=50,
    eps=0.005,
    schedule="linear",
    snr=0.1,
    corrector_steps=1,
    x_T=None,
    generator=None,
):
    """Sample from the conditional reverse process.

    Methods: ``euler``, ``heun``, ``rk4``, ``em`` (Euler-Maruyama) and ``pc``.
    NFE counts neural score evaluations, which makes equal-NFE comparisons easy.
    """
    method = method.lower()
    if method not in {"euler", "heun", "rk4", "em", "pc"}:
        raise ValueError(f"Unknown sampler {method!r}")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not (0.0 < float(eps) < float(sde.T)):
        raise ValueError(f"Need 0 < eps < T, got eps={eps}, T={sde.T}.")
    if x_T is None:
        x = sde.prior_sampling(y.shape, y).to(y.device)
    else:
        x = x_T.clone().to(y.device)
    grid = _time_grid(sde.T, eps, steps, y.device, y.real.dtype, schedule)
    nfe = 0
    x_mean = x

    for i in range(steps):
        t0, t1 = grid[i], grid[i + 1]
        dt = t1 - t0  # negative: backward integration
        vec0 = t0.expand(y.shape[0])

        if method == "euler":
            k1 = _ode_drift(sde, score_fn, x, y, vec0); nfe += 1
            x = x + dt * k1
            x_mean = x

        elif method == "heun":
            k1 = _ode_drift(sde, score_fn, x, y, vec0); nfe += 1
            proposal = x + dt * k1
            vec1 = t1.expand(y.shape[0])
            k2 = _ode_drift(sde, score_fn, proposal, y, vec1); nfe += 1
            x = x + 0.5 * dt * (k1 + k2)
            x_mean = x

        elif method == "rk4":
            half = t0 + 0.5 * dt
            vech = half.expand(y.shape[0])
            vec1 = t1.expand(y.shape[0])
            k1 = _ode_drift(sde, score_fn, x, y, vec0)
            k2 = _ode_drift(sde, score_fn, x + 0.5 * dt * k1, y, vech)
            k3 = _ode_drift(sde, score_fn, x + 0.5 * dt * k2, y, vech)
            k4 = _ode_drift(sde, score_fn, x + dt * k3, y, vec1)
            nfe += 4
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            x_mean = x

        else:
            # Optional Langevin corrector at fixed t before the reverse-SDE predictor.
            if method == "pc":
                for _ in range(int(corrector_steps)):
                    grad = _score_flat(score_fn, x, y, vec0); nfe += 1
                    noise = _noise_like(x, sde, generator)
                    grad_norm = grad.abs().square().sum(dim=1).sqrt().clamp_min(1e-12)
                    noise_norm = noise.abs().square().sum(dim=1).sqrt().clamp_min(1e-12)
                    step_size = 2.0 * (float(snr) * noise_norm / grad_norm).square()
                    x_mean = x + step_size[:, None] * grad
                    x = x_mean + torch.sqrt(2.0 * step_size)[:, None] * noise

            drift, B = _reverse_drift(sde, score_fn, x, y, vec0, probability_flow=False); nfe += 1
            x_mean = x + dt * drift
            z = _noise_like(x, sde, generator)
            diffusion_noise = torch.einsum("bij,bj->bi", B, z)
            x = x_mean + torch.sqrt((-dt).clamp_min(0)) * diffusion_noise

    # The state now lives at grid[-1] == eps.  There is deliberately NO
    # post-processing/denoising step after eps: the reverse process must stop at
    # the same t_epsilon used as the lower bound during training.
    final_time = float(grid[-1].detach().cpu())
    if abs(final_time - float(eps)) > 1e-7:
        raise RuntimeError(
            f"Reverse sampler endpoint mismatch: final_time={final_time}, eps={eps}."
        )

    return SamplerResult(
        sample=x_mean if method in {"em", "pc"} else x,
        nfe=nfe,
        steps=steps,
        method=method,
        final_time=final_time,
    )
