import abc

import numpy as np
import torch
from utils.registry import Registry
from .rfft_noise import standard_rfft_noise_like

PredictorRegistry = Registry("Predictor")


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn, probability_flow=probability_flow)
        self.score_fn = score_fn
        self.probability_flow = probability_flow

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """One update of the predictor.
        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes
        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

    def debug_update_fn(self, x, t, *args):
        raise NotImplementedError(
            f"Debug update function not implemented for predictor {self}."
        )


@PredictorRegistry.register("euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, t, *args, **kwargs):
        # args[0] is expected to be 'y' for the SDE
        y = args[0] if len(args) > 0 else None
        step = float(kwargs.get("dt", float(self.rsde.T) / float(self.rsde.N)))
        if step <= 0:
            raise ValueError(f"dt must be positive, got {step}.")
        dt = -step

        z = standard_rfft_noise_like(
            x,
            num_sources=int(getattr(self.sde, "feature_dim", 1)),
            n_fft=getattr(self.sde, "real_n_fft", None),
        )
        # SDE expects (x, y, t)
        f, g = self.rsde.sde(x, y, t)

        x_mean = x + f * dt

        # Handle diffusion matrix multiplication if g is a matrix
        if g.ndim == 3:  # (B, dim, dim)
            noise = torch.bmm(g, z.unsqueeze(-1)).squeeze(-1)
        else:
            noise = g * z

        x = x_mean + noise * np.sqrt(-dt)
        return x, x_mean


@PredictorRegistry.register("reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=True):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, t, *args, **kwargs):
        # args[0] is expected to be 'y'
        y = args[0] if len(args) > 0 else None

        # discretize returns the pre-computed increments: f = drift*dt, g = diffusion*sqrt(dt)
        step = float(kwargs.get("dt", float(self.rsde.T) / float(self.rsde.N)))
        if step <= 0:
            raise ValueError(f"dt must be positive, got {step}.")
        f, g = self.rsde.discretize(x, y, t, dt=step)

        z = standard_rfft_noise_like(
            x,
            num_sources=int(getattr(self.sde, "feature_dim", 1)),
            n_fft=getattr(self.sde, "real_n_fft", None),
        )
        x_mean = x + f

        # Handle diffusion matrix multiplication if g is a matrix (B, dim, dim)
        if g.ndim == 3:
            noise = torch.bmm(g, z.unsqueeze(-1)).squeeze(-1)
        else:
            noise = g * z

        x = x_mean + noise
        return x, x_mean


@PredictorRegistry.register("none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def update_fn(self, x, t, *args):
        return x, x
