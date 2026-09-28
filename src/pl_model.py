"""
Score model Class
Taken and adapted from https://github.com/fakufaku/diffusion-separation
"""

from torch.profiler import record_function
import matplotlib.pyplot as plt
import datetime
import itertools
import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from unittest import skip
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio as si_sdr_fn,
)

try:
    import fast_bss_eval
except ImportError:
    fast_bss_eval = None
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.utils import instantiate, to_absolute_path
from omegaconf.omegaconf import open_dict
from scipy.optimize import linear_sum_assignment

from src.ema import ExponentialMovingAverage
from src.sde.rfft_noise import standard_rfft_noise_like, project_rfft_endpoints
from src.score_parameterization import (
    ScoreParameterization,
    network_output_to_physical,
    network_output_to_whitened,
)

import src.sde.freqsde
import utils
from typing import Callable, Optional, Tuple

log = logging.getLogger(__name__)

Tensor = torch.Tensor
ScoreModel = Callable[[Tensor, Tensor, Tensor], Tensor]


class DCSModel(pl.LightningModule):
    """
    DCSModel: Diffusion-based Convolutional Source Separation Model.

    This module implements the training and inference pipeline for separating
    convolutive mixtures using a score-based generative model. It leverages
    a Variance-Exploding SDE (VE-SDE) defined in the frequency domain to
    model the distribution of source signals conditioned on the mixture.

    The training objective is Denoising Score Matching (DSM), where the model
    learns to predict the score function \(\nabla_{x_t} \log p_t(x_t | y)\).
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Score-based model (typically a UNet) that predicts the score function
        self.score_model = instantiate(self.config.model.score_model, _recursive_=False)
        self.num_sources = self.config.model.num_sources

        # Forward SDE defining the noise schedule and mixing matrix \(\mathcal{A}\)
        self.sde = instantiate(self.config.model.sde)

        self.t_eps = self.config.model.time_epsilon
        self.t_max = self.sde.T
        self.prob = config.model.prior_sampling_prob
        self.time_sampling_strategy = getattr(
            self.config.model, "time_sampling_strategy", "uniform"
        )
        self.score_parameterization = ScoreParameterization.parse(
            getattr(self.config.model, "score_parameterization", "physical")
        )
        log.info("Score parameterization: %s", self.score_parameterization.value)

        self.t_rev_init = getattr(self.config.model, "reverse_init_time", 0.03)
        log.info(f"Sampling time in [{self.t_eps, self.t_max}]")

        self.lr_warmup = getattr(config.model, "lr_warmup", None)
        self.lr_original = self.config.model.optimizer.lr

        self.train_source_order = getattr(
            self.config.model, "train_source_order", "random"
        )

        self.loss = instantiate(self.config.model.loss)

        # Exponential Moving Average (EMA) for model weights to improve stability
        self.ema_decay = float(getattr(self.config.model, "ema_decay", 0.999))
        self.ema_power = float(getattr(self.config.model, "ema_warmup_power", 0.75))
        self.ema_inv_gamma = float(getattr(self.config.model, "ema_inv_gamma", 1.0))
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._ema_applied = False
        self._error_loading_ema = False
        self.norm = self.config.model.normalization_type

    def separate(self, y, **kwargs):
        """Separate a batch with an explicitly configured reverse solver.

        New fixed-step methods: ``euler`` (probability-flow ODE), ``heun``,
        ``rk4``, ``em`` (reverse SDE Euler-Maruyama), and ``pc``.  Set
        ``method=legacy`` to retain the original predictor/corrector path.
        """
        sampler_kwargs = dict(self.config.model.sampler)
        sampler_kwargs.update(kwargs)
        method = str(sampler_kwargs.pop("method", "euler")).lower()
        steps = int(sampler_kwargs.pop("num_steps", sampler_kwargs.pop("steps", self.sde.N)))
        if method == "legacy":
            sampler = self.get_pc_sampler(
                "reverse_diffusion", "none", y, N=steps, **sampler_kwargs
            )
            sample, nfe = sampler()
            # Legacy grid is also constructed with N+1 points ending exactly
            # at self.t_eps; no post-eps denoise step is taken.
            self._last_sampling_final_time = float(self.t_eps)
            return sample, nfe

        from src.sde.fixed_samplers import sample_fixed
        requested_eps = float(sampler_kwargs.pop("eps", self.t_eps))
        if abs(requested_eps - float(self.t_eps)) > 1e-12:
            raise ValueError(
                f"Reverse endpoint is fixed to model.time_epsilon={self.t_eps}; "
                f"received sampler eps={requested_eps}."
            )
        result = sample_fixed(
            self.sde.copy(), self, y, method=method, steps=steps,
            eps=float(self.t_eps),
            schedule=str(sampler_kwargs.pop("schedule", "linear")),
            snr=float(sampler_kwargs.pop("signal_to_noise_ratio", sampler_kwargs.pop("snr", 0.1))),
            corrector_steps=int(sampler_kwargs.pop("num_corrector_steps", sampler_kwargs.pop("corrector_steps", 1))),
            **sampler_kwargs,
        )
        self._last_sampling_final_time = float(result.final_time)
        if abs(self._last_sampling_final_time - float(self.t_eps)) > 1e-7:
            raise RuntimeError(
                f"Reverse sampler stopped at {self._last_sampling_final_time}, "
                f"expected t_epsilon={self.t_eps}."
            )
        return result.sample, result.nfe

    def sample_time(self, x):
        """Sample diffusion times used by the DSM branch.

        Supported strategies
        --------------------
        uniform:
            t ~ U[t_eps, T]. This is the original training distribution.

        log_uniform:
            log(t) ~ U[log(t_eps), log(T)]. This deliberately allocates much
            more training mass to the difficult low-noise region near t_eps.

        mixed:
            Per sample, draw from log-uniform with probability
            ``time_sampling_log_fraction`` and otherwise from uniform.
            This keeps broad coverage while substantially increasing the
            number of low-t examples.

        varprop:
            Retained from the original project.

        Note
        ----
        log_uniform and mixed intentionally change the time-weighting of the
        training objective. They are experimental strategies, not importance-
        corrected estimators of the original uniform-in-time objective.
        """
        B = x.shape[0]
        device = x.device
        dtype = x.real.dtype

        strategy = str(self.time_sampling_strategy).lower()

        if strategy == "uniform":
            return torch.empty(B, device=device, dtype=dtype).uniform_(
                float(self.t_eps), float(self.t_max)
            )

        if strategy == "log_uniform":
            if float(self.t_eps) <= 0.0:
                raise ValueError("log_uniform time sampling requires t_eps > 0.")

            u = torch.rand(B, device=device, dtype=dtype)
            log_min = math.log(float(self.t_eps))
            log_max = math.log(float(self.t_max))
            return torch.exp(
                u * (log_max - log_min) + log_min
            )

        if strategy == "mixed":
            if float(self.t_eps) <= 0.0:
                raise ValueError("mixed time sampling requires t_eps > 0.")

            frac = float(
                getattr(
                    self.config.model,
                    "time_sampling_log_fraction",
                    0.5,
                )
            )
            if not 0.0 <= frac <= 1.0:
                raise ValueError(
                    "model.time_sampling_log_fraction must lie in [0,1], "
                    f"got {frac}."
                )

            u_uniform = torch.rand(B, device=device, dtype=dtype)
            t_uniform = (
                float(self.t_eps)
                + u_uniform * (float(self.t_max) - float(self.t_eps))
            )

            u_log = torch.rand(B, device=device, dtype=dtype)
            log_min = math.log(float(self.t_eps))
            log_max = math.log(float(self.t_max))
            t_log = torch.exp(
                u_log * (log_max - log_min) + log_min
            )

            choose_log = torch.rand(B, device=device) < frac
            return torch.where(choose_log, t_log, t_uniform)

        if strategy == "varprop":
            return self.sde.sample_time_varprop(
                B,
                t_eps=self.t_eps,
                device=device,
            ).to(dtype=dtype)

        raise NotImplementedError(
            f"No sampling strategy '{self.time_sampling_strategy}'. "
            "Supported: uniform, log_uniform, mixed, varprop."
        )

    def sample_prior(self, x, y):
        """Sample ``x_t`` using the structured covariance factor.

        Returns
        -------
        xt : [B,D]
            Perturbed state.
        time : [B]
            Sampled diffusion time.
        std_modes : [B,D]
            Diagonal modal factor in ``S_t = V diag(std_modes)``.
        z : [B,D]
            Modal standard-complex perturbation coordinate.  On terminal
            mismatch examples it includes the analytical mean-shift term, so
            the DSM target remains exactly ``-z``.
        """
        time = self.sample_time(x).to(dtype=x.real.dtype)
        terminal_mask = torch.rand_like(time) < float(self.prob)
        if terminal_mask.any():
            time = torch.where(
                terminal_mask,
                torch.full_like(time, float(self.sde.T)),
                time,
            )

        self.sde.send_to(x.device)
        mean = self.sde.apply_exp_Ft(x, time)
        std_modes = self.sde.modal_std(time)

        z = standard_rfft_noise_like(
            x,
            num_sources=self.num_sources,
            n_fft=getattr(self.score_model, "n_fft", None),
        )

        if terminal_mask.any():
            y_rep = y.repeat(1, self.num_sources)
            shift = self.sde.solve_S(
                y_rep - mean,
                time,
                std_modes=std_modes,
            )
            z = torch.where(terminal_mask[:, None], z + shift, z)
            z = project_rfft_endpoints(
                z,
                num_sources=self.num_sources,
                n_fft=getattr(self.score_model, "n_fft", None),
            )

        xt = mean + self.sde.apply_S(z, time, std_modes=std_modes)
        return xt, time, std_modes, z

    def predict_network_output(self, xt, y, t):
        """Return the raw neural-network output in its configured parameterization."""
        return self.score_model(xt, y, t)

    def forward(self, xt, y, t):
        """Return the *physical score* required by reverse-time dynamics."""
        raw = self.predict_network_output(xt, y, t)
        return network_output_to_physical(
            raw,
            sde=self.sde,
            t=t,
            parameterization=self.score_parameterization,
        )

    @staticmethod
    def _complex_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean squared modulus, equal to real-MSE + imag-MSE."""
        return (pred - target).abs().square().mean()

    def compute_score_loss(self, x, y):
        """Compute the covariance-weighted DSM objective.

        ``physical`` network:
            network -> s_theta, then analytically form S_t^H s_theta.

        ``whitened`` network:
            network -> u_theta = S_t^H s_theta directly.

        Both optimize the *same* target ``-z`` and therefore the same weighted
        DSM objective; only the neural output parameterization changes.
        """
        if y.ndim == 3:
            y = y.squeeze(1)

        xt, time, std_modes, z = self.sample_prior(x, y)
        self._last_sampled_times = time.detach()

        raw = self.predict_network_output(xt, y, time)
        pred_whitened = network_output_to_whitened(
            raw,
            sde=self.sde,
            t=time,
            parameterization=self.score_parameterization,
            std_modes=std_modes,
        )

        if pred_whitened.shape != z.shape:
            raise RuntimeError(
                f"Whitened prediction shape {tuple(pred_whitened.shape)} "
                f"does not match target {tuple(z.shape)}."
            )
        if not torch.isfinite(pred_whitened.real).all() or not torch.isfinite(pred_whitened.imag).all():
            raise RuntimeError("Network produced non-finite whitened score values.")

        return self._complex_mse(pred_whitened, -z)

    def on_train_epoch_start(self):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        B, K, Nfft = x.shape
        x = x.view(B, K * Nfft)
        y = y.view(B, Nfft)
        loss = self.compute_score_loss(x, y)

        cur_step = self.trainer.global_step
        self.last_step = getattr(self, "last_step", 0)

        # Optional diagnostic; disabled by default to avoid a GPU->CPU sync every step.
        is_spike = False
        if bool(getattr(self.config.model, "debug_loss_spikes", False)):
            self.loss_history = getattr(self, "loss_history", [])
            loss_scalar = float(loss.detach().cpu())
            self.loss_history.append(loss_scalar)
            window = int(getattr(self.config.model, "spike_window", 50))
            multiplier = float(getattr(self.config.model, "spike_multiplier", 2.0))
            if len(self.loss_history) >= window:
                reference = sum(self.loss_history[-window:]) / window
                is_spike = loss_scalar > reference * multiplier

        self.log("train/score_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        sampled_t = getattr(self, "_last_sampled_times", None)
        if sampled_t is not None:
            self.log(
                "train/time_mean",
                sampled_t.mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/time_frac_lt_0p05",
                (sampled_t < 0.05).float().mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/time_frac_lt_0p02",
                (sampled_t < 0.02).float().mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        if cur_step > self.last_step and cur_step % 10 == 0:
            self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True, prog_bar=False, sync_dist=False)
            self.do_lr_warmup()

        if is_spike:
            print(f"⚠️ Spike at step {cur_step}: loss={float(loss.detach().cpu()):.4f}")
            self.logger.log_metrics({"debug/spike_loss": float(loss.detach().cpu())}, step=cur_step)

            # Log input statistics
            self.logger.log_metrics(
                {
                    "debug/x_norm": x.norm(dim=-1).mean().item(),
                    "debug/x_max": x.abs().max().item(),
                    "debug/y_norm": y.norm(dim=-1).mean().item(),
                    "debug/y_max": y.abs().max().item(),
                },
                step=cur_step,
            )

        return loss

    def on_train_epoch_end(self):
        pass

    def _apply_ema_weights(self):
        if not self._ema_applied and not self._error_loading_ema:
            self.ema.store(self.parameters())
            self.ema.copy_to(self.parameters())
            self._ema_applied = True

    def _restore_online_weights(self):
        if self._ema_applied:
            self.ema.restore(self.parameters())
            self._ema_applied = False

    def on_validation_epoch_start(self):
        self.n_batches_est_done = 0
        self._apply_ema_weights()

    def validation_step(self, batch, batch_idx, dataset_i=0):
        x, y = batch
        B, K, Nfft = x.shape
        x = x.view(B, K * Nfft)
        y = y.view(B, Nfft)
        loss = self.compute_score_loss(x, y)
        self.log("val/score_loss", loss, on_epoch=True, sync_dist=True)
        with torch.no_grad():
            res, res_ns = self.separate(y)
        res_time = torch.fft.irfft(res.view(B, K, Nfft), norm=self.norm)
        x_time = torch.fft.irfft(x.view(B, K, Nfft), norm=self.norm)
        loss_rec = si_sdr_fn(res_time, x_time)
        if loss_rec.ndim == 2:
            loss_rec = loss_rec.mean()
        # loss_rec = self.loss(res.real, x.real) + self.loss(res.imag, x.imag)

        self.log("val/loss_rec", loss_rec, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self._restore_online_weights()

    def on_test_epoch_start(self):
        self._apply_ema_weights()

    def test_step(self, batch, batch_idx, dataset_i=None):
        return self.validation_step(batch, batch_idx, dataset_i=dataset_i)

    def on_test_epoch_end(self):
        self._restore_online_weights()

    def configure_optimizers(self):
        # we may have some frozen layers, so we remove these parameters
        # from the optimization
        log.info(f"set optim with {self.config.model.optimizer}")

        opt_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = instantiate(
            {**{"params": opt_params}, **self.config.model.optimizer}
        )

        if getattr(self.config.model, "scheduler", None) is not None:
            scheduler = instantiate(
                {**self.config.model.scheduler, **{"optimizer": optimizer}}
            )
        else:
            scheduler = None

        # Gradient clipping is handled by Lightning Trainer after gradient
        # accumulation, which avoids clipping individual micro-batches.
        self.grad_clipper = None

        if scheduler is None:
            return [optimizer]
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.config.model.main_val_loss,
            }

    def optimizer_step(self, *args, **kwargs):
        # EMA is updated exactly once per optimizer step (not once per micro-batch).
        super().optimizer_step(*args, **kwargs)
        j = self.ema.num_updates + 1
        warm_decay = 1.0 - (1.0 + j / max(self.ema_inv_gamma, 1e-8)) ** (-self.ema_power)
        decay = min(self.ema_decay, max(0.0, warm_decay))
        self.ema.update(self.parameters(), decay=decay)

    def on_after_backward(self):
        # Logging only. Actual clipping is performed by Lightning after any
        # configured gradient accumulation.
        if self.trainer.global_step % 25 == 0:
            grad_norm = utils.grad_norm(self)
            opt = self.trainer.optimizers[0]
            current_lr = opt.param_groups[0]["lr"]
            self.log("grad/norm", grad_norm, on_step=True, prog_bar=False, sync_dist=False)
            self.log("grad/step_size", current_lr * grad_norm, on_step=True, prog_bar=False, sync_dist=False)

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get("ema", None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self._error_loading_ema = True
            log.warning("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def do_lr_warmup(self):
        if self.lr_warmup is not None and self.trainer.global_step < self.lr_warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.lr_warmup)
            optimizer = self.trainer.optimizers[0]
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr_original

    def get_pc_sampler(
        self,
        predictor_name,
        corrector_name,
        y,
        N=None,
        minibatch=None,
        schedule=None,
        **kwargs,
    ):
        import src.sde as sde_pkg

        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        if "eps" in kwargs and abs(float(kwargs["eps"]) - float(self.t_eps)) > 1e-12:
            raise ValueError(
                f"Reverse endpoint is fixed to model.time_epsilon={self.t_eps}; "
                f"received eps={kwargs['eps']}."
            )
        kwargs = {**kwargs, "eps": float(self.t_eps)}
        if minibatch is None:
            if schedule is None:
                return sde_pkg.get_pc_sampler(
                    predictor_name,
                    corrector_name,
                    sde=sde,
                    score_fn=self,
                    y=y,
                    **kwargs,
                )
            else:
                return sde_pkg.get_pc_scheduled_sampler(
                    predictor_name,
                    corrector_name,
                    sde=sde,
                    score_fn=self,
                    y=y,
                    schedule=schedule,
                    **kwargs,
                )
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns, intmet = [], [], []
                for i in range(int(math.ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    if schedule is None:
                        sampler = sde.get_pc_sampler(
                            predictor_name,
                            corrector_name,
                            sde=sde,
                            score_fn=self,
                            y=y_mini,
                            **kwargs,
                        )
                    else:
                        sampler = sde.get_pc_scheduled_sampler(
                            predictor_name,
                            corrector_name,
                            sde=sde,
                            score_fn=self,
                            y=y_mini,
                            schedule=schedule,
                            **kwargs,
                        )
                    sample, n, *other = sampler()
                    samples.append(sample)
                    ns.append(n)
                    if len(other) > 0:
                        intmet.append(other[0])
                samples = torch.cat(samples, dim=0)
                if len(intmet) > 0:
                    return samples, ns, intmet
                else:
                    return samples, ns

            return batched_sampling_fn
