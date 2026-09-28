"""Deterministic held-out evaluation with both latent-source and source-image metrics."""
from __future__ import annotations
import json
from pathlib import Path

import hydra
import numpy as np
import torch
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

from src.dataset import MixData_Module
from src.pl_model import DCSModel
from train import load_filters, infer_n_fft


def normalized_mse(target, estimate, eps=1e-12):
    num = (estimate - target).abs().square().sum(dim=-1)
    den = target.abs().square().sum(dim=-1).clamp_min(eps)
    return num / den


def normalized_mae(target, estimate, eps=1e-12):
    num = (estimate - target).abs().mean(dim=-1)
    den = target.abs().mean(dim=-1).clamp_min(eps)
    return num / den


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg):
    if cfg.checkpoint is None:
        raise ValueError("Set checkpoint=/path/to/model.ckpt")
    pl.seed_everything(int(cfg.seed), workers=True)
    filters = load_filters(cfg.path.datasets.filters_path)
    n_fft = infer_n_fft(cfg, filters)
    n_bins = n_fft // 2 + 1
    kernels = torch.stack([
        torch.fft.rfft(torch.as_tensor(h, dtype=torch.float32), n=n_fft)
        for h in filters
    ]).to(torch.complex64)

    model = DCSModel.load_from_checkpoint(
        to_absolute_path(str(cfg.checkpoint)),
        strict=True,
        config=cfg,
        map_location="cpu",
        # PyTorch >= 2.6 defaults torch.load(..., weights_only=True).
        # Lightning checkpoints contain OmegaConf objects (e.g. DictConfig),
        # so trusted checkpoints created by this project must be loaded with
        # weights_only=False.
        weights_only=False,
    )
    model.sde.get_mix_mat(kernels, n_bins)
    model.sde.real_n_fft = n_fft
    model.score_model.set_operator(kernels, n_fft=n_fft)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.sde.send_to(device)
    model.eval()
    model._apply_ema_weights()

    dm = MixData_Module(cfg, num_fft_bins=n_fft, norm=model.norm)
    dm.setup("test")
    loader = dm.test_dataloader()

    rows = []
    all_latent_sisdr, all_image_sisdr = [], []
    all_nmse, all_nmae, all_resid = [], [], []
    total_nfe = 0
    count_batches = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device); y = y.to(device)
            B, K, F = x.shape
            flat_y = y.reshape(B, F)
            estimate_flat, nfe = model.separate(flat_y)
            final_time = float(getattr(model, "_last_sampling_final_time", model.t_eps))
            if abs(final_time - float(model.t_eps)) > 1e-7:
                raise RuntimeError(
                    f"Reverse sampler endpoint {final_time} != t_epsilon {model.t_eps}."
                )
            estimate = estimate_flat.reshape(B, K, F)
            total_nfe += int(nfe); count_batches += 1

            x_time = torch.fft.irfft(x, n=n_fft, dim=-1, norm=model.norm)
            e_time = torch.fft.irfft(estimate, n=n_fft, dim=-1, norm=model.norm)
            latent_sisdr = scale_invariant_signal_distortion_ratio(e_time, x_time)

            H = kernels.to(device).unsqueeze(0)
            x_img = x * H
            e_img = estimate * H
            x_img_time = torch.fft.irfft(x_img, n=n_fft, dim=-1, norm=model.norm)
            e_img_time = torch.fft.irfft(e_img, n=n_fft, dim=-1, norm=model.norm)
            image_sisdr = scale_invariant_signal_distortion_ratio(e_img_time, x_img_time)

            nmse = normalized_mse(x, estimate)
            nmae = normalized_mae(x, estimate)
            y_hat = (estimate * H).sum(dim=1)
            residual = (y_hat - flat_y).abs().square().sum(-1) / flat_y.abs().square().sum(-1).clamp_min(1e-12)

            all_latent_sisdr.append(latent_sisdr.cpu())
            all_image_sisdr.append(image_sisdr.cpu())
            all_nmse.append(nmse.cpu()); all_nmae.append(nmae.cpu()); all_resid.append(residual.cpu())

    latent = torch.cat(all_latent_sisdr).numpy()
    image = torch.cat(all_image_sisdr).numpy()
    nmse = torch.cat(all_nmse).numpy(); nmae = torch.cat(all_nmae).numpy(); resid = torch.cat(all_resid).numpy()
    mixture_mean = latent.mean(axis=1)
    summary = {
        "checkpoint": str(cfg.checkpoint),
        "sampler": dict(cfg.model.sampler),
        "num_examples": int(latent.shape[0]),
        "latent_si_sdr_db_mean": float(latent.mean()),
        "latent_si_sdr_db_per_source": [float(v) for v in latent.mean(axis=0)],
        "source_image_si_sdr_db_mean": float(image.mean()),
        "nmse_mean": float(nmse.mean()),
        "nmae_mean": float(nmae.mean()),
        "failure_rate_lt_5db": float(np.mean(mixture_mean < 5.0)),
        "measurement_residual_mean": float(resid.mean()),
        "mean_nfe": float(total_nfe / max(count_batches, 1)),
        "reverse_t_epsilon": float(model.t_eps),
        "reverse_final_time": float(getattr(model, "_last_sampling_final_time", model.t_eps)),
    }
    out = Path(to_absolute_path(str(cfg.evaluation_out)))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        out / "per_example.npz",
        latent_si_sdr_db=latent,
        source_image_si_sdr_db=image,
        nmse=nmse,
        nmae=nmae,
        measurement_residual=resid,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
