"""Hydra/Lightning training entry point.

This remains compatible with the original project layout while adding NPZ data,
explicit EMA evaluation, safer checkpointing, and physics-conditioned backbones.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from pytorch_lightning import loggers as pl_loggers

import utils
from src.dataset import MixData_Module
from src.pl_model import DCSModel

log = logging.getLogger(__name__)


def load_filters(path: str | Path):
    """Load legacy object-array .npy filters or rectangular filters.npz."""
    path = Path(to_absolute_path(str(path)))
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if "filters" not in arr.files:
            raise KeyError(f"{path} does not contain a 'filters' array")
        filters = arr["filters"]
        lengths = arr["filter_lengths"] if "filter_lengths" in arr.files else None
        if lengths is None:
            return [np.asarray(x, np.float32) for x in filters]
        return [np.asarray(filters[i, : int(lengths[i])], np.float32) for i in range(len(lengths))]
    return [np.asarray(x, np.float32) for x in arr]


def infer_n_fft(cfg, filters):
    """Prefer dataset metadata; fall back to the original signal+filter rule."""
    root = Path(to_absolute_path(str(cfg.path.datasets.mixdata)))
    split_npz = root / "tr.npz"
    if split_npz.exists():
        with np.load(split_npz, allow_pickle=False) as data:
            if "n_fft" in data.files:
                return int(data["n_fft"])
    return int(cfg.model.sampling_rate + max(len(h) for h in filters) - 1)


def load_model(cfg):
    ckpt = getattr(cfg, "load_pretrained", None)
    if ckpt is None:
        return DCSModel(cfg), False
    ckpt_path = Path(to_absolute_path(str(ckpt)))
    log.info("Loading pretrained model from %s", ckpt_path)
    if ckpt_path.suffix == ".ckpt":
        return DCSModel.load_from_checkpoint(str(ckpt_path), strict=True, config=cfg), True
    model = DCSModel(cfg)
    payload = torch.load(str(ckpt_path), map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    return model, True


def build_callbacks(cfg):
    callbacks = [pl.callbacks.RichProgressBar()]
    metric = str(cfg.model.main_val_loss)
    metric_name = metric.split("/")[-1]
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            monitor=metric,
            save_top_k=int(getattr(cfg, "save_top_k", 3)),
            save_last=True,
            mode=str(cfg.model.main_val_loss_mode),
            filename=f"epoch-{{epoch:03d}}_{metric_name}-{{{metric}:.3f}}",
            auto_insert_metric_name=False,
        )
    )
    callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))
    patience = getattr(cfg, "early_stopping_patience", None)
    if patience is not None:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=metric,
                patience=int(patience),
                mode=str(cfg.model.main_val_loss_mode),
                check_finite=True,
            )
        )
    return callbacks


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg):
    if utils.ddp.is_rank_zero():
        log.info("Start experiment: %s", HydraConfig().get().run.dir)
    else:
        os.chdir(hydra.utils.get_original_cwd())

    pl.seed_everything(int(cfg.seed), workers=True)
    if bool(getattr(cfg, "detect_anomaly", False)):
        torch.autograd.set_detect_anomaly(True)

    filters = load_filters(cfg.path.datasets.filters_path)
    if len(filters) != int(cfg.model.num_sources):
        raise ValueError(
            f"Configured K={cfg.model.num_sources}, but {len(filters)} filters were loaded"
        )
    n_fft = infer_n_fft(cfg, filters)
    n_bins = n_fft // 2 + 1

    model, _ = load_model(cfg)
    kernels = torch.stack([
        torch.fft.rfft(torch.as_tensor(h, dtype=torch.float32), n=n_fft)
        for h in filters
    ]).to(torch.complex64)
    model.sde.get_mix_mat(kernels, n_bins)
    model.sde.real_n_fft = n_fft
    # The legacy backbone ignores this; the enhanced backbone uses it for
    # normalized measurement-residual features.
    if hasattr(model.score_model, "set_operator"):
        model.score_model.set_operator(kernels, n_fft=n_fft)

    dm = MixData_Module(cfg, num_fft_bins=n_fft, norm=model.norm)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=".", name="", version="")
    trainer = instantiate(
        cfg.trainer,
        callbacks=build_callbacks(cfg),
        logger=tb_logger,
    )

    if bool(cfg.train):
        ckpt_path = getattr(cfg, "resume_from_checkpoint", None)
        trainer.fit(
            model,
            dm,
            ckpt_path=to_absolute_path(str(ckpt_path)) if ckpt_path else None,
        )

    if bool(cfg.test):
        # When training happened in this run, "best" resolves through the
        # ModelCheckpoint callback.  Otherwise Lightning uses current weights.
        best = "best" if bool(cfg.train) else None
        trainer.test(model, dm, ckpt_path=best)


if __name__ == "__main__":
    main()
