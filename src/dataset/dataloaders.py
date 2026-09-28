"""Dataset and LightningDataModule utilities.

Backward compatible with the original WAV tree while adding an NPZ backend.

Expected legacy WAV layout::

    root/
      tr/{mix,s1,s2,...}/*.wav
      cv/{mix,s1,s2,...}/*.wav
      tt/{mix,s1,s2,...}/*.wav

Expected NPZ layout::

    root/
      tr.npz
      cv.npz
      tt.npz

Each NPZ split may contain time-domain arrays
``sources`` [N,K,T] and ``mixtures`` [N,Tmix].  For fastest training it may
also contain precomputed complex arrays ``sources_fft`` [N,K,F] and
``mixtures_fft`` [N,F].  The converter in ``data/convert_wav_dataset_to_npz.py``
creates this format.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

log = logging.getLogger(__name__)

split_map = {"test": "tt", "val": "cv", "train": "tr"}


def _as_complex_tensor(x: np.ndarray) -> torch.Tensor:
    """Convert numpy complex array without an unnecessary dtype upcast."""
    if x.dtype == np.complex128:
        x = x.astype(np.complex64, copy=False)
    return torch.from_numpy(np.asarray(x))


class MixData(torch.utils.data.Dataset):
    """Source-separation dataset supporting legacy WAV and compact NPZ storage.

    Parameters are intentionally compatible with the original project.  Set
    ``storage='auto'`` to prefer ``<split>.npz`` when it exists and otherwise
    fall back to the WAV tree.  ``use_cached_fft=True`` uses cached FFT arrays
    when present and compatible with the requested FFT length/normalization.
    """

    def __init__(
        self,
        path: Union[str, Path],
        num_sources: Optional[int] = 2,
        sampling_rate: Optional[int] = 16000,
        split: Optional[str] = "train",
        max_length_seconds: Optional[float] = None,
        max_num_samples: Optional[int] = None,
        mix_dir: Optional[str] = "mix",
        num_fft_bins: Optional[int] = None,
        norm: Optional[str] = None,
        storage: str = "auto",
        use_cached_fft: bool = True,
    ):
        super().__init__()
        if split not in split_map:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        if storage not in {"auto", "wav", "npz"}:
            raise ValueError("storage must be one of: auto, wav, npz")

        self.base_folder = Path(to_absolute_path(str(path)))
        self.split = split
        self.split_code = split_map[split]
        self.num_sources = int(num_sources) if num_sources is not None else None
        self.sampling_rate = int(sampling_rate)
        self.max_len = (
            int(self.sampling_rate * max_length_seconds)
            if max_length_seconds is not None
            else None
        )
        self.Nf = int(num_fft_bins) if num_fft_bins is not None else None
        self.norm = norm
        self.use_cached_fft = bool(use_cached_fft)

        npz_path = self.base_folder / f"{self.split_code}.npz"
        if storage == "auto":
            storage = "npz" if npz_path.exists() else "wav"
        self.storage = storage

        if self.storage == "npz":
            if not npz_path.exists():
                raise FileNotFoundError(f"NPZ split not found: {npz_path}")

            # IMPORTANT: never keep an open NpzFile/ZipFile handle on the Dataset.
            # DataLoader workers are normally forked on Linux; an already-open ZIP
            # file descriptor is then inherited by all workers and concurrent reads
            # can corrupt the DEFLATE stream (zlib.error).  Load the arrays once in
            # the parent process, close the archive, and let forked workers share the
            # read-only pages via copy-on-write.  The paper datasets are small enough
            # that this is also considerably faster than repeatedly decompressing a
            # complete NPZ member for every sample.
            self.npz_path = npz_path
            self.sources = None
            self.mixtures = None
            self.sources_fft = None
            self.mixtures_fft = None

            with np.load(npz_path, allow_pickle=False) as archive:
                keys = set(archive.files)
                required = {"sources", "mixtures"}
                cached = {"sources_fft", "mixtures_fft"}
                if not required.issubset(keys) and not cached.issubset(keys):
                    raise KeyError(
                        f"{npz_path} must contain either {sorted(required)} or cached FFT arrays"
                    )

                archive_nfft = int(archive["n_fft"]) if "n_fft" in keys else None
                archive_norm = (
                    str(archive["fft_norm"].item()) if "fft_norm" in keys else None
                )
                self._cached_fft_ok = (
                    self.use_cached_fft
                    and cached.issubset(keys)
                    and (self.Nf is None or archive_nfft == self.Nf)
                    and (archive_norm is None or archive_norm == str(self.norm))
                )

                if self._cached_fft_ok:
                    self.sources_fft = np.asarray(
                        archive["sources_fft"], dtype=np.complex64
                    )
                    self.mixtures_fft = np.asarray(
                        archive["mixtures_fft"], dtype=np.complex64
                    )
                    probe = self.sources_fft
                else:
                    if not required.issubset(keys):
                        raise KeyError(
                            f"Cached FFTs in {npz_path} are incompatible with the requested "
                            "FFT settings and no time-domain sources/mixtures are available."
                        )
                    self.sources = np.asarray(archive["sources"], dtype=np.float32)
                    self.mixtures = np.asarray(archive["mixtures"], dtype=np.float32)
                    probe = self.sources

            if probe.ndim != 3:
                raise ValueError("sources/sources_fft must have shape [N,K,T/F]")
            inferred_k = int(probe.shape[1])
            if self.num_sources is None:
                self.num_sources = inferred_k
            elif inferred_k != self.num_sources:
                raise ValueError(
                    f"Dataset contains K={inferred_k}, config requests K={self.num_sources}"
                )
            self.length = int(probe.shape[0])
            if max_num_samples is not None:
                self.length = min(self.length, int(max_num_samples))
            return

        # Legacy WAV backend -------------------------------------------------
        self.path = self.base_folder / self.split_code
        self.path_mix = self.path / str(mix_dir)
        if not self.path_mix.exists():
            raise FileNotFoundError(f"Mixture directory not found: {self.path_mix}")
        self.path_src = [self.path / f"s{i+1}" for i in range(self.num_sources)]
        self.file_list = sorted(
            f for f in os.listdir(self.path_mix) if not f.startswith(".")
        )
        if max_num_samples is not None:
            self.file_list = self.file_list[: int(max_num_samples)]

    def __len__(self):
        return self.length if self.storage == "npz" else len(self.file_list)

    def _fft(self, sources: torch.Tensor, mixture: torch.Tensor):
        if self.Nf is None:
            raise ValueError("num_fft_bins/n_fft must be supplied by the DataModule")
        sources_fft = torch.fft.rfft(sources, n=self.Nf, dim=-1, norm=self.norm)
        mixture_fft = torch.fft.rfft(mixture, n=self.Nf, dim=-1, norm=self.norm)
        if mixture_fft.ndim == 2 and mixture_fft.shape[0] == 1:
            mixture_fft = mixture_fft.squeeze(0)
        return sources_fft.to(torch.complex64), mixture_fft.to(torch.complex64)

    def _get_npz(self, idx: int):
        if self._cached_fft_ok:
            x = _as_complex_tensor(self.sources_fft[idx]).to(torch.complex64)
            y = _as_complex_tensor(self.mixtures_fft[idx]).to(torch.complex64)
            return x, y

        sources = torch.from_numpy(self.sources[idx])
        mixture = torch.from_numpy(self.mixtures[idx])
        return self._fft(sources, mixture)

    def _get_wav(self, idx: int):
        filename = self.file_list[idx]
        mix, mix_sr = torchaudio.load(self.path_mix / filename)
        if int(mix_sr) != self.sampling_rate:
            raise ValueError(
                f"{filename}: sample rate {mix_sr} != configured {self.sampling_rate}"
            )
        target_parts = []
        for path in self.path_src:
            source, sr = torchaudio.load(path / filename)
            if int(sr) != self.sampling_rate:
                raise ValueError(
                    f"{path / filename}: sample rate {sr} != configured {self.sampling_rate}"
                )
            target_parts.append(source)
        tgt = torch.cat(target_parts, dim=0)

        if self.max_len is not None and tgt.shape[-1] > self.max_len:
            max_start = tgt.shape[-1] - self.max_len
            p = int(torch.randint(0, max_start + 1, size=(1,)).item())
            tgt = tgt[..., p : p + self.max_len]
            mix = mix[..., p : p + self.max_len]
        return self._fft(tgt, mix)

    def __getitem__(self, idx):
        return self._get_npz(idx) if self.storage == "npz" else self._get_wav(idx)


def max_collator(batch):
    """Pad frequency tensors in a batch to the longest frequency dimension."""
    max_len = max(row[0].shape[-1] for row in batch)
    new_batch = []
    for row in batch:
        new_row = []
        for el in row:
            if isinstance(el, torch.Tensor):
                off = max_len - el.shape[-1]
                new_row.append(torch.nn.functional.pad(el, (off // 2, off - off // 2)))
        new_batch.append(tuple(new_row))
    return torch.utils.data.default_collate(new_batch)


class MixData_Module(pl.LightningDataModule):
    def __init__(
        self,
        config: DictConfig,
        num_fft_bins: Optional[int] = None,
        norm: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = config
        self.num_fft_bins = num_fft_bins
        self.norm = norm
        self.datasets = {}

    def setup(self, stage=None):
        for split in self.cfg.datamodule:
            self.datasets[split] = instantiate(
                self.cfg.datamodule[split].dataset,
                num_fft_bins=self.num_fft_bins,
                norm=self.norm,
            )

    def _get_dataloader(self, split):
        opts = dict(self.cfg.datamodule[split].dl_opts)
        # Sensible loader defaults while retaining Hydra overrides.
        workers = int(opts.get("num_workers", 0))
        opts.setdefault("pin_memory", torch.cuda.is_available())
        if workers > 0:
            opts.setdefault("persistent_workers", True)
            opts.setdefault("prefetch_factor", 2)
        return torch.utils.data.DataLoader(
            self.datasets[split], collate_fn=max_collator, **opts
        )

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")
