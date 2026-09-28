#!/usr/bin/env python3
"""Generate publication-friendly NPZ datasets with unambiguous latent targets.

Unlike the legacy WAV generator, ``sources`` are the *latent clean sources* s_k.
The known FIRs are applied only to construct ``source_images = h_k * s_k`` and
``mixtures = sum_k source_images_k + noise``.  Full linear-convolution tails are
retained, eliminating the circular/linear-convolution ambiguity.

Examples
--------
Mechanical K=2::

    python data/create_dataset_npz.py --out data/mechanical_k2_npz --kind mechanical --sources 2

Stationary synthetic music K=3::

    python data/create_dataset_npz.py --out data/music_k3_npz --kind music --sources 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

# Allow execution as `python data/create_dataset_npz.py` from repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.gear_signal import (
    EpicyclicGearParams,
    FaultParams,
    GearParams,
    GearSignalGenerator,
    HarmonicParams,
    ModulationParams,
    SignalConfig,
)
from data.stationary_music import generate_stationary_music_source


def sample_fault(rng: np.random.Generator, probability: float) -> FaultParams:
    if rng.random() >= probability:
        return FaultParams(enabled=False)
    return FaultParams(
        enabled=True,
        impulse_amplitude=1.0,
        resonance_freq=200.0,
        damping_ratio=0.08,
    )


def mechanical_config(family: int, seed: int, fs: int, duration: float, fault_probability: float):
    rng = np.random.default_rng(seed)
    family = int(family) % 3
    fault = sample_fault(rng, fault_probability)
    if family == 0:
        return SignalConfig(
            duration=duration,
            sampling_rate=fs,
            gear=GearParams(teeth=int(rng.integers(12, 20)), speed_rpm=float(rng.uniform(200, 400))),
            harmonic=HarmonicParams(M=int(rng.integers(4, 7))),
            modulation=ModulationParams(K=int(rng.integers(2, 5))),
            fault=fault, noise_std=0.0, seed=seed,
        )
    if family == 1:
        return SignalConfig(
            duration=duration,
            sampling_rate=fs,
            gear=GearParams(teeth=int(rng.integers(15, 25)), speed_rpm=float(rng.uniform(500, 1000))),
            harmonic=HarmonicParams(M=int(rng.integers(2, 4))),
            modulation=ModulationParams(K=int(rng.integers(3, 6))),
            fault=fault, noise_std=0.0, seed=seed,
        )
    z_s = int(rng.integers(18, 25)); z_p = int(rng.integers(25, 36)); z_r = int(rng.integers(70, 91))
    return SignalConfig(
        duration=duration,
        sampling_rate=fs,
        gear=EpicyclicGearParams(
            speed_rpm=float(rng.uniform(30, 80)), teeth=z_s, N_p=3,
            Z_s=z_s, Z_p=z_p, Z_r=z_r,
        ),
        harmonic=HarmonicParams(M=int(rng.integers(4, 7))),
        modulation=ModulationParams(K=int(rng.integers(3, 6))),
        fault=fault, noise_std=0.0, seed=seed,
    )


def generate_latent(kind: str, family: int, seed: int, fs: int, duration: float, fault_probability: float):
    if kind == "music":
        return generate_stationary_music_source(fs, duration, seed, family)
    cfg = mechanical_config(family, seed, fs, duration, fault_probability)
    return GearSignalGenerator(cfg).generate()["signal"].astype(np.float32)


def make_filters(
    rng: np.random.Generator,
    k: int,
    min_taps: int,
    max_taps: int,
    model: str = "legacy",
):
    """Generate fixed source-specific FIRs.

    ``legacy`` reproduces the exponential-decay family used by the original
    scripts. ``reflections`` adds weak signed echoes as an optional robustness
    dataset without changing the default paper setup.
    """
    filters = []
    for _ in range(k):
        taps = int(rng.integers(min_taps, max_taps + 1))
        decay = float(rng.uniform(0.5, 0.8))
        h = np.exp(-decay * np.arange(taps, dtype=np.float64))
        if model == "reflections" and taps > 4:
            for _ in range(int(rng.integers(1, 4))):
                pos = int(rng.integers(1, taps))
                h[pos] += float(rng.uniform(-0.15, 0.15)) * np.exp(-0.5 * decay * pos)
        elif model != "legacy":
            raise ValueError("filter model must be legacy or reflections")
        filters.append(h.astype(np.float32))
    return filters


def add_noise_at_snr(clean: np.ndarray, snr_db: float, rng: np.random.Generator):
    power = float(np.mean(clean.astype(np.float64) ** 2))
    noise_power = power / (10.0 ** (snr_db / 10.0))
    return clean + rng.normal(0.0, np.sqrt(noise_power + 1e-12), clean.shape).astype(np.float32)


def build_split(
    count: int,
    k: int,
    kind: str,
    filters: list[np.ndarray],
    fs: int,
    duration: float,
    snr_low: float,
    snr_high: float,
    seed: int,
    fault_probability: float,
    fft_norm: str,
    cache_fft: bool,
):
    rng = np.random.default_rng(seed)
    signal_len = int(round(fs * duration))
    n_fft = signal_len + max(len(h) for h in filters) - 1
    sources = np.empty((count, k, signal_len), np.float32)
    source_images = np.zeros((count, k, n_fft), np.float32)
    mixtures = np.empty((count, n_fft), np.float32)
    snr = np.empty(count, np.float32)
    seeds = np.empty((count, k), np.int64)

    for i in range(count):
        for j in range(k):
            src_seed = int(rng.integers(0, 2**31 - 1))
            seeds[i, j] = src_seed
            s = generate_latent(kind, j, src_seed, fs, duration, fault_probability)
            if len(s) != signal_len:
                s = np.asarray(s[:signal_len], np.float32)
                if len(s) < signal_len:
                    s = np.pad(s, (0, signal_len - len(s)))
            sources[i, j] = s
            img = np.convolve(s, filters[j], mode="full")
            source_images[i, j, : len(img)] = img.astype(np.float32)
        clean_mix = source_images[i].sum(axis=0)
        snr[i] = float(rng.uniform(snr_low, snr_high))
        mixtures[i] = add_noise_at_snr(clean_mix, float(snr[i]), rng)

    payload = {
        "sources": sources,
        "source_images": source_images,
        "mixtures": mixtures,
        "snr_db": snr,
        "source_seeds": seeds,
        "sample_ids": np.asarray([f"sample_{i:06d}" for i in range(count)], dtype="U"),
        "sample_rate": np.int64(fs),
        "n_fft": np.int64(n_fft),
        "fft_norm": np.asarray(fft_norm),
    }
    if cache_fft:
        x = torch.from_numpy(sources)
        y = torch.from_numpy(mixtures)
        payload["sources_fft"] = torch.fft.rfft(x, n=n_fft, dim=-1, norm=fft_norm).numpy().astype(np.complex64)
        payload["mixtures_fft"] = torch.fft.rfft(y, n=n_fft, dim=-1, norm=fft_norm).numpy().astype(np.complex64)
    return payload, n_fft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kind", choices=["mechanical", "music"], default="mechanical")
    ap.add_argument("--sources", type=int, default=2)
    ap.add_argument("--sample-rate", type=int, default=2000)
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--train", type=int, default=3400)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--train-snr", nargs=2, type=float, default=(1.0, 4.0))
    ap.add_argument("--eval-snr", nargs=2, type=float, default=(1.0, 2.0))
    ap.add_argument("--fault-probability", type=float, default=0.05)
    ap.add_argument("--filter-taps", nargs=2, type=int, default=(5, 15))
    ap.add_argument("--filter-model", choices=["legacy", "reflections"], default="legacy")
    ap.add_argument("--fft-norm", type=str, default="ortho")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cache-fft", action="store_true")
    ap.add_argument("--uncompressed", action="store_true")
    args = ap.parse_args()
    if args.sources < 2:
        raise ValueError("At least two sources are required")

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    filters = make_filters(rng, args.sources, args.filter_taps[0], args.filter_taps[1], args.filter_model)
    max_taps = max(len(h) for h in filters)
    filter_array = np.zeros((args.sources, max_taps), np.float32)
    lengths = np.asarray([len(h) for h in filters], np.int64)
    for i, h in enumerate(filters):
        filter_array[i, : len(h)] = h
    np.savez_compressed(args.out / "filters.npz", filters=filter_array, filter_lengths=lengths)

    split_specs = {
        "tr": (args.train, args.train_snr, args.seed + 101),
        "cv": (args.val, args.eval_snr, args.seed + 202),
        "tt": (args.test, args.eval_snr, args.seed + 303),
    }
    metadata = vars(args).copy()
    metadata["out"] = str(args.out)
    metadata["splits"] = {}
    for code, (count, snr_range, seed) in split_specs.items():
        payload, n_fft = build_split(
            count, args.sources, args.kind, filters, args.sample_rate, args.duration,
            float(snr_range[0]), float(snr_range[1]), seed, args.fault_probability,
            args.fft_norm, not args.no_cache_fft,
        )
        saver = np.savez if args.uncompressed else np.savez_compressed
        saver(args.out / f"{code}.npz", **payload)
        metadata["splits"][code] = {"count": count, "snr_db": list(map(float, snr_range))}
        metadata["n_fft"] = n_fft

    with open(args.out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


if __name__ == "__main__":
    main()
