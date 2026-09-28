"""
create_dataset.py
-----------------
Generates a synthetic gear-vibration dataset using the new SignalConfig-based
GearSignalGenerator API (gear_signal.py).  Output layout mirrors the original
script: WAV files split into tr / cv / tt folders with s1…sN sub-folders for
each source and a mix/ folder for the mixture.

Epicyclic gears now synthesize TWO mesh components per planet:
    • Ring–planet mesh  :  f_mesh_ring = f_c * Z_r
    • Sun–planet mesh   :  f_mesh_sun  = f_c * (Z_r + Z_s)
so Z_s, Z_p, Z_r, and N_p all meaningfully affect the signal.

Nyquist constraint for EPI (FS=2000, M up to 6):
    f_c_max = (FS/2) / (M_max * (Z_r_max + Z_s_max))
    → carrier RPM capped at 80 to be safe across all random tooth combinations.
"""

from __future__ import annotations

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
from scipy.io.wavfile import write
from sklearn.model_selection import train_test_split

from gear_signal import (
    EpicyclicGearParams,
    FaultParams,
    GearParams,
    GearSignalGenerator,
    HarmonicParams,
    ModulationParams,
    SignalConfig,
)

# =====================================================================
# CONFIGURATION
# =====================================================================
BASE_FOLDER = (
    "/gpfs/data/ssa/users/d634375/FourierDiffusionConvolutiveSourceSeparation/data/sim"
)

NUM_SAMPLES = 4000
DURATION = 1.0
FS = 2_000

TRAIN_RATIO = 0.85
VAL_RATIO = 0.05
TEST_RATIO = 0.10

MIXTURE_NOISE_LEVEL = 2
FAULTY_PROBABILITY = 0.05

# Each entry is one source type: "A" (slow spur), "B" (fast spur), "EPI"
SOURCE_CONFIG = ["A", "B", "EPI"]
# =====================================================================


# ------------------------------------------------------------------
# Parameter samplers — one per source type
# ------------------------------------------------------------------


def _sample_fault(rng: np.random.Generator) -> FaultParams:
    faulty = rng.choice([True, False], p=[FAULTY_PROBABILITY, 1 - FAULTY_PROBABILITY])
    return FaultParams(
        enabled=bool(faulty),
        impulse_amplitude=1.0,
        resonance_freq=200.0,
        damping_ratio=0.08,
    )


def build_config_A(seed: int) -> SignalConfig:
    """Slow spur gear: 200–400 RPM, 12–19 teeth."""
    rng = np.random.default_rng(seed)
    return SignalConfig(
        duration=DURATION,
        sampling_rate=FS,
        gear=GearParams(
            teeth=int(rng.integers(12, 20)),
            speed_rpm=float(rng.uniform(200, 400)),
        ),
        harmonic=HarmonicParams(M=int(rng.integers(4, 7))),
        modulation=ModulationParams(K=int(rng.integers(2, 5))),
        fault=_sample_fault(rng),
        noise_std=0.0,
        seed=seed,
    )


def build_config_B(seed: int) -> SignalConfig:
    """Fast spur gear: 500–1000 RPM, 15–24 teeth."""
    rng = np.random.default_rng(seed)
    return SignalConfig(
        duration=DURATION,
        sampling_rate=FS,
        gear=GearParams(
            teeth=int(rng.integers(15, 25)),
            speed_rpm=float(rng.uniform(500, 1_000)),
        ),
        harmonic=HarmonicParams(M=int(rng.integers(2, 4))),
        modulation=ModulationParams(K=int(rng.integers(3, 6))),
        fault=_sample_fault(rng),
        noise_std=0.0,
        seed=seed,
    )


def build_config_EPI(seed: int) -> SignalConfig:
    """
    Epicyclic gear — fixed ring, carrier as input.

    Carrier RPM capped at 80 to keep ALL harmonics of BOTH mesh frequencies
    (ring and sun) below Nyquist for the full tooth-count ranges:
        Z_s ∈ [18, 24],  Z_r ∈ [70, 90],  M ∈ [4, 6]
        f_mesh_sun_max * M_max = f_c * (Z_r_max + Z_s_max) * M_max
        → f_c_max = 1000 / (6 * 114) ≈ 1.46 Hz  → RPM_max ≈ 87.7
        We use 80 RPM for a comfortable margin.
    """
    rng = np.random.default_rng(seed)
    Z_s = int(rng.integers(18, 25))  # 18–24
    Z_p = int(rng.integers(25, 36))  # 25–35
    Z_r = int(rng.integers(70, 90))  # 70–89
    return SignalConfig(
        duration=DURATION,
        sampling_rate=FS,
        gear=EpicyclicGearParams(
            speed_rpm=float(rng.uniform(30, 80)),  # safe for all tooth combos
            teeth=Z_s,  # fills base GearParams.teeth
            N_p=3,
            Z_s=Z_s,
            Z_p=Z_p,
            Z_r=Z_r,
        ),
        harmonic=HarmonicParams(M=int(rng.integers(4, 7))),
        modulation=ModulationParams(K=int(rng.integers(3, 6))),
        fault=_sample_fault(rng),
        noise_std=0.0,
        seed=seed,
    )


_CONFIG_BUILDERS = {
    "A": build_config_A,
    "B": build_config_B,
    "EPI": build_config_EPI,
}


def generate_source(src_type: str, seed: int) -> np.ndarray:
    """Build config → generate → return raw signal (no noise)."""
    builder = _CONFIG_BUILDERS[src_type]
    cfg = builder(seed)
    result = GearSignalGenerator(cfg).generate()
    return result["signal"]


# ------------------------------------------------------------------
# Minimal FIR filter helpers (replaces FIRFilterGenerator dependency)
# ------------------------------------------------------------------


def make_weight_decay_filter(decay_rate: float, num_taps: int) -> np.ndarray:
    """Exponentially-decaying FIR impulse response (not unit-norm — raw decay)."""
    taps = int(round(num_taps))
    h = np.exp(-decay_rate * np.arange(taps))
    return h.astype(np.float32)


def apply_filter(signal: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Linear (full) convolution, trimmed to input length."""
    return np.convolve(signal, h, mode="full")[: len(signal)]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    rng_global = np.random.default_rng(0)
    num_sources = len(SOURCE_CONFIG)

    print(f"Sources      : {SOURCE_CONFIG}")
    print(f"Faulty prob  : {FAULTY_PROBABILITY}")
    print(f"Samples      : {NUM_SAMPLES}  |  fs={FS} Hz  |  dur={DURATION}s")

    # ------------------------------------------------------------------
    # 1. Fixed per-source FIR filters
    # ------------------------------------------------------------------
    print("\nGenerating filters …")
    fixed_filters = [
        make_weight_decay_filter(
            decay_rate=float(rng_global.uniform(0.5, 0.8)),
            num_taps=float(rng_global.uniform(5, 15)),
        )
        for _ in SOURCE_CONFIG
    ]

    filters_dir = os.path.join(BASE_FOLDER, "filters")
    os.makedirs(filters_dir, exist_ok=True)
    np.save(
        os.path.join(filters_dir, "filters.npy"), np.array(fixed_filters, dtype=object)
    )
    print("Filters saved (.npy)")

    # ------------------------------------------------------------------
    # 2. Generate samples
    # ------------------------------------------------------------------
    data_sources: list[list[np.ndarray]] = [[] for _ in range(num_sources)]
    data_y: list[np.ndarray] = []

    print(f"\nGenerating {NUM_SAMPLES} samples …")

    for i in range(NUM_SAMPLES):
        current_sources = []

        for s_idx, src_type in enumerate(SOURCE_CONFIG):
            # Deterministic but unique seed per (sample, source)
            seed = int(rng_global.integers(0, 2**31))
            x = generate_source(src_type, seed)

            z = apply_filter(x, fixed_filters[s_idx])
            current_sources.append(z)
            data_sources[s_idx].append(z)

        # Mixture
        y = np.sum(current_sources, axis=0)
        if MIXTURE_NOISE_LEVEL > 0:
            noise_std = float(
                rng_global.uniform(MIXTURE_NOISE_LEVEL / 3, MIXTURE_NOISE_LEVEL)
            )
            y = y + rng_global.normal(noise_std / 2, noise_std, size=len(y))

        data_y.append(y)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{NUM_SAMPLES}")

    data_sources_np = [np.array(s) for s in data_sources]  # (N, T) each
    data_y_np = np.array(data_y)  # (N, T)

    # ------------------------------------------------------------------
    # 3. Train / val / test split
    # ------------------------------------------------------------------
    indices = np.arange(NUM_SAMPLES)
    train_idx, temp_idx = train_test_split(
        indices, train_size=TRAIN_RATIO, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=VAL_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=42,
    )
    splits = {"tr": train_idx, "cv": val_idx, "tt": test_idx}
    print(
        f"\nSplit  →  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}"
    )

    # ------------------------------------------------------------------
    # 4. Save WAV files
    # ------------------------------------------------------------------
    print("\nSaving WAV files …")
    source_folders = [f"s{i + 1}" for i in range(num_sources)]
    all_folders = source_folders + ["mix"]

    for split_name, idxs in splits.items():
        for folder in all_folders:
            os.makedirs(os.path.join(BASE_FOLDER, split_name, folder), exist_ok=True)

    for split_name, idxs in splits.items():
        for i in idxs:
            for s_idx in range(num_sources):
                write(
                    os.path.join(
                        BASE_FOLDER, split_name, f"s{s_idx + 1}", f"sample_{i}.wav"
                    ),
                    FS,
                    data_sources_np[s_idx][i].astype(np.float32),
                )
            write(
                os.path.join(BASE_FOLDER, split_name, "mix", f"sample_{i}.wav"),
                FS,
                data_y_np[i].astype(np.float32),
            )

    print("WAV files saved.")

    # ------------------------------------------------------------------
    # Shared plot style
    # ------------------------------------------------------------------
    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    STYLE = "seaborn-v0_8-whitegrid"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "lines.linewidth": 1.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # ------------------------------------------------------------------
    # 5. Plot: filter RDFT magnitude spectra (zero-padded to N_signal + max_filter_len)
    # ------------------------------------------------------------------
    N_signal = int(FS * DURATION)
    max_filter_len = max(len(h) for h in fixed_filters)
    nfft = N_signal + max_filter_len
    freqs_h = np.fft.rfftfreq(nfft, d=1.0 / FS)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))
        for idx, h in enumerate(fixed_filters):
            H = np.fft.rfft(h, n=nfft)
            ax.plot(
                freqs_h,
                np.abs(H),
                color=COLORS[idx % len(COLORS)],
                label=f"Source s{idx + 1}  ({SOURCE_CONFIG[idx]})",
                alpha=0.9,
            )
        ax.set_xlabel("Frequency (Hz)", labelpad=6)
        ax.set_ylabel("|H(f)|", labelpad=6)
        ax.set_title(
            f"Filter frequency responses  "
            f"(nfft = {N_signal} + {max_filter_len} = {nfft})",
            pad=10,
        )
        ax.legend(framealpha=0.7, loc="upper right")
        ax.set_xlim(0, FS / 2)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        fig.tight_layout()
        fig.savefig(
            os.path.join(filters_dir, "filters_plot.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    print(f"Filter RDFT plot saved (nfft={nfft} = {N_signal} + {max_filter_len}).")

    # ------------------------------------------------------------------
    # 6. Plot: one sample — time domain (separate axes)
    # ------------------------------------------------------------------
    sample_idx = random.randint(0, NUM_SAMPLES - 1)
    n_plots = num_sources + 1
    labels_td = [f"s{i + 1}  ({SOURCE_CONFIG[i]})" for i in range(num_sources)] + [
        "Mixture"
    ]

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(n_plots, 1, figsize=(11, 2.8 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        t_axis = np.arange(N_signal) / FS

        for s_idx in range(num_sources):
            axes[s_idx].plot(
                t_axis,
                data_sources_np[s_idx][sample_idx],
                color=COLORS[s_idx % len(COLORS)],
                lw=1.2,
                alpha=0.85,
            )
            axes[s_idx].set_ylabel(labels_td[s_idx], fontsize=10)

        axes[-1].plot(
            t_axis, data_y_np[sample_idx], color="#333333", lw=1.2, alpha=0.85
        )
        axes[-1].set_ylabel("Mixture", fontsize=10)
        axes[-1].set_xlabel("Time (s)", labelpad=6)

        for ax in axes:
            ax.margins(x=0.01)

        fig.suptitle(
            f"Time-domain signals  (sample #{sample_idx})", fontsize=13, y=1.01
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(BASE_FOLDER, "sample_multiaxis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    print(f"Time-domain plot saved (sample index={sample_idx}).")

    # ------------------------------------------------------------------
    # 7. Plot: one sample — frequency domain (RDFT magnitude)
    # ------------------------------------------------------------------
    N = N_signal
    freqs = np.fft.rfftfreq(N, d=1.0 / FS)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(n_plots, 1, figsize=(11, 2.8 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        for s_idx in range(num_sources):
            X = fft.rfft(data_sources_np[s_idx][sample_idx])
            axes[s_idx].plot(
                freqs,
                np.abs(X),
                color=COLORS[s_idx % len(COLORS)],
                lw=1.2,
                alpha=0.85,
            )
            axes[s_idx].set_ylabel(labels_td[s_idx], fontsize=10)

        Y = fft.rfft(data_y_np[sample_idx])
        axes[-1].plot(freqs, np.abs(Y), color="#333333", lw=1.2, alpha=0.85)
        axes[-1].set_ylabel("Mixture", fontsize=10)
        axes[-1].set_xlabel("Frequency (Hz)", labelpad=6)

        for ax in axes:
            ax.set_xlim(0, FS / 2)
            ax.margins(x=0.01)

        fig.suptitle(
            f"RDFT magnitude spectra  (sample #{sample_idx})", fontsize=13, y=1.01
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(BASE_FOLDER, "sample_dft_multiaxis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    print(f"DFT plot saved (sample index={sample_idx}).")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n── DONE ──────────────────────────────────────────────")
    print(f"Dataset   : {BASE_FOLDER}")
    print(f"Filters   : {filters_dir}")
    print(f"Sources   : {SOURCE_CONFIG}")
    print(f"Fault prob: {FAULTY_PROBABILITY}")
    print(f"Train={len(train_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")


if __name__ == "__main__":
    main()
