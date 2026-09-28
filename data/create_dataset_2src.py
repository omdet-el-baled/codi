"""
create_dataset_2src.py
----------------------
Same as create_dataset.py but with only TWO spur gear sources (A and B).
No epicyclic gear. Output folder: .../data/sim_2src
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
BASE_FOLDER = "/gpfs/data/ssa/users/d634375/FourierDiffusionConvolutiveSourceSeparation/data/sim_2src"

NUM_SAMPLES = 4000
DURATION = 1.0
FS = 2_000

TRAIN_RATIO = 0.85
VAL_RATIO = 0.05
TEST_RATIO = 0.10

MIXTURE_NOISE_LEVEL = 1
FAULTY_PROBABILITY = 0.05

# Each entry is one source type: "A" (slow spur), "B" (fast spur)
SOURCE_CONFIG = ["A", "B"]
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


_CONFIG_BUILDERS = {
    "A": build_config_A,
    "B": build_config_B,
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
    rng_global = np.random.default_rng(42)
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
            num_taps=float(rng_global.uniform(7, 18)),
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
            y = y + rng_global.normal(0.0, noise_std, size=len(y))

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
    # 8. README
    # ------------------------------------------------------------------
    import datetime

    # Compute per-source Nyquist info
    N_signal = int(FS * DURATION)
    source_rows = []
    for s_type in SOURCE_CONFIG:
        if s_type == "A":
            source_rows.append(
                "| s1 (A) | Spur gear — slow | 200–400 RPM | 12–19 teeth | "
                "M∈[4,6]  K∈[2,4] |"
            )
        elif s_type == "B":
            source_rows.append(
                "| s2 (B) | Spur gear — fast | 500–1000 RPM | 15–24 teeth | "
                "M∈[2,3]  K∈[3,5] |"
            )

    filter_rows = "\n".join(
        f"| s{idx + 1} ({SOURCE_CONFIG[idx]}) | {len(h)} taps | exponential decay |"
        for idx, h in enumerate(fixed_filters)
    )

    readme = f"""# Dataset README
Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview
Synthetic gear-vibration blind source separation (BSS) dataset — **2 spur gear sources**.
Each sample contains {num_sources} gear sources convolved with fixed FIR filters
and mixed together with additive Gaussian noise.

The signal model follows the **McFadden–Smith** formulation:
- Spur gears : Σ_m A_m [1 + a_m(t)] cos(2π m f_mesh t + φ_m + b_m(t))

---

## Signal Parameters

| Parameter | Value |
|---|---|
| Sampling rate | {FS} Hz |
| Duration | {DURATION} s |
| Samples per signal | {N_signal} |
| Mixture noise level | U[{MIXTURE_NOISE_LEVEL / 3:.2f}, {MIXTURE_NOISE_LEVEL:.2f}] |
| Fault probability | {FAULTY_PROBABILITY} |
| Fault resonance freq | 200 Hz |
| Fault damping ratio | 0.08 |

---

## Sources

| Source | Type | Speed range | Tooth parameters | Harmonics |
|---|---|---|---|---|
{chr(10).join(source_rows)}

---

## Filters

One fixed exponentially-decaying FIR filter per source, applied via linear convolution.
Filters are saved in `filters/filters.npy` (array of variable-length float32 arrays).

| Source | Length | Shape |
|---|---|---|
{filter_rows}

Filter RDFT responses are plotted in `filters/filters_plot.png`
(zero-padded to nfft = {N_signal} + {max_filter_len} = {nfft}).

---

## Dataset Split

| Split | Folder | Samples |
|---|---|---|
| Train | `tr/` | {len(train_idx)} |
| Validation | `cv/` | {len(val_idx)} |
| Test | `tt/` | {len(test_idx)} |
| **Total** | | **{NUM_SAMPLES}** |

---

## Folder Structure

```
{os.path.basename(BASE_FOLDER)}/
├── filters/
│   ├── filters.npy          # fixed FIR filters [list of arrays]
│   └── filters_plot.png     # RDFT magnitude responses
├── sample_multiaxis.png     # time-domain diagnostic plot
├── sample_dft_multiaxis.png # RDFT diagnostic plot
├── tr/                      # training split
│   ├── mix/                 # mixtures
│   ├── s1/                  # source 1 (A)
│   └── s2/                  # source 2 (B)
├── cv/                      # validation split (same structure)
└── tt/                      # test split (same structure)
```

All audio files are **float32 WAV** at {FS} Hz, {DURATION}s ({N_signal} samples).
Sources are noise-free; noise is added only to the mixture.

---

## Loading Example

```python
from dataset import make_dataset

train_ds = make_dataset(
    base_dir     = "{BASE_FOLDER}",
    split        = "tr",
    num_sources  = {num_sources},
    sample_rate  = {FS},
)
mixture, sources = train_ds[0]
# mixture : Tensor [1, {N_signal}]
# sources : Tensor [{num_sources}, {N_signal}]
```
"""

    readme_path = os.path.join(BASE_FOLDER, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme)
    print(f"README written → {readme_path}")

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
