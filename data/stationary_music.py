"""Small stationary synthetic music-like source generator.

This benchmark is intentionally stationary over the complete record so it is
compatible with the paper's full-signal DFT assumption.  It is not intended to
replace MUSDB18 or a real music benchmark.
"""
from __future__ import annotations
import numpy as np


def generate_stationary_music_source(
    sample_rate: int,
    duration: float,
    seed: int,
    family: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(sample_rate * duration))
    t = np.arange(n, dtype=np.float64) / sample_rate
    nyquist = sample_rate / 2.0

    # Different stationary spectral envelopes emulate distinct timbres.
    family = int(family) % 4
    if family == 0:      # bright harmonic
        f0 = rng.uniform(55.0, 130.0); max_h = 10; decay = 0.65; odd_only = False
    elif family == 1:    # warm / smooth
        f0 = rng.uniform(70.0, 180.0); max_h = 8; decay = 1.25; odd_only = False
    elif family == 2:    # reed/square-like
        f0 = rng.uniform(45.0, 115.0); max_h = 13; decay = 0.9; odd_only = True
    else:                # formant-shaped harmonic stack
        f0 = rng.uniform(80.0, 220.0); max_h = 9; decay = 0.85; odd_only = False

    harmonics = np.arange(1, max_h + 1)
    if odd_only:
        harmonics = harmonics[harmonics % 2 == 1]
    harmonics = harmonics[harmonics * f0 < 0.95 * nyquist]
    phases = rng.uniform(-np.pi, np.pi, size=len(harmonics))

    amps = 1.0 / np.maximum(harmonics.astype(np.float64), 1.0) ** decay
    if family == 3:
        center = rng.uniform(250.0, min(700.0, 0.8 * nyquist))
        width = rng.uniform(120.0, 280.0)
        hf = harmonics * f0
        amps *= 0.35 + np.exp(-0.5 * ((hf - center) / width) ** 2)
    amps *= rng.uniform(0.75, 1.25, size=len(amps))

    signal = np.zeros_like(t)
    for h, a, p in zip(harmonics, amps, phases):
        signal += a * np.sin(2.0 * np.pi * h * f0 * t + p)
    signal -= signal.mean()
    rms = np.sqrt(np.mean(signal**2) + 1e-12)
    signal = signal / rms
    return signal.astype(np.float32)
