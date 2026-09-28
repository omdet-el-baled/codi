import os
import time
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt
import musdb
import librosa
import soundfile as sf
import scipy.signal
from tqdm import tqdm

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SOURCE_CONFIG = ["drums", "bass"]
FS = 8000
DURATION = 1.0
HOP_DURATION = 0.5
ENERGY_THRESHOLD = 1e-3
MAX_SAMPLES_PER_TRACK = 40
MUSDB_ROOT = "/gpfs/data/ssa/users/d634375/FourierDiffusionConvolutiveSourceSeparation/data/musdb18"
IS_WAV = True
BASE_FOLDER = "./data/musdb18_convolutive_dataset"
DECAY_RATE_RANGE = (0.5, 0.8)
NUM_TAPS_RANGE = (7, 18)

SNR_RANGE_DB = (-10.0, 10.0)
# ------------------------------------------------------------------


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    tqdm.write(f"  ⏱  {label:<35} {elapsed:.3f}s")


def make_weight_decay_filter(decay_rate: float, num_taps: int) -> np.ndarray:
    taps = int(round(num_taps))
    h = np.exp(-decay_rate * np.arange(taps))
    return h.astype(np.float32)


def apply_filter(signal: np.ndarray, h: np.ndarray) -> np.ndarray:
    out = scipy.signal.fftconvolve(signal, h, mode="full")[: len(signal)]
    return out.astype(np.float32)


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def resample_chunk(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    return librosa.resample(
        audio, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq"
    ).astype(np.float32)


def generate_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise, unit variance."""
    return rng.standard_normal(length).astype(np.float32)


def mix_with_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Scale noise so that SNR = 10 log10(P_signal / P_noise) = snr_db,
    then return signal + scaled_noise.
    """
    p_signal = np.mean(signal**2)
    p_noise = np.mean(noise**2)
    if p_noise < 1e-12 or p_signal < 1e-12:
        return signal
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = np.sqrt(p_signal / (snr_linear * p_noise))
    return (signal + noise_scale * noise).astype(np.float32)


def uniform_subsample(items: list, k: int) -> list:
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    sf.write(path, audio, sr)


# ------------------------------------------------------------------
# Filter plots
# ------------------------------------------------------------------
def plot_filters(filters: list, out_path: str, sr: int) -> None:
    fig, axes = plt.subplots(
        len(filters), 1, figsize=(10, 3 * len(filters)), sharex=True
    )
    if len(filters) == 1:
        axes = [axes]
    colors = ["#e07b54", "#5b9bd5"]
    for ax, h, src, color in zip(axes, filters, SOURCE_CONFIG, colors):
        N = max(512, len(h))
        H = np.fft.rfft(h, n=N)
        freqs = np.fft.rfftfreq(N, d=1.0 / sr)
        mag_db = 20 * np.log10(np.abs(H) + 1e-9)
        ax.plot(freqs, mag_db, color=color, lw=1.5)
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"FIR filter — {src}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frequency (Hz)")
    plt.suptitle("Filter rDFT Magnitude Responses", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved filter plot → {out_path}")


# ------------------------------------------------------------------
# Diagnostic plots
# ------------------------------------------------------------------
def plot_sample_time(
    sources: list,
    clean_mixture: np.ndarray,
    noisy_mixture: np.ndarray,
    out_path: str,
    sr: int,
) -> None:
    n_rows = len(sources) + 2  # stems + clean mix + noisy mix
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)
    colors = ["#e07b54", "#5b9bd5"]
    t = np.linspace(0, len(clean_mixture) / sr, len(clean_mixture))

    for ax, s, src, color in zip(axes, sources, SOURCE_CONFIG, colors):
        ax.plot(t, s, color=color, lw=0.5)
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{src} (filtered)")
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.2)

    axes[-2].plot(t, clean_mixture, color="gray", lw=0.5)
    axes[-2].set_ylabel("Amplitude")
    axes[-2].set_title("Clean mixture (drums + bass)")
    axes[-2].set_ylim(-1, 1)
    axes[-2].grid(True, alpha=0.2)

    axes[-1].plot(t, noisy_mixture, color="#c0392b", lw=0.5)
    axes[-1].set_ylabel("Amplitude")
    axes[-1].set_title("Noisy mixture (with additive noise)")
    axes[-1].set_ylim(-1, 1)
    axes[-1].set_xlabel("Time (s)")
    axes[-1].grid(True, alpha=0.2)

    plt.suptitle("Time-domain diagnostic", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved time-domain plot → {out_path}")


def plot_sample_dft(
    sources: list,
    clean_mixture: np.ndarray,
    noisy_mixture: np.ndarray,
    out_path: str,
    sr: int,
) -> None:
    n_rows = len(sources) + 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)
    colors = ["#e07b54", "#5b9bd5"]

    for ax, s, src, color in zip(axes, sources, SOURCE_CONFIG, colors):
        freqs = np.fft.rfftfreq(len(s), d=1.0 / sr)
        mag_db = 20 * np.log10(np.abs(np.fft.rfft(s)) + 1e-9)
        ax.plot(freqs, mag_db, color=color, lw=0.6)
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"rDFT — {src}")
        ax.grid(True, alpha=0.3)

    for ax, sig, label, color in zip(
        axes[-2:],
        [clean_mixture, noisy_mixture],
        ["rDFT — Clean mixture", "rDFT — Noisy mixture"],
        ["gray", "#c0392b"],
    ):
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / sr)
        mag_db = 20 * np.log10(np.abs(np.fft.rfft(sig)) + 1e-9)
        ax.plot(freqs, mag_db, color=color, lw=0.6)
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frequency (Hz)")
    plt.suptitle("rDFT diagnostic", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved rDFT plot → {out_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    rng_global = np.random.default_rng(42)
    rng_noise = np.random.default_rng(99)  # separate seed → reproducible SNRs
    chunk_len = int(round(DURATION * FS))
    hop_len = int(round(HOP_DURATION * FS))

    print(f"Sources          : {SOURCE_CONFIG}")
    print(f"FS               : {FS} Hz  |  chunk={DURATION}s  |  hop={HOP_DURATION}s")
    print(f"Energy threshold : {ENERGY_THRESHOLD}")
    print(f"Max samples/track: {MAX_SAMPLES_PER_TRACK}")
    print(f"SNR range        : {SNR_RANGE_DB[0]} dB → {SNR_RANGE_DB[1]} dB")

    # ------------------------------------------------------------------
    # 1. Fixed per-source FIR filters
    # ------------------------------------------------------------------
    print("\nGenerating filters …")
    fixed_filters = [
        make_weight_decay_filter(
            decay_rate=float(rng_global.uniform(*DECAY_RATE_RANGE)),
            num_taps=float(rng_global.uniform(*NUM_TAPS_RANGE)),
        )
        for _ in SOURCE_CONFIG
    ]
    filter_dir = os.path.join(BASE_FOLDER, "filters")
    os.makedirs(filter_dir, exist_ok=True)
    np.save(
        os.path.join(filter_dir, "filters.npy"), np.array(fixed_filters, dtype=object)
    )
    plot_filters(
        fixed_filters, out_path=os.path.join(filter_dir, "filters_plot.png"), sr=FS
    )

    # ------------------------------------------------------------------
    # 2. Load MUSDB18-HQ and build splits
    # ------------------------------------------------------------------
    print("\nLoading MUSDB18-HQ …")
    mus = musdb.DB(root=MUSDB_ROOT, is_wav=IS_WAV)
    train_tracks = mus.load_mus_tracks(subsets="train")
    test_tracks = mus.load_mus_tracks(subsets="test")

    rng_split = np.random.default_rng(42)
    n_cv = max(1, int(round(0.1 * len(train_tracks))))
    idx = rng_split.permutation(len(train_tracks))
    cv_tracks = [train_tracks[i] for i in idx[:n_cv]]
    train_tracks = [train_tracks[i] for i in idx[n_cv:]]

    splits = {"tr": train_tracks, "cv": cv_tracks, "tt": test_tracks}
    print(f"  train={len(train_tracks)}  cv={len(cv_tracks)}  test={len(test_tracks)}")

    # ------------------------------------------------------------------
    # 3. Build dataset
    # ------------------------------------------------------------------
    diagnostic_saved = False
    global_stats = {}

    for split_name, tracks in splits.items():
        print(f"\n── Split '{split_name}' ({len(tracks)} tracks) ──")

        mix_dir = os.path.join(BASE_FOLDER, split_name, "mix")
        s1_dir = os.path.join(BASE_FOLDER, split_name, "s1")
        s2_dir = os.path.join(BASE_FOLDER, split_name, "s2")
        for d in [mix_dir, s1_dir, s2_dir]:
            os.makedirs(d, exist_ok=True)

        total_kept = 0
        total_dropped = 0

        for track in tqdm(tracks, desc=f"[{split_name}]", unit="track"):
            name = track.name.replace(" ", "_")
            tqdm.write(f"\n  Track: {name}")

            # ── mono mix-down ─────────────────────────────────────────
            with timer("mono mix-down"):
                stems_mono = [
                    to_mono(track.targets[src].audio) for src in SOURCE_CONFIG
                ]

            # ── resample full signal once ─────────────────────────────
            with timer("resample"):
                stems_resampled = [
                    resample_chunk(s, track.rate, FS) for s in stems_mono
                ]

            # ── vectorised stride + energy gate ───────────────────────
            with timer("stride + energy gate"):
                chunk_len_orig = int(round(chunk_len * track.rate / FS))
                hop_len_orig = int(round(hop_len * track.rate / FS))
                max_len = max(len(s) for s in stems_mono)

                stems_padded = []
                for s in stems_mono:
                    pad = max(
                        0,
                        chunk_len_orig
                        - (max_len - len(s) + chunk_len_orig) % hop_len_orig,
                    )
                    stems_padded.append(np.pad(s, (0, pad + chunk_len_orig)))

                def make_strided(x: np.ndarray) -> np.ndarray:
                    n_win = (len(x) - chunk_len_orig) // hop_len_orig + 1
                    shape = (n_win, chunk_len_orig)
                    strides = (x.strides[0] * hop_len_orig, x.strides[0])
                    return np.lib.stride_tricks.as_strided(
                        x, shape=shape, strides=strides
                    )

                strided = [make_strided(s) for s in stems_padded]
                n_windows = strided[0].shape[0]

                rms_all = np.stack(
                    [np.sqrt(np.mean(s**2, axis=1)) for s in strided]
                )  # (n_stems, n_windows)
                valid_mask = np.all(rms_all >= ENERGY_THRESHOLD, axis=0)
                valid_indices = np.where(valid_mask)[0]
                total_dropped += n_windows - len(valid_indices)

            tqdm.write(
                f"       windows={n_windows}  valid={len(valid_indices)}"
                f"  dropped={n_windows - len(valid_indices)}"
            )

            # ── uniform subsample ─────────────────────────────────────
            with timer("subsample"):
                if len(valid_indices) > MAX_SAMPLES_PER_TRACK:
                    step = len(valid_indices) / MAX_SAMPLES_PER_TRACK
                    valid_indices = [
                        valid_indices[int(i * step)]
                        for i in range(MAX_SAMPLES_PER_TRACK)
                    ]

            # ── convolve + noise + save ───────────────────────────────
            t_conv = t_save = 0.0

            for w_idx in tqdm(
                valid_indices, desc=f"  {name[:28]:<28}", unit="chunk", leave=False
            ):
                # slice from already-resampled signal
                start = w_idx * hop_len
                raw_chunks = [s[start : start + chunk_len] for s in stems_resampled]
                raw_chunks = [
                    c if len(c) == chunk_len else np.pad(c, (0, chunk_len - len(c)))
                    for c in raw_chunks
                ]

                # convolve
                t0 = time.perf_counter()
                filtered = [
                    apply_filter(c, h) for c, h in zip(raw_chunks, fixed_filters)
                ]
                clean_mixture = sum(filtered)

                # additive noise at random SNR ∈ SNR_RANGE_DB
                snr_db = float(rng_noise.uniform(*SNR_RANGE_DB))
                noise = generate_noise(len(clean_mixture), rng_noise)
                noisy_mixture = mix_with_snr(clean_mixture, noise, snr_db)
                t_conv += time.perf_counter() - t0

                # save — mix gets noisy mixture, s1/s2 get clean stems
                t0 = time.perf_counter()
                fname = f"{name}_w{w_idx:04d}.wav"
                save_wav(os.path.join(mix_dir, fname), noisy_mixture, FS)
                save_wav(os.path.join(s1_dir, fname), filtered[0], FS)
                save_wav(os.path.join(s2_dir, fname), filtered[1], FS)
                t_save += time.perf_counter() - t0

                if not diagnostic_saved:
                    plot_sample_time(
                        filtered,
                        clean_mixture,
                        noisy_mixture,
                        os.path.join(BASE_FOLDER, "sample_multiaxis.png"),
                        FS,
                    )
                    plot_sample_dft(
                        filtered,
                        clean_mixture,
                        noisy_mixture,
                        os.path.join(BASE_FOLDER, "sample_dft_multiaxis.png"),
                        FS,
                    )
                    diagnostic_saved = True

                total_kept += 1

            tqdm.write(f"  ⏱  {'convolve + noise (all chunks)':<35} {t_conv:.3f}s")
            tqdm.write(f"  ⏱  {'save wavs (all chunks)':<35} {t_save:.3f}s")

        global_stats[split_name] = {"kept": total_kept, "dropped": total_dropped}
        tqdm.write(
            f"\n  → split '{split_name}' done  "
            f"kept={total_kept}  dropped={total_dropped}"
        )

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 55}")
    print(f"Dataset written to: {os.path.abspath(BASE_FOLDER)}")
    for split_name, s in global_stats.items():
        total = s["kept"] + s["dropped"]
        ratio = 100 * s["dropped"] / total if total else 0
        print(
            f"  {split_name}:  kept={s['kept']}  "
            f"dropped={s['dropped']}  ({ratio:.1f}% discarded)"
        )
    print(f"{'=' * 55}")
    print(f"\n{os.path.basename(BASE_FOLDER)}/")
    print("├── filters/")
    print("│   ├── filters.npy")
    print("│   └── filters_plot.png")
    print("├── sample_multiaxis.png     ← stems + clean mix + noisy mix")
    print("├── sample_dft_multiaxis.png ← rDFT of all signals")
    print("├── tr/                      (90 tracks, ≤3 600 chunks)")
    print("│   ├── mix/  ← noisy mixture")
    print("│   ├── s1/   ← drums (clean, filtered)")
    print("│   └── s2/   ← bass  (clean, filtered)")
    print("├── cv/                      (10 tracks, ≤400 chunks)")
    print("│   ├── mix/")
    print("│   ├── s1/")
    print("│   └── s2/")
    print("└── tt/                      (50 tracks, ≤2 000 chunks)")
    print("    ├── mix/")
    print("    ├── s1/")
    print("    └── s2/")


if __name__ == "__main__":
    main()
