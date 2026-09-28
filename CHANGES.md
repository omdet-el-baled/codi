# Changes relative to the uploaded project

The guiding rule was to preserve the uploaded code as the source of truth and
add features around it rather than replace it with a new framework.

## Data
- `MixData` now auto-detects split-wise NPZ archives.
- Cached FFT arrays are supported.
- WAV loading remains supported.
- `convert_wav_dataset_to_npz.py` migrates an existing working dataset.
- `create_dataset_npz.py` creates latent-source-consistent new datasets and
  retains full linear convolution tails.
- Added stationary synthetic music generator.

## Model
- Added `PhysicsConditionedUNet` but retained `ncsn.UNet`.
- Added optional fixed-kernel conditioning through `ScoreModel.set_operator`.
- Enforces real DC/Nyquist network outputs for rFFT spectra.

## Training
- Local EMA removes the silent no-op fallback that existed when `torch_ema` was
  unavailable.
- EMA update is optimizer-step-aware and has warm-up.
- EMA evaluation no longer overrides PyTorch `train()` / `eval()`.
- Lightning now handles gradient clipping after gradient accumulation.
- Default optimizer changed to AdamW 1e-4 with weight decay 1e-4.
- Broad warning suppression and always-on anomaly detection were removed.
- Checkpoints save top 3 plus `last.ckpt`.

## Diffusion / inference
- Existing VEConvMixSDE is retained.
- `drift_mode=none` provides a matched no-drift ablation.
- The full global mixing matrix is no longer allocated unless explicitly asked;
  the original analytical eigensystem remains unchanged.
- Added Euler/Heun/RK4 probability-flow solvers and EM/PC stochastic samplers.
- Added NFE accounting.

## Metrics
- Fixed validation SI-SDR argument ordering (`prediction, target`).
- Added explicit latent-source and source-image evaluation.
