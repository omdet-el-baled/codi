# FourierDiffusionConvolutiveSourceSeparation — enhanced in-place version

This version keeps the structure and mathematical implementation of the code you
provided.  It is an **in-place enhancement**, not a replacement framework.
The original WAV loader, original `ncsn.UNet`, Hydra entry point, and legacy
sampler path are retained.

## Main changes

- split-wise **NPZ data backend** with optional cached complex FFTs;
- a converter for your already-generated WAV data;
- a new generator whose primary target is the **latent source** `s_k`, while
  `h_k*s_k` is stored separately as `source_images`;
- full linear-convolution tails are retained in the new generator;
- optional stationary synthetic music data compatible with the full-record DFT;
- stronger `PhysicsConditionedUNet`, selectable independently of the diffusion;
- the known transfer functions are attached once to the network and used to
  create normalized `H^H(Hx-y)` features;
- dependency-free, checkpointed **EMA with warm-up**;
- EMA is swapped in only during validation/test instead of overriding
  `Module.train()` / `eval()`;
- AdamW configuration and accumulation-safe Lightning gradient clipping;
- explicit fixed-step samplers: Euler, Heun, RK4, reverse-SDE EM, and PC;
- NFE reporting;
- latent-source and source-image metrics are reported separately;
- no-drift / no-mismatch / legacy-network experiment configs;
- the unused full global mixing matrix is not allocated by default
  (`store_full_matrix: false`), while the original eigensystem is retained.

Per your request, this package was **not run or numerically verified** after the
edits. It is intended as a clean starting point based directly on your working
code rather than another independent reimplementation.

## 1. Keep your current Conda environment

Do not replace the working PyTorch/CUDA build. Install only missing packages:

```bash
conda activate <your-existing-env>
pip install -r requirements.txt
```

`requirements.txt` deliberately does not pin/install PyTorch.

## 2. Fastest route: convert your existing WAV dataset to NPZ

If the dataset that already worked is at `data/sim_2src`:

```bash
python data/convert_wav_dataset_to_npz.py \
  --input data/sim_2src \
  --output data/sim_2src_npz \
  --filters data/sim_2src/filters/filters.npy \
  --num-sources 2 \
  --sample-rate 2000 \
  --fft-norm ortho \
  --cache-fft
```

Then train with:

```bash
python train.py \
  path.datasets.mixdata=data/sim_2src_npz \
  path.datasets.filters_path=data/sim_2src_npz/filters.npz
```

The loader uses NPZ automatically if `<split>.npz` exists. Otherwise it falls
back to the old WAV tree, so old commands remain valid.

For less CPU decompression and more disk use, add `--uncompressed` to the
converter. Cached FFTs avoid doing `rfft` for every sample at every epoch.

## 3. Regenerate the paper data with unambiguous latent targets

The new generator saves clean latent sources, source images, mixtures, SNRs,
seeds and cached FFTs together.

```bash
python data/create_dataset_npz.py \
  --out data/mechanical_k2_npz \
  --kind mechanical \
  --sources 2 \
  --train 3400 --val 200 --test 400 \
  --sample-rate 2000 --duration 1.0 \
  --seed 42
```

Three sources:

```bash
python data/create_dataset_npz.py \
  --out data/mechanical_k3_npz \
  --kind mechanical --sources 3 \
  --train 3400 --val 200 --test 400 --seed 42
```

The NPZ files contain:

```text
sources         [N,K,T]       latent clean s_k
source_images   [N,K,Nfft]    h_k * s_k
mixtures        [N,Nfft]      sum_k h_k*s_k + noise
sources_fft     [N,K,F]       optional cached rFFT
mixtures_fft    [N,F]         optional cached rFFT
snr_db          [N]
source_seeds    [N,K]
```

The generator uses train SNR `[1,4]` dB and validation/test `[1,2]` dB by
default. The default FIR family remains the legacy exponentially decaying
filter family. `--filter-model reflections` creates a separate robustness
variant with weak echoes.

### Stationary synthetic music

```bash
python data/create_dataset_npz.py \
  --out data/music_k2_npz \
  --kind music --sources 2 --seed 42
```

These are fixed harmonic stacks over the entire record: no note changes, ADSR,
vibrato or moving transfer functions. That keeps the benchmark aligned with the
full-signal DFT assumption.

## 4. Enhanced network

The default config now uses:

```yaml
_target_: src.models.enhanced_unet.PhysicsConditionedUNet
```

The network receives complex state/mixture features, log magnitude features,
and, for the fixed known kernels,

```text
H^H (H X_t - Y) / (1 + ||H||^2)
```

as a feature. This is not a hard projection and does not change the SDE.
The operator is attached once in `train.py`; there is no SVD inside the network.

To return to your original network:

```bash
python train.py experiment=legacy_network
```

## 5. EMA

EMA is updated once per **optimizer update**, including when gradient
accumulation is used. Its decay warms up toward `0.999`. Both online weights and
EMA state remain in the Lightning checkpoint. Validation and test automatically
use EMA weights and restore the online model afterward.

## 6. Reverse samplers

Configure:

```yaml
model:
  sampler:
    method: euler      # euler | heun | rk4 | em | pc | legacy
    num_steps: 50
    schedule: linear   # linear | quadratic | log
```

- `euler`, `heun`, `rk4`: probability-flow ODE
- `em`: reverse SDE Euler-Maruyama
- `pc`: Langevin corrector + reverse-SDE predictor
- `legacy`: original project sampler path

Sampler output includes NFE. Approximate equal-NFE comparison:

```bash
python scripts/run_sampler_sweep.py \
  --checkpoint /path/to/best.ckpt \
  --experiment paper_k2_npz \
  --budget 50
```

## 7. Paper ablations already exposed

Full model:

```bash
python train.py experiment=paper_k2_npz
```

No terminal mismatch correction:

```bash
python train.py experiment=no_mismatch \
  path.datasets.mixdata=data/mechanical_k2_npz \
  path.datasets.filters_path=data/mechanical_k2_npz/filters.npz
```

No mixing-aware drift, while retaining the same network/data/noise operator:

```bash
python train.py experiment=no_drift \
  path.datasets.mixdata=data/mechanical_k2_npz \
  path.datasets.filters_path=data/mechanical_k2_npz/filters.npz
```

Both off:

```bash
python train.py experiment=no_drift_no_mismatch \
  path.datasets.mixdata=data/mechanical_k2_npz \
  path.datasets.filters_path=data/mechanical_k2_npz/filters.npz
```

Three-source paper settings:

```bash
python train.py experiment=paper_k3
```

## 8. Evaluation

```bash
python evaluate.py \
  experiment=paper_k2_npz \
  checkpoint=/path/to/best.ckpt \
  evaluation_out=evaluation/k2_full
```

Outputs:

```text
summary.json
per_example.npz
```

The summary separates:

- latent-source SI-SDR;
- convolutive source-image SI-SDR;
- nMSE / nMAE on latent spectra;
- failure rate (`mean source SI-SDR < 5 dB`);
- measurement residual;
- NFE.

## 9. Useful runtime overrides

Gradient accumulation without changing effective batch size:

```bash
python train.py \
  datamodule.train.dl_opts.batch_size=4 \
  trainer.accumulate_grad_batches=4
```

Mixed precision can be tried without changing source code:

```bash
python train.py trainer.precision=16-mixed
```

If complex linear algebra on your GPU/build does not support the desired mixed
precision path, keep the default `32-true`.

Resume:

```bash
python train.py resume_from_checkpoint=/path/to/last.ckpt
```

Hydra saves the resolved configuration with every run, so an experiment can be
reconstructed later from its run directory.
