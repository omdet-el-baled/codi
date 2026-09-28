# Re-coded diffusion core

Run unit tests first:

```bash
python -m pytest
```

The suite checks:
- mixing eigendecomposition against the explicit operator;
- instantaneous drift vs derivative of the integrated drift;
- structured mean propagator vs dense construction;
- structured `S`, `S^H`, `S^{-1}`, and `S^{-H}` operations;
- modal covariance factor against the scalar Lyapunov ODE;
- `B B^H` structured action against the dense diffusion matrix;
- exact reverse endpoint and interval count;
- Euler convergence on a known smooth ODE;
- Heun/RK4 endpoint and NFE accounting;
- rFFT endpoint noise constraints;
- physical/whitened score parameterization equivalence.

The default remains:

```yaml
model.score_parameterization: physical
model.time_sampling_strategy: uniform
```

To train the numerically better-conditioned whitened parameterization:

```bash
python train.py experiment=paper_k2_npz \
  model.score_parameterization=whitened \
  model.time_sampling_strategy=uniform
```

Matched physical-vs-whitened experiment:

```bash
python scripts/run_score_parameterization_sweep.py \
  --parameterizations physical,whitened \
  --alpha 3 --beta 3 \
  --max-epochs 100 \
  --steps 5,10,20,50,100 \
  --seed 2
```

For a final experiment, increase `--max-epochs` and repeat with multiple seeds.
