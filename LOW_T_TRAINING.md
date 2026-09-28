# Low-t time-sampling experiment

The fixed-time diagnostic showed that the learned score is much less accurate near `t_epsilon`.
This patch adds two experimental DSM time distributions:

- `model.time_sampling_strategy=log_uniform`
- `model.time_sampling_strategy=mixed`
- `model.time_sampling_log_fraction=0.5` controls the mixed strategy.

The original `uniform` strategy is unchanged.

`log_uniform` and `mixed` intentionally change the training-time weighting; they are not
importance-corrected estimators of the old uniform-time objective.

Run:

```bash
python scripts/run_time_sampling_sweep.py   --strategies uniform,log_uniform,mixed   --alpha 3 --beta 3   --max-epochs 100   --steps 5,10,20,50,100   --seed 2
```

TensorBoard:

```bash
tensorboard --logdir sweeps/time_sampling --port 6006
```

Useful logged diagnostics:
- `train/time_mean`
- `train/time_frac_lt_0p05`
- `train/time_frac_lt_0p02`
