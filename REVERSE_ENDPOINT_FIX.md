# Reverse endpoint fix

All reverse paths now terminate exactly at `model.time_epsilon` and never take a hidden update from `t_epsilon` toward zero.

Key changes:

- Fixed Euler/Heun/RK4/EM/PC use `N+1` grid points for exactly `N` intervals from `T` to `t_epsilon`.
- Legacy PC samplers use the actual interval `dt = t_i - t_{i+1}` instead of an implicit `1/N`.
- `RSDE.discretize` uses the explicit interval length and the continuous reverse drift.
- The black-box ODE returns the state at `t_epsilon`; endpoint denoising after `t_epsilon` is disabled.
- `DCSModel.separate` rejects attempts to override the endpoint away from `model.time_epsilon`.
- `SDE.time_steps` now returns `N+1` points for `N` reverse steps.
- Added `scripts/check_reverse_endpoint.py`.

This patch retains the corrected non-zero-mode drift `alpha * log(lambda) * exp(-alpha*t)`.

Runtime enforcement:

- `DCSModel.separate()` records and verifies the sampler's final time.
- `evaluate.py` aborts if the final reverse time differs from `model.time_epsilon` and writes both values to `summary.json`.
