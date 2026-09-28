# CUDA determinism + time-conditioning patch

This patch makes two focused changes to the enhanced project:

1. `PhysicsConditionedUNet` decoder upsampling now uses nearest-neighbor interpolation followed by the existing learned `Conv1d`, instead of `mode="linear"`. This avoids `upsample_linear1d_backward_out_cuda`, which has no deterministic CUDA backward implementation while `Trainer(deterministic=true)` is enabled.
2. The Gaussian Fourier frequencies in `GaussianFourierTimeEmbedding` are now `nn.Parameter(..., requires_grad=False)` rather than a registered buffer. The frequencies remain fixed, but Lightning now includes them in the `Non-trainable params` count. The MLP that maps the fixed Fourier features remains trainable.

No training, tests, smoke tests, or numerical verification were run after applying this patch.
