# rFFT endpoint-noise fix

This patch makes stochastic noise consistent with the one-sided FFT representation of real-valued signals.

For interior rFFT bins, noise is proper complex Gaussian CN(0,1), so real and imaginary parts each have variance 1/2.
For DC and, when the full FFT length is even, Nyquist, noise is real N(0,1).

The same rule is used in:

- forward DSM perturbations in `DCSModel.sample_prior`;
- terminal prior initialization in `VEConvMixSDE.prior_sampling`;
- fixed-step reverse-SDE / predictor-corrector stochastic increments;
- legacy stochastic predictors.

After adding the finite-T mismatch shift during training, DC/Nyquist are projected back to the real rFFT subspace to remove numerical imaginary endpoint components.

The score network's existing DC/Nyquist real-output constraint is retained.

No training, numerical verification, or tests were run after this edit, per the user's instruction.
