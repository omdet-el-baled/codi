# Reverse dynamics fix

This patch fixes three consistency issues in the reverse dynamics.

1. `VEConvMixSDE._eigval_Ft()` now uses
   `alpha * log(lambda) * exp(-alpha*t)`, which is the derivative of
   `phi(t)=log(lambda)*(1-exp(-alpha*t))`.  The previous implementation
   omitted the `alpha` prefactor, so alpha=1 was special while alpha>1
   produced a reverse drift inconsistent with the training marginal.

2. The legacy `Predictor` now forwards the requested `probability_flow`
   flag into `sde.reverse(...)`.

3. The legacy reverse-discretization path now uses `-f` for the forward
   drift increment when stepping backward in time.

The newer fixed-step Euler/Heun/RK4 samplers already use a negative time
step explicitly, so only item 1 changes their dynamics.
