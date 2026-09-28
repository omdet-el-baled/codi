"""
Abstract SDE base class, Reverse SDE, and related utilities.

Adapted from:
https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py

"""

import abc
import logging
import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import warnings

from .rfft_noise import standard_rfft_noise_like

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Tensor = torch.Tensor
ScoreModel = Callable[[Tensor, Tensor, Tensor], Tensor]


# ---------------------------------------------------------------------------
# Base SDE
# ---------------------------------------------------------------------------


def adapt(vec: Tensor) -> Tensor:
    """Flatten every non-batch dimension while accepting already-flat scores."""
    if vec.ndim < 2:
        raise ValueError(f"Expected a batched tensor, got shape {tuple(vec.shape)}")
    return vec.reshape(vec.shape[0], -1)


class SDE(abc.ABC):
    """
    Abstract base class for Stochastic Differential Equations (SDEs).

    Represents a forward-time SDE of the form:
        dx = f(x, t) dt + g(x, t) dW

    where:
        f  is the drift coefficient
        g  is the diffusion coefficient
        W  is a standard Wiener process

    All methods are designed to operate on mini-batches. Time `t` is always a
    1-D tensor of shape ``(B,)`` matching the batch dimension of `x`.

    Args:
        T (float): Terminal time of the forward SDE (typically 1.0).
        N (int):   Number of discretisation steps for the reverse process.
    """

    def __init__(self, T: float, N: int) -> None:
        if T <= 0:
            raise ValueError(f"Terminal time T must be positive, got {T}.")
        if N < 1:
            raise ValueError(f"Number of steps N must be >= 1, got {N}.")
        super().__init__()
        self.T = T
        self.N = N

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def sde(self, x: Tensor, y: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the drift and diffusion coefficients of the forward SDE.

        Args:
            x: Noisy input tensor of shape ``(B, *spatial)``.
            y: Conditioning tensor (e.g. clean signal or label).
            t: Time tensor of shape ``(B,)`` with values in ``[0, T]``.

        Returns:
            drift:     Tensor matching ``x`` shape — the drift  f(x, t).
            diffusion: Tensor broadcastable to ``x`` — the diffusion g(t).
        """

    @abc.abstractmethod
    def marginal_prob(self, x: Tensor, y: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters of the marginal distribution p_t(x | y).

        For most SDEs this returns the mean and standard deviation of the
        (Gaussian) marginal so that the training objective can be computed
        analytically.

        Args:
            x: Clean input tensor of shape ``(B, *spatial)``.
            y: Conditioning tensor.
            t: Time tensor of shape ``(B,)``.

        Returns:
            mean: Tensor of shape matching ``x``.
            std:  Scalar-broadcastable tensor (often shape ``(B, 1, 1, 1)``).
        """

    @abc.abstractmethod
    def prior_sampling(self, shape: Tuple[int, ...], *args) -> Tensor:
        """
        Draw a sample from the prior distribution p_T(x | args).

        Args:
            shape: Output shape, e.g. ``(B, C, H, W)``.
            *args: Optional conditioning arguments (SDE-specific).

        Returns:
            Sample tensor of the requested shape.
        """

    @abc.abstractmethod
    def prior_logp(self, z: Tensor) -> Tensor:
        """
        Log-density of the prior distribution log p_T(z).

        Used for likelihood computation via the probability-flow ODE.

        Args:
            z: Latent tensor of shape ``(B, *spatial)``.

        Returns:
            Log-probability tensor of shape ``(B,)``.
        """

    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Register SDE-specific CLI arguments on an ``argparse.ArgumentParser``.

        Subclasses should add arguments such as ``--sigma-min``, ``--beta-max``
        etc. and return the modified parser.

        Args:
            parent_parser: An existing ``argparse.ArgumentParser`` instance.

        Returns:
            The modified parser.
        """

    @abc.abstractmethod
    def copy(self) -> "SDE":
        """
        Return a deep copy of this SDE instance.

        Useful when the reverse SDE needs its own independent object without
        sharing mutable state with the forward SDE.

        Returns:
            A new ``SDE`` instance with identical parameters.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def discretize(
        self, x: Tensor, y: Tensor, t: Tensor, *, dt: Optional[float] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Euler-Maruyama discretisation of the forward SDE.

        Computes:
            f_i = drift * dt
            B_i = diffusion * sqrt(dt)

        Override this in subclasses that require a different integrator
        (e.g. Milstein, stochastic Runge-Kutta) or non-uniform step sizes.

        Args:
            x: Noisy input tensor of shape ``(B, *spatial)``.
            y: Conditioning tensor.
            t: Time tensor of shape ``(B,)``.

        Returns:
            f: Deterministic increment of shape matching ``x``.
            G: Stochastic coefficient of shape matching ``x``.
        """
        # ``N`` denotes the number of intervals.  When an explicit step size is
        # not supplied, use T/N rather than assuming T == 1.  Reverse samplers
        # pass their actual interval length explicitly so they terminate at eps.
        step = float(self.T) / float(self.N) if dt is None else float(dt)
        if step <= 0:
            raise ValueError(f"dt must be positive, got {step}.")
        drift, diffusion = self.sde(x, y, t)
        f = drift * step
        B = diffusion * math.sqrt(step)
        return f, B

    def marginal_prob_std(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        """
        Convenience: return only the standard deviation of p_t(x | y).

        Args:
            x: Input tensor.
            y: Conditioning tensor.
            t: Time tensor of shape ``(B,)``.

        Returns:
            Standard deviation tensor broadcastable to ``x``.
        """
        _, std = self.marginal_prob(x, y, t)
        return std

    def time_steps(
        self,
        device: torch.device,
        *,
        eps: float = 1e-5,
        schedule: str = "uniform",
    ) -> Tensor:
        """
        Generate a sequence of reverse-process time steps.

        Args:
            device:   Target device for the output tensor.
            eps:      Small value to avoid evaluating at exactly t = 0.
            schedule: One of ``"uniform"`` (linear spacing) or
                      ``"quadratic"`` (concentrate steps near t = 0, which
                      can improve sample quality for some SDEs).

        Returns:
            1-D tensor of shape ``(N+1,)`` defining exactly ``N`` intervals,
            descending from ``T`` to ``eps``.

        Raises:
            ValueError: If ``schedule`` is not recognised.
        """
        if schedule == "uniform":
            return torch.linspace(self.T, eps, self.N + 1, device=device)
        elif schedule == "quadratic":
            # Steps denser near t = 0 for better score accuracy.
            t = torch.linspace(math.sqrt(self.T), math.sqrt(eps), self.N + 1, device=device)
            return t**2
        else:
            raise ValueError(
                f"Unknown schedule '{schedule}'. Choose 'uniform' or 'quadratic'."
            )

    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        eps: float = 1e-5,
    ) -> Tensor:
        """
        Sample uniformly random time steps for training.

        Args:
            batch_size: Number of time values to sample.
            device:     Target device.
            eps:        Lower bound to avoid t = 0.

        Returns:
            Tensor of shape ``(batch_size,)`` with values in ``[eps, T]``.
        """
        return torch.rand(batch_size, device=device) * (self.T - eps) + eps

    # ------------------------------------------------------------------
    # Reverse SDE factory
    # ------------------------------------------------------------------

    def reverse(
        self,
        score_model: ScoreModel,
        probability_flow: bool = False,
    ) -> "SDE":
        """
        Construct the reverse-time SDE (or ODE for probability flow).

        The reverse SDE is:
            dx = [f(x, t) - g(t)^2 * s_θ(x, t)] dt + g(t) dW̄

        where s_θ is the learned score function and W̄ is a reverse-time
        Wiener process.  When ``probability_flow=True`` the diffusion term
        is set to zero, yielding the probability-flow ODE.

        Args:
            score_model:      A callable ``(x, y, t) -> score`` implementing
                              the learned score function s_θ(x, t).
            probability_flow: If ``True``, build the deterministic ODE
                              instead of the stochastic SDE.

        Returns:
            An ``SDE`` instance whose ``sde`` and ``discretize`` methods
            implement the reverse process.
        """
        # Capture forward-SDE state; use `oself` to avoid shadowing inside RSDE.
        oself = self
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        class RSDE(oself.__class__):
            """
            Reverse-time SDE / probability-flow ODE.

            Constructed via :meth:`SDE.reverse`; do not instantiate directly.
            """

            def __init__(self) -> None:
                # Deliberately bypass the parent __init__ to avoid requiring
                # subclass-specific arguments.  T and N are set explicitly.
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self) -> float:  # type: ignore[override]
                return T

            # -- Forward interface (required by abc but unused in RSDE) -------

            def marginal_prob(self, x, y, t):
                return oself.marginal_prob(x, y, t)

            def prior_sampling(self, shape, *args):
                return oself.prior_sampling(shape, *args)

            def prior_logp(self, z):
                return oself.prior_logp(z)

            @staticmethod
            def add_argparse_args(parent_parser):
                return oself.__class__.add_argparse_args(parent_parser)

            def copy(self):
                return oself.copy().reverse(score_model, probability_flow)

            # -- Reverse-SDE core -------------------------------------------

            def sde(self, x: Tensor, y: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
                """Drift and diffusion of the reverse-time SDE."""
                parts = self.rsde_parts(x, y, t)
                return parts["total_drift"], parts["diffusion"]

            def rsde_parts(self, x: Tensor, y: Tensor, t: Tensor) -> dict:
                """
                Decomposed reverse-SDE terms — useful for diagnostics and logging.

                Returns a dict with keys:
                    ``total_drift``, ``diffusion``, ``sde_drift``,
                    ``sde_diffusion``, ``score_drift``, ``score``.
                """
                sde_drift, sde_diffusion = sde_fn(x, y, t)

                score = adapt(score_model(x, y, t))

                score_drift = torch.einsum(
                    "bmn,bn->bm",
                    -torch.bmm(sde_diffusion, sde_diffusion.transpose(1, 2).conj()),
                    score,
                ) * (0.5 if self.probability_flow else 1.0)
                diffusion = (
                    torch.zeros_like(sde_diffusion)
                    if self.probability_flow
                    else sde_diffusion
                )
                total_drift = sde_drift + score_drift

                return {
                    "total_drift": total_drift,
                    "diffusion": diffusion,
                    "sde_drift": sde_drift,
                    "sde_diffusion": sde_diffusion,
                    "score_drift": score_drift,
                    "score": score,
                }

            def discretize(
                self, x: Tensor, y: Tensor, t: Tensor, *args, **kwargs
            ) -> Tuple[Tensor, Tensor]:
                """One explicit reverse step over the supplied positive interval.

                ``dt`` is the *magnitude* ``t_i - t_{i+1}`` of the reverse
                interval.  The returned deterministic increment already includes
                the negative time direction.  Using the actual interval is
                essential when the solver stops at ``t_epsilon`` rather than 0.
                """
                step = float(kwargs.pop("dt", float(T) / float(N)))
                if step <= 0:
                    raise ValueError(f"dt must be positive, got {step}.")

                forward_drift, diffusion = sde_fn(x, y, t)
                score = adapt(score_model(x, y, t))
                covariance_score = torch.einsum(
                    "bmn,bn->bm",
                    torch.bmm(diffusion, diffusion.transpose(1, 2).conj()),
                    score,
                )
                coefficient = 0.5 if self.probability_flow else 1.0
                reverse_drift = forward_drift - coefficient * covariance_score

                # Reverse integration uses delta_t = -step.
                rev_f = -step * reverse_drift
                if self.probability_flow:
                    rev_B = torch.zeros_like(diffusion)
                else:
                    rev_B = math.sqrt(step) * diffusion
                return rev_f, rev_B

        return RSDE()

    # ------------------------------------------------------------------
    # Validation helpers (available to subclasses)
    # ------------------------------------------------------------------

    def _check_tensor(self, x: Tensor, name: str = "x") -> None:
        """
        Raise ``ValueError`` if ``x`` contains NaN or Inf values.

        Args:
            x:    Tensor to validate.
            name: Human-readable name used in the error message.
        """
        if not torch.isfinite(x).all():
            raise ValueError(
                f"SDE received non-finite values in '{name}'. "
                "Check your inputs or SDE parameters."
            )

    def _check_time(self, t: Tensor) -> None:
        """
        Raise ``ValueError`` if any time value is outside ``[0, T]``.

        Args:
            t: Time tensor of shape ``(B,)``.
        """
        if t.min() < 0 or t.max() > self.T:
            raise ValueError(
                f"Time values must be in [0, {self.T}], "
                f"got min={t.min().item():.4f}, max={t.max().item():.4f}."
            )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(T={self.T}, N={self.N})"


class VEConvMixSDE(SDE):
    """
    Variance-Exploding SDE with a Convolutional Mixing structure,
    designed for source separation in the complex (frequency) domain.

    Forward SDE (conceptual form):
        dx = f(x, A, t) dt + sigma(t) dW

    where:
        A        is the mixing matrix built from convolution kernels
        sigma(t) grows from sigma_min to sigma_max as t -> T
        alpha    controls the exponential decay rate of the drift
        beta     controls the decay rate for zero-eigenvalue modes

    Args:
        ndim      (int):   Spatial/feature dimension of each source.
                           (reserved for future use -- stored but not yet active)
        alpha     (float): Decay rate in the drift exponent. Must be > 0.
        beta      (float): Decay rate for zero-eigenvalue modes. Must be > 0.
        sigma_min (float): Minimum noise level (at t=0). Must be > 0.
        sigma_max (float): Maximum noise level (at t=T). Must be > sigma_min.
        steps     (int):   Number of solver sub-steps inside each SDE step.
                           (reserved for future use -- stored but not yet active)
        T         (float): Terminal time of the forward process. Must be > 0.
        N         (int):   Number of reverse-process discretisation steps.
                           Defaults to 1000.
    """

    @staticmethod
    def add_argparse_args(parser):
        """Register VEConvMixSDE-specific CLI arguments."""
        parser.add_argument(
            "--sde-n",
            type=int,
            default=1000,
            help="Number of reverse discretisation steps (default: 1000).",
        )
        parser.add_argument(
            "--theta",
            type=float,
            default=1.5,
            help="Theta parameter (default: 1.5).",
        )
        parser.add_argument(
            "--sigma-min",
            type=float,
            default=0.05,
            help="Minimum sigma for the noise schedule (default: 0.05).",
        )
        parser.add_argument(
            "--sigma-max",
            type=float,
            default=0.5,
            help="Maximum sigma for the noise schedule (default: 0.5).",
        )
        return parser

    def __init__(
        self,
        feature_dim: int,
        drift_decay_rate: float,
        zero_mode_decay_rate: float,
        sigma_min: float,
        sigma_max: float,
        steps: int,
        terminal_time: float,
        num_reverse_steps: int = 1000,
        drift_mode: str = "mixing",
        store_full_matrix: bool = False,
    ) -> None:
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be >= 1, got {feature_dim}.")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}.")

        # -- 2. Base-class init (validates T > 0, N >= 1, sets self.T/self.N) -
        super().__init__(terminal_time, num_reverse_steps)

        # -- 3. Store SDE-specific parameters ---------------------------------
        self.feature_dim = feature_dim
        self.drift_decay_rate = drift_decay_rate
        self.zero_mode_decay_rate = zero_mode_decay_rate
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.steps = steps
        self.drift_mode = str(drift_mode)
        if self.drift_mode not in {"mixing", "none"}:
            raise ValueError("drift_mode must be 'mixing' or 'none'")
        self.store_full_matrix = bool(store_full_matrix)

        # -- 4. Derived constants (computed once, reused everywhere) ----------
        self.ratiosig = sigma_max / sigma_min
        self.logsig = math.log(self.ratiosig)

        # -- 5. Mixing-matrix tensors (populated by get_mix_mat) --------------
        self.A = None  # Full mixing matrix \(\mathcal{A}\) (K*L, K*L)
        self.eigenvalues = None  # Eigenvalues \(\{\lambda_i\}\) (K*L,)
        self.eigenvector_matrix = (
            None  # Eigenvector matrix \(\widetilde{\mathbf{V}}\) (K*L, K*L)
        )
        self.inv_eigenvector_matrix = (
            None  # Inverse eigenvectors \(\widetilde{\mathbf{V}}^{-1}\) (K*L, K*L)
        )
        self.num_fft_bins = None  # number of non-redundant rFFT bins
        self.real_n_fft = None  # full real-signal FFT length; set by train/eval entry points

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def _ensure_mix_mat_ready(self) -> None:
        """
        Raise a descriptive RuntimeError if get_mix_mat() has not been called.

        Raises:
            RuntimeError: If the mixing matrix has not been built yet.
        """
        if self.A is None:
            raise RuntimeError(
                "Mixing matrix not initialised. "
                "Call get_mix_mat(kernels, num_fft_bins) before using this method."
            )

    @property
    def _mix_mat_device(self) -> torch.device:
        """
        Return the device where the mixing-matrix tensors currently live.

        Raises:
            RuntimeError: If the mixing matrix has not been built yet.
        """
        self._ensure_mix_mat_ready()
        return self.eigenvalues.device

    def send_to(self, device: torch.device) -> None:
        """
        Move all mixing-matrix tensors to ``device`` (in-place).

        Args:
            device (torch.device | str): Target device.

        Returns:
            self — for optional method chaining.

        Raises:
            RuntimeError: If get_mix_mat() has not been called yet.
        """
        self._ensure_mix_mat_ready()

        if self.A.device != device:
            self.A = self.A.to(device)
        if self.eigenvalues.device != device:
            self.eigenvalues = self.eigenvalues.to(device)
        if self.eigenvector_matrix.device != device:
            self.eigenvector_matrix = self.eigenvector_matrix.to(device)
        if self.inv_eigenvector_matrix.device != device:
            self.inv_eigenvector_matrix = self.inv_eigenvector_matrix.to(device)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def _build_zero_eigvecs(
        self,
        kernels: Tensor,
        K: int,
        L: int,
    ) -> Tensor:
        """
        Build the eigenvector columns corresponding to zero eigenvalues.

        For each free variable index f in {1, ..., K-1}, a zero-eigenvalue
        eigenvector is constructed as a weighted difference between source f
        and source 0, chosen so that A @ v = 0.

        The weight for source 0 is:
            alpha_0[freq] = -sum_{k=1}^{K-1} kernel[k, freq] * alpha_k[freq]
                            / kernel[0, freq]

        This gives L*(K-1) linearly independent null-space vectors.

        Args:
            kernels: Complex kernel tensor of shape (K, L).
            K:       Number of sources / microphones.
            L:       Number of frequency bins (FFT size).

        Returns:
            V_zero: Tensor of shape (K*L, L*(K-1)) — null-space eigenvectors.
        """
        V_zero = torch.zeros(K * L, L * (K - 1), dtype=kernels.dtype)

        for col_block, f in enumerate(range(1, K)):
            # alpha coefficients: shape (K, L)
            alpha = torch.zeros((K, L), dtype=kernels.dtype)
            alpha[f, :] = 1.0
            # Enforce A @ v = 0 by solving for alpha[0]
            alpha[0, :] = (
                -torch.sum(kernels[1:, :] * alpha[1:, :], dim=0) / kernels[0, :]
            )
            # Place diag(alpha[block]) in the correct block row and column
            for block in range(K):
                r0, r1 = block * L, (block + 1) * L
                c0, c1 = col_block * L, (col_block + 1) * L
                V_zero[r0:r1, c0:c1] = torch.diag(alpha[block])

        return V_zero

    def get_mix_mat(self, kernels: Tensor, num_fft_bins: int):
        r"""
        Build the convolutive mixing matrix A and its eigendecomposition.

        Mathematical structure
        ----------------------
        Given K sources and L = num_fft_bins frequency bins, the mixing matrix A is
        a (K*L) x (K*L) block matrix:

            A = [ diag(h_1)   diag(h_1)  ...  diag(h_1) ]
                [ diag(h_2)   diag(h_2)  ...  diag(h_2) ]
                [   ...                                 ]
                [ diag(h_K)   diag(h_K)  ...  diag(h_K) ]

        where h_i is the frequency-domain representation of the i-th filter.

        Because A is constructed from rank-1 outer products in the frequency
        domain, it has a structured eigendecomposition:

            Non-zero eigenvalues (L of them):
                lambda_l = sum_k  kernel[k, l]      for l = 0, ..., L-1

            Zero eigenvalues (L*(K-1) of them):
                Null-space spanned by weighted-difference vectors (see
                _build_zero_eigvecs for construction details).

        The full eigenvector matrix \(\widetilde{\mathbf{V}}\) satisfies:  \(\mathcal{A} = \widetilde{\mathbf{V}} \text{diag}(\lambda) \widetilde{\mathbf{V}}^{-1}\)

        Args:
            kernels (Tensor): Frequency-domain filter coefficients of shape
                              (K, L), where K = number of sources and
                              L = number of frequency bins. Will be cast to
                              complex64 internally.
            num_fft_bins (int): FFT size used to produce ``kernels``. Stored as
                              ``self.num_fft_bins``.

        Returns:
            self — for optional method chaining.

        Raises:
            ValueError: If ``kernels`` is not 2-D.
            ValueError: If ``num_fft_bins`` does not match ``kernels.shape[1]``.
            ValueError: If any frequency bin of ``kernels[0]`` is zero.
            RuntimeError: If the eigenvector matrix is singular.
        """
        # -- Input validation -------------------------------------------------
        if kernels.ndim != 2:
            raise ValueError(f"kernels must be 2-D, got shape {tuple(kernels.shape)}.")

        K, L = kernels.shape

        if K < 2:
            raise ValueError(f"Need at least 2 sources (K >= 2), got K={K}.")

        if num_fft_bins != L:
            raise ValueError(
                f"num_fft_bins ({num_fft_bins}) must match kernels.shape[1] ({L})."
            )

        if (kernels[0] == 0).any():
            raise ValueError("kernels[0] contains zero entries.")

        # -- Cast to complex64 -------------------------------------------------
        kernels = kernels.to(torch.complex64)
        self.num_fft_bins = num_fft_bins

        # -- Build full mixing matrix A  (K*L, K*L) ---------------------------
        if self.store_full_matrix:
            diags = torch.diag_embed(kernels)
            row_blocks = diags.unsqueeze(0).repeat(K, 1, 1, 1)
            self.A = row_blocks.permute(0, 2, 1, 3).reshape(K * L, K * L)
        else:
            # The full A is never needed by training/sampling once the analytic
            # eigensystem is available. Keep a zero-size sentinel to preserve
            # the legacy readiness/device interface without allocating O((KF)^2).
            self.A = torch.empty(0, dtype=kernels.dtype, device=kernels.device)

        # -- Eigenvalues ------------------------------------------------------
        eigvals_nonzero = kernels.sum(dim=0)
        eigvals_zero = torch.zeros(L * (K - 1), dtype=kernels.dtype)
        self.eigenvalues = torch.cat([eigvals_nonzero, eigvals_zero])

        # -- Eigenvectors -----------------------------------------------------
        V_nonzero = torch.eye(L).repeat(K, 1)
        V_zero = self._build_zero_eigvecs(kernels, K, L)
        self.eigenvector_matrix = torch.cat([V_nonzero, V_zero], dim=1)

        # -- Singularity check ------------------------------------------------
        cond = torch.linalg.cond(self.eigenvector_matrix).real.item()
        if not torch.isfinite(torch.tensor(cond)):
            raise RuntimeError("Eigenvector matrix is singular.")

        logger.info(
            f"Mixing operator initialized conceptually as {(K * L, K * L)}; full_matrix={self.store_full_matrix}. Condition Number: {cond:.2e}"
        )
        if cond > 1e6:
            warnings.warn(
                f"High condition number ({cond:.2e}) detected. Inversion may be unstable.",
                UserWarning,
            )

        # -- Invert eigenvector matrix ----------------------------------------
        self.inv_eigenvector_matrix = torch.linalg.inv(self.eigenvector_matrix)

        return self

    def _eigval_Ft_integral(self, t: Tensor) -> Tensor:
        """

        Returns:
            Tensor of shape (B, K*L) — one scalar per (batch, eigenvalue).
        """
        if self.drift_mode == "none":
            return torch.zeros((t.shape[0], self.eigenvalues.shape[0]), device=t.device, dtype=self.eigenvalues.dtype)
        # t[:, None] broadcasts over eigenvalues -> shape (B, K*L)
        is_zero = self.eigenvalues == 0.0  # (K*L,)

        # Safe log: add eps to zero eigenvalues to avoid log(0) = -inf
        log_eig = torch.log(self.eigenvalues + is_zero * 1e-4)  # (K*L,)

        # Non-zero branch: exp( log(lambda) * (1 - exp(-alpha * t)) )
        nonzero_vals = log_eig * (1.0 - torch.exp(-self.drift_decay_rate * t[:, None]))

        # Zero branch: exp(-beta * t)
        zero_vals = -self.zero_mode_decay_rate * t[:, None].expand(
            -1, self.eigenvalues.shape[0]
        )

        return torch.where(is_zero, zero_vals, nonzero_vals)  # (B, K*L)

    def _eigval_Ft(self, t: Tensor) -> Tensor:
        """
        Returns:
            Tensor of shape (B, K*L) — one scalar per (batch, eigenvalue).
        """
        if self.drift_mode == "none":
            return torch.zeros((t.shape[0], self.eigenvalues.shape[0]), device=t.device, dtype=self.eigenvalues.dtype)
        # t[:, None] broadcasts over eigenvalues -> shape (B, K*L)
        is_zero = self.eigenvalues == 0.0  # (K*L,)

        # Safe log: add eps to zero eigenvalues to avoid log(0) = -inf
        log_eig = torch.log(self.eigenvalues + is_zero * 1e-4)  # (K*L,)

        # Non-zero branch: phi'(t) for
        #   phi(t) = log(lambda) * (1 - exp(-alpha*t))
        # hence
        #   phi'(t) = alpha * log(lambda) * exp(-alpha*t).
        #
        # The alpha prefactor is essential: without it, integrating this
        # instantaneous drift gives (log(lambda)/alpha)*(1-exp(-alpha*t)),
        # which is inconsistent with _eigval_Ft_integral() for alpha != 1.
        nonzero_vals = (
            self.drift_decay_rate
            * log_eig
            * torch.exp(-self.drift_decay_rate * t[:, None])
        )  # (B, K*L)

        # Zero branch: -beta (constant, broadcast over batch and eigenvalue dims)
        zero_vals = torch.full_like(nonzero_vals, fill_value=-self.zero_mode_decay_rate)

        return torch.where(is_zero, zero_vals, nonzero_vals)  # (B, K*L)

    def get_Ft_integral(self, t: Tensor) -> Tensor:
        """
        Returns:
            Tensor of shape (B, K*L, K*L) — one full matrix per batch element.

        """

        if t.device != self.eigenvalues.device:
            raise ValueError(
                f"t is on device '{t.device}' but mixing matrix is on "
                f"'{self.eigenvalues.device}'. Call send_to() first."
            )

        # -- Per-eigenvalue scalars: shape (B, K*L) --------------------------------
        f_diag_integral = self._eigval_Ft_integral(t)  # (B, K*L)

        # -- Reconstruct full matrix: \(\widetilde{\mathbf{V}} \text{diag}(\mathbf{f}) \widetilde{\mathbf{V}}^{-1}\) ------------------
        #   step 1: scale columns of \(\widetilde{\mathbf{V}}\) by f_diag  ->  V_scaled: (B, K*L, K*L)
        #   step 2: right-multiply by \(\widetilde{\mathbf{V}}^{-1}\)      ->  F_t:      (B, K*L, K*L)
        V = self.eigenvector_matrix  # (K*L, K*L)
        V_inv = self.inv_eigenvector_matrix  # (K*L, K*L)

        # V_scaled[b, j, i] = V[j, i] * f_diag[b, i]
        V_scaled = V.unsqueeze(0) * f_diag_integral.unsqueeze(1)  # (B, K*L, K*L)
        F_t_integral = V_scaled @ V_inv  # (B, K*L, K*L)

        # -- Finite-value check ---------------------------------------------------
        if not torch.isfinite(F_t_integral).all():
            raise RuntimeError(
                "get_Ft_integral() produced non-finite values. "
                "Check for near-zero eigenvalues or extreme t values."
            )

        return F_t_integral  # (B, K*L, K*L)

    def get_exp_Ft_integral(self, t: Tensor) -> Tensor:
        """
        Returns:
            Tensor of shape (B, K*L, K*L) — one full matrix per batch element.
        """

        if t.device != self.eigenvalues.device:
            raise ValueError(
                f"t is on device '{t.device}' but mixing matrix is on "
                f"'{self.eigenvalues.device}'. Call send_to() first."
            )

        # -- Per-eigenvalue scalars: shape (B, K*L) --------------------------------
        exp_f_diag_integral = torch.exp(self._eigval_Ft_integral(t))  # (B, K*L)

        # -- Reconstruct full matrix: \(\widetilde{\mathbf{V}} \text{diag}(\exp(\mathbf{f})) \widetilde{\mathbf{V}}^{-1}\) ------------------
        #   step 1: scale columns of \(\widetilde{\mathbf{V}}\) by exp_f_diag  ->  V_scaled: (B, K*L, K*L)
        #   step 2: right-multiply by \(\widetilde{\mathbf{V}}^{-1}\)      ->  exp_F_t:      (B, K*L, K*L)
        V = self.eigenvector_matrix  # (K*L, K*L)
        V_inv = self.inv_eigenvector_matrix  # (K*L, K*L)

        # V_scaled[b, j, i] = V[j, i] * exp_f_diag[b, i]
        V_scaled = V.unsqueeze(0) * exp_f_diag_integral.unsqueeze(1)  # (B, K*L, K*L)
        exp_F_t_integral = V_scaled @ V_inv  # (B, K*L, K*L)

        # -- Finite-value check ---------------------------------------------------
        if not torch.isfinite(exp_F_t_integral).all():
            raise RuntimeError(
                "get_exp_Ft_integral() produced non-finite values. "
                "Check for near-zero eigenvalues or extreme t values."
            )

        return exp_F_t_integral  # (B, K*L, K*L)

    def get_Ft(self, t: Tensor) -> Tensor:
        """
        Returns:
            Tensor of shape (B, K*L, K*L) — one full matrix per batch element.
        """

        if t.device != self.eigenvalues.device:
            raise ValueError(
                f"t is on device '{t.device}' but mixing matrix is on "
                f"'{self.eigenvalues.device}'. Call send_to() first."
            )

        # -- Per-eigenvalue scalars: shape (B, K*L) --------------------------------
        f_diag = self._eigval_Ft(t)  # (B, K*L)

        # -- Reconstruct full matrix: \(\widetilde{\mathbf{V}} \text{diag}(\mathbf{f}) \widetilde{\mathbf{V}}^{-1}\) ------------------
        V = self.eigenvector_matrix  # (K*L, K*L)
        V_inv = self.inv_eigenvector_matrix  # (K*L, K*L)

        V_scaled = V.unsqueeze(0) * f_diag.unsqueeze(1)  # (B, K*L, K*L)
        F_t = V_scaled @ V_inv  # (B, K*L, K*L)

        # -- Finite-value check ---------------------------------------------------
        if not torch.isfinite(F_t).all():
            raise RuntimeError(
                "get_Ft() produced non-finite values. "
                "Check for near-zero eigenvalues or extreme t values."
            )

        return F_t  # (B, K*L, K*L)

    def apply_Ft(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Computes F(t) @ x without constructing F(t).
        """

        if t.device != self.eigenvalues.device:
            raise ValueError(
                f"t is on device '{t.device}' but mixing matrix is on "
                f"'{self.eigenvalues.device}'. Call send_to() first."
            )

        # eigenvalues in spectral domain
        f = self._eigval_Ft(t)  # (B, N)

        # V^{-1} x
        x_eig = torch.matmul(
            self.inv_eigenvector_matrix.unsqueeze(0), x.unsqueeze(-1)
        ).squeeze(-1)

        # apply diagonal in eigen space
        x_eig = x_eig * f

        # back transform
        out = torch.matmul(
            self.eigenvector_matrix.unsqueeze(0), x_eig.unsqueeze(-1)
        ).squeeze(-1)

        return out

    def sde(self, x: Tensor, y: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Drift and diffusion coefficients of the forward VEConvMix SDE.

        The forward SDE in the original (non-eigenbasis) domain is:

            dx = F(t) @ x  dt  +  B(t) dW

        where:
            F(t)     is the instantaneous mixing-rate matrix (Step 4)
            B(t) is the noise schedule:
                        B(t) = sigma(t) \(\widetilde{\mathbf{V}}\)
                        sigma(t) = sigma_min * (sigma_max/sigma_min)^(t/T)
                                = sigma_min * ratiosig^(t/T)

        Args:
            x:
            y:
            t: Time tensor of shape (B,), values in [0, T].

        Returns:
            drift:     Tensor of shape (B, K*L) — F(t) @ x per batch element.
            diffusion: Tensor of shape (B, K*L, K*L ) — Matrix per batch element.
        """

        # -- Drift -----------------------------------------------------

        drift = self.apply_Ft(x, t)

        # -- Diffusion---------------------------------
        # -- Noise schedule: sigma(t) = sigma_min * (sigma_max/sigma_min)^(t/T) --
        # Equivalent to: exp( log(sigma_min) + (t/T) * log(sigma_max/sigma_min) )
        sigma_t = self.sigma_min * (self.ratiosig ** (t / self.T))  # (B,)
        diff = (
            sigma_t * np.sqrt(2 * self.logsig)
        )  # (B, ) diffusion scalar. The matrix \(\mathbf{B}_t\) is \(g(t)\widetilde{\mathbf{V}}\)

        diffusion = diff[:, None, None] * self.eigenvector_matrix

        return drift, diffusion

    def _std(self, t: Tensor, chunk_size: int = 512) -> Tensor:
        """Return the diagonal modal covariance factor ``s_i(t)``.

        The covariance factor is represented as

            S_t = V diag(s(t)),

        where ``V`` is the (generally non-unitary) eigenvector matrix of the
        mixing operator.  For each mode,

            q_i(t) = |s_i(t)|^2

        satisfies

            dq_i/dt = 2 Re(phi_i'(t)) q_i + g(t)^2,
            q_i(0) = 0.

        Hence

            q_i(t) = exp(2 Re(phi_i(t)))
                     * integral_0^t exp(-2 Re(phi_i(u))) g(u)^2 du.

        The factor phase can be chosen as ``Im(phi_i(t))``.  This is the
        phase obtained directly from the scalar factor ODE when ``g(t)`` is
        real.  Importantly, we do *not* integrate ``Im(phi_i)`` itself.

        Numerical integration is performed on a normalized grid u in [0,1]
        with s=t*u.  This keeps the implementation correct for arbitrary
        terminal time ``T`` and avoids constructing any dense covariance
        matrix.
        """
        self._ensure_mix_mat_ready()
        if t.ndim != 1:
            raise ValueError(f"t must have shape (B,), got {tuple(t.shape)}")
        if t.device != self.eigenvalues.device:
            raise ValueError(
                f"t is on '{t.device}' but eigenvalues are on "
                f"'{self.eigenvalues.device}'. Call send_to() first."
            )
        if torch.any(t < 0) or torch.any(t > self.T):
            raise ValueError(f"t must lie in [0,{self.T}].")

        # A modest but deterministic quadrature grid. ``steps`` is retained as
        # the user-facing covariance-integration resolution.
        q_steps = max(8, int(self.steps))
        u = torch.linspace(0.0, 1.0, q_steps, device=t.device, dtype=t.dtype)
        s_grid = t[:, None] * u[None, :]  # [B,Q]

        # VE diffusion coefficient squared:
        # g(s)^2 = 2 log(r) sigma_min^2 r^(2 s / T).
        g2 = (
            2.0
            * float(self.logsig)
            * float(self.sigma_min) ** 2
            * torch.pow(
                torch.as_tensor(float(self.ratiosig), device=t.device, dtype=t.dtype),
                2.0 * s_grid / float(self.T),
            )
        )  # [B,Q]

        B = t.shape[0]
        D = self.eigenvalues.numel()
        out = torch.empty(B, D, device=t.device, dtype=self.eigenvalues.dtype)

        alpha = float(self.drift_decay_rate)
        beta = float(self.zero_mode_decay_rate)

        for j0 in range(0, D, int(chunk_size)):
            j1 = min(j0 + int(chunk_size), D)
            lam = self.eigenvalues[j0:j1]  # [C]
            is_zero = lam.abs() == 0

            # Avoid evaluating log(0), even on the branch later discarded by
            # torch.where. This prevents hidden NaN/Inf propagation.
            safe_lam = torch.where(is_zero, torch.ones_like(lam), lam)
            log_lam = torch.log(safe_lam)  # [C]

            s3 = s_grid[:, None, :]  # [B,1,Q]
            phi_nonzero = log_lam[None, :, None] * (
                1.0 - torch.exp(-alpha * s3)
            )
            phi_zero = (-beta * s3).to(phi_nonzero.dtype).expand(-1, j1 - j0, -1)
            phi = torch.where(is_zero[None, :, None], phi_zero, phi_nonzero)

            # integral_0^t f(s) ds = t * integral_0^1 f(t*u) du
            integrand = torch.exp(-2.0 * phi.real) * g2[:, None, :]
            integral = t[:, None] * torch.trapezoid(integrand, x=u, dim=-1)

            phi_t = phi[:, :, -1]
            q = torch.exp(2.0 * phi_t.real) * integral
            q = q.clamp_min(0.0)

            phase = phi_t.imag
            out[:, j0:j1] = torch.sqrt(q).to(out.dtype) * torch.exp(1j * phase)

        if not torch.isfinite(out.real).all() or not torch.isfinite(out.imag).all():
            raise RuntimeError(
                "_std() produced non-finite values. Check the SDE parameters "
                "and mixing eigenvalues."
            )
        return out

    # ------------------------------------------------------------------
    # Structured covariance-factor operations
    # ------------------------------------------------------------------

    def modal_std(self, t: Tensor) -> Tensor:
        """Public alias for the diagonal modal factor ``s(t)``."""
        return self._std(t)

    def apply_S(self, z: Tensor, t: Tensor, std_modes: Optional[Tensor] = None) -> Tensor:
        """Apply ``S_t = V diag(s(t))`` without materialising ``S_t``."""
        self.send_to(z.device)
        s = self._std(t) if std_modes is None else std_modes
        if s.shape != z.shape:
            raise ValueError(f"std_modes shape {tuple(s.shape)} != z shape {tuple(z.shape)}")
        return torch.matmul(
            self.eigenvector_matrix.unsqueeze(0),
            (s * z).unsqueeze(-1),
        ).squeeze(-1)

    def apply_SH(self, x: Tensor, t: Tensor, std_modes: Optional[Tensor] = None) -> Tensor:
        """Apply ``S_t^H`` without materialising ``S_t``."""
        self.send_to(x.device)
        s = self._std(t) if std_modes is None else std_modes
        vh_x = torch.matmul(
            self.eigenvector_matrix.conj().transpose(0, 1).unsqueeze(0),
            x.unsqueeze(-1),
        ).squeeze(-1)
        return s.conj() * vh_x

    def solve_S(self, rhs: Tensor, t: Tensor, std_modes: Optional[Tensor] = None) -> Tensor:
        """Solve ``S_t z = rhs`` for modal noise coordinate ``z``."""
        self.send_to(rhs.device)
        s = self._std(t) if std_modes is None else std_modes
        modal_rhs = torch.matmul(
            self.inv_eigenvector_matrix.unsqueeze(0),
            rhs.unsqueeze(-1),
        ).squeeze(-1)
        tiny = torch.finfo(s.real.dtype).tiny
        if torch.any(s.abs() <= tiny):
            raise RuntimeError("S_t is numerically singular at the requested time.")
        return modal_rhs / s

    def solve_SH(self, rhs: Tensor, t: Tensor, std_modes: Optional[Tensor] = None) -> Tensor:
        """Solve ``S_t^H x = rhs`` for the physical score ``x``."""
        self.send_to(rhs.device)
        s = self._std(t) if std_modes is None else std_modes
        tiny = torch.finfo(s.real.dtype).tiny
        if torch.any(s.abs() <= tiny):
            raise RuntimeError("S_t^H is numerically singular at the requested time.")
        modal = rhs / s.conj()
        # (V^H)^-1 = V^-H = (V^-1)^H
        return torch.matmul(
            self.inv_eigenvector_matrix.conj().transpose(0, 1).unsqueeze(0),
            modal.unsqueeze(-1),
        ).squeeze(-1)

    def apply_diffusion_covariance(self, score: Tensor, t: Tensor) -> Tensor:
        """Apply ``B_t B_t^H`` to ``score`` without forming the covariance."""
        self.send_to(score.device)
        vh_score = torch.matmul(
            self.eigenvector_matrix.conj().transpose(0, 1).unsqueeze(0),
            score.unsqueeze(-1),
        ).squeeze(-1)
        g2 = (
            2.0
            * float(self.logsig)
            * float(self.sigma_min) ** 2
            * torch.pow(
                torch.as_tensor(float(self.ratiosig), device=t.device, dtype=t.dtype),
                2.0 * t / float(self.T),
            )
        )
        vvhs = torch.matmul(
            self.eigenvector_matrix.unsqueeze(0),
            vh_score.unsqueeze(-1),
        ).squeeze(-1)
        return g2[:, None] * vvhs

    def apply_exp_Ft(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Computes F(t) @ x without constructing F(t).
        """

        if t.device != self.eigenvalues.device:
            raise ValueError(
                f"t is on device '{t.device}' but mixing matrix is on "
                f"'{self.eigenvalues.device}'. Call send_to() first."
            )

        # 1. spectral exponential (B, N)
        exp_f = torch.exp(self._eigval_Ft_integral(t))

        # 2. V^{-1} x
        x_eig = torch.matmul(
            self.inv_eigenvector_matrix.unsqueeze(0), x.unsqueeze(-1)
        ).squeeze(-1)

        # 3. apply diagonal in eigen space
        x_eig = x_eig * exp_f

        # 4. back transform
        out = torch.matmul(
            self.eigenvector_matrix.unsqueeze(0), x_eig.unsqueeze(-1)
        ).squeeze(-1)

        return out

    def marginal_prob(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the Gaussian marginal mean and a dense covariance factor.

        This compatibility method materialises ``S_t``.  Training and sampling
        should prefer :meth:`modal_std`, :meth:`apply_S`, :meth:`apply_SH`, and
        the corresponding solve methods, which avoid O(D^2) batch tensors.
        """
        self.send_to(x.device)
        std_modes = self._std(t).contiguous()
        mean = self.apply_exp_Ft(x, t)
        factor = self.eigenvector_matrix.unsqueeze(0) * std_modes.unsqueeze(1)
        return mean, factor

    def prior_sampling(self, shape, y):
        """Sample the practical terminal law N_C([y,...,y], Sigma_T)."""
        self.send_to(y.device)
        y_rep = y.repeat(1, self.feature_dim)
        t = torch.full(
            (y.shape[0],),
            float(self.T),
            device=y.device,
            dtype=y.real.dtype,
        )
        std_modes = self._std(t)
        z = standard_rfft_noise_like(
            y_rep,
            num_sources=self.feature_dim,
            n_fft=self.real_n_fft,
        )
        return y_rep + self.apply_S(z, t, std_modes=std_modes)

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for VESDE not yet implemented!")

    def copy(self):
        new_sde = VEConvMixSDE(
            self.feature_dim,
            self.drift_decay_rate,
            self.zero_mode_decay_rate,
            self.sigma_min,
            self.sigma_max,
            self.steps,
            terminal_time=self.T,
            num_reverse_steps=self.N,
            drift_mode=self.drift_mode,
            store_full_matrix=self.store_full_matrix,
        )
        new_sde.real_n_fft = self.real_n_fft

        # Copy mixing matrix state if it exists
        if self.A is not None:
            new_sde.A = self.A.clone()
            new_sde.eigenvalues = self.eigenvalues.clone()
            new_sde.eigenvector_matrix = self.eigenvector_matrix.clone()
            new_sde.inv_eigenvector_matrix = self.inv_eigenvector_matrix.clone()
            new_sde.num_fft_bins = self.num_fft_bins
        return new_sde
