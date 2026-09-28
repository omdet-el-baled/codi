"""Complex spectral score-model wrapper."""
from __future__ import annotations

import torch
from hydra.utils import instantiate


class ScoreModel(torch.nn.Module):
    def __init__(
        self,
        num_sources,
        backbone_args,
        transform="exponent",
        abs_exponent=1.0,
        factor=1.0,
        enforce_real_rfft_bins=True,
    ):
        super().__init__()
        self.backbone = instantiate(backbone_args)
        self.num_sources = int(num_sources)
        self.transform = str(transform)
        self.abs_exponent = float(abs_exponent)
        self.factor = float(factor)
        self.enforce_real_rfft_bins = bool(enforce_real_rfft_bins)
        self.n_fft = None

    @torch.no_grad()
    def set_operator(self, kernels: torch.Tensor, n_fft: int | None = None):
        """Pass fixed frequency responses to backbones that use physics features."""
        self.n_fft = int(n_fft) if n_fft is not None else None
        if hasattr(self.backbone, "set_operator"):
            self.backbone.set_operator(kernels, n_fft=n_fft)

    def transform_forward(self, x):
        if self.transform == "exponent":
            if self.abs_exponent != 1.0:
                e = abs(self.abs_exponent)
                x = x.abs().clamp_min(1e-12).pow(e) * torch.exp(1j * x.angle())
            return x * self.factor
        if self.transform == "none":
            return x
        raise ValueError("transform must be 'exponent' or 'none'")

    def transform_backward(self, x):
        if self.transform == "exponent":
            x = x / abs(self.factor)
            if self.abs_exponent != 1.0:
                e = abs(self.abs_exponent)
                x = x.abs().clamp_min(1e-12).pow(1.0 / e) * torch.exp(1j * x.angle())
            return x
        if self.transform == "none":
            return x
        raise ValueError("transform must be 'exponent' or 'none'")

    def _real_rfft_endpoints(self, x):
        if not self.enforce_real_rfft_bins:
            return x
        x = x.clone()
        x[..., 0] = x[..., 0].real.to(x.dtype)
        if self.n_fft is not None and self.n_fft % 2 == 0 and x.shape[-1] == self.n_fft // 2 + 1:
            x[..., -1] = x[..., -1].real.to(x.dtype)
        return x

    def forward(self, xt, y, time_cond):
        batch_size, n_bins = y.shape
        xt = xt.view(batch_size, self.num_sources, n_bins)
        y = y.view(batch_size, 1, n_bins)
        x_in = self.transform_forward(xt)
        y_in = self.transform_forward(y)
        out = self.backbone(x_in, y_in, time_cond.real)
        out = self.transform_backward(out)
        out = self._real_rfft_endpoints(out)
        return out.reshape(batch_size, self.num_sources * n_bins)
