"""Stronger physics-conditioned spectral U-Net.

Drop-in backbone for :class:`src.score_model.ScoreModel` with the same public
``forward(X, Y, t)`` interface as the original network.  The known frequency
responses can optionally be attached once with ``set_operator``.  When present,
the network receives the normalized adjoint measurement residual
``H^H(HX-Y)`` as a *feature*; it is not used as a hard projection.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int, preferred: int = 8) -> int:
    for g in range(min(preferred, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_groups(channels), channels, eps=1e-6)


class GaussianFourierTimeEmbedding(nn.Module):
    def __init__(self, time_dim: int = 256, scale: float = 16.0):
        super().__init__()
        if time_dim % 2:
            raise ValueError("time_dim must be even")
        half = time_dim // 2
        # Fixed Gaussian Fourier frequencies.  They are model state but are not
        # optimized.  Using an nn.Parameter with requires_grad=False makes this
        # explicit in Lightning's parameter summary (unlike a buffer, which is
        # not counted as a non-trainable parameter).
        self.freqs = nn.Parameter(
            torch.randn(half) * scale, requires_grad=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.real.reshape(-1, 1)
        a = 2.0 * math.pi * t * self.freqs.reshape(1, -1)
        return self.mlp(torch.cat((a.sin(), a.cos()), dim=-1))


class ConditionedResidualBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, dilation: int = 1, dropout: float = 0.05):
        super().__init__()
        self.norm1 = _gn(channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = _gn(channels)
        self.gate = nn.Conv1d(channels, 2 * channels, 3, padding=1)
        self.out = nn.Conv1d(channels, channels, 1)
        self.time_film = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * channels))
        self.cond_film = nn.Conv1d(channels, 2 * channels, 1)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.time_film[-1].weight); nn.init.zeros_(self.time_film[-1].bias)
        nn.init.zeros_(self.cond_film.weight); nn.init.zeros_(self.cond_film.bias)

    def forward(self, x, t_emb, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        scale_shift = self.time_film(t_emb).unsqueeze(-1) + self.cond_film(F.silu(cond))
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        value, gate = self.gate(F.silu(h)).chunk(2, dim=1)
        h = self.out(self.dropout(value * torch.sigmoid(gate)))
        return x + h


class SpectralAttention(nn.Module):
    """Self-attention across frequency plus cross-attention to conditioning."""
    def __init__(self, channels: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.norm_x1 = nn.LayerNorm(channels)
        self.norm_x2 = nn.LayerNorm(channels)
        self.norm_c = nn.LayerNorm(channels)
        self.norm_ff = nn.LayerNorm(channels)
        self.self_attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * channels, channels),
        )

    def forward(self, x, cond):
        seq = x.transpose(1, 2)
        ctx = cond.transpose(1, 2)
        q = self.norm_x1(seq)
        seq = seq + self.self_attn(q, q, q, need_weights=False)[0]
        seq = seq + self.cross_attn(self.norm_x2(seq), self.norm_c(ctx), self.norm_c(ctx), need_weights=False)[0]
        seq = seq + self.ff(self.norm_ff(seq))
        return seq.transpose(1, 2)


class PhysicsConditionedUNet(nn.Module):
    """Frequency-domain U-Net for K complex sources and one complex mixture."""
    def __init__(
        self,
        K: int,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 3, 4),
        blocks_per_level: int = 2,
        time_dim: int = 256,
        dropout: float = 0.05,
        attention_heads: int = 4,
        attention_levels: Sequence[int] = (2, 3),
        T: float = 1.0,
    ):
        super().__init__()
        self.K = int(K)
        self.T = float(T)
        self.n_fft = None
        self.register_buffer("operator", torch.empty(0, dtype=torch.complex64), persistent=False)
        widths = [base_channels * int(m) for m in channel_mults]
        if any(w % attention_heads for w in widths if widths.index(w) in attention_levels):
            raise ValueError("Attention widths must be divisible by attention_heads")
        self.widths = widths
        self.time_embed = GaussianFourierTimeEmbedding(time_dim)

        # X (real/imag), log|X|, normalized adjoint residual (real/imag) => 5K
        self.state_stem = nn.Conv1d(5 * self.K, widths[0], 3, padding=1)
        # y (real/imag/logmag) + H (real/imag/logmag) => 3 + 3K
        self.cond_stem = nn.Sequential(
            nn.Conv1d(3 + 3 * self.K, widths[0], 3, padding=1),
            _gn(widths[0]), nn.SiLU(),
        )

        self.enc_blocks = nn.ModuleList()
        self.enc_attn = nn.ModuleDict()
        self.state_down = nn.ModuleList()
        self.cond_down = nn.ModuleList()
        for i, w in enumerate(widths):
            self.enc_blocks.append(nn.ModuleList([
                ConditionedResidualBlock(w, time_dim, dilation=2 ** (j % 3), dropout=dropout)
                for j in range(blocks_per_level)
            ]))
            if i in attention_levels:
                self.enc_attn[str(i)] = SpectralAttention(w, attention_heads, dropout=0.0)
            if i < len(widths) - 1:
                nw = widths[i + 1]
                self.state_down.append(nn.Conv1d(w, nw, 4, stride=2, padding=1))
                self.cond_down.append(nn.Sequential(
                    nn.Conv1d(w, nw, 4, stride=2, padding=1), _gn(nw), nn.SiLU()
                ))

        self.mid1 = ConditionedResidualBlock(widths[-1], time_dim, 1, dropout)
        self.mid_attn = SpectralAttention(widths[-1], attention_heads, dropout=0.0)
        self.mid2 = ConditionedResidualBlock(widths[-1], time_dim, 2, dropout)

        self.up_proj = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_attn = nn.ModuleDict()
        current = widths[-1]
        for i in reversed(range(len(widths) - 1)):
            w = widths[i]
            self.up_proj.append(nn.Conv1d(current, w, 3, padding=1))
            self.fuse.append(nn.Conv1d(2 * w, w, 1))
            self.dec_blocks.append(nn.ModuleList([
                ConditionedResidualBlock(w, time_dim, dilation=2 ** (j % 3), dropout=dropout)
                for j in range(blocks_per_level)
            ]))
            if i in attention_levels:
                self.dec_attn[str(i)] = SpectralAttention(w, attention_heads, dropout=0.0)
            current = w

        self.out_norm = _gn(widths[0])
        self.out = nn.Conv1d(widths[0], 2 * self.K, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    @torch.no_grad()
    def set_operator(self, kernels: torch.Tensor, n_fft: int | None = None):
        """Attach fixed transfer functions H[k,f] once for physics features."""
        if kernels.ndim != 2 or kernels.shape[0] != self.K:
            raise ValueError(f"Expected kernels [K,F] with K={self.K}, got {tuple(kernels.shape)}")
        self.operator = kernels.detach().to(dtype=torch.complex64, device=self.operator.device)
        self.n_fft = int(n_fft) if n_fft is not None else None

    def _operator_for(self, x):
        if self.operator.numel() == 0:
            return torch.zeros((self.K, x.shape[-1]), dtype=x.dtype, device=x.device)
        if self.operator.shape[-1] != x.shape[-1]:
            raise ValueError(
                f"Operator has {self.operator.shape[-1]} bins but state has {x.shape[-1]}"
            )
        return self.operator.to(device=x.device, dtype=x.dtype)

    def _physics_features(self, X, Y):
        H = self._operator_for(X)
        if self.operator.numel() == 0:
            adjoint = torch.zeros_like(X)
        else:
            residual = (X * H.unsqueeze(0)).sum(dim=1, keepdim=True) - Y
            normalizer = 1.0 + H.abs().square().sum(dim=0, keepdim=True)
            adjoint = H.conj().unsqueeze(0) * residual / normalizer.unsqueeze(0)
        return H, adjoint

    def forward(self, X, Y, t):
        if X.ndim != 3 or X.shape[1] != self.K:
            raise ValueError(f"X must be [B,{self.K},F]")
        if Y.ndim != 3 or Y.shape[1] != 1:
            raise ValueError("Y must be [B,1,F]")
        H, adjoint = self._physics_features(X, Y)
        B, _, Freq = X.shape
        H_b = H.unsqueeze(0).expand(B, -1, -1)

        state_in = torch.cat(
            [X.real, X.imag, torch.log1p(X.abs()), adjoint.real, adjoint.imag], dim=1
        )
        cond_in = torch.cat(
            [Y.real, Y.imag, torch.log1p(Y.abs()), H_b.real, H_b.imag, torch.log1p(H_b.abs())], dim=1
        )
        h = self.state_stem(state_in)
        c = self.cond_stem(cond_in)
        te = self.time_embed(t.real / self.T)

        skips, conds = [], []
        for i, blocks in enumerate(self.enc_blocks):
            for block in blocks:
                h = block(h, te, c)
            if str(i) in self.enc_attn:
                h = self.enc_attn[str(i)](h, c)
            skips.append(h); conds.append(c)
            if i < len(self.widths) - 1:
                h = self.state_down[i](h)
                c = self.cond_down[i](c)

        h = self.mid1(h, te, c)
        h = self.mid_attn(h, c)
        h = self.mid2(h, te, c)

        for j, i in enumerate(reversed(range(len(self.widths) - 1))):
            skip, c = skips[i], conds[i]
            # nearest + learned convolution is deterministic on CUDA and avoids
            # the non-deterministic upsample_linear1d backward kernel.
            h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = self.up_proj[j](h)
            h = self.fuse[j](torch.cat([h, skip], dim=1))
            for block in self.dec_blocks[j]:
                h = block(h, te, c)
            if str(i) in self.dec_attn:
                h = self.dec_attn[str(i)](h, c)

        out = self.out(F.silu(self.out_norm(h)))
        z = torch.complex(out[:, : self.K], out[:, self.K :])
        # rFFT spectra of real signals have real-valued DC and (for even n_fft) Nyquist bins.
        z = z.clone()
        z[..., 0] = z[..., 0].real.to(z.dtype)
        if self.n_fft is not None and self.n_fft % 2 == 0 and z.shape[-1] == self.n_fft // 2 + 1:
            z[..., -1] = z[..., -1].real.to(z.dtype)
        return z
