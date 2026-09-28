import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ============================================================
# Fourier Time Embedding (unchanged)
# ============================================================


class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        # Fixed Gaussian Fourier frequencies. Keep them as a frozen Parameter
        # so Lightning reports them explicitly as non-trainable model parameters.
        self.W = nn.Parameter(
            torch.randn(embedding_size) * scale, requires_grad=False
        )

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_proj = t * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


# ============================================================
# BigGAN++ style 1D Residual Block
# ============================================================


class ResnetBlock1DBigGANpp(nn.Module):
    def __init__(
        self,
        channels,
        t_emb_dim=None,
        out_channels=None,
        up=False,
        down=False,
        dropout=0.1,
        skip_rescale=True,
        init_scale=0.0,
        groups=8,
    ):
        super().__init__()
        out_channels = out_channels or channels
        self.up = up
        self.down = down
        self.skip_rescale = skip_rescale

        self.norm1 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-6)

        self.conv1 = nn.Conv1d(channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        if channels != out_channels or up or down:
            self.skip_conv = nn.Conv1d(channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = None

        # FiLM projection from time embedding — no activation since
        # Fourier features are already non-linear (sin/cos)
        if t_emb_dim is not None:
            self.t_proj = nn.Linear(t_emb_dim, out_channels * 2)
            nn.init.zeros_(self.t_proj.bias)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        if init_scale == 0.0:
            nn.init.zeros_(self.conv2.weight)
        else:
            nn.init.normal_(self.conv2.weight, std=init_scale)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, t_emb=None):
        h = self.act(self.norm1(x))

        if self.up:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        elif self.down:
            h = F.avg_pool1d(h, kernel_size=2, stride=2)
            x = F.avg_pool1d(x, kernel_size=2, stride=2)

        h = self.conv1(h)

        # FiLM conditioning — direct projection, no activation needed
        if t_emb is not None:
            gamma_beta = self.t_proj(t_emb)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            gamma = gamma[:, :, None]
            beta = beta[:, :, None]
            h = h * (1 + gamma) + beta

        h = self.conv2(self.dropout(self.act(self.norm2(h))))

        if self.skip_conv is not None:
            x = self.skip_conv(x)

        if self.skip_rescale:
            return (x + h) / math.sqrt(2.0)
        else:
            return x + h


# ============================================================
# DownBlock, UpBlock, Bottleneck
# ============================================================


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim):
        super().__init__()
        self.res = ResnetBlock1DBigGANpp(in_ch, t_emb_dim)
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, t_emb_dim):
        super().__init__()
        self.res = ResnetBlock1DBigGANpp(in_ch + skip_ch, t_emb_dim)
        self.conv = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1)

    def match_length(self, x, target):
        diff = target.shape[-1] - x.shape[-1]
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[..., : target.shape[-1]]
        return x

    def forward(self, x, skip, t_emb):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.match_length(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        x = self.conv(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, channels, t_emb_dim):
        super().__init__()
        self.res1 = ResnetBlock1DBigGANpp(channels, t_emb_dim)
        self.attn = AttentionBlock(channels)
        self.res2 = ResnetBlock1DBigGANpp(channels, t_emb_dim)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.attn(x)
        x = self.res2(x, t_emb)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x):
        x_t = x.transpose(1, 2)  # [B, L, C]

        h = self.norm1(x_t)
        attn_out, _ = self.attn(h, h, h)
        x_t = x_t + attn_out

        h = self.norm2(x_t)
        x_t = x_t + self.ff(h)

        return x_t.transpose(1, 2)


# ============================================================
# FULL UNET
# ============================================================


class UNet(nn.Module):
    """
    Predicts a K-channel complex output given:
      - X: [B, K, L] complex   -- K-channel conditioning spectrum
      - Y: [B, 1, L] complex   -- single noisy/target channel
      - t: [B]                 -- diffusion timestep

    Time embedding is a fixed (grad-free) Gaussian Fourier projection.
    t_emb_dim = 2 * t_features  (sin + cos concatenation, no MLP).
    """

    def __init__(
        self,
        K,
        base_channels=64,
        channel_mults=(1, 2, 4, 8, 4),
        t_features=256,
        T=1,
        y_channels=1,
    ):
        super().__init__()

        self.K = K
        self.y_channels = y_channels

        # t_emb_dim is 2*t_features because GaussianFourierProjection
        # concatenates sin and cos, no MLP on top
        t_emb_dim = 2 * t_features

        self.time_embed = GaussianFourierProjection(t_features, scale=1 / T)

        in_channels = 2 * (K + y_channels)  # *2 for real+imag
        self.in_proj = nn.Conv1d(in_channels, base_channels, kernel_size=1)

        # ---------------- Down Path ----------------
        channels = [base_channels * m for m in channel_mults]

        self.down_blocks = nn.ModuleList()
        skip_channels = []

        in_ch = base_channels
        for ch in channels:
            self.down_blocks.append(DownBlock(in_ch, ch, t_emb_dim))
            skip_channels.append(in_ch)
            in_ch = ch

        # ---------------- Bottleneck ----------------
        self.bottleneck = Bottleneck(in_ch, t_emb_dim)

        # ---------------- Up Path ----------------
        self.up_blocks = nn.ModuleList()

        for ch, skip_ch in zip(reversed(channels), reversed(skip_channels)):
            self.up_blocks.append(UpBlock(in_ch, skip_ch, ch, t_emb_dim))
            in_ch = ch

        # ---------------- Output ----------------
        self.out_proj = nn.Conv1d(base_channels, 2 * K, kernel_size=1)

    def forward(self, X, Y, t):
        """
        X: complex tensor [B, K, L]
        Y: complex tensor [B, y_channels, L]
        t: [B]
        """
        assert X.shape[1] == self.K, f"X has {X.shape[1]} channels, expected K={self.K}"
        assert Y.shape[1] == self.y_channels, (
            f"Y has {Y.shape[1]} channels, expected y_channels={self.y_channels}"
        )
        assert torch.is_complex(X) and torch.is_complex(Y), (
            "X and Y must be complex tensors"
        )

        XY = torch.cat([X, Y], dim=1)
        XY = torch.cat([XY.real, XY.imag], dim=1)

        x = self.in_proj(XY)
        t_emb = self.time_embed(t)  # [B, 2*t_features], no grad on W

        skips = []
        for down in self.down_blocks:
            x, skip = down(x, t_emb)
            skips.append(skip)

        x = self.bottleneck(x, t_emb)

        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip, t_emb)

        out = self.out_proj(x)

        r = out[:, : self.K, :]
        i = out[:, self.K :, :]
        return torch.complex(r, i)
