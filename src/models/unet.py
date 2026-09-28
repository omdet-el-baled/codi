import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale)

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_proj = t * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, t_features, hidden_dim, T):
        super().__init__()
        self.fourier = GaussianFourierProjection(t_features, scale=1 / T)
        self.mlp = nn.Sequential(
            nn.Linear(2 * t_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        return self.mlp(self.fourier(t))


class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DownsampleN(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1))

    def forward(self, x):
        if x.shape[-1] % 2 == 1:
            x = F.pad(x, (0, 1))
        return self.conv(x)


class UpsampleN(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x, target_n: int):
        x = F.interpolate(
            x, size=(x.shape[-2], target_n), mode="bilinear", align_corners=False
        )
        return self.conv(x)


class SelfAttention2D(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (ch // heads) ** -0.5
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, K, N = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)

        def reshape(t):
            return t.reshape(B, self.heads, C // self.heads, K * N)

        q, k, v = reshape(q), reshape(k), reshape(v)
        attn = torch.softmax((q.transpose(-2, -1) @ k) * self.scale, dim=-1)
        out = v @ attn.transpose(-2, -1)
        out = out.reshape(B, C, K, N)
        return x + self.proj(out)


class UNet(nn.Module):
    """
    Input
    -----
    x : [B, 2, K+1, N]
        x[:, :, :K, :]   -> K signal rows to denoise
        x[:, :, K:, :]   -> 1 conditioning row (the last row)

    t : [B]

    Output
    ------
    [B, 2, K, N]  -- denoised prediction for the K signal rows
    N can be any positive integer.
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        base_channels=64,
        depth=3,
        time_dim=256,
        T=1,
        use_attention=True,
    ):
        super().__init__()

        self.time = TimeEmbedding(time_dim, time_dim, T)

        # Encode signal rows: [B, 2, K, N] -> [B, base_channels, K, N]
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encode conditioning row: [B, 2, 1, N] -> [B, base_channels, 1, N]
        # then broadcast across K via interpolation in forward
        self.cond_proj = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        )

        # ---------------- Encoder ----------------
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_channels
        skip_channels = []

        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.down_blocks.append(ResBlock2D(ch, out_ch, time_dim))
            self.downsamples.append(DownsampleN(out_ch))
            skip_channels.append(out_ch)
            ch = out_ch

        # ---------------- Bottleneck ----------------
        self.mid1 = ResBlock2D(ch, ch, time_dim)
        self.attn = SelfAttention2D(ch) if use_attention else nn.Identity()
        self.mid2 = ResBlock2D(ch, ch, time_dim)

        # ---------------- Decoder ----------------
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for i in reversed(range(depth)):
            skip_ch = skip_channels[i]
            out_ch = base_channels * (2**i)
            self.upsamples.append(UpsampleN(ch))
            self.up_blocks.append(ResBlock2D(ch + skip_ch, out_ch, time_dim))
            ch = out_ch

        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, t):
        # x: [B, 2, K+1, N]
        signal = x[:, :, :-1, :]  # [B, 2, K,   N]  rows to denoise
        cond = x[:, :, -1:, :]  # [B, 2, 1,   N]  conditioning row

        t_emb = self.time(t)  # [B, time_dim]

        # Encode signal
        h = self.in_conv(signal)  # [B, base_ch, K, N]

        # Encode conditioning and broadcast to match K
        c = self.cond_proj(cond)  # [B, base_ch, 1, N]
        c = c.expand(-1, -1, h.shape[2], -1)  # [B, base_ch, K, N]  no new memory
        h = h + c

        # Encoder
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid1(h, t_emb)
        h = self.attn(h)
        h = self.mid2(h, t_emb)

        # Decoder
        for up, block, skip in zip(self.upsamples, self.up_blocks, reversed(skips)):
            h = up(h, skip.shape[-1])
            if h.shape[-2] != skip.shape[-2]:
                h = h[:, :, : skip.shape[-2], :]
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)

        return self.out_conv(h)  # [B, 2, K, N]
