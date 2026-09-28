import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ------------------------------------------------------------
# Time Embedding
# ------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------


def gn(ch):
    return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)


# ------------------------------------------------------------
# Residual Block (FiLM time conditioning)
# ------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch * 2)  # FiLM scale + shift

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))

        scale, shift = self.time_proj(t).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.conv2(F.silu(self.norm2(h)))

        return h + self.skip(x)


# ------------------------------------------------------------
# Attention
# ------------------------------------------------------------


class SelfAttention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (ch // heads) ** -0.5

        self.norm = gn(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, K, N = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)

        q = q.reshape(B, self.heads, C // self.heads, K * N)
        k = k.reshape(B, self.heads, C // self.heads, K * N)
        v = v.reshape(B, self.heads, C // self.heads, K * N)

        attn = torch.softmax((q.transpose(-2, -1) @ k) * self.scale, dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, K, N)

        return x + self.proj(out)


# ------------------------------------------------------------
# Down / Up Sampling  (N axis only — K is never touched)
# ------------------------------------------------------------


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1))

    def forward(self, x):
        if x.shape[-1] % 2 == 1:
            x = F.pad(x, (0, 1))
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x, target_n: int):
        x = F.interpolate(
            x, size=(x.shape[-2], target_n), mode="bilinear", align_corners=False
        )
        return self.conv(x)


# ------------------------------------------------------------
# Full U-Net
# ------------------------------------------------------------


class UNet(nn.Module):
    """
    Input
    -----
    x : [B, 2, K+1, N]
        x[:, :, :K, :]  -> K signal rows to denoise
        x[:, :, K:, :]  -> 1 conditioning row (last row)

    t : [B]

    Output
    ------
    [B, 2, K, N]

    Design notes
    ------------
    - Conditioning row kept as an extra spatial row from the first conv —
      model sees it at every depth level for free via skip connections.
    - Downsample/upsample along N axis only; K+1 rows preserved throughout.
    - Time conditioning via FiLM (scale+shift) inside every ResBlock.
    - Attention inserted at levels where attn_levels[i] is True.
    - channel_mults controls channel width at each level.
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        base_channels=64,
        channel_mults=(1, 2, 4, 4),
        time_dim=256,
        attn_levels=(False, False, True, True),
    ):
        super().__init__()

        assert len(channel_mults) == len(attn_levels), (
            "channel_mults and attn_levels must be the same length"
        )

        self.channel_mults = channel_mults
        self.attn_levels = attn_levels

        # time embedding
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            TimeMLP(time_dim),
        )

        # first conv — receives full [B, 2, K+1, N] including cond row
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ---------------- Encoder ----------------
        # One ResBlock (+ optional SelfAttention) + one Downsample per level.
        # Stored as parallel ModuleLists so the forward loop is explicit.
        self.enc_res = nn.ModuleList()
        self.enc_attn = nn.ModuleList()  # nn.Identity when attn_levels[i]=False
        self.enc_down = nn.ModuleList()

        ch = base_channels
        self.skip_chs = []

        for mult, use_attn in zip(channel_mults, attn_levels):
            out_ch = base_channels * mult
            self.enc_res.append(ResBlock(ch, out_ch, time_dim))
            self.enc_attn.append(SelfAttention(out_ch) if use_attn else nn.Identity())
            self.enc_down.append(Downsample(out_ch))
            self.skip_chs.append(out_ch)
            ch = out_ch

        # ---------------- Bottleneck ----------------
        self.mid1 = ResBlock(ch, ch, time_dim)
        self.mid_attn = SelfAttention(ch)
        self.mid2 = ResBlock(ch, ch, time_dim)

        # ---------------- Decoder ----------------
        # One Upsample + one ResBlock (+ optional SelfAttention) per level.
        self.dec_up = nn.ModuleList()
        self.dec_res = nn.ModuleList()
        self.dec_attn = nn.ModuleList()

        for mult, use_attn, skip_ch in zip(
            reversed(channel_mults),
            reversed(attn_levels),
            reversed(self.skip_chs),
        ):
            out_ch = base_channels * mult
            self.dec_up.append(Upsample(ch))
            self.dec_res.append(ResBlock(ch + skip_ch, out_ch, time_dim))
            self.dec_attn.append(SelfAttention(out_ch) if use_attn else nn.Identity())
            ch = out_ch

        # output head
        self.out = nn.Sequential(
            gn(ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
        )

    def forward(self, x, t):
        # x : [B, 2, K+1, N]  — cond row is the last spatial row
        t_emb = self.time(t)

        h = self.in_conv(x)  # [B, base_ch, K+1, N]

        # ── Encoder ──────────────────────────────────────────────────────────
        skips = []
        for res, attn, down in zip(self.enc_res, self.enc_attn, self.enc_down):
            h = res(h, t_emb)
            h = attn(h)
            skips.append(h)  # save before downsampling
            h = down(h)

        # ── Bottleneck ───────────────────────────────────────────────────────
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # ── Decoder ──────────────────────────────────────────────────────────
        for up, res, attn, skip in zip(
            self.dec_up, self.dec_res, self.dec_attn, reversed(skips)
        ):
            h = up(h, skip.shape[-1])
            h = torch.cat([h, skip], dim=1)
            h = res(h, t_emb)
            h = attn(h)

        h = self.out(h)  # [B, 2, K+1, N]
        return h[:, :, :-1, :]  # [B, 2, K,   N]  drop cond row
