"""Capacity-first residual U-Net++ for binary spectrogram segmentation.

The nested decoder and deep supervision follow established U-Net++ ideas.
A spatial attention block lets low-resolution features combine information
across the image. Neither labels nor SNR are inputs to the network. This is an
accuracy-oriented combination of existing components, not a novelty claim.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _normalization(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _normalization(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _normalization(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.features(x) + self.shortcut(x))


class _SpatialContext(nn.Module):
    """Pre-normalized global attention at the smallest spatial resolution.

Depthwise convolution supplies local positional context without a learned
fixed-size coordinate table. PyTorch's scaled-dot-product attention can select
CUDA attention kernels when the device and precision support them.
"""

    def __init__(self, channels):
        super().__init__()
        self.heads = math.gcd(8, channels)
        self.head_dim = channels // self.heads
        self.position = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.attention_norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.projection = nn.Linear(channels, channels)
        self.feedforward_norm = nn.LayerNorm(channels)
        self.feedforward = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        tokens = (x + self.position(x)).flatten(2).transpose(1, 2)
        qkv = self.qkv(self.attention_norm(tokens))
        qkv = qkv.reshape(batch, height * width, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, height * width, channels)
        tokens = tokens + self.projection(attended)
        tokens = tokens + self.feedforward(self.feedforward_norm(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class _NestedFusion(nn.Module):
    def __init__(self, channels, lateral_count):
        super().__init__()
        self.up_projection = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.fusion = _ResidualBlock(channels * (lateral_count + 1), channels)

    def forward(self, lower, *lateral):
        up = F.interpolate(
            self.up_projection(lower),
            size=lateral[0].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fusion(torch.cat((*lateral, up), dim=1))


class F1UNet(nn.Module):
    """Residual nested U-Net with bottleneck attention and deep supervision.

    ``forward(x)`` returns raw main-head logits with the same B1HW shape as x.
    ``forward(x, return_aux=True)`` returns ``(main_logits, (early, middle))``;
    all three logits have full input resolution. Auxiliary heads are training
    aids; the final decoder head alone defines inference predictions.

    Inputs must be floating point, nonempty B1HW images with H and W >= 8.
    The default width is 64, giving encoder widths 64, 128, 256, and 512.
    """

    def __init__(self, base_channels=64):
        super().__init__()
        if isinstance(base_channels, bool) or not isinstance(base_channels, int):
            raise TypeError("base_channels must be an integer")
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")
        self.base_channels = base_channels
        widths = [base_channels * 2**level for level in range(4)]
        self.pool = nn.MaxPool2d(2)
        self.encoder = nn.ModuleList(
            [_ResidualBlock(1, widths[0])]
            + [_ResidualBlock(widths[i - 1], widths[i]) for i in range(1, 4)]
        )
        self.context = _SpatialContext(widths[3])
        self.x01 = _NestedFusion(widths[0], 1)
        self.x11 = _NestedFusion(widths[1], 1)
        self.x21 = _NestedFusion(widths[2], 1)
        self.x02 = _NestedFusion(widths[0], 2)
        self.x12 = _NestedFusion(widths[1], 2)
        self.x03 = _NestedFusion(widths[0], 3)
        self.auxiliary_heads = nn.ModuleList(
            [nn.Conv2d(widths[0], 1, 1), nn.Conv2d(widths[0], 1, 1)]
        )
        self.output = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x, return_aux=False):
        if x.ndim != 4 or x.shape[0] == 0 or x.shape[1] != 1:
            raise ValueError("Expected a nonempty batch of B1HW images")
        if min(x.shape[-2:]) < 8:
            raise ValueError("Image height and width must both be at least 8")
        if not x.is_floating_point():
            raise TypeError("Input images must have a floating-point dtype")

        x00 = self.encoder[0](x)
        x10 = self.encoder[1](self.pool(x00))
        x20 = self.encoder[2](self.pool(x10))
        x30 = self.context(self.encoder[3](self.pool(x20)))
        x01 = self.x01(x10, x00)
        x11 = self.x11(x20, x10)
        x21 = self.x21(x30, x20)
        x02 = self.x02(x11, x00, x01)
        x12 = self.x12(x21, x10, x11)
        x03 = self.x03(x12, x00, x01, x02)
        main_logits = self.output(x03)
        if return_aux:
            return main_logits, (
                self.auxiliary_heads[0](x01),
                self.auxiliary_heads[1](x02),
            )
        return main_logits
