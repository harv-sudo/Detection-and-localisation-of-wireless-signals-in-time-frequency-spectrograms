"""Small dense U-Net with RepVGG-style branches that disappear at deployment.

Training uses 3x3, 1x1 and (when possible) identity branches, each with BN.
In eval mode their affine maps merge exactly, up to floating-point rounding,
into one 3x3 convolution. No quantization, pruning or data changes are involved.
This is an application of established structural reparameterization, not a
claim that the technique or its combination with U-Net is novel.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import fuse_conv_bn_eval


class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False, branches=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.deploy = deploy
        if deploy:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.spatial = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.pointwise = (
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
                if branches
                else None
            )
            self.identity = (
                nn.BatchNorm2d(in_channels)
                if branches and in_channels == out_channels
                else None
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.activation(self.conv(x))
        value = self.spatial(x)
        if self.pointwise is not None:
            value = value + self.pointwise(x)
        if self.identity is not None:
            value = value + self.identity(x)
        return self.activation(value)

    @torch.no_grad()
    def merge(self):
        if self.training:
            raise ValueError("Call eval() before merging training branches.")
        if self.deploy:
            return
        conv = fuse_conv_bn_eval(self.spatial[0], self.spatial[1])
        if self.pointwise is not None:
            point = fuse_conv_bn_eval(self.pointwise[0], self.pointwise[1])
            conv.weight.add_(F.pad(point.weight, (1, 1, 1, 1)))
            conv.bias.add_(point.bias)
        if self.identity is not None:
            norm = self.identity
            if norm.training:
                raise ValueError("All normalization layers must be in eval mode.")
            scale = norm.weight / torch.sqrt(norm.running_var + norm.eps)
            indices = torch.arange(self.in_channels, device=conv.weight.device)
            conv.weight[indices, indices, 1, 1] += scale
            conv.bias.add_(norm.bias - norm.running_mean * scale)
        self.conv = conv
        del self.spatial, self.pointwise, self.identity
        self.deploy = True


def _double(in_channels, out_channels, deploy, branches):
    return nn.Sequential(
        RepConv(in_channels, out_channels, deploy, branches),
        RepConv(out_channels, out_channels, deploy, branches),
    )


class _Up(nn.Module):
    def __init__(self, in_channels, out_channels, deploy, branches):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = _double(2 * out_channels, out_channels, deploy, branches)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.conv(torch.cat((skip, x), dim=1))


class FastUNet(nn.Module):
    """Same input/output contract as UNet; widths b, 2b, 4b and 8b.

    Raw binary images in floating Bx1xHxW form, H,W >= 8; returns raw logits.
    The default b=8 is a latency candidate, whose accuracy must be measured.
    ``model.eval().to_deploy()`` returns a separate model with merged branches.
    """

    def __init__(self, base_channels=8, deploy=False, branches=True):
        super().__init__()
        if (
            isinstance(base_channels, bool)
            or not isinstance(base_channels, int)
            or base_channels < 1
        ):
            raise ValueError("base_channels must be a positive integer")
        self.base_channels = base_channels
        self.branches = branches
        b = base_channels
        self.encoder0 = _double(1, b, deploy, branches)
        self.encoder1 = nn.Sequential(
            nn.MaxPool2d(2), _double(b, 2 * b, deploy, branches)
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2), _double(2 * b, 4 * b, deploy, branches)
        )
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2), _double(4 * b, 8 * b, deploy, branches)
        )
        self.decoder2 = _Up(8 * b, 4 * b, deploy, branches)
        self.decoder1 = _Up(4 * b, 2 * b, deploy, branches)
        self.decoder0 = _Up(2 * b, b, deploy, branches)
        self.output = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[0] < 1 or min(x.shape[-2:]) < 8:
            raise ValueError("Expected nonempty Bx1xHxW input with H,W >= 8")
        if not x.is_floating_point():
            raise TypeError("Input must have floating dtype")
        first = self.encoder0(x)
        second = self.encoder1(first)
        third = self.encoder2(second)
        value = self.decoder2(self.bottleneck(third), third)
        return self.output(self.decoder0(self.decoder1(value, second), first))

    def to_deploy(self):
        if self.training:
            raise ValueError("Call eval() before deployment conversion.")
        result = copy.deepcopy(self)
        for module in list(result.modules()):
            if isinstance(module, RepConv):
                module.merge()
        return result


def fuse_unet_batchnorm(model):
    """Copy and fold adjacent Conv2d/BN pairs in the original U-Net too.

    Giving the reference the same deployment optimization avoids attributing
    the entire gain from ordinary BN folding to the new architecture.
    """
    if model.training:
        raise ValueError("Call eval() before deployment conversion.")
    result = copy.deepcopy(model)
    for sequence in result.modules():
        if isinstance(sequence, nn.Sequential):
            for index in range(len(sequence) - 1):
                if isinstance(sequence[index], nn.Conv2d) and isinstance(
                    sequence[index + 1], nn.BatchNorm2d
                ):
                    sequence[index] = fuse_conv_bn_eval(
                        sequence[index], sequence[index + 1]
                    )
                    sequence[index + 1] = nn.Identity()
    return result
