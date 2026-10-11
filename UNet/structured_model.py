"""Compact structured segmentation candidate for the existing binary images.

EvidenceAxialNet accepts the same B x 1 x H x W floating-point binary inputs and
returns raw segmentation logits, so the existing BCE-with-logits plus Dice loss
can be used unchanged. Height is the time axis and width is the frequency axis.

The network augments each observation with six local directional means. These
are normalized binary counts, with padding excluded from their denominators.
No SNR, labels, coordinates, fixed active-band position, or metadata are inputs.
A compact encoder/decoder uses residual depthwise axial convolutions and
GroupNorm. Decoder interpolation targets the actual skip sizes, including odd
input dimensions.

An optional finite refinement stage couples immediate neighbours through learned
undirected edge weights. Every iteration starts from the fixed unary logits;
it adds a bounded average of neighbouring signed probabilities, normalized by
one plus the incident edge-weight sum. The unit no-change anchor lets small
affinities suppress the correction rather than cancel through normalization.
The scalar coupling starts near zero but remains differentiable. This is a
mean-field-style refinement, not a claim of exact probabilistic inference.

This module is a research candidate, not an originality or accuracy claim.
Local means can blur boundaries, learned smoothing can merge nearby signals,
and binary observations cannot recover information discarded before the model.
GroupNorm uses spatial statistics; the network is not a causal streaming model.
Returned diagnostics show internal quantities, not validated faithful explanations.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _group_norm(channels):
    """Allow arbitrary supported channel widths without divisibility surprises."""
    return nn.GroupNorm(math.gcd(8, channels), channels)


class _AxialResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = _group_norm(channels)
        self.frequency_conv = nn.Conv2d(
            channels,
            channels,
            (1, 5),
            padding=(0, 2),
            groups=channels,
            bias=False,
        )
        self.time_conv = nn.Conv2d(
            channels,
            channels,
            (5, 1),
            padding=(2, 0),
            groups=channels,
            bias=False,
        )
        self.mix = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        residual = F.silu(self.norm(x))
        residual = F.silu(self.frequency_conv(residual))
        residual = self.time_conv(residual)
        return x + self.mix(residual)


class _ConvStage(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            _group_norm(out_channels),
            nn.SiLU(),
        )
        self.residual = _AxialResidualBlock(out_channels)

    def forward(self, x):
        return self.residual(self.project(x))


class _DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(),
        )
        self.residual = _AxialResidualBlock(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.residual(self.fuse(torch.cat((x, skip), dim=1)))


class EvidenceAxialNet(nn.Module):
    """Binary-image segmentation with optional directional evidence/refinement.

    Args:
        base_channels: Integer >= 4. Encoder widths are b, 2b and 4b.
        refinement_steps: Nonnegative number of finite refinement iterations.
        use_evidence: Append directional local means to the observed image.
        use_refinement: Enable learned neighbour coupling after the unary head.

    ``forward(x)`` returns raw logits with the same B x 1 x H x W shape as x.
    Inputs must have a nonempty batch, floating dtype and H,W >= 8. Binary
    values are expected; no costly value/range check is performed in forward.

    ``forward(x, return_aux=True)`` returns ``(logits, diagnostics)``. Evidence
    channels, when enabled, are frequency means at spans 3,9,17,33, followed by
    time means at spans 3,7. When disabled, evidence_maps has zero channels.
    Edge tensors hold one weight per undirected pair: horizontal Bx1xHx(W-1),
    vertical Bx1x(H-1)xW. Inactive refinement returns zero edge weights/strength.
    Diagnostic tensors retain gradients; detach them when storing visualizations.
    """

    FREQUENCY_SPANS = (3, 9, 17, 33)
    TIME_SPANS = (3, 7)
    MAX_REFINEMENT_STRENGTH = 2.0

    def __init__(
        self,
        base_channels=16,
        refinement_steps=3,
        use_evidence=True,
        use_refinement=True,
    ):
        super().__init__()
        if (
            isinstance(base_channels, bool)
            or not isinstance(base_channels, int)
            or base_channels < 4
        ):
            raise ValueError("base_channels must be an integer >= 4")
        if (
            isinstance(refinement_steps, bool)
            or not isinstance(refinement_steps, int)
            or refinement_steps < 0
        ):
            raise ValueError("refinement_steps must be a nonnegative integer")
        if not isinstance(use_evidence, bool) or not isinstance(use_refinement, bool):
            raise TypeError("use_evidence and use_refinement must be booleans")

        self.base_channels = base_channels
        self.refinement_steps = refinement_steps
        self.use_evidence = use_evidence
        self.use_refinement = use_refinement
        evidence_channels = (
            len(self.FREQUENCY_SPANS) + len(self.TIME_SPANS) if use_evidence else 0
        )
        self.encoder0 = _ConvStage(1 + evidence_channels, base_channels)
        self.encoder1 = _ConvStage(base_channels, 2 * base_channels, stride=2)
        self.encoder2 = _ConvStage(2 * base_channels, 4 * base_channels, stride=2)
        self.decoder1 = _DecoderStage(
            4 * base_channels, 2 * base_channels, 2 * base_channels
        )
        self.decoder0 = _DecoderStage(2 * base_channels, base_channels, base_channels)
        self.unary_head = nn.Conv2d(base_channels, 1, 1)

        if use_refinement:
            self.edge_gate = nn.Sequential(
                nn.Conv2d(2 * base_channels, base_channels, 1),
                nn.SiLU(),
                nn.Conv2d(base_channels, 1, 1),
            )
            # Effective strength is approximately 0.036 at initialization.
            self.raw_refinement_strength = nn.Parameter(torch.tensor(-4.0))
        else:
            self.edge_gate = None
            self.register_parameter("raw_refinement_strength", None)

    def _evidence(self, x):
        if not self.use_evidence:
            return x.new_empty((x.shape[0], 0, x.shape[2], x.shape[3]))
        frequency = [
            F.avg_pool2d(
                x,
                kernel_size=(1, span),
                stride=1,
                padding=(0, span // 2),
                count_include_pad=False,
            )
            for span in self.FREQUENCY_SPANS
        ]
        time = [
            F.avg_pool2d(
                x,
                kernel_size=(span, 1),
                stride=1,
                padding=(span // 2, 0),
                count_include_pad=False,
            )
            for span in self.TIME_SPANS
        ]
        return torch.cat(frequency + time, dim=1)

    def _pair_weights(self, first, second):
        # Mean and absolute difference are invariant to exchanging endpoints.
        pair_features = torch.cat(
            ((first + second) * 0.5, (first - second).abs()), dim=1
        )
        return torch.sigmoid(self.edge_gate(pair_features))

    @staticmethod
    def _neighbour_sum(values, horizontal, vertical):
        # Each stored undirected edge contributes to both incident pixels.
        return (
            F.pad(horizontal * values[..., 1:], (0, 1, 0, 0))
            + F.pad(horizontal * values[..., :-1], (1, 0, 0, 0))
            + F.pad(vertical * values[..., 1:, :], (0, 0, 0, 1))
            + F.pad(vertical * values[..., :-1, :], (0, 0, 1, 0))
        )

    def forward(self, x, return_aux=False):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("x must have shape (batch, 1, height, width)")
        if x.shape[0] < 1 or min(x.shape[-2:]) < 8:
            raise ValueError("x requires a nonempty batch and height,width >= 8")
        if not x.is_floating_point():
            raise TypeError("x must have floating dtype, with binary observations")
        if not isinstance(return_aux, bool):
            raise TypeError("return_aux must be a boolean")

        evidence = self._evidence(x)
        model_input = torch.cat((x, evidence), dim=1) if self.use_evidence else x
        skip0 = self.encoder0(model_input)
        skip1 = self.encoder1(skip0)
        encoded = self.encoder2(skip1)
        features = self.decoder0(self.decoder1(encoded, skip1), skip0)
        unary = self.unary_head(features)
        logits = unary

        if self.use_refinement and self.refinement_steps > 0:
            horizontal = self._pair_weights(features[..., :-1], features[..., 1:])
            vertical = self._pair_weights(features[..., :-1, :], features[..., 1:, :])
            strength = self.MAX_REFINEMENT_STRENGTH * torch.sigmoid(
                self.raw_refinement_strength
            )
            # A unit no-change anchor preserves absolute affinity strength:
            # as all incident edge weights vanish, so does the correction.
            # At boundaries only the neighbours that exist contribute.
            normalizer = 1.0 + self._neighbour_sum(
                torch.ones_like(unary), horizontal, vertical
            )
            for _ in range(self.refinement_steps):
                signed_probs = 2.0 * torch.sigmoid(logits) - 1.0
                message = (
                    self._neighbour_sum(signed_probs, horizontal, vertical) / normalizer
                )
                # Reset to unary each iteration; do not accumulate logits.
                logits = unary + strength * message
        elif return_aux:
            horizontal = unary.new_zeros((*unary.shape[:-1], unary.shape[-1] - 1))
            vertical = unary.new_zeros(
                (unary.shape[0], 1, unary.shape[-2] - 1, unary.shape[-1])
            )
            strength = unary.new_zeros(())

        if return_aux:
            return logits, {
                "unary_logits": unary,
                "evidence_maps": evidence,
                "horizontal_edge_weights": horizontal,
                "vertical_edge_weights": vertical,
                "refined_logits": logits,
                "refinement_strength": strength,
            }
        return logits
