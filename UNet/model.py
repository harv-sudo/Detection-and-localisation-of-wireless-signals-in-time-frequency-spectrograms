import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions.

    Each convolution is followed by:
        Batch Normalisation
        ReLU activation

    This is the basic building block used throughout the U-Net.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """
    Encoder block.

    Max pooling reduces the spatial resolution by a factor of 2,
    followed by a DoubleConv block.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    Decoder block.

    The feature map is first upsampled, then concatenated with the
    corresponding encoder feature map (skip connection).

    A DoubleConv block then processes the combined features.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = self.up(x)

        # A 50x100 input does not divide perfectly through every
        # downsampling stage. Make sure the decoder feature map
        # exactly matches the corresponding encoder feature map.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Skip connection
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """
    Compact U-Net for binary signal-area segmentation.

    Input:
        (batch_size, 1, 50, 100)

    Output:
        (batch_size, 1, 50, 100)

    The output contains raw logits.

    Do NOT apply sigmoid here during training because
    BCEWithLogitsLoss applies sigmoid internally.
    """

    def __init__(self):
        super().__init__()

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

        self.enc1 = DoubleConv(1, 32)
        self.enc2 = Down(32, 64)
        self.enc3 = Down(64, 128)

        # -------------------------------------------------
        # Bottleneck
        # -------------------------------------------------

        self.bottleneck = Down(128, 256)

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------

        self.dec3 = Up(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
        )

        self.dec2 = Up(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
        )

        self.dec1 = Up(
            in_channels=64,
            skip_channels=32,
            out_channels=32,
        )

        # -------------------------------------------------
        # Final segmentation layer
        # -------------------------------------------------

        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=1,
        )

    def forward(self, x):

        # Encoder
        x1 = self.enc1(x)  # 32 channels
        x2 = self.enc2(x1)  # 64 channels
        x3 = self.enc3(x2)  # 128 channels

        # Bottleneck
        x4 = self.bottleneck(x3)  # 256 channels

        # Decoder + skip connections
        x = self.dec3(x4, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)

        # Raw segmentation logits
        return self.output(x)


# ---------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------

if __name__ == "__main__":
    model = UNet()

    # Simulate a batch of 4 spectrograms
    x = torch.randn(4, 1, 50, 100)

    y = model(x)

    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters: {parameters:,}")
