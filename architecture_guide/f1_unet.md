# F1UNet: later accuracy-oriented candidate

**Status:** this candidate is not one of the four completed standardised runs. Its name describes its intended accuracy focus, not a demonstrated F1 improvement. No matched performance or speed figures are claimed here.

F1UNet combines residual convolutional blocks, a U-Net++-style nested decoder, bottleneck attention and auxiliary training heads. These are established components; this implementation does not establish a novel research contribution by itself.

**Working:**

1. A four-level encoder extracts features using two 3 x 3 convolutions per residual block, GroupNorm and SiLU. Default channel widths are 64, 128, 256 and 512. For a 50 x 100 image, the spatial sizes are 50 x 100, 25 x 50, 12 x 25 and 6 x 12.
2. At the smallest resolution, a depthwise convolution supplies local positional context. Multi-head attention then lets the 72 spatial locations exchange information across the whole reduced image, followed by a feed-forward block.
3. The nested decoder progressively combines upsampled deeper features with multiple earlier features at the same resolution. Compared with a single skip connection per level, this provides more paths for combining fine detail with broader context.
4. The final head produces one raw logit per input pixel. Two earlier full-resolution heads are available for auxiliary losses during training (`return_aux=True`). Standard inference uses only the final head; it does not ensemble the three predictions.

```text
Binary grid -> residual encoder -> bottleneck attention
            -> nested multilevel feature fusion -> pixel logits
                        |             |
                  auxiliary head  auxiliary head
                    (training aids only)
```

**Trade-offs:** the wider layers and extra decoder paths prioritise capacity rather than minimum computation. Bottleneck attention may help combine distant context, but its benefit needs controlled testing. Attention is an internal computation, not automatically a faithful explanation. The network receives neither labels nor true SNR as inputs.

Sources: [UNet/f1_model.py](../UNet/f1_model.py), [UNet/f1_experiment.py](../UNet/f1_experiment.py). Consult that runner's configuration separately; do not assume it uses the historical eight-epoch protocol.
