# FastUNet

FastUNet preserves the U-Net encoder, decoder and skip connections, but uses narrower layers and simplifies its training blocks for deployment. The aim is to retain useful pixel-level accuracy while reducing U-Net's storage and inference cost.

```text
Input: 1 x 50 x 100
  -> 8 x 50 x 100 -> 16 x 25 x 50 -> 32 x 12 x 25
  -> bottleneck: 64 x 6 x 12
  -> 32 x 12 x 25 -> 16 x 25 x 50 -> 8 x 50 x 100
  -> output: 1 x 50 x 100 logits
```

Dimensions are channels x time x frequency. The standardised run uses base width 8. As in the original U-Net, skip connections preserve detail and upsampled features are resized to match skip dimensions where needed.

**Training block:** a 3 x 3 convolution and a 1 x 1 convolution operate in parallel. Each has BatchNorm; an identity-plus-BatchNorm branch is added when input and output channel counts match. Their outputs are summed before ReLU. Each stage contains two such blocks.

**Deployment block:** after `eval()`, `to_deploy()` merges the branches and frozen BatchNorm transformations into one 3 x 3 convolution followed by ReLU. This preserves the evaluation-mode computation up to floating-point rounding. It is established structural reparameterisation, not quantisation or pruning, and does not imply that the architecture combination is novel.

- **Parameters:** 134,257 during multi-branch training; **120,681 deployed**. The smaller deployment count is the one used in our comparison tables.
- **Training:** BCE plus per-image soft Dice loss. Output logits become a mask using sigmoid and a validation-selected threshold.
- **Efficiency:** narrower layers reduce convolutional work; merged branches remove inference operations. CPU performance also depends strongly on the execution settings. The optional prepared CPU graph is an additional execution optimisation, separate from branch merging.
- **Explainability:** similar to U-Net; a readable pixel mask does not establish a faithful explanation of the internal decision process.

Sources: [UNet/fast_model.py](../UNet/fast_model.py), [UNet/inference.py](../UNet/inference.py), [UNet/deployment.py](../UNet/deployment.py).
