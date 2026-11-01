# Original U-Net

Our reference segmentation model reads a binary spectrogram and predicts a signal score for every pixel. It keeps the image's two-dimensional layout throughout the network.

**Working:** the encoder repeatedly applies filters and reduces the image resolution, learning broader patterns. The decoder enlarges these features back to the input size. Skip connections copy fine detail from each encoder stage into the corresponding decoder stage, helping reconstruct signal boundaries.

```text
Input: 1 x 50 x 100
  -> 32 x 50 x 100 -> 64 x 25 x 50 -> 128 x 12 x 25
  -> bottleneck: 256 x 6 x 12
  -> 128 x 12 x 25 -> 64 x 25 x 50 -> 32 x 50 x 100
  -> output: 1 x 50 x 100 logits
```

Dimensions above are channels x time x frequency; the batch dimension is omitted. Each main stage uses two 3 x 3 convolutions with BatchNorm and ReLU. Downsampling uses max pooling; upsampling uses transposed convolutions, with interpolation to match odd-sized skip features. A final 1 x 1 convolution produces logits. Sigmoid followed by a validation-selected threshold converts these into a binary signal mask.

- **Parameters:** 1,926,433 in the original model; 1,925,025 after optional BatchNorm folding at deployment. Folding is a deployment transformation of the same trained model, not a separate learned architecture.
- **Training:** BCE plus per-image soft Dice loss against clean masks.
- **Compute:** filters are reused at many image locations, so this can require much more computation than a fully connected network despite sharing weights spatially.
- **Explainability:** its output shows where it predicts a signal, but that localisation is not by itself an explanation of why the model made the prediction.

Source: [UNet/model.py](../UNet/model.py). Optional folding: [UNet/fast_model.py](../UNet/fast_model.py), `fuse_unet_batchnorm`.
