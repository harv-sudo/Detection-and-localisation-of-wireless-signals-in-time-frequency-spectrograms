# FCNN recreation

The local FCNN is a fully connected, paper-style DL-ANN baseline. Instead of predicting each pixel independently, it predicts rectangular signal areas directly.

```text
Binary 50 x 100 grid -> flatten to 5,000 values
  -> 256 -> 256 -> 256 -> 256
  -> 80 sigmoid outputs = 20 (x, y, width, height) boxes
```

Each hidden layer uses ReLU and dropout with probability 0.4. Sigmoid keeps the final normalised coordinates between 0 and 1. During evaluation, dropout is disabled.

**Working:** flattening lets the dense layers combine information from anywhere in the image. The model learns a fixed-size list of up to 20 box slots. Its outputs are rounded and clipped to image bounds, then the union of the boxes is turned into a binary mask for pixel-based comparison with the segmentation models. Invalid or zero-area boxes contribute no pixels. There is no pixel-probability threshold.

- **Parameters:** 1,498,192.
- **Training:** mean squared error on normalised box coordinates, including the padded zero box slots. The matched runner uses the same dataset split, eight-epoch budget and seed as the other models, but the target representation and loss differ.
- **Speed:** the network has relatively few layer operations and only 80 outputs. Dense weights are not repeatedly applied over every spatial position as convolutional filters are, explaining how this model can run faster than FastUNet despite having more parameters. Our forward-only timings exclude box-to-mask decoding.
- **Explainability:** the box coordinates are easy to read, but they do not explain which input evidence caused a prediction. The dense hidden representation is not intrinsically transparent.
- **Reference status:** this is our recreation, not the paper's original implementation or weights. Our dropout is placed after every hidden layer; the paper's diagram shows input dropout. It is closest in input information to DL-ANN Option 3 because no SNR input is supplied.

Sources: [baseline_recreation/model.py](../baseline_recreation/model.py), [baseline_recreation/matched_experiment.py](../baseline_recreation/matched_experiment.py). Run instructions: [baseline_recreation/README.md](../baseline_recreation/README.md).
