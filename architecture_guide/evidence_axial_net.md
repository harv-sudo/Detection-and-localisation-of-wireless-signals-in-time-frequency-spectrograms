# EvidenceAxialNet

EvidenceAxialNet combines explicit local signal-density measurements, a small segmentation network and learned neighbour refinement. It offers inspectable intermediate quantities while keeping the number of parameters small.

**1. Local evidence.** From the input binary grid, it calculates six moving averages: four across frequency (window widths 3, 9, 17 and 33) and two across time (heights 3 and 7). These measure the fraction of active pixels nearby at different scales. Padding is excluded from the average. The original image plus these six maps form a seven-channel input; no true SNR or fixed signal position is supplied.

**2. Compact encoder/decoder.** Strided convolutions reduce resolution. Residual blocks use separate depthwise 1 x 5 frequency and 5 x 1 time convolutions, then 1 x 1 channel mixing. GroupNorm and SiLU are used. The decoder interpolates features to the skip sizes, combines them with encoder detail and produces initial, or *unary*, pixel logits.

```text
Input + evidence: 7 x 50 x 100
  -> 16 x 50 x 100 -> 32 x 25 x 50 -> 64 x 13 x 25
  -> 32 x 25 x 50 -> 16 x 50 x 100
  -> unary logits -> 3 neighbour-refinement steps -> final logits
```

Dimensions are channels x time x frequency. These widths describe the standardised base-width-16 model.

**3. Neighbour refinement.** Learned gates assign symmetric weights between adjacent pixels. Each step adds a bounded neighbour correction to the original unary logits, using the previous step's signed probabilities. It does not keep accumulating logits. The correction strength is learned and bounded below 2. This is a finite, mean-field-style procedure, not exact probabilistic inference.

- **Parameters:** 37,347 with evidence and refinement enabled.
- **Training:** BCE plus per-image soft Dice loss; sigmoid and a validation-selected threshold produce the final mask.
- **Inspection:** `forward(x, return_aux=True)` exposes evidence maps, unary and refined logits, neighbour weights and refinement strength. They help inspect what changes inside the model, but have not been validated as faithful explanations.
- **Limitations:** local averaging can blur boundaries and refinement can merge nearby signals. GroupNorm uses spatial statistics, so this is not a causal streaming model. Small parameter count does not guarantee low latency: multiple small operations and refinement steps have execution overhead.

Source: [UNet/structured_model.py](../UNet/structured_model.py).
