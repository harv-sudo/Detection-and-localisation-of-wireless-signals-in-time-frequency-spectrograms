Model-only baseline results — 11 October 2026

The new EvidenceAxialNet establishes a compact alternative on the existing binary dataset. In this single-seed comparison it loses 0.40 percentage points of test macro F1, uses 98.1% fewer parameters and 94.3% fewer counted convolution MACs, and runs about 5.65 times faster on the tested CPU. It is slower on the tested GPU. This is an accuracy/size/CPU-latency trade-off, not an accuracy improvement or a verified novelty claim.

| Measurement | Original U-Net | EvidenceAxialNet |
|---|---:|---:|
| Test macro F1 | 0.944639 | 0.940678 |
| Test micro F1 | 0.940485 | 0.935869 |
| Test macro IoU | 0.907625 | 0.900806 |
| Test micro IoU | 0.887656 | 0.879468 |
| Parameters | 1,926,433 | 37,347 |
| Convolution MACs/image | 685,554,176 | 38,810,400 |
| CPU median forward, batch 1 | 10.311 ms | 1.823 ms |
| GPU median forward, batch 1 | 1.339 ms | 2.094 ms |
| GPU median forward, batch 64 | 8.006 ms | 9.749 ms |

Both models trained for eight epochs with Adam, learning rate 0.001, batch size 64 and BCE + per-image Dice loss. Seed 42 produced the same 62,400 training, 20,800 validation and 20,800 test indices; data hashes and exact test IDs were checked equal. U-Net's checkpoint was selected from epoch 5 and its threshold was 0.35. The candidate's checkpoint was selected from epoch 8 and its threshold was 0.45. Validation loss selected the checkpoint; validation macro F1 selected its threshold. There was no test-set threshold tuning.

The MATLAB generators, existing dataset loader, saved arrays, original U-Net and FCNN source were not edited. The candidate reads the same floating-point binary image and receives no SNR, true channel geometry or labels at inference. It combines directional local means, a compact axial convolutional encoder/decoder, and bounded edge-gated neighbour refinement. The latter remains anchored to unary logits and can suppress smoothing when incident edge weights are small.

Benchmarks used FP32, 50x100 binary inputs, random model weights, 30 warmup passes, 200 timed passes at batch one and 100 at batch 64. CPU: AMD Ryzen 7 9800X3D, four PyTorch threads. GPU: AMD Radeon RX 9070 XT, PyTorch 2.12.0+rocm7.14.0. Timings cover synchronized model forward execution, including dispatch, but exclude loading, transfers and output decoding. MAC counts exclude pooling, normalization, activation, interpolation, elementwise refinement and memory traffic. Random weights are appropriate for these dense architecture timings, but latency remains separate from trained-model accuracy.

The installed MIOpen BatchNorm compiler failed because it could not find the C++ type_traits header. U-Net training/evaluation therefore used the optional native-BatchNorm wrapper, preserving optimized convolution dispatch and the original layer equations. Both GPU architecture benchmarks used that same dispatch policy. The candidate uses GroupNorm and trained through the ordinary runner. Two earlier diagnostic U-Net runs were retained: one failed before training, and an all-native backend run was stopped after its first completed epoch in favour of the faster BatchNorm-only workaround. Neither contributes to the table.

All 15 regression tests passed. Checks cover forward/backward and checkpoint roundtrips, odd image dimensions, evidence padding, refinement suppression, metrics, same-size data modification detection, and BatchNorm dispatch/statistics. A real-data miniature train/evaluate integration run also completed. These checks establish implementation behavior, not publication novelty or robustness outside the current dataset.

Use the original U-Net as the accuracy reference and the candidate as the compact CPU-oriented baseline. Full component ablations and multiple training seeds remain outstanding. Pixel segmentation metrics do not establish per-transmission AP or robustness to changed signal layouts; the existing generator's fixed channel geometry limits conclusions.

Artifacts:

- [Architecture and commands](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/UNet/README.md>)
- [Candidate implementation](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/UNet/structured_model.py>)
- [Candidate trained checkpoint](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/runs/evidence_axial_seed42/best.pt>)
- [U-Net trained checkpoint](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/runs/unet_bn_native_seed42/best.pt>)
- [Machine-readable comparison, including per-SNR results](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/review_artifacts/model_comparison.json>)

The complete runs, checkpoints, source snapshots, manifests and split IDs remain under runs/. That directory is excluded from Git to avoid adding training artifacts. All implementation changes are uncommitted.

