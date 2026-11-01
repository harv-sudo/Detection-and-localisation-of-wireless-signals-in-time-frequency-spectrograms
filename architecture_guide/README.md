# Architecture guide

Brief explanations of how the models turn noisy binary time-frequency grids
into estimated signal areas. Inputs are **50 time bins x 100 frequency bins**.
None of these models receives the true SNR as an input.

| Model | Main idea | Read more |
|---|---|---|
| Original U-Net | Compress the image to learn context, then reconstruct a pixel mask using skip connections. | [Architecture and working](original_unet.md) |
| FastUNet | Use a narrower U-Net and merge its training branches into simple convolutions for deployment. | [Architecture and working](fast_unet.md) |
| EvidenceAxialNet | Combine local counts, directional filtering and learned neighbour refinement. | [Architecture and working](evidence_axial_net.md) |
| FCNN recreation | Flatten the image and predict rectangle coordinates through dense layers. | [Architecture and working](fcnn.md) |
| F1-oriented U-Net++ candidate | Use nested decoder connections, residual blocks and spatial attention to pursue accuracy. | [Architecture and working](f1_unet.md) |

The first four have completed the standardised eight-epoch comparison below.
The F1-oriented candidate is a separate development; these scores do not describe it.

## Current results

Same 62,400 training / 20,800 validation / 20,800 test images; eight epochs;
Adam at 0.001; batch 64; seed 42. F1 is foreground pixel F1 calculated for each
image and then averaged. The FCNN retains coordinate MSE, while the segmentation
models use BCE + Dice. Matching data and budget does not make their losses identical.

| Model | Test macro F1 | Test macro IoU | Parameters | CPU batch 1 | GPU batch 1 | GPU batch 64 |
|---|---:|---:|---:|---:|---:|---:|
| Original U-Net | 0.944639 | 0.907625 | 1,926,433 | 1.953 ms | 1.432 ms | 7.957 ms |
| EvidenceAxialNet | 0.940678 | 0.900806 | 37,347 | 1.577 ms | 2.143 ms | 9.820 ms |
| FastUNet, deployed | 0.943762 | 0.905776 | 120,681 | 0.755 ms | 0.874 ms | 1.987 ms |
| FCNN recreation | 0.821329 | 0.711164 | 1,498,192 | 0.265 ms | 0.804 ms | 0.699 ms |

Timings are median model-forward times on a Ryzen 7 9800X3D (four CPU threads)
and Radeon RX 9070 XT, FP32. CPU values select each model's fastest measured
mode from the same three offered modes; GPU execution is eager. Each setting
uses three shuffled rounds with 30 warmups and 100 timed passes per model per
round. Loading, transfers and output decoding are excluded. GPU batch 64 is the
time for the whole batch. Single-seed scores do not establish statistical equivalence.

**Why FCNN is faster:** parameters count stored weights, not how many times they
are used. FCNN has about 1.50 million convolution/linear multiply-accumulates per
image, versus FastUNet's 43.15 million and U-Net's 685.55 million. Convolutions
reuse weights at many image positions. FastUNet improves the U-Net speed/accuracy
trade-off; it does not beat FCNN on raw forward time in this experiment.

## Documents and reproducibility

- [Performance figure](../output/pdf/Performance.pdf)
- [Inference speed figure](../output/pdf/Inference%20Speed.pdf)
- [Hardware and training setup](../output/pdf/Experiment%20Setup%20and%20Metrics.pdf)
- [Saved standardised results and plotting instructions](../review_artifacts/standardised_results/README.md)
- [Matched FCNN training and benchmarking commands](../baseline_recreation/README.md)
- [Segmentation experiments](../UNet/README.md) and [FastUNet deployment](../UNet/FAST_MODEL.md)

The tracked result snapshot makes the measurements available without training
again. Generated datasets, model checkpoints and Python environments remain
excluded from Git. Running or retraining the models on another computer requires
those inputs and the appropriate dependencies. Exact historical retraining also
requires the saved training-source snapshots: the current trainer has evolved
since the eight-epoch runs. Use new run directories rather than overwriting them.

Older figures in `review_artifacts/supervisor_comparison` and the LaTeX examples
precede the matched FCNN run; use the PDFs linked above for the current results.
The paper curves/timings are cross-study references, not the results of our FCNN
recreation or controlled hardware comparisons. No architectural novelty or
validated explanation-faithfulness claim follows from these experiments alone.
