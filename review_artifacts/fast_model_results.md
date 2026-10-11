FastUNet speed experiment — 11 October 2026

All accuracy scores below use the same 20,800 test images and saved validation thresholds. Training, validation and test indices and dataset hashes were checked equal. CPU timings select each model's lowest median among three measured modes: eager, channels-last eager, and inference-graph preparation. All three modes are offered to every model; GPU timings use eager execution with native BatchNorm where present. All computations are FP32.

| Model | Test macro F1 | Test macro IoU | Parameters | CPU batch 1 | GPU batch 1 | GPU batch 64 |
|---|---:|---:|---:|---:|---:|---:|
| Original U-Net | 0.944639 | 0.907625 | 1,926,433 | 3.747 ms | 2.626 ms | 14.450 ms |
| U-Net, BN folded | 0.944639 | 0.907625 | 1,925,025 | 3.789 ms | 2.168 ms | 12.942 ms |
| EvidenceAxialNet | 0.940678 | 0.900806 | 37,347 | 2.412 ms | 3.796 ms | 15.076 ms |
| FastUNet, deployed | 0.943762 | 0.905776 | 120,681 | 1.231 ms | 1.698 ms | 4.025 ms |

Selected CPU modes: Original U-Net: cpu; U-Net, BN folded: cpu; EvidenceAxialNet: cpu_channels_last; FastUNet, deployed: cpu.

Speed ratios are reference latency divided by FastUNet latency:

- Versus Original U-Net: 3.04x CPU, 1.55x GPU batch 1, 3.59x GPU batch 64; macro F1 difference -0.088 percentage points.
- Versus U-Net, BN folded: 3.08x CPU, 1.28x GPU batch 1, 3.22x GPU batch 64; macro F1 difference -0.088 percentage points.
- Versus EvidenceAxialNet: 1.96x CPU, 2.23x GPU batch 1, 3.75x GPU batch 64; macro F1 difference +0.308 percentage points.

There is also an option requiring no retraining: optimize the existing U-Net checkpoint. Its CPU latency changes from 14.254 ms eager to 3.789 ms with CPU preparation (3.76x), and GPU latency changes from 2.626 ms to 2.168 ms by folding BatchNorm (1.21x). This preserves the learned model, subject to floating-point rounding.

FastUNet's validation-loss-selected checkpoint is epoch 7; its validation-selected threshold is 0.40. It trained for eight epochs on 62,400 images with 20,800 validation images, Adam at 0.001, batch 64, BCE + per-image Dice, training seed 42 and fixed split seed 42. Its auxiliary branches are merged after training. Deployment uses 43,147,136 counted convolution MACs per 50x100 image.

Both the reloaded deployed GPU models and CPU inference graphs were evaluated over the full test set. The following conversion checks compare their logits/predictions with the unfused training graph on the same device, using the same threshold:

- fast_gpu: maximum absolute logit difference 3.5762787e-05; 1/104,000,000 prediction pixels changed.
- fast_cpu: maximum absolute logit difference 3.1471252e-05; 0/104,000,000 prediction pixels changed.
- unet_cpu: maximum absolute logit difference 1.0490417e-05; 0/104,000,000 prediction pixels changed.
- unet_gpu: maximum absolute logit difference 2.7179718e-05; 0/104,000,000 prediction pixels changed.

Timing protocol: AMD Ryzen 7 9800X3D CPU, four PyTorch threads; AMD Radeon RX 9070 XT GPU; PyTorch 2.12.0+rocm7.14.0. Three shuffled rounds, 30 warmups and 100 measured passes per model per round, with device synchronization. Trained weights are loaded. Timings exclude graph preparation, data loading, host/device transfers and prediction decoding. Batch-64 times are for the entire batch. The model's smaller parameter/MAC count does not by itself explain these measured latencies.

Absolute timings differed from the earlier random-weight baseline session, so speed ratios here use the new matched runs rather than mixing old and new measurements. CPU round-to-round variability was material; the JSON includes every round median and p95. These are local measurements, not a guarantee for other machines or background workloads.

Deployment details matter: ordinary eager CPU execution hit slow small-convolution fallbacks on this build. FastUNet's prepared CPU result requires prepare_cpu(), which selects/fuses oneDNN kernels and rewrites eligible nonoverlapping transposed convolutions as equivalent 1x1 convolutions plus pixel shuffle. Every model is tested with that CPU optimization and both eager layouts; the table uses each model's fastest measured mode. This TorchScript-based runtime option is deprecated upstream and must be revalidated on future environments; portable state dictionaries are saved, and the CPU graph is prepared locally after loading. GPU inference uses ordinary folded modules.

Interpretation limits: this is one training seed and one fixed simulated distribution. Close point estimates do not establish statistical equivalence. Pixel F1/IoU do not establish per-transmission localization AP or robustness to new frequency layouts. There is no quantization, distillation or data-generation change. Narrow U-Nets, BN folding and structural reparameterization are established techniques; novelty has not been demonstrated. A plain-block ablation and additional seeds remain research follow-ups.

All 23 regression tests passed, covering branch/BN merging, checkpoint roundtrips, variable-batch CPU graphs, spatial guards, the subpixel weight permutation, training gradients, metric conventions and the existing structured-model checks. Original generator, loader and model source diffs are empty.

Artifacts: [model and commands](../UNet/FAST_MODEL.md), [machine-readable comparison](fast_model_comparison.json), [deployed weights](../runs/fast_unet_b8_seed42/deploy.pt). Complete runs and source snapshots are in runs/. The original implementation and data-generation files remain unchanged.
