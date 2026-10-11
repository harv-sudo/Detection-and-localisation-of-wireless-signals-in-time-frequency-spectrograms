FastUNet: speed experiment on the existing dataset

Completed results are in [the speed report](../review_artifacts/fast_model_results.md).
Test macro F1 is 0.943762 versus original U-Net's 0.944639 and EvidenceAxialNet's
0.940678. The deployed model measured 1.231 ms on CPU with preparation and
1.698 ms on GPU at batch one. These are matched local FP32 measurements from one
training seed; the report includes the equally optimized reference and runtime
variability. For zero changed test predictions, the original U-Net's optimized
checkpoint is also available at `runs/unet_bn_native_seed42/deploy.pt`.

FastUNet retains the original U-Net's three downsampling stages, skip connections,
transposed convolutions and exact-size decoder interpolation, with eight base
channels instead of 32. During training each convolution has a 3x3 branch, a 1x1
branch and, where dimensions agree, an identity branch. Each branch has its own
BatchNorm. During deployment their weights and frozen normalization statistics
merge into one 3x3 convolution. The deployed model has 120,681 parameters and
43,147,136 convolution MACs for a 50x100 image.

The input remains the existing floating-point binary image and the output remains
raw pixel logits. There is no signal-band coordinate, SNR or simulator metadata
input. The generator, saved data and dataset loader remain the existing ones.
Training uses the original comparison runner's split, optimizer, loss, eight-epoch
budget and validation selection rules. The additional training branches disappear
at inference; they increase training work rather than deployed compute.

Structural reparameterization is established prior art, including
[Ding et al., RepVGG (CVPR 2021)](https://arxiv.org/abs/2101.03697).
Neither branch merging nor its use here establishes a new research contribution.
An ablation with `--plain-blocks` can test whether the auxiliary training branches
help this task. Accuracy/latency measurements and multiple seeds are still needed
to substantiate any research claim.

Training and deployment commands from the repository root:

```powershell
.\.venv\Scripts\python.exe UNet/fast_experiment.py train --run-dir runs/fast_unet_b8_seed42 --base-channels 8 --epochs 8 --batch-size 64 --workers 4 --seed 42 --device cuda --native-batchnorm
.\.venv\Scripts\python.exe UNet/deployment.py --run-dir runs/fast_unet_b8_seed42 --device cuda --workers 4 --native-batchnorm
.\.venv\Scripts\python.exe UNet/deployment.py --run-dir runs/fast_unet_b8_seed42 --device cpu --workers 4 --optimize-cpu
```

Run the same deployment commands with `--run-dir runs/unet_bn_native_seed42`
to export and verify the original U-Net without retraining.

Use a fresh run directory for training. `--native-batchnorm` is a workaround for
the installed Windows ROCm BatchNorm compiler; it leaves convolution dispatch
enabled. It is unnecessary on installations with working optimized BatchNorm.
The runner fingerprints and archives the training source without changing the
older runner or invalidating its saved checkpoints. Deployment verifies the
training source, dataset and split fingerprints, exports `deploy.pt`, reloads it,
and evaluates it at the saved validation threshold. Conversion error and changed
prediction pixels are measured against the training graph over the full test set.

Load for GPU inference:

```python
import torch
from UNet.deployment import load_deployed_model

model, threshold = load_deployed_model("runs/fast_unet_b8_seed42/deploy.pt")
model = model.to("cuda").eval()
with torch.inference_mode():
    mask = model(binary_images.to("cuda")).sigmoid() > threshold
```

Prepare once for CPU inference:

```python
import torch
from UNet.deployment import load_deployed_model
from UNet.inference import prepare_cpu

torch.set_num_threads(4)
model, threshold = load_deployed_model("runs/fast_unet_b8_seed42/deploy.pt")
model = prepare_cpu(model, torch.zeros(1, 1, 50, 100))
with torch.inference_mode():
    mask = model(binary_images.cpu().float()).sigmoid() > threshold
```

The CPU preparation uses PyTorch's inference graph pass to select/fuse oneDNN
kernels. It also replaces nonoverlapping 2x2/stride-2 transposed convolutions
with an algebraically equivalent 1x1 convolution and pixel shuffle by permuting
their weights. The same replacement is available to the U-Net reference.
This matters on the local build: ordinary eager execution selects slow
fallback kernels for some small convolutions. The graph supports variable batch
sizes but is guarded to the prepared height/width and FP32 CPU input. Prepare a
separate graph for other image sizes. Preparation time is outside steady-state
latency. Save the portable weight artifact and prepare after loading; optimized
oneDNN graphs are not treated as portable serialized models.

This CPU path is tested on the installed PyTorch 2.12.0+rocm7.14.0/Python 3.14
environment. PyTorch marks the TorchScript inference pass as deprecated; treat
it as a measured local runtime option and revalidate after upgrades. The
[PyTorch implementation](https://github.com/pytorch/pytorch/blob/main/torch/jit/_freeze.py)
also documents its build-specific behavior. The model and portable weight
artifact remain ordinary PyTorch modules/state dictionaries.

Matched latency commands (run when training is finished):

```powershell
.\.venv\Scripts\python.exe UNet/speed_benchmark.py --device cpu --optimize-cpu --fast-run-dir runs/fast_unet_b8_seed42 --output review_artifacts/fast_cpu_optimized.json
.\.venv\Scripts\python.exe UNet/speed_benchmark.py --device cpu --fast-run-dir runs/fast_unet_b8_seed42 --output review_artifacts/fast_cpu_eager.json
.\.venv\Scripts\python.exe UNet/speed_benchmark.py --device cpu --channels-last --fast-run-dir runs/fast_unet_b8_seed42 --output review_artifacts/fast_cpu_channels_last.json
.\.venv\Scripts\python.exe UNet/speed_benchmark.py --device cuda --native-batchnorm --fast-run-dir runs/fast_unet_b8_seed42 --output review_artifacts/fast_gpu_batch1.json
.\.venv\Scripts\python.exe UNet/speed_benchmark.py --device cuda --batch-size 64 --native-batchnorm --fast-run-dir runs/fast_unet_b8_seed42 --output review_artifacts/fast_gpu_batch64.json
```

The benchmark includes the original U-Net, that same model with BatchNorm folded,
EvidenceAxialNet and the deployed FastUNet. CPU graph optimization is applied to
every model when requested. Timings use FP32, three shuffled rounds, 30 warmups
and 100 measurements per model per round, with GPU synchronization. They measure
model forward execution including dispatch, excluding loading, transfers and
mask decoding. GPU batch-64 timing is per batch; amortized per-image throughput
is a separate field. No quantization is used.

After the evaluations and all five benchmark commands finish, rebuild the
comparison with `.\.venv\Scripts\python.exe review_artifacts/summarize_fast_results.py`.

The data distribution still has fixed channel geometry. Pixel F1/IoU do not
establish arbitrary-frequency localization, per-transmission AP, real RF
robustness or statistical equivalence across training seeds.
