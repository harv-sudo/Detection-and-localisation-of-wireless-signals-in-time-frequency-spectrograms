EvidenceAxialNet is a compact segmentation candidate for the existing binary spectrograms. `structured_model.py` contains the candidate and `experiment.py` compares it with the original U-Net using a common training and evaluation protocol. The original `model.py`, dataset loaders, MATLAB generation code and generated data are unchanged.

For the subsequent CPU/GPU speed experiment, see [FastUNet and deployment commands](FAST_MODEL.md) and [measured results](../review_artifacts/fast_model_results.md). It adds a narrow dense model, branch merging, and optional CPU inference preparation while preserving this baseline and its source fingerprints.

Completed single-seed results are recorded in `review_artifacts/model_baseline_results.md` and `review_artifacts/model_comparison.json`: the candidate's test macro F1 is 0.9407 versus U-Net's 0.9446, with 98.1% fewer parameters. It is faster on the tested CPU but slower on the tested GPU. These results establish a compact baseline, not an accuracy or originality claim. Existing completed run directories must not be reused for training; choose a fresh run name for subsequent experiments.

Both models receive floating-point binary tensors shaped `(batch, 1, 50, 100)` and return raw segmentation logits with the same shape. Height represents time and width represents frequency. The candidate receives no SNR, coordinates, labels or other metadata as model inputs. SNR is used only to group evaluation results.

The candidate has three components:

1. **Directional count evidence.** Six deterministic local means are appended to the observed binary image: frequency spans 3, 9, 17 and 33, followed by time spans 3 and 7. Each mean is the fraction of observed ones in that window. Padding is excluded from the denominator. These channels summarize nearby binary observations; they are not calibrated signal probabilities or recovered spectrogram power.
2. **Compact axial convolutions.** The encoder uses widths `b`, `2b` and `4b`, with `b=16` by default. Residual blocks combine depthwise `1x5` frequency and `5x1` time convolutions, pointwise channel mixing and GroupNorm. Two decoder stages resize to the actual skip sizes, including odd spatial dimensions.
3. **Anchored neighbour refinement.** A shared learned gate predicts an undirected weight for each pair of adjacent pixels from the mean and absolute difference of their features. The gate is symmetric under exchanging the endpoints. Three refinement iterations are used by default.

For unary logits `u`, refinement starts at `z_0 = u` and updates each pixel `i` as:

```text
lambda = 2 * sigmoid(raw_strength)
z_(k+1)[i] = u[i] + lambda * sum_j(w_ij * (2 * sigmoid(z_k[j]) - 1)) / (1 + sum_j(w_ij))
```

The sum includes only existing immediate neighbours. The unit term in the denominator anchors the correction: when edge weights approach zero, the correction also approaches zero. Every iteration resets to the fixed unary logits instead of accumulating previous logits. Coupling starts near 0.036 and remains below 2. This is a finite, mean-field-style update; it does not establish exact CRF inference or calibrated uncertainty.

The default architecture has the following static counts for one `50x100` input:

| Model | Trainable parameters | Convolution MACs per image |
| --- | ---: | ---: |
| Original U-Net | 1,926,433 | 685,554,176 |
| EvidenceAxialNet, width 16, three steps | 37,347 | 38,810,400 |

A MAC denotes one multiply-accumulate. These counts cover convolutions and exclude pooling, normalization, activations, interpolation, elementwise refinement and data movement. The candidate uses about 98.1% fewer parameters and 94.3% fewer convolution MACs. These are architecture counts, not measurements of latency, energy or accuracy. No accuracy advantage is implied by these counts.

`benchmark.py` measures both architectures on the same fixed binary input with warmup and synchronized host-wall timing. Run it after training finishes to avoid concurrent GPU work:

```powershell
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
& $python UNet/benchmark.py --device cuda --batch-size 1 --output runs/latency_batch1.json
& $python UNet/benchmark.py --device cuda --batch-size 64 --output runs/latency_batch64.json
```

These are evaluation-mode FP32 forward times with random weights, including Python dispatch and synchronization but excluding loading, transfer and mask/box decoding. Batch-one latency and amortized batch throughput are different quantities. The script reports both median and p95 batch times; lower MACs do not guarantee lower measured latency.

Use a Python environment containing `torch`, `numpy` and `h5py`. Install the PyTorch build appropriate to the machine's CPU, CUDA or ROCm setup; this project does not prescribe a GPU-specific wheel. In PowerShell, run the following from the repository root, adjusting the interpreter path if necessary:

```powershell
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
& $python -m pip install numpy h5py
& $python -B UNet/experiment.py --help
& $python -B UNet/experiment.py smoke --model unet --device cpu
& $python -B UNet/experiment.py smoke --model evidence_axial --device cpu
```

Smoke tests use synthetic binary tensors and check shape, finite loss and finite gradients. They do not measure model quality. Dataset imports are deferred, so help and smoke do not require HDF5 files. `--device auto` selects CUDA/ROCm when available and otherwise uses CPU; PyTorch exposes ROCm devices through its `cuda` API. `--threads 4` is the default CPU thread count.

On the tested Windows ROCm installation, optimized BatchNorm compilation failed with a missing `type_traits` C++ header. `native_backend.py` is an optional workaround using exactly the same CLI: it sends only BatchNorm through native PyTorch operations, restoring the backend flag before each convolution. It preserves the original model layers and loss, and writes `backend.json` after a completed training run. It uses a process-local function wrapper intended for a single training thread. For example:

```powershell
& $python UNet/native_backend.py train --model unet --run-dir runs/unet_bn_native_seed42 --epochs 8 --batch-size 64 --workers 4 --device cuda
& $python UNet/native_backend.py evaluate --run-dir runs/unet_bn_native_seed42 --workers 4 --device cuda
& $python UNet/benchmark.py --native-batchnorm --device cuda --batch-size 1 --output runs/latency_batch1.json
```

The benchmark applies this dispatch policy to both architectures. Backend changes can affect floating-point behavior and performance, so report the recorded backend alongside results. The ordinary CLI remains appropriate on systems without this compiler problem. Tests cover flag restoration even on failure and preservation of BatchNorm outputs, statistics and gradients on CPU.

Train both models with the same budget and evaluate their saved best checkpoints:

```powershell
& $python -B UNet/experiment.py train --model unet --run-dir UNet/runs/unet_seed42 --epochs 8 --batch-size 64 --lr 0.001 --seed 42 --device auto
& $python -B UNet/experiment.py train --model evidence_axial --run-dir UNet/runs/evidence_seed42 --epochs 8 --batch-size 64 --lr 0.001 --seed 42 --device auto
& $python -B UNet/experiment.py evaluate --run-dir UNet/runs/unet_seed42 --device auto
& $python -B UNet/experiment.py evaluate --run-dir UNet/runs/evidence_seed42 --device auto
```

Data defaults to `UNet/data`, resolved relative to the script. Set `--data-dir PATH` to select another existing dataset without changing generation. `--workers` defaults to zero. A training run requires a new or empty `--run-dir`; the CLI refuses to overwrite an existing run. Evaluation accepts `--checkpoint latest` as an alternative to the default `best`. It expects checkpoints from this harness, not the older standalone training script.

For a small integration check, use separate run directories and sample caps:

```powershell
& $python -B UNet/experiment.py train --model evidence_axial --run-dir UNet/runs/integration --epochs 1 --batch-size 8 --max-train-samples 16 --max-eval-samples 8 --device cpu
& $python -B UNet/experiment.py evaluate --run-dir UNet/runs/integration --max-eval-samples 8 --device cpu
```

During training, `--max-eval-samples` limits validation. During evaluation it limits the test split. Caps take the first entries of the saved randomized split, and their exact IDs are retained. Capped runs are integration checks, not full benchmark results. Full dataset hashes are still computed for capped runs.

Use this four-condition ablation to isolate the evidence channels and refinement. Keep all other settings identical and compare the resulting checkpoints on the same held-out IDs:

```powershell
& $python -B UNet/experiment.py train --model evidence_axial --run-dir UNet/runs/ablation_full_s42 --epochs 8 --batch-size 64 --seed 42 --device auto
& $python -B UNet/experiment.py train --model evidence_axial --no-evidence --run-dir UNet/runs/ablation_no_evidence_s42 --epochs 8 --batch-size 64 --seed 42 --device auto
& $python -B UNet/experiment.py train --model evidence_axial --no-refinement --run-dir UNet/runs/ablation_no_refinement_s42 --epochs 8 --batch-size 64 --seed 42 --device auto
& $python -B UNet/experiment.py train --model evidence_axial --no-evidence --no-refinement --run-dir UNet/runs/ablation_backbone_s42 --epochs 8 --batch-size 64 --seed 42 --device auto
```

Evaluate each directory with the same `evaluate --run-dir ...` command. Removing either component changes the architecture and its parameter count; report those counts alongside metrics. `--base-channels` and `--refinement-steps` provide further capacity/cost experiments. The model requires width at least four; use `--no-refinement` to disable refinement through the CLI. Repeat matched comparisons with several training seeds and new run directories before drawing research conclusions.

The harness preserves the original split rule: 60% training, 20% validation and the remainder test, using `torch.utils.data.random_split` with **split seed 42**. `--seed` controls model/training randomness and defaults to 42; changing it does not change the holdout. Both models use Adam and the same unweighted BCE-with-logits plus per-image soft Dice loss, with Dice smoothing `1e-6`. The default learning rate is `0.001`. Non-finite loss or gradient norms terminate the run. Seeded runs and deterministic cuDNN settings do not guarantee identical results across different software or hardware stacks.

At each epoch, validation sweeps probability thresholds `0.05, 0.10, ..., 0.95`. Maximum validation **macro F1** selects that epoch's threshold; ties prefer the value closest to 0.5, then the lower threshold. Minimum validation loss selects `best.pt`. The selected threshold is stored inside its checkpoint. Test evaluation uses that value directly and never tunes a threshold on test labels. A custom validation grid can be supplied with `--thresholds`.

Predictions use `sigmoid(logits) > threshold`, and targets use `target > 0.5`. JSON reports F1, IoU, precision and recall overall and per SNR:

- **Micro** scores are computed from summed pixel TP, FP and FN.
- **Macro** scores are the unweighted mean of per-image scores.
- If prediction and target are both empty, all four scores equal one. Otherwise, a precision or recall with a zero denominator equals zero. Empty-target counts are included.

Micro and macro scores answer different questions and should be compared using the same convention. Pixel segmentation metrics do not directly measure instance count, individual signal detections or physical time/frequency boundary accuracy.

Each run saves:

| File | Contents |
| --- | --- |
| `config.json` | CLI configuration, package versions, resolved device and source SHA256 hashes |
| `manifest.json` | Sorted data filenames, sizes, per-file SHA256 hashes, total sample count and combined fingerprint |
| `splits.pt` | Exact train/validation/test indices and effective capped training/validation IDs |
| `best.pt`, `latest.pt` | Model/optimizer states, epoch, configuration, validation metrics and threshold, data fingerprint and split-file hash |
| `history.json` | Epoch losses, validation results, threshold sweeps and elapsed times |
| `evaluation.json` | Full test metrics, exact test IDs and evaluation settings |
| `evaluation_subset.json` | Capped test evaluation, when requested |

Evaluation verifies the complete data manifest, the saved split-file hash and the training source hashes for `experiment.py`, `model.py`, `structured_model.py` and `dataset.py`. Even a source-only edit will require restoring the recorded source before evaluating that run. Hashes identify source versions but do not archive their contents; preserve the corresponding repository revision with the run. An unchanged copy of the dataset can be evaluated at another path through `--data-dir`. Rerunning evaluation replaces the corresponding evaluation JSON in that run directory. The CLI does not currently implement training resumption.

The model exposes diagnostics without changing its ordinary logits interface:

```python
import torch
from UNet.structured_model import EvidenceAxialNet

model = EvidenceAxialNet(base_channels=16, refinement_steps=3)
model.eval()
x = torch.randint(0, 2, (2, 1, 50, 100)).float()
with torch.no_grad():
    logits, auxiliary = model(x, return_aux=True)

evidence = auxiliary["evidence_maps"]
unary_logits = auxiliary["unary_logits"]
horizontal_weights = auxiliary["horizontal_edge_weights"]
vertical_weights = auxiliary["vertical_edge_weights"]
refined_logits = auxiliary["refined_logits"]
strength = auxiliary["refinement_strength"]
```

Evidence maps have six channels in the order listed above. Horizontal weights have shape `(B, 1, H, W-1)` and vertical weights `(B, 1, H-1, W)`. Disabled evidence produces zero channels; inactive refinement reports zero edge weights and strength. Diagnostic tensors retain gradients when called outside a no-gradient context, so detach them before retaining visualizations.

These maps make intermediate computations inspectable, but do not establish faithful explanations. Test their meaning through ablations and controlled interventions, such as deleting a signal segment or disrupting a boundary, and check whether the predicted changes follow the proposed explanation. Local averaging can blur narrow structures, and refinement can merge nearby signals. GroupNorm uses spatial statistics, so this model is not a causal streaming detector. The network cannot recover analog information discarded by the existing binarization. Keeping the current generator and split fixed also limits conclusions to the current experimental distribution.

The research hypothesis is that a compact, direction-aware model with inspectable binary-count evidence and bounded neighbour corrections can improve the accuracy/compute trade-off on this task. Establishing a contribution requires matched experiments, component ablations, multiple seeds and measured latency or energy on stated hardware. Parameter and MAC reductions alone do not establish either improved accuracy or deployment speed.

The ingredients have relevant prior art. [Zheng et al., *Conditional Random Fields as Recurrent Neural Networks* (ICCV 2015)](https://arxiv.org/abs/1502.03240) integrate recurrent mean-field CRF inference with segmentation networks. This candidate uses local learned affinities and a finite anchored update rather than reproducing that CRF formulation. [Hou et al., *Strip Pooling: Rethinking Spatial Pooling for Scene Parsing* (CVPR 2020)](https://arxiv.org/abs/2003.13328) study narrow directional pooling for spatial context. The binary-count channels here are task-specific local windows, but directional pooling itself is established. This implementation is a candidate for a model-focused study; it is not a claim of unverified originality.
