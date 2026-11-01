# FCNN under the saved comparison protocol

Use `matched_experiment.py` for the comparison with the plotted Original U-Net,
EvidenceAxialNet and FastUNet runs. The historical scripts and FCNN architecture
are preserved. No data needs generating or copying. Run these commands from the
repository root in PowerShell using the existing environment.

## Train and evaluate

```powershell
.\.venv\Scripts\python.exe baseline_recreation/matched_experiment.py check --device cuda
.\.venv\Scripts\python.exe baseline_recreation/matched_experiment.py train --device cuda
.\.venv\Scripts\python.exe baseline_recreation/matched_experiment.py evaluate --device cuda
```

`cuda` also selects the AMD GPU through PyTorch ROCm. Use `--device cpu` for CPU.
The check command fingerprints all files, checks a training sample from every
chunk against its clean mask, and checks finite model gradients without saving a
run. Training does not evaluate the test set; evaluate loads the best checkpoint.

All generated outputs stay in **`runs/fcnn_matched_seed42/`**, already excluded
from Git. The directory contains config, manifest, exact split IDs, source
snapshots, best/latest checkpoints, history and summary; evaluation adds one JSON
with overall and per-SNR F1/IoU/precision/recall. A non-empty run directory is
refused to prevent overwriting experiments. To repeat, pass a fresh `--run-dir`
to train and use that same path with evaluate/benchmark.

## What matches

- Exact data hashes and saved split IDs from `runs/unet_bn_native_seed42`:
  62,400 training / 20,800 validation / 20,800 test images.
- Eight epochs, batch 64, constant Adam learning rate 0.001, seed 42, FP32,
  four loader workers and four CPU threads; no scheduler or early stopping.
- Strictly lowest validation loss selects the checkpoint, as in the *saved*
  eight-epoch runs. Later edits to the U-Net trainer are not used for training.
- Shared macro/per-image and micro/pooled metrics, with the original saved clean
  masks as ground truth. No SNR is supplied to the network.

FCNN retains its coordinate MSE loss, including padded box slots. Segmentation
models used BCE + Dice. This matches data and training budget, not loss functions
or compute time. The existing FCNN architecture and its dropout placement are
preserved; this remains the local recreation, not an exact reimplementation of
every detail in the paper. Coordinates are rounded/clipped using the historical
rule and combined into a mask. There is no pixel-probability threshold to tune.

The reference data path is read from its saved config. Use `--data-dir` only if
the same files moved; hashes must still match. `--max-train-samples` and
`--max-eval-samples` are only for smoke tests, and mark a run as unmatched.

## Timing all models together

Run after training finishes, with no other training running. Each command
re-measures all models, including the original and BN-folded U-Net, in shuffled
rounds. It writes its result into the FCNN run directory. Do not append a new
FCNN time to the old October timings and call that a same-session comparison.

```powershell
# Offer the same three CPU modes to every model.
.\.venv\Scripts\python.exe baseline_recreation/matched_experiment.py benchmark --device cpu
.\.venv\Scripts\python.exe baseline_recreation/matched_experiment.py benchmark --device cpu --channels-last
.\.venv\Scripts\python.exe baseline_recreation/matched_experiment.py benchmark --device cpu --optimize-cpu

# GPU latency and batch throughput.
.\.venv\Scripts\python.exe baseline_recreation/matched_experiment.py benchmark --device cuda --batch-size 1
.\.venv\Scripts\python.exe baseline_recreation/matched_experiment.py benchmark --device cuda --batch-size 64
```

Protocol: FP32, 50x100 inputs; 30 warmups plus 100 timed forwards per model in
each of three shuffled rounds; explicit GPU synchronization. Results report
median/p95, per-round medians, parameter/MAC counts and checkpoint hashes.
Times exclude preparation, loading, transfers and output decoding. Batch-64
times cover the whole batch. The same preparation/layout is offered to each
model; CPU best-of-mode summaries must state that selection policy.

The benchmark checks inference-architecture fingerprints against trained
checkpoints. Formatting-only changes are accepted only when the parsed code is
identical to a hash-verified training snapshot. Unrelated later changes to a training runner do not block timing;
accuracy evaluation retains full source/dataset/split integrity checks. The
segmentation checkpoint paths default to the three saved runs above; alternative
paths can be passed through `UNet/speed_benchmark.py` directly.
