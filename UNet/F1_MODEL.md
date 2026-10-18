# F1-first experiment on an H100

This experiment targets **test macro pixel F1** on the existing dataset. F1UNet
has 11,733,827 parameters: a wider residual U-Net++ style decoder, dense skip
connections, global bottleneck attention and two auxiliary training heads.
The final head alone produces ordinary inference logits. No SNR value or
hardcoded occupied frequency band is supplied to the network.

The generator, loader, existing models and their experiment runners are
unchanged. Existing saved runs remain usable. New runs reproduce the original
60/20/20 split with split seed 42. With all 104,000 original images this means
62,400 training, 20,800 validation and 20,800 test images.

**Accuracy has not yet been measured for this model.** The reference to beat is
the original U-Net's test macro F1 of **0.944639**, including **0.780538 at
-20 dB**. Neither a larger model nor longer training guarantees improvement.

## Run on the remote H100

Run the commands below from the repository root on the remote machine. Copy
the existing `.mat` files from `UNet/data` as well as the code: **Git ignores
the data and saved checkpoints, so cloning the repository alone is insufficient**.
Use exactly the same files; there is no need to regenerate anything. You can
replace `UNet/data` below with their remote location.

Use the provider's working CUDA PyTorch environment if it already has one.
For a fresh Linux environment, an explicitly published CUDA build is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy h5py
```

That wheel requires a compatible NVIDIA driver. The
[official PyTorch build list](https://pytorch.org/get-started/previous-versions/)
lists other CUDA versions if the remote provider specifies one. This experiment
does not use the local AMD/ROCm installation or its BatchNorm workaround.

Check the remote GPU and run a short forward/backward/optimizer check:

```bash
python -c "import torch; assert torch.cuda.is_available() and torch.version.cuda, 'CUDA PyTorch required'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_bf16_supported(), 'BF16 support required'"
python UNet/f1_experiment.py smoke --device cuda --precision bf16
```

Train one model, then evaluate its validation-selected checkpoint:

```bash
python UNet/f1_experiment.py train --data-dir UNet/data --run-dir runs/f1_seed42 --device cuda --precision bf16 --epochs 100 --batch-size 64 --workers 8 --seed 42
python UNet/f1_experiment.py evaluate --data-dir UNet/data --run-dir runs/f1_seed42 --device cuda --workers 8
```

This uses one GPU. `--device cuda:1` selects another GPU if needed. `--workers 8`
controls CPU data-loading processes; reduce it if the remote machine has fewer
CPU cores or limited shared memory. The default model width is 64. Training
uses BF16 autocast with FP32 parameters/loss reductions; validation and test
use FP32 with TF32 disabled. [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html).

The run directory must initially be empty. If a session disconnects or a job
stops, keep the same source files and resume from the last completed epoch:

```bash
python UNet/f1_experiment.py resume --data-dir UNet/data --run-dir runs/f1_seed42 --device cuda --workers 8
```

Resume restores model, optimizer, EMA, learning-rate position and RNG states.
The unfinished epoch is repeated. Sampler and worker seeds are separate to
preserve epoch shuffle order across persistent-worker restarts. Resume keeps
the original training configuration, epoch budget and batch size; it is not a
way to extend the schedule or change model width. Exact numerical identity
across different hardware/PyTorch versions is not guaranteed. `latest.pt`
contains recovery state; evaluate always uses `best.pt`.

## Training choices

| Setting | Default |
| --- | --- |
| Objective | Per-image soft Dice loss + 0.5 × BCE, both reduced in FP32 |
| Deep supervision | Main loss weight 1; each of two auxiliary heads 0.25; normalize by total weight |
| Optimizer | AdamW, peak learning rate 0.0003, weight decay 0.0001 |
| Schedule | 100 epochs, 5 warmup epochs, cosine decay to 1% of peak |
| Gradient clipping | Global norm 1 |
| Weight averaging | EMA decay up to 0.999 with an initial ramp |
| Checkpoint selection | Highest validation macro F1 across epochs and raw/EMA weights |
| Threshold selection | Validation sweep 0.05, 0.06, …, 0.95 |
| Test threshold | Saved validation choice; never tuned on test labels |

Soft Dice is a differentiable F1 surrogate. BCE adds a per-pixel learning
signal. The extra decoder heads help intermediate features learn useful masks;
they receive full-resolution targets. EMA smooths parameter updates, but the
runner also evaluates raw weights each epoch so EMA is not assumed better.
All images keep equal weight; SNR is used only for reporting. There are no
training-data transformations, sampling changes by SNR, or generator edits.

Macro F1 is the unweighted mean of image F1 scores, not the micro score from
pooled pixel counts. Reports retain both scores, precision, recall, IoU and
per-SNR breakdowns, with the original empty-mask conventions. A histogram
threshold sweep gives exact pixel counts without constructing 91 full masks
for every image. Its results are tested against the original evaluator.

This changes architecture **and** the training/selection recipe relative to
the earlier eight-epoch runs. To separate those effects, the same new runner
can train the original U-Net under the new recipe:

```bash
python UNet/f1_experiment.py train --model unet --data-dir UNet/data --run-dir runs/unet_f1_recipe_seed42 --device cuda --precision bf16 --epochs 100 --batch-size 64 --workers 8 --seed 42
python UNet/f1_experiment.py evaluate --data-dir UNet/data --run-dir runs/unet_f1_recipe_seed42 --device cuda --workers 8
```

## Optional: spend more compute on an ensemble

After seed 42, train two additional independent seeds using the same split:

```bash
for seed in 43 44; do
  python UNet/f1_experiment.py train --data-dir UNet/data --run-dir "runs/f1_seed${seed}" --device cuda --precision bf16 --epochs 100 --batch-size 64 --workers 8 --seed "$seed"
done

python UNet/f1_ensemble.py --run-dirs runs/f1_seed42 runs/f1_seed43 runs/f1_seed44 --output-dir runs/f1_ensemble --data-dir UNet/data --device cuda --batch-size 64 --workers 8
```

The tool averages model probabilities and selects between no flips, time flips,
and time/frequency flips using validation macro F1, together with a fresh
ensemble threshold. Every flipped output is restored to its original
orientation before averaging. The best validation recipe is saved before one
test evaluation. Flips are optional inference transforms, not changes to the
training data, and may not help this distribution. Use `--tta none` to disable
flip selection. An ensemble is not guaranteed to outperform its members.

## Results to send back

- `runs/f1_seed42/evaluation.json`: overall and per-SNR held-out scores.
- `runs/f1_seed42/history.json`: training and validation progress, selected weights and thresholds.
- `runs/f1_seed42/config.json`: exact model and training settings.
- If used, `runs/f1_ensemble/selection.json` and `evaluation.json`.

Each run also saves source snapshots, data hashes and exact split indices.
Evaluation and resume check these to reject changed sources or datasets. Keep
the run directory and source snapshot together when transferring machines.
`--max-train-samples` and `--max-eval-samples` are only diagnostic caps; do not
use them for the reported full-data experiment.

The model combines established components: [UNet++ dense skip connections and
deep supervision](https://arxiv.org/abs/1807.10165), residual convolutional blocks
and [multi-head self-attention](https://arxiv.org/abs/1706.03762). This is an
accuracy experiment, not verified architectural novelty. The current dataset's
fixed frequency placement and binary inputs still limit the conclusions,
especially at very low SNR where a location prior already performs well.

Local verification covers forward/backward/optimizer execution at the full
default width, metric equivalence, raw/EMA checkpoint selection, exact CPU
epoch-boundary recovery, ensemble selection, and the existing regression suite.
The H100/CUDA execution must be checked with the remote smoke command; no full
dataset training was performed while implementing this experiment.
