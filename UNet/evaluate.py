import time

import torch
from torch.utils.data import DataLoader, random_split

from dataset import UNetSignalDataset
from model import UNet


# =========================================================
# Configuration
# =========================================================

DATA_DIR = "data"
CHECKPOINT_PATH = "checkpoints/best_model.pt"

BATCH_SIZE = 128
NUM_WORKERS = 16

TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
SEED = 42

THRESHOLD = 0.5


# =========================================================
# Metrics
# =========================================================


def calculate_metrics(preds, targets):
    """
    Calculate TP, FP, FN, intersection and union.

    preds and targets are binary tensors.
    """

    preds = preds.bool()
    targets = targets.bool()

    tp = torch.logical_and(preds, targets).sum().item()

    fp = torch.logical_and(preds, ~targets).sum().item()

    fn = torch.logical_and(~preds, targets).sum().item()

    intersection = torch.logical_and(preds, targets).sum().item()

    union = torch.logical_or(preds, targets).sum().item()

    return tp, fp, fn, intersection, union


def calculate_f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn

    if denominator == 0:
        return 1.0

    return (2 * tp) / denominator


def calculate_iou(intersection, union):
    if union == 0:
        return 1.0

    return intersection / union


# =========================================================
# Evaluation
# =========================================================


def evaluate():

    # -----------------------------------------------------
    # Device
    #
    # Works with:
    # NVIDIA CUDA
    # AMD ROCm
    # CPU fallback
    # -----------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}", flush=True)

    if torch.cuda.is_available():
        print(f"Accelerator: {torch.cuda.get_device_name(0)}", flush=True)

        if torch.version.hip is not None:
            print(f"Backend: AMD ROCm (HIP {torch.version.hip})", flush=True)

        else:
            print(f"Backend: NVIDIA CUDA ({torch.version.cuda})", flush=True)

    else:
        print("Backend: CPU", flush=True)

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    print("Loading dataset...", flush=True)

    dataset = UNetSignalDataset(DATA_DIR)

    n_total = len(dataset)

    n_train = int(TRAIN_SPLIT * n_total)

    n_val = int(VAL_SPLIT * n_total)

    n_test = n_total - n_train - n_val

    _, _, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    print(f"Test samples: {len(test_ds)}", flush=True)

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = UNet().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print(
        f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}", flush=True
    )

    # -----------------------------------------------------
    # Metric accumulators
    # -----------------------------------------------------

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_intersection = 0
    total_union = 0

    total_inference_time = 0.0
    total_samples = 0

    # -----------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------

    print("Evaluating...", flush=True)

    with torch.no_grad():
        for batch_idx, (x, y, _) in enumerate(test_loader, start=1):
            x = x.to(device, non_blocking=True)

            y = y.to(device, non_blocking=True)

            # ---------------------------------------------
            # Synchronise accelerator before timing
            # ---------------------------------------------

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            logits = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            # ---------------------------------------------
            # Convert logits -> probabilities -> mask
            # ---------------------------------------------

            probs = torch.sigmoid(logits)

            preds = probs > THRESHOLD

            targets = y > 0.5

            # ---------------------------------------------
            # Metrics
            # ---------------------------------------------

            (tp, fp, fn, intersection, union) = calculate_metrics(preds, targets)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            total_intersection += intersection

            total_union += union

            batch_size = x.size(0)

            total_samples += batch_size

            total_inference_time += end - start

            if batch_idx % 500 == 0:
                print(f"Processed {batch_idx}/{len(test_loader)} batches", flush=True)

    # =====================================================
    # Final metrics
    # =====================================================

    mean_f1 = calculate_f1(total_tp, total_fp, total_fn)

    mean_iou = calculate_iou(total_intersection, total_union)

    mean_inference_time = total_inference_time / total_samples * 1000

    # -----------------------------------------------------
    # Precision / Recall
    # -----------------------------------------------------

    precision_denominator = total_tp + total_fp

    recall_denominator = total_tp + total_fn

    precision = total_tp / precision_denominator if precision_denominator > 0 else 1.0

    recall = total_tp / recall_denominator if recall_denominator > 0 else 1.0

    # =====================================================
    # Results
    # =====================================================

    print("", flush=True)

    print("U-Net Evaluation Results", flush=True)

    print("------------------------", flush=True)

    print(f"Samples: {total_samples}", flush=True)

    print(f"F1 / Dice: {mean_f1:.4f}", flush=True)

    print(f"IoU:       {mean_iou:.4f}", flush=True)

    print(f"Precision: {precision:.4f}", flush=True)

    print(f"Recall:    {recall:.4f}", flush=True)

    print(f"Mean inference time/sample: {mean_inference_time:.4f} ms", flush=True)


if __name__ == "__main__":
    evaluate()
