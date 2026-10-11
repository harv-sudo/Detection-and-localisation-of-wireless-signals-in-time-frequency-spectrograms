from collections import defaultdict
import time

import numpy as np
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


def evaluate_by_snr():

    # -----------------------------------------------------
    # Device selection
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
    # Results container
    #
    # We store raw totals for each SNR.
    # -----------------------------------------------------

    results = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "intersection": 0,
            "union": 0,
            "samples": 0,
            "inference_time": 0.0,
        }
    )

    # -----------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------

    print("Evaluating by SNR...", flush=True)

    with torch.no_grad():
        for batch_idx, (x, y, snr_batch) in enumerate(test_loader, start=1):
            x = x.to(device, non_blocking=True)

            y = y.to(device, non_blocking=True)

            # ---------------------------------------------
            # Inference timing
            # ---------------------------------------------

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            logits = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            batch_inference_time = end - start

            # ---------------------------------------------
            # Prediction masks
            # ---------------------------------------------

            probs = torch.sigmoid(logits)

            preds = probs > THRESHOLD

            targets = y > 0.5

            # Bring SNR values to CPU
            snr_batch = snr_batch.detach().cpu().numpy()

            # ---------------------------------------------
            # Process each SNR group separately
            # ---------------------------------------------

            unique_snrs = np.unique(snr_batch)

            for snr_value in unique_snrs:
                snr = int(round(float(snr_value)))

                mask_np = snr_batch == snr_value

                indices = torch.from_numpy(np.where(mask_np)[0]).to(
                    device=device, dtype=torch.long
                )

                preds_snr = torch.index_select(preds, dim=0, index=indices)

                targets_snr = torch.index_select(targets, dim=0, index=indices)

                (tp, fp, fn, intersection, union) = calculate_metrics(
                    preds_snr, targets_snr
                )

                sample_count = len(indices)

                results[snr]["tp"] += tp
                results[snr]["fp"] += fp
                results[snr]["fn"] += fn

                results[snr]["intersection"] += intersection

                results[snr]["union"] += union

                results[snr]["samples"] += sample_count

                # Approximate inference time contribution
                # based on this group's share of the batch.
                results[snr]["inference_time"] += batch_inference_time * (
                    sample_count / x.size(0)
                )

            if batch_idx % 500 == 0:
                print(f"Processed {batch_idx}/{len(test_loader)} batches", flush=True)

    # =====================================================
    # Print results
    # =====================================================

    print("", flush=True)

    print("U-Net SNR-wise Evaluation", flush=True)

    print("-------------------------", flush=True)

    print("SNR_dB\tF1\tIoU\tPrecision\tRecall\tTime_ms", flush=True)

    for snr in sorted(results.keys()):
        r = results[snr]

        tp = r["tp"]
        fp = r["fp"]
        fn = r["fn"]

        intersection = r["intersection"]

        union = r["union"]

        samples = r["samples"]

        f1 = calculate_f1(tp, fp, fn)

        iou = calculate_iou(intersection, union)

        precision_denom = tp + fp

        recall_denom = tp + fn

        precision = tp / precision_denom if precision_denom > 0 else 1.0

        recall = tp / recall_denom if recall_denom > 0 else 1.0

        mean_time_ms = r["inference_time"] / samples * 1000 if samples > 0 else 0.0

        print(
            f"{snr:>+3d}\t"
            f"{f1:.4f}\t"
            f"{iou:.4f}\t"
            f"{precision:.4f}\t"
            f"{recall:.4f}\t"
            f"{mean_time_ms:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    evaluate_by_snr()
