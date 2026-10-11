from collections import defaultdict
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset_large import LargeSignalAreaDataset
from model import DLANNBaseline


DATA_DIR = "data/raw_large"
CHECKPOINT_PATH = "checkpoints/best_model.pt"

BATCH_SIZE = 25
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
SEED = 42

HEIGHT = 50
WIDTH = 100
MAX_BOXES = 20


def boxes_to_mask(boxes):
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue

        x1 = int(round(x * WIDTH))
        y1 = int(round(y * HEIGHT))
        x2 = int(round((x + w) * WIDTH))
        y2 = int(round((y + h) * HEIGHT))

        x1 = max(0, min(WIDTH, x1))
        x2 = max(0, min(WIDTH, x2))
        y1 = max(0, min(HEIGHT, y1))
        y2 = max(0, min(HEIGHT, y2))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1

    return mask


def f1_score(pred_mask, true_mask):
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()

    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else (2 * tp) / denom


def iou_score(pred_mask, true_mask):
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()

    return 1.0 if union == 0 else inter / union


def mse_score(pred_boxes, true_boxes):
    return np.mean((pred_boxes - true_boxes) ** 2)


def evaluate_by_snr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    dataset = LargeSignalAreaDataset(DATA_DIR)

    n_total = len(dataset)
    n_train = int(TRAIN_SPLIT * n_total)
    n_val = int(VAL_SPLIT * n_total)
    n_test = n_total - n_train - n_val

    _, _, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = DLANNBaseline().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = defaultdict(lambda: {"f1": [], "iou": [], "mse": [], "time": []})

    with torch.no_grad():
        for x, y, snr_batch in test_loader:
            x = x.to(device, non_blocking=True)

            start = time.perf_counter()
            preds = model(x)
            end = time.perf_counter()

            per_sample_time = (end - start) / x.size(0)

            preds = preds.cpu().numpy().reshape(-1, MAX_BOXES, 4)
            targets = y.numpy().reshape(-1, MAX_BOXES, 4)
            snr_batch = snr_batch.numpy()

            for pred_boxes, true_boxes, snr_value in zip(preds, targets, snr_batch):
                snr = int(round(float(snr_value)))

                pred_mask = boxes_to_mask(pred_boxes)
                true_mask = boxes_to_mask(true_boxes)

                results[snr]["f1"].append(f1_score(pred_mask, true_mask))
                results[snr]["iou"].append(iou_score(pred_mask, true_mask))
                results[snr]["mse"].append(mse_score(pred_boxes, true_boxes))
                results[snr]["time"].append(per_sample_time)

    print("SNR-wise evaluation", flush=True)
    print("-------------------", flush=True)
    print("SNR_dB\tF1\tIoU\tMSE\tTime_ms", flush=True)

    for snr in sorted(results.keys()):
        f1 = np.mean(results[snr]["f1"])
        iou = np.mean(results[snr]["iou"])
        mse = np.mean(results[snr]["mse"])
        t = np.mean(results[snr]["time"]) * 1000

        print(f"{snr:>+3d}\t{f1:.4f}\t{iou:.4f}\t{mse:.6f}\t{t:.4f}", flush=True)


if __name__ == "__main__":
    evaluate_by_snr()
