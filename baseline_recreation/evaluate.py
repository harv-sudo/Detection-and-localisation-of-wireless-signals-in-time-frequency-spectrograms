from pathlib import Path
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


def boxes_to_mask(boxes, height=HEIGHT, width=WIDTH):
    """
    boxes: array shape (20, 4), normalised [x, y, w, h]
    returns binary mask shape (height, width)
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue

        x1 = int(round(x * width))
        y1 = int(round(y * height))
        x2 = int(round((x + w) * width))
        y2 = int(round((y + h) * height))

        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

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


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DLANNBaseline().to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    f1_scores = []
    iou_scores = []
    inference_times = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            start = time.perf_counter()
            preds = model(x)
            end = time.perf_counter()

            batch_time = (end - start) / x.size(0)
            inference_times.extend([batch_time] * x.size(0))

            preds = preds.cpu().numpy().reshape(-1, MAX_BOXES, 4)
            targets = y.numpy().reshape(-1, MAX_BOXES, 4)

            for pred_boxes, true_boxes in zip(preds, targets):
                pred_mask = boxes_to_mask(pred_boxes)
                true_mask = boxes_to_mask(true_boxes)

                f1_scores.append(f1_score(pred_mask, true_mask))
                iou_scores.append(iou_score(pred_mask, true_mask))

    print("Evaluation results")
    print("------------------")
    print(f"Samples: {len(f1_scores)}")
    print(f"Mean F1: {np.mean(f1_scores):.4f}")
    print(f"Mean IoU: {np.mean(iou_scores):.4f}")
    print(f"Mean inference time/sample: {np.mean(inference_times) * 1000:.4f} ms")


if __name__ == "__main__":
    evaluate()