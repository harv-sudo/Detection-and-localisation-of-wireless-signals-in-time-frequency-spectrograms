from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import UNetSignalDataset
from model import UNet


# =========================================================
# Configuration
# =========================================================

# Fresh dataset generated locally using MATLAB
DATA_DIR = "data"

CHECKPOINT_DIR = "checkpoints"

# Quick local training configuration
BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 1e-3

TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
SEED = 42

NUM_WORKERS = 4


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Dice Loss
# =========================================================

class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    The model outputs raw logits.
    Sigmoid converts them to probabilities.
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probs.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


# =========================================================
# Combined BCE + Dice Loss
# =========================================================

class BCEDiceLoss(nn.Module):
    """
    Combination of:
        Binary Cross Entropy
        Dice Loss
    """

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return bce_loss + dice_loss


# =========================================================
# Metrics
# =========================================================

def batch_metrics(logits, targets, threshold=0.5):
    """
    Calculate segmentation statistics for a batch.
    """

    probs = torch.sigmoid(logits)

    preds = probs > threshold
    targets = targets > 0.5

    tp = torch.logical_and(
        preds,
        targets
    ).sum().item()

    fp = torch.logical_and(
        preds,
        ~targets
    ).sum().item()

    fn = torch.logical_and(
        ~preds,
        targets
    ).sum().item()

    intersection = torch.logical_and(
        preds,
        targets
    ).sum().item()

    union = torch.logical_or(
        preds,
        targets
    ).sum().item()

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
# Training
# =========================================================

def train():

    set_seed(SEED)

    # -----------------------------------------------------
    # Device selection
    #
    # Works with:
    #   NVIDIA + CUDA
    #   AMD + ROCm
    #   CPU fallback
    #
    # PyTorch ROCm uses the torch.cuda API.
    # -----------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(
        f"Using device: {device}",
        flush=True
    )

    if torch.cuda.is_available():

        accelerator_name = torch.cuda.get_device_name(0)

        print(
            f"Accelerator: {accelerator_name}",
            flush=True
        )

        if torch.version.hip is not None:

            print(
                f"Backend: AMD ROCm "
                f"(HIP {torch.version.hip})",
                flush=True
            )

        else:

            print(
                f"Backend: NVIDIA CUDA "
                f"({torch.version.cuda})",
                flush=True
            )

    else:

        print(
            "Backend: CPU",
            flush=True
        )

    # -----------------------------------------------------
    # Checkpoint directory
    # -----------------------------------------------------

    Path(CHECKPOINT_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    print(
        "Loading local dataset...",
        flush=True
    )

    dataset_start = time.time()

    dataset = UNetSignalDataset(
        DATA_DIR
    )

    dataset_time = (
        time.time()
        - dataset_start
    )

    print(
        f"Dataset loaded: "
        f"{len(dataset)} samples "
        f"in {dataset_time:.2f}s",
        flush=True
    )

    # -----------------------------------------------------
    # Train / validation / test split
    # -----------------------------------------------------

    n_total = len(dataset)

    n_train = int(
        TRAIN_SPLIT * n_total
    )

    n_val = int(
        VAL_SPLIT * n_total
    )

    n_test = (
        n_total
        - n_train
        - n_val
    )

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [
            n_train,
            n_val,
            n_test
        ],
        generator=torch.Generator().manual_seed(
            SEED
        ),
    )

    print(
        f"Train: {len(train_ds)} | "
        f"Val: {len(val_ds)} | "
        f"Test: {len(test_ds)}",
        flush=True
    )

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    use_pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = UNet().to(device)

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}",
        flush=True
    )

    # -----------------------------------------------------
    # Loss and optimiser
    # -----------------------------------------------------

    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    # =====================================================
    # Epoch loop
    # =====================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()

        # =================================================
        # Training phase
        # =================================================

        model.train()

        train_loss = 0.0

        train_tp = 0
        train_fp = 0
        train_fn = 0

        train_intersection = 0
        train_union = 0

        for batch_idx, (
            x,
            y,
            _
        ) in enumerate(
            train_loader,
            start=1
        ):

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * x.size(0)
            )

            (
                tp,
                fp,
                fn,
                intersection,
                union
            ) = batch_metrics(
                logits.detach(),
                y
            )

            train_tp += tp
            train_fp += fp
            train_fn += fn

            train_intersection += (
                intersection
            )

            train_union += union

            # More frequent output for the smaller local run
            if batch_idx % 250 == 0:

                print(
                    f"Epoch {epoch} | "
                    f"Batch "
                    f"{batch_idx}/"
                    f"{len(train_loader)} | "
                    f"Loss: "
                    f"{loss.item():.6f}",
                    flush=True
                )

        # -------------------------------------------------
        # Training metrics
        # -------------------------------------------------

        train_loss /= len(
            train_loader.dataset
        )

        train_f1 = calculate_f1(
            train_tp,
            train_fp,
            train_fn
        )

        train_iou = calculate_iou(
            train_intersection,
            train_union
        )

        # =================================================
        # Validation phase
        # =================================================

        model.eval()

        val_loss = 0.0

        val_tp = 0
        val_fp = 0
        val_fn = 0

        val_intersection = 0
        val_union = 0

        with torch.no_grad():

            for x, y, _ in val_loader:

                x = x.to(
                    device,
                    non_blocking=True
                )

                y = y.to(
                    device,
                    non_blocking=True
                )

                logits = model(x)

                loss = criterion(
                    logits,
                    y
                )

                val_loss += (
                    loss.item()
                    * x.size(0)
                )

                (
                    tp,
                    fp,
                    fn,
                    intersection,
                    union
                ) = batch_metrics(
                    logits,
                    y
                )

                val_tp += tp
                val_fp += fp
                val_fn += fn

                val_intersection += (
                    intersection
                )

                val_union += union

        # -------------------------------------------------
        # Validation metrics
        # -------------------------------------------------

        val_loss /= len(
            val_loader.dataset
        )

        val_f1 = calculate_f1(
            val_tp,
            val_fp,
            val_fn
        )

        val_iou = calculate_iou(
            val_intersection,
            val_union
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        # -------------------------------------------------
        # Epoch summary
        # -------------------------------------------------

        print(
            "",
            flush=True
        )

        print(
            f"Epoch "
            f"[{epoch:03d}/{EPOCHS}]",
            flush=True
        )

        print(
            f"Train Loss: "
            f"{train_loss:.6f} | "
            f"F1: {train_f1:.4f} | "
            f"IoU: {train_iou:.4f}",
            flush=True
        )

        print(
            f"Val   Loss: "
            f"{val_loss:.6f} | "
            f"F1: {val_f1:.4f} | "
            f"IoU: {val_iou:.4f}",
            flush=True
        )

        print(
            f"Epoch time: "
            f"{epoch_time / 60:.2f} min",
            flush=True
        )

        print(
            "",
            flush=True
        )

        # =================================================
        # Save best checkpoint
        # =================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_f1": val_f1,
                "val_iou": val_iou,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "dataset_size": len(dataset),
                "seed": SEED,
            }

            torch.save(
                checkpoint,
                Path(
                    CHECKPOINT_DIR
                ) / "best_model.pt"
            )

            print(
                "Saved new best model.",
                flush=True
            )

    # =====================================================
    # Finished
    # =====================================================

    print(
        f"Training complete. "
        f"Best validation loss: "
        f"{best_val_loss:.6f}",
        flush=True
    )


if __name__ == "__main__":
    train()