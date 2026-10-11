from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset_large import LargeSignalAreaDataset
from model import DLANNBaseline


DATA_DIR = "data/raw_large"
CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 25
EPOCHS = 1
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
SEED = 42
NUM_WORKERS = 8


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

    print("Loading dataset...", flush=True)
    start = time.time()
    dataset = LargeSignalAreaDataset(DATA_DIR)
    print(
        f"Dataset loaded: {len(dataset)} samples in {time.time() - start:.2f}s",
        flush=True,
    )

    n_total = len(dataset)
    n_train = int(TRAIN_SPLIT * n_total)
    n_val = int(VAL_SPLIT * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    print(
        f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )

    model = DLANNBaseline().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            preds = model(x)
            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

            if batch_idx % 1000 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.6f}",
                    flush=True,
                )

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                preds = model(x)
                loss = criterion(preds, y)

                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Time: {epoch_time / 60:.2f} min",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                Path(CHECKPOINT_DIR) / "best_model.pt",
            )
            print("Saved new best model.", flush=True)

    print(f"Training complete. Best val loss: {best_val_loss:.6f}", flush=True)


if __name__ == "__main__":
    train()
