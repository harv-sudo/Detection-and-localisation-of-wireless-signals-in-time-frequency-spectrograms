from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class LargeSignalAreaDataset(Dataset):
    def __init__(self, root_dir, max_boxes=20):
        self.root_dir = Path(root_dir)
        self.files = sorted(self.root_dir.glob("*.mat"))
        self.max_boxes = max_boxes

        if not self.files:
            raise RuntimeError(f"No chunk .mat files found in {root_dir}")

        self.index = []

        for file_idx, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                n = f["noisy_imgs"].shape[-1]

            for sample_idx in range(n):
                self.index.append((file_idx, sample_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index[idx]
        path = self.files[file_idx]

        with h5py.File(path, "r") as f:
            noisy = f["noisy_imgs"][:, :, sample_idx]
            boxes = f["labels"][:, :, sample_idx]

            snr_dataset = f["SNR_dB_all"]

            if snr_dataset.shape[0] == 1:
                snr = float(snr_dataset[0, sample_idx])
            elif snr_dataset.shape[1] == 1:
                snr = float(snr_dataset[sample_idx, 0])
            else:
                snr = float(snr_dataset[sample_idx])

        noisy = torch.tensor(noisy, dtype=torch.float32).T.reshape(-1)

        boxes = torch.tensor(boxes, dtype=torch.float32).T

        boxes[:, 0] -= 1
        boxes[:, 1] -= 1

        boxes[:, 0] /= 100
        boxes[:, 2] /= 100
        boxes[:, 1] /= 50
        boxes[:, 3] /= 50

        boxes = torch.clamp(boxes, min=0.0, max=1.0)

        return noisy, boxes.reshape(-1), torch.tensor(snr, dtype=torch.float32)