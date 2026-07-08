from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


class SignalAreaDataset(Dataset):
    """
    Loads MATLAB-generated signal area samples.

    Expected .mat keys:
        noisy_img: input binary spectrogram, shape (50, 100)
        boxes: ground-truth boxes, shape (num_boxes, 4), MATLAB 1-indexed
               format: [x, y, w, h]

    Output:
        x: flattened spectrogram, shape (5000,)
        y: flattened box target, shape (80,)
           20 boxes × 4 values, zero-padded
    """

    def __init__(self, root_dir, max_boxes=20, normalise_boxes=True):
        self.root_dir = Path(root_dir)
        self.files = sorted(self.root_dir.glob("*.mat"))
        self.max_boxes = max_boxes
        self.normalise_boxes = normalise_boxes

        if not self.files:
            raise RuntimeError(f"No .mat files found in {self.root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mat = scipy.io.loadmat(self.files[idx])

        noisy_img = mat["noisy_img"].astype(np.float32)
        boxes = mat["boxes"].astype(np.float32)

        height, width = noisy_img.shape

        # Flatten input: 50 x 100 -> 5000
        x = noisy_img.reshape(-1)

        # Make it 0-indexed
        if boxes.size > 0:
            boxes[:, 0] -= 1  # x
            boxes[:, 1] -= 1  # y

        # Keep at most max_boxes
        boxes = boxes[: self.max_boxes]

        # Normalise coordinates to [0, 1]
        if self.normalise_boxes and boxes.size > 0:
            boxes[:, 0] /= width   # x
            boxes[:, 2] /= width   # w
            boxes[:, 1] /= height  # y
            boxes[:, 3] /= height  # h

        # Pad to fixed 20 x 4 output
        target = np.zeros((self.max_boxes, 4), dtype=np.float32)

        if boxes.size > 0:
            target[: len(boxes)] = boxes

        return torch.from_numpy(x), torch.from_numpy(target.reshape(-1))