from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class UNetSignalDataset(Dataset):
    """
    Dataset for training the U-Net segmentation model.

    Each sample consists of:
        input  : noisy spectrogram  -> shape (1, 50, 100)
        target : clean signal mask  -> shape (1, 50, 100)
        snr    : SNR value in dB

    The dataset reads the same chunked MATLAB v7.3 .mat files
    generated for the large baseline experiment.
    """

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        self.files = sorted(self.root_dir.glob("*.mat"))

        if not self.files:
            raise RuntimeError(f"No chunk .mat files found in {self.root_dir}")

        # Maps a global dataset index to:
        # (chunk_file_index, sample_index_within_chunk)
        self.index = []

        print("Indexing dataset...")

        for file_idx, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                if "noisy_imgs" not in f:
                    raise RuntimeError(f"'noisy_imgs' not found in {path}")

                if "clean_imgs" not in f:
                    raise RuntimeError(f"'clean_imgs' not found in {path}")

                # MATLAB v7.3/HDF5 dimensions appear reversed
                # when read using h5py.
                #
                # Expected stored shape:
                # (100, 50, number_of_samples)
                n_samples = f["noisy_imgs"].shape[-1]

            for sample_idx in range(n_samples):
                self.index.append((file_idx, sample_idx))

        print(
            f"Dataset indexed: {len(self.index)} samples "
            f"across {len(self.files)} chunk files."
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        file_idx, sample_idx = self.index[idx]

        path = self.files[file_idx]

        with h5py.File(path, "r") as f:
            # -------------------------------------------------
            # Load noisy spectrogram
            # -------------------------------------------------

            noisy = f["noisy_imgs"][:, :, sample_idx]

            # MATLAB/HDF5 orientation -> PyTorch orientation
            noisy = torch.tensor(noisy, dtype=torch.float32).T

            # Add channel dimension:
            #
            # (50, 100)
            #      ↓
            # (1, 50, 100)
            noisy = noisy.unsqueeze(0)

            # -------------------------------------------------
            # Load clean target image
            # -------------------------------------------------

            clean = f["clean_imgs"][:, :, sample_idx]

            clean = torch.tensor(clean, dtype=torch.float32).T

            clean = clean.unsqueeze(0)

            # Convert target to a binary segmentation mask.
            #
            # 0 = background
            # 1 = signal
            clean = (clean > 0).float()

            # -------------------------------------------------
            # Load SNR
            # -------------------------------------------------

            snr_dataset = f["SNR_dB_all"]

            if snr_dataset.shape[0] == 1:
                snr = float(snr_dataset[0, sample_idx])

            elif snr_dataset.shape[1] == 1:
                snr = float(snr_dataset[sample_idx, 0])

            else:
                snr = float(snr_dataset[sample_idx])

        snr = torch.tensor(snr, dtype=torch.float32)

        return noisy, clean, snr
