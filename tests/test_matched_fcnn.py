"""FCNN split integrity, coordinate conventions and train/evaluate round trips."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from baseline_recreation import matched_experiment as m
from UNet.dataset import UNetSignalDataset


class MatchedFCNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_matlab_boxes_padding_and_boundaries(self):
        boxes = torch.zeros(20, 4)
        boxes[0] = torch.tensor([1, 1, 100, 50])
        normalized = m.normalize_boxes(boxes)
        self.assertTrue(torch.equal(normalized[4:], torch.zeros(76)))
        self.assertEqual(int(m.boxes_to_masks(normalized[None]).sum()), 5000)
        boxes[0] = torch.tensor([100, 50, 1, 1])
        mask = m.boxes_to_masks(m.normalize_boxes(boxes)[None])[0]
        self.assertEqual(int(mask.sum()), 1)
        self.assertTrue(mask[49, 99])
        boxes[1, 0] = 1
        with self.assertRaisesRegex(ValueError, "padding"):
            m.normalize_boxes(boxes)

    def test_rounding_matches_historical_numpy_decoder(self):
        boxes = np.zeros((3, 20, 4), dtype=np.float32)
        boxes[0, :4] = [[.005, .01, .015, .03], [.985, .97, .1, .1],
                        [-.05, -.02, .1, .1], [.01, .02, 0, .2]]
        boxes[1] = np.random.default_rng(7).uniform(-.1, 1.1, (20, 4))
        expected = np.zeros((3, 50, 100), dtype=bool)
        for sample, row in enumerate(boxes):
            for x, y, w, h in row:
                if w <= 0 or h <= 0:
                    continue
                x1 = max(0, min(100, int(round(x * 100))))
                y1 = max(0, min(50, int(round(y * 50))))
                x2 = max(0, min(100, int(round((x + w) * 100))))
                y2 = max(0, min(50, int(round((y + h) * 50))))
                if x2 > x1 and y2 > y1:
                    expected[sample, y1:y2, x1:x2] = True
        np.testing.assert_array_equal(m.boxes_to_masks(torch.from_numpy(boxes)).numpy(), expected)
        boxes[0, 0, 0] = np.nan
        with self.assertRaises(FloatingPointError):
            m.boxes_to_masks(torch.from_numpy(boxes))

    def test_saved_clean_mask_is_the_scoring_target(self):
        predictions = torch.zeros(2, 80)
        predictions[0, :4] = torch.tensor([0, 0, .01, .02])
        truth = torch.zeros(2, 50, 100, dtype=torch.bool)
        truth[0, 0, 0] = True
        truth[1, :3, :3] = True
        # Box training labels deliberately disagree with clean truth. Scoring
        # must use clean_imgs; MSE alone uses coordinate labels.
        batch = (predictions, torch.zeros_like(predictions), truth, torch.tensor([-20., 5.]))
        result = m.assess(torch.nn.Identity(), [batch], torch.device("cpu"))
        self.assertEqual(result["overall"]["macro"]["f1"], .5)
        self.assertAlmostEqual(result["overall"]["micro"]["f1"], 2/11)
        self.assertEqual(result["per_snr"]["-20"]["macro"]["f1"], 1)
        self.assertEqual(result["per_snr"]["5"]["macro"]["f1"], 0)

    def test_split_partition_and_capped_reference_guards(self):
        good = {"train": [2, 0], "val": [1], "test": [3]}
        m.validate_splits(good, 4)
        for bad in ({**good, "test": [0]}, {**good, "test": []},
                    {**good, "used_train": [2]}):
            with self.assertRaises(ValueError):
                m.validate_splits(bad, 4)

    def fixture(self, root):
        data, reference = root / "data", root / "reference"
        data.mkdir()
        reference.mkdir()
        clean = np.zeros((100, 50, 10), np.float32)
        clean[3:8, 5:12, :] = 1
        labels = np.zeros((4, 20, 10), np.float32)
        labels[:, 0, :] = np.array([4, 6, 5, 7])[:, None]
        with h5py.File(data / "snr_00.mat", "w") as f:
            f["noisy_imgs"] = clean
            f["clean_imgs"] = clean
            f["labels"] = labels
            f["SNR_dB_all"] = np.zeros((1, 10), np.float32)
        config = dict(data_dir=str(data), epochs=8, batch_size=64, lr=.001, seed=42)
        (reference / "config.json").write_text(json.dumps(config))
        dataset = m.MatchedDataset(data)
        manifest = m.common.dataset_manifest(dataset)
        (reference / "manifest.json").write_text(json.dumps(manifest))
        splits = {"train": [0, 1, 2, 3, 4, 5], "val": [6, 7], "test": [8, 9]}
        torch.save(splits, reference / "splits.pt")
        torch.save(dict(manifest_sha256=manifest["sha256"],
                        splits_sha256=m.common.file_sha256(reference / "splits.pt")), reference / "best.pt")
        return data, reference, dataset

    def test_loader_matches_unet_orientation(self):
        with tempfile.TemporaryDirectory() as folder:
            data, _, dataset = self.fixture(Path(folder))
            with contextlib.redirect_stdout(io.StringIO()):
                unet = UNetSignalDataset(data)
            x, boxes, clean, snr = dataset[0]
            ux, uc, us = unet[0]
            torch.testing.assert_close(x, ux.flatten())
            self.assertTrue(torch.equal(clean, uc.squeeze(0).bool()))
            torch.testing.assert_close(snr, us)
            self.assertTrue(torch.equal(m.boxes_to_masks(boxes[None])[0], clean))

    def test_training_evaluation_roundtrip_and_tamper_guards(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            data, reference, _ = self.fixture(root)
            run = root / "run"
            args = m.parser().parse_args(["train", "--reference-run", str(reference), "--run-dir", str(run),
                                         "--epochs", "1", "--workers", "0", "--device", "cpu",
                                         "--batch-size", "2", "--max-train-samples", "4"])
            m.train(args)
            checkpoint = torch.load(run / "best.pt", weights_only=True)
            self.assertFalse(checkpoint["config"]["matches_reference_budget"])
            self.assertFalse((run / "evaluation.json").exists())
            self.assertEqual(torch.load(run / "splits.pt", weights_only=True)["test"], [8, 9])
            with self.assertRaises(FileExistsError):
                m.train(args)
            evaluation = m.parser().parse_args(["evaluate", "--run-dir", str(run), "--workers", "0", "--device", "cpu"])
            m.evaluate(evaluation)
            result = json.loads((run / "evaluation.json").read_text())
            self.assertEqual(result["test_indices"], [8, 9])
            self.assertEqual(result["overall"]["samples"], 2)
            self.assertEqual(result["epoch"], checkpoint["epoch"])
            with patch.object(m, "fingerprints", return_value={}):
                with self.assertRaisesRegex(ValueError, "Source changed"):
                    m.evaluate(evaluation)
            split_bytes = (run / "splits.pt").read_bytes()
            torch.save({"train": [0], "val": [1], "test": [2]}, run / "splits.pt")
            with self.assertRaisesRegex(ValueError, "split changed"):
                m.evaluate(evaluation)
            (run / "splits.pt").write_bytes(split_bytes)
            with h5py.File(data / "snr_00.mat", "r+") as f:
                f["noisy_imgs"][0, 0, 0] = 1
            with self.assertRaisesRegex(ValueError, "Dataset changed"):
                m.evaluate(evaluation)
            with self.assertRaisesRegex(ValueError, "Dataset differs"):
                m.load_reference(reference, data)


if __name__ == "__main__":
    unittest.main()
