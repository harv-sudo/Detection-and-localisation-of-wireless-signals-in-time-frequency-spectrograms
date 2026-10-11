"""Regression tests for comparable metrics and immutable dataset fingerprints."""
import tempfile
import unittest
from pathlib import Path

import torch

from UNet.experiment import SegmentationMetrics, assess, dataset_manifest, scores


class ExperimentTests(unittest.TestCase):
    def test_macro_and_micro_have_distinct_denominators(self):
        metrics = SegmentationMetrics()
        metrics.add(1, 0, 0)
        metrics.add(0, 0, 9)
        result = metrics.result()
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['macro']['f1'], 0.5)
        self.assertAlmostEqual(result['micro']['f1'], 2 / 11)
        self.assertAlmostEqual(result['micro']['iou'], 0.1)

    def test_empty_target_conventions_do_not_reward_false_detections(self):
        self.assertEqual(scores(0, 0, 0), dict.fromkeys(('f1', 'iou', 'precision', 'recall'), 1.0))
        self.assertEqual(scores(0, 2, 0)['f1'], 0)
        self.assertEqual(scores(0, 0, 2)['f1'], 0)
        with self.assertRaises(ValueError):
            SegmentationMetrics().result()

    def test_threshold_metrics_preserve_snr_groups(self):
        logits = torch.tensor([[[[2.0, -2.0]]], [[[0.0, -2.0]]]])
        targets = torch.tensor([[[[1.0, 0.0]]], [[[1.0, 0.0]]]])
        batches = [(logits, targets, torch.tensor([-20.0, 5.0]))]
        low, high = assess(torch.nn.Identity(), batches, torch.device('cpu'), [0.25, 0.5])
        self.assertEqual(low['overall']['macro']['f1'], 1)
        self.assertEqual(high['overall']['macro']['f1'], 0.5)
        self.assertEqual(high['per_snr']['-20']['macro']['f1'], 1)
        self.assertEqual(high['per_snr']['5']['macro']['f1'], 0)
        self.assertAlmostEqual(high['loss'], low['loss'])

    def test_manifest_detects_same_size_content_changes(self):
        class IndexedFiles:
            def __init__(self, path):
                self.files = [path]

            def __len__(self):
                return 4

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'chunk.mat'
            path.write_bytes(b'aaaa')
            dataset = IndexedFiles(path)
            initial = dataset_manifest(dataset)
            self.assertEqual(initial, dataset_manifest(dataset))
            path.write_bytes(b'bbbb')
            modified = dataset_manifest(dataset)
            self.assertEqual(initial['files'][0]['bytes'], modified['files'][0]['bytes'])
            self.assertNotEqual(initial['sha256'], modified['sha256'])


if __name__ == '__main__':
    unittest.main()
