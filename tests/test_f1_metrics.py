"""The faster threshold sweep must retain every frozen metric convention."""

import unittest

import torch

from UNet.experiment import assess as reference_assess
from UNet.f1_metrics import assess, threshold_counts


class F1MetricTests(unittest.TestCase):
    def assert_reports_equal(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(list(actual), list(expected))
            for key in expected:
                self.assert_reports_equal(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_reports_equal(left, right)
        elif isinstance(expected, float):
            self.assertAlmostEqual(actual, expected, places=14)
        else:
            self.assertEqual(actual, expected)

    def test_histogram_counts_are_exact_at_boundaries_and_preserve_order(self):
        probabilities = torch.tensor([
            [[[0.0, 0.25, 0.5, 0.75, 1.0]]],
            [[[0.0, 0.25, 0.5, 0.75, 1.0]]],
        ])
        targets = torch.tensor([[[[0.0, 0.5, 1.0, 0.0, 1.0]]], [[[0.0] * 5]]])
        thresholds = [0.75, 0.0, 0.5, 1.0, 0.25]
        actual = threshold_counts(probabilities, targets, thresholds)
        expected = []
        truth = targets > 0.5
        for threshold in thresholds:
            prediction = probabilities > threshold
            expected.append(torch.stack([
                (prediction & truth).flatten(1).sum(1),
                (prediction & ~truth).flatten(1).sum(1),
                (~prediction & truth).flatten(1).sum(1),
            ], dim=-1))
        self.assertEqual(actual.dtype, torch.int64)
        self.assertTrue(torch.equal(actual, torch.stack(expected)))

    def test_reports_match_reference_for_empty_masks_exact_thresholds_and_snr(self):
        logits = torch.tensor([
            [[[-100.0, -100.0, -100.0, -100.0]]],
            [[[100.0, 0.0, -2.0, 2.0]]],
            [[[-2.0, 0.0, 2.0, -100.0]]],
            [[[100.0, 100.0, 100.0, 100.0]]],
        ])
        targets = torch.tensor([
            [[[0.0, 0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0, 0.0]]],
            [[[1.0, 1.0, 0.0, 1.0]]],
            [[[1.0, 1.0, 1.0, 1.0]]],
        ])
        snrs = torch.tensor([5.0, -20.0, -20.0, -10.0])
        thresholds = [float(torch.tensor(2.0).sigmoid()), 0.5, float(torch.tensor(-2.0).sigmoid())]
        batches = [(logits[:3], targets[:3], snrs[:3]), (logits[3:], targets[3:], snrs[3:])]
        expected = reference_assess(torch.nn.Identity(), batches, "cpu", thresholds)
        actual = assess(torch.nn.Identity(), batches, "cpu", thresholds)
        self.assert_reports_equal(actual, expected)
        self.assertEqual(actual[0]["overall"]["empty_targets"], 2)
        self.assertEqual(actual[0]["per_snr"]["5"]["macro"]["f1"], 1.0)

    def test_random_imbalanced_reports_match_reference_across_batches(self):
        generator = torch.Generator().manual_seed(931)
        logits = torch.randn((29, 1, 7, 11), generator=generator) * 3
        targets = (torch.rand(logits.shape, generator=generator) > 0.97).float()
        targets[::5] = 0
        targets[1::7] = 1
        snrs = torch.tensor([-20, -19, -17, -12, -5, 0, 5] * 5)[:29].float()
        batches = [(logits[i:i + 6], targets[i:i + 6], snrs[i:i + 6]) for i in range(0, 29, 6)]
        for thresholds in ([0.5], [i / 100 for i in range(1, 100)]):
            with self.subTest(thresholds=len(thresholds)):
                actual = assess(torch.nn.Identity(), batches, "cpu", thresholds)
                expected = reference_assess(torch.nn.Identity(), batches, "cpu", thresholds)
                self.assert_reports_equal(actual, expected)

    def test_inference_disables_outer_autocast(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.observed_dtype = None

            def forward(self, inputs):
                result = self.conv(inputs)
                self.observed_dtype = result.dtype
                return result

        model = RecordingModel()
        batches = [(torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 2, 3), torch.tensor([-20.0]))]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assess(model, batches, "cpu", [0.5])
        self.assertEqual(model.observed_dtype, torch.float32)
        self.assertFalse(model.training)

    def test_invalid_thresholds_inputs_and_empty_evaluation_are_rejected(self):
        probabilities = torch.ones(2, 1, 2, 2) / 2
        for thresholds in ([], [float("nan")], [float("inf")], [-0.1], [1.1], [0.5, 0.5], [0.5, 0.5000000001]):
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                threshold_counts(probabilities, probabilities, thresholds)
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                threshold_counts(torch.full_like(probabilities, value), probabilities, [0.5])
        with self.assertRaises(ValueError):
            threshold_counts(probabilities, probabilities[:1], [0.5])
        with self.assertRaises(ValueError):
            threshold_counts(probabilities[:0], probabilities[:0], [0.5])
        with self.assertRaises(ValueError):
            assess(torch.nn.Identity(), [], "cpu", [0.5])
        with self.assertRaises(FloatingPointError):
            assess(torch.nn.Identity(), [(probabilities * float("nan"), probabilities, torch.zeros(2))], "cpu", [0.5])


if __name__ == "__main__":
    unittest.main()
