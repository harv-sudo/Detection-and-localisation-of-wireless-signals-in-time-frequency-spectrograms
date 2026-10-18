"""Check probability averaging, spatial alignment and validation-only selection."""

import copy
import unittest

import torch
from torch import nn

from UNet.f1_ensemble import ProbabilityEnsemble, select_recipe, validate_compatible_runs


class ConstantLogits(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, inputs):
        return torch.full_like(inputs, self.value)


class PositionSensitiveModel(nn.Module):
    def forward(self, inputs):
        rows = torch.arange(inputs.shape[-2], device=inputs.device).view(1, 1, -1, 1)
        columns = torch.arange(inputs.shape[-1], device=inputs.device).view(1, 1, 1, -1)
        return inputs + rows / 3 + columns / 7


class EnsembleTests(unittest.TestCase):
    def test_average_probabilities_instead_of_logits(self):
        inputs = torch.zeros(2, 1, 3, 5)
        ensemble = ProbabilityEnsemble([ConstantLogits(0), ConstantLogits(4)])
        expected = (torch.sigmoid(torch.tensor(0.0)) + torch.sigmoid(torch.tensor(4.0))) / 2
        torch.testing.assert_close(ensemble(inputs).sigmoid(), torch.full_like(inputs, expected))
        self.assertNotAlmostEqual(expected.item(), torch.sigmoid(torch.tensor(2.0)).item())

    def test_each_flip_is_undone_before_averaging(self):
        inputs = torch.linspace(-2, 2, 15).reshape(1, 1, 3, 5)
        model = PositionSensitiveModel()
        for recipe, dimensions in (
            ("time", ((), (-2,))),
            ("both", ((), (-2,), (-1,), (-2, -1))),
        ):
            expected = sum(
                model(inputs.flip(axes)).sigmoid().flip(axes)
                if axes else model(inputs).sigmoid()
                for axes in dimensions
            ) / len(dimensions)
            actual = ProbabilityEnsemble([model], recipe)(inputs).sigmoid()
            torch.testing.assert_close(actual, expected)
        # An equivariant model retains exactly the same spatial prediction.
        torch.testing.assert_close(
            ProbabilityEnsemble([nn.Identity()], "both")(inputs).sigmoid(), inputs.sigmoid()
        )

    def test_saturated_predictions_produce_finite_logits(self):
        inputs = torch.zeros(1, 1, 2, 3)
        for value in (-1000, 1000):
            logits = ProbabilityEnsemble([ConstantLogits(value)])(inputs)
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(((logits.sigmoid() > 0.5) == (value > 0)).all())

    def test_validation_selects_recipe_and_threshold_without_test_data(self):
        validation_loader = object()
        calls = []

        def assess(model, loader, device, thresholds):
            self.assertIs(loader, validation_loader)
            calls.append(model.tta)
            return [
                {
                    "threshold": threshold,
                    "overall": {"macro": {"f1": {"none": 0.7, "time": 0.9, "both": 0.8}[model.tta] - abs(threshold - 0.4)}},
                }
                for threshold in thresholds
            ]

        ensemble, selection = select_recipe(
            [nn.Identity()], validation_loader, torch.device("cpu"), [0.3, 0.4, 0.5], assess_fn=assess
        )
        self.assertEqual(calls, ["none", "time", "both"])
        self.assertEqual(ensemble.tta, "time")
        self.assertEqual(selection["threshold"], 0.4)
        self.assertEqual(len(selection["candidates"]), 3)

    def test_ties_prefer_threshold_half_and_fewer_flips(self):
        def assess(model, loader, device, thresholds):
            return [{"threshold": value, "overall": {"macro": {"f1": 1.0}}} for value in thresholds]

        _, selected = select_recipe(
            [nn.Identity()], [], torch.device("cpu"), [0.2, 0.5, 0.8], assess_fn=assess
        )
        self.assertEqual(selected["tta"], "none")
        self.assertEqual(selected["threshold"], 0.5)

    def test_identical_data_and_effective_splits_are_required(self):
        split = {"train": [0, 1], "val": [2, 3], "test": [4, 5], "used_train": [0], "used_val": [2]}
        manifest = {"sha256": "same"}
        result = validate_compatible_runs([split, copy.deepcopy(split)], [manifest, manifest])
        self.assertTrue(result["training_was_capped"])
        self.assertTrue(result["validation_was_capped"])
        for key in split:
            other = copy.deepcopy(split)
            other[key] = list(reversed(other[key])) if len(other[key]) > 1 else [9]
            with self.assertRaisesRegex(ValueError, key):
                validate_compatible_runs([split, other], [manifest, manifest])
        with self.assertRaisesRegex(ValueError, "manifests"):
            validate_compatible_runs([split, split], [manifest, {"sha256": "changed"}])


if __name__ == "__main__":
    unittest.main()
