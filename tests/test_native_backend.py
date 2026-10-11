"""Verify that the ROCm workaround changes only BatchNorm dispatch."""

import unittest
from unittest.mock import patch

import torch

from UNet.native_backend import enable_native_batchnorm


class NativeBatchNormTests(unittest.TestCase):
    def test_dispatch_flag_is_scoped_and_restored_on_error(self):
        initial = torch.backends.cudnn.enabled
        try:
            torch.backends.cudnn.enabled = True
            for raise_error in (False, True):

                def original(*args, **kwargs):
                    self.assertFalse(torch.backends.cudnn.enabled)
                    if raise_error:
                        raise RuntimeError("simulated operation failure")
                    return "result"

                with patch("torch.nn.functional.batch_norm", original):
                    enable_native_batchnorm()
                    if raise_error:
                        with self.assertRaisesRegex(RuntimeError, "simulated"):
                            torch.nn.functional.batch_norm()
                    else:
                        self.assertEqual(torch.nn.functional.batch_norm(), "result")
                    self.assertTrue(torch.backends.cudnn.enabled)
        finally:
            torch.backends.cudnn.enabled = initial

    def test_native_dispatch_preserves_batchnorm_statistics_and_gradients(self):
        torch.manual_seed(9)
        reference = torch.nn.BatchNorm2d(3)
        candidate = torch.nn.BatchNorm2d(3)
        candidate.load_state_dict(reference.state_dict())
        inputs = torch.randn(2, 3, 7, 9)
        expected = reference(inputs)
        expected.square().mean().backward()
        original = torch.nn.functional.batch_norm
        with patch("torch.nn.functional.batch_norm", original):
            enable_native_batchnorm()
            actual = candidate(inputs)
            actual.square().mean().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(candidate.running_mean, reference.running_mean)
        torch.testing.assert_close(candidate.running_var, reference.running_var)
        torch.testing.assert_close(candidate.weight.grad, reference.weight.grad)
        self.assertEqual(candidate.num_batches_tracked, reference.num_batches_tracked)


if __name__ == "__main__":
    unittest.main()
