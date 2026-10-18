"""CPU contract checks; these do not establish segmentation accuracy."""

import io
import unittest

import torch
from torch.nn import functional as F

from UNet.f1_model import F1UNet, _SpatialContext


class F1UNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(1729)

    def test_full_resolution_outputs_for_dataset_odd_and_minimum_sizes(self):
        model = F1UNet(base_channels=8).eval()
        with torch.no_grad():
            for shape in [(2, 1, 50, 100), (1, 1, 17, 31), (1, 1, 8, 8)]:
                with self.subTest(shape=shape):
                    x = torch.rand(shape)
                    main, auxiliary = model(x, return_aux=True)
                    self.assertIsInstance(auxiliary, tuple)
                    self.assertEqual(len(auxiliary), 2)
                    for logits in (main, *auxiliary):
                        self.assertEqual(tuple(logits.shape), shape)
                        self.assertTrue(bool(torch.isfinite(logits).all()))
                    torch.testing.assert_close(model(x), main, rtol=0, atol=0)

    def test_deep_supervision_trains_every_parameter_including_attention(self):
        model = F1UNet(base_channels=8).train()
        x = torch.rand(2, 1, 17, 31)
        target = (torch.rand_like(x) > 0.65).float()
        main, auxiliary = model(x, return_aux=True)
        loss = sum(
            F.binary_cross_entropy_with_logits(logits, target)
            for logits in (main, *auxiliary)
        )
        loss.backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        self.assertGreater(float(model.context.qkv.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.context.position.weight.grad.abs().sum()), 0)
        for head in model.auxiliary_heads:
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)

    def test_attention_carries_information_between_distant_positions(self):
        context = _SpatialContext(32).eval()
        x = torch.randn(1, 32, 6, 12, requires_grad=True)
        context(x)[0, 0, 0, 0].backward()
        # This distant location is outside the positional 3x3 convolution.
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertGreater(float(x.grad[0, :, -1, -1].abs().sum()), 0)

    def test_batch_one_training_at_minimum_shape_and_constant_inputs(self):
        model = F1UNet(base_channels=4).train()
        for value in (0.0, 1.0):
            model.zero_grad(set_to_none=True)
            x = torch.full((1, 1, 8, 8), value)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(x))
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertTrue(bool(torch.isfinite(model.encoder[0].features[0].weight.grad).all()))

    def test_checkpoint_roundtrip_preserves_main_and_auxiliary_predictions(self):
        model = F1UNet(base_channels=8).eval()
        x = torch.rand(1, 1, 17, 31)
        with torch.no_grad():
            expected_main, expected_auxiliary = model(x, return_aux=True)
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = F1UNet(base_channels=8).eval()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        with torch.no_grad():
            actual_main, actual_auxiliary = restored(x, return_aux=True)
        for actual, expected in zip(
            (actual_main, *actual_auxiliary), (expected_main, *expected_auxiliary)
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_shapes_dtypes_and_widths_have_clear_errors(self):
        model = F1UNet(base_channels=8)
        for shape in [(1, 17, 31), (0, 1, 17, 31), (1, 2, 17, 31), (1, 1, 7, 31)]:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    model(torch.zeros(shape))
        with self.assertRaises(TypeError):
            model(torch.zeros(1, 1, 8, 8, dtype=torch.int64))
        for width in [0, 1, 3]:
            with self.assertRaises(ValueError):
                F1UNet(base_channels=width)
        for width in [True, 8.5]:
            with self.assertRaises(TypeError):
                F1UNet(base_channels=width)


if __name__ == "__main__":
    unittest.main()
