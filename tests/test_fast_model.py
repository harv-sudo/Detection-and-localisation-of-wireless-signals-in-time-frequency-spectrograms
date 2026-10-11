"""Deployment equivalence is essential: speed must not silently alter logits."""
import io
import unittest

import torch
from torch import nn

from UNet.fast_model import FastUNet, RepConv, fuse_unet_batchnorm
from UNet.fast_experiment import model_support
from UNet import experiment
from UNet.model import UNet
from UNet.inference import prepare_cpu, replace_nonoverlapping_upsampling


class FastModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_merging_learned_branch_statistics_preserves_border_outputs(self):
        for channels in ((1, 8), (8, 8)):
            torch.manual_seed(42)
            layer = RepConv(*channels)
            for norm in layer.modules():
                if isinstance(norm, nn.BatchNorm2d):
                    norm.running_mean.uniform_(-1, 1)
                    norm.running_var.uniform_(0.2, 2)
                    with torch.no_grad():
                        norm.weight.uniform_(-2, 2)
                        norm.bias.uniform_(-1, 1)
            layer.eval()
            x = torch.randn(2, channels[0], 13, 17)
            expected = layer(x)
            layer.merge()
            torch.testing.assert_close(layer(x), expected, rtol=2e-5, atol=2e-5)

    def test_training_gradients_and_odd_shapes(self):
        model = FastUNet()
        x = torch.rand(2, 1, 51, 101)
        logits = model(x)
        self.assertEqual(logits.shape, x.shape)
        experiment.bce_dice_loss(logits, (x > 0.5).float()).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_conversion_is_copy_and_deployed_checkpoint_roundtrips(self):
        model = FastUNet().eval()
        deployed = model.to_deploy()
        self.assertTrue(any(isinstance(m, nn.BatchNorm2d) for m in model.modules()))
        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in deployed.modules()))
        for shape in ((1, 1, 8, 8), (2, 1, 50, 100), (1, 1, 51, 101)):
            x = torch.rand(shape)
            torch.testing.assert_close(deployed(x), model(x), rtol=2e-5, atol=2e-5)
        buffer = io.BytesIO()
        torch.save(deployed.state_dict(), buffer)
        buffer.seek(0)
        restored = FastUNet(deploy=True).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(restored(x), deployed(x), rtol=0, atol=0)
        torch.testing.assert_close(deployed.to_deploy()(x), deployed(x), rtol=0, atol=0)

    def test_reference_folding_preserves_logits(self):
        model = UNet().eval()
        folded = fuse_unet_batchnorm(model)
        x = torch.rand(2, 1, 50, 100)
        torch.testing.assert_close(folded(x), model(x), rtol=2e-5, atol=2e-5)
        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in folded.modules()))

    def test_training_mode_conversion_rejected(self):
        with self.assertRaises(ValueError):
            FastUNet().to_deploy()
        with self.assertRaises(ValueError):
            fuse_unet_batchnorm(UNet())

    def test_cpu_graph_preserves_values_for_variable_batches_and_guards_shapes(self):
        model = FastUNet().eval().to_deploy()
        prepared = prepare_cpu(model, torch.rand(1, 1, 50, 100))
        for batch in (1, 3):
            x = torch.rand(batch, 1, 50, 100)
            torch.testing.assert_close(prepared(x), model(x), rtol=1e-4, atol=1e-4)
        with self.assertRaises(ValueError):
            prepared(torch.rand(1, 1, 51, 100))
        with self.assertRaises(ValueError):
            prepared.train()

    def test_subpixel_weight_permutation_matches_nonoverlapping_transpose(self):
        for bias in (True, False):
            original = nn.Sequential(nn.ConvTranspose2d(5, 3, 2, stride=2, bias=bias)).eval()
            converted = replace_nonoverlapping_upsampling(original)
            for spatial in ((6, 12), (7, 13)):
                x = torch.randn(2, 5, *spatial)
                torch.testing.assert_close(converted(x), original(x), rtol=2e-5, atol=2e-5)
        overlapping = nn.Sequential(nn.ConvTranspose2d(5, 3, 3, stride=2)).eval()
        self.assertIsInstance(replace_nonoverlapping_upsampling(overlapping)[0], nn.ConvTranspose2d)

    def test_runner_restores_baseline_hooks_even_on_failure(self):
        factory, sources = experiment.make_model, experiment.source_fingerprints
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with model_support():
                self.assertIsNot(experiment.make_model, factory)
                raise RuntimeError("sentinel")
        self.assertIs(experiment.make_model, factory)
        self.assertIs(experiment.source_fingerprints, sources)


if __name__ == "__main__":
    unittest.main()
