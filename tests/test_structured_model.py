"""Behavioural checks for the evidence-aware segmentation model.

Run from the repository root with:
    python -m unittest discover -s tests -p test_structured_model.py -v

These tests exercise inference/training contracts on CPU; they do not make
claims about detection accuracy or invariance to shifted inputs.
"""

import io
import unittest

import torch
import torch.nn.functional as F

from UNet.structured_model import EvidenceAxialNet


class EvidenceAxialNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(1729)

    def make_model(self, **kwargs):
        config = {"base_channels": 8, "refinement_steps": 2}
        config.update(kwargs)
        return EvidenceAxialNet(**config)

    def assert_finite(self, tensor):
        self.assertTrue(bool(torch.isfinite(tensor).all()))

    def test_odd_and_dataset_shapes_preserved_with_interpretable_auxiliary_maps(self):
        model = self.make_model().eval()
        with torch.no_grad():
            for shape in [(2, 1, 17, 31), (1, 1, 50, 100)]:
                with self.subTest(shape=shape):
                    x = torch.rand(shape)
                    logits, aux = model(x, return_aux=True)
                    self.assertEqual(tuple(logits.shape), shape)
                    self.assert_finite(logits)
                    for name in ["unary_logits", "refined_logits"]:
                        self.assertEqual(tuple(aux[name].shape), shape)
                        self.assert_finite(aux[name])
                    torch.testing.assert_close(logits, aux["refined_logits"])
                    correction = (logits - aux["unary_logits"]).abs()
                    strength = aux["refinement_strength"]
                    self.assertTrue(bool((correction <= strength + 1e-6).all()))
                    evidence = aux["evidence_maps"]
                    self.assertEqual(evidence.shape[0], shape[0])
                    self.assertGreater(evidence.shape[1], 0)
                    self.assertEqual(tuple(evidence.shape[-2:]), shape[-2:])
                    self.assert_finite(evidence)
                    expected_edges = {
                        "horizontal_edge_weights": (shape[0], 1, shape[2], shape[3] - 1),
                        "vertical_edge_weights": (shape[0], 1, shape[2] - 1, shape[3]),
                    }
                    for name, edge_shape in expected_edges.items():
                        weights = aux[name]
                        self.assertEqual(tuple(weights.shape), edge_shape)
                        self.assert_finite(weights)
                        self.assertTrue(bool(((weights >= 0) & (weights <= 1)).all()))

    def test_batch_one_training_and_smallest_supported_image(self):
        model = self.make_model().train()
        self.assertTrue(any(isinstance(m, torch.nn.GroupNorm) for m in model.modules()))
        x = torch.rand(1, 1, 8, 8)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), tuple(x.shape))
        self.assert_finite(logits)
        loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        loss.backward()
        self.assert_finite(loss)

    def test_loss_backpropagates_through_unary_and_learned_refinement_strength(self):
        model = self.make_model().train()
        x = torch.rand(2, 1, 17, 31)
        targets = (torch.rand_like(x) > 0.65).to(x.dtype)
        logits, aux = model(x, return_aux=True)
        aux["unary_logits"].retain_grad()
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()

        unary_gradient = aux["unary_logits"].grad
        self.assertIsNotNone(unary_gradient)
        self.assert_finite(unary_gradient)
        self.assertGreater(float(unary_gradient.abs().sum()), 0)
        refinement_gradient = model.raw_refinement_strength.grad
        self.assertIsNotNone(refinement_gradient)
        self.assert_finite(refinement_gradient)
        self.assertGreater(float(refinement_gradient.abs().sum()), 0)
        edge_gradients = [p.grad for p in model.edge_gate.parameters() if p.grad is not None]
        self.assertTrue(edge_gradients)
        self.assertGreater(sum(float(g.abs().sum()) for g in edge_gradients), 0)
        parameter_gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(parameter_gradients), 1)
        for gradient in parameter_gradients:
            self.assert_finite(gradient)

    def test_empty_and_full_binary_inputs_remain_finite(self):
        model = self.make_model().eval()
        with torch.no_grad():
            for value in [0.0, 1.0]:
                with self.subTest(value=value):
                    x = torch.full((1, 1, 17, 31), value)
                    logits, aux = model(x, return_aux=True)
                    self.assert_finite(logits)
                    # Padding must not turn an all-busy image into weak evidence
                    # at a frame boundary, including for kernels wider than x.
                    torch.testing.assert_close(
                        aux["evidence_maps"], torch.full_like(aux["evidence_maps"], value)
                    )
                    for value_tensor in aux.values():
                        if isinstance(value_tensor, torch.Tensor):
                            self.assert_finite(value_tensor)

    def test_disabling_refinement_or_using_zero_steps_returns_unary_exactly(self):
        x = torch.rand(1, 1, 17, 31)
        for config in [{"use_refinement": False}, {"refinement_steps": 0}]:
            with self.subTest(config=config):
                model = self.make_model(**config).eval()
                with torch.no_grad():
                    logits, aux = model(x, return_aux=True)
                torch.testing.assert_close(logits, aux["unary_logits"], rtol=0, atol=0)

    def test_uniformly_weak_edge_gates_suppress_refinement_locally(self):
        model = self.make_model(refinement_steps=3).eval()
        x = torch.zeros(1, 1, 8, 8)
        with torch.no_grad():
            # A coherent positive unary removes boundary cancellation as an
            # explanation for small updates. The graph can accept or suppress
            # this same neighbour evidence solely by changing its edge gates.
            model.unary_head.weight.zero_()
            model.unary_head.bias.fill_(1.0)
            model.raw_refinement_strength.zero_()
            for parameter in model.edge_gate.parameters():
                parameter.zero_()
            final_edge_layer = list(model.edge_gate.children())[-1]
            final_edge_layer.bias.fill_(-10.0)
            weak_logits, weak_aux = model(x, return_aux=True)
            final_edge_layer.bias.zero_()
            strong_logits, strong_aux = model(x, return_aux=True)

        # sigmoid(-10) is small but substantially above float32 epsilon:
        # normalization must retain gate magnitude, not merely ratios.
        self.assertGreater(float(weak_aux["horizontal_edge_weights"].min()), 1e-6)
        torch.testing.assert_close(weak_logits, weak_aux["unary_logits"], rtol=0, atol=1e-3)
        strong_correction = strong_logits - strong_aux["unary_logits"]
        self.assertTrue(bool((strong_correction > 0.1).all()))
        self.assert_finite(weak_logits)
        self.assert_finite(strong_logits)

    def test_disabling_evidence_preserves_forward_and_backward_contract(self):
        model = self.make_model(use_evidence=False).train()
        x = torch.rand(1, 1, 17, 31)
        logits, aux = model(x, return_aux=True)
        self.assertEqual(tuple(logits.shape), tuple(x.shape))
        self.assertEqual(tuple(aux["evidence_maps"].shape), (1, 0, 17, 31))
        self.assert_finite(logits)
        F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        for gradient in gradients:
            self.assert_finite(gradient)

    def test_checkpoint_roundtrip_preserves_deterministic_evaluation(self):
        model = self.make_model().eval()
        x = torch.rand(1, 1, 17, 31)
        with torch.no_grad():
            expected = model(x)
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = self.make_model().eval()
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        with torch.no_grad():
            actual = restored(x)
            repeated = restored(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(repeated, actual, rtol=0, atol=0)

    def test_invalid_batch_channel_or_spatial_shapes_are_rejected(self):
        model = self.make_model().eval()
        invalid_shapes = [(1, 17, 31), (1, 2, 17, 31), (0, 1, 17, 31),
                          (1, 1, 7, 17), (1, 1, 17, 7)]
        for shape in invalid_shapes:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    model(torch.zeros(shape))
        with self.assertRaises(TypeError):
            model(torch.zeros((1, 1, 17, 31), dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
