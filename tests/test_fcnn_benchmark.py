"""Checks for including FCNN in the same trained-model timing protocol."""

import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import torch

from baseline_recreation.model import DLANNBaseline
from UNet.fast_model import FastUNet
from UNet.inference import prepare_cpu
from UNet.model import UNet
from UNet.structured_model import EvidenceAxialNet
from UNet.speed_benchmark import (
    _sha256, benchmark, count_linear_macs, load_benchmark_model, verify_architecture_source,
)


ROOT = Path(__file__).resolve().parents[1]


class FCNNBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def checkpoint(self, directory, kind, model, source, **settings):
        directory.mkdir()
        config = dict(
            model=kind,
            source_sha256={source: _sha256(ROOT / source), "experiment.py": "old"},
            **settings,
        )
        torch.save(dict(config=config, model_state_dict=model.state_dict()), directory / "best.pt")
        return directory

    def test_fcnn_keeps_predictions_for_contiguous_and_channels_last_inputs(self):
        original = DLANNBaseline().eval()
        with tempfile.TemporaryDirectory() as folder:
            directory = self.checkpoint(Path(folder) / "fcnn", "fcnn", original,
                                        "baseline_recreation/model.py")
            restored, _ = load_benchmark_model(directory)
            inputs = torch.rand(3, 1, 50, 100)
            with torch.inference_mode():
                expected = original(inputs.flatten(1))
                torch.testing.assert_close(restored(inputs), expected)
                torch.testing.assert_close(
                    restored(inputs.to(memory_format=torch.channels_last)), expected
                )
                self.assertEqual(count_linear_macs(restored, inputs[:1]), 1_497_088)
                prepared = prepare_cpu(restored, inputs[:1])
                torch.testing.assert_close(prepared(inputs), expected)

    def test_changed_architecture_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = self.checkpoint(Path(folder) / "fcnn", "fcnn", DLANNBaseline(),
                                        "baseline_recreation/model.py")
            path = directory / "best.pt"
            checkpoint = torch.load(path, weights_only=True)
            checkpoint["config"]["source_sha256"]["baseline_recreation/model.py"] = "changed"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "Inference architecture differs"):
                load_benchmark_model(directory)

    def test_only_verified_formatting_changes_are_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            old, current = Path(folder) / "snapshot.py", Path(folder) / "current.py"
            old.write_text("value=3\n")
            current.write_text("# reformatted\nvalue = 3\n")
            digest = _sha256(old)
            self.assertIn("AST identical", verify_architecture_source(current, digest, old))
            current.write_text("value = 4\n")
            with self.assertRaisesRegex(ValueError, "architecture differs"):
                verify_architecture_source(current, digest, old)
            old.write_text("value = 4\n")
            with self.assertRaisesRegex(ValueError, "architecture differs"):
                verify_architecture_source(current, digest, old)

    def test_training_runner_change_does_not_block_inference_architecture(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = self.checkpoint(Path(folder) / "unet", "unet", UNet(), "UNet/model.py")
            model, _ = load_benchmark_model(directory)
            self.assertIsInstance(model, UNet)
            self.assertFalse(model.training)

    def test_fcnn_cannot_be_mixed_with_random_weight_screening(self):
        args = argparse.Namespace(fcnn_run_dir="some-run", fast_run_dir=None)
        with self.assertRaisesRegex(ValueError, "requires --fast-run-dir"):
            benchmark(args)

    def test_one_round_includes_all_models_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            unet = self.checkpoint(folder / "unet", "unet", UNet(), "UNet/model.py")
            evidence = self.checkpoint(
                folder / "evidence", "evidence_axial", EvidenceAxialNet(),
                "UNet/structured_model.py", base_channels=16, refinement_steps=3,
                no_evidence=False, no_refinement=False,
            )
            fast = self.checkpoint(folder / "fast", "fast_unet", FastUNet(),
                                   "UNet/fast_model.py", base_channels=8, plain_blocks=False)
            fcnn = self.checkpoint(folder / "fcnn", "fcnn", DLANNBaseline(),
                                   "baseline_recreation/model.py")
            args = argparse.Namespace(
                fcnn_run_dir=str(fcnn), fast_run_dir=str(fast), unet_run_dir=str(unet),
                evidence_run_dir=str(evidence), threads=2, native_batchnorm=False,
                device="cpu", optimize_cpu=False, channels_last=False, batch_size=1,
                rounds=1, warmup=1, iterations=2, output=None,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = benchmark(args)
            records = {r["model"]: r for r in result["records"]}
            self.assertEqual(set(records), {
                "unet", "unet_folded", "evidence_axial", "fast_unet_b8_deploy", "fcnn"
            })
            self.assertEqual(records["fcnn"]["parameters"], 1_498_192)
            self.assertEqual(records["fcnn"]["convolution_macs_per_image"], 0)
            self.assertEqual(records["fcnn"]["counted_macs_per_image"], 1_497_088)
            self.assertEqual(records["fcnn"]["checkpoint_sha256"], _sha256(fcnn / "best.pt"))
            for record in records.values():
                self.assertEqual(len(record["rounds"]), 1)
                self.assertGreater(record["median_batch_ms"], 0)


if __name__ == "__main__":
    unittest.main()
