"""Test F1 selection and interruption recovery using a tiny synthetic dataset."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torch.utils.data import Dataset

from UNet import f1_experiment as runner


class ToyDataset(Dataset):
    def __init__(self, path):
        self.files = [path]
        generator = torch.Generator().manual_seed(120)
        self.inputs = torch.rand(10, 1, 8, 9, generator=generator)
        self.targets = (self.inputs > 0.6).float()

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index], torch.tensor(-20. if index % 2 else 5.)


class F1ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def args(self, *words):
        parser = runner.parser()
        result = parser.parse_args(list(words))
        runner.validate_args(result, parser)
        return result

    def test_loss_is_fp32_and_prioritizes_per_image_dice(self):
        logits = torch.tensor([[[[-2., 2.]]], [[[1., -1.]]]], dtype=torch.bfloat16, requires_grad=True)
        target = torch.tensor([[[[0., 1.]]], [[[0., 1.]]]])
        loss = runner.segmentation_loss(logits, target)
        self.assertEqual(loss.dtype, torch.float32)
        probabilities = logits.float().sigmoid().flatten(1)
        truth = target.flatten(1)
        expected = .5 * nn.functional.binary_cross_entropy_with_logits(logits.float(), target)
        expected += 1 - ((2 * (probabilities * truth).sum(1) + 1e-6) / (probabilities.sum(1) + truth.sum(1) + 1e-6)).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_ema_updates_parameters_copies_buffers_and_has_no_gradients(self):
        model = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2))
        ema = runner.ModelEMA(model, .99)
        before = model[0].weight.detach().clone()
        with torch.no_grad():
            model[0].weight.add_(2)
            model[1].running_mean.fill_(5)
            model[1].num_batches_tracked.fill_(7)
        ema.update(model)
        torch.testing.assert_close(ema.model[0].weight, before + 2 * (1 - 2 / 11))
        torch.testing.assert_close(ema.model[1].running_mean, model[1].running_mean)
        self.assertEqual(ema.model[1].num_batches_tracked, 7)
        self.assertTrue(all(not parameter.requires_grad for parameter in ema.model.parameters()))

    def test_schedule_reaches_warmup_peak_and_cosine_floor(self):
        values = [runner.learning_rate(step, 10, 3, 1., .01) for step in range(10)]
        self.assertAlmostEqual(values[0], 1 / 3)
        self.assertEqual(values[2:4], [1., 1.])
        self.assertAlmostEqual(values[-1], .01)
        self.assertTrue(all(a >= b for a, b in zip(values[3:], values[4:])))

    def test_selection_uses_f1_not_loss_and_deterministic_threshold_ties(self):
        rows = [dict(threshold=t, loss=loss, overall={"macro": {"f1": f1}})
                for t, loss, f1 in ((.2, .01, .7), (.7, 9., .9), (.4, 10., .9), (.6, 10., .9))]
        self.assertEqual(runner.select_validation(rows)["threshold"], .4)

    def test_cpu_bf16_cli_rejected(self):
        with self.assertRaisesRegex(ValueError, "BF16 requires"):
            runner.resolve_precision("bf16", torch.device("cpu"))
        self.assertEqual(runner.resolve_precision("auto", torch.device("cpu")), "fp32")

    def test_worker_seed_draws_do_not_change_epoch_shuffle(self):
        config = dict(batch_size=2, seed=42, workers=0)
        loader = runner.make_training_loader(list(range(20)), list(range(20)), config, torch.device("cpu"))
        self.assertIsNot(loader.generator, loader.sampler.generator)
        loader.sampler.generator.manual_seed(43)
        expected = list(loader.sampler)
        loader.sampler.generator.manual_seed(43)
        # A fresh DataLoader consumes a worker base seed; a persistent one may not.
        torch.empty((), dtype=torch.int64).random_(generator=loader.generator)
        self.assertEqual(list(loader.sampler), expected)

    def test_ambiguous_fp32_thresholds_rejected_before_training(self):
        parser = runner.parser()
        args = parser.parse_args(["train", "--run-dir", "unused", "--thresholds", ".5", ".5000000001"])
        with mock.patch.object(parser, "error", side_effect=ValueError("invalid thresholds")):
            with self.assertRaisesRegex(ValueError, "invalid thresholds"):
                runner.validate_args(args, parser)

    def test_training_resume_and_heldout_evaluation(self):
        # Keep temporary writes inside the workspace on restricted Windows hosts.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            folder = Path(folder)
            data_file = folder / "toy.mat"
            data_file.write_bytes(b"synthetic-only")
            dataset = ToyDataset(data_file)
            complete, interrupted = folder / "complete", folder / "interrupted"
            common = ("--device", "cpu", "--base-channels", "4", "--epochs", "2",
                      "--warmup-epochs", "0", "--batch-size", "2", "--thresholds", ".25", ".5", ".75")
            with mock.patch.object(runner.base, "load_dataset", return_value=dataset), redirect_stdout(io.StringIO()):
                runner.train(self.args("train", "--run-dir", str(complete), *common))
                original_loss = runner.supervised_loss
                calls = 0

                def stop_after_epoch(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 4:
                        raise RuntimeError("simulated interruption")
                    return original_loss(*args, **kwargs)

                with mock.patch.object(runner, "supervised_loss", side_effect=stop_after_epoch):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        runner.train(self.args("train", "--run-dir", str(interrupted), *common))
                partial = torch.load(interrupted / "latest.pt", weights_only=True)
                self.assertEqual(partial["epoch"], 1)
                runner.train(self.args("resume", "--run-dir", str(interrupted), "--device", "cpu"))
                full = torch.load(complete / "latest.pt", weights_only=True)
                resumed = torch.load(interrupted / "latest.pt", weights_only=True)
                self.assertEqual(full["global_step"], 6)
                self.assertEqual(full["global_step"], resumed["global_step"])
                for key in ("model_state_dict", "ema_state_dict"):
                    for name in full[key]:
                        torch.testing.assert_close(full[key][name], resumed[key][name], rtol=0, atol=0)
                self.assertEqual(full["validation"], resumed["validation"])
                best = torch.load(interrupted / "best.pt", weights_only=True)
                self.assertEqual(best["validation"]["overall"]["macro"]["f1"],
                                 max(row["validation"]["overall"]["macro"]["f1"] for row in resumed["history"]))
                self.assertFalse((interrupted / "evaluation.json").exists())
                runner.evaluate(self.args("evaluate", "--run-dir", str(interrupted), "--device", "cpu"))
                result = json.loads((interrupted / "evaluation.json").read_text())
                splits = torch.load(interrupted / "splits.pt", weights_only=True)
                self.assertEqual(result["test_indices"], splits["test"])
                self.assertEqual(result["threshold"], best["threshold"])
                self.assertEqual(result["overall"]["samples"], 2)
                self.assertEqual(set(result["per_snr"]),
                                 {f"{float(dataset[index][2]):g}" for index in splits["test"]})
                # Exercise the real ensemble command across verified checkpoints.
                from UNet import f1_ensemble
                ensemble_dir = folder / "ensemble"
                ensemble_args = f1_ensemble.parser().parse_args([
                    "--run-dirs", str(complete), str(interrupted), "--output-dir", str(ensemble_dir),
                    "--device", "cpu", "--batch-size", "2",
                ])
                f1_ensemble.run(ensemble_args)
                selection = json.loads((ensemble_dir / "selection.json").read_text())
                ensemble_result = json.loads((ensemble_dir / "evaluation.json").read_text())
                self.assertEqual(ensemble_result["threshold"], selection["threshold"])
                self.assertEqual(ensemble_result["tta"], selection["tta"])
                self.assertEqual(ensemble_result["test_indices"], splits["test"])
                self.assertEqual(selection["used_validation_indices"], splits["used_val"])
                # An altered file or split must not silently change the holdout.
                data_file.write_bytes(b"altered-content")
                with self.assertRaisesRegex(ValueError, "Dataset differs"):
                    runner.verify_run(interrupted, best)


if __name__ == "__main__":
    unittest.main()
