"""Export and evaluate folded inference models on the saved test split."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

if __package__:
    from . import experiment, fast_experiment
    from .fast_model import FastUNet, fuse_unet_batchnorm
    from .native_backend import enable_native_batchnorm
    from .inference import prepare_cpu
else:
    import experiment
    import fast_experiment
    from fast_model import FastUNet, fuse_unet_batchnorm
    from native_backend import enable_native_batchnorm
    from inference import prepare_cpu


def load_training_model(run_dir):
    checkpoint = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["model"] == "fast_unet":
        fingerprints, factory = fast_experiment.source_fingerprints, fast_experiment.make_model
    else:
        fingerprints, factory = experiment.source_fingerprints, experiment.make_model
    if config.get("source_sha256") != fingerprints():
        raise ValueError("Training source differs; restore the saved source before exporting.")
    model = factory(config).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def convert(model):
    return model.to_deploy() if isinstance(model, FastUNet) else fuse_unet_batchnorm(model)


def load_deployed_model(path):
    """Load exported weights with their validation-selected decision threshold."""
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact["format"] != "folded-segmentation-v1":
        raise ValueError("Unsupported deployment artifact")
    config = artifact["config"]
    if config["model"] == "fast_unet":
        model = FastUNet(base_channels=config["base_channels"], deploy=True)
    else:
        model = convert(experiment.make_model(config).eval())
    model.load_state_dict(artifact["model_state_dict"])
    return model.eval(), artifact["threshold"]


class ConversionCheck(nn.Module):
    def __init__(self, original, deployed, threshold):
        super().__init__()
        self.original, self.deployed, self.threshold = original, deployed, threshold
        self.max_abs_logit_error = 0.0
        self.different_pixels = self.pixels = 0

    def forward(self, x):
        expected, actual = self.original(x), self.deployed(x)
        if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
            raise FloatingPointError("Non-finite logits during conversion check")
        # Accumulated over the actual held-out data, not just a synthetic sample.
        self.max_abs_logit_error = max(self.max_abs_logit_error, float((expected - actual).abs().max()))
        self.different_pixels += int(((expected.sigmoid() > self.threshold) !=
                                      (actual.sigmoid() > self.threshold)).sum())
        self.pixels += actual.numel()
        return actual


def evaluate_and_export(args):
    torch.set_num_threads(args.threads)
    if args.native_batchnorm:
        enable_native_batchnorm()
    run_dir = Path(args.run_dir).resolve()
    original, checkpoint = load_training_model(run_dir)
    config = checkpoint["config"].copy()
    experiment.seed_everything(config["seed"])
    device = experiment.choose_device(args.device)
    if args.optimize_cpu and device.type != "cpu":
        raise ValueError("--optimize-cpu requires --device cpu")
    config.update(batch_size=args.batch_size, workers=args.workers)
    dataset = experiment.load_dataset(config["data_dir"])
    print("Verifying saved dataset and split...", flush=True)
    manifest = experiment.dataset_manifest(dataset)
    saved_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest != saved_manifest or manifest["sha256"] != checkpoint["manifest_sha256"]:
        raise ValueError("Dataset changed since training")
    if experiment.file_sha256(run_dir / "splits.pt") != checkpoint["splits_sha256"]:
        raise ValueError("Split changed since training")
    splits = torch.load(run_dir / "splits.pt", weights_only=True)
    deployed = convert(original)
    output = run_dir / "deploy.pt"
    artifact = dict(format="folded-segmentation-v1", config=checkpoint["config"],
                    model_state_dict=deployed.state_dict(), threshold=checkpoint["threshold"],
                    epoch=checkpoint["epoch"], source_checkpoint_sha256=experiment.file_sha256(run_dir / "best.pt"),
                    export_source_sha256=experiment.file_sha256(Path(__file__)))
    # Round-trip the actual exported artifact before scoring it.
    if output.exists():
        previous = torch.load(output, map_location="cpu", weights_only=True)
        if previous.get("source_checkpoint_sha256") != artifact["source_checkpoint_sha256"]:
            raise FileExistsError("deploy.pt belongs to a different checkpoint; preserve it before exporting again")
    else:
        experiment.save_checkpoint(output, artifact)
    restored, threshold = load_deployed_model(output)
    deployed_parameters = sum(p.numel() for p in restored.parameters())
    restored = restored.to(device)
    if args.optimize_cpu:
        restored = prepare_cpu(restored, torch.zeros(1, 1, 50, 100))
    checked = ConversionCheck(original.to(device), restored, threshold).eval()
    result = experiment.assess(checked, experiment.make_loader(dataset, splits["test"], config, device),
                               device, [threshold])[0]
    result.update(model=config["model"], epoch=checkpoint["epoch"], split="test", test_indices=splits["test"],
                  manifest_sha256=manifest["sha256"], threshold_source="saved validation selection",
                  parameters=deployed_parameters, optimize_cpu=args.optimize_cpu,
                  source_checkpoint_sha256=artifact["source_checkpoint_sha256"],
                  deployment_sha256=experiment.file_sha256(output),
                  evaluation_sources_sha256={name: experiment.file_sha256(Path(__file__).parent / name)
                                             for name in ("deployment.py", "inference.py")},
                  conversion_check=dict(max_abs_logit_error=checked.max_abs_logit_error,
                                        different_prediction_pixels=checked.different_pixels,
                                        total_pixels=checked.pixels),
                  metric_conventions=experiment.METRIC_CONVENTIONS,
                  device=str(device), threads=args.threads, batch_size=args.batch_size)
    evaluation_path = run_dir / ("deployment_evaluation_cpu_optimized.json" if args.optimize_cpu else "deployment_evaluation.json")
    experiment.write_json(evaluation_path, result)
    print(json.dumps({k: result[k] for k in ("overall", "conversion_check", "parameters", "threshold")}, indent=2))
    print(f"Saved {output} and {evaluation_path.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=experiment.positive_int, default=4)
    parser.add_argument("--batch-size", type=experiment.positive_int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--native-batchnorm", action="store_true")
    parser.add_argument("--optimize-cpu", action="store_true")
    arguments = parser.parse_args()
    if arguments.workers < 0:
        parser.error("--workers cannot be negative")
    evaluate_and_export(arguments)
