"""Reproducible binary-input comparisons using the existing, unchanged dataset.

Run ``python UNet/experiment.py --help`` for commands. Thresholds are selected
on validation macro F1; the held-out test split never selects a threshold.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split


SPLIT_SEED = 42
METRIC_CONVENTIONS = {
    "prediction": "sigmoid(logits) > threshold; target > 0.5",
    "micro": "scores from summed pixel TP, FP and FN",
    "macro": "unweighted mean of per-image scores",
    "empty": "Both masks empty: all four scores=1. Otherwise undefined precision/recall=0.",
    "threshold_selection": "maximum validation macro F1; ties closest to 0.5, then lower threshold",
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest(dataset):
    files = [{"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
             for path in dataset.files]
    payload = {"samples": len(dataset), "files": files}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {**payload, "sha256": digest}


def source_fingerprints():
    directory = Path(__file__).resolve().parent
    return {name: file_sha256(directory / name)
            for name in ("experiment.py", "model.py", "structured_model.py", "dataset.py")}


def load_dataset(path):
    # Keep optional HDF5 imports out of --help and synthetic smoke tests.
    if __package__:
        from .dataset import UNetSignalDataset
    else:
        from dataset import UNetSignalDataset
    return UNetSignalDataset(path)


def make_model(config):
    if config["model"] == "unet":
        if __package__:
            from .model import UNet
        else:
            from model import UNet
        return UNet()
    if __package__:
        from .structured_model import EvidenceAxialNet
    else:
        from structured_model import EvidenceAxialNet
    return EvidenceAxialNet(base_channels=config["base_channels"],
                            refinement_steps=config["refinement_steps"],
                            use_evidence=not config["no_evidence"],
                            use_refinement=not config["no_refinement"])


def choose_device(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm requested but unavailable; use --device cpu.")
    return device


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def bce_dice_loss(logits, targets):
    probabilities = logits.sigmoid().flatten(1)
    target_pixels = targets.flatten(1)
    dice = (2 * (probabilities * target_pixels).sum(1) + 1e-6) / (
        probabilities.sum(1) + target_pixels.sum(1) + 1e-6)
    return F.binary_cross_entropy_with_logits(logits, targets) + 1 - dice.mean()


def scores(tp, fp, fn):
    both_empty = tp + fp + fn == 0
    return {"f1": 2 * tp / (2 * tp + fp + fn) if not both_empty else 1.0,
            "iou": tp / (tp + fp + fn) if not both_empty else 1.0,
            "precision": tp / (tp + fp) if tp + fp else float(both_empty),
            "recall": tp / (tp + fn) if tp + fn else float(both_empty)}


class SegmentationMetrics:
    """Accumulate image counts, keeping macro and micro denominators distinct."""
    def __init__(self):
        self.samples = self.tp = self.fp = self.fn = self.empty_targets = 0
        self.sums = dict.fromkeys(("f1", "iou", "precision", "recall"), 0.0)

    def add(self, tp, fp, fn):
        self.samples += 1
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.empty_targets += int(tp + fn == 0)
        for key, value in scores(tp, fp, fn).items():
            self.sums[key] += value

    def result(self):
        if not self.samples:
            raise ValueError("Cannot report metrics for an empty split.")
        return {"samples": self.samples, "empty_targets": self.empty_targets,
                "tp": self.tp, "fp": self.fp, "fn": self.fn,
                "micro": scores(self.tp, self.fp, self.fn),
                "macro": {key: value / self.samples for key, value in self.sums.items()}}


def make_loader(dataset, indices, config, device, training=False):
    return DataLoader(Subset(dataset, indices), batch_size=config["batch_size"],
                      shuffle=training, num_workers=config["workers"],
                      pin_memory=device.type == "cuda", persistent_workers=config["workers"] > 0,
                      generator=torch.Generator().manual_seed(config["seed"]))


def limited(indices, maximum):
    return indices if maximum is None else indices[:maximum]


@torch.inference_mode()
def assess(model, loader, device, thresholds):
    model.eval()
    overall = [SegmentationMetrics() for _ in thresholds]
    by_snr = [defaultdict(SegmentationMetrics) for _ in thresholds]
    total_loss = 0.0
    for inputs, targets, snrs in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = bce_dice_loss(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss.")
        total_loss += float(loss) * inputs.size(0)
        probabilities, truth = logits.sigmoid(), targets > 0.5
        snr_values = snrs.tolist()
        for threshold_index, threshold in enumerate(thresholds):
            predictions = probabilities > threshold
            counts = torch.stack(((predictions & truth).flatten(1).sum(1),
                                  (predictions & ~truth).flatten(1).sum(1),
                                  (~predictions & truth).flatten(1).sum(1)), dim=1).cpu().tolist()
            for (tp, fp, fn), snr in zip(counts, snr_values):
                overall[threshold_index].add(tp, fp, fn)
                by_snr[threshold_index][f"{snr:g}"].add(tp, fp, fn)
    return [{"threshold": threshold, "loss": total_loss / overall[index].samples,
             "overall": overall[index].result(),
             "per_snr": {key: value.result() for key, value in sorted(by_snr[index].items(), key=lambda item: float(item[0]))}}
            for index, threshold in enumerate(thresholds)]


def train(args):
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty to preserve existing experiments: {run_dir}")
    config = vars(args).copy()
    config.update(data_dir=str(Path(args.data_dir).resolve()), run_dir=str(run_dir),
                  split_seed=SPLIT_SEED, torch_version=str(torch.__version__), loss="BCE + per-image soft Dice",
                  source_sha256=source_fingerprints(),
                  dependencies={name: importlib.metadata.version(name) for name in ("torch", "numpy", "h5py")})
    seed_everything(args.seed)
    device = choose_device(args.device)
    config["resolved_device"] = str(device)
    config["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    dataset = load_dataset(config["data_dir"])
    n_train, n_val = int(0.6 * len(dataset)), int(0.2 * len(dataset))
    if min(n_train, n_val, len(dataset) - n_train - n_val) == 0:
        raise ValueError("Dataset needs non-empty training, validation and test splits.")
    subsets = random_split(dataset, [n_train, n_val, len(dataset) - n_train - n_val],
                           generator=torch.Generator().manual_seed(SPLIT_SEED))
    splits = {name: subset.indices for name, subset in zip(("train", "val", "test"), subsets)}
    # Full splits and the exact effective subset IDs are both retained for smoke caps.
    splits["used_train"] = limited(splits["train"], args.max_train_samples)
    splits["used_val"] = limited(splits["val"], args.max_eval_samples)
    print("Fingerprinting dataset files...", flush=True)
    manifest = dataset_manifest(dataset)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "manifest.json", manifest)
    save_checkpoint(run_dir / "splits.pt", splits)
    split_digest = file_sha256(run_dir / "splits.pt")
    train_loader = make_loader(dataset, splits["used_train"], config, device, training=True)
    val_loader = make_loader(dataset, splits["used_val"], config, device)
    model = make_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        accumulated_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = bce_dice_loss(model(inputs), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            accumulated_loss += float(loss.detach()) * inputs.size(0)
        sweep = assess(model, val_loader, device, args.thresholds)
        validation = max(sweep, key=lambda result: (result["overall"]["macro"]["f1"],
                                                    -abs(result["threshold"] - 0.5), -result["threshold"]))
        row = {"epoch": epoch, "train_loss": accumulated_loss / len(splits["used_train"]),
               "validation": validation, "seconds": time.perf_counter() - started,
               "threshold_sweep": [{"threshold": result["threshold"], "macro_f1": result["overall"]["macro"]["f1"]} for result in sweep]}
        history.append(row)
        checkpoint = {"epoch": epoch, "config": config, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "validation": validation,
                      "threshold": validation["threshold"], "manifest_sha256": manifest["sha256"],
                      "splits_sha256": split_digest, "metric_conventions": METRIC_CONVENTIONS,
                      "parameters": sum(parameter.numel() for parameter in model.parameters())}
        save_checkpoint(run_dir / "latest.pt", checkpoint)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            save_checkpoint(run_dir / "best.pt", checkpoint)
        write_json(run_dir / "history.json", history)
        print(f"Epoch {epoch}: train={row['train_loss']:.6f}, val={validation['loss']:.6f}, "
              f"val macro F1={validation['overall']['macro']['f1']:.4f}, threshold={validation['threshold']:.2f}", flush=True)
    print(f"Saved experiment to {run_dir}; test split remains unevaluated.", flush=True)


def evaluate(args):
    run_dir = Path(args.run_dir).resolve()
    checkpoint = torch.load(run_dir / f"{args.checkpoint}.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"].copy()
    if config.get("source_sha256") != source_fingerprints():
        raise ValueError("Experiment/model source differs from training; restore the recorded source before evaluation.")
    config.update(batch_size=args.batch_size or config["batch_size"],
                  workers=config["workers"] if args.workers is None else args.workers)
    seed_everything(config["seed"])
    device = choose_device(args.device)
    dataset = load_dataset(args.data_dir or config["data_dir"])
    print("Verifying dataset fingerprint...", flush=True)
    manifest = dataset_manifest(dataset)
    saved_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest != saved_manifest or manifest["sha256"] != checkpoint["manifest_sha256"]:
        raise ValueError("Dataset fingerprint differs from training; refusing a changed holdout.")
    if file_sha256(run_dir / "splits.pt") != checkpoint["splits_sha256"]:
        raise ValueError("Saved split file has changed since training.")
    splits = torch.load(run_dir / "splits.pt", map_location="cpu", weights_only=True)
    indices = limited(splits["test"], args.max_eval_samples)
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = assess(model, make_loader(dataset, indices, config, device), device, [checkpoint["threshold"]])[0]
    result.update(checkpoint=args.checkpoint, epoch=checkpoint["epoch"], model=config["model"],
                  manifest_sha256=manifest["sha256"], split="test", test_indices=indices,
                  full_test_samples=len(splits["test"]), metric_conventions=METRIC_CONVENTIONS,
                  threshold_source="saved validation selection", device=str(device),
                  batch_size=config["batch_size"], threads=args.threads,
                  parameters=checkpoint["parameters"])
    output = run_dir / ("evaluation.json" if args.max_eval_samples is None else "evaluation_subset.json")
    write_json(output, result)
    print(json.dumps(result["overall"], indent=2))
    print(f"Saved {output}")


def smoke(args):
    seed_everything(args.seed)
    device = choose_device(args.device)
    model = make_model(vars(args)).to(device)
    inputs = torch.randint(0, 2, (2, 1, 50, 100), device=device).float()
    targets = torch.randint(0, 2, inputs.shape, device=device).float()
    logits = model(inputs)
    if logits.shape != targets.shape:
        raise AssertionError(f"Unexpected output shape: {tuple(logits.shape)}")
    loss = bce_dice_loss(logits, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite smoke loss.")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
    print(json.dumps({"model": args.model, "device": str(device), "shape": list(logits.shape),
                      "loss": float(loss.detach()), "parameters": sum(p.numel() for p in model.parameters()),
                      "finite_gradients": True}, indent=2))


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate", "smoke"):
        sub = commands.add_parser(command)
        sub.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
        sub.add_argument("--threads", type=positive_int, default=4, help="PyTorch CPU thread count")
        if command in ("train", "smoke"):
            sub.add_argument("--model", choices=("unet", "evidence_axial"), default="evidence_axial")
            sub.add_argument("--seed", type=int, default=42, help="training RNG; split seed remains 42")
            sub.add_argument("--base-channels", type=positive_int, default=16)
            sub.add_argument("--refinement-steps", type=positive_int, default=3)
            sub.add_argument("--no-evidence", action="store_true")
            sub.add_argument("--no-refinement", action="store_true")
        if command in ("train", "evaluate"):
            sub.add_argument("--run-dir", required=True)
            sub.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data") if command == "train" else None)
            sub.add_argument("--batch-size", type=positive_int, default=64 if command == "train" else None)
            sub.add_argument("--workers", type=int, default=0 if command == "train" else None)
            sub.add_argument("--max-eval-samples", type=positive_int, help="validation cap during train; test cap during evaluate")
        if command == "train":
            sub.add_argument("--epochs", type=positive_int, default=8)
            sub.add_argument("--lr", type=float, default=1e-3)
            sub.add_argument("--max-train-samples", type=positive_int)
            sub.add_argument("--thresholds", type=float, nargs="+", default=[i / 20 for i in range(1, 20)])
        if command == "evaluate":
            sub.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    return root


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    if getattr(args, "workers", None) is not None and args.workers < 0:
        argument_parser.error("--workers cannot be negative")
    if args.command == "train" and (not math.isfinite(args.lr) or not args.lr > 0 or any(not 0 < threshold < 1 for threshold in args.thresholds)):
        argument_parser.error("--lr must be positive and --thresholds must lie strictly between 0 and 1")
    torch.set_num_threads(args.threads)
    {"train": train, "evaluate": evaluate, "smoke": smoke}[args.command](args)


if __name__ == "__main__":
    main()
