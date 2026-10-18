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
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
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
    files = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in dataset.files
    ]
    payload = {"samples": len(dataset), "files": files}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {**payload, "sha256": digest}


def source_fingerprints():
    directory = Path(__file__).resolve().parent
    return {
        name: file_sha256(directory / name)
        for name in ("experiment.py", "model.py", "structured_model.py", "dataset.py")
    }


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
    return EvidenceAxialNet(
        base_channels=config["base_channels"],
        refinement_steps=config["refinement_steps"],
        use_evidence=not config["no_evidence"],
        use_refinement=not config["no_refinement"],
    )


def choose_device(name):
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name == "auto"
        else torch.device(name)
    )
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
        probabilities.sum(1) + target_pixels.sum(1) + 1e-6
    )
    return F.binary_cross_entropy_with_logits(logits, targets) + 1 - dice.mean()


def scores(tp, fp, fn):
    both_empty = tp + fp + fn == 0
    return {
        "f1": 2 * tp / (2 * tp + fp + fn) if not both_empty else 1.0,
        "iou": tp / (tp + fp + fn) if not both_empty else 1.0,
        "precision": tp / (tp + fp) if tp + fp else float(both_empty),
        "recall": tp / (tp + fn) if tp + fn else float(both_empty),
    }


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
        return {
            "samples": self.samples,
            "empty_targets": self.empty_targets,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "micro": scores(self.tp, self.fp, self.fn),
            "macro": {key: value / self.samples for key, value in self.sums.items()},
        }


def make_loader(dataset, indices, config, device, training=False):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=config["batch_size"],
        shuffle=training,
        num_workers=config["workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["workers"] > 0,
        generator=torch.Generator().manual_seed(config["seed"]),
    )


def limited(indices, maximum):
    return indices if maximum is None else indices[:maximum]


@torch.inference_mode()
def assess(model, loader, device, thresholds):
    model.eval()
    overall = [SegmentationMetrics() for _ in thresholds]
    by_snr = [defaultdict(SegmentationMetrics) for _ in thresholds]
    total_loss = 0.0
    for inputs, targets, snrs in loader:
        inputs, targets = (
            inputs.to(device, non_blocking=True),
            targets.to(device, non_blocking=True),
        )
        logits = model(inputs)
        loss = bce_dice_loss(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss.")
        total_loss += float(loss) * inputs.size(0)
        probabilities, truth = logits.sigmoid(), targets > 0.5
        snr_values = snrs.tolist()
        for threshold_index, threshold in enumerate(thresholds):
            predictions = probabilities > threshold
            counts = (
                torch.stack(
                    (
                        (predictions & truth).flatten(1).sum(1),
                        (predictions & ~truth).flatten(1).sum(1),
                        (~predictions & truth).flatten(1).sum(1),
                    ),
                    dim=1,
                )
                .cpu()
                .tolist()
            )
            for (tp, fp, fn), snr in zip(counts, snr_values):
                overall[threshold_index].add(tp, fp, fn)
                by_snr[threshold_index][f"{snr:g}"].add(tp, fp, fn)
    return [
        {
            "threshold": threshold,
            "loss": total_loss / overall[index].samples,
            "overall": overall[index].result(),
            "per_snr": {
                key: value.result()
                for key, value in sorted(
                    by_snr[index].items(), key=lambda item: float(item[0])
                )
            },
        }
        for index, threshold in enumerate(thresholds)
    ]


def train(args):
    """Train while selecting the checkpoint and threshold by validation macro F1.

    The held-out test split is never read during training. The *same* threshold
    selected on validation is used to report training metrics for comparison.
    """
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory must be empty to preserve existing experiments: {run_dir}"
        )
    config = vars(args).copy()
    config.update(
        data_dir=str(Path(args.data_dir).resolve()),
        run_dir=str(run_dir),
        split_seed=SPLIT_SEED,
        torch_version=str(torch.__version__),
        loss="BCE + per-image soft Dice",
        checkpoint_selection="maximum validation macro F1; tie: lowest validation loss",
        early_stopping_metric="validation macro F1",
        source_sha256=source_fingerprints(),
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "h5py")
        },
    )
    seed_everything(args.seed)
    device = choose_device(args.device)
    config["resolved_device"] = str(device)
    config["device_name"] = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    dataset = load_dataset(config["data_dir"])
    n_train, n_val = int(0.6 * len(dataset)), int(0.2 * len(dataset))
    if min(n_train, n_val, len(dataset) - n_train - n_val) == 0:
        raise ValueError(
            "Dataset needs non-empty training, validation and test splits."
        )
    subsets = random_split(
        dataset,
        [n_train, n_val, len(dataset) - n_train - n_val],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    splits = {
        name: subset.indices for name, subset in zip(("train", "val", "test"), subsets)
    }
    # Preserve both full split IDs and optional reduced subsets used for testing.
    splits["used_train"] = limited(splits["train"], args.max_train_samples)
    splits["used_val"] = limited(splits["val"], args.max_eval_samples)
    print("Fingerprinting dataset files...", flush=True)
    manifest = dataset_manifest(dataset)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "manifest.json", manifest)
    save_checkpoint(run_dir / "splits.pt", splits)
    split_digest = file_sha256(run_dir / "splits.pt")

    train_loader = make_loader(
        dataset, splits["used_train"], config, device, training=True
    )
    val_loader = make_loader(dataset, splits["used_val"], config, device)
    model = make_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.min_delta,
        threshold_mode="abs",
        min_lr=args.min_lr,
    )

    # Save the highest validation macro F1, even when a small gain is not large
    # enough to reset early-stopping patience. The latter uses min_delta.
    best_f1 = -float("inf")
    best_val_loss = float("inf")
    best_epoch = None
    best_threshold = None
    significant_f1 = -float("inf")
    epochs_without_significant_gain = 0
    history = []
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        lr_used = optimizer.param_groups[0]["lr"]
        model.train()
        accumulated_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs, targets = (
                inputs.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = bce_dice_loss(model(inputs), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf"), error_if_nonfinite=True
            )
            optimizer.step()
            accumulated_loss += float(loss.detach()) * inputs.size(0)

        # Select the prediction threshold using validation *only*.
        sweep = assess(model, val_loader, device, args.thresholds)
        validation = max(
            sweep,
            key=lambda result: (
                result["overall"]["macro"]["f1"],
                -abs(result["threshold"] - 0.5),
                -result["threshold"],
            ),
        )
        val_f1 = validation["overall"]["macro"]["f1"]
        val_loss = validation["loss"]

        # A separate no-grad train-set evaluation ensures train/val metrics
        # represent the same completed epoch, same metric, same threshold.
        train_evaluation = assess(
            model, train_loader, device, [validation["threshold"]]
        )[0]
        scheduler.step(val_f1)
        next_lr = optimizer.param_groups[0]["lr"]

        if val_f1 > significant_f1 + args.min_delta:
            significant_f1 = val_f1
            epochs_without_significant_gain = 0
        else:
            epochs_without_significant_gain += 1

        # Main objective: best validation macro F1. Use loss only to break ties.
        is_best = (
            val_f1 > best_f1
            or (val_f1 == best_f1 and val_loss < best_val_loss)
        )
        if is_best:
            best_f1, best_val_loss = val_f1, val_loss
            best_epoch, best_threshold = epoch, validation["threshold"]

        row = {
            "epoch": epoch,
            "train_loss": accumulated_loss / len(splits["used_train"]),
            "train_evaluation": train_evaluation,
            "validation": validation,
            "training_threshold": validation["threshold"],
            "learning_rate": lr_used,
            "next_learning_rate": next_lr,
            "best_validation_macro_f1": best_f1,
            "best_epoch": best_epoch,
            "epochs_without_significant_gain": epochs_without_significant_gain,
            "seconds": time.perf_counter() - started,
            "threshold_sweep": [
                {
                    "threshold": result["threshold"],
                    "macro_f1": result["overall"]["macro"]["f1"],
                }
                for result in sweep
            ],
        }
        history.append(row)
        checkpoint = {
            "epoch": epoch,
            "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "validation": validation,
            "threshold": validation["threshold"],
            "best_validation_macro_f1": best_f1,
            "best_epoch": best_epoch,
            "manifest_sha256": manifest["sha256"],
            "splits_sha256": split_digest,
            "metric_conventions": METRIC_CONVENTIONS,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        save_checkpoint(run_dir / "latest.pt", checkpoint)
        if is_best:
            save_checkpoint(run_dir / "best.pt", checkpoint)
        write_json(run_dir / "history.json", history)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss={row['train_loss']:.6f} | "
            f"train macro F1={train_evaluation['overall']['macro']['f1']:.4f} | "
            f"val loss={val_loss:.6f} | val macro F1={val_f1:.4f} | "
            f"val macro IoU={validation['overall']['macro']['iou']:.4f} | "
            f"threshold={validation['threshold']:.2f} | "
            f"LR={lr_used:.2e} | "
            f"best F1={best_f1:.4f} (epoch {best_epoch}) | "
            f"patience={epochs_without_significant_gain}/{args.patience}",
            flush=True,
        )
        if next_lr != lr_used:
            print(f"  Learning rate reduced to {next_lr:.2e}", flush=True)
        if epochs_without_significant_gain >= args.patience:
            stopped_early = True
            print(
                f"Early stopping: validation macro F1 failed to improve "
                f"by more than {args.min_delta:g} for {args.patience} epochs.",
                flush=True,
            )
            break

    write_json(
        run_dir / "summary.json",
        {
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
            "best_validation_loss": best_val_loss,
            "best_threshold": best_threshold,
            "epochs_completed": len(history),
            "maximum_epochs": args.epochs,
            "early_stopped": stopped_early,
            "criterion": "maximum validation macro F1; ties lowest validation loss",
            "test_evaluated": False,
        },
    )
    print(
        f"Best validation macro F1={best_f1:.6f} at epoch {best_epoch}, "
        f"threshold={best_threshold:.2f}. Saved experiment to {run_dir}; "
        "test split remains unevaluated.",
        flush=True,
    )


def evaluate(args):
    run_dir = Path(args.run_dir).resolve()
    checkpoint = torch.load(
        run_dir / f"{args.checkpoint}.pt", map_location="cpu", weights_only=True
    )
    config = checkpoint["config"].copy()
    if config.get("source_sha256") != source_fingerprints():
        raise ValueError(
            "Experiment/model source differs from training; restore the recorded source before evaluation."
        )
    config.update(
        batch_size=args.batch_size or config["batch_size"],
        workers=config["workers"] if args.workers is None else args.workers,
    )
    seed_everything(config["seed"])
    device = choose_device(args.device)
    dataset = load_dataset(args.data_dir or config["data_dir"])
    print("Verifying dataset fingerprint...", flush=True)
    manifest = dataset_manifest(dataset)
    saved_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest != saved_manifest
        or manifest["sha256"] != checkpoint["manifest_sha256"]
    ):
        raise ValueError(
            "Dataset fingerprint differs from training; refusing a changed holdout."
        )
    if file_sha256(run_dir / "splits.pt") != checkpoint["splits_sha256"]:
        raise ValueError("Saved split file has changed since training.")
    splits = torch.load(run_dir / "splits.pt", map_location="cpu", weights_only=True)
    indices = limited(splits["test"], args.max_eval_samples)
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = assess(
        model,
        make_loader(dataset, indices, config, device),
        device,
        [checkpoint["threshold"]],
    )[0]
    result.update(
        checkpoint=args.checkpoint,
        epoch=checkpoint["epoch"],
        model=config["model"],
        manifest_sha256=manifest["sha256"],
        split="test",
        test_indices=indices,
        full_test_samples=len(splits["test"]),
        metric_conventions=METRIC_CONVENTIONS,
        threshold_source="saved validation selection",
        device=str(device),
        batch_size=config["batch_size"],
        threads=args.threads,
        parameters=checkpoint["parameters"],
    )
    output = run_dir / (
        "evaluation.json" if args.max_eval_samples is None else "evaluation_subset.json"
    )
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
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), float("inf"), error_if_nonfinite=True
    )
    print(
        json.dumps(
            {
                "model": args.model,
                "device": str(device),
                "shape": list(logits.shape),
                "loss": float(loss.detach()),
                "parameters": sum(p.numel() for p in model.parameters()),
                "finite_gradients": True,
            },
            indent=2,
        )
    )


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
        sub.add_argument(
            "--threads", type=positive_int, default=4, help="PyTorch CPU thread count"
        )
        if command in ("train", "smoke"):
            sub.add_argument(
                "--model", choices=("unet", "evidence_axial"), default="evidence_axial"
            )
            sub.add_argument(
                "--seed",
                type=int,
                default=42,
                help="training RNG; split seed remains 42",
            )
            sub.add_argument("--base-channels", type=positive_int, default=16)
            sub.add_argument("--refinement-steps", type=positive_int, default=3)
            sub.add_argument("--no-evidence", action="store_true")
            sub.add_argument("--no-refinement", action="store_true")
        if command in ("train", "evaluate"):
            sub.add_argument("--run-dir", required=True)
            sub.add_argument(
                "--data-dir",
                default=str(Path(__file__).resolve().parent / "data")
                if command == "train"
                else None,
            )
            sub.add_argument(
                "--batch-size",
                type=positive_int,
                default=64 if command == "train" else None,
            )
            sub.add_argument(
                "--workers", type=int, default=0 if command == "train" else None
            )
            sub.add_argument(
                "--max-eval-samples",
                type=positive_int,
                help="validation cap during train; test cap during evaluate",
            )
        if command == "train":
            sub.add_argument("--epochs", type=positive_int, default=8)
            sub.add_argument("--lr", type=float, default=1e-3)
            sub.add_argument("--patience", type=positive_int, default=20,
                             help="early stopping epochs without significant validation macro F1 gain")
            sub.add_argument("--min-delta", type=float, default=0.0005,
                             help="minimum validation macro F1 increase that resets patience")
            sub.add_argument("--lr-patience", type=positive_int, default=6,
                             help="ReduceLROnPlateau patience (validation macro F1)")
            sub.add_argument("--lr-factor", type=float, default=0.5,
                             help="multiply learning rate by this on a plateau")
            sub.add_argument("--min-lr", type=float, default=1e-5,
                             help="minimum learning rate")
            sub.add_argument("--max-train-samples", type=positive_int)
            sub.add_argument(
                "--thresholds",
                type=float,
                nargs="+",
                default=[i / 20 for i in range(1, 20)],
            )
        if command == "evaluate":
            sub.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    return root


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    if getattr(args, "workers", None) is not None and args.workers < 0:
        argument_parser.error("--workers cannot be negative")
    if args.command == "train" and (
        not math.isfinite(args.lr)
        or not args.lr > 0
        or not math.isfinite(args.min_delta)
        or args.min_delta < 0
        or not math.isfinite(args.min_lr)
        or not 0 <= args.min_lr < args.lr
        or not math.isfinite(args.lr_factor)
        or not 0 < args.lr_factor < 1
        or any(not math.isfinite(t) or not 0 < t < 1 for t in args.thresholds)
    ):
        argument_parser.error(
            "require lr > 0, 0 <= min-lr < lr, min-delta >= 0, "
            "0 < lr-factor < 1, and thresholds strictly between 0 and 1 "
            "(all values finite)"
        )
    torch.set_num_threads(args.threads)
    {"train": train, "evaluate": evaluate, "smoke": smoke}[args.command](args)


if __name__ == "__main__":
    main()
