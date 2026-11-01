"""FCNN comparison against the saved eight-epoch U-Net experiments.

Commands: check, train, evaluate, benchmark. See README.md for copyable commands.
Data generation and the historical FCNN implementation are unchanged.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys
import time

import h5py
import torch
from torch import nn
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from baseline_recreation.model import DLANNBaseline
from UNet import experiment as common

DEFAULT_REFERENCE = ROOT / "runs/unet_bn_native_seed42"
DEFAULT_RUN = ROOT / "runs/fcnn_matched_seed42"
METRICS = {
    **common.METRIC_CONVENTIONS,
    "prediction": "union of rounded, clipped normalized xywh boxes; target is saved clean_imgs",
    "threshold_selection": "none: outputs are coordinates, not pixel probabilities",
}


def normalize_boxes(boxes):
    """MATLAB 1-based xywh -> normalized coordinates; preserve zero padding."""
    boxes = torch.as_tensor(boxes, dtype=torch.float32).clone()
    if boxes.shape != (20, 4) or not torch.isfinite(boxes).all():
        raise ValueError("Expected 20 finite xywh boxes")
    valid = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    if torch.any(boxes[~valid] != 0):
        raise ValueError("Invalid box padding: absent boxes must be all zero")
    boxes[valid, :2] -= 1
    boxes /= boxes.new_tensor([100, 50, 100, 50])
    if torch.any((boxes < 0) | (boxes > 1)):
        raise ValueError("Box coordinates outside normalized range")
    return boxes.flatten()


def boxes_to_masks(boxes):
    """Decode Bx80 normalized xywh exactly as the historical rounded-box rule.

    Preserve float32 arithmetic and ties-to-even rounding from the old evaluator.
    Box-slot padding has zero area; no score threshold or test-tuned filtering.
    """
    boxes = boxes.detach().to(device="cpu", dtype=torch.float32).reshape(-1, 20, 4)
    if not torch.isfinite(boxes).all():
        raise FloatingPointError("Non-finite predicted boxes")
    start = torch.round(boxes[..., :2] * torch.tensor([100, 50]))
    end = torch.round((boxes[..., :2] + boxes[..., 2:]) * torch.tensor([100, 50]))
    coordinates = torch.cat((start, end), dim=-1).to(torch.int64).tolist()
    valid = (boxes[..., 2:] > 0).all(-1).tolist()
    masks = torch.zeros((len(boxes), 50, 100), dtype=torch.bool)
    for sample in range(len(boxes)):
        for (x1, y1, x2, y2), present in zip(coordinates[sample], valid[sample]):
            if not present:
                continue
            x1, y1 = max(0, min(100, x1)), max(0, min(50, y1))
            x2, y2 = max(0, min(100, x2)), max(0, min(50, y2))
            if x2 > x1 and y2 > y1:
                masks[sample, y1:y2, x1:x2] = True
    return masks


class MatchedDataset(Dataset):
    """Same sorted chunk/sample order as UNetSignalDataset, with both targets."""

    def __init__(self, directory):
        self.files = sorted(Path(directory).glob("*.mat"))
        if not self.files:
            raise ValueError(f"No MATLAB chunks in {directory}")
        self.index = []
        for file_index, path in enumerate(self.files):
            with h5py.File(path, "r") as chunk:
                n = chunk["noisy_imgs"].shape[-1]
                if (chunk["noisy_imgs"].shape != (100, 50, n)
                        or chunk["clean_imgs"].shape != (100, 50, n)
                        or chunk["labels"].shape != (4, 20, n)
                        or chunk["SNR_dB_all"].size != n):
                    raise ValueError(f"Unexpected dataset shape in {path}")
            self.index.extend((file_index, i) for i in range(n))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        file_index, sample = self.index[index]
        with h5py.File(self.files[file_index], "r") as chunk:
            inputs = torch.tensor(chunk["noisy_imgs"][:, :, sample].T.copy(), dtype=torch.float32)
            clean = torch.tensor(chunk["clean_imgs"][:, :, sample].T.copy()) > 0
            boxes = normalize_boxes(chunk["labels"][:, :, sample].T.copy())
            snrs = chunk["SNR_dB_all"]
            snr = float(snrs[0, sample] if snrs.shape[0] == 1 else snrs[sample, 0])
        return inputs.flatten(), boxes, clean, torch.tensor(snr, dtype=torch.float32)


def validate_splits(splits, samples):
    all_indices = [i for key in ("train", "val", "test") for i in splits[key]]
    if sorted(all_indices) != list(range(samples)) or any(not splits[k] for k in ("train", "val", "test")):
        raise ValueError("Splits must form a disjoint, complete, non-empty partition")
    for full, used in (("train", "used_train"), ("val", "used_val")):
        if splits.get(used, splits[full]) != splits[full]:
            raise ValueError("Reference run used capped data; expected full historical splits")


def load_reference(reference, data_dir):
    reference = Path(reference).resolve()
    config = json.loads((reference / "config.json").read_text())
    dataset = MatchedDataset(data_dir or config["data_dir"])
    print("Checking dataset hashes and saved split IDs...", flush=True)
    manifest = common.dataset_manifest(dataset)
    saved_manifest = json.loads((reference / "manifest.json").read_text())
    checkpoint = torch.load(reference / "best.pt", weights_only=True, map_location="cpu")
    if manifest != saved_manifest or manifest["sha256"] != checkpoint["manifest_sha256"]:
        raise ValueError("Dataset differs from the saved reference experiment")
    if common.file_sha256(reference / "splits.pt") != checkpoint["splits_sha256"]:
        raise ValueError("Reference split file changed since training")
    splits = torch.load(reference / "splits.pt", weights_only=True, map_location="cpu")
    validate_splits(splits, len(dataset))
    return dataset, manifest, splits, config


def fingerprints():
    return {p: common.file_sha256(ROOT / p) for p in (
        "baseline_recreation/matched_experiment.py", "baseline_recreation/model.py",
        "UNet/experiment.py",
    )}


@torch.inference_mode()
def assess(model, loader, device):
    model.eval()
    overall, per_snr = common.SegmentationMetrics(), defaultdict(common.SegmentationMetrics)
    loss_sum = 0.0
    for inputs, boxes, clean, snrs in loader:
        predictions = model(inputs.to(device, non_blocking=True))
        loss = nn.functional.mse_loss(predictions, boxes.to(device, non_blocking=True))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation/evaluation loss")
        loss_sum += float(loss) * len(inputs)
        masks = boxes_to_masks(predictions)
        truth = clean.bool()
        counts = torch.stack(((masks & truth).flatten(1).sum(1),
                              (masks & ~truth).flatten(1).sum(1),
                              (~masks & truth).flatten(1).sum(1)), dim=1).tolist()
        for (tp, fp, fn), snr in zip(counts, snrs.tolist()):
            overall.add(tp, fp, fn)
            per_snr[f"{snr:g}"].add(tp, fp, fn)
    return {"loss": loss_sum / overall.samples, "overall": overall.result(),
            "per_snr": {k: v.result() for k, v in sorted(per_snr.items(), key=lambda kv: float(kv[0]))}}


def check(args):
    dataset, _, splits, _ = load_reference(args.reference_run, args.data_dir)
    # Only training/validation samples; checking does not score the holdout.
    selected = []
    seen_chunks = set()
    for index in splits["train"]:
        chunk = dataset.index[index][0]
        if chunk not in seen_chunks:
            selected.append(index)
            seen_chunks.add(chunk)
    for index in selected:
        inputs, boxes, clean, _ = dataset[index]
        if not torch.equal(boxes_to_masks(boxes[None])[0], clean):
            raise ValueError(f"Box/clean-mask mismatch at sample {index}")
    common.seed_everything(42)
    device = common.choose_device(args.device)
    model = DLANNBaseline().to(device)
    x = torch.stack([dataset[i][0] for i in selected[:2]]).to(device)
    y = torch.stack([dataset[i][1] for i in selected[:2]]).to(device)
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
    print(f"PASS: {len(dataset):,} images; "
          f"{len(splits['train']):,}/{len(splits['val']):,}/{len(splits['test']):,} split. "
          f"Box/mask checks across {len(selected)} chunks; finite gradients on {device}.")


def train(args):
    run = Path(args.run_dir).resolve()
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run}; choose a new --run-dir")
    dataset, manifest, splits, ref_config = load_reference(args.reference_run, args.data_dir)
    reference = Path(args.reference_run).resolve()
    config = vars(args).copy()
    config.update(model="fcnn", run_dir=str(run), data_dir=str(dataset.files[0].parent.resolve()),
                  reference_run=str(reference), loss="MSE on 20 normalized xywh boxes, including padding",
                  checkpoint_selection="minimum validation MSE; no scheduler or early stopping",
                  source_sha256=fingerprints(), torch_version=str(torch.__version__),
                  reference_checkpoint_sha256=common.file_sha256(reference / "best.pt"),
                  reference_config_sha256=common.file_sha256(reference / "config.json"),
                  reference_splits_sha256=common.file_sha256(reference / "splits.pt"),
                  reference_settings={k: ref_config[k] for k in ("epochs", "batch_size", "lr", "seed")})
    # Caps are for smoke/integration runs and are labelled explicitly in outputs.
    splits = {k: list(v) for k, v in splits.items()}
    splits["used_train"] = common.limited(splits["train"], args.max_train_samples)
    splits["used_val"] = common.limited(splits["val"], args.max_eval_samples)
    config["full_data_run"] = not (args.max_train_samples or args.max_eval_samples)
    config["matches_reference_budget"] = config["full_data_run"] and all(
        config[k] == ref_config[k] for k in ("epochs", "batch_size", "lr", "seed"))
    common.seed_everything(args.seed)
    device = common.choose_device(args.device)
    config.update(resolved_device=str(device), device_name=torch.cuda.get_device_name(device)
                  if device.type == "cuda" else "CPU")
    run.mkdir(parents=True, exist_ok=True)
    common.write_json(run / "config.json", config)
    common.write_json(run / "manifest.json", manifest)
    common.save_checkpoint(run / "splits.pt", splits)
    for relative in config["source_sha256"]:
        target = run / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    split_digest = common.file_sha256(run / "splits.pt")
    train_loader = common.make_loader(dataset, splits["used_train"], config, device, training=True)
    val_loader = common.make_loader(dataset, splits["used_val"], config, device)
    model = DLANNBaseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss, best_epoch, history = float("inf"), None, []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        total_loss = 0.0
        for inputs, boxes, _clean, _snrs in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(inputs.to(device, non_blocking=True)),
                                          boxes.to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(inputs)
        validation = assess(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / len(splits["used_train"]),
               "validation": validation, "seconds": time.perf_counter() - start}
        history.append(row)
        checkpoint = {"epoch": epoch, "config": config, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "validation": validation,
                      "manifest_sha256": manifest["sha256"], "splits_sha256": split_digest,
                      "parameters": sum(p.numel() for p in model.parameters()), "metric_conventions": METRICS}
        common.save_checkpoint(run / "latest.pt", checkpoint)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            common.save_checkpoint(run / "best.pt", checkpoint)
        common.write_json(run / "history.json", history)
        print(f"Epoch {epoch}/{args.epochs} | train MSE={row['train_loss']:.6f} | "
              f"val MSE={validation['loss']:.6f} | val macro F1={validation['overall']['macro']['f1']:.4f} | "
              f"{row['seconds']:.1f}s", flush=True)
    common.write_json(run / "summary.json", {"best_epoch": best_epoch, "best_validation_mse": best_loss,
                      "epochs_completed": len(history), "matches_reference_budget": config["matches_reference_budget"],
                      "test_evaluated": False})
    print(f"Saved to {run}. Best validation MSE at epoch {best_epoch}. Run evaluate separately for test scores.")


def evaluate(args):
    run = Path(args.run_dir).resolve()
    checkpoint = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
    config = checkpoint["config"].copy()
    if config["source_sha256"] != fingerprints():
        raise ValueError("Source changed since training; restore the snapshots saved in this run")
    if common.file_sha256(run / "splits.pt") != checkpoint["splits_sha256"]:
        raise ValueError("Saved split changed since training")
    dataset = MatchedDataset(args.data_dir or config["data_dir"])
    manifest = common.dataset_manifest(dataset)
    if (manifest != json.loads((run / "manifest.json").read_text())
            or manifest["sha256"] != checkpoint["manifest_sha256"]):
        raise ValueError("Dataset changed since training")
    splits = torch.load(run / "splits.pt", weights_only=True, map_location="cpu")
    indices = common.limited(splits["test"], args.max_eval_samples)
    config.update(batch_size=args.batch_size, workers=args.workers)
    device = common.choose_device(args.device)
    model = DLANNBaseline().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = assess(model, common.make_loader(dataset, indices, config, device), device)
    result.update(model="fcnn", epoch=checkpoint["epoch"], split="test", test_indices=indices,
                  full_test_samples=len(splits["test"]), manifest_sha256=manifest["sha256"],
                  checkpoint_sha256=common.file_sha256(run / "best.pt"), metric_conventions=METRICS,
                  parameters=checkpoint["parameters"], device=str(device), batch_size=args.batch_size,
                  matches_reference_budget=config["matches_reference_budget"])
    name = "evaluation_subset.json" if args.max_eval_samples else "evaluation.json"
    common.write_json(run / name, result)
    print("SNR (dB)   Macro F1   Macro IoU")
    for snr, score in result["per_snr"].items():
        print(f"{float(snr):+8.0f}   {score['macro']['f1']:.6f}   {score['macro']['iou']:.6f}")
    print(f"Overall macro F1: {result['overall']['macro']['f1']:.6f}. Saved {run / name}")


def benchmark(args):
    from UNet.speed_benchmark import benchmark as run_benchmark
    run = Path(args.run_dir).resolve()
    # Re-benchmark every model together; do not mix these times with old sessions.
    options = argparse.Namespace(
        device=args.device, threads=args.threads, batch_size=args.batch_size,
        fcnn_run_dir=str(run), fast_run_dir=str(ROOT / "runs/fast_unet_b8_seed42"),
        unet_run_dir=str(DEFAULT_REFERENCE), evidence_run_dir=str(ROOT / "runs/evidence_axial_seed42"),
        native_batchnorm=True, optimize_cpu=args.optimize_cpu, channels_last=args.channels_last,
        rounds=3, warmup=30, iterations=100, widths=[8],
        output=str(run / f"benchmark_{args.device.replace(':', '_')}_b{args.batch_size}"
                   f"_{'prepared' if args.optimize_cpu else 'channels_last' if args.channels_last else 'eager'}.json"),
    )
    run_benchmark(options)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("check", "train", "evaluate", "benchmark"):
        sub = commands.add_parser(command)
        sub.add_argument("--device", default="cpu" if command == "benchmark" else "auto")
        sub.add_argument("--threads", type=common.positive_int, default=4)
        if command != "check":
            sub.add_argument("--run-dir", default=str(DEFAULT_RUN))
            sub.add_argument("--batch-size", type=common.positive_int, default=1 if command == "benchmark" else 64)
        if command != "benchmark":
            sub.add_argument("--data-dir", help="Defaults to the recorded reference/training data directory")
            if command != "evaluate":
                sub.add_argument("--reference-run", default=str(DEFAULT_REFERENCE))
        if command in ("train", "evaluate"):
            sub.add_argument("--workers", type=int, default=4)
            sub.add_argument("--max-eval-samples", type=common.positive_int, help="Smoke tests only: cap validation or test samples")
        if command == "train":
            sub.add_argument("--epochs", type=common.positive_int, default=8)
            sub.add_argument("--lr", type=float, default=0.001)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--max-train-samples", type=common.positive_int, help="Smoke tests only")
        if command == "benchmark":
            sub.add_argument("--optimize-cpu", action="store_true")
            sub.add_argument("--channels-last", action="store_true")
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    if getattr(args, "workers", 0) < 0 or not (0 < getattr(args, "lr", 0.001) < float("inf")):
        raise ValueError("Workers must be non-negative and learning rate positive and finite")
    torch.set_num_threads(args.threads)
    {"check": check, "train": train, "evaluate": evaluate, "benchmark": benchmark}[args.command](args)
