"""Accuracy-first training on the original data and fixed baseline split.

Train with BF16 on CUDA; select raw/EMA weights and a threshold using validation
macro F1 in FP32. Test labels are only read by the separate evaluate command.
"""

import argparse
from contextlib import nullcontext
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
from torch.utils.data import random_split

if __package__:
    from . import experiment as base
    from .f1_model import F1UNet
    from .f1_metrics import assess
    from .model import UNet
else:
    import experiment as base
    from f1_model import F1UNet
    from f1_metrics import assess
    from model import UNet


SOURCE_NAMES = (
    "f1_experiment.py", "f1_model.py", "f1_metrics.py", "experiment.py",
    "model.py", "dataset.py",
)
THRESHOLDS = [i / 100 for i in range(5, 96)]


def source_fingerprints():
    folder = Path(__file__).resolve().parent
    return {name: base.file_sha256(folder / name) for name in SOURCE_NAMES}


def make_model(config):
    if config["model"] == "f1_unet":
        return F1UNet(base_channels=config["base_channels"])
    if config["model"] == "unet":
        return UNet()
    raise ValueError(f"Unknown model: {config['model']}")


def setup_device(name):
    device = base.choose_device(name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    # Use the same FP32 arithmetic settings for validation and test. No TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return device


def resolve_precision(requested, device):
    supported = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if requested == "auto":
        return "bf16" if supported else "fp32"
    if requested == "bf16" and not supported:
        raise ValueError("BF16 requires a supporting CUDA GPU. Use --precision fp32 on CPU.")
    return requested


def autocast_context(device, precision):
    return (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if precision == "bf16" else nullcontext()
    )


def segmentation_loss(logits, targets, bce_weight=0.5):
    # FP32 sigmoid, reductions and BCE even when convolutions run in BF16.
    logits, targets = logits.float(), targets.float()
    probabilities = logits.sigmoid().flatten(1)
    truth = targets.flatten(1)
    dice = (2 * (probabilities * truth).sum(1) + 1e-6) / (
        probabilities.sum(1) + truth.sum(1) + 1e-6
    )
    return bce_weight * torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets
    ) + 1 - dice.mean()


def supervised_loss(model, inputs, targets, config):
    if config["model"] == "f1_unet":
        logits, auxiliary = model(inputs, return_aux=True)
    else:
        logits, auxiliary = model(inputs), ()
    loss = segmentation_loss(logits, targets, config["bce_weight"])
    weight = config["aux_weight"]
    for head in auxiliary:
        loss = loss + weight * segmentation_loss(head, targets, config["bce_weight"])
    return loss / (1 + weight * len(auxiliary))


class ModelEMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0
        self.parameter_names = set(dict(self.model.named_parameters()))

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # Short initial averaging window avoids retaining random initialization.
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            if value.is_floating_point() and name in self.parameter_names:
                value.lerp_(source[name].detach(), 1 - decay)
            else:
                # In particular, copy normalization buffers/counters, not EMA them.
                value.copy_(source[name])


def learning_rate(step, total_steps, warmup_steps, peak, minimum_ratio):
    if step < warmup_steps:
        return peak * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)))
    return peak * (minimum_ratio + (1 - minimum_ratio) * (1 + math.cos(math.pi * progress)) / 2)


def select_validation(sweep):
    return max(sweep, key=lambda result: (
        result["overall"]["macro"]["f1"], -abs(result["threshold"] - 0.5),
        -result["threshold"],
    ))


def make_training_loader(dataset, indices, config, device):
    loader = base.make_loader(dataset, indices, config, device, training=True)
    # Persistent workers consume base seeds differently after process restart.
    # Keep shuffling on its own generator so that cannot alter sample order.
    loader.sampler.generator = torch.Generator()
    return loader


def rng_state(device):
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def verify_run(run_dir, checkpoint, data_dir=None):
    run_dir = Path(run_dir)
    config = checkpoint["config"]
    if config["source_sha256"] != source_fingerprints():
        raise ValueError("Source differs from training; restore UNet files from the run's source/ snapshot.")
    print("Verifying dataset fingerprint...", flush=True)
    dataset = base.load_dataset(data_dir or config["data_dir"])
    manifest = base.dataset_manifest(dataset)
    saved = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest != saved or manifest["sha256"] != checkpoint["manifest_sha256"]:
        raise ValueError("Dataset differs from training; refusing a changed holdout.")
    if base.file_sha256(run_dir / "splits.pt") != checkpoint["splits_sha256"]:
        raise ValueError("Saved split file has changed since training.")
    splits = torch.load(run_dir / "splits.pt", map_location="cpu", weights_only=True)
    return dataset, splits, manifest


def create_run(args, device):
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Use an empty run directory, or the resume command: {run_dir}")
    config = vars(args).copy()
    config.update(
        data_dir=str(Path(args.data_dir).resolve()), run_dir=str(run_dir),
        split_seed=base.SPLIT_SEED, source_sha256=source_fingerprints(),
        precision=resolve_precision(args.precision, device), inference_precision="fp32",
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        loss="bce_weight * BCE + per-image soft Dice; normalized auxiliary supervision",
        checkpoint_selection="maximum validation macro F1 across epochs, raw/EMA weights, thresholds",
        dependencies={name: importlib.metadata.version(name) for name in ("torch", "numpy", "h5py")},
    )
    dataset = base.load_dataset(config["data_dir"])
    n_train, n_val = int(0.6 * len(dataset)), int(0.2 * len(dataset))
    lengths = [n_train, n_val, len(dataset) - n_train - n_val]
    if min(lengths) <= 0:
        raise ValueError("Training, validation and test splits must all be nonempty.")
    subsets = random_split(dataset, lengths, generator=torch.Generator().manual_seed(base.SPLIT_SEED))
    splits = {name: subset.indices for name, subset in zip(("train", "val", "test"), subsets)}
    splits["used_train"] = base.limited(splits["train"], args.max_train_samples)
    splits["used_val"] = base.limited(splits["val"], args.max_eval_samples)
    print("Fingerprinting unchanged dataset files...", flush=True)
    manifest = base.dataset_manifest(dataset)
    run_dir.mkdir(parents=True, exist_ok=True)
    base.write_json(run_dir / "config.json", config)
    base.write_json(run_dir / "manifest.json", manifest)
    base.save_checkpoint(run_dir / "splits.pt", splits)
    snapshot = run_dir / "source"
    snapshot.mkdir()
    for name in SOURCE_NAMES:
        shutil.copy2(Path(__file__).resolve().parent / name, snapshot / name)
    return config, dataset, splits, manifest, None


def train(args):
    run_dir = Path(args.run_dir).resolve()
    device = setup_device(args.device)
    if args.command == "resume":
        state = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=True)
        config = state["config"].copy()
        dataset, splits, manifest = verify_run(run_dir, state, args.data_dir)
        resolve_precision(config["precision"], device)
        if args.workers is not None:
            config["workers"] = args.workers
    else:
        config, dataset, splits, manifest, state = create_run(args, device)
    base.seed_everything(config["seed"])
    model = make_model(config).to(device)
    ema = ModelEMA(model, config["ema_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    start_epoch, global_step, best_score, history = 1, 0, -1.0, []
    if state is not None:
        model.load_state_dict(state["model_state_dict"])
        ema.model.load_state_dict(state["ema_state_dict"])
        ema.updates = state["ema_updates"]
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch, global_step = state["epoch"] + 1, state["global_step"]
        best_score, history = state["best_score"], state["history"]
        if (run_dir / "best.pt").exists():
            saved_best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
            best_score = max(best_score, saved_best["validation"]["overall"]["macro"]["f1"])
        restore_rng(state["rng_state"], device)
    train_loader = make_training_loader(dataset, splits["used_train"], config, device)
    val_loader = base.make_loader(dataset, splits["used_val"], config, device)
    total_steps = config["epochs"] * len(train_loader)
    warmup_steps = config["warmup_epochs"] * len(train_loader)
    split_digest = base.file_sha256(run_dir / "splits.pt")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"{config['model']}: {parameters:,} parameters, {device}, training={config['precision']}, validation=fp32", flush=True)
    for epoch in range(start_epoch, config["epochs"] + 1):
        started = time.perf_counter()
        # Epoch-addressable shuffling makes an epoch-boundary resume reproducible.
        train_loader.sampler.generator.manual_seed(config["seed"] + epoch)
        model.train()
        total_loss = 0.0
        for batch_index, (inputs, targets, _) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            lr = learning_rate(global_step, total_steps, warmup_steps, config["lr"], config["min_lr_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config["precision"]):
                loss = supervised_loss(model, inputs, targets, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, batch {batch_index}.")
            # BF16 has FP32's exponent range; loss scaling is not required.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            ema.update(model)
            global_step += 1
            total_loss += float(loss.detach()) * inputs.size(0)
            if batch_index % config["log_every"] == 0:
                print(f"Epoch {epoch}/{config['epochs']} batch {batch_index}/{len(train_loader)} loss={float(loss.detach()):.5f}", flush=True)
        candidates = {}
        for kind, candidate in (("raw", model), ("ema", ema.model)):
            print(f"Epoch {epoch}: validating {kind} weights...", flush=True)
            candidates[kind] = select_validation(assess(candidate, val_loader, device, config["thresholds"]))
        # On an exact F1 tie, use raw weights; across epochs keep the earlier one.
        kind = max(candidates, key=lambda key: candidates[key]["overall"]["macro"]["f1"])
        validation = candidates[kind]
        score = validation["overall"]["macro"]["f1"]
        improved = score > best_score
        best_score = max(best_score, score)
        row = dict(epoch=epoch, train_loss=total_loss / len(splits["used_train"]),
                   lr=lr, validation=validation, weights_kind=kind,
                   candidates=candidates, seconds=time.perf_counter() - started)
        history.append(row)
        common = dict(epoch=epoch, config=config, validation=validation,
                      threshold=validation["threshold"], weights_kind=kind,
                      manifest_sha256=manifest["sha256"], splits_sha256=split_digest,
                      metric_conventions=base.METRIC_CONVENTIONS, parameters=parameters)
        if improved:
            selected = model if kind == "raw" else ema.model
            base.save_checkpoint(run_dir / "best.pt", {**common, "model_state_dict": selected.state_dict()})
        latest = dict(common, model_state_dict=model.state_dict(), ema_state_dict=ema.model.state_dict(),
                      ema_updates=ema.updates, optimizer_state_dict=optimizer.state_dict(),
                      global_step=global_step, best_score=best_score, history=history,
                      rng_state=rng_state(device))
        base.save_checkpoint(run_dir / "latest.pt", latest)
        base.write_json(run_dir / "history.json", history)
        print(f"Epoch {epoch}: val macro F1={score:.6f}, threshold={validation['threshold']:.2f}, weights={kind}, best={best_score:.6f}, seconds={row['seconds']:.1f}", flush=True)
    print(f"Saved {run_dir}. Test split remains unevaluated; use the evaluate command.", flush=True)


def evaluate(args):
    run_dir = Path(args.run_dir).resolve()
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    dataset, splits, manifest = verify_run(run_dir, checkpoint, args.data_dir)
    config = checkpoint["config"].copy()
    config["batch_size"] = args.batch_size or config["batch_size"]
    config["workers"] = config["workers"] if args.workers is None else args.workers
    device = setup_device(args.device)
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    indices = base.limited(splits["test"], args.max_eval_samples)
    result = assess(model, base.make_loader(dataset, indices, config, device), device, [checkpoint["threshold"]])[0]
    result.update(
        model=config["model"], epoch=checkpoint["epoch"], weights_kind=checkpoint["weights_kind"],
        checkpoint_sha256=base.file_sha256(run_dir / "best.pt"), manifest_sha256=manifest["sha256"],
        split="test", test_indices=indices, full_test_samples=len(splits["test"]),
        threshold_source="saved validation macro F1 selection", inference_precision="fp32",
        metric_conventions=base.METRIC_CONVENTIONS, device=str(device), parameters=checkpoint["parameters"],
    )
    output = run_dir / ("evaluation.json" if args.max_eval_samples is None else "evaluation_subset.json")
    base.write_json(output, result)
    print(json.dumps(result["overall"], indent=2), flush=True)
    print(f"Saved overall and per-SNR metrics to {output}", flush=True)


def smoke(args):
    device = setup_device(args.device)
    base.seed_everything(args.seed)
    config = vars(args).copy()
    config["precision"] = resolve_precision(args.precision, device)
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    inputs = torch.randint(0, 2, (2, 1, 50, 100), device=device).float()
    targets = torch.randint(0, 2, inputs.shape, device=device).float()
    with autocast_context(device, config["precision"]):
        loss = supervised_loss(model, inputs, targets, config)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    ema = ModelEMA(model, 0.999)
    ema.update(model)
    result = assess(ema.model, [(inputs.cpu(), targets.cpu(), torch.tensor([-20., 5.]))], device, THRESHOLDS)
    print(json.dumps(dict(model=args.model, device=str(device), precision=config["precision"],
                          parameters=sum(p.numel() for p in model.parameters()), loss=float(loss.detach()),
                          finite_gradients=True, validation_thresholds=len(result)), indent=2))


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("train", "resume", "evaluate", "smoke"):
        sub = commands.add_parser(command)
        sub.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N")
        sub.add_argument("--threads", type=base.positive_int, default=4)
        if command != "smoke":
            sub.add_argument("--run-dir", required=True)
            sub.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data") if command == "train" else None)
            sub.add_argument("--workers", type=int, default=0 if command == "train" else None)
        if command in ("train", "smoke"):
            sub.add_argument("--model", choices=("f1_unet", "unet"), default="f1_unet")
            sub.add_argument("--base-channels", type=base.positive_int, default=64)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
            sub.add_argument("--bce-weight", type=float, default=0.5)
            sub.add_argument("--aux-weight", type=float, default=0.25)
        if command in ("train", "evaluate"):
            sub.add_argument("--batch-size", type=base.positive_int, default=64 if command == "train" else None)
            sub.add_argument("--max-eval-samples", type=base.positive_int)
        if command == "train":
            sub.add_argument("--epochs", type=base.positive_int, default=100)
            sub.add_argument("--warmup-epochs", type=int, default=5)
            sub.add_argument("--lr", type=float, default=3e-4)
            sub.add_argument("--min-lr-ratio", type=float, default=0.01)
            sub.add_argument("--weight-decay", type=float, default=1e-4)
            sub.add_argument("--grad-clip", type=float, default=1.0)
            sub.add_argument("--ema-decay", type=float, default=0.999)
            sub.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
            sub.add_argument("--max-train-samples", type=base.positive_int)
            sub.add_argument("--log-every", type=base.positive_int, default=100)
    return root


def validate_args(args, argument_parser):
    if getattr(args, "workers", None) is not None and args.workers < 0:
        argument_parser.error("--workers must be nonnegative")
    if args.command in ("train", "smoke"):
        if args.model == "f1_unet" and args.base_channels < 4:
            argument_parser.error("--base-channels must be at least 4")
        if not math.isfinite(args.bce_weight) or args.bce_weight <= 0 or not math.isfinite(args.aux_weight) or args.aux_weight < 0:
            argument_parser.error("--bce-weight must be positive; --aux-weight must be nonnegative")
    if args.command == "train":
        if not 0 <= args.warmup_epochs < args.epochs:
            argument_parser.error("--warmup-epochs must be nonnegative and less than --epochs")
        for name in ("lr", "grad_clip"):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                argument_parser.error(f"--{name.replace('_', '-')} must be finite and positive")
        if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
            argument_parser.error("--weight-decay must be finite and nonnegative")
        if not 0 < args.min_lr_ratio <= 1 or not 0 <= args.ema_decay < 1:
            argument_parser.error("--min-lr-ratio must be in (0,1]; --ema-decay in [0,1)")
        if (any(not 0 < value < 1 for value in args.thresholds)
                or torch.tensor(args.thresholds, dtype=torch.float32).unique().numel() != len(args.thresholds)):
            argument_parser.error("--thresholds must be unique in FP32 and strictly between 0 and 1")


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    validate_args(args, argument_parser)
    torch.set_num_threads(args.threads)
    {"train": train, "resume": train, "evaluate": evaluate, "smoke": smoke}[args.command](args)


if __name__ == "__main__":
    main()
