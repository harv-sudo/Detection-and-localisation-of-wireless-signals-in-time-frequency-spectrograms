"""Train FastUNet with the frozen baseline runner's exact protocol.

Only the model factory and source manifest are replaced for the duration of
the call. This keeps the original experiment.py and its checkpoint fingerprints
intact. The scoped hooks are intended for one command per process.
"""
import argparse
from contextlib import contextmanager
import math
from pathlib import Path
import shutil

import torch

if __package__:
    from . import experiment
    from .fast_model import FastUNet
    from .native_backend import enable_native_batchnorm
else:
    import experiment
    from fast_model import FastUNet
    from native_backend import enable_native_batchnorm


SOURCE_NAMES = ("experiment.py", "model.py", "structured_model.py", "dataset.py",
                "fast_model.py", "fast_experiment.py", "native_backend.py")


def source_fingerprints():
    folder = Path(__file__).resolve().parent
    return {name: experiment.file_sha256(folder / name) for name in SOURCE_NAMES}


def make_model(config):
    if config["model"] != "fast_unet":
        raise ValueError("This runner accepts only fast_unet checkpoints.")
    return FastUNet(base_channels=config["base_channels"], branches=not config["plain_blocks"])


@contextmanager
def model_support():
    previous_factory, previous_sources = experiment.make_model, experiment.source_fingerprints
    experiment.make_model, experiment.source_fingerprints = make_model, source_fingerprints
    try:
        yield
    finally:
        experiment.make_model, experiment.source_fingerprints = previous_factory, previous_sources


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate", "smoke"):
        sub = commands.add_parser(command)
        sub.add_argument("--device", default="auto")
        sub.add_argument("--threads", type=experiment.positive_int, default=4)
        sub.add_argument("--native-batchnorm", action="store_true",
                         help="Installed Windows ROCm BatchNorm compiler workaround")
        if command in ("train", "smoke"):
            sub.set_defaults(model="fast_unet")
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--base-channels", type=experiment.positive_int, default=8)
            sub.add_argument("--plain-blocks", action="store_true", help="Train without auxiliary branches")
        if command in ("train", "evaluate"):
            sub.add_argument("--run-dir", required=True)
            sub.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data") if command == "train" else None)
            sub.add_argument("--batch-size", type=experiment.positive_int, default=64 if command == "train" else None)
            sub.add_argument("--workers", type=int, default=0 if command == "train" else None)
            sub.add_argument("--max-eval-samples", type=experiment.positive_int)
        if command == "train":
            sub.add_argument("--epochs", type=experiment.positive_int, default=8)
            sub.add_argument("--lr", type=float, default=0.001)
            sub.add_argument("--max-train-samples", type=experiment.positive_int)
            sub.add_argument("--thresholds", type=float, nargs="+", default=[i / 20 for i in range(1, 20)])
        if command == "evaluate":
            sub.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    return root


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    if getattr(args, "workers", None) is not None and args.workers < 0:
        argument_parser.error("--workers cannot be negative")
    if args.command == "train" and (not math.isfinite(args.lr) or args.lr <= 0 or any(not 0 < t < 1 for t in args.thresholds)):
        argument_parser.error("--lr must be positive and thresholds must lie between 0 and 1")
    torch.set_num_threads(args.threads)
    if args.native_batchnorm:
        enable_native_batchnorm()
    with model_support():
        {"train": experiment.train, "evaluate": experiment.evaluate, "smoke": experiment.smoke}[args.command](args)
    if args.command == "train":
        destination = Path(args.run_dir) / "source"
        destination.mkdir(exist_ok=True)
        for name in SOURCE_NAMES:
            shutil.copy2(Path(__file__).resolve().parent / name, destination / name)


if __name__ == "__main__":
    main()
