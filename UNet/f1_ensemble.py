"""Optional validation-selected probability ensemble for F1UNet runs.

This is an exploratory accuracy option, not a guaranteed improvement. It uses
saved best checkpoints, selects one threshold and flip recipe on validation,
then evaluates that frozen recipe once on the held-out test split.
"""

import argparse
from pathlib import Path

import torch
from torch import nn

if __package__:
    from .experiment import (
        METRIC_CONVENTIONS,
        file_sha256,
        make_loader,
        positive_int,
        seed_everything,
        write_json,
    )
else:
    from experiment import (
        METRIC_CONVENTIONS,
        file_sha256,
        make_loader,
        positive_int,
        seed_everything,
        write_json,
    )


TTA_FLIPS = {
    "none": ((),),
    "time": ((), (-2,)),
    "both": ((), (-2,), (-1,), (-2, -1)),
}
THRESHOLDS = [index / 100 for index in range(5, 96)]
SPLIT_KEYS = ("train", "val", "test", "used_train", "used_val")


class ProbabilityEnsemble(nn.Module):
    """Average model/flip probabilities and expose compatible binary logits.

    Time is the penultimate image dimension; frequency is the last dimension.
    Every prediction is returned to its original orientation before averaging.
    Model weights and forward computations remain FP32 at evaluation time.
    """

    def __init__(self, models, tta="none"):
        super().__init__()
        if not models:
            raise ValueError("An ensemble needs at least one model.")
        if tta not in TTA_FLIPS:
            raise ValueError(f"Unknown TTA recipe: {tta}")
        self.models = nn.ModuleList(models)
        self.tta = tta

    def forward(self, inputs):
        probability_sum = None
        for model in self.models:
            for dimensions in TTA_FLIPS[self.tta]:
                transformed = inputs.flip(dimensions) if dimensions else inputs
                logits = model(transformed)
                if logits.shape != inputs.shape:
                    raise ValueError("Each model must return B1HW logits matching its input.")
                probabilities = logits.float().sigmoid()
                if dimensions:
                    probabilities = probabilities.flip(dimensions)
                probability_sum = (
                    probabilities
                    if probability_sum is None
                    else probability_sum + probabilities
                )
        average = probability_sum / (len(self.models) * len(TTA_FLIPS[self.tta]))
        # Avoid infinite logits for saturated predictions. All selectable
        # thresholds are well inside these endpoints.
        epsilon = torch.finfo(average.dtype).eps
        return torch.logit(average.clamp(epsilon, 1 - epsilon))


def validate_compatible_runs(splits, manifests):
    """Require identical data and full/effective splits, allowing matched caps."""
    if not splits or len(splits) != len(manifests):
        raise ValueError("Every run must provide splits and a dataset manifest.")
    reference = splits[0]
    for member in splits:
        for key in SPLIT_KEYS:
            if key not in member or not member[key]:
                raise ValueError(f"Missing or empty saved {key} indices.")
            if member[key] != reference[key]:
                raise ValueError(f"Ensemble members have different {key} indices.")
    if any(manifest != manifests[0] for manifest in manifests[1:]):
        raise ValueError("Ensemble members have different dataset manifests.")
    return {
        "identical_full_and_effective_splits": True,
        "samples": {key: len(reference[key]) for key in SPLIT_KEYS},
        "training_was_capped": reference["used_train"] != reference["train"],
        "validation_was_capped": reference["used_val"] != reference["val"],
    }


def select_recipe(models, validation_loader, device, thresholds, tta="auto", assess_fn=None):
    """Select exclusively from validation; this helper never receives test data."""
    if tta not in ("auto", *TTA_FLIPS):
        raise ValueError(f"Unknown TTA selection: {tta}")
    if assess_fn is None:
        if __package__:
            from .f1_metrics import assess as assess_fn
        else:
            from f1_metrics import assess as assess_fn
    recipes = tuple(TTA_FLIPS) if tta == "auto" else (tta,)
    candidates = []
    for recipe in recipes:
        ensemble = ProbabilityEnsemble(models, recipe).eval()
        sweep = assess_fn(ensemble, validation_loader, device, thresholds)
        best = max(
            sweep,
            key=lambda result: (
                result["overall"]["macro"]["f1"],
                -abs(result["threshold"] - 0.5),
                -result["threshold"],
            ),
        )
        candidates.append(
            {
                "tta": recipe,
                "forwards_per_image": len(models) * len(TTA_FLIPS[recipe]),
                "validation": best,
                "threshold_sweep": [
                    {
                        "threshold": result["threshold"],
                        "macro_f1": result["overall"]["macro"]["f1"],
                    }
                    for result in sweep
                ],
            }
        )
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["validation"]["overall"]["macro"]["f1"],
            -candidate["forwards_per_image"],
        ),
    )
    selection = {
        "tta": selected["tta"],
        "threshold": selected["validation"]["threshold"],
        "validation": selected["validation"],
        "forwards_per_image": selected["forwards_per_image"],
        "candidates": candidates,
        "selection_rule": (
            "Maximum validation macro F1; threshold ties closest to 0.5, then lower; "
            "recipe ties prefer fewer forward passes."
        ),
    }
    return ProbabilityEnsemble(models, selected["tta"]).eval(), selection


def run(args):
    if __package__:
        from .f1_experiment import make_model, setup_device, source_fingerprints, verify_run
        from .f1_metrics import assess
    else:
        from f1_experiment import make_model, setup_device, source_fingerprints, verify_run
        from f1_metrics import assess

    run_dirs = [Path(directory).resolve() for directory in args.run_dirs]
    if len(run_dirs) < 2 or len(set(run_dirs)) != len(run_dirs):
        raise ValueError("Provide at least two distinct training run directories.")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("The output directory must be empty.")
    device = setup_device(args.device)
    seed_everything(42)
    members, models, all_splits, manifests = [], [], [], []
    dataset = None
    for run_dir in run_dirs:
        checkpoint_path = run_dir / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        member_dataset, splits, manifest = verify_run(run_dir, checkpoint, args.data_dir)
        if dataset is None:
            dataset = member_dataset
        all_splits.append(splits)
        manifests.append(manifest)
        model = make_model(checkpoint["config"]).float().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)
        members.append(
            {
                "run_dir": str(run_dir),
                "checkpoint": "best.pt",
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "epoch": checkpoint["epoch"],
                "weights_kind": checkpoint["weights_kind"],
                "individual_validation_threshold": checkpoint["threshold"],
                "individual_validation_macro_f1": checkpoint["validation"]["overall"]["macro"]["f1"],
            }
        )
    compatibility = validate_compatible_runs(all_splits, manifests)
    splits = all_splits[0]
    loader_config = {"seed": 42, "batch_size": args.batch_size, "workers": args.workers}
    validation_loader = make_loader(dataset, splits["used_val"], loader_config, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Selecting ensemble threshold and flip recipe on validation...", flush=True)
    ensemble, selection = select_recipe(
        models, validation_loader, device, THRESHOLDS, args.tta, assess_fn=assess
    )
    metadata = {
        "model": "F1 experiment probability ensemble",
        "member_runs": members,
        "manifest_sha256": manifests[0]["sha256"],
        "split_compatibility": compatibility,
        "used_validation_indices": splits["used_val"],
        "metric_conventions": METRIC_CONVENTIONS,
        "inference_precision": "float32",
        "device": str(device),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "threads": args.threads,
        "source_sha256": {
            **source_fingerprints(),
            "f1_ensemble.py": file_sha256(Path(__file__).resolve()),
        },
    }
    selection.update(metadata)
    # Persist the choice before any test labels enter evaluation.
    write_json(output_dir / "selection.json", selection)
    print(
        f"Selected TTA={selection['tta']}, threshold={selection['threshold']:.2f}; "
        "evaluating the frozen recipe on the full test split...",
        flush=True,
    )
    test_loader = make_loader(dataset, splits["test"], loader_config, device)
    result = assess(ensemble, test_loader, device, [selection["threshold"]])[0]
    result.update(
        **metadata,
        split="test",
        test_indices=splits["test"],
        full_test_samples=len(splits["test"]),
        tta=selection["tta"],
        forwards_per_image=selection["forwards_per_image"],
        threshold_source="frozen validation selection in selection.json",
    )
    write_json(output_dir / "evaluation.json", result)
    print(f"Test macro F1: {result['overall']['macro']['f1']:.6f}", flush=True)
    print(f"Saved ensemble results to {output_dir}", flush=True)


def parser():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--run-dirs", nargs="+", required=True)
    argument_parser.add_argument("--output-dir", required=True)
    argument_parser.add_argument("--data-dir")
    argument_parser.add_argument("--device", default="auto")
    argument_parser.add_argument("--batch-size", type=positive_int, default=64)
    argument_parser.add_argument("--workers", type=int, default=0)
    argument_parser.add_argument("--threads", type=positive_int, default=4)
    argument_parser.add_argument("--tta", choices=("auto", "none", "time", "both"), default="auto")
    return argument_parser


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    if args.workers < 0:
        argument_parser.error("--workers cannot be negative")
    if len(args.run_dirs) < 2:
        argument_parser.error("--run-dirs needs at least two distinct run directories")
    torch.set_num_threads(args.threads)
    run(args)


if __name__ == "__main__":
    main()
