"""Matched FP32 CPU/GPU latency, including an equally BN-folded U-Net reference.

Random weights are used for screening; --fast-run-dir loads all three trained
models. Multiple shuffled rounds reduce (but do not eliminate) order effects.
No loading, device transfers or prediction decoding is included in timings.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import torch

if __package__:
    from .benchmark import count_convolution_macs, synchronize
    from .deployment import convert, load_training_model
    from .experiment import choose_device, make_model, positive_int
    from .fast_model import FastUNet
    from .native_backend import enable_native_batchnorm
    from .inference import prepare_cpu
else:
    from benchmark import count_convolution_macs, synchronize
    from deployment import convert, load_training_model
    from experiment import choose_device, make_model, positive_int
    from fast_model import FastUNet
    from native_backend import enable_native_batchnorm
    from inference import prepare_cpu


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_architecture_source(current, expected, snapshot):
    """Accept exact bytes or verified formatting-only changes to saved source."""
    if expected == _sha256(current):
        return "exact hash"
    if (expected is not None and snapshot.is_file() and _sha256(snapshot) == expected
            and ast.dump(ast.parse(current.read_text(encoding="utf-8-sig")))
            == ast.dump(ast.parse(snapshot.read_text(encoding="utf-8-sig")))):
        return "AST identical to hash-verified training snapshot (formatting only)"
    raise ValueError(f"Inference architecture differs from training: {current}")


def load_benchmark_model(run_dir):
    """Verify inference architecture, without requiring unchanged training code.

    This is deliberately limited to timing saved weights. Training/evaluation
    provenance guards in deployment.py remain stricter and unchanged.
    """
    path = Path(run_dir) / "best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    kind = config["model"]
    sources = {
        "unet": "UNet/model.py",
        "evidence_axial": "UNet/structured_model.py",
        "fast_unet": "UNet/fast_model.py",
        "fcnn": "baseline_recreation/model.py",
    }
    if kind not in sources:
        raise ValueError(f"Unsupported benchmark model: {kind}")
    relative = sources[kind]
    root = Path(__file__).resolve().parents[1]
    recorded = config.get("source_sha256", {})
    # Older segmentation checkpoints record names relative to UNet/.
    expected = recorded.get(relative, recorded.get(Path(relative).name))
    snapshot = Path(run_dir) / "source" / (relative if kind == "fcnn" else Path(relative).name)
    checkpoint["benchmark_source_check"] = verify_architecture_source(root / relative, expected, snapshot)
    if kind == "fcnn":
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from baseline_recreation.model import DLANNBaseline

        model = DLANNBaseline()
    elif kind == "fast_unet":
        model = FastUNet(
            base_channels=config["base_channels"],
            branches=not config["plain_blocks"],
        )
    else:
        model = make_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    if kind == "fcnn":
        model = torch.nn.Sequential(torch.nn.Flatten(1), model)
    return model.eval(), checkpoint


def count_linear_macs(model, inputs):
    """Count dense-layer MACs for a single example, excluding bias/activation."""
    counts, handles = [], []

    def count(layer, arguments, output):
        counts.append(output.numel() * layer.in_features)

    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            handles.append(layer.register_forward_hook(count))
    try:
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return sum(counts)


@torch.inference_mode()
def benchmark(args):
    fcnn_run_dir = getattr(args, "fcnn_run_dir", None)
    if fcnn_run_dir and not args.fast_run_dir:
        raise ValueError("--fcnn-run-dir requires --fast-run-dir for trained comparisons")
    torch.set_num_threads(args.threads)
    if args.native_batchnorm:
        enable_native_batchnorm()
    torch.manual_seed(42)
    device = choose_device(args.device)
    if args.optimize_cpu and device.type != "cpu":
        raise ValueError("--optimize-cpu requires --device cpu")
    inputs = torch.randint(0, 2, (args.batch_size, 1, 50, 100)).float().to(device)
    if args.channels_last:
        inputs = inputs.to(memory_format=torch.channels_last)
    config = dict(
        base_channels=16, refinement_steps=3, no_evidence=False, no_refinement=False
    )
    checkpoint_paths = {}
    if args.fast_run_dir:
        loader = load_benchmark_model if fcnn_run_dir else load_training_model
        unet, unet_checkpoint = loader(args.unet_run_dir)
        evidence, evidence_checkpoint = loader(args.evidence_run_dir)
        fast, checkpoint = loader(args.fast_run_dir)
        width = checkpoint["config"]["base_channels"]
        models = {
            "unet": unet,
            "unet_folded": convert(unet),
            "evidence_axial": evidence,
            f"fast_unet_b{width}_deploy": convert(fast),
        }
        checkpoint_paths = {
            "unet": Path(args.unet_run_dir) / "best.pt",
            "unet_folded": Path(args.unet_run_dir) / "best.pt",
            "evidence_axial": Path(args.evidence_run_dir) / "best.pt",
            f"fast_unet_b{width}_deploy": Path(args.fast_run_dir) / "best.pt",
        }
        if fcnn_run_dir:
            models["fcnn"], fcnn_checkpoint = load_benchmark_model(fcnn_run_dir)
            if fcnn_checkpoint["config"]["model"] != "fcnn":
                raise ValueError("--fcnn-run-dir must contain an FCNN checkpoint")
            checkpoint_paths["fcnn"] = Path(fcnn_run_dir) / "best.pt"
            source_checks = {
                "unet": unet_checkpoint["benchmark_source_check"],
                "unet_folded": unet_checkpoint["benchmark_source_check"],
                "evidence_axial": evidence_checkpoint["benchmark_source_check"],
                f"fast_unet_b{width}_deploy": checkpoint["benchmark_source_check"],
                "fcnn": fcnn_checkpoint["benchmark_source_check"],
            }
    else:
        unet = make_model(dict(config, model="unet")).eval()
        models = {
            "unet": unet,
            "unet_folded": convert(unet),
            "evidence_axial": make_model(dict(config, model="evidence_axial")).eval(),
        }
        for width in args.widths:
            models[f"fast_unet_b{width}_deploy"] = (
                FastUNet(base_channels=width).eval().to_deploy()
            )
    samples = {name: [] for name in models}
    records = {}
    for name, model in models.items():
        model.to(device).eval()
        if args.channels_last:
            model.to(memory_format=torch.channels_last)
        records[name] = dict(
            model=name,
            parameters=sum(p.numel() for p in model.parameters()),
            convolution_macs_per_image=count_convolution_macs(model, inputs[:1]),
            rounds=[],
        )
        if fcnn_run_dir:
            linear_macs = count_linear_macs(model, inputs[:1])
            records[name].update(
                linear_macs_per_image=linear_macs,
                counted_macs_per_image=records[name]["convolution_macs_per_image"]
                + linear_macs,
                checkpoint_sha256=_sha256(checkpoint_paths[name]),
                architecture_source_check=source_checks[name],
            )
        if args.optimize_cpu:
            prepared = prepare_cpu(model, inputs[:1])
            torch.testing.assert_close(
                prepared(inputs), model(inputs), rtol=1e-4, atol=1e-4
            )
            models[name] = prepared
    order_rng = random.Random(42)
    for round_index in range(args.rounds):
        order = list(models)
        order_rng.shuffle(order)
        for name in order:
            model = models[name]
            for _ in range(args.warmup):
                model(inputs)
            synchronize(device)
            elapsed = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                model(inputs)
                synchronize(device)
                elapsed.append(1000 * (time.perf_counter() - start))
            samples[name].extend(elapsed)
            records[name]["rounds"].append(
                dict(round=round_index + 1, median_batch_ms=statistics.median(elapsed))
            )
        print(f"Finished timing round {round_index + 1}/{args.rounds}", flush=True)
    for name, values in samples.items():
        records[name].update(
            median_batch_ms=statistics.median(values),
            p95_batch_ms=sorted(values)[math.ceil(0.95 * len(values)) - 1],
            mean_batch_ms=statistics.mean(values),
            amortized_mean_ms_per_image=statistics.mean(values) / args.batch_size,
        )
    result = dict(
        device=str(device),
        device_name=torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        torch_version=str(torch.__version__),
        dtype="float32",
        cpu_threads=args.threads,
        batch_size=args.batch_size,
        input_shape=list(inputs.shape),
        warmup_per_round=args.warmup,
        iterations_per_round=args.iterations,
        rounds=args.rounds,
        trained_weights=bool(args.fast_run_dir),
        run_directories=dict(
            fast=args.fast_run_dir,
            unet=args.unet_run_dir,
            evidence=args.evidence_run_dir,
        ),
        native_batchnorm_only=args.native_batchnorm,
        records=list(records.values()),
        memory_format="channels_last" if args.channels_last else "contiguous",
        optimize_cpu=args.optimize_cpu,
        limitations=[
            "Synchronized host-wall model forward; excludes loading, transfers and decoding.",
            "Same FP32 policy; CPU inference graph optimization only when requested, for every model.",
            "Convolution MACs exclude other operations and memory traffic.",
        ],
    )
    if fcnn_run_dir:
        result["run_directories"]["fcnn"] = str(fcnn_run_dir)
        result["source_verification"] = (
            "Checkpoint architecture source hashes verified; training-runner changes "
            "are permitted for this inference-only benchmark."
        )
        result["limitations"].append(
            "Counted MACs include convolution and linear layers, not other operations. "
            "FCNN returns boxes; segmentation models return masks. Decoding excluded."
        )
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--widths", nargs="+", type=positive_int, default=[8, 12, 16])
    parser.add_argument("--warmup", type=positive_int, default=30)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--output")
    parser.add_argument("--native-batchnorm", action="store_true")
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Use the same channels-last layout for every model",
    )
    parser.add_argument(
        "--optimize-cpu",
        action="store_true",
        help="Prepare the same optimized CPU inference graph for every model",
    )
    parser.add_argument("--fast-run-dir")
    parser.add_argument("--fcnn-run-dir", help="Include trained FCNN in the same timing rounds")
    parser.add_argument("--unet-run-dir", default="runs/unet_bn_native_seed42")
    parser.add_argument("--evidence-run-dir", default="runs/evidence_axial_seed42")
    benchmark(parser.parse_args())
