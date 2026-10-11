"""Assemble the completed speed experiment, checking its comparison identities."""

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review_artifacts"
RUNS = {
    "unet": ROOT / "runs/unet_bn_native_seed42",
    "evidence_axial": ROOT / "runs/evidence_axial_seed42",
    "fast_unet_b8_deploy": ROOT / "runs/fast_unet_b8_seed42",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    evaluations = {
        name: read(
            folder
            / (
                "deployment_evaluation.json"
                if name.startswith("fast")
                else "evaluation.json"
            )
        )
        for name, folder in RUNS.items()
    }
    evaluations["unet_folded"] = read(RUNS["unet"] / "deployment_evaluation.json")
    cpu_evaluation = read(
        RUNS["fast_unet_b8_deploy"] / "deployment_evaluation_cpu_optimized.json"
    )
    unet_cpu_evaluation = read(
        RUNS["unet"] / "deployment_evaluation_cpu_optimized.json"
    )
    baseline = evaluations["unet"]
    baseline_splits = torch.load(RUNS["unet"] / "splits.pt", weights_only=True)
    for name, folder in RUNS.items():
        evaluation = evaluations[name]
        assert evaluation["test_indices"] == baseline["test_indices"], (
            f"Test IDs differ: {name}"
        )
        assert evaluation["manifest_sha256"] == baseline["manifest_sha256"], (
            f"Dataset differs: {name}"
        )
        splits = torch.load(folder / "splits.pt", weights_only=True)
        for split in ("train", "val", "test", "used_train", "used_val"):
            assert splits[split] == baseline_splits[split], (
                f"Split differs: {name}/{split}"
            )
    assert cpu_evaluation["test_indices"] == baseline["test_indices"]
    benchmark_files = dict(
        cpu="fast_cpu_optimized.json",
        cpu_eager="fast_cpu_eager.json",
        cpu_channels_last="fast_cpu_channels_last.json",
        gpu1="fast_gpu_batch1.json",
        gpu64="fast_gpu_batch64.json",
    )
    benchmarks = {
        name: read(OUT / filename) for name, filename in benchmark_files.items()
    }
    timing = {
        key: {row["model"]: row for row in value["records"]}
        for key, value in benchmarks.items()
    }
    rows = {}
    for name in ("unet", "unet_folded", "evidence_axial", "fast_unet_b8_deploy"):
        rows[name] = dict(
            metrics=evaluations[name]["overall"],
            per_snr=evaluations[name]["per_snr"],
            parameters=timing["gpu1"][name]["parameters"],
            convolution_macs=timing["gpu1"][name]["convolution_macs_per_image"],
            latency={key: timing[key][name] for key in timing},
        )
        rows[name]["best_cpu_mode"] = min(
            ("cpu", "cpu_eager", "cpu_channels_last"),
            key=lambda key: timing[key][name]["median_batch_ms"],
        )
        rows[name]["best_cpu_latency"] = timing[rows[name]["best_cpu_mode"]][name]
    fast = rows["fast_unet_b8_deploy"]
    differences = {
        name: dict(
            macro_f1_percentage_points=100
            * (fast["metrics"]["macro"]["f1"] - row["metrics"]["macro"]["f1"]),
            cpu_speedup=row["best_cpu_latency"]["median_batch_ms"]
            / fast["best_cpu_latency"]["median_batch_ms"],
            gpu_batch1_speedup=row["latency"]["gpu1"]["median_batch_ms"]
            / fast["latency"]["gpu1"]["median_batch_ms"],
            gpu_batch64_speedup=row["latency"]["gpu64"]["median_batch_ms"]
            / fast["latency"]["gpu64"]["median_batch_ms"],
        )
        for name, row in rows.items()
        if name != "fast_unet_b8_deploy"
    }
    result = dict(
        rows=rows,
        differences=differences,
        manifest_sha256=baseline["manifest_sha256"],
        split_sizes={
            name: len(baseline_splits[name]) for name in ("train", "val", "test")
        },
        benchmarks=benchmarks,
        conversion_checks=dict(
            fast_gpu=evaluations["fast_unet_b8_deploy"]["conversion_check"],
            fast_cpu=cpu_evaluation["conversion_check"],
            unet_cpu=unet_cpu_evaluation["conversion_check"],
            unet_gpu=evaluations["unet_folded"]["conversion_check"],
        ),
        fast_cpu_metrics=cpu_evaluation["overall"],
        unet_cpu_metrics=unet_cpu_evaluation["overall"],
        fast_config=read(RUNS["fast_unet_b8_deploy"] / "config.json"),
    )
    (OUT / "fast_model_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    labels = dict(
        unet="Original U-Net",
        unet_folded="U-Net, BN folded",
        evidence_axial="EvidenceAxialNet",
        fast_unet_b8_deploy="FastUNet, deployed",
    )
    lines = [
        "FastUNet speed experiment — 11 October 2026",
        "",
        "All accuracy scores below use the same 20,800 test images and saved validation thresholds. "
        "Training, validation and test indices and dataset hashes were checked equal. CPU timings select "
        "each model's lowest median among three measured modes: eager, channels-last eager, and inference-graph preparation. "
        "All three modes are offered to every model; GPU timings use eager execution with "
        "native BatchNorm where present. All computations are FP32.",
        "",
        "| Model | Test macro F1 | Test macro IoU | Parameters | CPU batch 1 | GPU batch 1 | GPU batch 64 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        ms = [
            row["best_cpu_latency"]["median_batch_ms"],
            row["latency"]["gpu1"]["median_batch_ms"],
            row["latency"]["gpu64"]["median_batch_ms"],
        ]
        lines.append(
            f"| {labels[name]} | {row['metrics']['macro']['f1']:.6f} | {row['metrics']['macro']['iou']:.6f} | "
            f"{row['parameters']:,} | {ms[0]:.3f} ms | {ms[1]:.3f} ms | {ms[2]:.3f} ms |"
        )
    lines += [
        "",
        "Selected CPU modes: "
        + "; ".join(
            f"{labels[name]}: {row['best_cpu_mode']}" for name, row in rows.items()
        )
        + ".",
    ]
    lines += ["", "Speed ratios are reference latency divided by FastUNet latency:", ""]
    for name, difference in differences.items():
        lines.append(
            f"- Versus {labels[name]}: {difference['cpu_speedup']:.2f}x CPU, "
            f"{difference['gpu_batch1_speedup']:.2f}x GPU batch 1, "
            f"{difference['gpu_batch64_speedup']:.2f}x GPU batch 64; "
            f"macro F1 difference {difference['macro_f1_percentage_points']:+.3f} percentage points."
        )
    original_eager = timing["cpu_eager"]["unet"]["median_batch_ms"]
    original_prepared = timing["cpu"]["unet_folded"]["median_batch_ms"]
    original_gpu = timing["gpu1"]["unet"]["median_batch_ms"]
    folded_gpu = timing["gpu1"]["unet_folded"]["median_batch_ms"]
    lines += [
        "",
        "There is also an option requiring no retraining: optimize the existing U-Net checkpoint. "
        f"Its CPU latency changes from {original_eager:.3f} ms eager to {original_prepared:.3f} ms with "
        f"CPU preparation ({original_eager / original_prepared:.2f}x), and GPU latency changes from "
        f"{original_gpu:.3f} ms to {folded_gpu:.3f} ms by folding BatchNorm "
        f"({original_gpu / folded_gpu:.2f}x). This preserves the learned model, subject to floating-point rounding.",
    ]
    candidate = evaluations["fast_unet_b8_deploy"]
    lines += [
        "",
        f"FastUNet's validation-loss-selected checkpoint is epoch {candidate['epoch']}; its "
        f"validation-selected threshold is {candidate['threshold']:.2f}. It trained for eight epochs on "
        "62,400 images with 20,800 validation images, Adam at 0.001, batch 64, BCE + per-image Dice, "
        "training seed 42 and fixed split seed 42. Its auxiliary branches are merged after training. "
        f"Deployment uses {fast['convolution_macs']:,} counted convolution MACs per 50x100 image.",
        "",
        "Both the reloaded deployed GPU models and CPU inference graphs were evaluated over the full test set. "
        "The following conversion checks compare their logits/predictions with the unfused training graph "
        "on the same device, using the same threshold:",
        "",
    ]
    for name, check in result["conversion_checks"].items():
        lines.append(
            f"- {name}: maximum absolute logit difference {check['max_abs_logit_error']:.8g}; "
            f"{check['different_prediction_pixels']:,}/{check['total_pixels']:,} prediction pixels changed."
        )
    lines += [
        "",
        "Timing protocol: AMD Ryzen 7 9800X3D CPU, four PyTorch threads; AMD Radeon RX 9070 XT GPU; "
        "PyTorch 2.12.0+rocm7.14.0. Three shuffled rounds, 30 warmups and 100 measured passes per model per round, "
        "with device synchronization. Trained weights are loaded. Timings exclude graph preparation, data loading, "
        "host/device transfers and prediction decoding. Batch-64 times are for the entire batch. "
        "The model's smaller parameter/MAC count does not by itself explain these measured latencies.",
        "",
        "Absolute timings differed from the earlier random-weight baseline session, so speed ratios here use "
        "the new matched runs rather than mixing old and new measurements. CPU round-to-round variability "
        "was material; the JSON includes every round median and p95. These are local measurements, not "
        "a guarantee for other machines or background workloads.",
        "",
        "Deployment details matter: ordinary eager CPU execution hit slow small-convolution fallbacks on this build. "
        "FastUNet's prepared CPU result requires prepare_cpu(), which selects/fuses oneDNN kernels and rewrites eligible "
        "nonoverlapping transposed convolutions as equivalent 1x1 convolutions plus pixel shuffle. "
        "Every model is tested with that CPU optimization and both eager layouts; the table uses each model's fastest measured mode. This TorchScript-based runtime "
        "option is deprecated upstream and must be revalidated on future environments; portable state dictionaries "
        "are saved, and the CPU graph is prepared locally after loading. GPU inference uses ordinary folded modules.",
        "",
        "Interpretation limits: this is one training seed and one fixed simulated distribution. Close point estimates "
        "do not establish statistical equivalence. Pixel F1/IoU do not establish per-transmission localization AP "
        "or robustness to new frequency layouts. There is no quantization, distillation or data-generation change. "
        "Narrow U-Nets, BN folding and structural reparameterization are established techniques; novelty has not "
        "been demonstrated. A plain-block ablation and additional seeds remain research follow-ups.",
        "",
        "All 23 regression tests passed, covering branch/BN merging, checkpoint roundtrips, variable-batch CPU "
        "graphs, spatial guards, the subpixel weight permutation, training gradients, metric conventions "
        "and the existing structured-model checks. Original generator, loader and model source diffs are empty.",
        "",
        "Artifacts: [model and commands](../UNet/FAST_MODEL.md), [machine-readable comparison](fast_model_comparison.json), "
        "[deployed weights](../runs/fast_unet_b8_seed42/deploy.pt). Complete runs and source snapshots are in runs/. "
        "The original implementation and data-generation files remain unchanged.",
        "",
    ]
    (OUT / "fast_model_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(differences, indent=2))


if __name__ == "__main__":
    main()
