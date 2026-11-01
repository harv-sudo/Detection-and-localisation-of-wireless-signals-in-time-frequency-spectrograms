"""Plot saved experiments against digitized Figure 16a; never retrain models.

Dependencies: matplotlib, numpy. Run from any directory.
The optional local plotting dependency directory avoids changing the ML environment.
"""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "runs/supervisor_plot_dependencies"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = ROOT / "review_artifacts"
OUT = ART / "supervisor_comparison"
OUT.mkdir(exist_ok=True)
comparison = json.loads((ART / "fast_model_comparison.json").read_text())
reference = json.loads((ART / "reference_figure16a_digitized.json").read_text())
snrs = np.arange(-20, 6)
models = [
    ("unet", "Original U-Net", "#1765AB", "o"),
    ("evidence_axial", "EvidenceAxialNet", "#D36A19", "s"),
    ("fast_unet_b8_deploy", "FastUNet", "#14866A", "^"),
]
fcnn = {}
for line in (ROOT / "FCNN results.txt").read_text().splitlines():
    match = re.match(r"\|\s*([+-]?\d+)\s*\|\s*\**([\d.]+)", line)
    if match:
        fcnn[int(match[1])] = float(match[2])
assert sorted(fcnn) == list(snrs)
for key, *_ in models:
    assert sorted(map(int, comparison["rows"][key]["per_snr"])) == list(snrs)
for curve in reference["curves"].values():
    assert [p["snr_db"] for p in curve] == list(snrs)
    assert all(0 <= p["f1"] <= 1 for p in curve)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "savefig.facecolor": "white",
})

paper_styles = {
    "ED": ("#959DA6", "-"),
    "CT-SA": ("#65717E", "--"),
    "SSA": ("#6B5550", ":"),
    "DL-ANN Option 1": ("#B08030", "-."),
    "DL-ANN Option 2": ("#7C8B33", (0, (5, 1, 1, 1))),
    "DL-ANN Option 3": ("#793DA6", "--"),
    "DL-ANN Option 4": ("#CC4885", ":"),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 7.3))
fig.subplots_adjust(left=.065, right=.98, top=.84, bottom=.34, wspace=.19)
fig.suptitle("F1 versus SNR: local models and the published reference", x=.065,
             y=.98, ha="left", fontsize=17, fontweight="bold")
fig.text(.065, .925, "Signal Area Estimation based on Deep Learning (2023), Figure 16a, AWGN channel",
         color="#4D5968", fontsize=11)
for ax in axes:
    for key, name, color, marker in models:
        row = comparison["rows"][key]
        vals = [row["per_snr"][str(int(s))]["macro"]["f1"] for s in snrs]
        ax.plot(snrs, vals, color=color, label=f"Local: {name}", lw=2.1,
                marker=marker, markersize=4, markevery=2, zorder=5)
    ax.plot(snrs, [fcnn[int(s)] for s in snrs], color="#434A50", lw=1.6,
            ls="--", label="Local: historical FCNN recreation*", zorder=4)
    for name, points in reference["curves"].items():
        if ax is axes[1] and name not in ("DL-ANN Option 3", "DL-ANN Option 4"):
            continue
        color, linestyle = paper_styles[name]
        ax.plot(snrs, [p["f1"] for p in points], color=color, ls=linestyle,
                lw=1.7 if "3" in name or "4" in name else 1.15,
                label=f"Paper: {name}", zorder=3)
    ax.set(xlim=(-20, 5), xlabel="SNR (dB)", ylabel="F1 score")
    ax.set_xticks([-20, -15, -10, -5, 0, 5])
    ax.grid(alpha=.18)
axes[0].set(ylim=(0, 1.025), title="All seven published curves + four local models")
axes[1].set(ylim=(.65, 1.01), title="Detail: local models + paper Options 3 and 4")
axes[1].set_yticks(np.arange(.65, 1.001, .05))
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.065, .155),
           ncol=3, frameon=False, fontsize=9.4, handlelength=3.0,
           columnspacing=2.1, labelspacing=.7)
fig.text(.065, .115, "Published curves are digitized approximations from the author manuscript, not raw measurements.", fontsize=9)
fig.text(.065, .089, "Local curves use mean per-image foreground pixel F1. Datasets, training and averaging protocols are not fully matched.", fontsize=9)
fig.text(.065, .063, "* Historical FCNN: separate 312,000-image test set; the three newer models share 20,800 test images.", fontsize=9)
fig.text(.065, .037, "Paper Option 3 uses all training SNRs; Option 4 additionally receives SNR as an input. DOI: 10.1016/j.phycom.2023.102113", fontsize=8.6)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"f1_vs_snr_reference_overlay.{ext}", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(12, 5.6))
fig.subplots_adjust(left=.16, right=.98, top=.76, bottom=.32, wspace=.28)
fig.suptitle("Measured inference time on the same computer", x=.055, y=.965,
             ha="left", fontsize=17, fontweight="bold")
fig.text(.055, .90, "Saved matched runs | FP32 | 50 x 100 inputs | median forward time | lower is better", fontsize=11, color="#4D5968")
specs = [("CPU, batch 1", "cpu"), ("GPU, batch 1", "gpu1"), ("GPU, batch 64", "gpu64")]
timing_rows = []
for ax, (title, mode) in zip(axes, specs):
    vals = []
    for key, *_ in models:
        r = comparison["rows"][key]
        vals.append((r["best_cpu_latency"] if mode == "cpu" else r["latency"][mode])["median_batch_ms"])
    ax.barh(np.arange(3), vals, color=[m[2] for m in models], height=.55)
    ax.set_yticks(np.arange(3), [m[1] for m in models] if mode == "cpu" else ["", "", ""])
    ax.invert_yaxis()
    ax.set(title=title, xlabel="Milliseconds per batch", xlim=(0, max(vals)*1.28))
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.18)
    for i, v in enumerate(vals):
        ax.text(v + max(vals)*.025, i, f"{v:.3f}", va="center", fontsize=10)
    ax.text(.03, -.28, f"FastUNet: {vals[0]/vals[2]:.2f}x faster than U-Net", transform=ax.transAxes, fontsize=9.4)
fig.text(.055, .17, "CPU: AMD Ryzen 7 9800X3D (4 threads). GPU: AMD Radeon RX 9070 XT. PyTorch 2.12.0+rocm7.14.0.", fontsize=9)
fig.text(.055, .135, "Three shuffled rounds; 30 warmups + 100 timed passes per model per round; GPU synchronization enabled.", fontsize=9)
fig.text(.055, .10, "CPU: best measured mode from the same three offered modes. GPU: eager execution; native BatchNorm where present.", fontsize=9)
fig.text(.055, .065, "Excludes preparation, loading, transfers and output decoding. Batch-64 times cover all 64 images. FCNN timing pending.", fontsize=9)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"matched_inference_times.{ext}", dpi=200)
plt.close(fig)

print(f"Saved comparison and timing figures to {OUT}")
for s in (-20, -15, -10, -5, 5):
    local = comparison["rows"]["fast_unet_b8_deploy"]["per_snr"][str(s)]["macro"]["f1"]
    paper = reference["curves"]["DL-ANN Option 3"][s + 20]["f1"]
    print(f"{s:+} dB: FastUNet={local:.4f}, figure Option3~{paper:.3f}, graphical gap={100*(local-paper):.1f}pp")
