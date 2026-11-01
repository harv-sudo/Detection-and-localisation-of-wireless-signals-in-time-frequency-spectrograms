"""Retain original Figure 16a pixels and overlay saved local model results.

No model training/evaluation. Axis coordinates are taken from the source PDF,
not estimated from a screenshot. Requires matplotlib, numpy and pdfplumber.
"""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "runs/supervisor_plot_dependencies"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import pdfplumber

ART = ROOT / "review_artifacts"
OUT = ART / "supervisor_comparison"
PDF_OUT = ROOT / "output/pdf"
OUT.mkdir(exist_ok=True)
PDF_OUT.mkdir(parents=True, exist_ok=True)
source = ART / "reference_signal_area_2023.pdf"
rows = json.loads((ART / "fast_model_comparison.json").read_text())["rows"]
reference = json.loads((ART / "reference_figure16a_digitized.json").read_text())
bounds = reference["plot_bounds_pdf_points"]
left, right, top, bottom = [bounds[k] for k in ("left", "right", "top", "bottom")]
crop = (64, 56, 273, 221)
with pdfplumber.open(source) as pdf:
    original = pdf.pages[11].crop(crop).to_image(resolution=600).original

snrs = np.arange(-20, 6)
specs = [
    ("unet", "Original U-Net", "#163A75", "o", (0, (6, 3)), 0),
    ("evidence_axial", "EvidenceAxialNet", "#D55E00", "s", (0, (1, 2)), 1),
    ("fast_unet_b8_deploy", "FastUNet", "#008577", "^", "-", 2),
    ("fcnn", "FCNN recreation*", "#56236D", "D", (0, (6, 2, 1, 2)), 0),
]
series = {key: np.array([rows[key]["per_snr"][str(int(s))]["macro"]["f1"] for s in snrs])
          for key, *_ in specs[:3]}
fcnn = {}
for line in (ROOT / "FCNN results.txt").read_text().splitlines():
    match = re.match(r"\|\s*([+-]?\d+)\s*\|\s*\**([\d.]+)", line)
    if match:
        fcnn[int(match[1])] = float(match[2])
assert sorted(fcnn) == list(snrs)
series["fcnn"] = np.array([fcnn[int(s)] for s in snrs])
for values in series.values():
    assert len(values) == 26 and np.all((values >= 0) & (values <= 1))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "svg.fonttype": "none", "savefig.facecolor": "white"})
fig = plt.figure(figsize=(13.6, 8.8), facecolor="white")
fig.text(.045, .95, "Our models on the paper's Figure 16a", fontsize=23, weight="bold", color="#172B42")
fig.text(.045, .904, "Original published graph retained  |  Our results added as thick lines with markers",
         fontsize=13, color="#526173")

# Raster source and vector overlays share the original PDF point coordinate system.
ax = fig.add_axes([.03, .15, .635, .71])
ax.imshow(original, extent=(crop[0], crop[2], crop[3], crop[1]), aspect="equal", zorder=0)
ax.set_xlim(crop[0], crop[2])
ax.set_ylim(crop[3], crop[1])
ax.axis("off")
xp = left + (snrs + 20) * (right - left) / 25
for key, label, color, marker, ls, offset in specs:
    yp = bottom - series[key] * (bottom - top)
    ax.plot(xp, yp, color=color, lw=2.4, linestyle=ls, zorder=3,
            path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()])
# Draw staggered markers last so all three almost-coincident curves remain visible.
for key, label, color, marker, ls, offset in specs:
    yp = bottom - series[key] * (bottom - top)
    ax.plot(xp, yp, lw=0, marker=marker, markevery=(offset, 3), ms=6.3,
            mfc="white", mec=color, mew=1.8, zorder=5)

handles = [Line2D([0], [0], color=c, ls=ls, lw=2.5, marker=m, ms=7,
                  mfc="white", mec=c, mew=1.7, label=label)
           for _, label, c, m, ls, _ in specs]
fig.text(.71, .824, "OUR RESULTS", fontsize=12, weight="bold", color="#172B42")
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.695, .805),
           frameon=False, fontsize=13, handlelength=3.6, labelspacing=.85)
fig.text(.71, .604, "The original legend inside the graph\nidentifies the paper's thin curves.",
         fontsize=10.5, color="#526173", linespacing=1.5)

zoom = fig.add_axes([.72, .26, .245, .28])
mask = snrs <= -12
for key, label, color, marker, ls, offset in specs[:3]:
    zoom.plot(snrs[mask], series[key][mask], color=color, ls=ls, lw=1.8)
for key, label, color, marker, ls, offset in specs[:3]:
    zoom.plot(snrs[mask], series[key][mask], color=color, lw=0, marker=marker,
              markevery=(offset, 3), ms=5, mfc="white", mec=color, mew=1.4)
zoom.set(title="Low-SNR detail: our three models", xlim=(-20.3, -11.7), ylim=(.765, .95),
         xlabel="SNR (dB)", ylabel="F1 score")
zoom.title.set_size(11)
zoom.set_xticks([-20, -18, -16, -14, -12])
zoom.set_yticks([.78, .82, .86, .90, .94])
zoom.tick_params(labelsize=10)
zoom.spines[["top", "right"]].set_visible(False)
zoom.grid(alpha=.16)
fig.text(.71, .153, "Near-overlapping lines mean similar scores.\nNo values have been shifted or smoothed.",
         fontsize=10.5, color="#526173", linespacing=1.5)

fig.text(.045, .087, "Source: Alammar & Lopez-Benitez (2023), Figure 16a, AWGN. DOI: 10.1016/j.phycom.2023.102113", fontsize=10)
fig.text(.045, .057, "Local results: mean per-image pixel F1. Training data and evaluation protocols differ from the paper.", fontsize=10)
fig.text(.045, .027, "* FCNN recreation is our earlier experiment on a separate test set; it is not the published paper curve.", fontsize=10)

png = OUT / "figure16a_with_our_models.png"
pdf = PDF_OUT / "figure16a_with_our_models.pdf"
fig.savefig(png, dpi=220)
fig.savefig(pdf, dpi=400)
plt.close(fig)
print(png)
print(pdf)

# Re-render the actual exported PDF for layout verification.
with pdfplumber.open(pdf) as document:
    assert len(document.pages) == 1
    document.pages[0].to_image(resolution=120).save(OUT / "figure16a_pdf_preview.png")

manifest = {
    "source_pdf": str(source.relative_to(ROOT)), "source_page_1based": 12,
    "crop_pdf_points": crop, "axis_bounds_pdf_points": bounds,
    "source_figure": "Unaltered colour raster of the original Figure 16a",
    "local_scores": "Saved macro per-image pixel F1; no offset or smoothing",
    "markers": "Different symbols staggered across actual SNR positions to limit occlusion",
    "accuracy_data": "review_artifacts/fast_model_comparison.json",
    "historical_fcnn_data": "FCNN results.txt",
    "new_inference_or_training": False,
}
(OUT / "figure16a_overlay_provenance.json").write_text(json.dumps(manifest, indent=2))
