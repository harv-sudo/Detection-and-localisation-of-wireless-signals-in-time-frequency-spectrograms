# Standardised results snapshot

Copied from the completed local run records on 1 November 2026. The JSON files
are unchanged copies, not new evaluations. Each of the four folders contains
the configuration, dataset manifest, training history and evaluation used for
the current comparison. The FCNN folder also contains the five timing sessions
(three CPU modes, GPU batch 1 and GPU batch 64); each session benchmarks all
four architectures plus the BatchNorm-folded U-Net variant.

The evaluation JSON records the exact test indices, overall and per-SNR metrics,
metric conventions and checkpoint provenance. Histories contain epoch durations
and validation results. Paths inside the records describe the original Windows
machine; they are provenance, not portable executable paths.

These files are sufficient to inspect and replot the reported measurements.
They do **not** contain the datasets, trained weights, complete training split
files or original training-source snapshots. Those remain in ignored local
`runs/` directories and must be transferred separately for exact reruns.

## Recreate the current figures

Install `matplotlib`, `numpy`, `pdfplumber` and `reportlab` in a suitable Python
environment. From the repository root:

```sh
python review_artifacts/plot_standardised_experiment.py
python review_artifacts/write_experiment_record.py
```

The plotting script first uses the original local `runs/` records if present,
and otherwise falls back to this tracked snapshot. The hardware/training summary
is the fixed description of these completed runs. Output goes into `output/pdf/`
using the existing names; rendered previews go to ignored `tmp/pdfs/`.

The original paper graph is sourced from the author manuscript stored at
`review_artifacts/reference_signal_area_2023.pdf`; see
[the reference notes](../reference_paper_notes.md) for its public source and
comparison limits. Older plotting scripts and LaTeX examples use historical
results and should not be substituted for `plot_standardised_experiment.py`.
