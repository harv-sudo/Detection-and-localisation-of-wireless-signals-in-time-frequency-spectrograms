# Supervisor comparison - 1 November 2026

**Historical figures:** this folder predates the matched FCNN evaluation and the
latest five-model timing session. For the current comparison, use
[the architecture guide](../../architecture_guide/README.md),
[the current PDFs](../../output/pdf/) and
[the saved standardised results](../standardised_results/README.md).
The old FCNN scores and October timings below are retained as historical records.

The PNG files can be inserted into an email; the SVG files are editable vector versions.

## Accuracy overlay

`f1_vs_snr_reference_overlay.png` combines the three matched local segmentation models, the historical local FCNN recreation, and all seven curves from Figure 16a of Alammar & Lopez-Benitez, *Signal area estimation based on deep learning* (2023), DOI 10.1016/j.phycom.2023.102113.

The published curves were digitized from vector paths in the public author manuscript at https://www.lopezbenitez.es/journals/Physical_Communication_2023.pdf . They are graphical approximations, not author-provided raw measurements. The right panel focuses on the local models and paper Options 3/4. The duplicate BN-folded U-Net accuracy curve is omitted.

Local results use foreground pixel F1 per image, then average over images at each SNR. The paper uses the same TP/FP/FN formula but does not explicitly resolve macro versus pooled micro averaging. Option 3 trains over all SNRs without SNR as an input; Option 4 also receives SNR.

| SNR (dB) | FastUNet, local macro F1 | Paper Option 3, digitized F1 |
| ---: | ---: | ---: |
| -20 | 0.7809 | approximately 0.709 |
| -15 | 0.8589 | approximately 0.763 |
| -10 | 0.9785 | approximately 0.849 |
| -5 | 1.0000 | approximately 0.961 |
| +5 | 1.0000 | approximately 0.957 |

The paper's prose describes Option 3 as approximately 0.73 at -20 dB, while its plotted Option 3 curve is approximately 0.709 (Option 4 is approximately 0.729). The overlay follows the figure, without silently reconciling this discrepancy.

The three local segmentation models share 20,800 test images and eight training epochs. The historical local FCNN results use a separate 312,000-image test set. The paper reports 60,000 grids/SNR and 100 training epochs, versus 4,000 grids/SNR locally. These cross-study curves are indicative; they do not establish a controlled superiority claim. The older local FCNN result is not the published paper curve.

## Measured speed

`matched_inference_times.png` re-plots the saved matched timing runs from 11 October 2026. No new timing experiment or training was performed for this figure.

| Model | CPU batch 1 | GPU batch 1 | GPU batch 64 |
| --- | ---: | ---: | ---: |
| Original U-Net | 3.747 ms | 2.626 ms | 14.450 ms |
| EvidenceAxialNet | 2.412 ms | 3.796 ms | 15.076 ms |
| FastUNet deployed | 1.231 ms | 1.698 ms | 4.025 ms |

AMD Ryzen 7 9800X3D, four CPU threads; AMD Radeon RX 9070 XT; PyTorch 2.12.0+rocm7.14.0; FP32, 50x100 inputs. Three shuffled rounds with 30 warmups and 100 timed passes per model per round, with synchronization. CPU results select the best measured mode from the same three offered modes, not an identical selected mode for every model. GPU inference is eager with native BatchNorm where present. Times exclude graph preparation, loading, transfers, and prediction decoding. Batch 64 means the entire batch, not per image.

FastUNet is 3.04x faster on CPU, 1.55x for GPU batch 1 and 3.59x for GPU batch 64 than the original U-Net in these measurements. Relative to the BN-folded U-Net, speedups are 3.08x, 1.28x and 3.22x respectively. FCNN still needs matched timing; its historical unsynchronized timing should not be used.

## Defensible claim and remaining checks

FastUNet uses 120,681 deployed parameters versus 1,926,433 for original U-Net (93.74% fewer) with overall test macro F1 0.943762 versus 0.944639, a decrease of 0.088 percentage points. These single-seed point estimates establish a promising measured efficiency trade-off, not statistical equivalence.

The dense architecture drawn in the reference paper implies 1,498,192 parameters including biases, making FastUNet approximately 91.94% smaller than that architecture. This is a derived count. Fewer parameters do not establish faster inference relative to the FCNN; its arithmetic workload is much lower than U-Net's.

Before a publication claim: evaluate/retrain FCNN under the same protocol, measure all four local architectures with the same timing harness, confirm the paper's F1 averaging, and repeat training across seeds. Fixed central-channel geometry also limits claims about generalisation to arbitrary signal locations.

## Reproduction

Sources: `../fast_model_comparison.json`, `../../FCNN results.txt`, `../reference_figure16a_digitized.json`.

Run `../digitize_reference_figure16a.py` with pdfplumber to regenerate the paper curves. Run `../plot_supervisor_comparison.py` with matplotlib/numpy to regenerate these figures. Detailed source notes are in `../reference_paper_notes.md`.
