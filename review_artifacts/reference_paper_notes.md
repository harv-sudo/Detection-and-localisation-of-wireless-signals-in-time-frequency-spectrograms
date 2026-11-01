# Reference paper audit (1 November 2026)

Source: Mohammed M. Alammar and Miguel Lopez-Benitez, **Signal area estimation based on deep learning**, Physical Communication 59 (2023), 102113. DOI: https://doi.org/10.1016/j.phycom.2023.102113

Public author accepted manuscript: https://www.lopezbenitez.es/journals/Physical_Communication_2023.pdf

University repository record: https://livrepository.liverpool.ac.uk/3170365/

The author manuscript was downloaded as `reference_signal_area_2023.pdf`. Its page 12 contains Figure 16. The page was rendered to `reference_figure16_page.png` and visually inspected. The publisher site could not be fetched directly; this audit uses the public accepted manuscript.

## Metric and appropriate reference

- Section IV-C, manuscript page 7, defines F1 as `2 TP / (2 TP + FP + FN)` using point-by-point comparison of the output spectrogram with ground truth. This is foreground pixel F1, not per-transmission/object detection F1.
- The definition matches the underlying confusion-count formula in the current segmentation experiments. The manuscript does not explicitly establish whether the plotted test curve is a mean of per-image F1 values or a pooled F1 over all pixels. Do not claim proven equality of macro/micro aggregation.
- Figure 16a uses an AWGN channel and includes ED, CT-SA, SSA, and four DL-ANN training options. Option 1 trains only at -5 dB; option 2 at -10, -7 and -5 dB; option 3 across -20 to +5 dB, in 1 dB increments; option 4 uses the same range plus SNR as a model input.
- Option 3 is the closest reference to the current U-Net models because all train across the SNR range without receiving SNR as an input. Option 4 can also be shown but should be marked as receiving additional SNR information.
- The older local FCNN results are a local recreation, not the numbers in Figure 16a. Keep the two curves distinct.

## Digitization

`reference_figure16a_digitized.json` stores all seven reference curves under `curves`. Keys are `ED`, `CT-SA`, `SSA`, `DL-ANN Option 1`, `DL-ANN Option 2`, `DL-ANN Option 3`, `DL-ANN Option 4`. Each value is a list of 26 objects with `snr_db` and `f1` fields.

`digitize_reference_figure16a.py` reproduces the extraction. Unlike manual visual estimates, it reads vector polyline vertices embedded in the PDF and scales their coordinates to the verified axis bounds. Nonetheless these are figure-derived graphical approximations, not released raw experimental data; present them as digitized reference curves and avoid implying four-decimal experimental precision.

Selected figure-derived values (rounded to three decimals):

| SNR | Option 3 | Option 4 |
| ---: | ---: | ---: |
| -20 | 0.709 | 0.729 |
| -15 | 0.763 | 0.783 |
| -10 | 0.849 | 0.863 |
| -5 | 0.961 | 0.968 |
| +5 | 0.957 | 0.968 |

There is a small source discrepancy: the prose on page 12 describes option 3 as about 73% at -20 dB, whereas the plotted green option-3 line is approximately 70.9% and the magenta option-4 line is approximately 72.9%. Preserve the plotted line for an overlay and disclose this discrepancy rather than adjusting it to match the prose.

## Simulation and training comparison

Section IV-A uses 50 time points by 100 frequency points; three equal-width channels with 5% guardbands; only the central channel active; minimum on/off durations 10/5 points; exponential rates 0.5 per point; CFAR false-alarm probability 0.01; SNR -20 through +5 dB. The current MATLAB generator uses those same stated geometry, duration and CFAR settings. Its `N=100` sensing sample count and AWGN probability formula produce a detection probability near 0.39 at -7 dB, consistent with the paper's quoted misdetection probability near 0.61. The paper does not state N directly in the inspected methodology text, so this is a consistency check, not verification of all implementation details.

Section IV-D states **60,000 independent grids per SNR** and a **60/20/20 split**. Across all 26 SNRs this corresponds to 1,560,000 grids and nominally 312,000 test grids. Table I lists 100 epochs. Current matched U-Net/EvidenceAxialNet/FastUNet experiments instead use 4,000 grids per SNR (104,000 total), 62,400/20,800/20,800 split and eight epochs. The random test grids are not identical to the paper's, and training budgets differ substantially. Report cross-study comparison as indicative, not a controlled head-to-head or proven accuracy improvement.

The local generator's `resolution='low'` still means 50x100; the paper calls those dimensions medium resolution. Compare dimensions rather than those labels.

## Architecture, parameter counts and time

Figure 9 (manuscript page 9) shows a dense network `5000 -> 256 -> 256 -> 256 -> 256 -> 80`, with an input dropout layer. Counting linear weights and biases gives **1,498,192 parameters** (a derived count, not a number quoted by the paper). Option 4 adds one SNR input and thus 256 parameters.

The local `baseline_recreation/model.py` has the same linear dimensions, ReLU hidden activations, sigmoid output and dropout rate 0.4; its dropout is applied after every hidden layer, rather than the input-only dropout shown in Figure 9. It is therefore a recreation with a training-architecture difference. Outputs are 20 bounding-box tuples rather than segmentation logits.

FastUNet's 120,681 deployed parameters represent approximately **91.94% fewer than the derived paper FCNN count**, and **93.74% fewer than the original local U-Net's 1,926,433**. The previously quoted 94% refers to the U-Net comparison, not the paper FCNN.

Section IV-C defines computation time as execution after training, excluding offline training. Figure 16b is in seconds. The inspected manuscript text does not identify the processor model for those timings, nor a sufficiently detailed timing/transfer/postprocessing protocol to compare its absolute times directly to our modern CPU/GPU results. No hardware-relative speed-up claim should be made from Figure 16b. All four local architectures should be measured in the same current environment for a controlled timing comparison.
