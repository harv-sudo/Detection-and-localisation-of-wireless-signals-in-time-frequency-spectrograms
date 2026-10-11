Implementation review and research recommendations — 11 October 2026

Historical audit of the repository before the subsequent model implementation and dependency installation. For completed model-only experiments and the current environment, see model_baseline_results.md in this directory and UNet/README.md.

Your stated priority is higher detection/localisation accuracy. My recommendation is to establish a common, defensible benchmark first, then investigate **noise-aware signal evidence and structured boundary prediction**. Compression and explanation tools can support that work, but changing U-Net alone would not establish a new research contribution.

This review covers the FCNN, U-Net, MATLAB generators, training/evaluation scripts and recorded result tables. Model architecture checks and synthetic prior calculations were run. No trained checkpoint evaluation or retraining was performed. Training/model/generator source files were left unchanged; reproducible review helpers and results are in review_artifacts.

**What the current results establish**

Your saved FCNN results report F1 0.8216 / IoU 0.7108; U-Net reports 0.8738 / 0.7759. Those numbers are promising observations, but their difference is not yet an apples-to-apples accuracy gain.

The FCNN averages per-image F1/IoU; U-Net accumulates pixel counts over the full test set before computing metrics. These are macro and micro averaging respectively. Differences in image occupancy and failure distribution affect them differently. The training configurations also point to different dataset locations and sizes; a common held-out sample manifest is not recorded. See [baseline_recreation/evaluate.py](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/baseline_recreation/evaluate.py:116>) and [UNet/evaluate.py](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/UNet/evaluate.py:311>).

The U-Net implementation has sensible foundations: spatial weight sharing, skip connections, raw output logits and BCE-with-logits plus Dice loss. The FCNN supplies a useful low-arithmetic-cost regression comparator. Both need stronger experimental controls before publication claims.

**Issues to fix before another expensive experiment**

1. **Current FCNN training and overall evaluation have a tuple-unpacking error.** The loader returns image, boxes and SNR, while training and overall evaluation expect only image and boxes. SNR evaluation already handles all three. This means the recorded results cannot be reproduced using these exact entry points as checked in. See [baseline_recreation/dataset_large.py](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/baseline_recreation/dataset_large.py:60>), [baseline_recreation/train.py](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/baseline_recreation/train.py:93>) and [baseline_recreation/evaluate.py](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/baseline_recreation/evaluate.py:103>).

2. **The generator makes frequency localisation largely predetermined.** Both generators set channel_map=[0 1 0]. At the current resolution, every true box has MATLAB x=35 and width=31; only temporal activity changes. A model can exploit this without learning general frequency localisation. Minimum on/off durations and their distributions are also fixed. See [Signal area generation code/generate_dataset.m](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/Signal area generation code/generate_dataset.m:18>), [Signal area generation code/generate_unet_local_dataset.m](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/Signal area generation code/generate_unet_local_dataset.m:26>) and [Signal area generation code/create_test_image_channelised_modified.m](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/Signal area generation code/create_test_image_channelised_modified.m:24>). Retain this as a controlled reproduction setting, but add unseen frequencies, widths, durations, simultaneous channels and explicit empty scenes for your contribution.

3. **The input is a binary detector-output map.** SNR controls an analytic detection probability, after which pixels undergo independent Bernoulli flips; this is not an IQ-to-STFT noise simulation. The model has no amplitude, phase, leakage or correlated-noise information. This is a legitimate restricted model of a detector output, but conclusions should match that scope. See [Signal area generation code/add_noise_to_image.m](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/Signal area generation code/add_noise_to_image.m:27>). More neural capacity cannot recover evidence discarded before the network.

4. **Detection/localisation is not fully measured by union-mask overlap.** The FCNN unions all predicted boxes; U-Net directly predicts foreground. Neither current evaluation establishes whether individual signals are counted, split, merged or duplicated correctly. Add one-to-one object matching with precision/recall at predeclared IoU thresholds, count error, start/end-time error and frequency-edge error. Use pixel units until the generator defines sample-time and frequency-bin scales. Report recall at a fixed false-alarm budget, especially on empty windows and weak/short/narrow signals.

5. **FCNN latency cannot support an efficiency claim yet.** Its timers stop before GPU work is synchronized, and the CPU copy occurs afterwards. U-Net does synchronize, but uses batch-amortized forward timing with no warmup; this is not batch-one request latency. Both omit data preparation and output decoding. See [baseline_recreation/evaluate_by_snr.py](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/baseline_recreation/evaluate_by_snr.py:110>) and [UNet/evaluate.py](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/UNet/evaluate.py:264>). Follow [PyTorch's timing guidance](https://docs.pytorch.org/docs/stable/notes/cuda): warm up, synchronize or use device events, and report batch-one median/p95, throughput, precision, hardware and end-to-end time separately.

**A revealing simple-baseline check**

A Python simulation of 100,000 clean scenes from the checked-in generator's duration laws, seed 42, estimates central-channel occupancy at 0.64253. Predicting that entire 31-column channel as busy at all times, without consulting the input, gives:

| Input-independent baseline | Simulated score |
|---|---:|
| Micro F1 | 0.78237 |
| Macro F1 | 0.78035 |
| IoU | 0.64253 |

These values exceed the recorded -20 dB F1 values numerically (U-Net 0.7241; FCNN 0.6988). **This is a distribution sanity check, not a re-evaluation on your saved test split.** It does not establish a better detector: the baseline cannot resolve separate bursts and generates false alarms during idle intervals. It shows why F1 alone can reward the strong occupancy/location prior.

At -20 dB, the generator's per-pixel detection probability is about 0.01375, close to its false-alarm probability of 0.01. Much of a model's apparent success can therefore come from temporal/location priors. At high SNR the opposite issue arises: a rule marking a row busy when at least four of its 31 known-channel pixels are one has analytic asymptotic micro F1 about 0.99993 from -5 dB upward under this independent-flip model. This rule uses a fixed threshold and the configured band; it does not receive true SNR. Its estimate is neither saved-data evaluation nor an unknown-band result.

Run the following baselines on exactly the same held-out samples: input mask unchanged, always-empty, always-central-band, row-count threshold, morphology/connected components and a temporal probabilistic model. Tune any thresholds on validation data only. Include both known-band and band-agnostic versions. The large gap between the simple high-SNR prediction and the recorded U-Net plateau merits checking input orientation, box-to-mask identity, thresholding, convergence and boundary errors before adding architectural complexity.

The simulation also finds repeated clean templates across random splits. This is **template overlap, not proof of identical noisy-sample leakage**. Preserve an IID test, but add a test grouped by latent clean scene and a test with held-out geometry/duration settings. When multiple SNR/noise versions of a scene are created, assign its split first.

Reproduce these calculations with [prior_sanity_check.py](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/review_artifacts/prior_sanity_check.py>); numerical output is in [prior_sanity_check_results.json](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/review_artifacts/prior_sanity_check_results.json>).

**Accuracy improvements, in recommended order**

| Change | Reason to try it | Decisive experiment |
|---|---|---|
| Shared evaluation, validation threshold and simple baselines | Establish whether observed improvements reflect detection ability | Same masks, sample IDs, macro/micro scores and object metrics |
| More diverse scenes and explicit empty windows | Remove fixed-position shortcuts and measure false alarms | Unseen frequency/width/duration and simultaneous-signal tests |
| Noise-normalized soft power input | Preserve weak-signal evidence before hard thresholding | Binary versus soft versus soft+binary input on identical underlying waveforms |
| Boundary-aware segmentation and event decoding | Reduce missed short signals and merged bursts | Small-signal recall, matched boundary error and object recall at fixed false alarms |
| Temporal structure with estimated noise reliability | Exploit persistence while exposing reliance on priors | Compare evidence-only, duration-only and combined models under duration/noise shift |

For the soft-input experiment, generate or record waveforms, add noise/channel effects, compute power spectrograms, estimate the noise floor and normalize consistently. Feed normalized log-power and optionally a CFAR mask/noise estimate. Preserve the present binary benchmark as a separate comparison. Never use clean labels or true test SNR to supply deployable input features; an oracle-SNR result must be labelled separately. Keep train/test captures, source waveforms and related crops disjoint. Correlated noise, fading, frequency offset, leakage and unequal-power overlapping signals are useful targeted stress tests, not substitutes for a controlled ablation.

For U-Net, first sweep the probability threshold on validation data under the chosen false-alarm constraint. Then compare BCE+Dice with a modest boundary-weighted auxiliary objective and explicit time/frequency edge prediction. Do not assume more positive-class weighting improves detection: it can increase false alarms. Connected components can decode masks into regions, but touching/overlapping signals need additional separation machinery, such as centers/offsets or instance heads.

For FCNN, add explicit box-presence confidence and apply coordinate regression only to real matched boxes, alongside a presence loss and an overlap/boundary objective. Current fixed-slot MSE gives many terms to padded boxes. The present generator orders events deterministically, so arbitrary label permutation is not a demonstrated current bug; set matching becomes useful as scenes generalize. Constrain coordinates to valid bounds. Compare this against a compact convolutional encoder, not only a larger flattened MLP.

Train to documented validation convergence. FCNN currently specifies one epoch; U-Net specifies eight on a different configured corpus. The existing baseline_recreation/checkpoints/best_model.pt was inspected with weights_only=True and records epoch 45 and validation loss 0.00081627; its layer shapes match the current FCNN, but its dataset/run provenance is absent. Therefore the current one-epoch setting does not establish how the saved results were trained. Epoch count alone is not a fair budget. Record examples seen, optimizer steps, wall-clock/accelerator time and validation curves. Select models using a predeclared validation criterion aligned with the intended detection task.

**A research hypothesis worth pursuing**

A useful target claim would be:

“A noise-aware model combining local time-frequency evidence with explicit boundary and persistence constraints improves weak-signal detection and localisation under noise-floor and scene changes, at a fixed false-alarm budget.”

This is a proposed hypothesis, not an established novel contribution. Its strongest version would work when the occupied frequency band is unknown and signal durations differ from training.

A particularly interpretable starting point follows directly from your current generator. For a candidate band of width w, let k(t) be the number of positive detection bits in row t. With known detector probabilities, its busy-versus-idle log-likelihood evidence is:

```text
e(t) = k(t) log(Pd/Pfa)
     + [w-k(t)] log[(1-Pd)/(1-Pfa)]
```

Under the current uniform, independent Bernoulli model, row counts retain the relevant within-band evidence about row occupancy. A model with explicit busy/idle duration distributions can combine this evidence across time and infer burst boundaries. It provides inspectable evidence and duration contributions while avoiding a full image network. A tiny 1-D temporal CNN is another necessary comparator.

The classical version is a baseline, not the novelty claim. Possible research work is robust estimation of detector reliability, learned but constrained temporal/boundary terms, or a frequency-shared extension handling unknown bands and bandwidths. Oracle known-band/known-Pd results should be separated from models that estimate these quantities. Measure performance when those assumptions are wrong; otherwise the method may simply reproduce the generator.

An alternative with less implementation risk is a boundary-aware U-Net trained on richer signal evidence, supported by an ablation of each new component. That direction still needs a specific methodological contribution beyond applying standard segmentation techniques.

**Compute headroom**

CPU forward passes verified the U-Net shapes and analytical convolution MAC counts below. The narrow variants were instantiated only for profiling and are untrained.

| Architecture | Trainable parameters | Linear/convolution MACs per image |
|---|---:|---:|
| Current FCNN | 1,498,192 | 1,497,088 |
| Current U-Net, base width 32 | 1,926,433 | 685,554,176 |
| U-Net, base width 16 | 482,449 | 171,788,544 |
| U-Net, base width 8 | 121,033 | 43,147,136 |

These counts exclude activation, normalization, pooling, interpolation and memory costs; one MAC is one multiply-accumulate, not two FLOPs. U-Net has about 458 times the counted arithmetic of FCNN, despite only about 29% more parameters. Even the width-8 U-Net has about 29 times the counted arithmetic of FCNN. A smaller parameter count does not establish lower latency.

Width-16 and width-8 are useful accuracy/compute ablations before depthwise convolutions, attention or transformers. Distillation from a demonstrably stronger teacher can be tested later; it cannot compensate for an unreliable teacher/benchmark. Reproducible profiling and per-layer results are in [profile_unet.py](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/review_artifacts/profile_unet.py>) and [unet_compute_profile.json](<C:/Users/Isaac/Desktop/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/review_artifacts/unet_compute_profile.json>).

Training throughput can improve independently: both loaders reopen HDF5 for every sample; benchmark worker-local lazy file handles, bounded caching and contiguous compact storage. Avoid sharing an already-open HDF5 handle across worker processes. U-Net computes several scalar GPU-to-CPU reductions every batch, including duplicate true-positive/intersection counts; accumulate counters on device and transfer less often. Test mixed precision, channels-last and batch/worker counts on the actual accelerator. These are engineering optimizations, not assumed accuracy improvements or novelty; see [PyTorch's performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

**Explainability that contributes evidence**

For each detected region, explain a clearly defined score, such as the mean logit in a fixed region. Compare region occlusion and Integrated Gradients using multiple plausible noise-only references. Record explanation generation time separately from detection latency. [Integrated Gradients](https://arxiv.org/abs/1703.01365) is an existing attribution method, not a new contribution by itself.

Use controlled interventions: remove a true signal and regenerate its observation, move it to another frequency, change its duration or amplitude, add an interferer, and alter only idle-region noise. Check whether predictions and attributed evidence respond appropriately. Report faithfulness, stability across noise draws and behavior under model randomization, alongside localisation accuracy. A heatmap overlapping the output mask is not sufficient validation; [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292) motivates testing whether explanations depend on the learned model.

In a structured detector, expose the observed signal evidence, estimated reliability, temporal-prior contribution and boundary confidence directly. Calibrated confidence addresses uncertainty; it does not by itself explain feature reliance.

**Novelty and related work**

The apparent FCNN reference is [Alammar and López-Benítez, Signal Area Estimation based on Deep Learning (2023)](https://livrepository.liverpool.ac.uk/3170365/). Confirm this against your intended reference before treating the implementation as an exact reproduction.

U-Net-style spectral segmentation with connected-component localisation already appears in [West, Roy and O'Shea (2021)](https://arxiv.org/html/2110.00583v1). Efficiency-focused spectrogram segmentation is also represented by [RecSpecUNet (2026), author repository](https://github.com/PhucThinhHuynh/RecSpecUNet). Their reported scores use their own tasks and datasets and cannot be compared directly with yours.

Model-based learned sensing has prior art in [PU-DetNet (2022)](https://livrepository.liverpool.ac.uk/3164581/), although its sensing task differs from arbitrary two-dimensional signal-area localisation. [Stitching the Spectrum (2024)](https://arxiv.org/abs/2402.03465) supplies relevant context for realistic data generation and broader spectrum segmentation. These sources show why “use U-Net,” “make it smaller” or “add explanations” is too broad a novelty claim. This was a targeted literature check, not an exhaustive novelty search.

**A manageable experiment sequence**

1. Repair the FCNN entry points; freeze a shared dataset manifest/split; check box-to-mask identity and orientation using a known asymmetric scene. Record dependencies, code revision, generator seed/config and split IDs with each result. The local Torch environment currently lacks numpy/h5py. A baseline checkpoint is present, but no U-Net checkpoint was found in its configured location, so saved scores were not reproduced.
2. Run the six cheap baselines and both models through one evaluator. Report macro/micro pixel scores, per-SNR object recall at fixed false alarms and boundary error. Audit empty and small-signal cases.
3. Generalize scene locations, widths, durations and occupancy; keep an IID set and held-out-condition sets. Split scene/source identity before generating noisy variants.
4. Test input representation and one structured/boundary modification with explicit ablations. If compute is limited, screen ideas with width-16 U-Net and verify finalists against the full model.
5. Evaluate finalists on at least three training seeds and independent test scenes/captures. Use paired scene-level bootstrap intervals for differences, not individual-pixel resampling. Include a small independent real-signal dataset if feasible.

The immediate decision gate is whether a model improves on the cheap baselines on the same samples and improves localisation at a controlled false-alarm rate. That evidence will determine which architectural change deserves the next training run.

