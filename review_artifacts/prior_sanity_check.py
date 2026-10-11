"""Reproduce data-prior sanity checks without training or third-party packages.

This simulates the distribution in the checked-in MATLAB low-resolution
generator, not the user's saved dataset. Python and MATLAB random streams
are different. Identical clean masks are template overlap, not evidence of
identical noisy samples crossing the train/test boundary.

The row-count baselines know the configured active frequency interval.
Each threshold is fixed across all SNRs; the analytic evaluation uses the
generator's Pd(SNR), but the baseline itself does not receive SNR.

Run: python review_artifacts/prior_sanity_check.py
"""

import argparse
import json
import math
import random
import statistics
from collections import Counter


def simulate_masks(count, seed):
    rng = random.Random(seed)
    masks = []
    busy_counts = []
    for _ in range(count):
        state = int(rng.random() < 0.5)
        position = 0
        mask = 0
        busy_count = 0
        while position < 50:
            duration = math.floor((10 if state else 5) + rng.expovariate(0.5))
            if state:
                used = min(50 - position, duration)
                mask |= ((1 << used) - 1) << position
                busy_count += used
            position += duration
            state = 1 - state
        masks.append(mask)
        busy_counts.append(busy_count)
    return masks, busy_counts, rng


def binomial_tail(probability, threshold, width=31):
    return sum(
        math.comb(width, successes)
        * probability**successes
        * (1 - probability)**(width - successes)
        for successes in range(threshold, width + 1)
    )


def run(count=100_000, seed=42):
    masks, busy_counts, rng = simulate_masks(count, seed)
    occupancy = sum(busy_counts) / (count * 50)
    template_counts = Counter(masks)
    rng.shuffle(masks)
    train_end = int(0.6 * count)
    test_start = train_end + int(0.2 * count)
    train_templates = set(masks[:train_end])
    test_masks = masks[test_start:]

    report = {
        "scope": "Synthetic distribution sanity check; no saved data or checkpoints loaded.",
        "seed": seed,
        "simulated_scenes": count,
        "generator_config": {
            "height": 50,
            "width": 100,
            "active_columns_matlab_inclusive": [35, 65],
            "min_on": 10,
            "min_off": 5,
            "lambda_on": 0.5,
            "lambda_off": 0.5,
            "Pfa": 0.01,
            "N": 100,
        },
        "estimated_active_channel_occupancy": occupancy,
        "always_predict_full_active_channel": {
            "micro_f1": 2 * sum(busy_counts) / (count * 50 + sum(busy_counts)),
            "macro_f1": statistics.mean(2 * busy / (50 + busy) for busy in busy_counts),
            "micro_and_macro_iou": occupancy,
        },
        "clean_template_overlap": {
            "unique_templates": len(template_counts),
            "largest_template_count": max(template_counts.values()),
            "fraction_beyond_first_occurrence": 1 - len(template_counts) / count,
            "test_template_seen_in_train_fraction": statistics.mean(
                mask in train_templates for mask in test_masks
            ),
            "caveat": "Clean-template overlap only; noisy observations have independent flips.",
        },
        "all_empty_scenes": busy_counts.count(0),
        "row_count_baselines": [],
    }

    q_inverse = statistics.NormalDist().inv_cdf(0.99)
    for threshold in [1, 2, 4, 16]:
        false_positive_rate = binomial_tail(0.01, threshold)
        by_snr = []
        for snr_db in range(-20, 6):
            gamma = 10 ** (snr_db / 10)
            pd = 0.5 * math.erfc(
                (q_inverse - math.sqrt(100) * gamma) / (1 + gamma) / math.sqrt(2)
            )
            recall = binomial_tail(pd, threshold)
            tp = occupancy * recall
            fp = (1 - occupancy) * false_positive_rate
            fn = occupancy * (1 - recall)
            by_snr.append({
                "snr_db": snr_db,
                "generator_Pd": pd,
                "analytic_recall": recall,
                "analytic_micro_f1": 2 * tp / (2 * tp + fp + fn),
                "analytic_micro_iou": tp / (tp + fp + fn),
            })
        report["row_count_baselines"].append({
            "rule": f"Mark row busy iff at least {threshold} of 31 active-channel pixels are one.",
            "threshold": threshold,
            "analytic_false_positive_rate_within_active_channel": false_positive_rate,
            "by_snr": by_snr,
        })

    report["interpretation_limits"] = [
        "Occupancy and template overlap are Monte Carlo estimates, not measurements on saved data.",
        "Analytic metrics are large-sample micro metrics under independent Bernoulli pixel flips.",
        "Configured active frequency interval is known to these baselines; evaluate an interval-blind variant too.",
        "Thresholds are illustrative fixed baselines, not validated or tuned on a held-out dataset.",
        "The row-count evaluation is analytic; it is not a measured runtime or trained-model comparison.",
    ]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.scenes < 10:
        parser.error("--scenes must be at least 10")
    print(json.dumps(run(args.scenes, args.seed), indent=2))
