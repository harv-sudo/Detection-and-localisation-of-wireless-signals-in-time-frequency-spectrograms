"""Efficient threshold sweeps with the frozen experiment's pixel metric rules.

Predictions use a strict ``probability > threshold`` comparison. A histogram
counts each pixel once, avoiding a separate image-sized mask per threshold.
"""

import math

import torch

if __package__:
    from .experiment import bce_dice_loss, scores
else:
    from experiment import bce_dice_loss, scores


_SCORE_NAMES = ("f1", "iou", "precision", "recall")


def _prepare_thresholds(thresholds):
    values = [float(value) for value in thresholds]
    if not values or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Provide at least one finite threshold between 0 and 1.")
    boundaries = torch.tensor(values, dtype=torch.float32)
    if boundaries.unique().numel() != len(values):
        raise ValueError("Thresholds must be distinct after float32 conversion.")
    boundaries, order = boundaries.sort()
    return values, boundaries, order.argsort()


def _counts(probabilities, targets, boundaries, original_order):
    pixels = probabilities.flatten(1)
    truth = (targets > 0.5).flatten(1)
    bins = boundaries.numel() + 1
    # right=False assigns an exact boundary value below that threshold. Thus
    # bucket k exceeds exactly the first k thresholds, matching strict >.
    buckets = torch.bucketize(pixels.contiguous(), boundaries, right=False)
    buckets = buckets + truth.to(torch.int64) * bins
    histogram = torch.zeros((pixels.size(0), 2 * bins), dtype=torch.int64, device=pixels.device)
    histogram.scatter_add_(1, buckets, torch.ones_like(buckets))
    negative, positive = histogram[:, :bins], histogram[:, bins:]
    positive_total = positive.sum(1, keepdim=True)
    tp = positive_total - positive.cumsum(1)[:, :-1]
    fp = negative.sum(1, keepdim=True) - negative.cumsum(1)[:, :-1]
    fn = positive_total - tp
    return torch.stack((tp, fp, fn), dim=-1).transpose(0, 1)[original_order]


def threshold_counts(probabilities, targets, thresholds):
    """Return int64 counts shaped ``[threshold, image, (TP, FP, FN)]``.

    Thresholds may be supplied in any order, which is preserved in the result.
    Comparisons are performed in float32, as in validation. Input probabilities
    must be finite and in [0, 1], and targets must share their shape/device.
    """
    _, boundaries, order = _prepare_thresholds(thresholds)
    if probabilities.ndim < 2 or probabilities.size(0) == 0 or probabilities.numel() == 0:
        raise ValueError("Expected a non-empty batch with at least one pixel per image.")
    if probabilities.shape != targets.shape or probabilities.device != targets.device:
        raise ValueError("Probabilities and targets must have matching shapes and devices.")
    probabilities = probabilities.float()
    if not bool(torch.isfinite(probabilities).all()) or bool(((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Probabilities must be finite and lie between 0 and 1.")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("Targets must be finite.")
    return _counts(probabilities, targets, boundaries.to(probabilities.device), order.to(probabilities.device))


def _per_image_scores(counts):
    tp, fp, fn = counts.to(torch.float64).unbind(-1)
    empty = tp + fp + fn == 0
    return torch.stack(
        (
            torch.where(empty, 1.0, 2 * tp / (2 * tp + fp + fn).clamp_min(1)),
            torch.where(empty, 1.0, tp / (tp + fp + fn).clamp_min(1)),
            torch.where(tp + fp > 0, tp / (tp + fp).clamp_min(1), empty.to(torch.float64)),
            torch.where(tp + fn > 0, tp / (tp + fn).clamp_min(1), empty.to(torch.float64)),
        ),
        dim=-1,
    )


class _MetricTotals:
    def __init__(self, thresholds):
        self.samples = self.empty_targets = 0
        self.counts = torch.zeros((thresholds, 3), dtype=torch.int64)
        self.score_sums = torch.zeros((thresholds, 4), dtype=torch.float64)

    def add(self, counts, per_image_scores):
        self.samples += counts.size(1)
        self.empty_targets += int((counts[0, :, 0] + counts[0, :, 2] == 0).sum())
        self.counts += counts.sum(1)
        self.score_sums += per_image_scores.sum(1)

    def result(self, index):
        if not self.samples:
            raise ValueError("Cannot report metrics for an empty split.")
        tp, fp, fn = self.counts[index].tolist()
        return {
            "samples": self.samples,
            "empty_targets": self.empty_targets,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "micro": scores(tp, fp, fn),
            "macro": dict(zip(_SCORE_NAMES, (self.score_sums[index] / self.samples).tolist())),
        }


@torch.inference_mode()
def assess(model, loader, device, thresholds):
    """Match experiment.assess reports, with vectorized threshold aggregation.

    Inference and loss use float32 even inside an outer autocast context. Macro
    scores are accumulated in CPU float64; micro scores use exact integer sums.
    """
    thresholds, boundaries, order = _prepare_thresholds(thresholds)
    device = torch.device(device)
    boundaries, order = boundaries.to(device), order.to(device)
    overall = _MetricTotals(len(thresholds))
    by_snr = {}
    total_loss = 0.0
    model.eval()
    with torch.autocast(device_type=device.type, enabled=False):
        for inputs, targets, snrs in loader:
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
            logits = model(inputs).float()
            if logits.shape != targets.shape:
                raise ValueError("Model logits and targets must have matching shapes.")
            loss = bce_dice_loss(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite evaluation loss.")
            total_loss += float(loss) * inputs.size(0)
            counts = _counts(logits.sigmoid(), targets, boundaries, order).cpu()
            image_scores = _per_image_scores(counts)
            overall.add(counts, image_scores)
            snr_values = snrs.detach().cpu().flatten().tolist()
            if len(snr_values) != inputs.size(0) or any(not math.isfinite(snr) for snr in snr_values):
                raise ValueError("Expected one finite SNR value per image.")
            positions = {}
            for index, snr in enumerate(snr_values):
                positions.setdefault(f"{snr:g}", []).append(index)
            for key, indices in positions.items():
                if key not in by_snr:
                    by_snr[key] = _MetricTotals(len(thresholds))
                by_snr[key].add(counts[:, indices], image_scores[:, indices])
    if not overall.samples:
        raise ValueError("Cannot report metrics for an empty split.")
    return [
        {
            "threshold": threshold,
            "loss": total_loss / overall.samples,
            "overall": overall.result(index),
            "per_snr": {
                key: by_snr[key].result(index) for key in sorted(by_snr, key=float)
            },
        }
        for index, threshold in enumerate(thresholds)
    ]
