"""Measure architecture counts and warmed forward latency on the same device.

Uses fixed binary inputs and eval-mode models. This is model-only latency,
excluding loading, transfers and output decoding, not end-to-end detection time.
Run after training has finished so concurrent GPU work does not bias timings.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

if __package__:
    from .experiment import choose_device, make_model, positive_int
else:
    from experiment import choose_device, make_model, positive_int


def count_convolution_macs(model, inputs):
    counts, handles = [], []

    def count(layer, arguments, output):
        kernel = layer.kernel_size[0] * layer.kernel_size[1]
        if isinstance(layer, torch.nn.ConvTranspose2d):
            counts.append(arguments[0].numel() * (layer.out_channels // layer.groups) * kernel)
        else:
            counts.append(output.numel() * (layer.in_channels // layer.groups) * kernel)

    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            handles.append(layer.register_forward_hook(count))
    try:
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return sum(counts)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark(args):
    torch.set_num_threads(args.threads)
    if args.native_backend:
        torch.backends.cudnn.enabled = False
    if args.native_batchnorm:
        if __package__:
            from .native_backend import enable_native_batchnorm
        else:
            from native_backend import enable_native_batchnorm
        enable_native_batchnorm()
    torch.manual_seed(42)
    device = choose_device(args.device)
    records = []
    # Create on CPU and copy once, outside the timed region.
    inputs = torch.randint(0, 2, (args.batch_size, 1, 50, 100)).float().to(device)
    for name in ('unet', 'evidence_axial'):
        config = dict(model=name, base_channels=16, refinement_steps=3,
                      no_evidence=False, no_refinement=False)
        model = make_model(config).to(device).eval()
        macs = count_convolution_macs(model, inputs[:1])
        for _ in range(args.warmup):
            model(inputs)
        synchronize(device)
        milliseconds = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            model(inputs)
            synchronize(device)
            milliseconds.append((time.perf_counter() - started) * 1000)
        ordered = sorted(milliseconds)
        p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
        records.append(dict(model=name, parameters=sum(p.numel() for p in model.parameters()),
                            convolution_macs_per_image=macs,
                            median_batch_ms=statistics.median(milliseconds), p95_batch_ms=p95,
                            mean_batch_ms=statistics.mean(milliseconds),
                            amortized_mean_ms_per_image=statistics.mean(milliseconds) / args.batch_size))
        del model
    result = dict(device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                  torch_version=str(torch.__version__), dtype='float32', batch_size=args.batch_size,
                  cudnn_miopen_enabled=torch.backends.cudnn.enabled,
                  native_batchnorm_only=args.native_batchnorm,
                  input_shape=list(inputs.shape), cpu_threads=args.threads, warmup=args.warmup,
                  iterations=args.iterations, records=records,
                  limitations=['Random weights; architecture benchmark, not accuracy evaluation.',
                               'Synchronized host-wall forward timings include Python dispatch and synchronization.',
                               'Excludes data loading, host/device transfer and prediction postprocessing.',
                               'MAC counts exclude pooling, normalization, activation, interpolation, refinement and memory traffic.',
                               'Do not compare timings obtained with concurrent accelerator workloads.'])
    payload = json.dumps(result, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(payload, encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=positive_int, default=1)
    parser.add_argument('--threads', type=positive_int, default=4)
    parser.add_argument('--warmup', type=positive_int, default=20)
    parser.add_argument('--iterations', type=positive_int, default=100)
    parser.add_argument('--output')
    backends = parser.add_mutually_exclusive_group()
    backends.add_argument('--native-backend', action='store_true', help='Use native PyTorch operations for both models')
    backends.add_argument('--native-batchnorm', action='store_true', help='Use native BatchNorm for both models, preserving optimized convolutions')
    benchmark(parser.parse_args())
