"""Read-only model profiling for the implementation review.

Run with a Python environment containing PyTorch. No datasets or checkpoints are
needed. Width variants are created in memory; repository model files are unchanged.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch


REPOSITORY = Path(__file__).resolve().parents[1]


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "review_unet_model", REPOSITORY / "UNet" / "model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_variant(module, base_width):
    model = module.UNet()
    if base_width != 32:
        model.enc1 = module.DoubleConv(1, base_width)
        model.enc2 = module.Down(base_width, base_width * 2)
        model.enc3 = module.Down(base_width * 2, base_width * 4)
        model.bottleneck = module.Down(base_width * 4, base_width * 8)
        model.dec3 = module.Up(base_width * 8, base_width * 4, base_width * 4)
        model.dec2 = module.Up(base_width * 4, base_width * 2, base_width * 2)
        model.dec1 = module.Up(base_width * 2, base_width, base_width)
        model.output = torch.nn.Conv2d(base_width, 1, kernel_size=1)
    return model.eval()


def profile_model(model, base_width):
    layers = []
    handles = []

    def hook_for(name):
        def record(layer, inputs, output):
            kernel_area = layer.kernel_size[0] * layer.kernel_size[1]
            if isinstance(layer, torch.nn.ConvTranspose2d):
                macs = (
                    inputs[0].numel()
                    * (layer.out_channels // layer.groups)
                    * kernel_area
                )
            else:
                macs = (
                    output.numel() * (layer.in_channels // layer.groups) * kernel_area
                )
            layers.append(
                {"name": name, "output_shape": list(output.shape), "macs": macs}
            )

        return record

    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            handles.append(layer.register_forward_hook(hook_for(name)))

    try:
        with torch.inference_mode():
            output = model(torch.zeros(1, 1, 50, 100))
    finally:
        for handle in handles:
            handle.remove()

    assert list(output.shape) == [1, 1, 50, 100]
    return {
        "base_width": base_width,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "convolution_macs": sum(layer["macs"] for layer in layers),
        "output_shape": list(output.shape),
        "layers": layers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    torch.set_num_threads(2)
    module = load_model_module()
    result = {
        "torch_version": str(torch.__version__),
        "device": "cpu",
        "input_shape": [1, 1, 50, 100],
        "method": "Analytical convolution multiply-accumulate counts from forward hooks",
        "limitations": [
            "No accuracy or latency measurements; width variants are untrained.",
            "Excludes batch normalization, activation, pooling, interpolation, bias additions and memory traffic.",
            "One multiply-accumulate is counted as one MAC, not two floating-point operations.",
        ],
        "variants": [
            profile_model(make_variant(module, width), width) for width in (32, 16, 8)
        ],
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
