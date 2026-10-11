"""Optional CPU graph preparation, with an explicit fixed spatial-size guard.

The installed CPU backend uses slow fallback kernels for some small dense
convolutions. PyTorch's inference graph pass selects/fuses oneDNN kernels.
Prepare once outside the timed/request path. Recreate the graph after loading
weights: optimized oneDNN graphs are not a portable serialization format.
"""
import copy
import warnings

import torch
from torch import nn


@torch.no_grad()
def replace_nonoverlapping_upsampling(model):
    """Copy: exact 2x2/stride-2 transposed convolution as 1x1 + pixel shuffle.

    With no overlapping output patches, each input pixel independently produces
    four output pixels. The weight permutation is algebraic, not an approximation.
    Other transposed convolutions are left untouched.
    """
    result = copy.deepcopy(model)

    def replace(parent):
        for name, child in list(parent.named_children()):
            if (isinstance(child, nn.ConvTranspose2d) and child.kernel_size == (2, 2)
                    and child.stride == (2, 2) and child.padding == (0, 0)
                    and child.output_padding == (0, 0) and child.dilation == (1, 1)
                    and child.groups == 1):
                projection = nn.Conv2d(child.in_channels, child.out_channels * 4, 1,
                                       bias=child.bias is not None,
                                       device=child.weight.device, dtype=child.weight.dtype)
                projection.weight.copy_(child.weight.permute(1, 2, 3, 0).reshape(
                    child.out_channels * 4, child.in_channels, 1, 1))
                if child.bias is not None:
                    projection.bias.copy_(child.bias.repeat_interleave(4))
                setattr(parent, name, nn.Sequential(projection, nn.PixelShuffle(2)).eval())
            else:
                replace(child)
    replace(result)
    return result


class PreparedCPUModel(nn.Module):
    def __init__(self, graph, height, width):
        super().__init__()
        self.graph, self.height, self.width = graph, height, width
        self.eval()

    def train(self, mode=True):
        if mode:
            raise ValueError("A prepared inference graph cannot be trained")
        return super().train(False)

    def forward(self, x):
        if (x.device.type != "cpu" or x.dtype != torch.float32 or x.ndim != 4
                or x.shape[0] < 1 or tuple(x.shape[1:]) != (1, self.height, self.width)):
            raise ValueError(f"Prepared graph expects FP32 CPU Bx1x{self.height}x{self.width}")
        return self.graph(x)


@torch.inference_mode()
def prepare_cpu(model, example):
    if model.training:
        raise ValueError("Call eval() before preparing an inference graph")
    if example.device.type != "cpu" or example.dtype != torch.float32:
        raise ValueError("CPU graph preparation requires an FP32 CPU example")
    # Tracing specializes spatial branches. The wrapper enforces those sizes;
    # batch size remains dynamic and is regression-tested separately.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(replace_nonoverlapping_upsampling(model), example)
    graph = torch.jit.optimize_for_inference(traced)
    return PreparedCPUModel(graph, *example.shape[-2:])
