"""Run the identical experiment with native BatchNorm and optimized convolutions.

Workaround for a Windows ROCm installation whose MIOpen BatchNorm kernel
compiler cannot find type_traits. Only BatchNorm dispatch changes; model layers,
data, loss and split stay unchanged. Runtime choice is saved in backend.json.
Use the same command arguments as experiment.py.
"""

from pathlib import Path

import torch

if __package__:
    from . import experiment
else:
    import experiment


def enable_native_batchnorm():
    """Scoped dispatch fallback; intended for a single training thread/process."""
    original = torch.nn.functional.batch_norm

    def native_batch_norm(*args, **kwargs):
        enabled = torch.backends.cudnn.enabled
        try:
            torch.backends.cudnn.enabled = False
            return original(*args, **kwargs)
        finally:
            torch.backends.cudnn.enabled = enabled

    torch.nn.functional.batch_norm = native_batch_norm


if __name__ == "__main__":
    arguments = experiment.parser().parse_args()
    enable_native_batchnorm()
    print("Using native BatchNorm; convolution backend unchanged.", flush=True)
    experiment.main()
    if arguments.command == "train":
        experiment.write_json(
            Path(arguments.run_dir) / "backend.json",
            {
                "entrypoint": "native_backend.py",
                "entrypoint_sha256": experiment.file_sha256(Path(__file__)),
                "native_batchnorm_only": True,
                "convolution_backend": "default",
                "reason": "Native backend avoids missing C++ type_traits header in installed MIOpen BatchNorm compiler.",
                "torch_version": str(torch.__version__),
            },
        )
