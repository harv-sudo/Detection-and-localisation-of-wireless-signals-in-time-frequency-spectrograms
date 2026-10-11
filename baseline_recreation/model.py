import torch
import torch.nn as nn


class DLANNBaseline(nn.Module):
    """
    Paper-style DL-ANN baseline for Signal Area Estimation.

    Input:
        flattened binary spectrogram, shape (batch, 5000)

    Output:
        20 boxes × 4 values = 80 values
        format: [x, y, w, h, x, y, w, h, ...]
    """

    def __init__(
        self,
        input_dim: int = 5000,
        hidden_dim: int = 256,
        output_dim: int = 80,
        num_hidden_layers: int = 4,
        dropout: float = 0.4,
    ):
        super().__init__()

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


if __name__ == "__main__":
    model = DLANNBaseline()

    dummy = torch.randn(4, 5000)
    out = model(dummy)

    print(model)
    print("Output shape:", out.shape)
