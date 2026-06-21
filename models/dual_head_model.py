from __future__ import annotations

import torch
from torch import nn


class DualHeadCNNGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        conv_channels: list[int] | tuple[int, ...] = (64, 128),
        kernel_size: int = 3,
        gru_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_size
        padding = kernel_size // 2
        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=gru_hidden,
            batch_first=True,
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(gru_hidden),
            nn.Dropout(dropout),
        )
        self.rul_head = nn.Linear(gru_hidden, 1)
        self.fault_head = nn.Linear(gru_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, window, sensors]
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        _, hidden = self.gru(x)
        features = self.shared(hidden[-1])
        rul = self.rul_head(features).squeeze(-1)
        fault_logit = self.fault_head(features).squeeze(-1)
        return rul, fault_logit


def build_model(input_size: int, config: dict) -> DualHeadCNNGRU:
    model_cfg = config["model"]
    return DualHeadCNNGRU(
        input_size=input_size,
        conv_channels=model_cfg["conv_channels"],
        kernel_size=model_cfg["kernel_size"],
        gru_hidden=model_cfg["gru_hidden"],
        dropout=model_cfg["dropout"],
    )
