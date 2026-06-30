from __future__ import annotations

import torch
from torch import nn

from models.dual_head_model import DualHeadCNNGRU


class HeterogeneousSensorAdapterModel(nn.Module):
    def __init__(
        self,
        client_input_sizes: dict[int, int],
        shared_input_size: int,
        config: dict,
    ) -> None:
        super().__init__()
        model_cfg = config["model"]
        self.adapters = nn.ModuleDict(
            {
                str(client_id): nn.Sequential(
                    nn.Linear(input_size, shared_input_size),
                    nn.LayerNorm(shared_input_size),
                    nn.ReLU(),
                    nn.Dropout(model_cfg["dropout"]),
                )
                for client_id, input_size in client_input_sizes.items()
            }
        )
        self.shared_model = DualHeadCNNGRU(
            input_size=shared_input_size,
            conv_channels=model_cfg["conv_channels"],
            kernel_size=model_cfg["kernel_size"],
            gru_hidden=model_cfg["gru_hidden"],
            dropout=model_cfg["dropout"],
        )

    def forward(self, x: torch.Tensor, client_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        adapter = self.adapters[str(client_id)]
        x = adapter(x)
        return self.shared_model(x)


def build_heterogeneous_sensor_model(
    client_input_sizes: dict[int, int],
    config: dict,
) -> HeterogeneousSensorAdapterModel:
    rq1_cfg = config["rq1"]
    return HeterogeneousSensorAdapterModel(
        client_input_sizes=client_input_sizes,
        shared_input_size=int(rq1_cfg["shared_input_size"]),
        config=config,
    )
