from __future__ import annotations

import copy

import torch


def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(model)


def fedavg_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    sample_counts: list[int],
) -> dict[str, torch.Tensor]:
    if len(state_dicts) != len(sample_counts):
        raise ValueError("state_dicts and sample_counts must have the same length")
    if not state_dicts:
        raise ValueError("No client states provided")

    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("Total sample count must be positive")

    averaged = {}
    for key in state_dicts[0].keys():
        value = None
        for state, count in zip(state_dicts, sample_counts):
            weighted = state[key].detach().cpu().float() * (count / total)
            value = weighted if value is None else value + weighted
        averaged[key] = value
    return averaged
