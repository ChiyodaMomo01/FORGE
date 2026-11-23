"""Full-grid Gaussian policy, physical decode and Monte Carlo credit."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..core.sampling import spatial_cost
from ..paired.model import prepare_observation


def raw_mean(actor, obs):
    """Works with already-loaded Component objects that have no raw() method."""
    obs = prepare_observation(obs, next(actor.parameters()).dtype)
    return actor.network(obs) * float(actor.amplitude)


def full_returns(costs, *, global_credit=False, groups=None):
    """Interval MC suffix returns, including unpaid absorbing failure tails."""
    costs = np.asarray(costs)
    if costs.ndim != 4 or not len(costs) or not costs.shape[1] or not np.isfinite(costs).all():
        raise ValueError("finite complete parent/time/field costs required")
    reward = costs.mean((-2, -1), keepdims=True) if global_credit else costs
    if global_credit and groups is not None:
        groups = [list(group) for group in groups]
        if any(not group for group in groups) or sorted(
            i for group in groups for i in group
        ) != list(range(len(costs))):
            raise ValueError("credit groups must partition all parent indices")
        grouped = np.empty_like(reward)
        for group in groups:
            grouped[group] = reward[group].mean(axis=0, keepdims=True)
        reward = grouped
    return (-np.cumsum(reward[:, ::-1], axis=1, dtype=np.float64)[:, ::-1] / costs.shape[1]).astype(
        np.float32
    )


class SpatialCritic(nn.Module):
    """Full-grid periodic critic: current observation and remaining time only."""

    def __init__(self, observation_channels, width=32, input_mean=None, input_std=None):
        super().__init__()
        self.register_buffer(
            "input_mean",
            torch.zeros(1, observation_channels, 1, 1)
            if input_mean is None
            else input_mean.detach().reshape(1, observation_channels, 1, 1).clone(),
        )
        self.register_buffer(
            "input_std",
            torch.ones(1, observation_channels, 1, 1)
            if input_std is None
            else input_std.detach().reshape(1, observation_channels, 1, 1).clone(),
        )
        self.layers = nn.Sequential(
            nn.Conv2d(observation_channels + 1, width, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv2d(width, 1, 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, obs, intervals, horizon):
        obs = (obs.float() - self.input_mean) / self.input_std
        remaining = (1 - intervals.to(obs) / horizon)[:, None, None, None]
        return self.layers(torch.cat((obs, remaining.expand(-1, 1, *obs.shape[-2:])), 1))[:, 0]


__all__ = ["SpatialCritic", "full_returns", "raw_mean", "spatial_cost"]
