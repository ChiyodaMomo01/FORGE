"""Residual neural closure parameterization: G_total(x) = G_coarse(x) + 0.1 * G_theta(x)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..networks import NetworkConfig, PeriodicFNO
from ..schemas import CANONICAL_CHECKPOINT_SEMANTICS
from .coarse import coarse_value

SEMANTICS = CANONICAL_CHECKPOINT_SEMANTICS


def prepare_observation(obs, dtype):
    obs = obs.to(dtype)
    if obs.shape[1] < 8:
        obs = F.pad(obs, (0, 0, 0, 0, 0, 8 - obs.shape[1]))
    if obs.shape[1] != 8:
        raise ValueError("at most eight public input channels")
    return obs


CASE_CHANNELS = {
    "cahn_hilliard": (2, 1),
    "gray_scott": (4, 1),
    "kolmogorov_ns": (6, 2),
    "mhd2d": (8, 3),
}

CASE_SPECTRAL_DEFAULTS = {
    "cahn_hilliard": (0, 0, 1),
    "gray_scott": (0, 0, 1),
    "kolmogorov_ns": (14, 16, 6),
    "mhd2d": (14, 16, 6),
}


def project_action(case_name, value):
    if case_name not in CASE_CHANNELS:
        raise ValueError(f"unsupported 2D benchmark case: {case_name}")
    return value


class NormalizedNetwork(PeriodicFNO):
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("input_mean", torch.zeros(1, 8, 1, 1))
        self.register_buffer("input_std", torch.ones(1, 8, 1, 1))

    def _normalize_input(self, x):
        x = prepare_observation(x, self.input_mean.dtype)
        return (x - self.input_mean) / self.input_std

    def forward(self, x):
        return super().forward(self._normalize_input(x))

    def split_head(self, x, detach_tail_trunk=False):
        return super().split_head(self._normalize_input(x), detach_tail_trunk)


class FullComponent(nn.Module):
    output_semantics = SEMANTICS
    residual_coefficient = 0.1

    def __init__(
        self,
        case_name,
        n=96,
        width=29,
        statistics=None,
        modes=6,
        layers=4,
        radial_bins=None,
        bilinear_channels=None,
        head_bands=None,
        nyquist_gate=False,
        nyquist_cutoff=42.0,
        nyquist_tau=5e-4,
    ):
        super().__init__()
        if case_name not in CASE_CHANNELS:
            raise ValueError(f"unsupported 2D benchmark case: {case_name}")
        def_rad, def_bilin, def_bands = CASE_SPECTRAL_DEFAULTS[case_name]
        radial_bins = def_rad if radial_bins is None else int(radial_bins)
        bilinear_channels = def_bilin if bilinear_channels is None else int(bilinear_channels)
        head_bands = def_bands if head_bands is None else int(head_bands)

        self.case_name = case_name
        self.observation_channels, channels = CASE_CHANNELS[case_name]
        self.action_channels = channels
        self.radial_bins = radial_bins
        self.bilinear_channels = bilinear_channels
        self.head_bands = head_bands
        self.nyquist_gate = bool(nyquist_gate)
        self.nyquist_cutoff = float(nyquist_cutoff)
        self.nyquist_tau = float(nyquist_tau)
        self.network = NormalizedNetwork(
            NetworkConfig(
                nx=n,
                action_grid=n,
                width=width,
                action_channels=channels,
                modes=modes,
                layers=layers,
                radial_bins=radial_bins,
                bilinear_channels=bilinear_channels,
                head_bands=head_bands,
                nyquist_gate=self.nyquist_gate,
                nyquist_cutoff=self.nyquist_cutoff,
                nyquist_tau=self.nyquist_tau,
            )
        )
        self.register_buffer("action_std", torch.ones(1, channels, 1, 1))
        self.register_buffer("output_scale", torch.ones(1, channels, 1, 1))
        self.amplitude = 1.0
        if statistics is not None:
            for name in ("input_mean", "input_std"):
                v = torch.as_tensor(statistics[name], dtype=torch.float32)
                getattr(self.network, name)[:, : len(v)] = v.reshape(1, -1, 1, 1)
            scale = (
                torch.as_tensor(statistics["g_theta_scale"])
                if "g_theta_scale" in statistics
                else torch.as_tensor(statistics["residual_rms"]) / self.residual_coefficient
            )
            self.output_scale.copy_(scale.reshape(1, -1, 1, 1))

    def forward(self, obs):
        return self.network(obs) * self.amplitude

    def split_predictions(self, obs, detach_tail_trunk=False):
        closure, tail = self.network.split_head(obs, detach_tail_trunk=detach_tail_trunk)
        return closure * self.amplitude, tail * self.amplitude

    def correction(self, coordinates):
        return self.residual_coefficient * coordinates * self.output_scale.to(coordinates)

    def physical(self, obs):
        return coarse_value(self.case_name, obs) + self.correction(self(obs)).to(obs.dtype)

    def residual_target(self, obs, y):
        return (y - coarse_value(self.case_name, obs)) / (
            self.residual_coefficient * self.output_scale.to(y)
        )


def vector(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()


def assign(model, values):
    parameter = next(model.parameters())
    torch.nn.utils.vector_to_parameters(values.detach().to(parameter).clone(), model.parameters())
