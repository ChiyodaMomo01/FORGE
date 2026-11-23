"""Periodic reaction and phase-field cases with internal, differentiable closure slots.

All spatial values represent cell-centred finite-volume fields. The Laplacian is
the periodic five-point operator, diagonalized by FFT; it is not a continuum
spectral derivative. Restriction is conservative block averaging. Consequently
the nonlinear labels below do not claim to remove every discretization error.
No concentration/order-parameter clipping or fallback is performed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GrayScottConfig:
    ood: bool = False
    length: float = 2.5
    diffusion_u: float = 2.0e-5
    diffusion_v: float = 1.0e-5
    feed: float = 0.035
    kill: float = 0.062
    dt: float = 4.0
    max_substep: float = 1.0
    action_scale: float = 0.01


@dataclass(frozen=True)
class CahnHilliardConfig:
    ood: bool = False
    length: float = 1.0
    mobility: float = 0.001
    kappa: float = 0.00015625  # fixed physical interface length sqrt(kappa)=.0125
    stabilization: float = 2.0
    dt: float = 0.02
    action_scale: float = 0.1
    mean: float = -0.1


def _config(cls, config):
    if config is None:
        return cls()
    if isinstance(config, Mapping):
        values = dict(config)
        # Shared factory execution hint: autograd permissions belong to the
        # caller's no_grad/inference_mode context, not the physical parameters.
        values.pop("differentiable", None)
        dtype_hint = values.pop("dtype", None)
        if dtype_hint not in (None, "float64", torch.float64):
            raise ValueError(
                f"Reaction cases require float64 state evolution, got dtype={dtype_hint!r}"
            )
        return cls(**values)
    return config


def _restrict(x: torch.Tensor, n: int) -> torch.Tensor:
    old = x.shape[-1]
    if x.shape[-2] != old or n < 1 or old % n:
        raise ValueError("Restriction requires square grids and an integer refinement ratio")
    return x if old == n else F.avg_pool2d(x, old // n)


def _lap(x: torch.Tensor, length: float) -> torch.Tensor:
    h = length / x.shape[-1]
    return (x.roll(1, -1) + x.roll(-1, -1) + x.roll(1, -2) + x.roll(-1, -2) - 4 * x) / (h * h)


class _PeriodicCase:
    spatial_dim = 2

    def __init__(self):
        self._symbols: dict[tuple, torch.Tensor] = {}

    def _symbol(self, x):
        n = x.shape[-1]
        key = (n, x.device, x.dtype)
        if key not in self._symbols:
            q = torch.arange(n, dtype=x.dtype, device=x.device)
            k = 4 * torch.sin(math.pi * q / n).square() / (self.config.length / n) ** 2
            self._symbols[key] = -(k[:, None] + k[None, : n // 2 + 1])
        return self._symbols[key]

    def fields(self, state):
        return state["fields"]

    def state_from_fields(self, fields, time=0.0):
        if (
            fields.ndim != 4
            or fields.shape[1] != self.field_channels
            or fields.shape[-1] != fields.shape[-2]
        ):
            raise ValueError(
                "Expected batched square fields with the case's physical channel count"
            )
        return {"fields": fields, "time": float(time)}

    def default_dt(self, n):
        return self.config.dt

    def restrict(self, state, n):
        return {**state, "fields": _restrict(self.fields(state), n)}

    def _action(self, x, model):
        if model is None:
            return torch.zeros_like(x[:, :1])
        action = model(self.observe({"fields": x}))
        expected = (x.shape[0], self.action_channels, *x.shape[-2:])
        if action.shape != expected:
            raise ValueError(f"Expected model action {expected}, got {tuple(action.shape)}")
        return action * self.config.action_scale

    def _grid(self, seed, n, device):
        if n < 8:
            raise ValueError("Use at least 8 cells per direction")
        # CPU generator makes initial realizations identical across CPU/GPU.
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        q = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n
        yy, xx = torch.meshgrid(q, q, indexing="ij")
        return gen, xx, yy


class GrayScottCase(_PeriodicCase):
    """Strang diffusion / midpoint reaction; closure is evaluated at both stages."""

    name = "gray_scott"
    field_channels = 2
    observation_channels = 4
    action_channels = 1

    def __init__(self, config=None):
        super().__init__()
        self.config = _config(GrayScottConfig, config)
        if (
            min(
                self.config.length,
                self.config.diffusion_u,
                self.config.diffusion_v,
                self.config.dt,
                self.config.max_substep,
            )
            <= 0
        ):
            raise ValueError("Length, diffusion coefficients, and timesteps must be positive")

    def initial(self, seed, n, device="cpu"):
        gen, xx, yy = self._grid(seed, n, device)
        spots = torch.rand((32 if self.config.ood else 16, 3), generator=gen).to(device)
        blob = torch.zeros_like(xx)
        for cx, cy, radius in spots:
            dx = torch.remainder(xx - cx + 0.5, 1.0) - 0.5
            dy = torch.remainder(yy - cy + 0.5, 1.0) - 0.5
            width = 0.018 + 0.018 * radius
            bump = torch.exp(-(dx.square() + dy.square()) / (2 * width.square()))
            blob = 1 - (1 - blob) * (1 - bump)
        x = torch.stack((1 - 0.55 * blob, 0.30 * blob))[None]
        return {"fields": x, "time": 0.0}

    def observe(self, state):
        x = self.fields(state)
        d = x.new_tensor([self.config.diffusion_u, self.config.diffusion_v])[None, :, None, None]
        return torch.cat((x, d * _lap(x, self.config.length) / 0.05), 1)

    def g(self, public_input):
        """Actual nonlinear reaction rate u * v^2 in physical units."""
        return public_input[:, :1] * public_input[:, 1:2].square()

    def _reaction(self, x, model):
        u, v = x[:, :1], x[:, 1:2]
        if getattr(model, "output_semantics", None) == "full-g-v1":
            rate = model(self.observe({"fields": x}))
        else:
            rate = self.g(self.observe({"fields": x}))
            if model is not None:
                rate = rate + self._action(x, model)
        return torch.cat(
            (
                -rate + self.config.feed * (1 - u),
                rate - (self.config.feed + self.config.kill) * v,
            ),
            1,
        )

    def step(self, state, model=None, dt=None):
        x = self.fields(state)
        interval = self.default_dt(x.shape[-1]) if dt is None else float(dt)
        if interval <= 0:
            raise ValueError("dt must be positive")
        count = max(1, math.ceil(interval / self.config.max_substep))
        h = interval / count
        d = x.new_tensor([self.config.diffusion_u, self.config.diffusion_v])[None, :, None, None]
        decay = torch.exp(0.5 * h * d * self._symbol(x))

        def diffuse(y):
            return torch.fft.irfft2(torch.fft.rfft2(y) * decay, s=y.shape[-2:])

        for _ in range(count):
            a = diffuse(x)
            midpoint = a + 0.5 * h * self._reaction(a, model)
            x = diffuse(a + h * self._reaction(midpoint, model))
        return {**state, "fields": x, "time": state.get("time", 0.0) + interval}

    def closure_target(self, fine_state, n):
        """Normalized unresolved reaction label, available only with HR permissions."""
        x = self.fields(fine_state)
        coarse = _restrict(x, n)
        return (
            _restrict(x[:, :1] * x[:, 1:2].square(), n) - coarse[:, :1] * coarse[:, 1:2].square()
        ) / self.config.action_scale

    def classical_closure(self, observation):
        """Leading box-filter gradient expansion; no fitted coefficient or HR input."""
        u, v = observation[:, :1], observation[:, 1:2]
        rate = torch.zeros_like(u)
        for axis in (-1, -2):
            du = 0.5 * (u.roll(-1, axis) - u.roll(1, axis))
            dv = 0.5 * (v.roll(-1, axis) - v.roll(1, axis))
            rate = rate + u * dv.square() + 2 * v * du * dv
        return rate / (12 * self.config.action_scale)

    def diagnostics(self, state):
        x = self.fields(state)
        return {
            "mass": x.mean((-2, -1)),
            "minimum": x.amin((-3, -2, -1)),
            "maximum": x.amax((-3, -2, -1)),
            "finite": x.isfinite().all(),
        }

    def metadata(self):
        return {
            "name": self.name,
            "config": asdict(self.config),
            "boundary": "periodic",
            "integrator": "Strang exact discrete diffusion / explicit midpoint reaction",
            "restriction": "conservative cell block average",
            "clipping": False,
            "closure": "one reaction rate with opposite signs in u and v at both reaction stages",
            "positivity": "not guaranteed; negative concentrations must be reported",
            "observation_scales": [1.0, 1.0, 0.05, 0.05],
            "observation_definition": "u, v, D_u Lap(u)/.05, D_v Lap(v)/.05",
            "field_scales": [1.0, 1.0],
            "action_scale": self.config.action_scale,
            "initial_dtype": "float32; cast fields before evolution for float64 campaigns",
            "ood_regime": "32 initial spots instead of 16; physical coefficients unchanged",
            "autograd": "supported; forward-only permissions are enforced by caller context",
            "batched": True,
        }


class CahnHilliardCase(_PeriodicCase):
    """First-order stabilized IMEX CH; correction enters mu before its Laplacian.

    (I - dt M S Lap + dt M kappa Lap^2)c_next
      = c + dt M Lap[c^3 - (1+S)c + delta_mu].
    S is numerical stabilization, not a change of the physical free energy.
    Mass is preserved to floating-point precision for arbitrary actions. Neither
    arbitrary learned actions nor unbounded c are promised to dissipate energy.
    """

    name = "cahn_hilliard"
    field_channels = 1
    observation_channels = 2
    action_channels = 1

    def __init__(self, config=None):
        super().__init__()
        self.config = _config(CahnHilliardConfig, config)
        if (
            min(
                self.config.length,
                self.config.mobility,
                self.config.kappa,
                self.config.dt,
            )
            <= 0
        ):
            raise ValueError("Length, mobility, kappa, and dt must be positive")
        if self.config.stabilization < 0:
            raise ValueError("stabilization must be nonnegative")

    def initial(self, seed, n, device="cpu"):
        gen, xx, yy = self._grid(seed, n, device)
        cutoff = 16 if self.config.ood else 12
        modes = torch.randint(-cutoff, cutoff + 1, (32, 2), generator=gen)
        phases = 2 * math.pi * torch.rand(32, generator=gen)
        amplitudes = torch.randn(32, generator=gen)
        x = torch.zeros_like(xx)
        for mode, phase, amplitude in zip(modes, phases, amplitudes, strict=True):
            x = x + amplitude.to(device) * torch.cos(
                2 * math.pi * (int(mode[0]) * xx + int(mode[1]) * yy) + phase.to(device)
            )
        x = x - x.mean()
        # Fixed coefficient normalization, independent of n and no value clipping.
        mean = self.config.mean + (0.25 if self.config.ood else 0.0)
        x = mean + 0.15 * x / torch.sqrt(amplitudes.square().sum().to(device) / 2)
        return {"fields": x[None, None], "time": 0.0}

    def observe(self, state):
        x = self.fields(state)
        return torch.cat((x, self.config.kappa * _lap(x, self.config.length)), 1)

    def g(self, public_input):
        """Actual nonlinear component value; no state/history in this interface.

        The public input is (c, kappa*Lap(c)); output is the full cubic
        chemical-potential component in physical units, NOT a correction.
        """
        return public_input[:, :1].pow(3)

    def step(self, state, model=None, dt=None):
        x = self.fields(state)
        h = self.default_dt(x.shape[-1]) if dt is None else float(dt)
        if h <= 0:
            raise ValueError("dt must be positive")
        c = self.config
        lap = self._symbol(x)
        if getattr(model, "output_semantics", None) == "full-g-v1":
            nonlinear = model(self.observe(state))
            mu_explicit = nonlinear - (1 + c.stabilization) * x
        else:
            nonlinear = self.g(self.observe(state))
            mu_explicit = nonlinear - (1 + c.stabilization) * x + self._action(x, model)
        next_hat = torch.fft.rfft2(x) + h * c.mobility * lap * torch.fft.rfft2(mu_explicit)
        next_hat = next_hat / (
            1 - h * c.mobility * c.stabilization * lap + h * c.mobility * c.kappa * lap.square()
        )
        y = torch.fft.irfft2(next_hat, s=x.shape[-2:])
        return {**state, "fields": y, "time": state.get("time", 0.0) + h}

    def closure_target(self, fine_state, n):
        x = self.fields(fine_state)
        coarse = _restrict(x, n)
        target = (_restrict(x.pow(3), n) - coarse.pow(3)) / self.config.action_scale
        return target - target.mean((-2, -1), keepdim=True)

    def classical_closure(self, observation):
        """Leading box-filter expansion of R(c^3) - R(c)^3."""
        x = observation[:, :1]
        variance = torch.zeros_like(x)
        for axis in (-1, -2):
            dx = 0.5 * (x.roll(-1, axis) - x.roll(1, axis))
            variance = variance + dx.square() / 12
        mu = 3 * x * variance / self.config.action_scale
        return mu - mu.mean((-2, -1), keepdim=True)

    def diagnostics(self, state):
        x = self.fields(state)
        h = self.config.length / x.shape[-1]
        grad2 = ((x.roll(-1, -1) - x).square() + (x.roll(-1, -2) - x).square()) / h**2
        energy_density = 0.25 * (x.square() - 1).square() + 0.5 * self.config.kappa * grad2
        return {
            "mass": x.mean((-2, -1)),
            "energy": energy_density.mean((-3, -2, -1)) * self.config.length**2,
            "minimum": x.amin((-3, -2, -1)),
            "maximum": x.amax((-3, -2, -1)),
            "finite": x.isfinite().all(),
        }

    def metadata(self):
        return {
            "name": self.name,
            "config": asdict(self.config),
            "boundary": "periodic",
            "integrator": "stabilized first-order IMEX Euler with exact linear FFT solve",
            "restriction": "conservative cell block average",
            "clipping": False,
            "closure": "chemical potential correction before conservative Laplacian",
            "energy": "diagnostic only; learned correction need not dissipate original free energy",
            "observation_scales": [1.0, 1.0],
            "observation_definition": "c, kappa Lap(c)",
            "field_scales": [1.0],
            "action_scale": self.config.action_scale,
            "initial_dtype": "float32; cast fields before evolution for float64 campaigns",
            "ood_regime": "initial mean shifted by +.25 and Fourier cutoff16 instead of12; physical coefficients unchanged",
            "autograd": "supported; forward-only permissions are enforced by caller context",
            "batched": True,
        }
