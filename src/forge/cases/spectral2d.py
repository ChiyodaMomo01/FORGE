"""Periodic 2D dealiased pseudospectral turbulence cases with ETD-RK4 integrators.

Implements two closed-loop subgrid-scale (SGS) closure benchmarks on [0, L)^2:
1. ``kolmogorov_ns``: Forced-dissipative 2D incompressible Navier-Stokes (Kolmogorov flow)
   in vorticity-streamfunction form.
2. ``mhd2d``: Forced-dissipative 2D incompressible magnetohydrodynamics with a uniform
   in-plane guide field B0 * e_x in vorticity-magnetic-potential form, using an exact
   coupled 2x2 Fourier matrix ETD-RK4 propagator for linear dissipation and Alfven waves.

Both cases share:
- Exact Fourier Galerkin restriction with zeroed Nyquist lines (exact commutation with all
  linear spectral diagnostics F and linear propagators in H: R F_H(q_H) == F_L(R q_H)).
- Parameter-free 3/2 zero-padded pseudospectral quadratic products in both native ``g``
  and public ``coarse_value``, ensuring bit-identical zero-residual correspondence.
- Hard structural projection in H (zero spatial mean / Coulomb gauge, zero Nyquist lines,
  and exact solenoidal velocity / magnetic fields) for arbitrary finite component outputs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

import torch

from ..spectral_ops import dealiased_mul, spectral_restrict


def _scalar_etdrk4_coeffs(
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact float64 Cox-Matthews ETD-RK4 coefficients via 4x4 block matrix_exp."""

    # Top row of exp([[z, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    # equals (exp(z), phi_1(z), phi_2(z), phi_3(z)) without small-z cancellation.
    def exp_and_phis(arg: torch.Tensor):
        aug = torch.zeros((*arg.shape, 4, 4), dtype=torch.float64, device=arg.device)
        aug[..., 0, 0] = arg
        aug[..., 0, 1] = 1.0
        aug[..., 1, 2] = 1.0
        aug[..., 2, 3] = 1.0
        ex = torch.linalg.matrix_exp(aug)
        return ex[..., 0, 0], ex[..., 0, 1], ex[..., 0, 2], ex[..., 0, 3]

    exp_half, phi1_half, _, _ = exp_and_phis(0.5 * z)
    exp_full, phi1, phi2, phi3 = exp_and_phis(z)
    alpha1 = phi1 - 3.0 * phi2 + 4.0 * phi3
    alpha2 = phi2 - 2.0 * phi3
    alpha3 = -phi2 + 4.0 * phi3
    return exp_half, phi1_half, exp_full, alpha1, alpha2, alpha3


def _matrix2x2_etdrk4_coeffs(
    a_mat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact 2x2 matrix Cox-Matthews ETD-RK4 coefficients via 8x8 block matrix_exp."""
    batch_shape = a_mat.shape[:-2]
    eye = torch.eye(2, dtype=torch.complex128, device=a_mat.device)

    def exp_and_phis(mat: torch.Tensor):
        aug = torch.zeros((*batch_shape, 8, 8), dtype=torch.complex128, device=mat.device)
        aug[..., 0:2, 0:2] = mat
        aug[..., 0:2, 2:4] = eye
        aug[..., 2:4, 4:6] = eye
        aug[..., 4:6, 6:8] = eye
        ex = torch.linalg.matrix_exp(aug)
        return (
            ex[..., 0:2, 0:2].contiguous(),
            ex[..., 0:2, 2:4].contiguous(),
            ex[..., 0:2, 4:6].contiguous(),
            ex[..., 0:2, 6:8].contiguous(),
        )

    exp_half, phi1_half, _, _ = exp_and_phis(0.5 * a_mat)
    exp_full, phi1, phi2, phi3 = exp_and_phis(a_mat)
    alpha1 = phi1 - 3.0 * phi2 + 4.0 * phi3
    alpha2 = phi2 - 2.0 * phi3
    alpha3 = -phi2 + 4.0 * phi3
    return exp_half, phi1_half, exp_full, alpha1, alpha2, alpha3


@dataclass(frozen=True)
class KolmogorovNSConfig:
    ood: bool = False
    length: float = 2.0 * math.pi
    viscosity: float = 5.0e-4
    drag: float = 0.1
    forcing_amp: float = 1.0
    forcing_wavenumber: int = 6
    dt: float = 0.02
    substeps: int = 2
    initial_rms: float = 2.5
    action_scale: float = 0.1


@dataclass(frozen=True)
class MHD2DConfig:
    ood: bool = False
    length: float = 2.0 * math.pi
    viscosity: float = 5.0e-4
    resistivity: float = 5.0e-4
    drag: float = 0.05
    guide_field: float = 0.5
    forcing_amp: float = 1.0
    forcing_wavenumber: int = 4
    dt: float = 0.025
    substeps: int = 2
    initial_vorticity_rms: float = 2.0
    initial_potential_rms: float = 0.2
    action_scale: float = 0.1


def _parse_config(cls, config):
    if config is None:
        return cls()
    if isinstance(config, Mapping):
        values = dict(config)
        values.pop("differentiable", None)
        values.pop("dtype", None)
        return cls(**values)
    return config


class _Spectral2DBase:
    spatial_dim = 2

    def __init__(self):
        self._spec_cache: dict[tuple, dict[str, torch.Tensor]] = {}
        self._step_cache: dict[tuple, dict[str, torch.Tensor]] = {}

    @staticmethod
    def _validate_n(n: int) -> None:
        if not isinstance(n, int) or n < 8 or n % 2:
            raise ValueError("spectral 2D cases require an even square grid >= 8")

    def default_dt(self, n: int) -> float:
        self._validate_n(n)
        return float(self.config.dt)

    def fields(self, state):
        return state["fields"]

    def state_from_fields(self, fields: torch.Tensor, time: float = 0.0):
        if (
            fields.ndim != 4
            or fields.shape[1] != self.field_channels
            or fields.shape[-1] != fields.shape[-2]
        ):
            raise ValueError(
                f"expected batched square fields with {self.field_channels} channels, got {tuple(fields.shape)}"
            )
        self._validate_n(fields.shape[-1])
        clean = self._project_fields(fields)
        return {"fields": clean, "time": float(time)}

    def restrict(self, state, n: int):
        self._validate_n(n)
        return {**state, "fields": spectral_restrict(self.fields(state), n)}

    def _spectral_ops(self, n: int, dtype: torch.dtype, device: torch.device):
        key = (n, dtype, device)
        if key not in self._spec_cache:
            length = self.config.length
            kx = (
                2.0
                * math.pi
                * torch.fft.rfftfreq(n, d=length / n, dtype=torch.float64, device=device)
            )
            ky = (
                2.0
                * math.pi
                * torch.fft.fftfreq(n, d=length / n, dtype=torch.float64, device=device)[:, None]
            )
            k2 = kx.square() + ky.square()
            mask = torch.ones_like(k2)
            mask[0, 0] = 0.0
            mask[n // 2, :] = 0.0
            mask[:, -1] = 0.0
            inv_lap = torch.where(mask > 0, -1.0 / k2.clamp(min=1e-30), torch.zeros_like(k2))
            q = torch.arange(n, dtype=torch.float64, device=device) * (length / n)
            yy, xx = torch.meshgrid(q, q, indexing="ij")
            forcing = self._build_forcing_hat(xx, yy, mask)
            cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
            self._spec_cache[key] = {
                "kx": kx.to(dtype),
                "ky": ky.to(dtype),
                "k2": k2.to(dtype),
                "mask": mask.to(dtype),
                "inv_lap": inv_lap.to(dtype),
                "forcing_hat": forcing.to(cdtype),
            }
        return self._spec_cache[key]

    def _project_fields(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        op = self._spectral_ops(n, x.dtype, x.device)
        xh = torch.fft.rfft2(x) * op["mask"]
        return torch.fft.irfft2(xh, s=(n, n))

    def _random_band_field(
        self, gen: torch.Generator, n: int, rms: float, device, dtype=torch.float64
    ) -> torch.Tensor:
        # Generate random spectrum at fixed canonical resolution n0=64 so initial energy
        # resides strictly in large/resolved scales (k in [2, 8]) on all grids >= 16.
        noise = torch.randn((1, 1, n, n), generator=gen, dtype=torch.float64).to(device)
        nh = torch.fft.rfft2(noise)
        kx = torch.fft.rfftfreq(n, d=1.0 / n, dtype=torch.float64, device=device)
        ky = torch.fft.fftfreq(n, d=1.0 / n, dtype=torch.float64, device=device)[:, None]
        kr = torch.sqrt(kx.square() + ky.square())
        envelope = torch.where(
            (kr >= 2.0) & (kr <= 8.0),
            1.0 / (1.0 + (kr / 4.0).square()),
            torch.zeros_like(kr),
        )
        envelope[n // 2, :] = 0.0
        envelope[:, -1] = 0.0
        envelope[0, 0] = 0.0
        field = torch.fft.irfft2(nh * envelope, s=(n, n))
        cur_rms = field.square().mean().sqrt().clamp(min=1e-30)
        return (field * (rms / cur_rms)).to(dtype)

    def _evaluate_flux(self, obs: torch.Tensor, model) -> torch.Tensor:
        if getattr(model, "output_semantics", None) == "full-g-v1":
            parameter = next(model.parameters(), None) if hasattr(model, "parameters") else None
            inp = obs.to(parameter.dtype) if parameter is not None else obs
            return model(inp).to(obs.dtype)
        flux = self.g(obs)
        if model is not None:
            parameter = next(model.parameters(), None) if hasattr(model, "parameters") else None
            inp = obs.to(parameter.dtype) if parameter is not None else obs
            raw = model(inp).to(obs.dtype)
            expected = (obs.shape[0], self.action_channels, *obs.shape[-2:])
            if raw.shape != expected:
                raise ValueError(f"expected action shape {expected}, got {tuple(raw.shape)}")
            flux = flux + raw * self.config.action_scale
        return flux


class KolmogorovNSCase(_Spectral2DBase):
    """2D forced incompressible Navier-Stokes in vorticity-streamfunction form."""

    name = "kolmogorov_ns"
    field_channels = 1
    observation_channels = 6
    action_channels = 2

    def __init__(self, config=None):
        super().__init__()
        self.config = _parse_config(KolmogorovNSConfig, config)
        if (
            min(
                self.config.length,
                self.config.viscosity,
                self.config.dt,
                self.config.initial_rms,
                self.config.action_scale,
            )
            <= 0
            or self.config.drag < 0
            or self.config.substeps < 1
            or self.config.forcing_wavenumber < 1
        ):
            raise ValueError("invalid KolmogorovNSConfig parameters")

    def _build_forcing_hat(self, xx, yy, mask):
        # Horizontal body force f_u = (A sin(k_f y), 0) => curl f_omega = -A k_f cos(k_f y).
        kf = int(self.config.forcing_wavenumber)
        fw = -self.config.forcing_amp * kf * torch.cos(kf * yy)
        return torch.fft.rfft2(fw) * mask

    def initial(self, seed: int, n: int, device="cpu"):
        self._validate_n(n)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        rms = self.config.initial_rms * (1.25 if self.config.ood else 1.0)
        omega = self._random_band_field(gen, n, rms, device=device, dtype=torch.float64)
        return {"fields": omega, "time": 0.0}

    def _diagnose_hat(self, wh: torch.Tensor, op: dict[str, torch.Tensor]):
        n = wh.shape[-2]
        wh = wh * op["mask"]
        psih = wh * op["inv_lap"]
        uh = -1j * op["ky"] * psih
        vh = 1j * op["kx"] * psih
        wxh = 1j * op["kx"] * wh
        wyh = 1j * op["ky"] * wh
        inv = lambda z: torch.fft.irfft2(z, s=(n, n))
        return inv(wh), inv(psih), inv(uh), inv(vh), inv(wxh), inv(wyh)

    def observe(self, state):
        w = self.fields(state)
        n = w.shape[-1]
        op = self._spectral_ops(n, w.dtype, w.device)
        wh = torch.fft.rfft2(w)
        omega, psi, u, v, wx, wy = self._diagnose_hat(wh, op)
        return torch.cat((omega, psi, u, v, wx, wy), dim=1)

    def g(self, public_input: torch.Tensor) -> torch.Tensor:
        """Actual dealiased advective vorticity flux (u * omega, v * omega)."""
        omega = public_input[:, 0:1]
        u = public_input[:, 2:3]
        v = public_input[:, 3:4]
        return torch.cat((dealiased_mul(u, omega), dealiased_mul(v, omega)), dim=1)

    def _etd_ops(self, n: int, h: float, dtype: torch.dtype, device: torch.device):
        key = (n, round(float(h), 15), dtype, device)
        if key not in self._step_cache:
            op64 = self._spectral_ops(n, torch.float64, device)
            lam = -(self.config.viscosity * op64["k2"] + self.config.drag) * op64["mask"]
            exp_half, phi1_half, exp_full, alpha1, alpha2, alpha3 = _scalar_etdrk4_coeffs(h * lam)
            mask = op64["mask"]
            self._step_cache[key] = {
                "exp_half": (exp_half * mask).to(dtype),
                "half_phi1_half": (0.5 * h * phi1_half * mask).to(dtype),
                "exp_full": (exp_full * mask).to(dtype),
                "h_alpha1": (h * alpha1 * mask).to(dtype),
                "two_h_alpha2": (2.0 * h * alpha2 * mask).to(dtype),
                "h_alpha3": (h * alpha3 * mask).to(dtype),
            }
        return self._step_cache[key]

    def _rhs_hat(self, wh: torch.Tensor, op: dict[str, torch.Tensor], model) -> torch.Tensor:
        omega, psi, u, v, wx, wy = self._diagnose_hat(wh, op)
        obs = torch.cat((omega, psi, u, v, wx, wy), dim=1)
        flux = self._evaluate_flux(obs, model)
        f1h = torch.fft.rfft2(flux[:, 0:1])
        f2h = torch.fft.rfft2(flux[:, 1:2])
        div_hat = 1j * op["kx"] * f1h + 1j * op["ky"] * f2h
        return (-div_hat + op["forcing_hat"][None, None]) * op["mask"]

    def step(self, state, model=None, dt=None):
        w = self.fields(state)
        n = w.shape[-1]
        interval = self.default_dt(n) if dt is None else float(dt)
        if interval <= 0:
            raise ValueError("dt must be positive")
        target_substep = self.config.dt / self.config.substeps
        count = max(1, int(round(interval / target_substep)))
        if abs(interval - count * target_substep) > 1e-12 * max(1.0, interval):
            count = max(self.config.substeps, math.ceil(interval / target_substep))
        h = interval / count
        op = self._spectral_ops(n, w.dtype, w.device)
        etd = self._etd_ops(n, h, w.dtype, w.device)
        mask = op["mask"]
        wh = torch.fft.rfft2(w) * mask
        for _ in range(count):
            n0 = self._rhs_hat(wh, op, model)
            base_half = etd["exp_half"] * wh
            ah = (base_half + etd["half_phi1_half"] * n0) * mask
            na = self._rhs_hat(ah, op, model)
            bh = (base_half + etd["half_phi1_half"] * na) * mask
            nb = self._rhs_hat(bh, op, model)
            ch = (etd["exp_half"] * ah + etd["half_phi1_half"] * (2.0 * nb - n0)) * mask
            nc = self._rhs_hat(ch, op, model)
            wh = (
                etd["exp_full"] * wh
                + etd["h_alpha1"] * n0
                + etd["two_h_alpha2"] * (na + nb)
                + etd["h_alpha3"] * nc
            ) * mask
        w_next = torch.fft.irfft2(wh, s=(n, n))
        return {**state, "fields": w_next, "time": state.get("time", 0.0) + interval}

    def closure_target(self, fine_state, n: int) -> torch.Tensor:
        obs_hr = self.observe(fine_state)
        obs_lr = spectral_restrict(obs_hr, n)
        return (spectral_restrict(self.g(obs_hr), n) - self.g(obs_lr)) / self.config.action_scale

    def diagnostics(self, state):
        w = self.fields(state)
        n = w.shape[-1]
        op = self._spectral_ops(n, w.dtype, w.device)
        omega, psi, u, v, wx, wy = self._diagnose_hat(torch.fft.rfft2(w), op)
        ke = 0.5 * (u.square() + v.square()).mean((-2, -1))
        enstrophy = 0.5 * omega.square().mean((-2, -1))
        palinstrophy = 0.5 * (wx.square() + wy.square()).mean((-2, -1))
        dx = self.config.length / n
        h_sub = self.config.dt / self.config.substeps
        cfl = (u.abs() + v.abs()).amax((-2, -1)) * h_sub / dx
        div_u = torch.fft.irfft2(
            1j * op["kx"] * torch.fft.rfft2(u) + 1j * op["ky"] * torch.fft.rfft2(v),
            s=(n, n),
        )
        return {
            "kinetic_energy": ke,
            "enstrophy": enstrophy,
            "palinstrophy": palinstrophy,
            "vorticity_rms": omega.square().mean((-2, -1)).sqrt(),
            "max_advective_cfl": cfl.amax(),
            "divergence_l2": div_u.square().mean().sqrt(),
            "mean_abs": omega.mean((-2, -1)).abs().amax(),
            "finite": w.isfinite().all(),
        }

    def metadata(self):
        return {
            "name": self.name,
            "config": asdict(self.config),
            "boundary": "periodic [0, 2*pi)^2",
            "forcing": "unidirectional sinusoidal Kolmogorov shear (-A * k_f * cos(k_f * y))",
            "integrator": "4-stage Cox-Matthews ETD-RK4 with exact diagonal viscous/drag exponential and 3/2 dealiasing",
            "restriction": "Fourier Galerkin truncation with zeroed coarse Nyquist lines",
            "clipping": False,
        }


class MHD2DCase(_Spectral2DBase):
    """2D forced incompressible guide-field MHD in (vorticity, magnetic potential) form."""

    name = "mhd2d"
    field_channels = 2
    observation_channels = 8
    action_channels = 3

    def __init__(self, config=None):
        super().__init__()
        self.config = _parse_config(MHD2DConfig, config)
        if (
            min(
                self.config.length,
                self.config.viscosity,
                self.config.resistivity,
                self.config.dt,
                self.config.initial_vorticity_rms,
                self.config.initial_potential_rms,
                self.config.action_scale,
            )
            <= 0
            or self.config.drag < 0
            or self.config.guide_field < 0
            or self.config.substeps < 1
            or self.config.forcing_wavenumber < 1
        ):
            raise ValueError("invalid MHD2DConfig parameters")

    def _build_forcing_hat(self, xx, yy, mask):
        # Cellular vortex forcing sustains x- and y-shears that continuously bend the
        # horizontal guide field B0 * e_x, preventing Zeldovich decay in 2D.
        kf = int(self.config.forcing_wavenumber)
        fw = -self.config.forcing_amp * kf * torch.cos(kf * xx) * torch.cos(kf * yy)
        return torch.fft.rfft2(fw) * mask

    def initial(self, seed: int, n: int, device="cpu"):
        self._validate_n(n)
        gen_w = torch.Generator(device="cpu").manual_seed(int(seed))
        gen_a = torch.Generator(device="cpu").manual_seed(int(seed) + 10_000_019)
        scale = 1.25 if self.config.ood else 1.0
        omega = self._random_band_field(
            gen_w, n, self.config.initial_vorticity_rms * scale, device=device
        )
        pot_a = self._random_band_field(
            gen_a, n, self.config.initial_potential_rms * scale, device=device
        )
        return {"fields": torch.cat((omega, pot_a), dim=1), "time": 0.0}

    def _diagnose_hat(self, xh: torch.Tensor, op: dict[str, torch.Tensor]):
        n = xh.shape[-2]
        xh = xh * op["mask"]
        wh = xh[:, 0:1]
        ah = xh[:, 1:2]
        psih = wh * op["inv_lap"]
        jh = -op["k2"] * ah
        uh = -1j * op["ky"] * psih
        vh = 1j * op["kx"] * psih
        bxh = -1j * op["ky"] * ah
        byh = 1j * op["kx"] * ah
        inv = lambda z: torch.fft.irfft2(z, s=(n, n))
        return (
            inv(wh),
            inv(ah),
            inv(psih),
            inv(jh),
            inv(uh),
            inv(vh),
            inv(bxh),
            inv(byh),
        )

    def observe(self, state):
        x = self.fields(state)
        n = x.shape[-1]
        op = self._spectral_ops(n, x.dtype, x.device)
        xh = torch.fft.rfft2(x)
        omega, pot_a, psi, cur_j, u, v, bx, by = self._diagnose_hat(xh, op)
        return torch.cat((omega, pot_a, psi, cur_j, u, v, bx, by), dim=1)

    def g(self, public_input: torch.Tensor) -> torch.Tensor:
        """Actual 3/2-dealiased Reynolds-Maxwell vorticity flux and out-of-plane EMF."""
        omega = public_input[:, 0:1]
        cur_j = public_input[:, 3:4]
        u = public_input[:, 4:5]
        v = public_input[:, 5:6]
        bx = public_input[:, 6:7]
        by = public_input[:, 7:8]
        g1 = dealiased_mul(u, omega) - dealiased_mul(bx, cur_j)
        g2 = dealiased_mul(v, omega) - dealiased_mul(by, cur_j)
        g3 = dealiased_mul(u, by) - dealiased_mul(v, bx)
        return torch.cat((g1, g2, g3), dim=1)

    def _etd_ops(self, n: int, h: float, dtype: torch.dtype, device: torch.device):
        key = (n, round(float(h), 15), dtype, device)
        if key not in self._step_cache:
            op64 = self._spectral_ops(n, torch.float64, device)
            kx = op64["kx"]
            k2 = op64["k2"]
            mask = op64["mask"]
            b0 = self.config.guide_field
            # Linear Fourier matrix L(k) on (omega_hat, a_hat)^T for k != 0:
            #   d/dt omega_hat = (-nu*k^2 - drag)*omega_hat + (-i*kx*B0*k^2)*a_hat
            #   d/dt a_hat     = (-i*kx*B0 / k^2)*omega_hat + (-eta*k^2)*a_hat
            l_mat = torch.zeros((*k2.shape, 2, 2), dtype=torch.complex128, device=device)
            l_mat[..., 0, 0] = -(self.config.viscosity * k2 + self.config.drag)
            l_mat[..., 0, 1] = -1j * kx * b0 * k2
            l_mat[..., 1, 0] = torch.where(
                mask > 0,
                -1j * kx * b0 / k2.clamp(min=1e-30),
                torch.zeros_like(k2, dtype=torch.complex128),
            )
            l_mat[..., 1, 1] = -(self.config.resistivity * k2)
            l_mat = l_mat * mask[..., None, None]
            exp_half, phi1_half, exp_full, alpha1, alpha2, alpha3 = _matrix2x2_etdrk4_coeffs(
                h * l_mat
            )
            m_mask = mask[..., None, None].to(torch.complex128)
            cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
            self._step_cache[key] = {
                "exp_half": (exp_half * m_mask).to(cdtype),
                "half_phi1_half": (0.5 * h * phi1_half * m_mask).to(cdtype),
                "exp_full": (exp_full * m_mask).to(cdtype),
                "h_alpha1": (h * alpha1 * m_mask).to(cdtype),
                "two_h_alpha2": (2.0 * h * alpha2 * m_mask).to(cdtype),
                "h_alpha3": (h * alpha3 * m_mask).to(cdtype),
            }
        return self._step_cache[key]

    @staticmethod
    def _matvec2(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        # mat: (H, W, 2, 2), vec: (B, 2, H, W) -> (B, 2, H, W)
        v0, v1 = vec[:, 0], vec[:, 1]
        out0 = mat[..., 0, 0] * v0 + mat[..., 0, 1] * v1
        out1 = mat[..., 1, 0] * v0 + mat[..., 1, 1] * v1
        return torch.stack((out0, out1), dim=1)

    def _rhs_hat(self, xh: torch.Tensor, op: dict[str, torch.Tensor], model) -> torch.Tensor:
        omega, pot_a, psi, cur_j, u, v, bx, by = self._diagnose_hat(xh, op)
        obs = torch.cat((omega, pot_a, psi, cur_j, u, v, bx, by), dim=1)
        flux = self._evaluate_flux(obs, model)
        g1h = torch.fft.rfft2(flux[:, 0:1])
        g2h = torch.fft.rfft2(flux[:, 1:2])
        g3h = torch.fft.rfft2(flux[:, 2:3])
        nw = (-1j * op["kx"] * g1h - 1j * op["ky"] * g2h + op["forcing_hat"][None, None]) * op[
            "mask"
        ]
        na = (-g3h) * op["mask"]
        return torch.cat((nw, na), dim=1)

    def step(self, state, model=None, dt=None):
        x = self.fields(state)
        n = x.shape[-1]
        interval = self.default_dt(n) if dt is None else float(dt)
        if interval <= 0:
            raise ValueError("dt must be positive")
        target_substep = self.config.dt / self.config.substeps
        count = max(1, int(round(interval / target_substep)))
        if abs(interval - count * target_substep) > 1e-12 * max(1.0, interval):
            count = max(self.config.substeps, math.ceil(interval / target_substep))
        h = interval / count
        op = self._spectral_ops(n, x.dtype, x.device)
        etd = self._etd_ops(n, h, x.dtype, x.device)
        mask = op["mask"]
        xh = torch.fft.rfft2(x) * mask
        for _ in range(count):
            n0 = self._rhs_hat(xh, op, model)
            base_half = self._matvec2(etd["exp_half"], xh)
            ah = (base_half + self._matvec2(etd["half_phi1_half"], n0)) * mask
            na = self._rhs_hat(ah, op, model)
            bh = (base_half + self._matvec2(etd["half_phi1_half"], na)) * mask
            nb = self._rhs_hat(bh, op, model)
            ch = (
                self._matvec2(etd["exp_half"], ah)
                + self._matvec2(etd["half_phi1_half"], 2.0 * nb - n0)
            ) * mask
            nc = self._rhs_hat(ch, op, model)
            xh = (
                self._matvec2(etd["exp_full"], xh)
                + self._matvec2(etd["h_alpha1"], n0)
                + self._matvec2(etd["two_h_alpha2"], na + nb)
                + self._matvec2(etd["h_alpha3"], nc)
            ) * mask
        x_next = torch.fft.irfft2(xh, s=(n, n))
        return {**state, "fields": x_next, "time": state.get("time", 0.0) + interval}

    def closure_target(self, fine_state, n: int) -> torch.Tensor:
        obs_hr = self.observe(fine_state)
        obs_lr = spectral_restrict(obs_hr, n)
        return (spectral_restrict(self.g(obs_hr), n) - self.g(obs_lr)) / self.config.action_scale

    def diagnostics(self, state):
        x = self.fields(state)
        n = x.shape[-1]
        op = self._spectral_ops(n, x.dtype, x.device)
        omega, pot_a, psi, cur_j, u, v, bx, by = self._diagnose_hat(torch.fft.rfft2(x), op)
        ke = 0.5 * (u.square() + v.square()).mean((-2, -1))
        me = 0.5 * (bx.square() + by.square()).mean((-2, -1))
        enstrophy = 0.5 * omega.square().mean((-2, -1))
        mean_sq_current = 0.5 * cur_j.square().mean((-2, -1))
        cross_helicity = (u * bx + v * by).mean((-2, -1))
        dx = self.config.length / n
        h_sub = self.config.dt / self.config.substeps
        b0 = self.config.guide_field
        adv_cfl = (u.abs() + v.abs()).amax((-2, -1)) * h_sub / dx
        alf_cfl = ((bx + b0).abs() + by.abs()).amax((-2, -1)) * h_sub / dx
        div_u = torch.fft.irfft2(
            1j * op["kx"] * torch.fft.rfft2(u) + 1j * op["ky"] * torch.fft.rfft2(v),
            s=(n, n),
        )
        div_b = torch.fft.irfft2(
            1j * op["kx"] * torch.fft.rfft2(bx) + 1j * op["ky"] * torch.fft.rfft2(by),
            s=(n, n),
        )
        return {
            "kinetic_energy": ke,
            "magnetic_energy": me,
            "total_energy": ke + me,
            "enstrophy": enstrophy,
            "mean_square_current": mean_sq_current,
            "cross_helicity": cross_helicity,
            "max_advective_cfl": adv_cfl.amax(),
            "max_alfven_cfl": alf_cfl.amax(),
            "divergence_u_l2": div_u.square().mean().sqrt(),
            "divergence_b_l2": div_b.square().mean().sqrt(),
            "mean_abs": x.mean((-2, -1)).abs().amax(),
            "finite": x.isfinite().all(),
        }

    def metadata(self):
        return {
            "name": self.name,
            "config": asdict(self.config),
            "boundary": "periodic [0, 2*pi)^2",
            "forcing": "2D cellular vortex forcing (-A * k_f * cos(k_f * x) * cos(k_f * y)) with uniform in-plane guide field B0 * e_x",
            "integrator": "4-stage matrix Cox-Matthews ETD-RK4 with exact coupled dissipation + guide-field Alfven exponential and 3/2 dealiasing",
            "restriction": "Fourier Galerkin truncation with zeroed coarse Nyquist lines",
            "clipping": False,
        }
