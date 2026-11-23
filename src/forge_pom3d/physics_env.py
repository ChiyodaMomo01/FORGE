from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from .config import ForgePOM3DConfig
from .ocean.contracts import CANONICAL_CORE_NAME, resolve_env_var
from .ocean.hybrid import (
    StateNormalizer,
    build_initialized_core,
    inject_core_state,
    pack_pom_batch,
)
from .ocean.hybrid.sparse_coupling import LEVELS, OUTPUT_FIELDS
from .ocean.io.checkpoint import capture, restore
from .ocean.io.core_spec import core_spec
from .ocean.io.data import FixedCoreDataset
from .ocean.semi_implicit.autograd import strain_magnitude
from .ocean.semi_implicit.gradient_kernels import CompiledVerticalCore


def resolve_input_stats_path(
    cfg: ForgePOM3DConfig,
    explicit_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve calibration statistics via CLI/config > FORGE_POM3D_INPUT_STATS > ${dataset_root}/assets/input_stats.npz."""
    if explicit_path is not None and str(explicit_path).strip():
        return Path(explicit_path)
    cfg_path = getattr(cfg, "input_stats_path", "")
    if cfg_path and str(cfg_path).strip():
        return Path(cfg_path)
    env_path = resolve_env_var("FORGE_POM3D_INPUT_STATS", "", environ=environ).strip()
    if env_path:
        return Path(env_path)
    return Path(cfg.dataset_root) / "assets" / "input_stats.npz"


def apply_projection_P(
    phys_pieces: dict[str, torch.Tensor],
    public_geom: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply exact W-orthogonal linear projection P(q) using static public geometry only.

    Satisfies P(0) = 0, P(P(q)) = P(q), and <p, P(q)>_W = <P(p), q>_W under static cell-volume
    weights W = (w_vol_u, w_vol_v, w_vol_t, w_vol_t).
    - Baroclinic velocity increments (u, v): staggered wet mask + vertical dz-weighted zero column mean.
    - Tracer increments (t, s): tracer wet mask + per-layer static-volume-weighted zero mean.
    """
    dum = public_geom["dum"]
    dvm = public_geom["dvm"]
    fsm = public_geom["fsm"]
    dz = public_geom["dz"]  # (1, 1, LEVELS)
    w_vol_t = public_geom["w_vol_t"]  # (im, jm, LEVELS)

    u = phys_pieces["u"] * dum[:, :, None]
    v = phys_pieces["v"] * dvm[:, :, None]
    t = phys_pieces["t"] * fsm[:, :, None]
    s = phys_pieces["s"] * fsm[:, :, None]

    dz_sum = dz.sum().clamp_min(1e-20)
    u_proj = (u - (u * dz).sum(dim=2, keepdim=True) / dz_sum) * dum[:, :, None]
    v_proj = (v - (v * dz).sum(dim=2, keepdim=True) / dz_sum) * dvm[:, :, None]

    layer_vol = w_vol_t.sum(dim=(0, 1), keepdim=True).clamp_min(1e-20)
    t_mean = (t * w_vol_t).sum(dim=(0, 1), keepdim=True) / layer_vol
    s_mean = (s * w_vol_t).sum(dim=(0, 1), keepdim=True) / layer_vol
    t_proj = (t - t_mean) * fsm[:, :, None]
    s_proj = (s - s_mean) * fsm[:, :, None]

    return {"u": u_proj, "v": v_proj, "t": t_proj, "s": s_proj}


def apply_projection_P_transpose(
    grad_pieces: dict[str, torch.Tensor],
    public_geom: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply exact Euclidean transpose P^T(g) of W-orthogonal projection P = I - 1 w^T / (w^T 1).

    For arbitrary positive cell-volume weights w_vol_t, satisfies <P(q), g>_2 = <q, P^T(g)>_2 to
    machine precision while using static public geometry only.
    """
    dum = public_geom["dum"]
    dvm = public_geom["dvm"]
    fsm = public_geom["fsm"]
    dz = public_geom["dz"]  # (1, 1, LEVELS)
    w_vol_t = public_geom["w_vol_t"]  # (im, jm, LEVELS)

    gu = grad_pieces["u"] * dum[:, :, None]
    gv = grad_pieces["v"] * dvm[:, :, None]
    gt = grad_pieces["t"] * fsm[:, :, None]
    gs = grad_pieces["s"] * fsm[:, :, None]

    dz_sum = dz.sum().clamp_min(1e-20)
    gu_adj = (gu - dz * gu.sum(dim=2, keepdim=True) / dz_sum) * dum[:, :, None]
    gv_adj = (gv - dz * gv.sum(dim=2, keepdim=True) / dz_sum) * dvm[:, :, None]

    layer_vol = w_vol_t.sum(dim=(0, 1), keepdim=True).clamp_min(1e-20)
    gt_sum = gt.sum(dim=(0, 1), keepdim=True)
    gs_sum = gs.sum(dim=(0, 1), keepdim=True)
    gt_adj = (gt - w_vol_t * (gt_sum / layer_vol)) * fsm[:, :, None]
    gs_adj = (gs - w_vol_t * (gs_sum / layer_vol)) * fsm[:, :, None]

    return {"u": gu_adj, "v": gv_adj, "t": gt_adj, "s": gs_adj}


class BlackBoxProvider3D:
    """Strict F -> G_theta -> H black-box environment wrapper for 3D Double-Gyre POM."""

    def __init__(
        self,
        cfg: ForgePOM3DConfig,
        device: torch.device,
        input_stats_path: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.spec = core_spec(CANONICAL_CORE_NAME)
        self.core = build_initialized_core(self.spec.core_class, self.device, cfg.im, cfg.jm)
        self.spec.configure(self.core)
        self._enable_fast_kernels()
        self.core.runtime_diagnostics = False
        self.baseline_state = capture(self.core, cpu=False)
        self.dataset = FixedCoreDataset(
            cfg.dataset_root,
            3600,
            self.spec.name,
            expected_grid=(cfg.im, cfg.jm, cfg.kb),
        )
        resolved_stats = resolve_input_stats_path(cfg, input_stats_path)
        self.normalizer = StateNormalizer.from_path(resolved_stats, self.device)
        self.scales = {
            "u": cfg.sigma_u,
            "v": cfg.sigma_v,
            "t": cfg.sigma_t,
            "s": cfg.sigma_s,
            "el": cfg.sigma_el,
        }
        self.action_scales = {
            "u": cfg.scale_u * cfg.action_gain,
            "v": cfg.scale_v * cfg.action_gain,
            "t": cfg.scale_t * cfg.action_gain,
            "s": cfg.scale_s * cfg.action_gain,
        }
        self.loss_scales = {
            "u": cfg.loss_scale_u,
            "v": cfg.loss_scale_v,
            "t": cfg.loss_scale_t,
            "s": cfg.loss_scale_s,
            "el": cfg.loss_scale_el,
        }
        dz = self.core.dz[:LEVELS].detach().clone().view(1, 1, LEVELS)
        art = self.core.art.detach().clone()
        h = self.core.h.detach().clone()
        fsm = self.core.fsm.detach().clone()
        dum = (self.core.dum * self.core.fsm).detach().clone()
        dvm = (self.core.dvm * self.core.fsm).detach().clone()
        self._public_geometry: dict[str, torch.Tensor] = {
            "dz": dz,
            "art": art,
            "h": h,
            "dum": dum,
            "dvm": dvm,
            "fsm": fsm,
            "w_vol_u": (art * dum * h)[:, :, None] * dz,
            "w_vol_v": (art * dvm * h)[:, :, None] * dz,
            "w_vol_t": (art * fsm * h)[:, :, None] * dz,
            "w_area_el": art * fsm,
        }
        self._continuation: dict[str, Any] | None = None
        self.predictor_substeps_executed: int = 0
        self.complete_substeps_executed: int = 0
        self.whitebox_ad_substeps_executed: int = 0

    def get_step_counters(self) -> dict[str, int]:
        """Return cumulative 10-minute internal solver step counters on this rank."""
        return {
            "predictor_substeps_executed": int(getattr(self, "predictor_substeps_executed", 0)),
            "complete_substeps_executed": int(getattr(self, "complete_substeps_executed", 0)),
            "whitebox_ad_substeps_executed": int(getattr(self, "whitebox_ad_substeps_executed", 0)),
        }

    def reset_step_counters(self) -> None:
        """Reset solver step counters on this rank."""
        self.predictor_substeps_executed = 0
        self.complete_substeps_executed = 0
        self.whitebox_ad_substeps_executed = 0

    def get_public_geometry(self) -> dict[str, torch.Tensor]:
        """Return static public grid geometry and masks (no native state or solver internals)."""
        return self._public_geometry

    def _enable_fast_kernels(self) -> None:
        self.core.__class__ = CompiledVerticalCore
        self.core._gradient_kernel_names = ("proft", "profq")
        self.core.use_triton_elliptic_operator = self.device.type == "cuda"
        self.core.use_triton_cg_updates = False
        self.core.compile_eta_solver = False
        self.core.runtime_diagnostics = False

    def reset(self, member: int, day: int) -> dict[str, torch.Tensor]:
        """Reset solver to exact coarse checkpoint at start of `day`."""
        restore(self.core, self.baseline_state)
        self._enable_fast_kernels()
        initial = self.dataset.load_initial(member, day, map_location=self.device)
        inject_core_state(self.core, initial)
        self.core.dt = self.core.h + self.core.et
        self.core.d = self.core.h + self.core.el
        self._continuation = None
        return capture(self.core, cpu=False)

    def get_state(self) -> dict[str, torch.Tensor]:
        return capture(self.core, cpu=False)

    def set_state(self, state: dict[str, torch.Tensor]) -> None:
        restore(self.core, state)
        self._enable_fast_kernels()
        self._continuation = None

    def _step_explicit_predictor_prefix(
        self, differentiable: bool = False
    ) -> dict[str, torch.Tensor]:
        """Run the explicit predictor prefix of 1 internal step (10 min), stopping right before proft."""
        ctx = torch.enable_grad() if differentiable else torch.no_grad()
        gen_fast = not differentiable
        with ctx:
            self.predictor_substeps_executed = (
                int(getattr(self, "predictor_substeps_executed", 0)) + 1
            )
            core = self.core
            s_i = slice(1, core.imm1)
            s_j = slice(1, core.jmm1)
            s_i_m1 = slice(0, core.imm1 - 1)
            s_i_p1 = slice(2, core.imm1 + 1)
            s_j_m1 = slice(0, core.jmm1 - 1)
            s_j_p1 = slice(2, core.jmm1 + 1)

            core.iint = int(getattr(core, "iint", 0)) + 1
            core.time = core.dti * core.iint / 86400.0 + core.time0
            core.ramp = min(1.0, float(core.time / core.period)) if core.lramp else 1.0
            core.apply_implicit_case_surface_forcing(generation_fast_path=gen_fast)

            if core.mode != 2:
                _, _, curv_new, advx_new, advy_new = core.advct(
                    core.a,
                    core.c,
                    core.ee,
                    core.advx,
                    core.advy,
                    core.dt,
                    core.u,
                    core.v,
                    core.w,
                    core.ub,
                    core.vb,
                    core.dti2,
                    core.dx,
                    core.dy,
                    core.dz,
                    core.h,
                    core.dum,
                    core.dvm,
                    core.art,
                    core.aru,
                    core.arv,
                    core.cbc,
                    core.uab,
                    core.vab,
                    core.im,
                    core.jm,
                    core.kb,
                    core.imm1,
                    core.jmm1,
                    core.kbm1,
                    core.drx2d,
                    core.dry2d,
                    core.cor,
                    core.curv,
                    core.aam,
                )
                core.advx, core.advy, core.curv = advx_new, advy_new, curv_new
                core.rho, core.drhox, core.drhoy = core.baropg()

                dx_sl = core.dx[s_i, s_j].unsqueeze(2)
                dy_sl = core.dy[s_i, s_j].unsqueeze(2)
                u_curr = core.u[s_i, s_j, : core.kbm1]
                u_ip1 = core.u[s_i_p1, s_j, : core.kbm1]
                v_curr = core.v[s_i, s_j, : core.kbm1]
                v_jp1 = core.v[s_i, s_j_p1, : core.kbm1]
                term1 = ((u_ip1 - u_curr) / dx_sl) ** 2
                term2 = ((v_jp1 - v_curr) / dy_sl) ** 2
                u_term = (
                    0.25
                    * (
                        core.u[s_i, s_j_p1, : core.kbm1]
                        + core.u[s_i_p1, s_j_p1, : core.kbm1]
                        - core.u[s_i, s_j_m1, : core.kbm1]
                        - core.u[s_i_p1, s_j_m1, : core.kbm1]
                    )
                    / dy_sl
                )
                v_term = (
                    0.25
                    * (
                        core.v[s_i_p1, s_j, : core.kbm1]
                        + core.v[s_i_p1, s_j_p1, : core.kbm1]
                        - core.v[s_i_m1, s_j, : core.kbm1]
                        - core.v[s_i_m1, s_j_p1, : core.kbm1]
                    )
                    / dx_sl
                )
                term3 = 0.5 * (u_term + v_term) ** 2
                smag_visc = core.horcon * dx_sl * dy_sl * strain_magnitude(term1 + term2 + term3)
                aam_min = float(getattr(core, "aam_min", 0.0) or 0.0)
                if aam_min > 0.0:
                    smag_visc = torch.clamp(smag_visc, min=aam_min)
                aam_new = core.aam.clone()
                aam_new[s_i, s_j, : core.kbm1] = smag_visc
                core.aam = core.apply_implicit_boundary_scalar3d(aam_new)

                dz_w = core.dz[: core.kbm1].view(1, 1, -1)
                core.adx2d = torch.sum(core.advx[:, :, : core.kbm1] * dz_w, dim=2)
                core.ady2d = torch.sum(core.advy[:, :, : core.kbm1] * dz_w, dim=2)
                core.drx2d = torch.sum(core.drhox[:, :, : core.kbm1] * dz_w, dim=2)
                core.dry2d = torch.sum(core.drhoy[:, :, : core.kbm1] * dz_w, dim=2)
                core.aam2d = torch.sum(core.aam[:, :, : core.kbm1] * dz_w, dim=2)

                (
                    core.psi,
                    core.advua,
                    core.advva,
                    core.fluxua,
                    core.fluxva,
                    core.wubot,
                    core.wvbot,
                    core.tps,
                ) = core.advave(
                    core.psi,
                    core.advua,
                    core.advva,
                    core.fluxua,
                    core.fluxva,
                    core.wubot,
                    core.wvbot,
                    core.tps,
                    core.mode,
                    core.aam2d,
                    core.uab,
                    core.vab,
                    core.dx,
                    core.dy,
                    core.ua,
                    core.va,
                    core.cbc,
                    core.aru,
                    core.arv,
                    core.d,
                )
                core.adx2d = core.adx2d - core.advua
                core.ady2d = core.ady2d - core.advva

            core.egf = core.el * core.ispi
            d_baro = core.h + core.elb
            d_term_u = core.d[1 : core.im, :] + core.d[0 : core.imm1, :]
            utf_new = core.utf.clone()
            utf_new[1 : core.im, :] = core.ua[1 : core.im, :] * d_term_u * core.isp2i
            d_term_v = core.d[:, 1 : core.jm] + core.d[:, 0 : core.jmm1]
            vtf_new = core.vtf.clone()
            vtf_new[:, 1 : core.jm] = core.va[:, 1 : core.jm] * d_term_v * core.isp2i
            core.utf, core.vtf = core.apply_implicit_boundary_velocity2d(utf_new, vtf_new)

            u_star, v_star, denom_u_star, denom_v_star = core._barotropic_predictor_with_bc()
            core.fluxua, core.fluxva, rhs_eta = core._barotropic_transport_rhs_full(u_star, v_star)
            prev_elf = getattr(core, "_eta_prev_solution", None)
            initial_guess = (
                2.0 * core.elf - prev_elf if prev_elf is not None else getattr(core, "elf", None)
            )
            eta_opts = core.implicit_numerics_policy().eta_solver_options(core)
            core.elf = core.solve_implicit_eta_cg(
                rhs_eta,
                core.d,
                max_iter=int(eta_opts.get("max_iter", 200)),
                tol=float(eta_opts.get("tol", 1.0e-12)),
                print_cg_residual=False,
                initial_guess=initial_guess,
            )
            core.elf = core._surface_bc_after_solver(core.elf)
            core._eta_prev_solution = core.elf.detach().clone()
            core.d = core.h + core.elf
            core.uaf, core.vaf = core._barotropic_corrector_with_bc(
                u_star, v_star, denom_u_star, denom_v_star
            )
            core.etf = core.elf * core.fsm

            core.ua = core.ua + 0.5 * core.smoth * (core.uab - 2.0 * core.ua + core.uaf)
            core.va = core.va + 0.5 * core.smoth * (core.vab - 2.0 * core.va + core.vaf)
            core.el = core.el + 0.5 * core.smoth * (core.elb - 2.0 * core.el + core.elf)
            core.elb = core.el.clone()
            core.el = core.elf.clone()
            core.d = core.h + core.el
            core.uab = core.ua.clone()
            core.ua = core.uaf.clone()
            core.vab = core.va.clone()
            core.va = core.vaf.clone()
            core.elb = core.apply_implicit_boundary_scalar2d(core.elb)
            core.el = core.apply_implicit_boundary_scalar2d(core.el)
            core.d = core.apply_implicit_boundary_scalar2d(core.d)
            core.uab, core.vab = core.apply_implicit_boundary_velocity2d(core.uab, core.vab)
            core.ua, core.va = core.apply_implicit_boundary_velocity2d(core.ua, core.va)
            core.egf = 0.5 * (core.elb + core.el)
            d_sum_u_new = core.d[1 : core.im, :] + core.d[0 : core.imm1, :]
            d_sum_u_baro = d_baro[1 : core.im, :] + d_baro[0 : core.imm1, :]
            utf_c = core.utf.clone()
            utf_c[1 : core.im, :] = 0.25 * (
                core.uab[1 : core.im, :] * d_sum_u_baro + core.ua[1 : core.im, :] * d_sum_u_new
            )
            d_sum_v_new = core.d[:, 1 : core.jm] + core.d[:, 0 : core.jmm1]
            d_sum_v_baro = d_baro[:, 1 : core.jm] + d_baro[:, 0 : core.jmm1]
            vtf_c = core.vtf.clone()
            vtf_c[:, 1 : core.jm] = 0.25 * (
                core.vab[:, 1 : core.jm] * d_sum_v_baro + core.va[:, 1 : core.jm] * d_sum_v_new
            )
            core.utf, core.vtf = core.apply_implicit_boundary_velocity2d(utf_c, vtf_c)

            dz_w = core.dz[: core.kbm1].view(1, 1, -1)
            tps_u = torch.sum(core.u[:, :, : core.kbm1] * dz_w, dim=2)
            dt_sum_u = core.dt[1 : core.im, :] + core.dt[0 : core.imm1, :]
            correction_u = (core.utb[1 : core.im, :] + core.utf[1 : core.im, :]) / (
                dt_sum_u + 1e-20
            )
            u_new = core.u.clone()
            u_new[1 : core.im, :, : core.kbm1] = (
                core.u[1 : core.im, :, : core.kbm1] - tps_u[1 : core.im, :].unsqueeze(2)
            ) + correction_u.unsqueeze(2)
            core.u = u_new

            tps_v = torch.sum(core.v[:, :, : core.kbm1] * dz_w, dim=2)
            dt_sum_v = core.dt[:, 1 : core.jm] + core.dt[:, 0 : core.jmm1]
            correction_v = (core.vtb[:, 1 : core.jm] + core.vtf[:, 1 : core.jm]) / (
                dt_sum_v + 1e-20
            )
            v_new = core.v.clone()
            v_new[:, 1 : core.jm, : core.kbm1] = (
                core.v[:, 1 : core.jm, : core.kbm1] - tps_v[:, 1 : core.jm].unsqueeze(2)
            ) + correction_v.unsqueeze(2)
            core.v = v_new
            core.u, core.v = core.apply_implicit_boundary_velocity3d(core.u, core.v)

            core.a, core.c, core.w = core.vertvl(
                core.a,
                core.c,
                core.w,
                core.dx,
                core.dy,
                core.dz,
                core.dt,
                core.u,
                core.v,
                core.vfluxb,
                core.vfluxf,
                core.etf,
                core.etb,
                core.dti2,
                core.im,
                core.jm,
                core.imm1,
                core.jmm1,
                core.kbm1,
            )
            core.elf, core.uaf, core.vaf, core.uf, core.vf, core.w = core.bcond(
                5, core.elf, core.uaf, core.vaf, core.uf, core.vf, core.w
            )
            core.vf = torch.zeros_like(core.vf)
            core.uf = torch.zeros_like(core.uf)

            core.q2b, core.q2, core.uf, core.a, core.c = core.advq(
                core.q2b,
                core.q2,
                core.uf,
                core.a,
                core.c,
                core.dt,
                core.dx,
                core.dy,
                core.dz,
                core.u,
                core.v,
                core.w,
                core.aam,
                core.h,
                core.dum,
                core.dvm,
                core.art,
                core.etb,
                core.etf,
                core.im,
                core.jm,
                core.imm1,
                core.jmm1,
                core.kbm1,
                core.dti2,
            )
            core.q2lb, core.q2l, core.vf, core.a, core.c = core.advq(
                core.q2lb,
                core.q2l,
                core.vf,
                core.a,
                core.c,
                core.dt,
                core.dx,
                core.dy,
                core.dz,
                core.u,
                core.v,
                core.w,
                core.aam,
                core.h,
                core.dum,
                core.dvm,
                core.art,
                core.etb,
                core.etf,
                core.im,
                core.jm,
                core.imm1,
                core.jmm1,
                core.kbm1,
                core.dti2,
            )
            (
                core.a,
                core.c,
                core.tps,
                core.dtef,
                core.ee,
                core.gg,
                core.l,
                core.kq,
                core.km,
                core.kh,
                core.uf,
                core.vf,
                core.q2b,
                core.q2lb,
                core.a,
                core.c,
            ) = core.profq(
                core.a,
                core.c,
                core.tps,
                core.dtef,
                core.ee,
                core.gg,
                core.l,
                core.kq,
                core.km,
                core.kh,
                core.uf,
                core.vf,
                core.q2,
                core.q2b,
                core.q2lb,
                core.a,
                core.c,
                core.h,
                core.etf,
                core.dti2,
                core.umol,
                core.dzz,
                core.grav,
                core.rho,
                core.kappa,
                core.u,
                core.v,
                core.dt,
                core.small,
                core.fsm,
                core.im,
                core.jm,
                core.kb,
                core.imm1,
                core.jmm1,
                core.kbm1,
                core.tbias,
                core.sbias,
                core.dz,
                core.wusurf,
                core.wubot,
                core.wvsurf,
                core.wvbot,
                core.t,
                core.s,
                core.rhoref,
                core.zz,
                core.z,
            )
            core.elf, core.uaf, core.vaf, core.uf, core.vf, core.w = core.bcond(
                6, core.elf, core.uaf, core.vaf, core.uf, core.vf, core.w
            )
            core.q2 = core.q2 + 0.5 * core.smoth * (core.uf + core.q2b - 2.0 * core.q2)
            core.q2l = core.q2l + 0.5 * core.smoth * (core.vf + core.q2lb - 2.0 * core.q2l)
            core.q2b = core.apply_implicit_boundary_scalar3d(core.q2.clone())
            core.q2 = core.apply_implicit_boundary_scalar3d(core.uf.clone())
            core.q2lb = core.apply_implicit_boundary_scalar3d(core.q2l.clone())
            core.q2l = core.apply_implicit_boundary_scalar3d(core.vf.clone())

            vertical_mixing_max = getattr(core, "vertical_mixing_max", None)
            if vertical_mixing_max is not None:
                vmax_mix = float(vertical_mixing_max)
                core.kh = torch.clamp(core.kh, min=0.0, max=vmax_mix)
                core.km = torch.clamp(core.km, min=0.0, max=vmax_mix)
                core.kq = torch.clamp(core.kq, min=0.0, max=vmax_mix)
            core.kh = core.apply_implicit_boundary_scalar3d(core.kh)
            core.km = core.apply_implicit_boundary_scalar3d(core.km)
            core.kq = core.apply_implicit_boundary_scalar3d(core.kq)

            # Explicit tracer advection (populates core.uf = t_prov, core.vf = s_prov)
            core.implicit_numerics_policy().advect_tracers(core)
            t_prov = core.uf[:, :, :LEVELS].clone()
            s_prov = core.vf[:, :, :LEVELS].clone()

            # Also compute explicit momentum predictor (u_prov, v_prov) using current baroclinic pressure gradient
            u_prov_full = core.advu(
                core.advx,
                core.aru,
                core.dz,
                core.cor,
                core.dt,
                core.e_atmos,
                core.dy,
                core.drhox,
                core.h,
                core.dti2,
                core.ub,
                core.uf.clone(),
                core.u,
                core.v,
                core.w,
                core.im,
                core.jm,
                core.kb,
                core.imm1,
                core.jmm1,
                core.kbm1,
                core.grav,
                core.egf,
                core.egb,
                core.etf,
                core.etb,
            )
            v_prov_full = core.advv(
                core.advy,
                core.arv,
                core.dz,
                core.cor,
                core.dt,
                core.e_atmos,
                core.dx,
                core.drhoy,
                core.h,
                core.dti2,
                core.vb,
                core.vf.clone(),
                core.u,
                core.v,
                core.w,
                core.im,
                core.jm,
                core.kb,
                core.imm1,
                core.jmm1,
                core.kbm1,
                core.grav,
                core.egf,
                core.egb,
                core.etf,
                core.etb,
            )
            u_prov = u_prov_full[:, :, :LEVELS]
            v_prov = v_prov_full[:, :, :LEVELS]

            du_exp = ((u_prov - core.ub[:, :, :LEVELS]) / self.cfg.scale_u) * core.dum[:, :, None]
            dv_exp = ((v_prov - core.vb[:, :, :LEVELS]) / self.cfg.scale_v) * core.dvm[:, :, None]
            dt_exp = ((t_prov - core.tb[:, :, :LEVELS]) / self.cfg.scale_t) * core.fsm[:, :, None]
            ds_exp = ((s_prov - core.sb[:, :, :LEVELS]) / self.cfg.scale_s) * core.fsm[:, :, None]

            return {
                "du_exp": du_exp,
                "dv_exp": dv_exp,
                "dt_exp": dt_exp,
                "ds_exp": ds_exp,
            }

    def _complete_step_from_pre_implicit(
        self,
        inc_pieces: dict[str, torch.Tensor] | None,
        differentiable: bool = False,
    ) -> None:
        """Complete 1 internal step from right before `proft`, injecting `inc_pieces` once pre-implicit."""
        ctx = torch.enable_grad() if differentiable else torch.no_grad()
        gen_fast = not differentiable
        with ctx:
            self.complete_substeps_executed = (
                int(getattr(self, "complete_substeps_executed", 0)) + 1
            )
            if differentiable:
                self.whitebox_ad_substeps_executed = (
                    int(getattr(self, "whitebox_ad_substeps_executed", 0)) + 1
                )
            core = self.core

            # 1. Single pre-implicit T/S injection into predictor buffers right before proft
            if inc_pieces is not None:
                core.uf = torch.cat(
                    (core.uf[:, :, :LEVELS] + inc_pieces["t"], core.uf[:, :, LEVELS:]), dim=2
                )
                core.vf = torch.cat(
                    (core.vf[:, :, :LEVELS] + inc_pieces["s"], core.vf[:, :, LEVELS:]), dim=2
                )

            (
                core.uf,
                core.wtsurf,
                core.tsurf,
                core.nbct,
                core.tps,
                core.a,
                core.c,
                core.ee,
                core.gg,
            ) = core.proft(
                core.uf,
                core.wtsurf,
                core.tsurf,
                core.nbct,
                core.tps,
                core.a,
                core.c,
                core.ee,
                core.gg,
                core.h,
                core.etf,
                core.dti2,
                core.dz,
                core.dzz,
                core.swrad,
                core.ntp,
                core.im,
                core.jm,
                core.kb,
                core.kbm1,
                core.kbm2,
                core.kh,
                core.umol,
            )
            (
                core.vf,
                core.wssurf,
                core.ssurf,
                core.nbcs,
                core.tps,
                core.a,
                core.c,
                core.ee,
                core.gg,
            ) = core.proft(
                core.vf,
                core.wssurf,
                core.ssurf,
                core.nbcs,
                core.tps,
                core.a,
                core.c,
                core.ee,
                core.gg,
                core.h,
                core.etf,
                core.dti2,
                core.dz,
                core.dzz,
                core.swrad,
                core.ntp,
                core.im,
                core.jm,
                core.kb,
                core.kbm1,
                core.kbm2,
                core.kh,
                core.umol,
            )
            core.elf, core.uaf, core.vaf, core.uf, core.vf, core.w = core.bcond(
                4, core.elf, core.uaf, core.vaf, core.uf, core.vf, core.w
            )
            core.t = core.t + 0.5 * core.smoth * (core.uf + core.tb - 2.0 * core.t)
            core.s = core.s + 0.5 * core.smoth * (core.vf + core.sb - 2.0 * core.s)
            core.uf, core.vf = core.implicit_numerics_policy().filter_tracers(
                core, core.uf, core.vf, generation_fast_path=gen_fast
            )
            core.tb = core.t.clone()
            core.t = core.uf.clone()
            core.sb = core.s.clone()
            core.s = core.vf.clone()
            core.implicit_numerics_policy().refresh_tracer_ghosts(
                core, generation_fast_path=gen_fast
            )
            core.s, core.t, core.rho = core.dens()

            # 2. Explicit momentum advection + single pre-implicit U/V injection right before profu/profv
            core.uf = core.advu(
                core.advx,
                core.aru,
                core.dz,
                core.cor,
                core.dt,
                core.e_atmos,
                core.dy,
                core.drhox,
                core.h,
                core.dti2,
                core.ub,
                core.uf,
                core.u,
                core.v,
                core.w,
                core.im,
                core.jm,
                core.kb,
                core.imm1,
                core.jmm1,
                core.kbm1,
                core.grav,
                core.egf,
                core.egb,
                core.etf,
                core.etb,
            )
            core.vf = core.advv(
                core.advy,
                core.arv,
                core.dz,
                core.cor,
                core.dt,
                core.e_atmos,
                core.dx,
                core.drhoy,
                core.h,
                core.dti2,
                core.vb,
                core.vf,
                core.u,
                core.v,
                core.w,
                core.im,
                core.jm,
                core.kb,
                core.imm1,
                core.jmm1,
                core.kbm1,
                core.grav,
                core.egf,
                core.egb,
                core.etf,
                core.etb,
            )
            if inc_pieces is not None:
                core.uf = torch.cat(
                    (core.uf[:, :, :LEVELS] + inc_pieces["u"], core.uf[:, :, LEVELS:]), dim=2
                )
                core.vf = torch.cat(
                    (core.vf[:, :, :LEVELS] + inc_pieces["v"], core.vf[:, :, LEVELS:]), dim=2
                )

            core.a, core.c, core.ee, core.gg, core.tps, core.uf, core.wubot = core.profu(
                core.a,
                core.c,
                core.ee,
                core.gg,
                core.tps,
                core.uf,
                core.wubot,
                core.etf,
                core.h,
                core.km,
                core.dti2,
                core.umol,
                core.dz,
                core.dzz,
                core.wusurf,
                core.cbc,
                core.dum,
                core.im,
                core.jm,
                core.kb,
                core.imm1,
                core.jmm1,
                core.kbm1,
                core.kbm2,
                core.ub,
                core.vb,
            )
            core.a, core.c, core.ee, core.gg, core.tps, core.vf, core.wvbot = core.profv(
                core.a,
                core.c,
                core.ee,
                core.gg,
                core.tps,
                core.vf,
                core.wvbot,
                core.dvm,
                core.dz,
                core.dzz,
                core.im,
                core.jm,
                core.kb,
                core.imm1,
                core.jmm1,
                core.kbm1,
                core.kbm2,
                core.km,
                core.cbc,
                core.ub,
                core.vb,
                core.umol,
                core.wvsurf,
                core.h,
                core.etf,
                core.dti2,
            )
            core.elf, core.uaf, core.vaf, core.uf, core.vf, core.w = core.bcond(
                3, core.elf, core.uaf, core.vaf, core.uf, core.vf, core.w
            )

            dz_w = core.dz[: core.kbm1].view(1, 1, -1)
            term_u = (
                core.uf[:, :, : core.kbm1]
                + core.ub[:, :, : core.kbm1]
                - 2.0 * core.u[:, :, : core.kbm1]
            )
            tps_u = torch.sum(term_u * dz_w, dim=2).unsqueeze(2)
            u_filtered = core.u.clone()
            u_filtered[:, :, : core.kbm1] = core.u[:, :, : core.kbm1] + 0.5 * core.smoth * (
                term_u - tps_u
            )

            term_v = (
                core.vf[:, :, : core.kbm1]
                + core.vb[:, :, : core.kbm1]
                - 2.0 * core.v[:, :, : core.kbm1]
            )
            tps_v = torch.sum(term_v * dz_w, dim=2).unsqueeze(2)
            v_filtered = core.v.clone()
            v_filtered[:, :, : core.kbm1] = core.v[:, :, : core.kbm1] + 0.5 * core.smoth * (
                term_v - tps_v
            )

            u_filtered, v_filtered = core.implicit_numerics_policy().filter_internal_velocity(
                core, u_filtered, v_filtered, generation_fast_path=gen_fast
            )
            core.ub, core.vb = core.apply_implicit_boundary_velocity3d(u_filtered, v_filtered)
            core.u, core.v = core.apply_implicit_boundary_velocity3d(
                core.uf.clone(), core.vf.clone()
            )

            core.egb = core.apply_implicit_boundary_scalar2d(core.egf.clone())
            core.etb = core.apply_implicit_boundary_scalar2d(core.et.clone())
            core.et = core.apply_implicit_boundary_scalar2d(core.etf.clone())
            core.dt = core.apply_implicit_boundary_scalar2d(core.h + core.et)
            core.utb, core.vtb = core.apply_implicit_boundary_velocity2d(
                core.utf.clone(), core.vtf.clone()
            )
            core.vfluxb = core.vfluxf.clone()

    @torch.no_grad()
    def observe_F(self) -> torch.Tensor:
        """Public feature extraction F(s_k) -> x_k in R^{161 x 90 x 82}."""
        base_101 = pack_pom_batch([self.core], self.normalizer)[0].to(self.device)  # (101, 90, 82)
        exp_inc = self._step_explicit_predictor_prefix()
        self._continuation = {"ready": True}
        inc_60 = torch.cat(
            [
                exp_inc["du_exp"].permute(2, 0, 1),
                exp_inc["dv_exp"].permute(2, 0, 1),
                exp_inc["dt_exp"].permute(2, 0, 1),
                exp_inc["ds_exp"].permute(2, 0, 1),
            ],
            dim=0,
        )  # (60, 90, 82)
        return torch.cat([base_101, inc_60], dim=0)

    def raw_to_physical_hourly(self, raw_g: torch.Tensor) -> dict[str, torch.Tensor]:
        """Step A (Unit conversion): map standardized [60, 90, 82] action to hourly physical units."""
        return {
            f: raw_g[i * LEVELS : (i + 1) * LEVELS].permute(1, 2, 0) * self.action_scales[f]
            for i, f in enumerate(OUTPUT_FIELDS)
        }

    def physical_hourly_to_raw(self, phys_pieces: dict[str, torch.Tensor]) -> torch.Tensor:
        """Inverse of Step A: map hourly physical increment dictionary back to standardized [60, 90, 82]."""
        return torch.cat(
            [(phys_pieces[f] / self.action_scales[f]).permute(2, 0, 1) for f in OUTPUT_FIELDS],
            dim=0,
        )

    def project_action(self, raw_g: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compose Unit Conversion -> W-orthogonal Projection P -> Substep Allocation (/ 6).

        Only the middle step `apply_projection_P` is the idempotent projection P (P^2 = P, P^T W = W P);
        `project_action` returns per-substep physical predictor increments `P(S * raw_g) / 6`.
        """
        phys_hourly = self.raw_to_physical_hourly(raw_g)
        proj_hourly = apply_projection_P(phys_hourly, self.get_public_geometry())
        div = self.cfg.substep_divisor
        return {f: proj_hourly[f] / div for f in OUTPUT_FIELDS}

    def project_standardized_action(self, raw_g: torch.Tensor) -> torch.Tensor:
        """Apply P_std(raw_g) = S^{-1} P(S * raw_g) and return projected standardized [60, 90, 82] tensor."""
        phys_hourly = self.raw_to_physical_hourly(raw_g)
        proj_hourly = apply_projection_P(phys_hourly, self.get_public_geometry())
        return self.physical_hourly_to_raw(proj_hourly)

    def project_standardized_action_transpose(self, grad_raw_g: torch.Tensor) -> torch.Tensor:
        """Apply exact Euclidean transpose P_std^T(grad_raw_g) = S P^T(S^{-1} grad_raw_g)."""
        inv_scaled = {
            f: grad_raw_g[i * LEVELS : (i + 1) * LEVELS].permute(1, 2, 0) / self.action_scales[f]
            for i, f in enumerate(OUTPUT_FIELDS)
        }
        adj_pieces = apply_projection_P_transpose(inv_scaled, self.get_public_geometry())
        return torch.cat(
            [(adj_pieces[f] * self.action_scales[f]).permute(2, 0, 1) for f in OUTPUT_FIELDS],
            dim=0,
        )

    @torch.no_grad()
    def step_H(
        self,
        raw_g: torch.Tensor | None,
    ) -> list[dict[str, torch.Tensor]]:
        """Advance solver 1 hour (6 internal steps of 10 min) with single pre-implicit predictor injection.

        Returns the 3 public 20-minute segment snapshots (at 20m, 40m, 60m).
        """
        inc_pieces = self.project_action(raw_g) if raw_g is not None else None
        segments: list[dict[str, torch.Tensor]] = []
        for sub in range(self.cfg.substeps_per_hour):
            if (
                sub == 0
                and self._continuation is not None
                and self._continuation.get("ready", False)
            ):
                self._continuation = None
            else:
                self._step_explicit_predictor_prefix()
            self._complete_step_from_pre_implicit(inc_pieces)
            if (sub + 1) % 2 == 0:
                segments.append(
                    {
                        "u": self.core.u[:, :, :LEVELS].clone(),
                        "v": self.core.v[:, :, :LEVELS].clone(),
                        "t": self.core.t[:, :, :LEVELS].clone(),
                        "s": self.core.s[:, :, :LEVELS].clone(),
                        "el": self.core.el.clone(),
                    }
                )
        return segments

    @torch.no_grad()
    def observe_public_scored_fields(self) -> dict[str, torch.Tensor]:
        """Return the 5 public scored coarse fields at the current timestamp (0 solver steps)."""
        return {
            "u": self.core.u[:, :, :LEVELS].clone(),
            "v": self.core.v[:, :, :LEVELS].clone(),
            "t": self.core.t[:, :, :LEVELS].clone(),
            "s": self.core.s[:, :, :LEVELS].clone(),
            "el": self.core.el.clone(),
        }

    @torch.no_grad()
    def preload_window_targets(self, member: int, day: int) -> list[dict[str, torch.Tensor]]:
        """Preload the 12 public target segment dicts and exact normalized field weights for `(member, day)`."""
        core = self.core
        dz = core.dz[:LEVELS].view(1, 1, -1)
        out: list[dict[str, torch.Tensor]] = []
        for j in range(self.cfg.segments_per_window):
            lead_seconds = (j + 1) * 1200
            ref = self.dataset.load_target(member, day, lead_seconds, map_location=self.device)
            h_ref = (core.h + ref["el"]).clamp_min(1.0)
            w_u = (core.art * core.dum * h_ref)[:, :, None] * dz
            w_v = (core.art * core.dvm * h_ref)[:, :, None] * dz
            w_t = (core.art * core.fsm * h_ref)[:, :, None] * dz
            w_el = core.art * core.fsm
            out.append(
                {
                    "u_ref": ref["u"][:, :, :LEVELS].detach().clone(),
                    "v_ref": ref["v"][:, :, :LEVELS].detach().clone(),
                    "t_ref": ref["t"][:, :, :LEVELS].detach().clone(),
                    "s_ref": ref["s"][:, :, :LEVELS].detach().clone(),
                    "el_ref": ref["el"].detach().clone(),
                    "u_norm_w": (w_u / (w_u.sum().clamp_min(1e-20) * (self.loss_scales["u"] ** 2)))
                    .detach()
                    .clone(),
                    "v_norm_w": (w_v / (w_v.sum().clamp_min(1e-20) * (self.loss_scales["v"] ** 2)))
                    .detach()
                    .clone(),
                    "t_norm_w": (w_t / (w_t.sum().clamp_min(1e-20) * (self.loss_scales["t"] ** 2)))
                    .detach()
                    .clone(),
                    "s_norm_w": (w_t / (w_t.sum().clamp_min(1e-20) * (self.loss_scales["s"] ** 2)))
                    .detach()
                    .clone(),
                    "el_norm_w": (
                        w_el / (w_el.sum().clamp_min(1e-20) * (self.loss_scales["el"] ** 2))
                    )
                    .detach()
                    .clone(),
                }
            )
        return out

    def compute_preloaded_12seg_loss(
        self,
        preloaded_targets: list[dict[str, torch.Tensor]],
        pred_segments: list[dict[str, torch.Tensor]],
        cap_per_seg: float | None = 200.0,
        init_fields: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Differentiable 12-segment normalized MSE matching `compute_12seg_loss_and_metrics`."""
        if len(pred_segments) != len(preloaded_targets):
            raise ValueError(
                f"Expected {len(preloaded_targets)} predicted segments, got {len(pred_segments)}"
            )
        seg_losses: list[torch.Tensor] = []
        for pred, targ in zip(pred_segments, preloaded_targets, strict=True):
            seg_terms: list[torch.Tensor] = []
            for f in ("u", "v", "t", "s", "el"):
                if init_fields is not None:
                    base_err = init_fields[f].double() - targ[f"{f}_ref"].double()
                    err = base_err + pred[f].double()
                else:
                    err = pred[f].double() - targ[f"{f}_ref"].double()
                seg_terms.append((err.square() * targ[f"{f}_norm_w"].double()).sum())
            seg_l = torch.stack(seg_terms).mean()
            if not torch.isfinite(seg_l):
                raise FloatingPointError("Non-finite segment loss in closed-loop emulator rollout")
            if cap_per_seg is not None:
                seg_l = seg_l.clamp(max=float(cap_per_seg))
            seg_losses.append(seg_l)
        return torch.stack(seg_losses).mean().float()

    @torch.no_grad()
    def compute_12seg_local_column_costs(
        self,
        member: int,
        day: int,
        pred_segments: list[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, float, dict[str, float]]:
        """Decompose 4-hour 12-segment loss into exact hourly wet-column local costs `(4, im, jm)`.

        Each active wet water column `(i, j)` with `fsm[i, j] > 0.5` receives its exact area-,
        depth-, layer-thickness-, and 5-field-scale-weighted squared error scaled by `N_wet_cols`,
        so `hourly_col_costs[:, col_wet].mean(dim=1).sum()` reproduces `loss_4h` to float64
        machine precision while dry columns carry identically zero cost.
        """
        core = self.core
        dz = core.dz[:LEVELS].view(1, 1, -1).double()
        fsm_bool = core.fsm > 0.5
        n_wet_cols = float(fsm_bool.double().sum().clamp_min(1.0).item())
        n_segs = len(pred_segments)
        if n_segs != self.cfg.segments_per_window:
            raise ValueError(f"Expected {self.cfg.segments_per_window} segments, got {n_segs}")

        hourly_costs = torch.zeros(
            (self.cfg.hours_per_window, self.cfg.im, self.cfg.jm),
            device=self.device,
            dtype=torch.float64,
        )
        field_sq_err = {f: 0.0 for f in ("u", "v", "t", "s", "el")}
        hourly_rmse: dict[str, float] = {}
        seg_losses_f64: list[float] = []

        for j, pred in enumerate(pred_segments):
            lead_seconds = (j + 1) * 1200
            ref = self.dataset.load_target(member, day, lead_seconds, map_location=self.device)
            h_ref = (core.h + ref["el"]).clamp_min(1.0).double()
            art_d = core.art.double()
            w_u = (art_d * core.dum.double() * h_ref)[:, :, None] * dz
            w_v = (art_d * core.dvm.double() * h_ref)[:, :, None] * dz
            w_t = (art_d * core.fsm.double() * h_ref)[:, :, None] * dz
            w_el = art_d * core.fsm.double()

            weights_map = {"u": w_u, "v": w_v, "t": w_t, "s": w_t, "el": w_el}
            col_seg = torch.zeros(
                (self.cfg.im, self.cfg.jm), device=self.device, dtype=torch.float64
            )
            seg_sum = 0.0

            for f in ("u", "v", "t", "s", "el"):
                ref_f = (ref[f][:, :, :LEVELS] if f != "el" else ref["el"]).double()
                pred_f = pred[f].double()
                w = weights_map[f]
                w_denom = w.sum().clamp_min(1e-30)
                sq_w = (pred_f - ref_f).square() * w
                col_f = sq_w.sum(dim=2) if f != "el" else sq_w

                # Fold any C-grid dry-cell face weight adjacent to an ocean cell onto its wet column
                dry_mask = ~fsm_bool
                if bool(sq_w.any()) and bool(
                    col_f.masked_select(dry_mask).abs().max().item() > 0.0
                ):
                    if f == "u":
                        shift_e = dry_mask[1:, :] & fsm_bool[:-1, :]
                        col_f[:-1, :] = col_f[:-1, :] + torch.where(
                            shift_e, col_f[1:, :], torch.zeros_like(col_f[1:, :])
                        )
                        col_f[1:, :] = torch.where(
                            shift_e, torch.zeros_like(col_f[1:, :]), col_f[1:, :]
                        )
                        shift_w = dry_mask[:-1, :] & fsm_bool[1:, :]
                        col_f[1:, :] = col_f[1:, :] + torch.where(
                            shift_w, col_f[:-1, :], torch.zeros_like(col_f[:-1, :])
                        )
                        col_f[:-1, :] = torch.where(
                            shift_w, torch.zeros_like(col_f[:-1, :]), col_f[:-1, :]
                        )
                    elif f == "v":
                        shift_n = dry_mask[:, 1:] & fsm_bool[:, :-1]
                        col_f[:, :-1] = col_f[:, :-1] + torch.where(
                            shift_n, col_f[:, 1:], torch.zeros_like(col_f[:, 1:])
                        )
                        col_f[:, 1:] = torch.where(
                            shift_n, torch.zeros_like(col_f[:, 1:]), col_f[:, 1:]
                        )
                        shift_s = dry_mask[:, :-1] & fsm_bool[:, 1:]
                        col_f[:, 1:] = col_f[:, 1:] + torch.where(
                            shift_s, col_f[:, :-1], torch.zeros_like(col_f[:, :-1])
                        )
                        col_f[:, :-1] = torch.where(
                            shift_s, torch.zeros_like(col_f[:, :-1]), col_f[:, :-1]
                        )
                    rem_dry = col_f.masked_select(dry_mask).sum()
                    if float(rem_dry.abs().item()) > 0.0:
                        col_f = torch.where(
                            fsm_bool, col_f + rem_dry / n_wet_cols, torch.zeros_like(col_f)
                        )
                    else:
                        col_f = torch.where(fsm_bool, col_f, torch.zeros_like(col_f))

                mse_f = float((sq_w.sum() / w_denom).item())
                inv_scale_sq = 1.0 / (float(self.loss_scales[f]) ** 2)
                norm_col_f = (col_f / w_denom) * inv_scale_sq
                col_seg = col_seg + (norm_col_f / 5.0)
                seg_sum += (mse_f * inv_scale_sq) / 5.0
                field_sq_err[f] += mse_f
                if j in (2, 5, 8, 11):
                    hr = (j + 1) // 3
                    hourly_rmse[f"h{hr}_{f}_rmse"] = mse_f**0.5

            seg_losses_f64.append(seg_sum)
            hr_idx = j // 3
            hourly_costs[hr_idx] = hourly_costs[hr_idx] + (col_seg * (n_wet_cols / float(n_segs)))

        hourly_costs = torch.where(
            fsm_bool.unsqueeze(0), hourly_costs, torch.zeros_like(hourly_costs)
        )
        total_loss_f64 = sum(seg_losses_f64) / float(n_segs)
        metrics: dict[str, float] = {"loss_4h": float(total_loss_f64)}
        for f in ("u", "v", "t", "s", "el"):
            rmse_f = (field_sq_err[f] / float(n_segs)) ** 0.5
            metrics[f"{f}_rmse_4h"] = rmse_f
            metrics[f"{f}_nrmse_4h"] = rmse_f / self.scales[f]
            metrics[f"{f}_loss_4h"] = (rmse_f / self.loss_scales[f]) ** 2
        metrics["momentum_loss_4h"] = (
            metrics["u_loss_4h"] + metrics["v_loss_4h"] + metrics["el_loss_4h"]
        ) / 3.0
        metrics["tracer_loss_4h"] = (metrics["t_loss_4h"] + metrics["s_loss_4h"]) / 2.0
        metrics["composite_nrmse_4h"] = (
            sum(metrics[f"{f}_nrmse_4h"] for f in ("u", "v", "t", "s", "el")) / 5.0
        )
        metrics.update(hourly_rmse)
        return hourly_costs, float(total_loss_f64), metrics

    def compute_12seg_loss_and_metrics(
        self,
        member: int,
        day: int,
        pred_segments: list[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute volume-weighted 12-segment (4h) normalized trajectory MSE and per-field RMSEs."""
        core = self.core
        dz = core.dz[:LEVELS].view(1, 1, -1)
        seg_losses: list[torch.Tensor] = []
        field_sq_err = {f: 0.0 for f in ("u", "v", "t", "s", "el")}
        hourly_rmse: dict[str, float] = {}

        for j, pred in enumerate(pred_segments):
            lead_seconds = (j + 1) * 1200  # 20 min per segment
            ref = self.dataset.load_target(member, day, lead_seconds, map_location=self.device)
            h_ref = (core.h + ref["el"]).clamp_min(1.0)
            w_u = (core.art * core.dum * h_ref)[:, :, None] * dz
            w_v = (core.art * core.dvm * h_ref)[:, :, None] * dz
            w_t = (core.art * core.fsm * h_ref)[:, :, None] * dz
            w_el = core.art * core.fsm

            weights_map = {"u": w_u, "v": w_v, "t": w_t, "s": w_t, "el": w_el}
            seg_terms: list[torch.Tensor] = []
            for f in ("u", "v", "t", "s", "el"):
                ref_f = ref[f][:, :, :LEVELS] if f != "el" else ref["el"]
                w = weights_map[f]
                diff = pred[f] - ref_f
                mse_f = (diff.square() * w).sum() / w.sum().clamp_min(1e-20)
                norm_mse_f = mse_f / (self.loss_scales[f] ** 2)
                seg_terms.append(norm_mse_f)
                field_sq_err[f] += float(mse_f.detach().item())
                if j in (2, 5, 8, 11):
                    hr = (j + 1) // 3
                    hourly_rmse[f"h{hr}_{f}_rmse"] = float(mse_f.detach().sqrt().item())

            seg_losses.append(torch.stack(seg_terms).mean())

        total_loss = torch.stack(seg_losses).mean()
        metrics: dict[str, float] = {"loss_4h": float(total_loss.detach().item())}
        for f in ("u", "v", "t", "s", "el"):
            rmse_f = (field_sq_err[f] / len(pred_segments)) ** 0.5
            metrics[f"{f}_rmse_4h"] = rmse_f
            metrics[f"{f}_nrmse_4h"] = rmse_f / self.scales[f]
            metrics[f"{f}_loss_4h"] = (rmse_f / self.loss_scales[f]) ** 2
        metrics["momentum_loss_4h"] = (
            metrics["u_loss_4h"] + metrics["v_loss_4h"] + metrics["el_loss_4h"]
        ) / 3.0
        metrics["tracer_loss_4h"] = (metrics["t_loss_4h"] + metrics["s_loss_4h"]) / 2.0
        metrics["composite_nrmse_4h"] = (
            sum(metrics[f"{f}_nrmse_4h"] for f in ("u", "v", "t", "s", "el")) / 5.0
        )
        metrics.update(hourly_rmse)
        return total_loss, metrics

    @torch.no_grad()
    def rollout_4h_window(
        self,
        member: int,
        day: int,
        policy_fn: Callable[[torch.Tensor, int], torch.Tensor | None] | None,
    ) -> tuple[float, dict[str, float], list[dict[str, torch.Tensor]]]:
        """Run a 4-hour black-box trajectory (4 hourly actions, 12 20-min output segments)."""
        self.reset(member, day)
        all_segments: list[dict[str, torch.Tensor]] = []
        for k in range(self.cfg.hours_per_window):
            if policy_fn is None:
                segs = self.step_H(None)
            else:
                x_k = self.observe_F()
                g_k = policy_fn(x_k, k)
                segs = self.step_H(g_k)
            all_segments.extend(segs)
        loss, metrics = self.compute_12seg_loss_and_metrics(member, day, all_segments)
        return float(loss.item()), metrics, all_segments

    def rollout_4h_window_whitebox_ad(
        self,
        member: int,
        day: int,
        policy_fn: Callable[[torch.Tensor, int], torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Privileged white-box 24-substep (4h) full-BPTT rollout exclusively for `stage8_ad_ref`.

        Executes the exact same 4 hourly `observe_F -> policy_fn -> project_action -> 6x10m step`
        dynamics as `rollout_4h_window` with autograd enabled and ZERO hourly detach, while
        incrementing `whitebox_ad_substeps_executed` so black-box stages can assert zero usage.
        """
        with torch.no_grad():
            self.reset(member, day)
        all_segments: list[dict[str, torch.Tensor]] = []
        with torch.enable_grad():
            for k in range(self.cfg.hours_per_window):
                base_101 = pack_pom_batch([self.core], self.normalizer)[0].to(self.device)
                exp_inc = self._step_explicit_predictor_prefix(differentiable=True)
                inc_60 = torch.cat(
                    [
                        exp_inc["du_exp"].permute(2, 0, 1),
                        exp_inc["dv_exp"].permute(2, 0, 1),
                        exp_inc["dt_exp"].permute(2, 0, 1),
                        exp_inc["ds_exp"].permute(2, 0, 1),
                    ],
                    dim=0,
                )
                x_k = torch.cat([base_101, inc_60], dim=0)
                raw_g = policy_fn(x_k, k)
                inc_pieces = self.project_action(raw_g)
                for sub in range(self.cfg.substeps_per_hour):
                    if sub > 0:
                        self._step_explicit_predictor_prefix(differentiable=True)
                    self._complete_step_from_pre_implicit(inc_pieces, differentiable=True)
                    if (sub + 1) % 2 == 0:
                        all_segments.append(
                            {
                                "u": self.core.u[:, :, :LEVELS].clone(),
                                "v": self.core.v[:, :, :LEVELS].clone(),
                                "t": self.core.t[:, :, :LEVELS].clone(),
                                "s": self.core.s[:, :, :LEVELS].clone(),
                                "el": self.core.el.clone(),
                            }
                        )
            loss_tensor, metrics = self.compute_12seg_loss_and_metrics(member, day, all_segments)
        return loss_tensor, metrics

    @torch.no_grad()
    def evaluate_hourly_rollout(
        self,
        member: int,
        start_day: int,
        hours: list[int],
        policy_fn: Callable[[torch.Tensor, int], torch.Tensor | None] | None,
    ) -> dict[str, Any]:
        """Score a continuous rollout at available reference horizons up to 48 h.

        Stored references cover the first four hours of each day and daily
        endpoints. Targets are checked before starting; the solver is reset only
        once, so later horizons measure out-of-window generalization.
        """
        from .config import validate_rollout_horizon_coverage

        horizons = list(validate_rollout_horizon_coverage([start_day], hours, min_day=1))
        for hour in horizons:
            path = self.dataset.label_path(member, start_day, hour * 3600)
            if not path.is_file():
                raise FileNotFoundError(path)
        self.reset(member, start_day)
        core = self.core
        dz = core.dz[:LEVELS].view(1, 1, -1)
        records = {}
        for hour in range(1, horizons[-1] + 1):
            action = None if policy_fn is None else policy_fn(self.observe_F(), (hour - 1) % 4)
            prediction = self.step_H(action)[-1]
            if hour not in horizons:
                continue
            reference = self.dataset.load_target(
                member, start_day, hour * 3600, map_location=self.device
            )
            depth = (core.h + reference["el"]).clamp_min(1.0)
            weights = {
                "u": (core.art * core.dum * depth)[:, :, None] * dz,
                "v": (core.art * core.dvm * depth)[:, :, None] * dz,
                "t": (core.art * core.fsm * depth)[:, :, None] * dz,
                "s": (core.art * core.fsm * depth)[:, :, None] * dz,
                "el": core.art * core.fsm,
            }
            metrics = {}
            for field, weight in weights.items():
                target = reference[field] if field == "el" else reference[field][:, :, :LEVELS]
                mse = (
                    (prediction[field] - target).square() * weight
                ).sum() / weight.sum().clamp_min(1e-20)
                rmse = float(mse.sqrt())
                metrics[f"{field}_rmse"] = rmse
                metrics[f"{field}_nrmse"] = rmse / self.scales[field]
            metrics["composite_nrmse"] = sum(metrics[f"{field}_nrmse"] for field in weights) / len(
                weights
            )
            records[str(hour)] = metrics
        return {"member": member, "start_day": start_day, "horizons": records}

    @torch.no_grad()
    def evaluate_multiday_rollout(
        self,
        member: int,
        start_day: int,
        days: int,
        policy_fn: Callable[[torch.Tensor, int], torch.Tensor | None] | None,
    ) -> dict[str, Any]:
        """Run a multi-day closed-loop rollout (up to 50 days) from `checkpoint_Day{start_day}.npz`.

        Evaluates:
        1. First 4-hour dense trajectory loss and hourly 1h..4h RMSEs (`seg00..11`);
        2. Daily state RMSE at Day +1, +2, ..., +`days` against `checkpoint_Day{start_day + d}.npz`.
        """
        self.reset(member, start_day)
        total_hours = days * 24
        first_4h_segments: list[dict[str, torch.Tensor]] = []
        daily_records: list[dict[str, float]] = []

        core = self.core
        dz = core.dz[:LEVELS].view(1, 1, -1)

        for hour_idx in range(total_hours):
            if policy_fn is None:
                segs = self.step_H(None)
            else:
                x_k = self.observe_F()
                g_k = policy_fn(x_k, hour_idx % 4)
                segs = self.step_H(g_k)

            if hour_idx < 4:
                first_4h_segments.extend(segs)

            if (hour_idx + 1) % 24 == 0:
                lead_day = (hour_idx + 1) // 24
                target_day = start_day + lead_day
                ref_daily = self.dataset.load_initial(member, target_day, map_location=self.device)
                h_ref = (core.h + ref_daily["el"]).clamp_min(1.0)
                w_u = (core.art * core.dum * h_ref)[:, :, None] * dz
                w_v = (core.art * core.dvm * h_ref)[:, :, None] * dz
                w_t = (core.art * core.fsm * h_ref)[:, :, None] * dz
                w_el = core.art * core.fsm
                weights_map = {"u": w_u, "v": w_v, "t": w_t, "s": w_t, "el": w_el}

                day_metrics: dict[str, float] = {"lead_day": float(lead_day)}
                for f in ("u", "v", "t", "s", "el"):
                    pred_f = getattr(core, f)[:, :, :LEVELS] if f != "el" else core.el
                    ref_f = ref_daily[f][:, :, :LEVELS] if f != "el" else ref_daily["el"]
                    w = weights_map[f]
                    rmse_f = float(
                        ((pred_f - ref_f).square() * w)
                        .sum()
                        .div(w.sum().clamp_min(1e-20))
                        .sqrt()
                        .item()
                    )
                    day_metrics[f"{f}_rmse"] = rmse_f
                    day_metrics[f"{f}_nrmse"] = rmse_f / self.scales[f]
                day_metrics["composite_nrmse"] = (
                    sum(day_metrics[f"{f}_nrmse"] for f in ("u", "v", "t", "s", "el")) / 5.0
                )
                daily_records.append(day_metrics)
                if self.device.index == 0 and (lead_day % 10 == 0 or lead_day == days):
                    print(
                        f"    [Rollout m{member} start={start_day}] Day {lead_day:02d}/{days} | "
                        f"composite_nrmse={day_metrics['composite_nrmse']:.5f}",
                        flush=True,
                    )

        _, intraday_metrics = self.compute_12seg_loss_and_metrics(
            member, start_day, first_4h_segments
        )

        summary: dict[str, Any] = {
            "member": member,
            "start_day": start_day,
            "days": days,
            "intraday_4h": intraday_metrics,
            "daily_records": daily_records,
        }
        for horizon in (10, 30, 50):
            if days >= horizon:
                sub = daily_records[:horizon]
                for f in ("u", "v", "t", "s", "el"):
                    summary[f"first{horizon}_{f}_rmse"] = sum(r[f"{f}_rmse"] for r in sub) / len(
                        sub
                    )
                    summary[f"first{horizon}_{f}_nrmse"] = sum(r[f"{f}_nrmse"] for r in sub) / len(
                        sub
                    )
                summary[f"first{horizon}_composite_nrmse"] = sum(
                    r["composite_nrmse"] for r in sub
                ) / len(sub)
        if daily_records:
            for f in ("u", "v", "t", "s", "el"):
                summary[f"mean_{f}_rmse"] = sum(r[f"{f}_rmse"] for r in daily_records) / len(
                    daily_records
                )
                summary[f"mean_{f}_nrmse"] = sum(r[f"{f}_nrmse"] for r in daily_records) / len(
                    daily_records
                )
            summary["mean_composite_nrmse"] = sum(
                r["composite_nrmse"] for r in daily_records
            ) / len(daily_records)
        return summary
