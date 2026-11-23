"""Stage I parameter-space subspace construction, readout drift compensation, and Gauss-Newton screening."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..config import ForgePOM3DConfig
from ..distributed import ddp_mean_dict
from ..network import ForgePOMNet
from ..packing import pack_reference_segments, pack_segments_to_tensor
from ..physics_env import BlackBoxProvider3D
from ..probes import measure_action_jacobian_ranks


class Stage1DriftCompensator:
    """Truncated-SVD readout compensation canceling second-order trunk feature drift.

    Given nominal parameters `theta0` and orthonormal mixed head+body directions `{u_m}_{m=0}^{M-1}`,
    any coordinate vector `c in R^M` defines a raw parameter candidate `theta_raw = theta0 + U c`
    and an exact first-order intended output `G_int(x) = G_{theta0}(x) + sum_m c_m (dG_{theta0}/dtheta u_m)(x)`
    evaluated on cached public nominal observations `x` (zero extra solver queries).
    When `comp_rank > 0`, a rank-`s` truncated pseudoinverse adjustment `Delta W_comp` on the
    augmented perturbed trunk features cancels the dominant `O(||Delta theta||^2)` nonlinear drift
    `G_int - G_{theta_raw}` while preserving the exact first-order directional derivative.
    """

    def __init__(
        self,
        model: ForgePOMNet,
        theta0: torch.Tensor,
        directions: list[torch.Tensor],
        obs_samples: list[torch.Tensor],
        wq_60: torch.Tensor,
        comp_rank: int,
    ) -> None:
        self.model = model
        self.theta0 = theta0.detach().clone()
        self.directions = directions
        self.comp_rank = max(int(comp_rank), 0)
        self.last_comp_l2_norm: float = 0.0
        self.enabled = (self.comp_rank > 0) and bool(obs_samples) and bool(directions)
        if not self.enabled:
            return

        was_tr = model.training
        model.eval()
        head_mask, body_mask = model.head_body_index_masks(device=self.theta0.device)
        self.body_mask = body_mask
        self.base_head_norm = max(
            float(torch.linalg.vector_norm(self.theta0[head_mask]).item()),
            1.0,
        )
        self.xb = torch.stack(obs_samples[:4], dim=0).detach()
        self.n_frames = int(self.xb.shape[0])
        self.p_aug = int(model.hidden_channels) + 1
        self.out_ch = int(model.out_channels)

        param_meta = [(name, p.numel(), p.shape) for name, p in model.named_parameters()]

        def _unflatten(vec: torch.Tensor) -> dict[str, torch.Tensor]:
            out_d: dict[str, torch.Tensor] = {}
            off = 0
            for name, numel, shape in param_meta:
                out_d[name] = vec[off : off + numel].view(shape)
                off += numel
            return out_d

        with torch.no_grad():
            params0 = _unflatten(self.theta0)

            def _f_call(pd: dict[str, torch.Tensor]) -> torch.Tensor:
                return torch.func.functional_call(model, pd, (self.xb,))

            self.base_out = _f_call(params0).detach()
            self.dir_resps: list[torch.Tensor] = []
            for u_m in directions:
                _, jvp_m = torch.func.jvp(_f_call, (params0,), (_unflatten(u_m),))
                self.dir_resps.append(jvp_m.detach())

        self.head_groups: list[tuple[int, int, torch.Tensor, float]] = []
        for ch_lo, ch_hi, w2d in (
            (0, 15, wq_60[0]),
            (15, 30, wq_60[15]),
            (30, 60, wq_60[30]),
        ):
            w_clamped = w2d.double().clamp_min(0.0)
            w_sqrt = w_clamped.sqrt()
            denom = max(float(w_clamped.sum().item()) * float(self.n_frames), 1.0)
            self.head_groups.append((ch_lo, ch_hi, w_sqrt, denom))

        model.set_flat_params(self.theta0)
        if was_tr:
            model.train()

    @torch.no_grad()
    def _compensate(
        self,
        cand_raw: torch.Tensor,
        intended_out: torch.Tensor,
        head_cap_mult: float,
    ) -> torch.Tensor:
        body_step_norm = float(
            torch.linalg.vector_norm((cand_raw - self.theta0)[self.body_mask]).item()
        )
        if body_step_norm <= 1e-30:
            self.last_comp_l2_norm = 0.0
            return cand_raw

        was_tr = self.model.training
        self.model.eval()
        self.model.set_flat_params(cand_raw)

        phi = self.model.extract_trunk_features(self.xb)
        actual_out = self.model.forward_from_features(phi)
        err = intended_out.double() - actual_out.double()

        ones_b = torch.ones(
            (phi.shape[0], 1, phi.shape[2], phi.shape[3]),
            device=phi.device,
            dtype=torch.float64,
        )
        phi_aug = torch.cat([phi.double(), ones_b], dim=1)

        delta_w_t = torch.zeros(
            (self.p_aug, self.out_ch),
            device=cand_raw.device,
            dtype=torch.float64,
        )
        for ch_lo, ch_hi, w_sqrt_2d, grp_denom in self.head_groups:
            w_4d = w_sqrt_2d.unsqueeze(0).unsqueeze(0)
            a_mat = (phi_aug * w_4d).permute(0, 2, 3, 1).reshape(-1, self.p_aug)
            e_mat = (err[:, ch_lo:ch_hi] * w_4d).permute(0, 2, 3, 1).reshape(-1, ch_hi - ch_lo)
            gram_g = (a_mat.T @ a_mat) / grp_denom
            rhs_g = (a_mat.T @ e_mat) / grp_denom
            if dist.is_initialized():
                dist.all_reduce(gram_g, op=dist.ReduceOp.AVG)
                dist.all_reduce(rhs_g, op=dist.ReduceOp.AVG)
            evals_g, evecs_g = torch.linalg.eigh(gram_g)
            lam_floor = max(float(evals_g[-1].item()) * 1e-12, 1e-24)
            valid_cnt = int((evals_g > lam_floor).sum().item())
            keep = min(self.comp_rank, valid_cnt)
            if keep > 0:
                v_k = evecs_g[:, self.p_aug - keep :]
                l_k = evals_g[self.p_aug - keep :]
                delta_w_t[:, ch_lo:ch_hi] = v_k @ ((v_k.T @ rhs_g) / l_k.unsqueeze(1))

        delta_head_flat = self.model.head_matrix_to_flat_delta(delta_w_t.T.to(dtype=cand_raw.dtype))
        comp_norm = float(torch.linalg.vector_norm(delta_head_flat).item())
        cap_lim = float(head_cap_mult) * self.base_head_norm
        if comp_norm > cap_lim and comp_norm > 1e-30:
            delta_head_flat = delta_head_flat * (cap_lim / comp_norm)
            comp_norm = cap_lim

        self.last_comp_l2_norm = comp_norm
        self.model.set_flat_params(self.theta0)
        if was_tr:
            self.model.train()
        return cand_raw + delta_head_flat

    @torch.no_grad()
    def apply_single_dir(
        self,
        m_idx: int,
        scale: float,
        head_cap_mult: float = 1.0,
    ) -> torch.Tensor:
        s = float(scale)
        cand_raw = self.theta0 + s * self.directions[m_idx]
        if not self.enabled or abs(s) <= 1e-30:
            self.last_comp_l2_norm = 0.0
            return cand_raw
        intended_out = self.base_out + s * self.dir_resps[m_idx]
        return self._compensate(cand_raw, intended_out, head_cap_mult=head_cap_mult)

    @torch.no_grad()
    def apply_coords(
        self,
        coords: torch.Tensor,
        head_cap_mult: float = 2.0,
    ) -> torch.Tensor:
        delta = torch.zeros_like(self.theta0)
        for m_idx, u_m in enumerate(self.directions):
            delta = delta + float(coords[m_idx].item()) * u_m
        cand_raw = self.theta0 + delta
        if not self.enabled:
            self.last_comp_l2_norm = 0.0
            return cand_raw
        intended_out = self.base_out.clone()
        for m_idx, jvp_m in enumerate(self.dir_resps):
            c_m = float(coords[m_idx].item())
            if abs(c_m) > 1e-30:
                intended_out = intended_out + c_m * jvp_m
        return self._compensate(cand_raw, intended_out, head_cap_mult=head_cap_mult)


@dataclass
class Stage1RoundResult:
    theta0: torch.Tensor
    theta_stage1: torch.Tensor
    chosen_alpha: float
    chosen_comp_l2: float
    eff_comp_rank: int
    pre_round_train_loss: float
    stage1_train_loss: float
    rank_info: dict[str, int]
    window_rollouts_used: int
    nom_obs_list: list[torch.Tensor]
    nom_actions_raw_list: list[list[torch.Tensor]]
    nom_actions_proj_list: list[list[torch.Tensor]]
    nom_segments_list: list[list[torch.Tensor]]
    nom_losses_list: list[float]
    ref_segments_list: list[list[torch.Tensor]]


def run_stage1_subspace_step(
    cfg: ForgePOM3DConfig,
    env: BlackBoxProvider3D,
    model: ForgePOMNet,
    my_windows: list[tuple[int, int]],
    round_idx: int,
    ablation: str,
    mask_60: torch.Tensor,
    wq_60: torch.Tensor,
    wet_61_cnt: float,
    world_size: int,
    device: torch.device,
) -> Stage1RoundResult:
    """Execute nominal window rollouts and Stage I antithetic Gauss-Newton subspace step."""
    model.eval()
    nom_obs_list: list[torch.Tensor] = []
    nom_actions_raw_list: list[list[torch.Tensor]] = []
    nom_actions_proj_list: list[list[torch.Tensor]] = []
    nom_segments_list: list[list[torch.Tensor]] = []
    nom_losses_list: list[float] = []
    ref_segments_list: list[list[torch.Tensor]] = []
    base_obs_samples: list[torch.Tensor] = []
    base_train_loss = 0.0

    with torch.no_grad():
        for m, d in my_windows:
            env.reset(m, d)
            xs_w: list[torch.Tensor] = []
            acts_raw_w: list[torch.Tensor] = []
            acts_proj_w: list[torch.Tensor] = []
            segs_w_dicts: list[dict[str, torch.Tensor]] = []
            for _k in range(cfg.hours_per_window):
                x_k = env.observe_F()
                g_k = model(x_k).detach()
                segs_w_dicts.extend(env.step_H(g_k))
                xs_w.append(x_k.detach())
                base_obs_samples.append(x_k.detach())
                acts_raw_w.append(g_k)
                acts_proj_w.append(env.project_standardized_action(g_k).detach())
            loss_w, _ = env.compute_12seg_loss_and_metrics(m, d, segs_w_dicts)
            lw_f = float(loss_w.item())
            base_train_loss += lw_f / len(my_windows)
            nom_losses_list.append(lw_f)
            nom_obs_list.append(torch.stack(xs_w, dim=0))
            nom_actions_raw_list.append(acts_raw_w)
            nom_actions_proj_list.append(acts_proj_w)
            nom_segments_list.append(pack_segments_to_tensor(env, segs_w_dicts))
            ref_segments_list.append(pack_reference_segments(env, m, d))

    rollouts_used = len(my_windows) * world_size
    pre_round_train_loss = ddp_mean_dict({"l": base_train_loss}, device)["l"]

    theta0 = model.get_flat_params().clone()
    eff_comp_rank = (
        0 if ablation in ("no_stage1", "no_body_comp") else max(int(cfg.forge_comp_rank), 0)
    )
    chosen_comp_l2 = 0.0

    if ablation == "no_stage1":
        rank_info = {"action_jac_rank_full": 0, "action_jac_rank_body_only": 0}
        chosen_alpha = 0.0
        theta_stage1 = theta0.clone()
        model.set_flat_params(theta_stage1)
        stage1_train_loss = pre_round_train_loss
    else:
        m_sub = int(cfg.forge_num_directions)
        directions = model.build_mixed_head_body_basis(
            num_directions=m_sub, seed=cfg.seed + round_idx * 313
        )
        rank_info = measure_action_jacobian_ranks(
            model, directions, base_obs_samples, mask_60, eps=cfg.forge_probe_eps
        )
        eps_s1 = float(cfg.forge_probe_eps)
        s1_comp = Stage1DriftCompensator(
            model=model,
            theta0=theta0,
            directions=directions,
            obs_samples=base_obs_samples,
            wq_60=wq_60,
            comp_rank=eff_comp_rank,
        )
        s1_probe_thetas = [
            (
                s1_comp.apply_single_dir(m_dir, +eps_s1, head_cap_mult=1.0),
                s1_comp.apply_single_dir(m_dir, -eps_s1, head_cap_mult=1.0),
            )
            for m_dir in range(m_sub)
        ]

        local_jtj = torch.zeros((m_sub, m_sub), device=device, dtype=torch.float64)
        local_grad = torch.zeros(m_sub, device=device, dtype=torch.float64)
        n_total_wet = float(len(my_windows) * 12) * wet_61_cnt

        with torch.no_grad():
            for w_idx, (m, d) in enumerate(my_windows):
                r0_segs = [
                    (nom_segments_list[w_idx][j] - ref_segments_list[w_idx][j]).double()
                    for j in range(12)
                ]
                j_cols: list[torch.Tensor] = []
                for m_dir in range(m_sub):
                    theta_pos_m, theta_neg_m = s1_probe_thetas[m_dir]
                    model.set_flat_params(theta_pos_m)
                    _, _, segs_pos_d = env.rollout_4h_window(m, d, lambda x_k, _k: model(x_k))
                    segs_pos = pack_segments_to_tensor(env, segs_pos_d)

                    model.set_flat_params(theta_neg_m)
                    _, _, segs_neg_d = env.rollout_4h_window(m, d, lambda x_k, _k: model(x_k))
                    segs_neg = pack_segments_to_tensor(env, segs_neg_d)

                    diff_col = torch.cat(
                        [
                            (
                                (segs_pos[j].double() - segs_neg[j].double()) / (2.0 * eps_s1)
                            ).reshape(-1)
                            for j in range(12)
                        ],
                        dim=0,
                    )
                    j_cols.append(diff_col)

                j_mat = torch.stack(j_cols, dim=1)
                r0_vec = torch.cat([s.reshape(-1) for s in r0_segs], dim=0)
                local_jtj += (j_mat.T @ j_mat) / n_total_wet
                local_grad += (j_mat.T @ r0_vec) / n_total_wet
            model.set_flat_params(theta0)
        rollouts_used += 2 * m_sub * len(my_windows) * world_size

        if dist.is_initialized():
            dist.all_reduce(local_jtj, op=dist.ReduceOp.AVG)
            dist.all_reduce(local_grad, op=dist.ReduceOp.AVG)

        evals_s1, evecs_s1 = torch.linalg.eigh(local_jtj)
        evals_s1 = evals_s1.clamp_min(0.0)
        top_r = min(int(cfg.forge_curv_rank), m_sub)
        lam_damp = float(cfg.forge_ridge_damping) + float(evals_s1.max().item()) * 1e-3
        h_approx = evecs_s1[:, m_sub - top_r :] @ torch.diag(evals_s1[m_sub - top_r :]) @ evecs_s1[
            :, m_sub - top_r :
        ].T + lam_damp * torch.eye(m_sub, device=device, dtype=torch.float64)
        c_star = -torch.linalg.solve(h_approx, local_grad)

        best_step_loss = pre_round_train_loss
        chosen_alpha = 0.0
        best_theta_stage1 = theta0.clone()
        with torch.no_grad():
            for alpha in cfg.forge_head_step_grid:
                cand_theta_s1 = s1_comp.apply_coords(c_star * float(alpha), head_cap_mult=2.0)
                cand_comp_l2 = s1_comp.last_comp_l2_norm
                model.set_flat_params(cand_theta_s1)
                cand_loss = 0.0
                for m, d in my_windows:
                    l_w, _, _ = env.rollout_4h_window(m, d, lambda x_k, _k: model(x_k))
                    cand_loss += float(l_w) / len(my_windows)
                rollouts_used += len(my_windows) * world_size
                sync_l = ddp_mean_dict({"l": cand_loss}, device)["l"]
                if math.isfinite(sync_l) and sync_l < best_step_loss - 1e-7:
                    best_step_loss = sync_l
                    chosen_alpha = float(alpha)
                    chosen_comp_l2 = cand_comp_l2
                    best_theta_stage1 = cand_theta_s1.clone()

            theta_stage1 = best_theta_stage1
            model.set_flat_params(theta_stage1)
        stage1_train_loss = best_step_loss

    return Stage1RoundResult(
        theta0=theta0,
        theta_stage1=theta_stage1,
        chosen_alpha=chosen_alpha,
        chosen_comp_l2=chosen_comp_l2,
        eff_comp_rank=eff_comp_rank,
        pre_round_train_loss=pre_round_train_loss,
        stage1_train_loss=stage1_train_loss,
        rank_info=rank_info,
        window_rollouts_used=rollouts_used,
        nom_obs_list=nom_obs_list,
        nom_actions_raw_list=nom_actions_raw_list,
        nom_actions_proj_list=nom_actions_proj_list,
        nom_segments_list=nom_segments_list,
        nom_losses_list=nom_losses_list,
        ref_segments_list=ref_segments_list,
    )


__all__ = [
    "Stage1DriftCompensator",
    "Stage1RoundResult",
    "run_stage1_subspace_step",
]
