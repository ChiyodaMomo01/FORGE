"""Causal 30-block segment response operator and 6-block observation response operator for 3D POM."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..probes import HADAMARD_4X4


def apply_wet_normalized_laplacian(a: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply wet-masked 5-point horizontal Laplacian scaled by 1/8 so ||L||_2 <= 1.0 and L^T = L."""
    single = a.dim() == 3
    if single:
        a = a.unsqueeze(0)
    m = mask.unsqueeze(0)
    am = a * m

    f_ip = F.pad(am[:, :, 1:, :], (0, 0, 0, 1))
    f_im = F.pad(am[:, :, :-1, :], (0, 0, 1, 0))
    f_jp = F.pad(am[:, :, :, 1:], (0, 1, 0, 0))
    f_jm = F.pad(am[:, :, :, :-1], (1, 0, 0, 0))

    m_ip = F.pad(m[:, :, 1:, :], (0, 0, 0, 1))
    m_im = F.pad(m[:, :, :-1, :], (0, 0, 1, 0))
    m_jp = F.pad(m[:, :, :, 1:], (0, 1, 0, 0))
    m_jm = F.pad(m[:, :, :, :-1], (1, 0, 0, 0))

    lap = 0.125 * ((f_ip + f_im + f_jp + f_jm) - (m_ip + m_im + m_jp + m_jm) * am) * m
    return lap[0] if single else lap


class CausalBlockResponseOperator3D(nn.Module):
    """Causal 30-block response operator K_{jk} = C_{jk} Pi_wet + D_{jk} L + P_{jk} for 3D POM."""

    def __init__(
        self,
        wet_mask_60: torch.Tensor,
        wet_mask_61: torch.Tensor,
        im: int = 90,
        jm: int = 82,
        in_ch: int = 60,
        out_ch: int = 61,
        num_hours: int = 4,
        num_segments: int = 12,
        low_rank: int = 4,
        gain_bound: float = 10.0,
        weights_61: torch.Tensor | None = None,
        action_weights_60: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.im = im
        self.jm = jm
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_hours = num_hours
        self.num_segments = num_segments
        self.low_rank = low_rank
        self.gain_bound = gain_bound

        m60 = wet_mask_60.float()
        m61 = wet_mask_61.float()
        self.register_buffer("wet_mask_60", m60)
        self.register_buffer("wet_mask_61", m61)

        wo_init = (weights_61.float() * m61) if weights_61 is not None else m61.clone()
        wo_init = wo_init * (m61.sum().clamp_min(1.0) / wo_init.sum().clamp_min(1e-20))
        wq_init = (
            (action_weights_60.float() * m60) if action_weights_60 is not None else m60.clone()
        )
        wq_init = wq_init * (m60.sum().clamp_min(1.0) / wq_init.sum().clamp_min(1e-20))
        self.register_buffer("default_wo_61", wo_init)
        self.register_buffer("default_wq_60", wq_init)

        self.causal_pairs: list[tuple[int, int]] = []
        self.pair_to_idx: dict[tuple[int, int], int] = {}
        for k in range(num_hours):
            for j in range(3 * k, num_segments):
                self.pair_to_idx[(j, k)] = len(self.causal_pairs)
                self.causal_pairs.append((j, k))
        num_blocks = len(self.causal_pairs)

        self.C_blocks = nn.Parameter(torch.zeros(num_blocks, out_ch, in_ch), requires_grad=False)
        self.D_blocks = nn.Parameter(torch.zeros(num_blocks, out_ch, in_ch), requires_grad=False)
        self.M_blocks = nn.Parameter(
            torch.zeros(num_blocks, low_rank, low_rank), requires_grad=False
        )

        basis = self._build_orthonormal_spatial_basis(wet_mask_60[0], im, jm, low_rank)
        self.register_buffer("spatial_basis", basis)

    @staticmethod
    def _build_orthonormal_spatial_basis(
        mask_2d: torch.Tensor, im: int, jm: int, rank: int
    ) -> torch.Tensor:
        device = mask_2d.device
        x = torch.linspace(0.0, math.pi, im, device=device).unsqueeze(1).expand(im, jm)
        y = torch.linspace(0.0, math.pi, jm, device=device).unsqueeze(0).expand(im, jm)
        modes = [
            torch.sin(x) * torch.sin(y),
            torch.sin(x) * torch.sin(2.0 * y),
            torch.sin(2.0 * x) * torch.sin(y),
            torch.sin(2.0 * x) * torch.sin(2.0 * y),
        ][:rank]
        mat = torch.stack([m * mask_2d for m in modes], dim=-1).reshape(im * jm, rank)
        q, _ = torch.linalg.qr(mat)
        return q.reshape(im, jm, rank)

    def apply_normalized_laplacian(
        self, a: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply wet-masked 5-point horizontal Laplacian scaled by 1/8 so ||L||_2 <= 1.0 and L^T = L."""
        return apply_wet_normalized_laplacian(a, self.wet_mask_60 if mask is None else mask)

    def forward_block(self, j: int, k: int, delta_a_k: torch.Tensor) -> torch.Tensor:
        """Apply K_{jk} to delta_a_k in R^{60 x 90 x 82} -> R^{61 x 90 x 82}."""
        idx = self.pair_to_idx[(j, k)]
        c_mat = self.C_blocks[idx]
        d_mat = self.D_blocks[idx]
        m_mat = self.M_blocks[idx]

        a_wet = delta_a_k * self.wet_mask_60
        lap_a = self.apply_normalized_laplacian(a_wet, self.wet_mask_60)

        out_c = torch.einsum("oi,ixy->oxy", c_mat, a_wet)
        out_d = torch.einsum("oi,ixy->oxy", d_mat, lap_a)

        b_spatial = self.spatial_basis
        a_mean = a_wet.mean(dim=0)
        proj_in = torch.einsum("xy,xyr->r", a_mean, b_spatial) / math.sqrt(self.in_ch)
        proj_out = torch.mv(m_mat, proj_in) / math.sqrt(self.out_ch)
        spatial_out = torch.einsum("xyr,r->xy", b_spatial, proj_out)
        out_p = spatial_out.unsqueeze(0).expand(self.out_ch, -1, -1)

        return (out_c + out_d + out_p) * self.wet_mask_61

    def adjoint_block(self, j: int, k: int, q_j: torch.Tensor) -> torch.Tensor:
        """Apply exact transpose K_{jk}^T to q_j in R^{61 x 90 x 82} -> R^{60 x 90 x 82}."""
        idx = self.pair_to_idx[(j, k)]
        c_mat = self.C_blocks[idx]
        d_mat = self.D_blocks[idx]
        m_mat = self.M_blocks[idx]

        q_wet = q_j * self.wet_mask_61
        adj_c = torch.einsum("oi,oxy->ixy", c_mat, q_wet) * self.wet_mask_60
        dt_q = torch.einsum("oi,oxy->ixy", d_mat, q_wet) * self.wet_mask_60
        adj_d = self.apply_normalized_laplacian(dt_q, self.wet_mask_60)

        b_spatial = self.spatial_basis
        q_sum = q_wet.sum(dim=0)
        q_proj = torch.einsum("xy,xyr->r", q_sum, b_spatial) / math.sqrt(self.out_ch)
        in_proj = torch.mv(m_mat.T, q_proj) / math.sqrt(self.in_ch)
        spatial_in = torch.einsum("xyr,r->xy", b_spatial, in_proj) / float(self.in_ch)
        adj_p = spatial_in.unsqueeze(0).expand(self.in_ch, -1, -1) * self.wet_mask_60

        return adj_c + adj_d + adj_p

    def forward_trajectory(self, delta_actions: list[torch.Tensor]) -> list[torch.Tensor]:
        """Map [delta_a_0, ..., delta_a_3] to 12 segment perturbations [delta_y_0, ..., delta_y_11]."""
        outputs: list[torch.Tensor] = []
        for j in range(self.num_segments):
            max_k = j // 3
            acc = torch.zeros(
                self.out_ch,
                self.im,
                self.jm,
                device=delta_actions[0].device,
                dtype=delta_actions[0].dtype,
            )
            for k in range(max_k + 1):
                acc = acc + self.forward_block(j, k, delta_actions[k])
            outputs.append(acc)
        return outputs

    def adjoint_trajectory(self, q_segments: list[torch.Tensor]) -> list[torch.Tensor]:
        """Map 12 segment covectors [q_0, ..., q_11] to 4 hourly action covectors [g_0, ..., g_3]."""
        grads: list[torch.Tensor] = []
        for k in range(self.num_hours):
            acc = torch.zeros(
                self.in_ch,
                self.im,
                self.jm,
                device=q_segments[0].device,
                dtype=q_segments[0].dtype,
            )
            for j in range(3 * k, self.num_segments):
                acc = acc + self.adjoint_block(j, k, q_segments[j])
            grads.append(acc)
        return grads

    @torch.no_grad()
    def _metric_factors(
        self,
        weights_61: torch.Tensor | None = None,
        action_weights_60: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        device = self.C_blocks.device
        m60 = self.wet_mask_60.double()
        m61 = self.wet_mask_61.double()
        wo = (
            (weights_61.to(device=device, dtype=torch.float64) * m61)
            if weights_61 is not None
            else self.default_wo_61.double()
        )
        wo = wo * (m61.sum().clamp_min(1.0) / wo.sum().clamp_min(1e-20))

        wq = (
            (action_weights_60.to(device=device, dtype=torch.float64) * m60)
            if action_weights_60 is not None
            else self.default_wq_60.double()
        )
        wq = wq * (m60.sum().clamp_min(1.0) / wq.sum().clamp_min(1e-20))

        d_o = (wo.sum(dim=(1, 2)) / m61.sum(dim=(1, 2)).clamp_min(1.0)).clamp_min(1e-20)
        d_q = (wq.sum(dim=(1, 2)) / m60.sum(dim=(1, 2)).clamp_min(1.0)).clamp_min(1e-20)

        so_wet = (wo / d_o[:, None, None])[m61 > 0.5]
        sq_wet = (wq / d_q[:, None, None])[m60 > 0.5]
        sigma_o_max = (
            float(so_wet.clamp_min(0.0).sqrt().max().item()) if so_wet.numel() > 0 else 1.0
        )
        sigma_q_inv = (
            float((1.0 / sq_wet.clamp_min(1e-20).sqrt()).max().item())
            if sq_wet.numel() > 0
            else 1.0
        )
        return d_o.sqrt(), (1.0 / d_q.sqrt()), sigma_o_max * sigma_q_inv

    @torch.no_grad()
    def block_norm_matrix(
        self,
        weights_61: torch.Tensor | None = None,
        action_weights_60: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sqrt_do, inv_sqrt_dq, sp_factor = self._metric_factors(weights_61, action_weights_60)
        lr_scale = (
            float(sqrt_do.norm().item())
            / math.sqrt(float(self.out_ch))
            * float(inv_sqrt_dq.norm().item())
            / math.sqrt(float(self.in_ch))
        )
        b_mat = torch.zeros(
            self.num_segments,
            self.num_hours,
            device=self.C_blocks.device,
            dtype=torch.float64,
        )
        for (j, k), idx in self.pair_to_idx.items():
            c_weighted = sqrt_do[:, None] * self.C_blocks[idx].double() * inv_sqrt_dq[None, :]
            d_weighted = sqrt_do[:, None] * self.D_blocks[idx].double() * inv_sqrt_dq[None, :]
            norm_c = float(torch.linalg.matrix_norm(c_weighted, ord=2).item())
            norm_d = float(torch.linalg.matrix_norm(d_weighted, ord=2).item())
            norm_m = (
                lr_scale
                * float(torch.linalg.matrix_norm(self.M_blocks[idx].double(), ord=2).item())
                / float(self.in_ch)
            )
            b_mat[j, k] = sp_factor * (norm_c + norm_d + norm_m)
        return b_mat

    @torch.no_grad()
    def trajectory_norm_bound(
        self,
        weights_61: torch.Tensor | None = None,
        action_weights_60: torch.Tensor | None = None,
    ) -> float:
        b_mat = self.block_norm_matrix(weights_61=weights_61, action_weights_60=action_weights_60)
        return float(torch.linalg.matrix_norm(b_mat, ord=2).item())

    @torch.no_grad()
    def enforce_gain_bounds(
        self,
        weights_61: torch.Tensor | None = None,
        action_weights_60: torch.Tensor | None = None,
    ) -> dict[str, float]:
        b_init = self.block_norm_matrix(weights_61=weights_61, action_weights_60=action_weights_60)
        max_block = 0.0
        for (j, k), idx in self.pair_to_idx.items():
            total = float(b_init[j, k].item())
            if total > self.gain_bound and total > 1e-12:
                scale = self.gain_bound / total
                self.C_blocks[idx].mul_(scale)
                self.D_blocks[idx].mul_(scale)
                self.M_blocks[idx].mul_(scale)
                total = self.gain_bound
            max_block = max(max_block, total)

        traj_bound = self.trajectory_norm_bound(
            weights_61=weights_61, action_weights_60=action_weights_60
        )
        if traj_bound > self.gain_bound and traj_bound > 1e-12:
            scale = self.gain_bound / traj_bound
            self.C_blocks.mul_(scale)
            self.D_blocks.mul_(scale)
            self.M_blocks.mul_(scale)
            max_block *= scale
            traj_bound = self.trajectory_norm_bound(
                weights_61=weights_61, action_weights_60=action_weights_60
            )
        return {"max_block_norm": max_block, "trajectory_norm_bound": traj_bound}

    @torch.no_grad()
    def mask_cross_hour_blocks_(self) -> None:
        """Zero all 18 cross-hour blocks (j // 3 != k), keeping only the 12 intra-hour blocks."""
        for (j, k), idx in self.pair_to_idx.items():
            if (j // 3) != k:
                self.C_blocks[idx].zero_()
                self.D_blocks[idx].zero_()
                self.M_blocks[idx].zero_()

    @torch.no_grad()
    def fit_from_probes(
        self,
        action_deltas: list[list[torch.Tensor]],
        segment_deltas: list[list[torch.Tensor]],
        ridge: float = 5e-2,
        ema_decay: float = 0.0,
        weights_61: torch.Tensor | None = None,
        action_weights_60: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Simultaneously solve all active causal blocks per segment j via Schur-complement elimination."""
        num_samples = len(action_deltas)
        if num_samples == 0:
            return {
                "max_block_norm": 0.0,
                "unconstrained_max_block_norm": 0.0,
                "trajectory_norm_bound": 0.0,
                "unconstrained_trajectory_norm_bound": 0.0,
                "gain_scale_applied": 1.0,
                "unconstrained_fit_rel_err": 1.0,
                "fit_rel_err": 1.0,
            }

        device = self.C_blocks.device
        f_per_hour = 2 * self.in_ch
        n_pix = self.im * self.jm
        n_rows = num_samples * n_pix

        mask_groups: list[tuple[torch.Tensor, list[int]]] = []
        mask_61_flat = self.wet_mask_61.double().reshape(self.out_ch, n_pix)
        for o_idx in range(self.out_ch):
            m_o = mask_61_flat[o_idx]
            placed = False
            for g_mask, g_chans in mask_groups:
                if torch.equal(m_o, g_mask):
                    g_chans.append(o_idx)
                    placed = True
                    break
            if not placed:
                mask_groups.append((m_o, [o_idx]))

        hour_feats: list[torch.Tensor] = []
        hour_lowrank: list[torch.Tensor] = []
        b_spatial_flat = self.spatial_basis.double().reshape(n_pix, self.low_rank)
        inv_sqrt_out = 1.0 / math.sqrt(float(self.out_ch))
        for k in range(self.num_hours):
            a_k = torch.stack(
                [sample[k].to(device) * self.wet_mask_60 for sample in action_deltas],
                dim=0,
            )
            la_k = self.apply_normalized_laplacian(a_k, self.wet_mask_60)
            feat_k = (
                torch.cat([a_k, la_k], dim=1)
                .permute(0, 2, 3, 1)
                .reshape(n_rows, f_per_hour)
                .double()
            )
            hour_feats.append(feat_k)

            a_mean = a_k.double().mean(dim=1).reshape(num_samples, n_pix)
            proj_in = (a_mean @ b_spatial_flat) / math.sqrt(float(self.in_ch))
            hour_lowrank.append(proj_in)

        for j in range(self.num_segments):
            max_k = j // 3
            active_hours = max_k + 1
            joint_feat = torch.cat(hour_feats[:active_hours], dim=1)
            total_f = joint_feat.shape[1]

            joint_lr_in = torch.cat(hour_lowrank[:active_hours], dim=1)
            q_dim = joint_lr_in.shape[1]
            m_dim = self.low_rank * q_dim
            psi_full = (
                inv_sqrt_out
                * b_spatial_flat.unsqueeze(0).unsqueeze(-1)
                * joint_lr_in.unsqueeze(1).unsqueeze(2)
            ).reshape(n_rows, m_dim)

            y_j = torch.stack(
                [
                    segment_deltas[s_idx][j].to(device) * self.wet_mask_61
                    for s_idx in range(num_samples)
                ],
                dim=0,
            ).double()
            target_flat = y_j.permute(0, 2, 3, 1).reshape(n_rows, self.out_ch)

            base_cd_diag = (joint_feat.square().sum(dim=0).mean()).clamp_min(1e-12)
            lam_cd = float(ridge) * base_cd_diag
            eye_cd = torch.eye(total_f, device=device, dtype=torch.float64)

            h_mm_raw = torch.zeros((m_dim, m_dim), device=device, dtype=torch.float64)
            schur_sub = torch.zeros((m_dim, m_dim), device=device, dtype=torch.float64)
            rhs_m_raw = torch.zeros(m_dim, device=device, dtype=torch.float64)
            rhs_m_sub = torch.zeros(m_dim, device=device, dtype=torch.float64)
            group_solves: list[tuple[list[int], torch.Tensor, torch.Tensor]] = []

            for g_mask_1d, g_chans in mask_groups:
                mu_col = g_mask_1d.unsqueeze(0).expand(num_samples, n_pix).reshape(n_rows, 1)
                phi_g = joint_feat * mu_col
                psi_g = psi_full * mu_col
                y_g = target_flat[:, g_chans] * mu_col
                c_g_count = float(len(g_chans))

                gram_cd_g = (phi_g.T @ phi_g) + lam_cd * eye_cd
                cross_g = phi_g.T @ psi_g
                rhs_cd_g = phi_g.T @ y_g

                solved_g = torch.linalg.solve(gram_cd_g, torch.cat([cross_g, rhs_cd_g], dim=1))
                a_g = solved_g[:, :m_dim]
                b_g = solved_g[:, m_dim:]

                h_mm_raw = h_mm_raw + c_g_count * (psi_g.T @ psi_g)
                schur_sub = schur_sub + c_g_count * (cross_g.T @ a_g)
                rhs_m_raw = rhs_m_raw + (psi_g.T @ y_g.sum(dim=1))
                rhs_m_sub = rhs_m_sub + (cross_g.T @ b_g.sum(dim=1))
                group_solves.append((g_chans, a_g, b_g))

            lam_m = float(ridge) * h_mm_raw.diag().mean().clamp_min(1e-12)
            eye_m = torch.eye(m_dim, device=device, dtype=torch.float64)
            schur_mat = (h_mm_raw - schur_sub) + lam_m * eye_m
            schur_rhs = rhs_m_raw - rhs_m_sub
            m_vec = torch.linalg.solve(schur_mat, schur_rhs)

            sol_cd_all = torch.zeros((self.out_ch, total_f), device=device, dtype=torch.float64)
            for g_chans, a_g, b_g in group_solves:
                theta_g_t = b_g - (a_g @ m_vec).unsqueeze(1)
                sol_cd_all[g_chans, :] = theta_g_t.T

            sol_cd_f = sol_cd_all.float()
            sol_m_f = m_vec.reshape(self.low_rank, q_dim).float()

            for k in range(active_hours):
                idx = self.pair_to_idx[(j, k)]
                block_cd = sol_cd_f[:, k * f_per_hour : (k + 1) * f_per_hour]
                new_c = block_cd[:, : self.in_ch]
                new_d = block_cd[:, self.in_ch :]
                new_m = sol_m_f[:, k * self.low_rank : (k + 1) * self.low_rank]
                if ema_decay > 0.0 and float(self.C_blocks[idx].abs().max().item()) > 0.0:
                    self.C_blocks[idx].copy_(
                        ema_decay * self.C_blocks[idx] + (1.0 - ema_decay) * new_c
                    )
                    self.D_blocks[idx].copy_(
                        ema_decay * self.D_blocks[idx] + (1.0 - ema_decay) * new_d
                    )
                    self.M_blocks[idx].copy_(
                        ema_decay * self.M_blocks[idx] + (1.0 - ema_decay) * new_m
                    )
                else:
                    self.C_blocks[idx].copy_(new_c)
                    self.D_blocks[idx].copy_(new_d)
                    self.M_blocks[idx].copy_(new_m)

        b_unc = self.block_norm_matrix(weights_61=weights_61, action_weights_60=action_weights_60)
        unconstrained_max_block = float(b_unc.max().item()) if b_unc.numel() > 0 else 0.0
        unconstrained_traj_bound = float(torch.linalg.matrix_norm(b_unc, ord=2).item())
        unconstrained_metrics = self.evaluate_probes(
            action_deltas, segment_deltas, weights_61=weights_61
        )
        bounds = self.enforce_gain_bounds(
            weights_61=weights_61, action_weights_60=action_weights_60
        )
        train_metrics = self.evaluate_probes(action_deltas, segment_deltas, weights_61=weights_61)
        gain_scale = (
            bounds["trajectory_norm_bound"] / unconstrained_traj_bound
            if unconstrained_traj_bound > 1e-12
            else 1.0
        )
        return {
            "max_block_norm": bounds["max_block_norm"],
            "unconstrained_max_block_norm": unconstrained_max_block,
            "trajectory_norm_bound": bounds["trajectory_norm_bound"],
            "unconstrained_trajectory_norm_bound": unconstrained_traj_bound,
            "gain_scale_applied": gain_scale,
            "unconstrained_fit_rel_err": unconstrained_metrics["relative_error"],
            "fit_rel_err": train_metrics["relative_error"],
            "fit_cosine": train_metrics["cosine"],
            "fit_r2": train_metrics["r2"],
            "fit_worst_probe_rel_err": train_metrics["worst_probe_rel_err"],
            "m_blocks_fro_norm": float(self.M_blocks.norm().item()),
        }

    @torch.no_grad()
    def evaluate_probes(
        self,
        action_deltas: list[list[torch.Tensor]],
        segment_deltas: list[list[torch.Tensor]],
        weights_61: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        device = self.C_blocks.device
        m61 = self.wet_mask_61.double()
        w = (
            (weights_61.to(device).double() * m61)
            if weights_61 is not None
            else self.default_wo_61.double()
        )
        w = w * (m61.sum().clamp_min(1.0) / w.sum().clamp_min(1e-20))
        sq_err = 0.0
        sq_tot = 0.0
        sq_pred = 0.0
        dot_sum = 0.0
        y_sum = 0.0
        w_sum = 0.0
        per_probe_rel: list[float] = []
        min_probe_sig_sq = float("inf") if len(action_deltas) > 0 else 0.0

        for s_idx in range(len(action_deltas)):
            preds = self.forward_trajectory([a.to(device) for a in action_deltas[s_idx]])
            p_err = 0.0
            p_tot = 0.0
            for j in range(self.num_segments):
                p_j = preds[j].double() * m61
                t_j = segment_deltas[s_idx][j].to(device).double() * m61
                diff = p_j - t_j
                e_j = float((diff.square() * w).sum().item())
                s_j = float((t_j.square() * w).sum().item())
                p_err += e_j
                p_tot += s_j
                sq_err += e_j
                sq_tot += s_j
                sq_pred += float((p_j.square() * w).sum().item())
                dot_sum += float((p_j * t_j * w).sum().item())
                y_sum += float((t_j * w).sum().item())
                w_sum += float(w.sum().item())
            min_probe_sig_sq = min(min_probe_sig_sq, p_tot)
            if p_tot <= 1e-20:
                per_probe_rel.append(float("inf"))
            else:
                per_probe_rel.append(math.sqrt(p_err / p_tot))

        if sq_tot <= 1e-20:
            rel_err = float("inf")
            cosine = 0.0
            r2 = -float("inf")
        else:
            rel_err = math.sqrt(sq_err / sq_tot)
            cosine = dot_sum / math.sqrt(max(sq_tot * sq_pred, 1e-30))
            centered_var = max(sq_tot - (y_sum * y_sum) / max(w_sum, 1e-30), 1e-30)
            r2 = 1.0 - sq_err / centered_var
        worst_probe = max(per_probe_rel) if per_probe_rel else float("inf")
        return {
            "relative_error": rel_err,
            "cosine": cosine,
            "r2": r2,
            "signal_squared": sq_tot,
            "min_probe_signal_squared": min_probe_sig_sq,
            "worst_probe_rel_err": worst_probe,
            "per_probe_rel_errors": per_probe_rel,
        }

    @torch.no_grad()
    def validate_heldout_probes(
        self,
        action_deltas: list[list[torch.Tensor]],
        segment_deltas: list[list[torch.Tensor]],
        weights_61: torch.Tensor | None = None,
        action_weights_60: torch.Tensor | None = None,
        relative_limit: float = 0.50,
        cosine_limit: float = 0.80,
        r2_limit: float = 0.50,
        per_probe_limit: float = 1.00,
    ) -> dict[str, Any]:
        metrics = self.evaluate_probes(action_deltas, segment_deltas, weights_61=weights_61)
        b_mat = self.block_norm_matrix(weights_61=weights_61, action_weights_60=action_weights_60)
        max_block = float(b_mat.max().item()) if b_mat.numel() > 0 else 0.0
        traj_bound = float(torch.linalg.matrix_norm(b_mat, ord=2).item())
        passed = bool(
            len(action_deltas) > 0
            and math.isfinite(metrics["relative_error"])
            and math.isfinite(metrics["worst_probe_rel_err"])
            and metrics["signal_squared"] > 1e-20
            and metrics["min_probe_signal_squared"] > 1e-20
            and metrics["relative_error"] <= relative_limit
            and metrics["cosine"] >= cosine_limit
            and metrics["r2"] >= r2_limit
            and metrics["worst_probe_rel_err"] <= per_probe_limit
            and traj_bound <= self.gain_bound * (1.0 + 1e-5)
        )
        return {
            "passed": passed,
            "metrics": metrics,
            "max_block_norm": max_block,
            "trajectory_norm_bound": traj_bound,
            "gain_bound": self.gain_bound,
            "relative_limit": relative_limit,
            "cosine_limit": cosine_limit,
            "r2_limit": r2_limit,
            "per_probe_limit": per_probe_limit,
            "heldout_samples": len(action_deltas),
        }

    @torch.no_grad()
    def solve_adjoint_cg(
        self,
        residual_segments: list[torch.Tensor],
        weights_61: torch.Tensor | None = None,
        action_weights_60: torch.Tensor | None = None,
        project_fn: Any | None = None,
        project_transpose_fn: Any | None = None,
        iterations: int = 32,
        damping: float = 0.02,
        trust_rms: float = 0.15,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        """Minimize 0.5 * ||r^I + K P u||_{W_o}^2 + 0.5 * lambda_u * ||u||_{W_q}^2 via float64 PCG."""
        device = self.C_blocks.device
        out_dtype = residual_segments[0].dtype
        mask60_d = self.wet_mask_60.double()
        mask61_d = self.wet_mask_61.double()
        wet_60_count = mask60_d.sum().clamp_min(1.0)
        wet_61_count = mask61_d.sum().clamp_min(1.0)

        w_o = (
            (weights_61.to(device=device, dtype=torch.float64) * mask61_d)
            if weights_61 is not None
            else self.default_wo_61.double()
        )
        w_o = w_o * (wet_61_count / w_o.sum().clamp_min(1e-20))

        w_q = (
            (action_weights_60.to(device=device, dtype=torch.float64) * mask60_d)
            if action_weights_60 is not None
            else self.default_wq_60.double()
        )
        w_q = w_q * (wet_60_count / w_q.sum().clamp_min(1e-20))
        inv_w_q = torch.where(mask60_d > 0.5, 1.0 / w_q.clamp_min(1e-20), torch.zeros_like(w_q))
        lam_u = max(float(damping), 1e-12)

        adj_proj_fn = project_transpose_fn if project_transpose_fn is not None else project_fn

        def proj_forward(acts: list[torch.Tensor]) -> list[torch.Tensor]:
            if project_fn is None:
                return [a * mask60_d for a in acts]
            return [project_fn(a) * mask60_d for a in acts]

        def proj_transpose(grads: list[torch.Tensor]) -> list[torch.Tensor]:
            if adj_proj_fn is None:
                return [g * mask60_d for g in grads]
            return [adj_proj_fn(g) * mask60_d for g in grads]

        def forward_op_d(acts: list[torch.Tensor]) -> list[torch.Tensor]:
            p_acts = proj_forward(acts)
            outputs: list[torch.Tensor] = []
            for j in range(self.num_segments):
                max_k = j // 3
                acc = torch.zeros(
                    (self.out_ch, self.im, self.jm), device=device, dtype=torch.float64
                )
                for k in range(max_k + 1):
                    idx = self.pair_to_idx[(j, k)]
                    c_d = self.C_blocks[idx].double()
                    d_d = self.D_blocks[idx].double()
                    m_d = self.M_blocks[idx].double()
                    a_wet = p_acts[k] * mask60_d
                    lap_a = self.apply_normalized_laplacian(a_wet, mask60_d)
                    out_c = torch.einsum("oi,ixy->oxy", c_d, a_wet)
                    out_d = torch.einsum("oi,ixy->oxy", d_d, lap_a)
                    b_sp = self.spatial_basis.double()
                    a_mean = a_wet.mean(dim=0)
                    proj_in = torch.einsum("xy,xyr->r", a_mean, b_sp) / math.sqrt(float(self.in_ch))
                    proj_out = torch.mv(m_d, proj_in) / math.sqrt(float(self.out_ch))
                    spatial_out = torch.einsum("xyr,r->xy", b_sp, proj_out)
                    acc = (
                        acc
                        + (out_c + out_d + spatial_out.unsqueeze(0).expand(self.out_ch, -1, -1))
                        * mask61_d
                    )
                outputs.append(acc)
            return outputs

        def adjoint_op_d(segs: list[torch.Tensor]) -> list[torch.Tensor]:
            weighted_segs = [s * w_o for s in segs]
            grads: list[torch.Tensor] = []
            b_sp = self.spatial_basis.double()
            for k in range(self.num_hours):
                acc = torch.zeros(
                    (self.in_ch, self.im, self.jm), device=device, dtype=torch.float64
                )
                for j in range(3 * k, self.num_segments):
                    idx = self.pair_to_idx[(j, k)]
                    c_d = self.C_blocks[idx].double()
                    d_d = self.D_blocks[idx].double()
                    m_d = self.M_blocks[idx].double()
                    q_wet = weighted_segs[j] * mask61_d
                    adj_c = torch.einsum("oi,oxy->ixy", c_d, q_wet) * mask60_d
                    dt_q = torch.einsum("oi,oxy->ixy", d_d, q_wet) * mask60_d
                    adj_d = self.apply_normalized_laplacian(dt_q, mask60_d)
                    q_sum = q_wet.sum(dim=0)
                    q_proj = torch.einsum("xy,xyr->r", q_sum, b_sp) / math.sqrt(float(self.out_ch))
                    in_proj = torch.mv(m_d.T, q_proj) / math.sqrt(float(self.in_ch))
                    spatial_in = torch.einsum("xyr,r->xy", b_sp, in_proj) / float(self.in_ch)
                    adj_p = spatial_in.unsqueeze(0).expand(self.in_ch, -1, -1) * mask60_d
                    acc = acc + adj_c + adj_d + adj_p
                grads.append(acc)
            return proj_transpose(grads)

        n_out_dof = float(self.num_segments) * float(wet_61_count.item())
        n_in_dof = float(self.num_hours) * float(wet_60_count.item())
        lam_eff = lam_u * (n_out_dof / max(n_in_dof, 1.0))

        def dot_euc(a_list: list[torch.Tensor], b_list: list[torch.Tensor]) -> torch.Tensor:
            return sum((a * b * mask60_d).sum() for a, b in zip(a_list, b_list, strict=True))

        def data_misfit_half(segs: list[torch.Tensor]) -> float:
            total = 0.5 * sum((s.square() * w_o).sum() for s in segs) / n_out_dof
            return float(total.item())

        def action_reg_half(acts: list[torch.Tensor]) -> float:
            total = 0.5 * lam_u * sum((a.square() * w_q).sum() for a in acts) / n_in_dof
            return float(total.item())

        res_clean = [r.to(device=device, dtype=torch.float64) * mask61_d for r in residual_segments]
        rhs = [-g for g in adjoint_op_d(res_clean)]
        x = [torch.zeros_like(g) for g in rhs]
        r = [g.clone() for g in rhs]
        z = [rk * inv_w_q for rk in r]
        p = [zk.clone() for zk in z]
        rz = dot_euc(r, z)
        initial_rr = float(dot_euc(r, r).item())
        completed = 0

        for completed in range(1, iterations + 1):
            if float(rz.item()) <= 1e-28:
                completed -= 1
                break
            fp = forward_op_d(p)
            ap = [ag + lam_eff * (w_q * pg) for ag, pg in zip(adjoint_op_d(fp), p, strict=True)]
            denom = dot_euc(p, ap)
            if not torch.isfinite(denom) or float(denom.item()) <= 0.0:
                break
            alpha = float((rz / denom).item())
            x = [xk + alpha * pk for xk, pk in zip(x, p, strict=True)]
            r = [rk - alpha * apk for rk, apk in zip(r, ap, strict=True)]
            final_rr = float(dot_euc(r, r).item())
            if final_rr < max(initial_rr * 1e-20, 1e-28):
                break
            z = [rk * inv_w_q for rk in r]
            rz_new = dot_euc(r, z)
            beta = float((rz_new / rz).item())
            p = [zk + beta * pk for zk, pk in zip(z, p, strict=True)]
            rz = rz_new

        final_rr = float(dot_euc(r, r).item())
        x_proj = proj_forward(x)
        rms = math.sqrt(
            float(sum((xk.square() * w_q).sum() for xk in x_proj).item())
            / float(self.num_hours * wet_60_count.item())
        )
        scale = min(1.0, trust_rms / max(rms, 1e-20))
        x_scaled = [xk * scale for xk in x_proj]
        u_scaled = [xk * scale for xk in x]

        misfit_before = data_misfit_half(res_clean)
        pred_segs = [rk + dk for rk, dk in zip(res_clean, forward_op_d(u_scaled), strict=True)]
        misfit_after = data_misfit_half(pred_segs)
        reg_after = action_reg_half(u_scaled)
        obj_before = misfit_before
        obj_after = misfit_after + reg_after
        usable = bool(
            obj_after < obj_before and misfit_after < misfit_before and rms * scale > 1e-12
        )
        if not usable:
            x_out = [
                torch.zeros((self.in_ch, self.im, self.jm), device=device, dtype=out_dtype)
                for _ in range(self.num_hours)
            ]
            misfit_after = misfit_before
            reg_after = 0.0
            obj_after = obj_before
        else:
            x_out = [xk.to(dtype=out_dtype) for xk in x_scaled]

        return x_out, {
            "initial_normal_residual_squared": initial_rr,
            "final_normal_residual_squared": final_rr,
            "action_rms_before_trust": rms,
            "trust_scale": scale,
            "trust_rms": trust_rms,
            "lambda_u": lam_u,
            "damping": lam_u,
            "CG_iterations": completed,
            "data_misfit_before": misfit_before,
            "data_misfit_after": misfit_after,
            "action_reg_term": reg_after,
            "total_objective_before": obj_before,
            "total_objective_after": obj_after,
            "center_model_mse": 2.0 * misfit_before,
            "candidate_model_mse": 2.0 * misfit_after,
            "usable": usable,
        }


def apply_response_trajectory_differentiable(
    delta_actions: torch.Tensor,
    c_blocks: torch.Tensor,
    d_blocks: torch.Tensor,
    m_blocks: torch.Tensor,
    spatial_basis: torch.Tensor,
    wet_mask_60: torch.Tensor,
    wet_mask_61: torch.Tensor,
    pair_to_idx: dict[tuple[int, int], int],
) -> torch.Tensor:
    """Differentiably map `delta_actions` `(4, 60, im, jm)` to `(12, 61, im, jm)` segment shifts."""
    in_ch = int(delta_actions.shape[1])
    out_ch = int(c_blocks.shape[1])
    im = int(delta_actions.shape[2])
    jm = int(delta_actions.shape[3])

    a_wet_all = delta_actions * wet_mask_60.unsqueeze(0)
    lap_all = apply_wet_normalized_laplacian(a_wet_all, wet_mask_60)
    a_mean_all = a_wet_all.mean(dim=1)
    proj_in_all = torch.einsum("hxy,xyr->hr", a_mean_all, spatial_basis) / math.sqrt(float(in_ch))
    inv_sqrt_out = 1.0 / math.sqrt(float(out_ch))

    outputs: list[torch.Tensor] = []
    for j in range(12):
        max_k = j // 3
        acc = torch.zeros((out_ch, im, jm), device=delta_actions.device, dtype=delta_actions.dtype)
        for k in range(max_k + 1):
            idx = pair_to_idx[(j, k)]
            out_c = torch.einsum("oi,ixy->oxy", c_blocks[idx], a_wet_all[k])
            out_d = torch.einsum("oi,ixy->oxy", d_blocks[idx], lap_all[k])
            proj_out = torch.mv(m_blocks[idx], proj_in_all[k]) * inv_sqrt_out
            spatial_out = torch.einsum("xyr,r->xy", spatial_basis, proj_out)
            acc = acc + out_c + out_d + spatial_out.unsqueeze(0).expand(out_ch, -1, -1)
        outputs.append(acc * wet_mask_61)
    return torch.stack(outputs, dim=0)


OBS_CAUSAL_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
    (3, 2),
)


class CausalObservationResponseOperator3D(nn.Module):
    """Non-neural causal linear response `K_obs` mapping action shifts `delta_a_l` (`l < k`) to `delta_x_k`."""

    def __init__(
        self,
        wet_mask_60: torch.Tensor,
        obs_wet_mask_161: torch.Tensor,
        im: int = 90,
        jm: int = 82,
        gain_bound: float = 6.0,
    ) -> None:
        super().__init__()
        self.in_ch = int(wet_mask_60.shape[0])
        self.obs_ch = int(obs_wet_mask_161.shape[0])
        self.im = int(im)
        self.jm = int(jm)
        self.gain_bound = float(gain_bound)
        self.pairs = OBS_CAUSAL_PAIRS
        self.pair_to_idx: dict[tuple[int, int], int] = {p: i for i, p in enumerate(self.pairs)}

        self.register_buffer("wet_mask_60", wet_mask_60.float().clone())
        self.register_buffer("obs_wet_mask_161", obs_wet_mask_161.float().clone())
        self.register_buffer("C_obs", torch.zeros(len(self.pairs), self.obs_ch, self.in_ch))
        self.register_buffer("D_obs", torch.zeros(len(self.pairs), self.obs_ch, self.in_ch))

    def forward_obs_step(self, k_hour: int, prev_delta_actions: list[torch.Tensor]) -> torch.Tensor:
        if k_hour <= 0:
            return torch.zeros(
                (self.obs_ch, self.im, self.jm),
                device=self.C_obs.device,
                dtype=prev_delta_actions[0].dtype,
            )
        acc = torch.zeros(
            (self.obs_ch, self.im, self.jm),
            device=prev_delta_actions[0].device,
            dtype=prev_delta_actions[0].dtype,
        )
        for l_hr in range(k_hour):
            idx = self.pair_to_idx[(k_hour, l_hr)]
            a_wet = prev_delta_actions[l_hr] * self.wet_mask_60
            lap_a = apply_wet_normalized_laplacian(a_wet, self.wet_mask_60)
            out_c = torch.einsum("oi,ixy->oxy", self.C_obs[idx], a_wet)
            out_d = torch.einsum("oi,ixy->oxy", self.D_obs[idx], lap_a)
            acc = acc + out_c + out_d
        return acc * self.obs_wet_mask_161

    @torch.no_grad()
    def fit_from_probes(
        self,
        action_deltas: list[list[torch.Tensor]],
        obs_deltas_123: list[list[torch.Tensor]],
        ridge: float = 5e-2,
    ) -> dict[str, float]:
        num_samples = len(action_deltas)
        if num_samples == 0:
            return {
                "max_block_norm": 0.0,
                "gain_scale_applied": 1.0,
                "fit_rel_err": 1.0,
            }

        device = self.C_obs.device
        f_per_hour = 2 * self.in_ch
        n_pix = self.im * self.jm
        n_rows = num_samples * n_pix

        mask_groups: list[tuple[torch.Tensor, list[int]]] = []
        mask_161_flat = self.obs_wet_mask_161.double().reshape(self.obs_ch, n_pix)
        for o_idx in range(self.obs_ch):
            m_o = mask_161_flat[o_idx]
            placed = False
            for g_mask, g_chans in mask_groups:
                if torch.equal(m_o, g_mask):
                    g_chans.append(o_idx)
                    placed = True
                    break
            if not placed:
                mask_groups.append((m_o, [o_idx]))

        hour_feats: list[torch.Tensor] = []
        for l_hr in range(3):
            a_l = torch.stack(
                [sample[l_hr].to(device) * self.wet_mask_60 for sample in action_deltas],
                dim=0,
            )
            la_l = apply_wet_normalized_laplacian(a_l, self.wet_mask_60)
            feat_l = (
                torch.cat([a_l, la_l], dim=1)
                .permute(0, 2, 3, 1)
                .reshape(n_rows, f_per_hour)
                .double()
            )
            hour_feats.append(feat_l)

        for k_hour in (1, 2, 3):
            joint_feat = torch.cat(hour_feats[:k_hour], dim=1)
            total_f = joint_feat.shape[1]
            y_k = torch.stack(
                [
                    obs_deltas_123[s_idx][k_hour - 1].to(device) * self.obs_wet_mask_161
                    for s_idx in range(num_samples)
                ],
                dim=0,
            ).double()
            target_flat = y_k.permute(0, 2, 3, 1).reshape(n_rows, self.obs_ch)

            base_diag = (joint_feat.square().sum(dim=0).mean()).clamp_min(1e-12)
            lam = float(ridge) * base_diag
            eye_f = torch.eye(total_f, device=device, dtype=torch.float64)
            w_joint = torch.zeros((total_f, self.obs_ch), device=device, dtype=torch.float64)

            for g_mask_1d, g_chans in mask_groups:
                mu_col = g_mask_1d.unsqueeze(0).expand(num_samples, n_pix).reshape(n_rows, 1)
                phi_g = joint_feat * mu_col
                y_g = target_flat[:, g_chans] * mu_col
                gram_g = (phi_g.T @ phi_g) + lam * eye_f
                rhs_g = phi_g.T @ y_g
                w_joint[:, g_chans] = torch.linalg.solve(gram_g, rhs_g)

            for l_hr in range(k_hour):
                idx = self.pair_to_idx[(k_hour, l_hr)]
                sl = w_joint[l_hr * f_per_hour : (l_hr + 1) * f_per_hour, :]
                self.C_obs[idx].copy_(sl[: self.in_ch, :].T.float())
                self.D_obs[idx].copy_(sl[self.in_ch :, :].T.float())

        max_norm = 0.0
        scale_factor = math.sqrt(float(self.in_ch) / float(self.obs_ch))
        for idx in range(len(self.pairs)):
            bn = scale_factor * (
                float(torch.linalg.matrix_norm(self.C_obs[idx].double(), ord=2).item())
                + float(torch.linalg.matrix_norm(self.D_obs[idx].double(), ord=2).item())
            )
            max_norm = max(max_norm, bn)

        gain_scale = 1.0
        if max_norm > self.gain_bound and max_norm > 0.0:
            gain_scale = self.gain_bound / max_norm
            self.C_obs.mul_(gain_scale)
            self.D_obs.mul_(gain_scale)

        err_sq = 0.0
        ref_sq = 0.0
        for s_idx in range(num_samples):
            acts_s = [t.to(device) for t in action_deltas[s_idx]]
            for k_hour in (1, 2, 3):
                pred_x = self.forward_obs_step(k_hour, acts_s)
                targ_x = obs_deltas_123[s_idx][k_hour - 1].to(device) * self.obs_wet_mask_161
                err_sq += float((pred_x - targ_x).double().square().sum().item())
                ref_sq += float(targ_x.double().square().sum().item())

        return {
            "unconstrained_max_block_norm": max_norm,
            "max_block_norm": max_norm * gain_scale,
            "gain_scale_applied": gain_scale,
            "fit_rel_err": math.sqrt(err_sq / max(ref_sq, 1e-30)),
        }

    @torch.no_grad()
    def validate_heldout_probes(
        self,
        val_action_deltas: list[list[torch.Tensor]],
        val_obs_deltas_123: list[list[torch.Tensor]],
        cosine_min: float = 0.50,
        r2_min: float = 0.15,
    ) -> dict[str, Any]:
        device = self.C_obs.device
        err_sq = 0.0
        ref_sq = 0.0
        pred_sq = 0.0
        dot_sum = 0.0
        for s_idx in range(len(val_action_deltas)):
            acts_s = [t.to(device) for t in val_action_deltas[s_idx]]
            for k_hour in (1, 2, 3):
                pred_x = self.forward_obs_step(k_hour, acts_s).double()
                targ_x = (
                    val_obs_deltas_123[s_idx][k_hour - 1].to(device) * self.obs_wet_mask_161
                ).double()
                err_sq += float((pred_x - targ_x).square().sum().item())
                ref_sq += float(targ_x.square().sum().item())
                pred_sq += float(pred_x.square().sum().item())
                dot_sum += float((pred_x * targ_x).sum().item())

        rel_err = math.sqrt(err_sq / max(ref_sq, 1e-30))
        cosine = dot_sum / max(math.sqrt(pred_sq * ref_sq), 1e-30)
        r2 = 1.0 - (err_sq / max(ref_sq, 1e-30))
        passed = bool(
            math.isfinite(cosine) and math.isfinite(r2) and cosine >= cosine_min and r2 >= r2_min
        )
        gamma_obs = float(min(max(r2, 0.0), 1.0)) if passed else 0.0
        return {
            "passed": passed,
            "gamma_obs": gamma_obs,
            "relative_error": rel_err,
            "cosine": cosine,
            "r2": r2,
        }


def apply_obs_step_differentiable(
    k_hour: int,
    prev_delta_actions: list[torch.Tensor],
    c_obs: torch.Tensor,
    d_obs: torch.Tensor,
    wet_mask_60: torch.Tensor,
    obs_wet_mask_161: torch.Tensor,
    pair_to_idx: dict[tuple[int, int], int],
) -> torch.Tensor:
    """Differentiably map `prev_delta_actions[:k_hour]` to `delta_x_k` (`161, im, jm`)."""
    im = int(prev_delta_actions[0].shape[1])
    jm = int(prev_delta_actions[0].shape[2])
    obs_ch = int(c_obs.shape[1])
    acc = torch.zeros(
        (obs_ch, im, jm),
        device=prev_delta_actions[0].device,
        dtype=prev_delta_actions[0].dtype,
    )
    for l_hr in range(k_hour):
        idx = pair_to_idx[(k_hour, l_hr)]
        a_wet = prev_delta_actions[l_hr] * wet_mask_60
        lap_a = apply_wet_normalized_laplacian(a_wet, wet_mask_60)
        acc = (
            acc
            + torch.einsum("oi,ixy->oxy", c_obs[idx], a_wet)
            + torch.einsum("oi,ixy->oxy", d_obs[idx], lap_a)
        )
    return acc * obs_wet_mask_161


__all__ = [
    "CausalBlockResponseOperator3D",
    "CausalObservationResponseOperator3D",
    "HADAMARD_4X4",
    "OBS_CAUSAL_PAIRS",
    "apply_obs_step_differentiable",
    "apply_response_trajectory_differentiable",
    "apply_wet_normalized_laplacian",
]
