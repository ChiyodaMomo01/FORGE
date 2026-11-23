"""Stage II Hadamard probe collection, operator fitting, adjoint PCG inversion, and replay optimization."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import ForgePOM3DConfig
from ..distributed import ddp_all_gather_objects, ddp_mean_dict
from ..network import ForgePOMNet
from ..packing import pack_segments_to_tensor
from ..physics_env import BlackBoxProvider3D
from ..probes import (
    HADAMARD_4X4,
    build_projected_action_probe_direction,
    measure_action_jacobian_ranks,
)
from ..sampling import sample_epoch_windows
from .response_operator import (
    CausalBlockResponseOperator3D,
    CausalObservationResponseOperator3D,
    apply_obs_step_differentiable,
    apply_response_trajectory_differentiable,
)
from .stage1 import Stage1RoundResult


@dataclass
class Stage2RoundResult:
    chosen_alpha_s2: float
    stage2_accepted: bool
    stage2_source: str
    post_s2_sync: float
    stage2_best_cand_loss: float | None
    stage2_max_body_diff: float
    stage2_body_l2_diff: float
    curr_trust_rms: float
    curr_lambda_act: float
    fit_stats: dict[str, Any]
    heldout_gate: dict[str, Any]
    obs_gate: dict[str, Any]
    cg_info: dict[str, Any]
    window_rollouts_used: int


@torch.no_grad()
def solve_replay_head_ridge(
    cfg: ForgePOM3DConfig,
    env: BlackBoxProvider3D,
    model: ForgePOMNet,
    replay_buffer: deque[dict[str, Any]],
    wq_60: torch.Tensor,
    base_theta: torch.Tensor,
    current_round: int,
    device: torch.device,
) -> torch.Tensor:
    """Solve global DDP-averaged `W_q`-weighted readout head ridge regression on `base_theta` features."""
    if not replay_buffer:
        return base_theta.clone()
    was_tr = model.training
    model.eval()
    model.set_flat_params(base_theta)

    p_aug = int(cfg.hidden_channels) + 1
    head_groups = (
        (0, 15, wq_60[0]),
        (15, 30, wq_60[15]),
        (30, 60, wq_60[30]),
    )
    total_weight = sum(
        float(cfg.forge_stage2_replay_decay) ** max(current_round - int(ent["round"]), 0)
        for ent in replay_buffer
    ) * float(cfg.hours_per_window)
    total_weight = max(total_weight, 1e-8)

    cached_feats: list[tuple[float, torch.Tensor, torch.Tensor]] = []
    for ent in replay_buffer:
        w_ent = float(cfg.forge_stage2_replay_decay) ** max(current_round - int(ent["round"]), 0)
        x_seq = ent["x_nom"].to(device=device, dtype=torch.float32)
        a_star_seq = ent["a_target_proj"].to(device=device, dtype=torch.float32)
        for k_hr in range(cfg.hours_per_window):
            phi_k = model.extract_trunk_features(x_seq[k_hr])
            g_cur_proj = env.project_standardized_action(model.forward_from_features(phi_k))
            rem_da = (a_star_seq[k_hr] - g_cur_proj).detach()
            cached_feats.append((w_ent, phi_k.detach(), rem_da))

    delta_w_t = torch.zeros((p_aug, cfg.out_channels), device=device, dtype=torch.float64)
    ones_2d = torch.ones((1, cfg.im, cfg.jm), device=device, dtype=torch.float64)
    for ch_lo, ch_hi, w2d_grp in head_groups:
        w_sqrt_2d = w2d_grp.double().clamp_min(0.0).sqrt()
        grp_denom = max(float(w2d_grp.double().sum().item()) * total_weight, 1.0)
        gram_g = torch.zeros((p_aug, p_aug), device=device, dtype=torch.float64)
        rhs_g = torch.zeros((p_aug, ch_hi - ch_lo), device=device, dtype=torch.float64)
        for w_ent, phi_k, rem_da in cached_feats:
            phi_aug = (
                torch.cat([phi_k.double(), ones_2d], dim=0) * w_sqrt_2d.unsqueeze(0)
            ).reshape(p_aug, -1)
            da_grp = (rem_da[ch_lo:ch_hi].double() * w_sqrt_2d.unsqueeze(0)).reshape(
                ch_hi - ch_lo, -1
            )
            gram_g.add_((phi_aug @ phi_aug.T) * (w_ent / grp_denom))
            rhs_g.add_((phi_aug @ da_grp.T) * (w_ent / grp_denom))
        if dist.is_initialized():
            dist.all_reduce(gram_g, op=dist.ReduceOp.AVG)
            dist.all_reduce(rhs_g, op=dist.ReduceOp.AVG)
        lam_g = 3e-2 * gram_g.diag().mean().clamp_min(1e-8)
        delta_w_t[:, ch_lo:ch_hi] = torch.linalg.solve(
            gram_g + lam_g * torch.eye(p_aug, device=device, dtype=torch.float64),
            rhs_g,
        )

    w_head_cur = model.get_head_matrix()
    model.set_head_matrix(w_head_cur + delta_w_t.T.to(dtype=w_head_cur.dtype))
    out_theta = model.get_flat_params().clone()
    if was_tr:
        model.train()
    return out_theta


@torch.no_grad()
def _recapture_stage1_center_if_needed(
    cfg: ForgePOM3DConfig,
    env: BlackBoxProvider3D,
    model: ForgePOMNet,
    my_windows: list[tuple[int, int]],
    s1_res: Stage1RoundResult,
    world_size: int,
) -> tuple[
    list[torch.Tensor],
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
    list[float],
    int,
]:
    if s1_res.chosen_alpha == 0.0:
        return (
            s1_res.nom_obs_list,
            s1_res.nom_actions_raw_list,
            s1_res.nom_actions_proj_list,
            s1_res.nom_segments_list,
            s1_res.nom_losses_list,
            0,
        )

    s1_obs_list: list[torch.Tensor] = []
    s1_actions_raw_list: list[list[torch.Tensor]] = []
    s1_actions_proj_list: list[list[torch.Tensor]] = []
    s1_segments_list: list[list[torch.Tensor]] = []
    s1_losses_list: list[float] = []
    for m, d in my_windows:
        env.reset(m, d)
        xs_w: list[torch.Tensor] = []
        acts_raw_w: list[torch.Tensor] = []
        acts_proj_w: list[torch.Tensor] = []
        segs_w: list[dict[str, torch.Tensor]] = []
        for _k in range(cfg.hours_per_window):
            x_k = env.observe_F()
            g_k = model(x_k).detach()
            segs_w.extend(env.step_H(g_k))
            xs_w.append(x_k.detach())
            acts_raw_w.append(g_k)
            acts_proj_w.append(env.project_standardized_action(g_k).detach())
        lw_s1, _ = env.compute_12seg_loss_and_metrics(m, d, segs_w)
        s1_losses_list.append(float(lw_s1.item()))
        s1_obs_list.append(torch.stack(xs_w, dim=0))
        s1_actions_raw_list.append(acts_raw_w)
        s1_actions_proj_list.append(acts_proj_w)
        s1_segments_list.append(pack_segments_to_tensor(env, segs_w))
    return (
        s1_obs_list,
        s1_actions_raw_list,
        s1_actions_proj_list,
        s1_segments_list,
        s1_losses_list,
        len(my_windows) * world_size,
    )


@torch.no_grad()
def collect_stage2_hadamard_probes(
    cfg: ForgePOM3DConfig,
    env: BlackBoxProvider3D,
    my_windows: list[tuple[int, int]],
    s1_obs_list: list[torch.Tensor],
    s1_actions_raw_list: list[list[torch.Tensor]],
    s1_actions_proj_list: list[list[torch.Tensor]],
    s1_segments_list: list[list[torch.Tensor]],
    mask_60: torch.Tensor,
    h_norm: torch.Tensor,
    round_idx: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[
    list[tuple[int, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]],
    list[tuple[int, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]],
    int,
]:
    """Collect 64 Hadamard fit probes and 16 unseen held-out probes across DDP ranks."""
    m0, d0 = my_windows[0]
    probe_amp = float(cfg.forge_stage2_probe_amp)
    local_fit_records: list[
        tuple[int, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
    ] = []
    local_val_records: list[
        tuple[int, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
    ] = []

    for pair_slot in range(4):
        global_pair_id = rank * 4 + pair_slot
        dir_seed = cfg.seed + round_idx * 10007 + global_pair_id * 17
        d_spatial = build_projected_action_probe_direction(env, mask_60, dir_seed, device)
        t_code = h_norm[global_pair_id % 4]
        amp_mag = probe_amp if (pair_slot < 3) else (0.5 * probe_amp)
        for sgn_idx, sgn in enumerate((1.0, -1.0)):
            env.reset(m0, d0)
            segs_p: list[dict[str, torch.Tensor]] = []
            act_deltas_p: list[torch.Tensor] = []
            obs_deltas_123_p: list[torch.Tensor] = []
            for k in range(cfg.hours_per_window):
                x_p_k = env.observe_F()
                if k >= 1:
                    obs_deltas_123_p.append((x_p_k - s1_obs_list[0][k]).cpu())
                g_nom = s1_actions_raw_list[0][k]
                g_pert = g_nom + float(sgn * amp_mag * float(t_code[k].item())) * d_spatial
                segs_p.extend(env.step_H(g_pert))
                exec_pert = env.project_standardized_action(g_pert)
                act_deltas_p.append((exec_pert - s1_actions_proj_list[0][k]).cpu())
            packed_p = pack_segments_to_tensor(env, segs_p)
            seg_deltas_p = [(packed_p[j] - s1_segments_list[0][j]).cpu() for j in range(12)]
            local_fit_records.append(
                (
                    global_pair_id * 2 + sgn_idx,
                    act_deltas_p,
                    seg_deltas_p,
                    obs_deltas_123_p,
                )
            )

    # Held-out 1: unseen single-hour impulse
    imp_seed = cfg.seed + round_idx * 20011 + rank * 29 + 500000
    d_imp = build_projected_action_probe_direction(env, mask_60, imp_seed, device)
    imp_hour = rank % cfg.hours_per_window
    env.reset(m0, d0)
    segs_imp: list[dict[str, torch.Tensor]] = []
    act_deltas_imp: list[torch.Tensor] = []
    obs_deltas_123_imp: list[torch.Tensor] = []
    for k in range(cfg.hours_per_window):
        x_imp_k = env.observe_F()
        if k >= 1:
            obs_deltas_123_imp.append((x_imp_k - s1_obs_list[0][k]).cpu())
        g_nom = s1_actions_raw_list[0][k]
        c_k = probe_amp if k == imp_hour else 0.0
        g_pert = g_nom + c_k * d_imp
        segs_imp.extend(env.step_H(g_pert))
        exec_pert = env.project_standardized_action(g_pert)
        act_deltas_imp.append((exec_pert - s1_actions_proj_list[0][k]).cpu())
    packed_imp = pack_segments_to_tensor(env, segs_imp)
    seg_deltas_imp = [(packed_imp[j] - s1_segments_list[0][j]).cpu() for j in range(12)]
    local_val_records.append((rank * 2, act_deltas_imp, seg_deltas_imp, obs_deltas_123_imp))

    # Held-out 2: unseen multi-hour orthogonal combination
    cmb_seed = cfg.seed + round_idx * 30013 + rank * 37 + 700000
    d_cmb = build_projected_action_probe_direction(env, mask_60, cmb_seed, device)
    t_cmb = (h_norm[(rank + 1) % 4] + h_norm[(rank + 2) % 4]) / math.sqrt(2.0)
    sgn_cmb = -1.0 if (rank % 2 == 1) else 0.75
    env.reset(m0, d0)
    segs_cmb: list[dict[str, torch.Tensor]] = []
    act_deltas_cmb: list[torch.Tensor] = []
    obs_deltas_123_cmb: list[torch.Tensor] = []
    for k in range(cfg.hours_per_window):
        x_cmb_k = env.observe_F()
        if k >= 1:
            obs_deltas_123_cmb.append((x_cmb_k - s1_obs_list[0][k]).cpu())
        g_nom = s1_actions_raw_list[0][k]
        g_pert = g_nom + float(sgn_cmb * probe_amp * float(t_cmb[k].item())) * d_cmb
        segs_cmb.extend(env.step_H(g_pert))
        exec_pert = env.project_standardized_action(g_pert)
        act_deltas_cmb.append((exec_pert - s1_actions_proj_list[0][k]).cpu())
    packed_cmb = pack_segments_to_tensor(env, segs_cmb)
    seg_deltas_cmb = [(packed_cmb[j] - s1_segments_list[0][j]).cpu() for j in range(12)]
    local_val_records.append((rank * 2 + 1, act_deltas_cmb, seg_deltas_cmb, obs_deltas_123_cmb))

    all_fit = sorted(
        [it for sub in ddp_all_gather_objects(local_fit_records, world_size) for it in sub],
        key=lambda x: int(x[0]),
    )
    all_val = sorted(
        [it for sub in ddp_all_gather_objects(local_val_records, world_size) for it in sub],
        key=lambda x: int(x[0]),
    )
    return all_fit, all_val, 10 * world_size


def optimize_stage2_replay_network(
    cfg: ForgePOM3DConfig,
    env: BlackBoxProvider3D,
    model: ForgePOMNet,
    resp_op: CausalBlockResponseOperator3D,
    obs_resp_op: CausalObservationResponseOperator3D,
    replay_buffer: deque[dict[str, Any]],
    mask_60: torch.Tensor,
    mask_61: torch.Tensor,
    obs_mask_161: torch.Tensor,
    wq_60: torch.Tensor,
    wo_61: torch.Tensor,
    wet_60_cnt: float,
    wet_61_cnt: float,
    n_body_params: float,
    curr_trust_rms: float,
    curr_lambda_act: float,
    ablation: str,
    round_idx: int,
    rank: int,
    device: torch.device,
) -> None:
    """Execute decoupled-LR AdamW replay-buffer optimization through the fitted response operator."""
    head_params = [p for n, p in model.named_parameters() if n.startswith("head.")]
    body_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    freeze_body_s2 = ablation == "head_only_stage2"
    if freeze_body_s2:
        for p in body_params:
            p.requires_grad_(False)

    ddp_net: torch.nn.Module | None = None
    try:
        ddp_net = (
            DDP(model, device_ids=[device.index])
            if (dist.is_initialized() and device.type == "cuda")
            else (DDP(model) if dist.is_initialized() else model)
        )
        body_refs = [p.detach().clone() for p in body_params]

        opt_param_groups: list[dict[str, Any]] = [
            {
                "params": head_params,
                "lr": float(cfg.forge_stage2_head_lr),
                "weight_decay": 0.0,
            },
        ]
        if not freeze_body_s2:
            opt_param_groups.append(
                {
                    "params": body_params,
                    "lr": float(cfg.forge_stage2_body_lr),
                    "weight_decay": 1e-4,
                }
            )
        opt_s2 = torch.optim.AdamW(opt_param_groups)
        n_proj_steps = int(cfg.forge_proj_steps)
        sched_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_s2,
            T_max=max(n_proj_steps, 1),
            eta_min=float(cfg.forge_stage2_body_lr) * 0.2,
        )
        trainable_params = head_params if freeze_body_s2 else list(model.parameters())

        n_buf = len(replay_buffer)
        mb_size = min(int(cfg.forge_stage2_minibatch_windows), n_buf)
        rng_mb = np.random.default_rng(int(cfg.seed) + round_idx * 6151 + rank * 97)
        spatial_basis_dev = resp_op.spatial_basis.detach()
        trust_hinge_sq = (1.5 * curr_trust_rms) ** 2

        ddp_net.train()
        for _step_s2 in range(n_proj_steps):
            if n_buf <= mb_size:
                mb_indices = list(range(n_buf))
            else:
                older = rng_mb.choice(n_buf - 1, size=mb_size - 1, replace=False).tolist()
                mb_indices = [int(ix) for ix in older] + [n_buf - 1]

            opt_s2.zero_grad(set_to_none=True)
            w_sum_mb = 0.0
            loss_mb = torch.tensor(0.0, device=device)

            for b_idx in mb_indices:
                ent = replay_buffer[b_idx]
                w_ent = float(cfg.forge_stage2_replay_decay) ** max(
                    round_idx - int(ent["round"]), 0
                )
                w_sum_mb += w_ent

                x_nom_b = ent["x_nom"].to(device=device, dtype=torch.float32)
                a_nom_b = ent["a_nom_proj"].to(device=device, dtype=torch.float32)
                r_nom_b = ent["r_nom"].to(device=device, dtype=torch.float32)
                c_blk_b = ent["c_blocks"].to(device=device, dtype=torch.float32)
                d_blk_b = ent["d_blocks"].to(device=device, dtype=torch.float32)
                m_blk_b = ent["m_blocks"].to(device=device, dtype=torch.float32)
                use_obs_b = (ablation != "no_kobs") and (float(ent["gamma_obs"]) > 1e-6)
                c_obs_b = ent["c_obs"].to(device=device, dtype=torch.float32) if use_obs_b else None
                d_obs_b = ent["d_obs"].to(device=device, dtype=torch.float32) if use_obs_b else None

                delta_acts_b: list[torch.Tensor] = []
                null_pen_b = torch.tensor(0.0, device=device)
                for k_hr in range(cfg.hours_per_window):
                    if k_hr == 0 or not use_obs_b:
                        x_hat_k = x_nom_b[k_hr]
                    else:
                        assert c_obs_b is not None and d_obs_b is not None
                        dx_k = apply_obs_step_differentiable(
                            k_hr,
                            delta_acts_b,
                            c_obs_b,
                            d_obs_b,
                            mask_60,
                            obs_mask_161,
                            obs_resp_op.pair_to_idx,
                        )
                        x_hat_k = x_nom_b[k_hr] + dx_k
                    g_raw_k = ddp_net(x_hat_k.unsqueeze(0))[0]
                    a_proj_k = env.project_standardized_action(g_raw_k)
                    delta_acts_b.append(a_proj_k - a_nom_b[k_hr])
                    null_pen_b = null_pen_b + ((g_raw_k - a_proj_k).square() * wq_60).sum()

                da_stack_b = torch.stack(delta_acts_b, dim=0)
                da_ms_b = (da_stack_b.square() * wq_60.unsqueeze(0)).sum() / (
                    float(cfg.hours_per_window) * wet_60_cnt
                )
                null_reg_b = 1e-4 * (null_pen_b / (float(cfg.hours_per_window) * wet_60_cnt))
                act_reg_b = curr_lambda_act * da_ms_b + 0.50 * F.relu(da_ms_b - trust_hinge_sq)

                if ablation == "action_l2":
                    da_cg_b = (ent["a_target_proj"] - ent["a_nom_proj"]).to(
                        device=device, dtype=torch.float32
                    )
                    obj_loss_b = ((da_stack_b - da_cg_b).square() * wq_60.unsqueeze(0)).sum() / (
                        float(cfg.hours_per_window) * wet_60_cnt
                    )
                else:
                    dy_segs_b = apply_response_trajectory_differentiable(
                        da_stack_b,
                        c_blk_b,
                        d_blk_b,
                        m_blk_b,
                        spatial_basis_dev,
                        mask_60,
                        mask_61,
                        resp_op.pair_to_idx,
                    )
                    pred_res_b = (r_nom_b + dy_segs_b) * wo_61.unsqueeze(0)
                    obj_loss_b = pred_res_b.square().sum() / (12.0 * wet_61_cnt)

                loss_mb = loss_mb + w_ent * (obj_loss_b + act_reg_b + null_reg_b)

            loss_mb = loss_mb / max(w_sum_mb, 1e-8)
            anchor_pen = torch.tensor(0.0, device=device)
            if not freeze_body_s2:
                for p_cur, p_ref in zip(body_params, body_refs, strict=True):
                    anchor_pen = anchor_pen + (p_cur - p_ref).square().sum()
            total_s2_loss = loss_mb + float(cfg.forge_anchor_weight) * (anchor_pen / n_body_params)

            total_s2_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt_s2.step()
            sched_s2.step()
    finally:
        del ddp_net
        if freeze_body_s2:
            for p in body_params:
                p.requires_grad_(True)
        model.zero_grad(set_to_none=True)


def run_stage2_response_step(
    cfg: ForgePOM3DConfig,
    env: BlackBoxProvider3D,
    model: ForgePOMNet,
    resp_op: CausalBlockResponseOperator3D,
    obs_resp_op: CausalObservationResponseOperator3D,
    replay_buffer: deque[dict[str, Any]],
    train_pool: list[tuple[int, int]],
    my_windows: list[tuple[int, int]],
    s1_res: Stage1RoundResult,
    mask_60: torch.Tensor,
    mask_61: torch.Tensor,
    obs_mask_161: torch.Tensor,
    wq_60: torch.Tensor,
    wo_61: torch.Tensor,
    wet_60_cnt: float,
    wet_61_cnt: float,
    body_mask_bool: torch.Tensor,
    n_body_params: float,
    h_norm: torch.Tensor,
    curr_trust_rms: float,
    curr_lambda_act: float,
    n_s2_candidates: int,
    n_extra_screen_per_gpu: int,
    stage_name: str,
    ablation: str,
    round_idx: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Stage2RoundResult:
    """Execute Stage II probe fitting, held-out gating, adjoint PCG, replay projection, and screening."""
    (
        s1_obs_list,
        s1_actions_raw_list,
        s1_actions_proj_list,
        s1_segments_list,
        s1_losses_list,
        recap_rollouts,
    ) = _recapture_stage1_center_if_needed(cfg, env, model, my_windows, s1_res, world_size)
    rollouts_used = recap_rollouts

    all_fit, all_val, probe_rollouts = collect_stage2_hadamard_probes(
        cfg,
        env,
        my_windows,
        s1_obs_list,
        s1_actions_raw_list,
        s1_actions_proj_list,
        s1_segments_list,
        mask_60,
        h_norm,
        round_idx,
        rank,
        world_size,
        device,
    )
    rollouts_used += probe_rollouts

    fit_acts = [[t.to(device) for t in item[1]] for item in all_fit]
    fit_segs = [[t.to(device) for t in item[2]] for item in all_fit]
    fit_obs_123 = [[t.to(device) for t in item[3]] for item in all_fit]

    val_acts = [[t.to(device) for t in item[1]] for item in all_val]
    val_segs = [[t.to(device) for t in item[2]] for item in all_val]
    val_obs_123 = [[t.to(device) for t in item[3]] for item in all_val]

    fit_stats = resp_op.fit_from_probes(
        fit_acts,
        fit_segs,
        ridge=cfg.forge_response_ridge,
        weights_61=wo_61,
        action_weights_60=wq_60,
    )
    heldout_gate = resp_op.validate_heldout_probes(
        val_acts,
        val_segs,
        weights_61=wo_61,
        action_weights_60=wq_60,
        relative_limit=cfg.response_gate_rel_err_max,
        cosine_limit=cfg.response_gate_cosine_min,
        r2_limit=cfg.response_gate_r2_min,
        per_probe_limit=cfg.response_gate_per_probe_max,
    )

    obs_fit_stats = obs_resp_op.fit_from_probes(
        fit_acts, fit_obs_123, ridge=cfg.forge_response_ridge
    )
    obs_gate = obs_resp_op.validate_heldout_probes(
        val_acts,
        val_obs_123,
        cosine_min=cfg.obs_response_cosine_min,
        r2_min=cfg.obs_response_r2_min,
    )
    obs_gate["fit_rel_err"] = obs_fit_stats["fit_rel_err"]
    raw_gamma_obs = float(obs_gate["gamma_obs"])
    obs_gate["raw_gamma_obs"] = raw_gamma_obs
    if ablation == "no_kobs":
        gamma_obs = 0.0
        obs_gate["obs_feedback_enabled"] = False
    else:
        gamma_obs = raw_gamma_obs
        obs_gate["obs_feedback_enabled"] = True
    obs_gate["gamma_obs"] = gamma_obs

    del fit_acts, fit_segs, fit_obs_123, val_acts, val_segs, val_obs_123, all_fit, all_val

    chosen_alpha_s2 = 0.0
    stage2_accepted = False
    stage2_source = "none"
    post_s2_sync = s1_res.stage1_train_loss
    stage2_best_cand_loss: float | None = None
    stage2_max_body_diff = 0.0
    stage2_body_l2_diff = 0.0
    cg_info: dict[str, Any] = {}
    theta_stage1 = s1_res.theta_stage1

    if not heldout_gate["passed"]:
        model.set_flat_params(theta_stage1)
        return Stage2RoundResult(
            chosen_alpha_s2=chosen_alpha_s2,
            stage2_accepted=stage2_accepted,
            stage2_source=stage2_source,
            post_s2_sync=post_s2_sync,
            stage2_best_cand_loss=stage2_best_cand_loss,
            stage2_max_body_diff=stage2_max_body_diff,
            stage2_body_l2_diff=stage2_body_l2_diff,
            curr_trust_rms=curr_trust_rms,
            curr_lambda_act=curr_lambda_act,
            fit_stats=fit_stats,
            heldout_gate=heldout_gate,
            obs_gate=obs_gate,
            cg_info=cg_info,
            window_rollouts_used=rollouts_used,
        )

    if ablation == "intra_hour_only":
        resp_op.mask_cross_hour_blocks_()

    any_cg_usable = False
    for w_i in range(len(my_windows)):
        residuals_w = [
            (s1_segments_list[w_i][j] - s1_res.ref_segments_list[w_i][j]).detach()
            for j in range(12)
        ]
        delta_acts_star, cg_info = resp_op.solve_adjoint_cg(
            residuals_w,
            weights_61=wo_61,
            action_weights_60=wq_60,
            project_fn=env.project_standardized_action,
            project_transpose_fn=env.project_standardized_action_transpose,
            iterations=32,
            damping=curr_lambda_act,
            trust_rms=curr_trust_rms,
        )
        if cg_info["usable"]:
            any_cg_usable = True

        a_nom_stack = torch.stack(s1_actions_proj_list[w_i], dim=0)
        da_star_stack = torch.stack(delta_acts_star, dim=0)
        replay_buffer.append(
            {
                "round": int(round_idx),
                "x_nom": s1_obs_list[w_i].detach().cpu(),
                "a_nom_proj": a_nom_stack.detach().cpu(),
                "r_nom": torch.stack(residuals_w, dim=0).detach().cpu(),
                "a_target_proj": (a_nom_stack + da_star_stack).detach().cpu(),
                "c_blocks": resp_op.C_blocks.detach().cpu().clone(),
                "d_blocks": resp_op.D_blocks.detach().cpu().clone(),
                "m_blocks": resp_op.M_blocks.detach().cpu().clone(),
                "gamma_obs": gamma_obs,
                "c_obs": (obs_resp_op.C_obs.detach() * gamma_obs).cpu().clone(),
                "d_obs": (obs_resp_op.D_obs.detach() * gamma_obs).cpu().clone(),
            }
        )

    usable_sync = ddp_mean_dict({"u": 1.0 if any_cg_usable else 0.0}, device)["u"]
    if usable_sync <= 0.0 or len(replay_buffer) == 0:
        model.set_flat_params(theta_stage1)
        return Stage2RoundResult(
            chosen_alpha_s2=chosen_alpha_s2,
            stage2_accepted=stage2_accepted,
            stage2_source=stage2_source,
            post_s2_sync=post_s2_sync,
            stage2_best_cand_loss=stage2_best_cand_loss,
            stage2_max_body_diff=stage2_max_body_diff,
            stage2_body_l2_diff=stage2_body_l2_diff,
            curr_trust_rms=curr_trust_rms,
            curr_lambda_act=curr_lambda_act,
            fit_stats=fit_stats,
            heldout_gate=heldout_gate,
            obs_gate=obs_gate,
            cg_info=cg_info,
            window_rollouts_used=rollouts_used,
        )

    theta_head_warm = solve_replay_head_ridge(
        cfg, env, model, replay_buffer, wq_60, theta_stage1, round_idx, device
    )
    model.set_flat_params(theta_head_warm)
    model.zero_grad(set_to_none=True)

    optimize_stage2_replay_network(
        cfg,
        env,
        model,
        resp_op,
        obs_resp_op,
        replay_buffer,
        mask_60,
        mask_61,
        obs_mask_161,
        wq_60,
        wo_61,
        wet_60_cnt,
        wet_61_cnt,
        n_body_params,
        curr_trust_rms,
        curr_lambda_act,
        ablation,
        round_idx,
        rank,
        device,
    )

    freeze_body_s2 = ablation == "head_only_stage2"
    model.eval()
    with torch.no_grad():
        theta_full = model.get_flat_params().clone()
        delta_from_head = theta_full - theta_head_warm
        theta_half = theta_head_warm + 0.5 * delta_from_head
        theta_half_refit = solve_replay_head_ridge(
            cfg, env, model, replay_buffer, wq_60, theta_half, round_idx, device
        )
        theta_quarter = theta_head_warm + 0.25 * delta_from_head

        candidates_s2 = [
            ("head_warm", 1.0, theta_head_warm),
            ("full_resp", 1.0, theta_full),
            ("full_resp", 0.5, theta_half),
            ("full_resp_refit", 0.5, theta_half_refit),
            ("full_resp", 0.25, theta_quarter),
        ]
        if freeze_body_s2:
            for cand_tag, cand_alpha, cand_v in candidates_s2:
                cand_body_diff = float(((cand_v - theta_stage1)[body_mask_bool]).abs().max().item())
                if cand_body_diff > 0.0:
                    raise RuntimeError(
                        f"[{stage_name}] Trunk weights changed in head_only_stage2 candidate "
                        f"{cand_tag}@{cand_alpha}: max_diff={cand_body_diff}"
                    )

        if len(candidates_s2) != n_s2_candidates:
            raise RuntimeError(
                f"[{stage_name}] Expected {n_s2_candidates} Stage II candidates, got {len(candidates_s2)}"
            )

        extra_screen_windows: list[tuple[int, int]] = []
        if n_extra_screen_per_gpu > 0:
            extra_all = sample_epoch_windows(
                train_pool,
                round_idx=round_idx,
                batch_size=n_extra_screen_per_gpu * world_size,
                seed=cfg.seed,
                stream_id=4049,
            )
            extra_screen_windows = extra_all[rank::world_size]

        screen_windows = list(my_windows) + list(extra_screen_windows)
        n_scr = max(len(screen_windows), 1)

        model.set_flat_params(theta_stage1)
        local_s1_scr_sum = float(sum(s1_losses_list))
        if extra_screen_windows:
            local_ext_s1 = 0.0
            for m, d in extra_screen_windows:
                l_ext, _, _ = env.rollout_4h_window(m, d, lambda x_k, _k: model(x_k))
                local_ext_s1 += float(l_ext)
            rollouts_used += len(extra_screen_windows) * world_size
            local_s1_scr_sum += local_ext_s1

        s1_scr_loss = ddp_mean_dict({"l": local_s1_scr_sum / float(n_scr)}, device)["l"]
        best_scr_loss = s1_scr_loss
        best_cand_vec = theta_stage1

        for src_tag, alpha_s2, cand_vec in candidates_s2:
            model.set_flat_params(cand_vec)
            cand_sum = 0.0
            for m, d in screen_windows:
                l_w, _, _ = env.rollout_4h_window(m, d, lambda x_k, _k: model(x_k))
                cand_sum += float(l_w)
            rollouts_used += len(screen_windows) * world_size
            cand_sync_l = ddp_mean_dict({"l": cand_sum / float(n_scr)}, device)["l"]
            if math.isfinite(cand_sync_l):
                if stage2_best_cand_loss is None or cand_sync_l < stage2_best_cand_loss:
                    stage2_best_cand_loss = cand_sync_l
                if cand_sync_l < best_scr_loss - 1e-7:
                    best_scr_loss = cand_sync_l
                    post_s2_sync = cand_sync_l
                    chosen_alpha_s2 = float(alpha_s2)
                    stage2_source = src_tag
                    stage2_accepted = True
                    best_cand_vec = cand_vec.clone()

        if stage2_accepted:
            body_delta_vec = (best_cand_vec - theta_stage1)[body_mask_bool]
            stage2_max_body_diff = float(body_delta_vec.abs().max().item())
            stage2_body_l2_diff = float(torch.linalg.vector_norm(body_delta_vec).item())
            if ablation == "head_only_stage2" and stage2_max_body_diff > 0.0:
                raise RuntimeError(
                    f"[{stage_name}] Trunk weights changed in head_only_stage2: max_diff={stage2_max_body_diff}"
                )
            model.set_flat_params(best_cand_vec)
            if chosen_alpha_s2 >= 0.5:
                curr_trust_rms = min(float(cfg.forge_cg_trust_max), curr_trust_rms * 1.08)
                curr_lambda_act = max(0.003, curr_lambda_act * 0.94)
        else:
            model.set_flat_params(theta_stage1)
            curr_trust_rms = max(float(cfg.forge_cg_trust_min), curr_trust_rms * 0.85)
            curr_lambda_act = min(0.15, curr_lambda_act * 1.20)

    return Stage2RoundResult(
        chosen_alpha_s2=chosen_alpha_s2,
        stage2_accepted=stage2_accepted,
        stage2_source=stage2_source,
        post_s2_sync=post_s2_sync,
        stage2_best_cand_loss=stage2_best_cand_loss,
        stage2_max_body_diff=stage2_max_body_diff,
        stage2_body_l2_diff=stage2_body_l2_diff,
        curr_trust_rms=curr_trust_rms,
        curr_lambda_act=curr_lambda_act,
        fit_stats=fit_stats,
        heldout_gate=heldout_gate,
        obs_gate=obs_gate,
        cg_info=cg_info,
        window_rollouts_used=rollouts_used,
    )


__all__ = [
    "HADAMARD_4X4",
    "Stage2RoundResult",
    "build_projected_action_probe_direction",
    "collect_stage2_hadamard_probes",
    "measure_action_jacobian_ranks",
    "optimize_stage2_replay_network",
    "run_stage2_response_step",
    "solve_replay_head_ridge",
]
