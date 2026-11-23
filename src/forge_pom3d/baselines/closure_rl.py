"""Stage 7 public-interface wet-column Closure-RL baseline for 3D POM."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ..artifacts import (
    append_jsonl,
    relativize_artifact_path,
    save_atomic_json,
    sha256_file,
)
from ..config import ForgePOM3DConfig
from ..distributed import (
    ddp_mean_dict,
    get_ddp_solver_window_equivalents,
    is_main_process,
)
from ..evaluation import (
    Pure4hSelectionTracker,
    can_afford_next_round,
    evaluate_shared_windows_ddp,
    evaluate_validation_4h_with_worst,
)
from ..network import ForgePOMNet
from ..packing import build_wet_masks_60_61
from ..physics_env import BlackBoxProvider3D
from ..sampling import ensure_common_zero_ckpt, sample_epoch_windows


def compute_wet_column_ppo_stats(
    a_sampled: torch.Tensor,
    mu_old: torch.Tensor,
    mu_new: torch.Tensor,
    mask_60: torch.Tensor,
    col_wet: torch.Tensor,
    std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact per-wet-column 60-channel joint Gaussian log-ratio and KL in float64."""
    single = a_sampled.dim() == 3
    if single:
        a_sampled = a_sampled.unsqueeze(0)
        mu_old = mu_old.unsqueeze(0)
        mu_new = mu_new.unsqueeze(0)

    inv_var = 1.0 / max(float(std) * float(std), 1e-30)
    mask_d = mask_60.double().unsqueeze(0)
    col_d = col_wet.to(dtype=torch.float64).unsqueeze(0)

    delta = (mu_new.double() - mu_old.double()) * mask_d
    noise = (a_sampled.double() - mu_old.double()) * mask_d

    delta_sq_col = delta.square().sum(dim=1) * col_d
    dot_col = (noise * delta).sum(dim=1) * col_d

    log_ratio_col = inv_var * dot_col - 0.5 * inv_var * delta_sq_col
    kl_col = 0.5 * inv_var * delta_sq_col

    if single:
        return log_ratio_col[0], kl_col[0]
    return log_ratio_col, kl_col


class WetColumnSpatialCritic3D(nn.Module):
    """Lightweight public-feature spatial value baseline `V_psi(x_k, k)` on `(im, jm)` wet columns."""

    def __init__(self, obs_channels: int = 161, width: int = 64) -> None:
        super().__init__()
        in_ch = obs_channels + 4
        groups = 8 if width % 8 == 0 else 4
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=3, padding=1),
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GroupNorm(groups, width),
            nn.GELU(),
        )
        self.head = nn.Conv2d(width, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        obs_b: torch.Tensor,
        hour_idx_b: torch.Tensor,
        col_wet: torch.Tensor,
    ) -> torch.Tensor:
        bsz, _, im, jm = obs_b.shape
        phase = torch.zeros((bsz, 4, im, jm), device=obs_b.device, dtype=obs_b.dtype)
        for bi in range(bsz):
            phase[bi, int(hour_idx_b[bi].item()) % 4] = 1.0
        inp = torch.cat([obs_b, phase], dim=1)
        v = self.head(self.net(inp)).squeeze(1)
        return v * col_wet.to(dtype=v.dtype).unsqueeze(0)


class _ClosureRLActorShard3D(nn.Module):
    """DDP-wrappable per-wet-column clipped PPO surrogate objective."""

    def __init__(
        self,
        actor: ForgePOMNet,
        mask_60: torch.Tensor,
        col_wet: torch.Tensor,
        std: float,
        clip_eps: float,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.register_buffer("mask_60", mask_60.detach().clone(), persistent=False)
        self.register_buffer("col_wet", col_wet.detach().clone(), persistent=False)
        self.n_wet_cols = float(col_wet.double().sum().clamp_min(1.0).item())
        self.std = float(std)
        self.clip_eps = float(clip_eps)

    def forward(
        self,
        obs_mb: torch.Tensor,
        act_mb: torch.Tensor,
        old_mb: torch.Tensor,
        adv_mb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_new_mb = self.actor(obs_mb)
        log_ratio_col, kl_col = compute_wet_column_ppo_stats(
            act_mb, old_mb, mu_new_mb, self.mask_60, self.col_wet, self.std
        )
        ratio_col = torch.exp(log_ratio_col)
        surr1 = ratio_col * adv_mb
        surr2 = torch.clamp(ratio_col, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mb
        col_mask_d = self.col_wet.to(dtype=torch.float64).unsqueeze(0)
        bsz = float(max(int(obs_mb.shape[0]), 1))
        denom = bsz * self.n_wet_cols
        ppo_loss = ((-torch.minimum(surr1, surr2)) * col_mask_d).sum() / denom
        mean_kl = (kl_col * col_mask_d).sum() / denom
        max_frame_kl = ((kl_col * col_mask_d).sum(dim=(1, 2)) / self.n_wet_cols).max()
        return ppo_loss, mean_kl.detach(), max_frame_kl.detach()


class _ClosureRLCriticShard3D(nn.Module):
    """DDP-wrappable wet-column value regression shard."""

    def __init__(self, critic: WetColumnSpatialCritic3D, col_wet: torch.Tensor) -> None:
        super().__init__()
        self.critic = critic
        self.register_buffer("col_wet", col_wet.detach().clone(), persistent=False)
        self.n_wet_cols = float(col_wet.double().sum().clamp_min(1.0).item())

    def forward(
        self,
        obs_b: torch.Tensor,
        hour_b: torch.Tensor,
        ret_b: torch.Tensor,
    ) -> torch.Tensor:
        v_pred = self.critic(obs_b, hour_b, self.col_wet).double()
        col_mask_d = self.col_wet.to(dtype=torch.float64).unsqueeze(0)
        diff = (v_pred - ret_b.double()) * col_mask_d
        bsz = float(max(int(obs_b.shape[0]), 1))
        return F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="sum") / (
            bsz * self.n_wet_cols
        )


@torch.no_grad()
def evaluate_closure_rl_batch_stats_ddp(
    actor: ForgePOMNet,
    obs_b: torch.Tensor,
    act_b: torch.Tensor,
    old_b: torch.Tensor,
    adv_b: torch.Tensor,
    mask_60: torch.Tensor,
    col_wet: torch.Tensor,
    std: float,
    clip_eps: float,
    device: torch.device,
) -> dict[str, float]:
    """Compute exact global DDP wet-column KL, max frame KL, clip fraction, and ratio bounds."""
    was_tr = actor.training
    actor.eval()
    mu_new_b = actor(obs_b)
    if was_tr:
        actor.train()

    log_ratio_col, kl_col = compute_wet_column_ppo_stats(
        act_b, old_b, mu_new_b, mask_60, col_wet, std
    )
    ratio_col = torch.exp(log_ratio_col)

    wet_3d = col_wet.unsqueeze(0).expand_as(ratio_col)
    lr_wet = log_ratio_col.masked_select(wet_3d)
    rat_wet = ratio_col.masked_select(wet_3d)
    kl_wet = kl_col.masked_select(wet_3d)
    adv_wet = adv_b.masked_select(wet_3d)

    is_finite = bool(
        torch.isfinite(lr_wet).all().item()
        and torch.isfinite(rat_wet).all().item()
        and torch.isfinite(kl_wet).all().item()
    )
    nf_flag = torch.tensor([0.0 if is_finite else 1.0], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(nf_flag, op=dist.ReduceOp.MAX)
    if float(nf_flag[0].item()) > 0.0:
        return {
            "nf": 1.0,
            "mean_kl": float("inf"),
            "max_frame_kl": float("inf"),
            "clip_fraction": 1.0,
            "ratio_min": 0.0,
            "ratio_max": float("inf"),
            "unclipped_max_adv": 0.0,
        }

    clipped_mask = ((adv_wet > 0.0) & (rat_wet > 1.0 + clip_eps)) | (
        (adv_wet < 0.0) & (rat_wet < 1.0 - clip_eps)
    )
    unclipped_adv = adv_wet.abs().masked_select(~clipped_mask)
    unclipped_max = float(unclipped_adv.max().item()) if unclipped_adv.numel() > 0 else 0.0

    n_wet_cols = float(col_wet.double().sum().clamp_min(1.0).item())
    frame_kls = (kl_col * col_wet.to(dtype=torch.float64).unsqueeze(0)).sum(dim=(1, 2)) / n_wet_cols
    max_frame_kl = float(frame_kls.max().item()) if frame_kls.numel() > 0 else 0.0

    mean_sync = ddp_mean_dict(
        {
            "mean_kl": float(kl_wet.mean().item()),
            "clip_fraction": float(clipped_mask.double().mean().item()),
        },
        device,
    )

    rmin_local = float(rat_wet.min().item())
    rmax_local = float(rat_wet.max().item())
    ext_t = torch.tensor(
        [-rmin_local, rmax_local, max_frame_kl, unclipped_max],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(ext_t, op=dist.ReduceOp.MAX)
    return {
        "nf": 0.0,
        "mean_kl": mean_sync["mean_kl"],
        "max_frame_kl": float(ext_t[2].item()),
        "clip_fraction": mean_sync["clip_fraction"],
        "ratio_min": -float(ext_t[0].item()),
        "ratio_max": float(ext_t[1].item()),
        "unclipped_max_adv": float(ext_t[3].item()),
    }


def run_closure_rl_baseline(
    cfg: ForgePOM3DConfig,
    env: BlackBoxProvider3D,
    common_ckpt: Path,
    train_pool: list[tuple[int, int]],
    val_windows: list[tuple[int, int]],
    rank: int,
    world_size: int,
    device: torch.device,
    max_window_budget: float | None = None,
) -> dict[str, Any]:
    """Stage 7 (Pure-4h): Public-Interface Wet-Column Closure-RL (`stage7_closure_rl`)."""
    out_root_path = Path(cfg.output_root)
    stage_dir = out_root_path / "stage7_closure_rl"
    summary_path = stage_dir / "summary.json"
    best_ckpt = stage_dir / "best_model.pt"
    critic_ckpt = stage_dir / "critic_last.pt"
    if summary_path.exists() and best_ckpt.exists():
        if is_main_process(rank):
            print(
                "[Stage 7 Closure-RL] Already completed, loading summary.json",
                flush=True,
            )
        return json.loads(summary_path.read_text(encoding="utf-8"))

    t0 = time.time()
    zero_ckpt = ensure_common_zero_ckpt(cfg, rank, device, explicit_ckpt=common_ckpt)
    s_cnt_start = get_ddp_solver_window_equivalents(env, device, forbid_whitebox=True)
    torch.manual_seed(int(cfg.seed))

    base_model = ForgePOMNet(
        cfg.in_channels, cfg.hidden_channels, cfg.out_channels, cfg.num_blocks
    ).to(device)
    base_model.load_state_dict(torch.load(zero_ckpt, map_location=device, weights_only=True))

    critic = WetColumnSpatialCritic3D(
        obs_channels=cfg.in_channels, width=cfg.closure_rl_critic_width
    ).to(device)
    critic_opt = torch.optim.AdamW(
        critic.parameters(), lr=cfg.closure_rl_critic_lr, weight_decay=1e-4
    )
    actor_opt = torch.optim.Adam(base_model.parameters(), lr=cfg.closure_rl_lr)

    mask_60, _ = build_wet_masks_60_61(env)
    col_wet = (env.get_public_geometry()["fsm"] > 0.5).contiguous()
    if bool(mask_60[:, ~col_wet].abs().max().item() > 0.0):
        raise RuntimeError("Active coordinates in mask_60 fell outside col_wet = (fsm > 0.5)")

    actor_shard = _ClosureRLActorShard3D(
        base_model,
        mask_60=mask_60,
        col_wet=col_wet,
        std=cfg.closure_rl_explore_std,
        clip_eps=cfg.closure_rl_clip_eps,
    ).to(device)
    actor_ddp: nn.Module = (
        DDP(
            actor_shard,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
        )
        if dist.is_initialized()
        else actor_shard
    )

    critic_shard = _ClosureRLCriticShard3D(critic, col_wet=col_wet).to(device)
    critic_ddp: nn.Module = (
        DDP(
            critic_shard,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
        )
        if dist.is_initialized()
        else critic_shard
    )

    init_metrics = evaluate_validation_4h_with_worst(
        env, base_model, val_windows, rank, world_size, device
    )
    train_window_rollouts = 0
    val_4h_window_rollouts = len(val_windows)

    tracker = Pure4hSelectionTracker(
        stage_dir=stage_dir,
        rank=rank,
        init_state=base_model.state_dict(),
        init_val_metrics=init_metrics,
        init_cum_windows=float(val_4h_window_rollouts),
        anchor_window_budget=cfg.anchor_window_budget,
        improve_tol=cfg.val_improve_tol,
    )

    stop_reason = "completed_max_rounds"
    accepted_rounds = 0
    total_nonfinite_attempts = 0
    total_nonfinite_failures = 0
    total_shrunk_rounds = 0
    max_spatial_decomp_error = 0.0

    std = float(cfg.closure_rl_explore_std)
    target_kl = float(cfg.closure_rl_target_kl)
    max_ex_kl = float(cfg.closure_rl_max_example_kl)
    clip_eps = float(cfg.closure_rl_clip_eps)
    alphas = tuple(float(a) for a in cfg.closure_rl_step_alphas)
    batch_per_round = int(cfg.closure_rl_windows_per_gpu) * max(world_size, 1)
    round_train_val_cost = float(batch_per_round * (2 + len(alphas)) + len(val_windows))
    post_eval_reserve = float(len(val_windows))
    wet_cnt_60 = float(mask_60.double().sum().clamp_min(1.0).item())
    n_wet_cols = float(col_wet.double().sum().clamp_min(1.0).item())
    max_rounds = int(cfg.baseline_rounds)

    if is_main_process(rank):
        print(
            f"[Stage 7 Closure-RL] Starting pure-4h training from common_zero.pt "
            f"(val_4h={tracker.init_val_4h:.6f}, wet_cols={int(col_wet.sum().item()):,}, "
            f"critic_params={sum(p.numel() for p in critic.parameters()):,}, "
            f"max_window_budget={max_window_budget}, round_win={round_train_val_cost:.0f})...",
            flush=True,
        )

    for r in range(1, max_rounds + 1):
        current_cum = float(train_window_rollouts + val_4h_window_rollouts)
        allowed, _ = can_afford_next_round(
            current_cum + post_eval_reserve, round_train_val_cost, max_window_budget
        )
        if not allowed:
            stop_reason = "reached_max_window_budget_cap"
            break

        r_t0 = time.time()

        base_model.eval()
        critic.eval()
        round_wins = sample_epoch_windows(
            train_pool,
            round_idx=r,
            batch_size=batch_per_round,
            seed=cfg.seed,
            stream_id=7071,
        )
        my_windows = round_wins[rank::world_size]

        obs_list: list[torch.Tensor] = []
        act_list: list[torch.Tensor] = []
        old_mean_list: list[torch.Tensor] = []
        hour_list: list[int] = []
        ret_list: list[torch.Tensor] = []

        local_base_loss_sum = 0.0
        local_base_tr_sum = 0.0
        local_exp_loss_sum = 0.0
        local_decomp_max_err = 0.0
        local_pre_proj_sq = 0.0
        local_post_proj_sq = 0.0
        local_noise_frames = 0.0

        with torch.no_grad():
            for w_idx, (m, d) in enumerate(my_windows):
                _, _, segs_base = env.rollout_4h_window(m, d, lambda x_k, _k: base_model(x_k))
                base_h_costs, base_loss_f64, base_met = env.compute_12seg_local_column_costs(
                    m, d, segs_base
                )
                base_recon = float(base_h_costs[:, col_wet].mean(dim=1).sum().item())
                err_base = abs(base_recon - base_loss_f64)

                env.reset(m, d)
                xs_w: list[torch.Tensor] = []
                acts_w: list[torch.Tensor] = []
                means_w: list[torch.Tensor] = []
                segs_exp: list[dict[str, torch.Tensor]] = []
                for k in range(cfg.hours_per_window):
                    x_k = env.observe_F()
                    mu_k = base_model(x_k)
                    gen_n = torch.Generator(device="cpu").manual_seed(
                        cfg.seed + r * 12007 + rank * 157 + w_idx * 23 + k
                    )
                    noise = (
                        std
                        * torch.randn(mu_k.shape, generator=gen_n, dtype=torch.float32).to(
                            device=device, dtype=mu_k.dtype
                        )
                        * mask_60
                    )
                    noise_proj = env.project_standardized_action(noise) * mask_60
                    local_pre_proj_sq += float(noise.double().square().sum().item()) / wet_cnt_60
                    local_post_proj_sq += (
                        float(noise_proj.double().square().sum().item()) / wet_cnt_60
                    )
                    local_noise_frames += 1.0

                    a_k = mu_k + noise
                    segs_exp.extend(env.step_H(a_k))
                    xs_w.append(x_k.detach())
                    acts_w.append(a_k.detach())
                    means_w.append(mu_k.detach())

                exp_h_costs, exp_loss_f64, _ = env.compute_12seg_local_column_costs(m, d, segs_exp)
                exp_recon = float(exp_h_costs[:, col_wet].mean(dim=1).sum().item())
                err_exp = abs(exp_recon - exp_loss_f64)
                win_decomp_err = max(err_base, err_exp)
                if win_decomp_err > 1e-9:
                    raise RuntimeError(
                        f"[Stage 7 Closure-RL] Spatial cost decomposition mismatch on (m={m}, d={d}): "
                        f"err_base={err_base:.3e}, err_exp={err_exp:.3e}"
                    )
                local_decomp_max_err = max(local_decomp_max_err, win_decomp_err)

                local_base_loss_sum += float(base_loss_f64)
                local_base_tr_sum += float(base_met["tracer_loss_4h"])
                local_exp_loss_sum += float(exp_loss_f64)

                diff_h = (base_h_costs - exp_h_costs) * col_wet.to(dtype=torch.float64).unsqueeze(0)
                running_suffix = torch.zeros_like(diff_h[0])
                suffix_returns: list[torch.Tensor] = [
                    torch.empty(0) for _ in range(cfg.hours_per_window)
                ]
                for k in range(cfg.hours_per_window - 1, -1, -1):
                    running_suffix = running_suffix + diff_h[k]
                    suffix_returns[k] = running_suffix.clone()

                for k in range(cfg.hours_per_window):
                    obs_list.append(xs_w[k])
                    act_list.append(acts_w[k])
                    old_mean_list.append(means_w[k])
                    hour_list.append(k)
                    ret_list.append(suffix_returns[k])

        train_window_rollouts += 2 * len(my_windows) * world_size

        roll_sync = torch.tensor(
            [
                local_base_loss_sum,
                local_base_tr_sum,
                local_exp_loss_sum,
                local_pre_proj_sq,
                local_post_proj_sq,
                local_noise_frames,
            ],
            device=device,
            dtype=torch.float64,
        )
        decomp_t = torch.tensor([local_decomp_max_err], device=device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(roll_sync, op=dist.ReduceOp.SUM)
            dist.all_reduce(decomp_t, op=dist.ReduceOp.MAX)

        center_loss_4h = float(roll_sync[0].item()) / max(len(round_wins), 1)
        mean_exp_loss_4h = float(roll_sync[2].item()) / max(len(round_wins), 1)
        pre_proj_rms = math.sqrt(float(roll_sync[3].item()) / max(float(roll_sync[5].item()), 1.0))
        post_proj_rms = math.sqrt(float(roll_sync[4].item()) / max(float(roll_sync[5].item()), 1.0))
        round_decomp_err = float(decomp_t[0].item())
        max_spatial_decomp_error = max(max_spatial_decomp_error, round_decomp_err)

        obs_batch = torch.stack(obs_list, dim=0)
        act_batch = torch.stack(act_list, dim=0)
        old_batch = torch.stack(old_mean_list, dim=0)
        hour_batch = torch.tensor(hour_list, device=device, dtype=torch.long)
        ret_batch = torch.stack(ret_list, dim=0)

        with torch.no_grad():
            v_old = critic(obs_batch, hour_batch, col_wet).double()
            col_mask_d = col_wet.to(dtype=torch.float64).unsqueeze(0)
            ret_masked = ret_batch * col_mask_d
            raw_adv = (ret_batch - v_old) * col_mask_d
            n_frames_local = float(obs_batch.shape[0])
            adv_stats = torch.tensor(
                [
                    float(raw_adv.sum().item()),
                    float(raw_adv.square().sum().item()),
                    n_frames_local * n_wet_cols,
                    float(ret_masked.sum().item()),
                    float(ret_masked.square().sum().item()),
                ],
                device=device,
                dtype=torch.float64,
            )
            if dist.is_initialized():
                dist.all_reduce(adv_stats, op=dist.ReduceOp.SUM)
            denom_adv = adv_stats[2].clamp_min(1.0)
            adv_mean = adv_stats[0] / denom_adv
            adv_var = (adv_stats[1] / denom_adv - adv_mean.square()).clamp_min(1e-12)
            adv_std = adv_var.sqrt()
            ret_mean = adv_stats[3] / denom_adv
            ret_std = (adv_stats[4] / denom_adv - ret_mean.square()).clamp_min(0.0).sqrt()
            raw_adv_std_val = float(adv_std.item())
            ret_std_val = float(ret_std.item())
            adv_norm_batch = ((raw_adv - adv_mean) / adv_std) * col_mask_d

        critic.train()
        critic_losses: list[float] = []
        for _ce in range(4):
            critic_opt.zero_grad(set_to_none=True)
            c_loss = critic_ddp(obs_batch, hour_batch, ret_batch)
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
            critic_opt.step()
            cl_v = c_loss.detach().double()
            if dist.is_initialized():
                dist.all_reduce(cl_v, op=dist.ReduceOp.AVG)
            critic_losses.append(float(cl_v.item()))
        critic.eval()

        theta_center = base_model.get_flat_params()
        prev_actor_opt_state = copy.deepcopy(actor_opt.state_dict())
        n_local_frames = int(obs_batch.shape[0])
        mb_size = max(1, min(int(cfg.closure_rl_minibatch_frames), n_local_frames))

        head_params = [p for n, p in base_model.named_parameters() if n.startswith("head.")]
        body_params = [p for n, p in base_model.named_parameters() if not n.startswith("head.")]
        head_grad_norms: list[float] = []
        body_grad_norms: list[float] = []
        nonfinite_attempts = 0
        nonfinite_failures = 0
        kl_early_stopped = False

        for ep_idx in range(int(cfg.closure_rl_epochs)):
            rng_mb = np.random.default_rng(cfg.seed + r * 40009 + rank * 71 + ep_idx)
            perm = rng_mb.permutation(n_local_frames)
            for start_i in range(0, n_local_frames, mb_size):
                pre_stats = evaluate_closure_rl_batch_stats_ddp(
                    base_model,
                    obs_batch,
                    act_batch,
                    old_batch,
                    adv_norm_batch,
                    mask_60,
                    col_wet,
                    std,
                    clip_eps,
                    device,
                )
                if pre_stats["nf"] > 0.0:
                    nonfinite_attempts += 1
                    nonfinite_failures += 1
                    total_nonfinite_attempts += 1
                    total_nonfinite_failures += 1
                    raise FloatingPointError(
                        f"[Stage 7 Closure-RL] Non-finite pre-step statistics at round {r}, epoch {ep_idx}"
                    )
                if pre_stats["mean_kl"] > target_kl or pre_stats["max_frame_kl"] > max_ex_kl:
                    kl_early_stopped = True
                    break
                if pre_stats["unclipped_max_adv"] == 0.0:
                    kl_early_stopped = True
                    break

                idx_mb = torch.as_tensor(
                    perm[start_i : start_i + mb_size],
                    device=device,
                    dtype=torch.long,
                )
                base_model.train()
                actor_opt.zero_grad(set_to_none=True)
                ppo_loss, _, _ = actor_ddp(
                    obs_batch[idx_mb],
                    act_batch[idx_mb],
                    old_batch[idx_mb],
                    adv_norm_batch[idx_mb],
                )
                ppo_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(), 1.0, error_if_nonfinite=True
                )
                hg = math.sqrt(
                    sum(
                        float(p.grad.double().square().sum().item())
                        for p in head_params
                        if p.grad is not None
                    )
                )
                bg = math.sqrt(
                    sum(
                        float(p.grad.double().square().sum().item())
                        for p in body_params
                        if p.grad is not None
                    )
                )
                head_grad_norms.append(hg)
                body_grad_norms.append(bg)
                actor_opt.step()
            if kl_early_stopped:
                break

        base_model.eval()
        post_stats = evaluate_closure_rl_batch_stats_ddp(
            base_model,
            obs_batch,
            act_batch,
            old_batch,
            adv_norm_batch,
            mask_60,
            col_wet,
            std,
            clip_eps,
            device,
        )
        shrink_used = 1.0
        if (
            post_stats["nf"] > 0.0
            or post_stats["mean_kl"] > target_kl
            or post_stats["max_frame_kl"] > max_ex_kl
        ):
            if post_stats["nf"] > 0.0:
                nonfinite_attempts += 1
            theta_raw = base_model.get_flat_params()
            delta_raw = theta_raw - theta_center
            alpha_sh = (
                min(
                    0.75,
                    math.sqrt(target_kl / max(post_stats["mean_kl"], target_kl * 1.05)),
                )
                if (
                    post_stats["nf"] == 0.0
                    and math.isfinite(post_stats["mean_kl"])
                    and post_stats["mean_kl"] > 0.0
                )
                else 0.5
            )
            accepted_sh = False
            for _ in range(10):
                base_model.set_flat_params(theta_center + float(alpha_sh) * delta_raw)
                post_stats = evaluate_closure_rl_batch_stats_ddp(
                    base_model,
                    obs_batch,
                    act_batch,
                    old_batch,
                    adv_norm_batch,
                    mask_60,
                    col_wet,
                    std,
                    clip_eps,
                    device,
                )
                if post_stats["nf"] > 0.0:
                    nonfinite_attempts += 1
                if (
                    post_stats["nf"] == 0.0
                    and post_stats["mean_kl"] <= target_kl
                    and post_stats["max_frame_kl"] <= max_ex_kl
                ):
                    shrink_used = float(alpha_sh)
                    accepted_sh = True
                    total_shrunk_rounds += 1
                    break
                alpha_sh *= 0.5

            if not accepted_sh:
                base_model.set_flat_params(theta_center)
                actor_opt.load_state_dict(prev_actor_opt_state)
                shrink_used = 0.0
                post_stats = evaluate_closure_rl_batch_stats_ddp(
                    base_model,
                    obs_batch,
                    act_batch,
                    old_batch,
                    adv_norm_batch,
                    mask_60,
                    col_wet,
                    std,
                    clip_eps,
                    device,
                )
                if post_stats["nf"] > 0.0:
                    nonfinite_failures += 1
                    total_nonfinite_attempts += nonfinite_attempts
                    total_nonfinite_failures += nonfinite_failures
                    raise FloatingPointError(
                        f"[Stage 7 Closure-RL] Persistent non-finite statistics after rollback at round {r}"
                    )

        total_nonfinite_attempts += nonfinite_attempts
        total_nonfinite_failures += nonfinite_failures

        theta_prop = base_model.get_flat_params()
        delta_theta = theta_prop - theta_center
        prop_param_delta_norm = float(delta_theta.double().norm().item())

        best_alpha = 0.0
        best_cand_loss = center_loss_4h
        screen_table: list[dict[str, float]] = []
        for alpha in alphas:
            base_model.set_flat_params(theta_center + float(alpha) * delta_theta)
            cand_res = evaluate_shared_windows_ddp(
                env,
                base_model,
                round_wins,
                rank,
                world_size,
                device,
                capture_obs=False,
            )
            assert len(cand_res) == 2
            cand_met, n_eval_w = cand_res
            train_window_rollouts += int(n_eval_w)
            c_loss = float(cand_met["loss_4h"])
            c_tr = float(cand_met["tracer_loss_4h"])
            screen_table.append({"alpha": float(alpha), "loss_4h": c_loss, "tracer_loss_4h": c_tr})
            if math.isfinite(c_loss) and c_loss < best_cand_loss - 1e-7:
                best_cand_loss = c_loss
                best_alpha = float(alpha)

        if best_alpha > 0.0:
            base_model.set_flat_params(theta_center + best_alpha * delta_theta)
            accepted_rounds += 1
        else:
            base_model.set_flat_params(theta_center)

        val_metrics = evaluate_validation_4h_with_worst(
            env, base_model, val_windows, rank, world_size, device
        )
        val_4h_window_rollouts += len(val_windows)
        cumulative_window_rollouts = float(train_window_rollouts + val_4h_window_rollouts)

        upd = tracker.update(
            round_idx=r,
            val_metrics=val_metrics,
            candidate_state=base_model.state_dict(),
            cum_windows=cumulative_window_rollouts,
        )
        if not upd["finite"]:
            base_model.load_state_dict(tracker.selected_state)

        r_wall = time.time() - r_t0
        if is_main_process(rank):
            rec = {
                "round": r,
                "train_window_rollouts": train_window_rollouts,
                "val_4h_window_rollouts": val_4h_window_rollouts,
                "feasibility_window_equivalents": 0.0,
                "cumulative_window_rollouts": cumulative_window_rollouts,
                "mean_wet_col_kl": post_stats["mean_kl"],
                "max_frame_wet_col_kl": post_stats["max_frame_kl"],
                "clip_fraction": post_stats["clip_fraction"],
                "ratio_min": post_stats["ratio_min"],
                "ratio_max": post_stats["ratio_max"],
                "kl_shrink_factor": shrink_used,
                "kl_early_stopped": kl_early_stopped,
                "nonfinite_attempts": nonfinite_attempts,
                "nonfinite_failures": nonfinite_failures,
                "spatial_decomp_max_err": round_decomp_err,
                "explore_pre_proj_rms": pre_proj_rms,
                "explore_post_proj_rms": post_proj_rms,
                "wet_col_return_std": ret_std_val,
                "wet_col_raw_adv_std": raw_adv_std_val,
                "proposed_param_delta_norm": prop_param_delta_norm,
                "accepted_param_delta_norm": float(best_alpha) * prop_param_delta_norm,
                "critic_loss_first": critic_losses[0] if critic_losses else 0.0,
                "critic_loss_last": critic_losses[-1] if critic_losses else 0.0,
                "actor_head_grad_norm_max": max(head_grad_norms) if head_grad_norms else 0.0,
                "actor_body_grad_norm_max": max(body_grad_norms) if body_grad_norms else 0.0,
                "center_train_loss_4h": center_loss_4h,
                "explore_train_loss_4h": mean_exp_loss_4h,
                "accepted_alpha": best_alpha,
                "accepted_train_loss_4h": best_cand_loss,
                "candidate_screen": screen_table,
                "val_loss_4h": upd["val_loss_4h"],
                "val_tracer_loss_4h": val_metrics["tracer_loss_4h"],
                "val_momentum_loss_4h": val_metrics["momentum_loss_4h"],
                "val_composite_nrmse_4h": val_metrics["composite_nrmse_4h"],
                "val_worst_window_loss_4h": val_metrics["worst_window_loss_4h"],
                "selected_round_so_far": tracker.selected_round,
                "selected_val_4h_so_far": tracker.selected_val_4h,
                "no_improve": tracker.no_improve,
                "round_wall_seconds": r_wall,
            }
            append_jsonl(stage_dir / "metrics.jsonl", rec)
            print(
                f"  [Stage 7 Closure-RL r{r:03d}] col_kl={post_stats['mean_kl']:.5f} "
                f"(max_f={post_stats['max_frame_kl']:.4f}) | clip={post_stats['clip_fraction']:.3f} | "
                f"sh={shrink_used:.2f} | alpha={best_alpha:.2f} | "
                f"train_4h={center_loss_4h:.5f}->{best_cand_loss:.5f} | "
                f"val_4h={upd['val_loss_4h']:.6f} (best={tracker.selected_val_4h:.6f} @ r{tracker.selected_round}) | "
                f"no_imp={tracker.no_improve} | cum_win={cumulative_window_rollouts:.0f} | {r_wall:.1f}s",
                flush=True,
            )

        if tracker.no_improve >= int(cfg.baseline_patience):
            stop_reason = f"early_stop_patience_{cfg.baseline_patience}"
            break

    base_model.load_state_dict(tracker.selected_state)
    if is_main_process(rank):
        torch.save(critic.state_dict(), critic_ckpt)

    val_4h = evaluate_validation_4h_with_worst(
        env, base_model, val_windows, rank, world_size, device
    )
    post_selection_eval_window_equivalents = float(len(val_windows))
    cumulative_window_rollouts = float(train_window_rollouts + val_4h_window_rollouts)
    total_stage_window_equivalents = (
        cumulative_window_rollouts + post_selection_eval_window_equivalents
    )

    fin_meta = tracker.finalize(cumulative_window_rollouts)
    s_cnt_end = get_ddp_solver_window_equivalents(env, device, forbid_whitebox=True)
    solver_verified_win = float(s_cnt_end["window_equivalents"] - s_cnt_start["window_equivalents"])
    if abs(solver_verified_win - total_stage_window_equivalents) > 1e-6:
        raise RuntimeError(
            f"[stage7_closure_rl] Ledger vs solver substep mismatch: ledger_total={total_stage_window_equivalents} "
            f"vs solver_verified={solver_verified_win}"
        )

    summary = {
        "stage": "stage7_closure_rl",
        "method": "closure_rl",
        "protocol_mode": "pure_4h",
        "status": "completed",
        "stop_reason": stop_reason,
        "seed": cfg.seed,
        "source_version": cfg.source_version,
        "source_sha256": cfg.source_sha256,
        "source_tree_sha256": cfg.source_tree_sha256,
        "effective_config_sha256": cfg.effective_config_sha256,
        "source_bundle_sha256": cfg.source_bundle_sha256,
        "common_ckpt": relativize_artifact_path(zero_ckpt, out_root_path),
        "common_sha256": sha256_file(zero_ckpt),
        "best_model_path": relativize_artifact_path(best_ckpt, out_root_path),
        "best_model_sha256": fin_meta["best_model_sha256"],
        "last_round_model_path": relativize_artifact_path(tracker.last_round_ckpt, out_root_path),
        "last_round_model_sha256": fin_meta["last_round_model_sha256"],
        "critic_parameters": int(sum(p.numel() for p in critic.parameters())),
        "active_wet_columns": int(col_wet.sum().item()),
        "last_completed_round": tracker.last_completed_round,
        "accepted_rounds": accepted_rounds,
        "selected_round": tracker.selected_round,
        "selected_val_4h": tracker.selected_val_4h,
        "total_nonfinite_attempts": total_nonfinite_attempts,
        "total_nonfinite_failures": total_nonfinite_failures,
        "total_shrunk_rounds": total_shrunk_rounds,
        "max_spatial_decomp_error": max_spatial_decomp_error,
        "train_window_rollouts": train_window_rollouts,
        "val_4h_window_rollouts": val_4h_window_rollouts,
        "feasibility_window_equivalents": 0.0,
        "cumulative_window_rollouts": cumulative_window_rollouts,
        "post_selection_eval_window_equivalents": post_selection_eval_window_equivalents,
        "total_stage_window_equivalents": total_stage_window_equivalents,
        "solver_verified_window_equivalents": solver_verified_win,
        "anchor_window_budget": float(cfg.anchor_window_budget),
        "budget_checkpoints": fin_meta["milestones"],
        "wall_seconds": time.time() - t0,
        "val_4h": val_4h,
    }
    if is_main_process(rank):
        save_atomic_json(summary_path, summary)
        print(
            f"[stage7_closure_rl Done] selected_round={tracker.selected_round}, "
            f"val_4h={val_4h['loss_4h']:.6f}, cum_win={cumulative_window_rollouts:.0f}, "
            f"wall={summary['wall_seconds']:.1f}s",
            flush=True,
        )
    if dist.is_initialized():
        dist.barrier()
    return summary


run_stage7_closure_rl = run_closure_rl_baseline

__all__ = [
    "WetColumnSpatialCritic3D",
    "compute_wet_column_ppo_stats",
    "evaluate_closure_rl_batch_stats_ddp",
    "run_closure_rl_baseline",
    "run_stage7_closure_rl",
]
