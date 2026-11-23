"""Stage 5 Clipped Proximal Policy Optimization (PPO) black-box baseline for 3D POM."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
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
    evaluate_validation_4h_with_worst,
)
from ..network import ForgePOMNet
from ..packing import build_wet_masks_60_61
from ..physics_env import BlackBoxProvider3D
from ..sampling import ensure_common_zero_ckpt, sample_epoch_windows


def compute_joint_gaussian_ppo_stats(
    a_sampled: torch.Tensor,
    mu_old: torch.Tensor,
    mu_new: torch.Tensor,
    mask_60: torch.Tensor,
    std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact joint Gaussian log-probability ratio and KL over active wet coordinates only."""
    inv_var = 1.0 / max(float(std) * float(std), 1e-30)
    mask_d = mask_60.double()
    delta = (mu_new.double() - mu_old.double()) * mask_d
    noise = (a_sampled.double() - mu_old.double()) * mask_d
    delta_sq = delta.square().sum()
    log_ratio = inv_var * (noise * delta).sum() - 0.5 * inv_var * delta_sq
    kl = 0.5 * inv_var * delta_sq
    return log_ratio, kl


def run_ppo_baseline(
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
    """Run Pure-4h joint-Gaussian Clipped PPO baseline (`stage5_ppo`)."""
    out_root_path = Path(cfg.output_root)
    stage_dir = out_root_path / "stage5_ppo"
    summary_path = stage_dir / "summary.json"
    best_ckpt = stage_dir / "best_model.pt"
    if summary_path.exists() and best_ckpt.exists():
        if is_main_process(rank):
            print("[Stage 5 PPO] Already completed, loading summary.json", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    t0 = time.time()
    zero_ckpt = ensure_common_zero_ckpt(cfg, rank, device, explicit_ckpt=common_ckpt)
    s_cnt_start = get_ddp_solver_window_equivalents(env, device, forbid_whitebox=True)
    base_model = ForgePOMNet(
        cfg.in_channels, cfg.hidden_channels, cfg.out_channels, cfg.num_blocks
    ).to(device)
    base_model.load_state_dict(torch.load(zero_ckpt, map_location=device, weights_only=True))
    ddp_model = DDP(base_model, device_ids=[device.index]) if dist.is_initialized() else base_model
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=cfg.ppo_lr, weight_decay=1e-4)
    mask_60, _ = build_wet_masks_60_61(env)

    val_init = evaluate_validation_4h_with_worst(
        env, base_model, val_windows, rank, world_size, device
    )
    train_window_rollouts = 0
    val_4h_window_rollouts = len(val_windows)

    tracker = Pure4hSelectionTracker(
        stage_dir=stage_dir,
        rank=rank,
        init_state=base_model.state_dict(),
        init_val_metrics=val_init,
        init_cum_windows=float(val_4h_window_rollouts),
        anchor_window_budget=cfg.anchor_window_budget,
        improve_tol=cfg.val_improve_tol,
    )

    total_nonfinite_attempts = 0
    total_nonfinite_failures = 0
    total_rolled_back_epochs = 0
    stop_reason = "completed_max_rounds"

    std = cfg.ppo_explore_std
    target_kl = cfg.ppo_target_kl
    clip_eps = cfg.ppo_clip_eps
    batch_per_round = 4 * max(world_size, 1)
    round_cost = float(2 * batch_per_round + len(val_windows))
    post_eval_reserve = float(len(val_windows))
    max_rounds = int(cfg.baseline_rounds)

    if is_main_process(rank):
        print(
            f"[Stage 5 PPO] Starting pure-4h training from common_zero.pt "
            f"(val_4h={tracker.init_val_4h:.6f}, max_window_budget={max_window_budget}, "
            f"round_win={round_cost:.0f})...",
            flush=True,
        )

    @torch.no_grad()
    def _eval_ppo_batch_stats_ddp(
        obs_b: torch.Tensor,
        act_b: torch.Tensor,
        old_b: torch.Tensor,
        adv_list: list[float],
    ) -> dict[str, float]:
        was_tr = base_model.training
        base_model.eval()
        mu_new_b = base_model(obs_b)
        n_b = int(obs_b.shape[0])
        local_nf = 0.0
        kls: list[float] = []
        clips: list[float] = []
        r_mins: list[float] = []
        r_maxs: list[float] = []
        unclipped_max_adv = 0.0
        for idx_i in range(n_b):
            lr_val, kl_v = compute_joint_gaussian_ppo_stats(
                act_b[idx_i], old_b[idx_i], mu_new_b[idx_i], mask_60, std
            )
            rat = torch.exp(lr_val)
            if (
                not bool(torch.isfinite(lr_val).item())
                or not bool(torch.isfinite(rat).item())
                or not bool(torch.isfinite(kl_v).item())
            ):
                local_nf = 1.0
                break
            rv = float(rat.item())
            av = float(adv_list[idx_i])
            is_clipped = (av > 0.0 and rv > 1.0 + clip_eps) or (av < 0.0 and rv < 1.0 - clip_eps)
            if not is_clipped:
                unclipped_max_adv = max(unclipped_max_adv, abs(av))
            kls.append(float(kl_v.item()))
            clips.append(1.0 if is_clipped else 0.0)
            r_mins.append(rv)
            r_maxs.append(rv)
        if was_tr:
            base_model.train()

        mean_sync = ddp_mean_dict(
            {
                "nf": local_nf,
                "kl": sum(kls) / max(len(kls), 1),
                "clip": sum(clips) / max(len(clips), 1),
            },
            device,
        )
        rmin_val = min(r_mins) if (r_mins and local_nf == 0.0) else float("inf")
        rmax_val = max(r_maxs) if (r_maxs and local_nf == 0.0) else 0.0
        ext_t = torch.tensor(
            [-rmin_val, rmax_val, unclipped_max_adv],
            device=device,
            dtype=torch.float64,
        )
        if dist.is_initialized():
            dist.all_reduce(ext_t, op=dist.ReduceOp.MAX)
        return {
            "nf": mean_sync["nf"],
            "kl": mean_sync["kl"],
            "clip": mean_sync["clip"],
            "rmin": -float(ext_t[0].item()),
            "rmax": float(ext_t[1].item()),
            "unclipped_max_adv": float(ext_t[2].item()),
        }

    for r in range(1, max_rounds + 1):
        curr_cum = float(train_window_rollouts + val_4h_window_rollouts)
        allowed, _ = can_afford_next_round(
            curr_cum + post_eval_reserve, round_cost, max_window_budget
        )
        if not allowed:
            stop_reason = "reached_max_window_budget_cap"
            break

        r_t0 = time.time()
        base_model.eval()
        round_wins = sample_epoch_windows(
            train_pool,
            round_idx=r,
            batch_size=batch_per_round,
            seed=cfg.seed,
            stream_id=5051,
        )
        my_windows = round_wins[rank::world_size]

        obs_buffer: list[torch.Tensor] = []
        act_buffer: list[torch.Tensor] = []
        old_mean_buffer: list[torch.Tensor] = []
        adv_buffer: list[float] = []

        with torch.no_grad():
            for w_idx, (m, d) in enumerate(my_windows):
                base_loss, _, _ = env.rollout_4h_window(m, d, lambda x_k, _k: base_model(x_k))
                env.reset(m, d)
                xs_w: list[torch.Tensor] = []
                acts_w: list[torch.Tensor] = []
                means_w: list[torch.Tensor] = []
                segs_w: list[dict[str, torch.Tensor]] = []
                for k in range(cfg.hours_per_window):
                    x_k = env.observe_F()
                    mu_k = base_model(x_k)
                    gen_n = torch.Generator(device="cpu").manual_seed(
                        cfg.seed + r * 9001 + rank * 131 + w_idx * 17 + k
                    )
                    noise = (
                        std
                        * torch.randn(mu_k.shape, generator=gen_n, dtype=torch.float32).to(
                            device=device, dtype=mu_k.dtype
                        )
                        * mask_60
                    )
                    a_k = mu_k + noise
                    segs_w.extend(env.step_H(a_k))
                    xs_w.append(x_k.detach())
                    acts_w.append(a_k.detach())
                    means_w.append(mu_k.detach())
                pert_loss, _ = env.compute_12seg_loss_and_metrics(m, d, segs_w)
                adv = float(base_loss - float(pert_loss.item()))
                for k in range(cfg.hours_per_window):
                    obs_buffer.append(xs_w[k])
                    act_buffer.append(acts_w[k])
                    old_mean_buffer.append(means_w[k])
                    adv_buffer.append(adv)
        train_window_rollouts += 2 * len(my_windows) * world_size

        adv_t = torch.tensor(adv_buffer, device=device, dtype=torch.float64)
        adv_mean = adv_t.mean()
        adv_sq = adv_t.square().mean()
        if dist.is_initialized():
            dist.all_reduce(adv_mean, op=dist.ReduceOp.AVG)
            dist.all_reduce(adv_sq, op=dist.ReduceOp.AVG)
        adv_std = (adv_sq - adv_mean.square()).clamp_min(1e-12).sqrt()
        adv_norm = ((adv_t - adv_mean) / adv_std).tolist()

        obs_batch = torch.stack(obs_buffer, dim=0)
        act_batch = torch.stack(act_buffer, dim=0)
        old_batch = torch.stack(old_mean_buffer, dim=0)

        round_kl = 0.0
        round_clip_frac = 0.0
        round_ratio_min = 1.0
        round_ratio_max = 1.0
        round_last_shrink = 1.0
        rolled_back_epochs = 0
        nonfinite_attempts = 0
        nonfinite_failures = 0

        for _ppo_epoch in range(cfg.ppo_epochs):
            pre_stats = _eval_ppo_batch_stats_ddp(obs_batch, act_batch, old_batch, adv_norm)
            if pre_stats["nf"] > 0.0:
                nonfinite_attempts += 1
                nonfinite_failures += 1
                total_nonfinite_attempts += 1
                total_nonfinite_failures += 1
                raise FloatingPointError(
                    f"Non-finite pre-step PPO statistics at round {r}, epoch {_ppo_epoch}"
                )
            pre_kl = pre_stats["kl"]
            if pre_kl > target_kl:
                break
            if pre_stats["unclipped_max_adv"] == 0.0:
                round_kl = pre_kl
                round_clip_frac = pre_stats["clip"]
                round_ratio_min = pre_stats["rmin"]
                round_ratio_max = pre_stats["rmax"]
                break

            prev_theta = base_model.get_flat_params()
            prev_opt_state = copy.deepcopy(optimizer.state_dict())

            ddp_model.train()
            optimizer.zero_grad(set_to_none=True)
            mu_new_batch = ddp_model(obs_batch)
            ppo_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
            n_b = int(obs_batch.shape[0])
            for idx_b in range(n_b):
                log_ratio, _ = compute_joint_gaussian_ppo_stats(
                    act_batch[idx_b],
                    old_batch[idx_b],
                    mu_new_batch[idx_b],
                    mask_60,
                    std,
                )
                ratio = torch.exp(log_ratio)
                adv_b = float(adv_norm[idx_b])
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
                ppo_loss = ppo_loss + (-torch.minimum(surr1, surr2)) / float(n_b)

            ppo_loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()

            post_stats = _eval_ppo_batch_stats_ddp(obs_batch, act_batch, old_batch, adv_norm)
            if post_stats["nf"] > 0.0:
                nonfinite_attempts += 1
            shrink_used = 1.0
            if post_stats["nf"] > 0.0 or post_stats["kl"] > target_kl:
                if pre_kl >= 0.75 * target_kl:
                    base_model.set_flat_params(prev_theta)
                    optimizer.load_state_dict(prev_opt_state)
                    rolled_back_epochs += 1
                    break
                full_theta = base_model.get_flat_params()
                delta_theta = full_theta - prev_theta
                alpha_sh = (
                    min(
                        0.5,
                        math.sqrt(
                            max(0.5 * target_kl - pre_kl, 0.1 * target_kl)
                            / max(post_stats["kl"], target_kl)
                        ),
                    )
                    if (
                        post_stats["nf"] == 0.0
                        and math.isfinite(post_stats["kl"])
                        and post_stats["kl"] > 0.0
                    )
                    else 0.5
                )
                accepted_shrunk = False
                for _ in range(8):
                    base_model.set_flat_params(prev_theta + float(alpha_sh) * delta_theta)
                    post_stats = _eval_ppo_batch_stats_ddp(
                        obs_batch, act_batch, old_batch, adv_norm
                    )
                    if post_stats["nf"] > 0.0:
                        nonfinite_attempts += 1
                    if post_stats["nf"] == 0.0 and post_stats["kl"] <= target_kl:
                        shrink_used = float(alpha_sh)
                        accepted_shrunk = True
                        break
                    alpha_sh *= 0.5
                if not accepted_shrunk:
                    base_model.set_flat_params(prev_theta)
                    optimizer.load_state_dict(prev_opt_state)
                    if post_stats["nf"] > 0.0:
                        nonfinite_failures += 1
                        total_nonfinite_attempts += nonfinite_attempts
                        total_nonfinite_failures += nonfinite_failures
                        raise FloatingPointError(
                            f"Persistent non-finite post-step PPO statistics at round {r}, epoch {_ppo_epoch}"
                        )
                    rolled_back_epochs += 1
                    break

            round_kl = post_stats["kl"]
            round_clip_frac = post_stats["clip"]
            round_ratio_min = post_stats["rmin"]
            round_ratio_max = post_stats["rmax"]
            round_last_shrink = shrink_used

        total_nonfinite_attempts += nonfinite_attempts
        total_nonfinite_failures += nonfinite_failures
        total_rolled_back_epochs += rolled_back_epochs

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
                "mean_joint_kl": round_kl,
                "clip_fraction": round_clip_frac,
                "ratio_min": round_ratio_min,
                "ratio_max": round_ratio_max,
                "last_shrink_factor": round_last_shrink,
                "rolled_back_epochs": rolled_back_epochs,
                "nonfinite_attempts": nonfinite_attempts,
                "nonfinite_failures": nonfinite_failures,
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
                f"  [Stage 5 PPO r{r:03d}] joint_kl={round_kl:.4f} | clip={round_clip_frac:.2f} | "
                f"sh={round_last_shrink:.2f} | rb={rolled_back_epochs} | nf_att={nonfinite_attempts} "
                f"| val_4h={upd['val_loss_4h']:.6f} (best={tracker.selected_val_4h:.6f} @ r{tracker.selected_round}) "
                f"| no_imp={tracker.no_improve} | cum_win={cumulative_window_rollouts:.0f} | {r_wall:.1f}s",
                flush=True,
            )

        if tracker.no_improve >= int(cfg.baseline_patience):
            stop_reason = f"early_stop_patience_{cfg.baseline_patience}"
            break

    base_model.load_state_dict(tracker.selected_state)
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
            f"[stage5_ppo] Ledger vs solver substep mismatch: ledger_total={total_stage_window_equivalents} "
            f"vs solver_verified={solver_verified_win}"
        )

    summary = {
        "stage": "stage5_ppo",
        "method": "ppo",
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
        "last_completed_round": tracker.last_completed_round,
        "selected_round": tracker.selected_round,
        "selected_val_4h": tracker.selected_val_4h,
        "total_nonfinite_attempts": total_nonfinite_attempts,
        "total_nonfinite_failures": total_nonfinite_failures,
        "total_rolled_back_epochs": total_rolled_back_epochs,
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
            f"[stage5_ppo Done] selected_round={tracker.selected_round}, "
            f"val_4h={val_4h['loss_4h']:.6f}, cum_win={cumulative_window_rollouts:.0f}, "
            f"wall={summary['wall_seconds']:.1f}s",
            flush=True,
        )
    if dist.is_initialized():
        dist.barrier()
    return summary


__all__ = ["compute_joint_gaussian_ppo_stats", "run_ppo_baseline"]
