"""Stage 3 antithetic Evolution Strategies (ES) black-box baseline for 3D POM."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ..artifacts import (
    append_jsonl,
    relativize_artifact_path,
    save_atomic_json,
    sha256_file,
)
from ..config import ForgePOM3DConfig
from ..distributed import get_ddp_solver_window_equivalents, is_main_process
from ..evaluation import (
    Pure4hSelectionTracker,
    can_afford_next_round,
    evaluate_shared_windows_ddp,
    evaluate_validation_4h_with_worst,
)
from ..network import ForgePOMNet
from ..packing import build_wet_masks_60_61
from ..physics_env import BlackBoxProvider3D
from ..probes import measure_action_jacobian_ranks
from ..sampling import ensure_common_zero_ckpt, sample_epoch_windows


def run_es_baseline(
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
    """Run Pure-4h antithetic Evolution Strategies baseline (`stage3_es`)."""
    out_root_path = Path(cfg.output_root)
    stage_dir = out_root_path / "stage3_es"
    summary_path = stage_dir / "summary.json"
    best_ckpt = stage_dir / "best_model.pt"
    if summary_path.exists() and best_ckpt.exists():
        if is_main_process(rank):
            print("[Stage 3 ES] Already completed, loading summary.json", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    t0 = time.time()
    zero_ckpt = ensure_common_zero_ckpt(cfg, rank, device, explicit_ckpt=common_ckpt)
    s_cnt_start = get_ddp_solver_window_equivalents(env, device, forbid_whitebox=True)
    model = ForgePOMNet(cfg.in_channels, cfg.hidden_channels, cfg.out_channels, cfg.num_blocks).to(
        device
    )
    model.load_state_dict(torch.load(zero_ckpt, map_location=device, weights_only=True))
    mask_60, _ = build_wet_masks_60_61(env)

    val_init = evaluate_validation_4h_with_worst(env, model, val_windows, rank, world_size, device)
    train_window_rollouts = 0
    val_4h_window_rollouts = len(val_windows)

    tracker = Pure4hSelectionTracker(
        stage_dir=stage_dir,
        rank=rank,
        init_state=model.state_dict(),
        init_val_metrics=val_init,
        init_cum_windows=float(val_4h_window_rollouts),
        anchor_window_budget=cfg.anchor_window_budget,
        improve_tol=cfg.val_improve_tol,
    )

    m_sub = int(cfg.es_eki_num_directions)
    n_shared = 4
    round_worst_cost = float(
        n_shared + 2 * n_shared * world_size + len(cfg.es_step_radii) * n_shared + len(val_windows)
    )
    post_eval_reserve = float(len(val_windows))
    max_rounds = int(cfg.baseline_rounds)
    stop_reason = "completed_max_rounds"
    accepted_rounds = 0

    if is_main_process(rank):
        print(
            f"[Stage 3 ES] Starting pure-4h training from common_zero.pt "
            f"(val_4h={tracker.init_val_4h:.6f}, M={m_sub}, max_window_budget={max_window_budget}, "
            f"worst_round_win={round_worst_cost:.0f})...",
            flush=True,
        )

    for r in range(1, max_rounds + 1):
        curr_cum = float(train_window_rollouts + val_4h_window_rollouts)
        allowed, _ = can_afford_next_round(
            curr_cum + post_eval_reserve, round_worst_cost, max_window_budget
        )
        if not allowed:
            stop_reason = "reached_max_window_budget_cap"
            break

        r_t0 = time.time()
        shared_windows = sample_epoch_windows(
            train_pool,
            round_idx=r,
            batch_size=n_shared,
            seed=cfg.seed,
            stream_id=2027,
        )

        directions = model.build_mixed_head_body_basis(
            num_directions=m_sub, seed=cfg.seed + r * 313
        )
        basis_mat = torch.stack(directions, dim=1)

        theta_center = model.get_flat_params()
        center_eval_res = evaluate_shared_windows_ddp(
            env, model, shared_windows, rank, world_size, device, capture_obs=True
        )
        assert len(center_eval_res) == 3
        center_eval, c_cost, sample_obs = center_eval_res
        train_window_rollouts += c_cost
        center_loss = float(center_eval["loss_4h"])
        rank_info = measure_action_jacobian_ranks(
            model, directions, sample_obs, mask_60, eps=cfg.es_sigma
        )

        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + r * 4099 + rank)
        z_noise = torch.randn(m_sub, dtype=torch.float32, generator=gen).to(
            device=device, dtype=theta_center.dtype
        )
        eps_vec = basis_mat @ z_noise

        model.set_flat_params(theta_center + cfg.es_sigma * eps_vec)
        loss_pos = sum(
            env.rollout_4h_window(m, d, lambda x_k, _k: model(x_k))[0] for m, d in shared_windows
        ) / len(shared_windows)

        model.set_flat_params(theta_center - cfg.es_sigma * eps_vec)
        loss_neg = sum(
            env.rollout_4h_window(m, d, lambda x_k, _k: model(x_k))[0] for m, d in shared_windows
        ) / len(shared_windows)
        train_window_rollouts += 2 * len(shared_windows) * world_size

        grad_sub = ((loss_pos - loss_neg) / (2.0 * cfg.es_sigma)) * z_noise.double()
        if dist.is_initialized():
            dist.all_reduce(grad_sub, op=dist.ReduceOp.AVG)

        grad_norm = float(grad_sub.norm().item())
        direction = (-grad_sub / max(grad_norm, 1e-30)).to(dtype=theta_center.dtype)
        step_vec = basis_mat @ direction

        best_cand_loss = center_loss
        accepted_radius = 0.0
        best_theta = theta_center
        if math.isfinite(grad_norm) and grad_norm > 0.0:
            for radius in cfg.es_step_radii:
                cand_theta = theta_center + float(radius) * step_vec
                model.set_flat_params(cand_theta)
                cand_res = evaluate_shared_windows_ddp(
                    env, model, shared_windows, rank, world_size, device
                )
                assert len(cand_res) == 2
                cand_eval, r_cost = cand_res
                train_window_rollouts += r_cost
                c_loss = float(cand_eval["loss_4h"])
                if math.isfinite(c_loss) and c_loss < best_cand_loss - 1e-7:
                    best_cand_loss = c_loss
                    accepted_radius = float(radius)
                    best_theta = cand_theta.clone()

        if accepted_radius > 0.0:
            accepted_rounds += 1
        model.set_flat_params(best_theta)
        val_metrics = evaluate_validation_4h_with_worst(
            env, model, val_windows, rank, world_size, device
        )
        val_4h_window_rollouts += len(val_windows)
        cumulative_window_rollouts = float(train_window_rollouts + val_4h_window_rollouts)

        upd = tracker.update(
            round_idx=r,
            val_metrics=val_metrics,
            candidate_state=model.state_dict(),
            cum_windows=cumulative_window_rollouts,
        )
        if not upd["finite"]:
            model.load_state_dict(tracker.selected_state)

        r_wall = time.time() - r_t0
        if is_main_process(rank):
            rec = {
                "round": r,
                "action_jac_rank_full": rank_info["action_jac_rank_full"],
                "action_jac_rank_body_only": rank_info["action_jac_rank_body_only"],
                "center_train_loss": center_loss,
                "post_step_train_loss": best_cand_loss,
                "accepted_radius": accepted_radius,
                "grad_sub_norm": grad_norm,
                "train_window_rollouts": train_window_rollouts,
                "val_4h_window_rollouts": val_4h_window_rollouts,
                "feasibility_window_equivalents": 0.0,
                "cumulative_window_rollouts": cumulative_window_rollouts,
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
                f"  [Stage 3 ES r{r:03d}] rank=({rank_info['action_jac_rank_full']},{rank_info['action_jac_rank_body_only']}) "
                f"| rad={accepted_radius:.2f} | val_4h={upd['val_loss_4h']:.6f} "
                f"(best={tracker.selected_val_4h:.6f} @ r{tracker.selected_round}) | no_imp={tracker.no_improve} "
                f"| cum_win={cumulative_window_rollouts:.0f} | {r_wall:.1f}s",
                flush=True,
            )

        if tracker.no_improve >= int(cfg.baseline_patience):
            stop_reason = f"early_stop_patience_{cfg.baseline_patience}"
            break

    model.load_state_dict(tracker.selected_state)
    val_4h = evaluate_validation_4h_with_worst(env, model, val_windows, rank, world_size, device)
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
            f"[stage3_es] Ledger vs solver substep mismatch: ledger_total={total_stage_window_equivalents} "
            f"vs solver_verified={solver_verified_win}"
        )

    summary = {
        "stage": "stage3_es",
        "method": "es",
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
        "accepted_rounds": accepted_rounds,
        "selected_round": tracker.selected_round,
        "selected_val_4h": tracker.selected_val_4h,
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
            f"[stage3_es Done] selected_round={tracker.selected_round}, "
            f"val_4h={val_4h['loss_4h']:.6f}, cum_win={cumulative_window_rollouts:.0f}, "
            f"wall={summary['wall_seconds']:.1f}s",
            flush=True,
        )
    if dist.is_initialized():
        dist.barrier()
    return summary


__all__ = ["run_es_baseline"]
