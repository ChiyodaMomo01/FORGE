"""Stage 4 Ensemble Kalman Inversion (EKI) black-box baseline for 3D POM."""

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
from ..packing import (
    build_wet_masks_60_61,
    pack_reference_segments,
    pack_segments_to_tensor,
)
from ..physics_env import BlackBoxProvider3D
from ..probes import measure_action_jacobian_ranks
from ..sampling import ensure_common_zero_ckpt, sample_epoch_windows


@torch.no_grad()
def compute_streamed_eki_ensemble_update(
    xi_ensemble: torch.Tensor,
    residual_chunks: list[torch.Tensor],
    eki_reg: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute exact subspace EKI update from streamed signed full-field residual chunks."""
    n_ens = int(xi_ensemble.shape[0])
    denom_ens = math.sqrt(float(max(n_ens - 1, 1)))
    device = xi_ensemble.device

    xi_d = xi_ensemble.double()
    xi_bar = xi_d.mean(dim=0, keepdim=True)
    d_xi = (xi_d - xi_bar) / denom_ens

    gram_ens = torch.zeros((n_ens, n_ens), device=device, dtype=torch.float64)
    cross_rhs = torch.zeros(n_ens, device=device, dtype=torch.float64)
    mean_sq_norm = 0.0

    for chunk in residual_chunks:
        yc = chunk.to(device=device, dtype=torch.float64)
        y_bar_c = yc.mean(dim=0, keepdim=True)
        dy_c = (yc - y_bar_c) / denom_ens
        gram_ens = gram_ens + (dy_c @ dy_c.T)
        cross_rhs = cross_rhs + (dy_c @ (-y_bar_c.squeeze(0)))
        mean_sq_norm += float(y_bar_c.square().sum().item())

    reg = max(float(eki_reg), 1e-8)
    sys_mat = gram_ens + reg * torch.eye(n_ens, device=device, dtype=torch.float64)
    weights_ens = torch.linalg.solve(sys_mat, cross_rhs)
    delta_xi = (d_xi.T @ weights_ens).to(dtype=xi_ensemble.dtype)

    evals = torch.linalg.eigvalsh(gram_ens).clamp_min(0.0)
    eff_rank = int((evals > max(float(evals.max().item()) * 1e-6, 1e-12)).sum().item())
    return delta_xi, {
        "ensemble_gram_trace": float(gram_ens.diag().sum().item()),
        "ensemble_gram_max_eig": float(evals.max().item()),
        "ensemble_obs_effective_rank": float(eff_rank),
        "ensemble_mean_residual_sq": mean_sq_norm,
    }


def run_eki_baseline(
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
    """Run Pure-4h Ensemble Kalman Inversion baseline (`stage4_eki`)."""
    out_root_path = Path(cfg.output_root)
    stage_dir = out_root_path / "stage4_eki"
    summary_path = stage_dir / "summary.json"
    best_ckpt = stage_dir / "best_model.pt"
    if summary_path.exists() and best_ckpt.exists():
        if is_main_process(rank):
            print("[Stage 4 EKI] Already completed, loading summary.json", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    t0 = time.time()
    zero_ckpt = ensure_common_zero_ckpt(cfg, rank, device, explicit_ckpt=common_ckpt)
    s_cnt_start = get_ddp_solver_window_equivalents(env, device, forbid_whitebox=True)
    model = ForgePOMNet(cfg.in_channels, cfg.hidden_channels, cfg.out_channels, cfg.num_blocks).to(
        device
    )
    model.load_state_dict(torch.load(zero_ckpt, map_location=device, weights_only=True))
    mask_60, mask_61 = build_wet_masks_60_61(env)
    wet_61_count = float(mask_61.double().sum().clamp_min(1.0).item())

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
    local_ens_per_rank = max(cfg.eki_ensemble_size // max(world_size, 1), 1)
    round_worst_cost = float(
        n_shared
        + local_ens_per_rank * n_shared * world_size
        + len(cfg.eki_step_radii) * n_shared
        + len(val_windows)
    )
    post_eval_reserve = float(len(val_windows))
    max_rounds = int(cfg.baseline_rounds)
    stop_reason = "completed_max_rounds"
    accepted_rounds = 0

    if is_main_process(rank):
        print(
            f"[Stage 4 EKI] Starting pure-4h training from common_zero.pt "
            f"(val_4h={tracker.init_val_4h:.6f}, M={m_sub}, N_ens={local_ens_per_rank * world_size}, "
            f"max_window_budget={max_window_budget}, worst_round_win={round_worst_cost:.0f})...",
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
            stream_id=3037,
        )
        ref_by_win = [pack_reference_segments(env, m, d) for m, d in shared_windows]

        directions = model.build_mixed_head_body_basis(
            num_directions=m_sub, seed=cfg.seed + r * 313
        )
        basis_mat = torch.stack(directions, dim=1)

        theta_mean = model.get_flat_params()
        center_eval_res = evaluate_shared_windows_ddp(
            env, model, shared_windows, rank, world_size, device, capture_obs=True
        )
        assert len(center_eval_res) == 3
        center_eval, c_cost, sample_obs = center_eval_res
        train_window_rollouts += c_cost
        center_loss = float(center_eval["loss_4h"])
        rank_info = measure_action_jacobian_ranks(
            model, directions, sample_obs, mask_60, eps=cfg.eki_sigma
        )

        norm_denom = math.sqrt(float(len(shared_windows) * 12) * wet_61_count)
        my_coords: list[torch.Tensor] = []
        my_res_by_win: list[list[torch.Tensor]] = [[] for _ in range(len(shared_windows))]

        for local_e in range(local_ens_per_rank):
            e_id = rank * local_ens_per_rank + local_e
            gen = torch.Generator(device="cpu").manual_seed(cfg.seed + r * 8191 + e_id)
            xi_e = float(cfg.eki_sigma) * torch.randn(m_sub, dtype=torch.float32, generator=gen).to(
                device=device, dtype=theta_mean.dtype
            )
            model.set_flat_params(theta_mean + basis_mat @ xi_e)
            my_coords.append(xi_e)
            for w_i, (m, d) in enumerate(shared_windows):
                _, _, segs_e = env.rollout_4h_window(m, d, lambda x_k, _k: model(x_k))
                packed_e = pack_segments_to_tensor(env, segs_e)
                signed_flat = torch.cat(
                    [(packed_e[j] - ref_by_win[w_i][j]).reshape(-1) for j in range(12)],
                    dim=0,
                ) / float(norm_denom)
                my_res_by_win[w_i].append(signed_flat.cpu())

        train_window_rollouts += local_ens_per_rank * len(shared_windows) * world_size

        coords_local = torch.stack(my_coords, dim=0)
        if dist.is_initialized():
            coords_all = [torch.zeros_like(coords_local) for _ in range(world_size)]
            dist.all_gather(coords_all, coords_local)
            xi_mat = torch.cat(coords_all, dim=0)
        else:
            xi_mat = coords_local

        gathered_chunks: list[torch.Tensor] = []
        for w_i in range(len(shared_windows)):
            chunk_local = torch.stack(my_res_by_win[w_i], dim=0).to(device)
            if dist.is_initialized():
                chunk_all = [torch.zeros_like(chunk_local) for _ in range(world_size)]
                dist.all_gather(chunk_all, chunk_local)
                gathered_chunks.append(torch.cat(chunk_all, dim=0))
            else:
                gathered_chunks.append(chunk_local)

        delta_xi, eki_stats = compute_streamed_eki_ensemble_update(
            xi_mat, gathered_chunks, eki_reg=cfg.eki_reg
        )
        del gathered_chunks
        delta_theta = basis_mat @ delta_xi
        step_norm = float(delta_xi.double().norm().item())

        best_cand_loss = center_loss
        accepted_radius = 0.0
        best_theta = theta_mean
        if math.isfinite(step_norm) and step_norm > 0.0:
            for radius in cfg.eki_step_radii:
                cand_theta = theta_mean + float(radius) * delta_theta
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
                "coordinate_step_norm_before_linesearch": step_norm,
                "eki_stats": eki_stats,
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
                f"  [Stage 4 EKI r{r:03d}] rank=({rank_info['action_jac_rank_full']},{rank_info['action_jac_rank_body_only']}) "
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
            f"[stage4_eki] Ledger vs solver substep mismatch: ledger_total={total_stage_window_equivalents} "
            f"vs solver_verified={solver_verified_win}"
        )

    summary = {
        "stage": "stage4_eki",
        "method": "eki",
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
            f"[stage4_eki Done] selected_round={tracker.selected_round}, "
            f"val_4h={val_4h['loss_4h']:.6f}, cum_win={cumulative_window_rollouts:.0f}, "
            f"wall={summary['wall_seconds']:.1f}s",
            flush=True,
        )
    if dist.is_initialized():
        dist.barrier()
    return summary


__all__ = ["compute_streamed_eki_ensemble_update", "run_eki_baseline"]
