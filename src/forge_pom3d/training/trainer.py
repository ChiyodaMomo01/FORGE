"""Orchestrator for two-stage FORGE 3D POM closed-loop training and single-factor ablations."""

from __future__ import annotations

import json
import time
from collections import deque
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
from ..config import VALID_ABLATION_VARIANTS, ForgePOM3DConfig
from ..distributed import get_ddp_solver_window_equivalents, is_main_process
from ..evaluation import (
    Pure4hSelectionTracker,
    can_afford_next_round,
    evaluate_validation_4h_with_worst,
)
from ..network import ForgePOMNet
from ..packing import (
    build_obs_wet_mask_161,
    build_volume_weights_60_61,
    build_wet_masks_60_61,
)
from ..physics_env import BlackBoxProvider3D
from ..probes import HADAMARD_4X4
from ..sampling import ensure_common_zero_ckpt, sample_epoch_windows
from .response_operator import (
    CausalBlockResponseOperator3D,
    CausalObservationResponseOperator3D,
)
from .stage1 import run_stage1_subspace_step
from .stage2 import run_stage2_response_step


def _compute_round_worst_cost(
    cfg: ForgePOM3DConfig,
    stage_one_only: bool,
    ablation: str,
    world_size: int,
    n_val_windows: int,
) -> tuple[float, int, int, int]:
    m_sub = int(cfg.forge_num_directions)
    n_base_per_gpu = int(cfg.forge_windows_per_gpu)
    n_extra_screen_per_gpu = max(int(cfg.forge_screen_windows_per_gpu) - n_base_per_gpu, 0)
    n_s2_candidates = 0 if stage_one_only else 5

    round_s1_cost = (
        1 + 2 * m_sub + len(cfg.forge_head_step_grid)
    ) * n_base_per_gpu * world_size + n_val_windows
    if stage_one_only:
        worst_cost = float(round_s1_cost)
    elif ablation == "no_stage1":
        worst_cost = float(
            (
                n_base_per_gpu
                + 10
                + n_extra_screen_per_gpu
                + n_s2_candidates * (n_base_per_gpu + n_extra_screen_per_gpu)
            )
            * world_size
            + n_val_windows
        )
    else:
        worst_cost = float(
            round_s1_cost
            + n_base_per_gpu * world_size
            + 10 * world_size
            + n_extra_screen_per_gpu * world_size
            + n_s2_candidates * (n_base_per_gpu + n_extra_screen_per_gpu) * world_size
        )
    return worst_cost, n_base_per_gpu, n_extra_screen_per_gpu, n_s2_candidates


def run_forge_main_4h(
    cfg: ForgePOM3DConfig,
    env: BlackBoxProvider3D,
    common_ckpt: Path,
    train_pool: list[tuple[int, int]],
    val_windows: list[tuple[int, int]],
    rank: int,
    world_size: int,
    device: torch.device,
    stage_one_only: bool = False,
    ablation_variant: str | None = None,
    max_window_budget: float | None = None,
) -> dict[str, Any]:
    """Run Pure-4h FORGE Main, Stage-I-Only, or single-factor Stage I/II ablations."""
    ablation = (
        str(ablation_variant if ablation_variant is not None else cfg.ablation_variant)
        .strip()
        .lower()
    )
    if ablation not in VALID_ABLATION_VARIANTS:
        raise ValueError(
            f"Invalid ablation_variant={ablation!r}; expected one of {VALID_ABLATION_VARIANTS}"
        )
    if stage_one_only and ablation != "none":
        raise ValueError(f"Cannot combine stage_one_only=True with ablation_variant={ablation!r}")
    cfg.ablation_variant = ablation

    if stage_one_only:
        stage_name = "stage2_forge_stage1_only"
    elif ablation == "none":
        stage_name = "stage2_forge_main"
    else:
        stage_name = f"stage2_forge_ablation_{ablation}"

    out_root_path = Path(cfg.output_root)
    stage_dir = out_root_path / stage_name
    summary_path = stage_dir / "summary.json"
    best_ckpt = stage_dir / "best_model.pt"
    if summary_path.exists() and best_ckpt.exists():
        if is_main_process(rank):
            print(
                f"[{stage_name}] Already completed, loading summary.json",
                flush=True,
            )
        return json.loads(summary_path.read_text(encoding="utf-8"))

    t0 = time.time()
    zero_ckpt = ensure_common_zero_ckpt(cfg, rank, device, explicit_ckpt=common_ckpt)
    s_cnt_start = get_ddp_solver_window_equivalents(env, device, forbid_whitebox=True)

    model = ForgePOMNet(cfg.in_channels, cfg.hidden_channels, cfg.out_channels, cfg.num_blocks).to(
        device
    )
    model.load_state_dict(torch.load(zero_ckpt, map_location=device, weights_only=True))

    mask_60, mask_61 = build_wet_masks_60_61(env)
    wq_60, wo_61 = build_volume_weights_60_61(env)
    obs_mask_161 = build_obs_wet_mask_161(env, mask_60)
    wet_60_cnt = float(mask_60.double().sum().clamp_min(1.0).item())
    wet_61_cnt = float(mask_61.double().sum().clamp_min(1.0).item())
    _, body_mask_bool = model.head_body_index_masks(device=device)
    n_body_params = float(max(int(body_mask_bool.sum().item()), 1))

    resp_op = CausalBlockResponseOperator3D(
        mask_60,
        mask_61,
        im=cfg.im,
        jm=cfg.jm,
        gain_bound=cfg.forge_full_gain_bound,
        weights_61=wo_61,
        action_weights_60=wq_60,
    ).to(device)
    obs_resp_op = CausalObservationResponseOperator3D(
        mask_60,
        obs_mask_161,
        im=cfg.im,
        jm=cfg.jm,
        gain_bound=cfg.obs_response_gain_bound,
    ).to(device)

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

    m_sub = int(cfg.forge_num_directions)
    h_norm = torch.tensor(HADAMARD_4X4, device=device, dtype=torch.float32) * 0.5
    max_rounds = int(cfg.forge_rounds)
    (
        round_worst_cost,
        n_base_per_gpu,
        n_extra_screen_per_gpu,
        n_s2_candidates,
    ) = _compute_round_worst_cost(cfg, stage_one_only, ablation, world_size, len(val_windows))
    post_eval_reserve = float(len(val_windows))

    replay_buffer: deque[dict[str, Any]] = deque(maxlen=max(int(cfg.forge_stage2_replay_rounds), 1))
    curr_trust_rms = float(cfg.forge_cg_trust_rms)
    curr_lambda_act = float(cfg.forge_action_reg_weight)
    stage2_adopted_rounds = 0
    stage2_trunk_adopted_rounds = 0
    stop_reason = "completed_max_rounds"

    if is_main_process(rank):
        print(
            f"[{stage_name}] Starting pure-4h training from common_zero.pt "
            f"(val_4h={tracker.init_val_4h:.6f}, M={m_sub}, stage_one_only={stage_one_only}, "
            f"ablation={ablation}, max_window_budget={max_window_budget}, worst_round_win={round_worst_cost:.0f})...",
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
        batch_windows = sample_epoch_windows(
            train_pool,
            round_idx=r,
            batch_size=n_base_per_gpu * world_size,
            seed=cfg.seed,
            stream_id=1009,
        )
        my_windows = batch_windows[rank::world_size]

        s1_res = run_stage1_subspace_step(
            cfg=cfg,
            env=env,
            model=model,
            my_windows=my_windows,
            round_idx=r,
            ablation=ablation,
            mask_60=mask_60,
            wq_60=wq_60,
            wet_61_cnt=wet_61_cnt,
            world_size=world_size,
            device=device,
        )
        train_window_rollouts += s1_res.window_rollouts_used

        if stage_one_only:
            chosen_alpha_s2 = 0.0
            stage2_accepted = False
            stage2_source = "none"
            post_s2_sync = s1_res.stage1_train_loss
            stage2_best_cand_loss: float | None = None
            stage2_max_body_diff = 0.0
            stage2_body_l2_diff = 0.0
            fit_stats: dict[str, Any] = {}
            heldout_gate: dict[str, Any] = {"passed": False, "skipped": True}
            obs_gate: dict[str, Any] = {"passed": False, "gamma_obs": 0.0}
            cg_info: dict[str, Any] = {}
        else:
            s2_res = run_stage2_response_step(
                cfg=cfg,
                env=env,
                model=model,
                resp_op=resp_op,
                obs_resp_op=obs_resp_op,
                replay_buffer=replay_buffer,
                train_pool=train_pool,
                my_windows=my_windows,
                s1_res=s1_res,
                mask_60=mask_60,
                mask_61=mask_61,
                obs_mask_161=obs_mask_161,
                wq_60=wq_60,
                wo_61=wo_61,
                wet_60_cnt=wet_60_cnt,
                wet_61_cnt=wet_61_cnt,
                body_mask_bool=body_mask_bool,
                n_body_params=n_body_params,
                h_norm=h_norm,
                curr_trust_rms=curr_trust_rms,
                curr_lambda_act=curr_lambda_act,
                n_s2_candidates=n_s2_candidates,
                n_extra_screen_per_gpu=n_extra_screen_per_gpu,
                stage_name=stage_name,
                ablation=ablation,
                round_idx=r,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            train_window_rollouts += s2_res.window_rollouts_used
            chosen_alpha_s2 = s2_res.chosen_alpha_s2
            stage2_accepted = s2_res.stage2_accepted
            stage2_source = s2_res.stage2_source
            post_s2_sync = s2_res.post_s2_sync
            stage2_best_cand_loss = s2_res.stage2_best_cand_loss
            stage2_max_body_diff = s2_res.stage2_max_body_diff
            stage2_body_l2_diff = s2_res.stage2_body_l2_diff
            curr_trust_rms = s2_res.curr_trust_rms
            curr_lambda_act = s2_res.curr_lambda_act
            fit_stats = s2_res.fit_stats
            heldout_gate = s2_res.heldout_gate
            obs_gate = s2_res.obs_gate
            cg_info = s2_res.cg_info
            if stage2_accepted:
                stage2_adopted_rounds += 1
                if stage2_max_body_diff > 0.0:
                    stage2_trunk_adopted_rounds += 1

        val_metrics = evaluate_validation_4h_with_worst(
            env, model, val_windows, rank, world_size, device
        )
        val_4h_window_rollouts += len(val_windows)
        cum_windows = float(train_window_rollouts + val_4h_window_rollouts)

        upd = tracker.update(
            round_idx=r,
            val_metrics=val_metrics,
            candidate_state=model.state_dict(),
            cum_windows=cum_windows,
        )
        if not upd["finite"]:
            model.load_state_dict(tracker.selected_state)

        r_wall = time.time() - r_t0
        if is_main_process(rank):
            rec = {
                "round": r,
                "ablation_variant": ablation,
                "stage1_comp_rank": s1_res.eff_comp_rank,
                "stage1_comp_head_l2": s1_res.chosen_comp_l2,
                "action_jac_rank_full": s1_res.rank_info["action_jac_rank_full"],
                "action_jac_rank_body_only": s1_res.rank_info["action_jac_rank_body_only"],
                "chosen_alpha": s1_res.chosen_alpha,
                "chosen_alpha_stage2": chosen_alpha_s2,
                "stage2_source": stage2_source,
                "pre_round_train_loss": s1_res.pre_round_train_loss,
                "stage1_train_loss": s1_res.stage1_train_loss,
                "stage2_best_cand_loss": stage2_best_cand_loss,
                "stage2_train_loss": post_s2_sync,
                "stage2_accepted": stage2_accepted,
                "stage2_max_body_diff": stage2_max_body_diff,
                "stage2_body_l2_diff": stage2_body_l2_diff,
                "stage2_adopted_rounds": stage2_adopted_rounds,
                "stage2_trunk_adopted_rounds": stage2_trunk_adopted_rounds,
                "curr_trust_rms": curr_trust_rms,
                "curr_lambda_act": curr_lambda_act,
                "replay_buffer_len": len(replay_buffer),
                "obs_response_gate": obs_gate,
                "response_fit_stats": fit_stats,
                "heldout_response_gate": {k: v for k, v in heldout_gate.items() if k != "metrics"}
                | (
                    {
                        "metrics": {
                            mk: mv
                            for mk, mv in heldout_gate["metrics"].items()
                            if mk != "per_probe_rel_errors"
                        }
                    }
                    if "metrics" in heldout_gate
                    else {}
                ),
                "cg_info": cg_info,
                "train_window_rollouts": train_window_rollouts,
                "val_4h_window_rollouts": val_4h_window_rollouts,
                "cumulative_window_rollouts": cum_windows,
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
                f"  [{stage_name} r{r:03d}] a1={s1_res.chosen_alpha:.2f} | "
                f"s1_l={s1_res.stage1_train_loss:.6f} | "
                f"s2={stage2_accepted} ({stage2_source}, a2={chosen_alpha_s2:.2f}, s2_l={post_s2_sync:.6f}, "
                f"gamma_obs={obs_gate.get('gamma_obs', 0.0):.2f}) | val_4h={upd['val_loss_4h']:.6f} "
                f"(best={tracker.selected_val_4h:.6f} @ r{tracker.selected_round}) | no_imp={tracker.no_improve} | "
                f"cum_win={cum_windows:.0f} | {r_wall:.1f}s",
                flush=True,
            )

        if tracker.no_improve >= int(cfg.forge_patience):
            stop_reason = f"early_stop_patience_{cfg.forge_patience}"
            break

    model.load_state_dict(tracker.selected_state)
    val_final = evaluate_validation_4h_with_worst(env, model, val_windows, rank, world_size, device)
    post_eval_windows = float(len(val_windows))
    cum_windows = float(train_window_rollouts + val_4h_window_rollouts)
    total_stage_windows = cum_windows + post_eval_windows

    fin_meta = tracker.finalize(cum_windows)
    s_cnt_end = get_ddp_solver_window_equivalents(env, device, forbid_whitebox=True)
    solver_verified_win = float(s_cnt_end["window_equivalents"] - s_cnt_start["window_equivalents"])
    if abs(solver_verified_win - total_stage_windows) > 1e-6:
        raise RuntimeError(
            f"[{stage_name}] Ledger vs solver substep mismatch: ledger_total={total_stage_windows} "
            f"vs solver_verified={solver_verified_win}"
        )

    summary = {
        "stage": stage_name,
        "stage_one_only": bool(stage_one_only),
        "ablation_variant": ablation,
        "forge_comp_rank": (
            0 if ablation in ("no_stage1", "no_body_comp") else max(int(cfg.forge_comp_rank), 0)
        ),
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
        "stage2_adopted_rounds": stage2_adopted_rounds,
        "stage2_trunk_adopted_rounds": stage2_trunk_adopted_rounds,
        "train_window_rollouts": train_window_rollouts,
        "val_4h_window_rollouts": val_4h_window_rollouts,
        "feasibility_window_equivalents": 0.0,
        "cumulative_window_rollouts": cum_windows,
        "post_selection_eval_window_equivalents": post_eval_windows,
        "total_stage_window_equivalents": total_stage_windows,
        "solver_verified_window_equivalents": solver_verified_win,
        "budget_checkpoints": fin_meta["milestones"],
        "wall_seconds": time.time() - t0,
        "val_4h": val_final,
    }
    if is_main_process(rank):
        save_atomic_json(summary_path, summary)
        print(
            f"[{stage_name} Done] selected_round={tracker.selected_round}, "
            f"s2_adopted={stage2_adopted_rounds} (trunk={stage2_trunk_adopted_rounds}), "
            f"val_4h={val_final['loss_4h']:.6f}, cum_win={cum_windows:.0f}, wall={summary['wall_seconds']:.1f}s",
            flush=True,
        )
    if dist.is_initialized():
        dist.barrier()
    return summary


run_stage2_forge_main_4h = run_forge_main_4h

__all__ = [
    "VALID_ABLATION_VARIANTS",
    "run_forge_main_4h",
    "run_stage2_forge_main_4h",
]
