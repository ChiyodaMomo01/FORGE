"""Stage 8 privileged white-box 24-substep Solver BPTT AD reference for 3D POM."""

from __future__ import annotations

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
from ..evaluation import Pure4hSelectionTracker, evaluate_validation_4h_with_worst
from ..network import ForgePOMNet
from ..physics_env import BlackBoxProvider3D
from ..sampling import ensure_common_zero_ckpt, sample_epoch_windows


def verify_whitebox_ad_parity_and_gradient(
    cfg: ForgePOM3DConfig,
    env: BlackBoxProvider3D,
    common_ckpt: Path,
    val_windows: list[tuple[int, int]],
    rank: int,
    device: torch.device,
    eps: float = 1e-2,
) -> dict[str, Any]:
    """Verify forward parity (`rel_diff < 1e-5`) and head/body/mixed FD gradient consistency before AD training.

    In `D = 4.27M` parameter dimensions, a random unit vector `u` has expected inner product
    `|<g, u>| ~ ||g|| / sqrt(D) ~ 1e-4`, making `2 * eps * |<g, u>| ~ 1e-6` comparable to float32
    24-substep elliptic-CG roundoff noise. Testing along the normalized AD head gradient `g_head / ||g_head||`,
    normalized AD trunk gradient `g_body / ||g_body||`, and their balanced combination yields first-order
    loss differences `dL ~ 3e-4 .. 6e-4` (`> 200x` above float32 roundoff) while independently certifying
    both readout-head and 4-hour closed-loop trunk backpropagation paths.
    """
    test_net = ForgePOMNet(
        cfg.in_channels, cfg.hidden_channels, cfg.out_channels, cfg.num_blocks
    ).to(device)
    test_net.load_state_dict(torch.load(common_ckpt, map_location=device, weights_only=True))
    test_net.eval()

    gen_p = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + 918273)
    theta0 = test_net.get_flat_params().detach().clone()
    head_mask, body_mask = test_net.head_body_index_masks(device=device)
    noise_cpu = torch.randn(theta0.numel(), generator=gen_p, dtype=torch.float32).to(device=device)
    theta_probe = (
        theta0 + 0.02 * noise_cpu * head_mask.float() + 0.005 * noise_cpu * body_mask.float()
    )
    test_net.set_flat_params(theta_probe)

    m_chk, d_chk = val_windows[int(rank) % len(val_windows)]

    # 1. Black-box vs White-box forward parity check
    loss_bb, _, _ = env.rollout_4h_window(m_chk, d_chk, lambda x_k, _k: test_net(x_k))
    test_net.zero_grad(set_to_none=True)
    loss_wb_t, _ = env.rollout_4h_window_whitebox_ad(m_chk, d_chk, lambda x_k, _k: test_net(x_k))
    loss_wb = float(loss_wb_t.detach().item())
    rel_fwd_diff = abs(loss_wb - float(loss_bb)) / max(abs(float(loss_bb)), 1e-12)

    # 2. Full 24-substep backprop vs black-box central finite differences on head, trunk, and mixed directions
    loss_wb_t.backward()
    grads_flat = torch.cat(
        [p.grad.detach().reshape(-1).double() for p in test_net.parameters()], dim=0
    )
    grad_norm = float(grads_flat.norm().item())

    g_head = grads_flat * head_mask.double()
    g_body = grads_flat * body_mask.double()
    head_grad_norm = float(g_head.norm().item())
    body_grad_norm = float(g_body.norm().item())

    u_head = (g_head / max(head_grad_norm, 1e-20)).to(dtype=torch.float32).contiguous()
    u_body = (g_body / max(body_grad_norm, 1e-20)).to(dtype=torch.float32).contiguous()
    u_mix = (math.sqrt(0.5) * u_head + math.sqrt(0.5) * u_body).contiguous()

    def _fd_check(u_vec: torch.Tensor) -> tuple[float, float, float]:
        ad_v = float((grads_flat * u_vec.double()).sum().item())
        with torch.no_grad():
            test_net.set_flat_params(theta_probe + float(eps) * u_vec)
            lp, _, _ = env.rollout_4h_window(m_chk, d_chk, lambda x_k, _k: test_net(x_k))
            test_net.set_flat_params(theta_probe - float(eps) * u_vec)
            lm, _, _ = env.rollout_4h_window(m_chk, d_chk, lambda x_k, _k: test_net(x_k))
        fd_v = (float(lp) - float(lm)) / (2.0 * float(eps))
        r_err = abs(ad_v - fd_v) / max(abs(ad_v), abs(fd_v), 1e-8)
        return ad_v, fd_v, r_err

    ad_head, fd_head, err_head = _fd_check(u_head)
    ad_body, fd_body, err_body = _fd_check(u_body)
    ad_dir_deriv, fd_dir_deriv, rel_grad_err = _fd_check(u_mix)

    vec_max = torch.tensor(
        [rel_fwd_diff, rel_grad_err, err_head, err_body],
        device=device,
        dtype=torch.float64,
    )
    vec_min = torch.tensor([body_grad_norm, head_grad_norm], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(vec_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(vec_min, op=dist.ReduceOp.MIN)

    max_rel_fwd = float(vec_max[0].item())
    max_rel_grad = float(vec_max[1].item())
    max_err_head = float(vec_max[2].item())
    max_err_body = float(vec_max[3].item())
    min_body_gnorm = float(vec_min[0].item())
    min_head_gnorm = float(vec_min[1].item())

    if not math.isfinite(max_rel_fwd) or max_rel_fwd > 1e-5:
        raise RuntimeError(
            f"[stage8_ad_ref] Forward parity check FAILED: max_rel_fwd_diff={max_rel_fwd:.3e} (> 1e-5)"
        )
    if min_body_gnorm <= 0.0 or min_head_gnorm <= 0.0:
        raise RuntimeError(
            f"[stage8_ad_ref] Zero gradient norm detected: min_body_gnorm={min_body_gnorm:.3e}, "
            f"min_head_gnorm={min_head_gnorm:.3e}"
        )
    if (
        not math.isfinite(max_rel_grad)
        or max_rel_grad > 0.12
        or max_err_head > 0.08
        or max_err_body > 0.20
    ):
        raise RuntimeError(
            f"[stage8_ad_ref] Directional FD gradient check FAILED: "
            f"max_mix_err={max_rel_grad:.4f}, max_head_err={max_err_head:.4f}, max_body_err={max_err_body:.4f} "
            f"(rank0 mix ad={ad_dir_deriv:.6e} vs fd={fd_dir_deriv:.6e}; "
            f"head ad={ad_head:.6e} vs fd={fd_head:.6e}; body ad={ad_body:.6e} vs fd={fd_body:.6e})"
        )

    return {
        "passed": True,
        "eps": float(eps),
        "rank0_window": [int(m_chk), int(d_chk)],
        "rank0_loss_blackbox": float(loss_bb),
        "rank0_loss_whitebox": float(loss_wb),
        "max_rel_forward_diff_all_ranks": max_rel_fwd,
        "rank0_ad_directional_deriv": ad_dir_deriv,
        "rank0_fd_directional_deriv": fd_dir_deriv,
        "rank0_head_ad_deriv": ad_head,
        "rank0_head_fd_deriv": fd_head,
        "rank0_body_ad_deriv": ad_body,
        "rank0_body_fd_deriv": fd_body,
        "max_rel_directional_grad_err_all_ranks": max_rel_grad,
        "max_rel_head_grad_err_all_ranks": max_err_head,
        "max_rel_body_grad_err_all_ranks": max_err_body,
        "rank0_full_grad_norm": grad_norm,
        "min_head_grad_norm_all_ranks": min_head_gnorm,
        "min_body_grad_norm_all_ranks": min_body_gnorm,
    }


def run_ad_reference(
    cfg: ForgePOM3DConfig,
    env: BlackBoxProvider3D,
    common_ckpt: Path,
    train_pool: list[tuple[int, int]],
    val_windows: list[tuple[int, int]],
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train privileged white-box 24-substep BPTT AD reference from `common_zero.pt` under pure `val_loss_4h`."""
    stage_name = "stage8_ad_ref"
    out_root_path = Path(cfg.output_root)
    stage_dir = out_root_path / stage_name
    summary_path = stage_dir / "summary.json"
    best_ckpt = stage_dir / "best_model.pt"
    if summary_path.exists() and best_ckpt.exists():
        if is_main_process(rank):
            print(f"[{stage_name}] Already completed, loading summary.json", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    t0 = time.time()
    zero_ckpt = ensure_common_zero_ckpt(cfg, rank, device, explicit_ckpt=common_ckpt)
    if is_main_process(rank):
        stage_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{stage_name}] Running pre-flight forward-parity & directional FD gradient verification...",
            flush=True,
        )

    env.reset_step_counters()
    s_pre_start = get_ddp_solver_window_equivalents(env, device, forbid_whitebox=False)
    grad_check_report = verify_whitebox_ad_parity_and_gradient(
        cfg, env, zero_ckpt, val_windows, rank, device, eps=1e-2
    )
    s_pre_end = get_ddp_solver_window_equivalents(env, device, forbid_whitebox=False)
    preflight_fwd_win = float(s_pre_end["window_equivalents"] - s_pre_start["window_equivalents"])
    preflight_wb_win = float(
        s_pre_end["whitebox_ad_window_equivalents"] - s_pre_start["whitebox_ad_window_equivalents"]
    )
    preflight_bwd_win = int(world_size)
    expected_preflight_fwd = float(8 * world_size)
    if abs(preflight_fwd_win - expected_preflight_fwd) > 1e-6:
        raise RuntimeError(
            f"[{stage_name}] Preflight forward window mismatch: expected={expected_preflight_fwd} "
            f"vs solver_verified={preflight_fwd_win}"
        )
    grad_check_report["preflight_forward_window_equivalents"] = preflight_fwd_win
    grad_check_report["preflight_whitebox_ad_window_equivalents"] = preflight_wb_win
    grad_check_report["preflight_backward_ad_windows"] = preflight_bwd_win
    if is_main_process(rank):
        save_atomic_json(stage_dir / "gradient_check.json", grad_check_report)
        print(
            f"[{stage_name}] Precheck PASSED: max_fwd_rel_diff={grad_check_report['max_rel_forward_diff_all_ranks']:.3e}, "
            f"max_mix_err={grad_check_report['max_rel_directional_grad_err_all_ranks']:.4f}, "
            f"max_head_err={grad_check_report['max_rel_head_grad_err_all_ranks']:.4f}, "
            f"max_body_err={grad_check_report['max_rel_body_grad_err_all_ranks']:.4f} "
            f"(preflight_fwd_win={preflight_fwd_win:.0f}, preflight_bwd_win={preflight_bwd_win})",
            flush=True,
        )

    env.reset_step_counters()
    s_cnt_start = get_ddp_solver_window_equivalents(env, device, forbid_whitebox=False)

    base_model = ForgePOMNet(
        cfg.in_channels, cfg.hidden_channels, cfg.out_channels, cfg.num_blocks
    ).to(device)
    base_model.load_state_dict(torch.load(zero_ckpt, map_location=device, weights_only=True))
    ddp_model = DDP(base_model, device_ids=[device.index]) if dist.is_initialized() else base_model

    head_params: list[torch.nn.Parameter] = []
    body_params: list[torch.nn.Parameter] = []
    for name, p in base_model.named_parameters():
        if name.startswith("head."):
            head_params.append(p)
        else:
            body_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": float(cfg.ad_head_lr), "weight_decay": 0.0},
            {
                "params": body_params,
                "lr": float(cfg.ad_body_lr),
                "weight_decay": float(cfg.ad_weight_decay),
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg.ad_lr_factor),
        patience=int(cfg.ad_lr_patience),
        min_lr=float(cfg.ad_min_lr),
    )

    val_init = evaluate_validation_4h_with_worst(
        env, base_model, val_windows, rank, world_size, device
    )
    val_4h_window_rollouts = len(val_windows)
    forward_train_windows = 0
    backward_ad_windows = 0
    skipped_nonfinite_steps = 0

    tracker = Pure4hSelectionTracker(
        stage_dir=stage_dir,
        rank=rank,
        init_state=base_model.state_dict(),
        init_val_metrics=val_init,
        init_cum_windows=float(val_4h_window_rollouts),
        anchor_window_budget=cfg.anchor_window_budget,
        improve_tol=cfg.val_improve_tol,
    )

    if is_main_process(rank):
        print(
            f"[{stage_name}] Starting privileged 24-substep AD from common_zero.pt "
            f"(val_4h={tracker.init_val_4h:.6f}, max_epochs={cfg.ad_max_epochs}, "
            f"steps/ep={cfg.ad_steps_per_epoch}, head_lr={cfg.ad_head_lr}, body_lr={cfg.ad_body_lr})...",
            flush=True,
        )

    stop_reason = "completed_max_epochs"
    global_step = 0

    for epoch in range(1, int(cfg.ad_max_epochs) + 1):
        ep_t0 = time.time()
        ddp_model.train()
        ep_loss_sum = 0.0
        ep_grad_norm_sum = 0.0
        valid_steps_this_ep = 0

        for _s_idx in range(1, int(cfg.ad_steps_per_epoch) + 1):
            global_step += 1
            batch_windows = sample_epoch_windows(
                train_pool,
                round_idx=global_step,
                batch_size=world_size,
                seed=cfg.seed,
                stream_id=8801,
            )
            m_w, d_w = batch_windows[rank]

            optimizer.zero_grad(set_to_none=True)
            loss_t, _ = env.rollout_4h_window_whitebox_ad(
                m_w,
                d_w,
                lambda x_k, _k: ddp_model(x_k.unsqueeze(0))[0],
            )
            forward_train_windows += world_size
            loss_t.backward()
            backward_ad_windows += world_size

            local_finite = 1.0 if bool(torch.isfinite(loss_t.detach()).item()) else 0.0
            for p in base_model.parameters():
                if p.grad is not None and not bool(torch.isfinite(p.grad).all().item()):
                    local_finite = 0.0
                    break
            fin_t = torch.tensor([local_finite], device=device, dtype=torch.float64)
            if dist.is_initialized():
                dist.all_reduce(fin_t, op=dist.ReduceOp.MIN)

            if float(fin_t[0].item()) < 0.5:
                optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite_steps += 1
                continue

            gnorm = float(
                torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(), float(cfg.ad_grad_clip)
                ).item()
            )
            optimizer.step()
            ep_loss_sum += float(loss_t.detach().item())
            ep_grad_norm_sum += gnorm
            valid_steps_this_ep += 1

        sync_ep = ddp_mean_dict(
            {
                "train_loss": ep_loss_sum / max(valid_steps_this_ep, 1),
                "grad_norm": ep_grad_norm_sum / max(valid_steps_this_ep, 1),
            },
            device,
        )

        val_metrics = evaluate_validation_4h_with_worst(
            env, base_model, val_windows, rank, world_size, device
        )
        val_4h_window_rollouts += len(val_windows)
        cum_fwd_windows = float(forward_train_windows + val_4h_window_rollouts)

        upd = tracker.update(
            round_idx=epoch,
            val_metrics=val_metrics,
            candidate_state=base_model.state_dict(),
            cum_windows=cum_fwd_windows,
        )
        if not upd["finite"]:
            base_model.load_state_dict(tracker.selected_state)
        else:
            scheduler.step(float(upd["val_loss_4h"]))

        cur_head_lr = float(optimizer.param_groups[0]["lr"])
        cur_body_lr = float(optimizer.param_groups[1]["lr"])
        ep_wall = time.time() - ep_t0

        if is_main_process(rank):
            rec = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss_4h": sync_ep["train_loss"],
                "grad_norm": sync_ep["grad_norm"],
                "head_lr": cur_head_lr,
                "body_lr": cur_body_lr,
                "val_loss_4h": upd["val_loss_4h"],
                "val_tracer_loss_4h": val_metrics["tracer_loss_4h"],
                "val_momentum_loss_4h": val_metrics["momentum_loss_4h"],
                "val_composite_nrmse_4h": val_metrics["composite_nrmse_4h"],
                "val_worst_window_loss_4h": val_metrics["worst_window_loss_4h"],
                "selected_epoch_so_far": tracker.selected_round,
                "selected_val_4h_so_far": tracker.selected_val_4h,
                "no_improve": tracker.no_improve,
                "forward_train_windows": forward_train_windows,
                "backward_ad_windows": backward_ad_windows,
                "val_4h_window_rollouts": val_4h_window_rollouts,
                "cumulative_forward_windows": cum_fwd_windows,
                "epoch_wall_seconds": ep_wall,
            }
            append_jsonl(stage_dir / "metrics.jsonl", rec)
            print(
                f"  [{stage_name} ep{epoch:02d}] train_4h={sync_ep['train_loss']:.6f} | "
                f"val_4h={upd['val_loss_4h']:.6f} (best={tracker.selected_val_4h:.6f} @ ep{tracker.selected_round}) | "
                f"lr=({cur_head_lr:.1e},{cur_body_lr:.1e}) | no_imp={tracker.no_improve} | "
                f"fwd_win={cum_fwd_windows:.0f} | {ep_wall:.1f}s",
                flush=True,
            )

        if tracker.no_improve >= int(cfg.ad_patience):
            stop_reason = f"early_stop_patience_{cfg.ad_patience}"
            break

    base_model.load_state_dict(tracker.selected_state)
    val_final = evaluate_validation_4h_with_worst(
        env, base_model, val_windows, rank, world_size, device
    )
    post_eval_windows = float(len(val_windows))
    cum_fwd_windows = float(forward_train_windows + val_4h_window_rollouts)
    total_stage_fwd_windows = cum_fwd_windows + post_eval_windows

    fin_meta = tracker.finalize(cum_fwd_windows)
    s_cnt_end = get_ddp_solver_window_equivalents(env, device, forbid_whitebox=False)
    solver_verified_win = float(s_cnt_end["window_equivalents"] - s_cnt_start["window_equivalents"])
    whitebox_train_win = float(
        s_cnt_end["whitebox_ad_window_equivalents"] - s_cnt_start["whitebox_ad_window_equivalents"]
    )
    if abs(solver_verified_win - total_stage_fwd_windows) > 1e-6:
        raise RuntimeError(
            f"[{stage_name}] Ledger vs solver substep mismatch: expected={total_stage_fwd_windows} "
            f"vs solver_verified={solver_verified_win}"
        )
    if abs(whitebox_train_win - float(forward_train_windows)) > 1e-6:
        raise RuntimeError(
            f"[{stage_name}] White-box AD window mismatch: expected={forward_train_windows} "
            f"vs whitebox_verified={whitebox_train_win}"
        )

    summary = {
        "stage": stage_name,
        "privileged_whitebox_reference": True,
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
        "preflight_forward_window_equivalents": preflight_fwd_win,
        "preflight_whitebox_ad_window_equivalents": preflight_wb_win,
        "preflight_backward_ad_windows": preflight_bwd_win,
        "forward_train_windows": forward_train_windows,
        "backward_ad_windows": backward_ad_windows,
        "total_backward_ad_windows_including_preflight": int(
            backward_ad_windows + preflight_bwd_win
        ),
        "whitebox_ad_window_equivalents": whitebox_train_win,
        "total_whitebox_ad_window_equivalents_including_preflight": float(
            whitebox_train_win + preflight_wb_win
        ),
        "skipped_nonfinite_steps": skipped_nonfinite_steps,
        "val_4h_window_rollouts": val_4h_window_rollouts,
        "cumulative_window_rollouts": cum_fwd_windows,
        "post_selection_eval_window_equivalents": post_eval_windows,
        "total_stage_window_equivalents": total_stage_fwd_windows,
        "solver_verified_window_equivalents": solver_verified_win,
        "total_stage_including_preflight_window_equivalents": float(
            total_stage_fwd_windows + preflight_fwd_win
        ),
        "solver_verified_including_preflight_window_equivalents": float(
            solver_verified_win + preflight_fwd_win
        ),
        "gradient_check": grad_check_report,
        "budget_checkpoints": fin_meta["milestones"],
        "wall_seconds": time.time() - t0,
        "val_4h": val_final,
    }
    if is_main_process(rank):
        save_atomic_json(summary_path, summary)
        print(
            f"[{stage_name} Done] selected_epoch={tracker.selected_round}, "
            f"val_4h={val_final['loss_4h']:.6f}, fwd_win={cum_fwd_windows:.0f}, "
            f"bwd_win={backward_ad_windows}, wall={summary['wall_seconds']:.1f}s",
            flush=True,
        )
    if dist.is_initialized():
        dist.barrier()
    return summary


run_stage8_whitebox_ad = run_ad_reference

__all__ = [
    "run_ad_reference",
    "run_stage8_whitebox_ad",
    "verify_whitebox_ad_parity_and_gradient",
]
