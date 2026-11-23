"""Stage 0 unparameterized coarse physics baseline (G_theta == 0) for 3D POM."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ..artifacts import (
    all_numeric_leaves_finite,
    relativize_artifact_path,
    save_atomic_json,
)
from ..config import ForgePOM3DConfig
from ..distributed import get_ddp_solver_window_equivalents, is_main_process
from ..evaluation import (
    evaluate_frozen_test_4h_ddp,
    evaluate_validation_4h_with_worst,
)
from ..physics_env import BlackBoxProvider3D


def run_physics_baseline(
    cfg: ForgePOM3DConfig,
    env: BlackBoxProvider3D,
    val_windows: list[tuple[int, int]],
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run Pure-4h coarse physics baseline on validation and frozen Test ID splits."""
    out_root_path = Path(cfg.output_root)
    stage_dir = out_root_path / "stage0_physics"
    summary_path = stage_dir / "summary.json"
    test_4h_path = stage_dir / "test_4h.json"
    if summary_path.exists() and test_4h_path.exists():
        if is_main_process(rank):
            print("[Stage 0] Already completed, loading summary.json", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    t0 = time.time()
    s_cnt_start = get_ddp_solver_window_equivalents(env, device, forbid_whitebox=True)
    if is_main_process(rank):
        print(
            "[Stage 0] Running Pure-4h Physics Baseline (16 val windows + 48 strided & 8 continuity Test ID windows)...",
            flush=True,
        )

    val_4h = evaluate_validation_4h_with_worst(env, None, val_windows, rank, world_size, device)
    test_4h = evaluate_frozen_test_4h_ddp(
        cfg, env, None, rank, world_size, physics_ref_test_4h=None
    )
    expected_total_win = float(len(val_windows)) + float(test_4h["eval_window_equivalents"])

    s_cnt_end = get_ddp_solver_window_equivalents(env, device, forbid_whitebox=True)
    solver_verified_win: float | None = None
    if s_cnt_end["has_counters"]:
        solver_verified_win = float(
            s_cnt_end["window_equivalents"] - s_cnt_start["window_equivalents"]
        )
        if abs(solver_verified_win - expected_total_win) > 1e-6:
            raise RuntimeError(
                f"[stage0_physics] Ledger vs solver substep mismatch: ledger_total={expected_total_win} "
                f"vs solver_verified={solver_verified_win}"
            )
    if not all_numeric_leaves_finite(val_4h) or not all_numeric_leaves_finite(test_4h):
        raise FloatingPointError("Non-finite metric detected in stage0_physics")

    test_4h["status"] = "completed"
    test_4h["source_stage"] = "stage0_physics"
    test_4h["protocol_mode"] = "pure_4h"
    test_4h["source_version"] = cfg.source_version
    test_4h["source_sha256"] = cfg.source_sha256
    test_4h["source_tree_sha256"] = cfg.source_tree_sha256
    test_4h["effective_config_sha256"] = cfg.effective_config_sha256
    test_4h["source_bundle_sha256"] = cfg.source_bundle_sha256
    test_4h["solver_verified_window_equivalents"] = (
        float(test_4h["eval_window_equivalents"]) if solver_verified_win is not None else None
    )
    test_4h["all_metrics_finite"] = True

    summary = {
        "stage": "stage0_physics",
        "protocol_mode": "pure_4h",
        "status": "completed",
        "source_version": cfg.source_version,
        "source_sha256": cfg.source_sha256,
        "source_tree_sha256": cfg.source_tree_sha256,
        "effective_config_sha256": cfg.effective_config_sha256,
        "source_bundle_sha256": cfg.source_bundle_sha256,
        "val_4h_window_rollouts": len(val_windows),
        "test_4h_window_rollouts": int(round(float(test_4h["eval_window_equivalents"]))),
        "total_stage_window_equivalents": expected_total_win,
        "solver_verified_window_equivalents": solver_verified_win,
        "wall_seconds": time.time() - t0,
        "val_4h": val_4h,
        "test_4h_summary": {
            "loss_4h": test_4h["loss_4h"],
            "composite_nrmse_4h": test_4h["composite_nrmse_4h"],
            "momentum_loss_4h": test_4h["momentum_loss_4h"],
            "tracer_loss_4h": test_4h["tracer_loss_4h"],
            "worst_window_loss_4h": test_4h["worst_window_loss_4h"],
            "continuity8_loss_4h": test_4h["continuity8_loss_4h"],
        },
        "test_4h_json_path": relativize_artifact_path(test_4h_path, out_root_path),
    }
    if is_main_process(rank):
        save_atomic_json(test_4h_path, test_4h)
        save_atomic_json(summary_path, summary)
        print(
            f"[Stage 0 Done] val_4h={val_4h['loss_4h']:.6f} | "
            f"test_4h_48={test_4h['loss_4h']:.6f} (continuity8={test_4h['continuity8_loss_4h']:.6f}) | "
            f"win={expected_total_win:.0f} | wall={summary['wall_seconds']:.1f}s",
            flush=True,
        )
    if dist.is_initialized():
        dist.barrier()
    return summary


__all__ = ["run_physics_baseline"]
