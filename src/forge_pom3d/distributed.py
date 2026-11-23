"""Distributed data-parallel utilities and hardware solver substep verification."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist

from .physics_env import BlackBoxProvider3D


def setup_ddp() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def ddp_mean_dict(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    keys = sorted(metrics.keys())
    vec = torch.tensor([float(metrics[k]) for k in keys], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(vec, op=dist.ReduceOp.AVG)
    return {k: float(vec[i].item()) for i, k in enumerate(keys)}


def ddp_all_gather_objects(obj: Any, world_size: int) -> list[Any]:
    if not dist.is_initialized() or world_size == 1:
        return [obj]
    out: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(out, obj)
    return out


def get_ddp_solver_window_equivalents(
    env: BlackBoxProvider3D,
    device: torch.device,
    forbid_whitebox: bool = True,
) -> dict[str, Any]:
    """Sum hardware-executed 10-minute solver substeps across all DDP ranks (24 substeps == 1 x 4h window)."""
    has_counters = hasattr(env, "get_step_counters")
    if not has_counters:
        return {
            "has_counters": False,
            "predictor_substeps": 0,
            "complete_substeps": 0,
            "whitebox_ad_substeps": 0,
            "whitebox_ad_window_equivalents": 0.0,
            "window_equivalents": 0.0,
        }
    cnt = env.get_step_counters()
    vec = torch.tensor(
        [
            float(cnt["predictor_substeps_executed"]),
            float(cnt["complete_substeps_executed"]),
            float(cnt.get("whitebox_ad_substeps_executed", 0)),
        ],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    pred_steps = int(round(float(vec[0].item())))
    comp_steps = int(round(float(vec[1].item())))
    wb_steps = int(round(float(vec[2].item())))
    if forbid_whitebox and wb_steps > 0:
        raise RuntimeError(
            f"Illegal white-box AD substep execution inside strict black-box stage: whitebox_ad_substeps={wb_steps}"
        )
    return {
        "has_counters": True,
        "predictor_substeps": pred_steps,
        "complete_substeps": comp_steps,
        "whitebox_ad_substeps": wb_steps,
        "whitebox_ad_window_equivalents": float(wb_steps) / 24.0,
        "window_equivalents": float(pred_steps) / 24.0,
    }


__all__ = [
    "cleanup_ddp",
    "ddp_all_gather_objects",
    "ddp_mean_dict",
    "get_ddp_solver_window_equivalents",
    "is_main_process",
    "setup_ddp",
]
