"""Validation, frozen Test ID evaluation, budget gating, and pure-4h model selection."""

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

from .artifacts import (
    all_numeric_leaves_finite,
    atomic_torch_save,
    relativize_artifact_path,
    save_atomic_json,
    sha256_file,
)
from .config import ForgePOM3DConfig
from .distributed import (
    ddp_all_gather_objects,
    ddp_mean_dict,
    get_ddp_solver_window_equivalents,
    is_main_process,
)
from .network import ForgePOMNet
from .physics_env import BlackBoxProvider3D
from .sampling import build_test_id_48_windows, build_test_id_continuity8_windows

BUDGET_FRACTIONS: tuple[tuple[str, float], ...] = (
    ("10pct", 0.10),
    ("25pct", 0.25),
    ("50pct", 0.50),
    ("100pct", 1.00),
)


def can_afford_next_round(
    curr_cum_windows: float,
    round_base_cost: float,
    max_window_budget: float | None,
) -> tuple[bool, float]:
    """Return `(allowed, projected_cum)` under the pure-4h query ceiling."""
    projected = float(curr_cum_windows) + float(round_base_cost)
    if max_window_budget is None:
        return True, projected
    return (projected <= float(max_window_budget) + 1e-9), projected


@torch.no_grad()
def evaluate_validation_4h(
    env: BlackBoxProvider3D,
    model: ForgePOMNet | None,
    val_windows: list[tuple[int, int]],
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate 4-hour dense trajectory loss across all 16 validation windows across DDP ranks."""
    if model is not None:
        model.eval()

        def _policy(x_k: torch.Tensor, _k: int) -> torch.Tensor:
            return model(x_k)

        policy_fn = _policy
    else:
        policy_fn = None

    if not val_windows:
        raise ValueError("val_windows must be non-empty")
    my_windows = val_windows[rank::world_size]
    if dist.is_initialized() and world_size > 1 and len(val_windows) % world_size != 0:
        local_rows: list[dict[str, float]] = []
        for m, d in my_windows:
            _, metrics, _ = env.rollout_4h_window(m, d, policy_fn)
            local_rows.append({k: float(v) for k, v in metrics.items()})
        all_rows = [
            row for shard in ddp_all_gather_objects(local_rows, world_size) for row in shard
        ]
        keys = sorted(all_rows[0].keys())
        return {k: float(sum(r[k] for r in all_rows) / len(all_rows)) for k in keys}

    accum: dict[str, float] = {}
    for m, d in my_windows:
        _, metrics, _ = env.rollout_4h_window(m, d, policy_fn)
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + float(v) / max(len(my_windows), 1)
    return ddp_mean_dict(accum, device)


@torch.no_grad()
def evaluate_validation_4h_with_worst(
    env: BlackBoxProvider3D,
    model: ForgePOMNet | None,
    val_windows: list[tuple[int, int]],
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate 4h validation metrics across 16 windows plus worst single-window loss."""
    if model is not None:
        model.eval()

        def _policy(x_k: torch.Tensor, _k: int) -> torch.Tensor:
            return model(x_k)

        policy_fn = _policy
    else:
        policy_fn = None

    if not val_windows:
        raise ValueError("val_windows must be non-empty")
    my_windows = val_windows[rank::world_size]
    if dist.is_initialized() and world_size > 1 and len(val_windows) % world_size != 0:
        local_rows: list[tuple[float, dict[str, float]]] = []
        for m, d in my_windows:
            l_w, metrics, _ = env.rollout_4h_window(m, d, policy_fn)
            local_rows.append((float(l_w), {k: float(v) for k, v in metrics.items()}))
        all_rows = [
            row for shard in ddp_all_gather_objects(local_rows, world_size) for row in shard
        ]
        keys = sorted(all_rows[0][1].keys())
        out = {k: float(sum(m_d[k] for _, m_d in all_rows) / len(all_rows)) for k in keys}
        out["worst_window_loss_4h"] = float(max(lw for lw, _ in all_rows))
        return out

    accum: dict[str, float] = {}
    local_worst = 0.0
    for m, d in my_windows:
        l_w, metrics, _ = env.rollout_4h_window(m, d, policy_fn)
        local_worst = max(local_worst, float(l_w))
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + float(v) / max(len(my_windows), 1)

    keys = sorted(accum.keys())
    vec = torch.tensor([float(accum[k]) for k in keys], device=device, dtype=torch.float64)
    worst_t = torch.tensor([local_worst], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(vec, op=dist.ReduceOp.AVG)
        dist.all_reduce(worst_t, op=dist.ReduceOp.MAX)
    out = {k: float(vec[i].item()) for i, k in enumerate(keys)}
    out["worst_window_loss_4h"] = float(worst_t[0].item())
    return out


@torch.no_grad()
def evaluate_shared_windows_ddp(
    env: BlackBoxProvider3D,
    model: ForgePOMNet,
    shared_windows: list[tuple[int, int]],
    rank: int,
    world_size: int,
    device: torch.device,
    capture_obs: bool = False,
) -> tuple[dict[str, float], int] | tuple[dict[str, float], int, list[torch.Tensor]]:
    """Evaluate deterministic policy on `shared_windows` without duplicate PDE queries across ranks."""
    model.eval()
    n_win = max(len(shared_windows), 1)
    local_loss_sum = 0.0
    local_tr_sum = 0.0
    first_obs_rank0: torch.Tensor | None = None
    for idx in range(rank, len(shared_windows), world_size):
        m, d = shared_windows[idx]
        if capture_obs and idx == 0:
            captured_box: list[torch.Tensor] = []

            def _pol_capture(
                x_k: torch.Tensor,
                step_k: int,
                box: list[torch.Tensor] = captured_box,
            ) -> torch.Tensor:
                if step_k == 0 and not box:
                    box.append(x_k.detach().clone())
                return model(x_k)

            l_w, met_w, _ = env.rollout_4h_window(m, d, _pol_capture)
            if captured_box:
                first_obs_rank0 = captured_box[0]
        else:
            l_w, met_w, _ = env.rollout_4h_window(m, d, lambda x_k, _k: model(x_k))
        local_loss_sum += float(l_w)
        local_tr_sum += float(met_w["tracer_loss_4h"])
    vec = torch.tensor([local_loss_sum, local_tr_sum], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    metrics_out = {
        "loss_4h": float(vec[0].item()) / float(n_win),
        "tracer_loss_4h": float(vec[1].item()) / float(n_win),
    }
    if not capture_obs:
        return metrics_out, len(shared_windows)

    obs_shape = (env.cfg.in_channels, env.cfg.im, env.cfg.jm)
    if rank == 0 and first_obs_rank0 is not None:
        obs_bcast = first_obs_rank0.to(device=device, dtype=torch.float32).contiguous()
    else:
        obs_bcast = torch.zeros(obs_shape, device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.broadcast(obs_bcast, src=0)
    return metrics_out, len(shared_windows), [obs_bcast]


class Pure4hSelectionTracker:
    """Unified pure-4h validation model-selection & budget-milestone state machine."""

    def __init__(
        self,
        stage_dir: Path,
        rank: int,
        init_state: dict[str, torch.Tensor],
        init_val_metrics: dict[str, float],
        init_cum_windows: float,
        anchor_window_budget: float = 121504.0,
        improve_tol: float = 1e-6,
    ) -> None:
        self.stage_dir = stage_dir
        self.rank = int(rank)
        self.is_main = self.rank == 0
        self.best_ckpt = stage_dir / "best_model.pt"
        self.last_round_ckpt = stage_dir / "last_round_model.pt"
        self.milestones_dir = stage_dir / "milestones"
        self.anchor_window_budget = float(anchor_window_budget)
        self.improve_tol = float(improve_tol)

        init_4h = float(init_val_metrics["loss_4h"])
        if not math.isfinite(init_4h):
            raise FloatingPointError(f"Initial round-0 val_loss_4h is non-finite: {init_4h}")

        self.init_val_4h = init_4h
        self.init_val_metrics = copy.deepcopy(init_val_metrics)
        self.selected_round: int = 0
        self.selected_val_4h: float = init_4h
        self.patience_ref_val_4h: float = init_4h
        self.selected_val_metrics: dict[str, float] = copy.deepcopy(init_val_metrics)
        self.selected_cum_windows: float = float(init_cum_windows)
        self.selected_state: dict[str, torch.Tensor] = copy.deepcopy(init_state)

        self.last_completed_round: int = 0
        self.last_finite_state: dict[str, torch.Tensor] = copy.deepcopy(init_state)
        self.no_improve: int = 0
        self.nonfinite_val_rounds: int = 0

        self._thresholds: dict[str, float] = {
            label: frac * self.anchor_window_budget for label, frac in BUDGET_FRACTIONS
        }
        self._crossed: dict[str, bool] = {label: False for label, _ in BUDGET_FRACTIONS}
        self._milestone_meta: dict[str, dict[str, Any]] = {}
        for label, thr in self._thresholds.items():
            self._milestone_meta[label] = {
                "fraction": dict(BUDGET_FRACTIONS)[label],
                "anchor_window_budget": self.anchor_window_budget,
                "threshold_window_equivalents": thr,
                "selected_round": 0,
                "selected_val_4h": init_4h,
                "selected_at_cumulative_window_rollouts": float(init_cum_windows),
                "last_round_within_threshold": 0,
                "last_cum_windows_within_threshold": float(init_cum_windows),
            }

        if self.is_main:
            self.stage_dir.mkdir(parents=True, exist_ok=True)
            self.milestones_dir.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(self.selected_state, self.best_ckpt)
            atomic_torch_save(self.last_finite_state, self.last_round_ckpt)
            for label in self._thresholds:
                atomic_torch_save(self.selected_state, self.milestones_dir / f"cost-{label}.pt")

    def update(
        self,
        round_idx: int,
        val_metrics: dict[str, float],
        candidate_state: dict[str, torch.Tensor],
        cum_windows: float,
    ) -> dict[str, Any]:
        """Record round `round_idx` validation metrics and update `best_model.pt` & milestones."""
        self.last_completed_round = int(round_idx)
        finite = all_numeric_leaves_finite(val_metrics)
        val_loss = float(val_metrics.get("loss_4h", float("nan"))) if finite else float("nan")
        strictly_better = bool(finite and (val_loss < self.selected_val_4h))
        patience_improved = bool(
            finite and (val_loss < self.patience_ref_val_4h - self.improve_tol)
        )

        if not finite:
            self.nonfinite_val_rounds += 1
            self.no_improve += 1
        else:
            self.last_finite_state = copy.deepcopy(candidate_state)
            if strictly_better:
                self.selected_round = int(round_idx)
                self.selected_val_4h = val_loss
                self.selected_val_metrics = copy.deepcopy(val_metrics)
                self.selected_cum_windows = float(cum_windows)
                self.selected_state = copy.deepcopy(candidate_state)
                if self.is_main:
                    atomic_torch_save(self.selected_state, self.best_ckpt)
            if patience_improved:
                self.patience_ref_val_4h = val_loss
                self.no_improve = 0
            else:
                self.no_improve += 1
            if self.is_main:
                atomic_torch_save(self.last_finite_state, self.last_round_ckpt)

        for label, thr in self._thresholds.items():
            if float(cum_windows) <= thr + 1e-9:
                meta = self._milestone_meta[label]
                meta["last_round_within_threshold"] = int(round_idx)
                meta["last_cum_windows_within_threshold"] = float(cum_windows)
                if strictly_better:
                    meta["selected_round"] = self.selected_round
                    meta["selected_val_4h"] = self.selected_val_4h
                    meta["selected_at_cumulative_window_rollouts"] = self.selected_cum_windows
                    if self.is_main:
                        atomic_torch_save(
                            self.selected_state,
                            self.milestones_dir / f"cost-{label}.pt",
                        )
            else:
                self._crossed[label] = True

        return {
            "finite": finite,
            "improved": strictly_better,
            "patience_reset": patience_improved,
            "val_loss_4h": val_loss,
            "selected_round": self.selected_round,
            "selected_val_4h": self.selected_val_4h,
            "no_improve": self.no_improve,
        }

    def finalize(self, final_cum_windows: float) -> dict[str, Any]:
        """Finalize `best_model.pt` and all `cost-{10,25,50,100}pct.pt` metadata with SHA-256."""
        out_milestones: dict[str, Any] = {}
        best_sha = ""
        last_sha = ""
        base_out_dir = self.stage_dir.parent
        if self.is_main:
            atomic_torch_save(self.selected_state, self.best_ckpt)
            atomic_torch_save(self.last_finite_state, self.last_round_ckpt)
            best_sha = sha256_file(self.best_ckpt)
            last_sha = sha256_file(self.last_round_ckpt)
            for label, thr in self._thresholds.items():
                ckpt_p = self.milestones_dir / f"cost-{label}.pt"
                if not ckpt_p.exists():
                    atomic_torch_save(self.selected_state, ckpt_p)
                meta = dict(self._milestone_meta[label])
                meta["early_stopped_before_threshold"] = bool(float(final_cum_windows) < thr - 1e-9)
                meta["final_cumulative_window_rollouts"] = float(final_cum_windows)
                meta["checkpoint_path"] = relativize_artifact_path(ckpt_p, base_out_dir)
                meta["checkpoint_sha256"] = sha256_file(ckpt_p)
                out_milestones[label] = meta

        payload_list: list[Any] = [
            {
                "best_model_sha256": best_sha,
                "last_round_model_sha256": last_sha,
                "milestones": out_milestones,
            }
        ]
        if dist.is_initialized():
            dist.broadcast_object_list(payload_list, src=0)
        return payload_list[0]


def _summarize_window_records(
    records: list[dict[str, Any]],
    physics_by_day: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Empty window record list in Test ID evaluation")
    records_sorted = sorted(records, key=lambda r: int(r["start_day"]))
    n = len(records_sorted)
    scalar_keys = sorted(
        k
        for k, v in records_sorted[0].items()
        if isinstance(v, (int, float)) and k not in ("member", "start_day")
    )
    summary: dict[str, Any] = {
        "member": int(records_sorted[0]["member"]),
        "num_windows": n,
        "start_days": [int(r["start_day"]) for r in records_sorted],
    }
    for k in scalar_keys:
        vals = np.asarray([float(r[k]) for r in records_sorted], dtype=np.float64)
        summary[k] = float(vals.mean())

    losses = np.asarray([float(r["loss_4h"]) for r in records_sorted], dtype=np.float64)
    summary["median_loss_4h"] = float(np.median(losses))
    summary["p90_loss_4h"] = float(np.percentile(losses, 90.0))
    summary["worst_window_loss_4h"] = float(np.max(losses))
    summary["best_window_loss_4h"] = float(np.min(losses))
    summary["std_loss_4h"] = float(np.std(losses, ddof=1)) if n > 1 else 0.0
    summary["se_loss_4h"] = float(summary["std_loss_4h"] / math.sqrt(float(n))) if n > 1 else 0.0

    if physics_by_day is not None:
        deltas: list[float] = []
        wins = 0
        for r in records_sorted:
            d_key = int(r["start_day"])
            if d_key in physics_by_day:
                diff = float(r["loss_4h"]) - float(physics_by_day[d_key]["loss_4h"])
                deltas.append(diff)
                if diff < -1e-9:
                    wins += 1
        if len(deltas) == n:
            arr_d = np.asarray(deltas, dtype=np.float64)
            std_d = float(np.std(arr_d, ddof=1)) if n > 1 else 0.0
            phys_mean = float(
                np.mean(
                    [float(physics_by_day[int(r["start_day"])]["loss_4h"]) for r in records_sorted]
                )
            )
            summary["paired_vs_physics"] = {
                "physics_mean_loss_4h": phys_mean,
                "mean_delta_loss_4h": float(arr_d.mean()),
                "relative_change_pct": float(100.0 * arr_d.mean() / max(phys_mean, 1e-20)),
                "median_delta_loss_4h": float(np.median(arr_d)),
                "worst_delta_loss_4h": float(arr_d.max()),
                "paired_se_delta_loss_4h": (float(std_d / math.sqrt(float(n))) if n > 1 else 0.0),
                "wins_vs_physics": int(wins),
                "num_windows": int(n),
                "win_fraction": float(wins) / float(n),
            }

    summary["per_window_records"] = records_sorted
    return summary


@torch.no_grad()
def evaluate_window_list_ddp(
    env: BlackBoxProvider3D,
    model: ForgePOMNet | None,
    windows: list[tuple[int, int]],
    rank: int,
    world_size: int,
    physics_by_day: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate `windows` across DDP ranks without duplicate solver queries and gather records."""
    if model is not None:
        model.eval()

        def _policy(x_k: torch.Tensor, _k: int) -> torch.Tensor:
            return model(x_k)

        policy_fn = _policy
    else:
        policy_fn = None

    my_windows = windows[rank::world_size]
    my_records: list[dict[str, Any]] = []
    for m, d in my_windows:
        _, met, _ = env.rollout_4h_window(int(m), int(d), policy_fn)
        rec: dict[str, Any] = {"member": int(m), "start_day": int(d)}
        for k, v in met.items():
            rec[k] = float(v)
        my_records.append(rec)

    if dist.is_initialized() and world_size > 1:
        gathered: list[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, my_records)
        all_records = [item for sub in gathered for item in sub]
    else:
        all_records = my_records

    return _summarize_window_records(all_records, physics_by_day=physics_by_day)


@torch.no_grad()
def evaluate_frozen_test_4h_ddp(
    cfg: ForgePOM3DConfig,
    env: BlackBoxProvider3D,
    model: ForgePOMNet | None,
    rank: int,
    world_size: int,
    physics_ref_test_4h: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate frozen model on 48 strided `member=7` Test ID windows + 8 continuity windows."""
    phys_48_map: dict[int, dict[str, Any]] | None = None
    phys_8_map: dict[int, dict[str, Any]] | None = None
    if physics_ref_test_4h is not None:
        if (
            "test_4h_48" in physics_ref_test_4h
            and "per_window_records" in physics_ref_test_4h["test_4h_48"]
        ):
            phys_48_map = {
                int(r["start_day"]): r
                for r in physics_ref_test_4h["test_4h_48"]["per_window_records"]
            }
        if (
            "test_4h_continuity8" in physics_ref_test_4h
            and "per_window_records" in physics_ref_test_4h["test_4h_continuity8"]
        ):
            phys_8_map = {
                int(r["start_day"]): r
                for r in physics_ref_test_4h["test_4h_continuity8"]["per_window_records"]
            }

    win_48 = build_test_id_48_windows(cfg)
    win_8 = build_test_id_continuity8_windows(cfg)

    rep_48 = evaluate_window_list_ddp(
        env, model, win_48, rank, world_size, physics_by_day=phys_48_map
    )
    rep_8 = evaluate_window_list_ddp(env, model, win_8, rank, world_size, physics_by_day=phys_8_map)

    return {
        "primary_metric": "test_4h_48.loss_4h",
        "loss_4h": float(rep_48["loss_4h"]),
        "composite_nrmse_4h": float(rep_48["composite_nrmse_4h"]),
        "momentum_loss_4h": float(rep_48["momentum_loss_4h"]),
        "tracer_loss_4h": float(rep_48["tracer_loss_4h"]),
        "worst_window_loss_4h": float(rep_48["worst_window_loss_4h"]),
        "continuity8_loss_4h": float(rep_8["loss_4h"]),
        "eval_window_equivalents": float(len(win_48) + len(win_8)),
        "test_4h_48": rep_48,
        "test_4h_continuity8": rep_8,
    }


def run_standalone_test_4h(
    cfg: ForgePOM3DConfig,
    env: BlackBoxProvider3D,
    common_ckpt: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    eval_ckpt_arg: str | None,
    eval_stage_summary_arg: str | None = None,
    physics_test_4h_json_arg: str | None = None,
    eval_out_json_arg: str | None = None,
) -> dict[str, Any]:
    """Standalone frozen Test ID (`test_4h`) evaluator (48 strided + 8 continuity windows)."""
    if not eval_ckpt_arg:
        raise ValueError("--eval-ckpt is required when running standalone test_4h stage")

    eval_ckpt_path = Path(eval_ckpt_arg).resolve()
    if not eval_ckpt_path.exists():
        raise FileNotFoundError(f"Frozen --eval-ckpt does not exist: {eval_ckpt_path.name}")
    eval_ckpt_sha = sha256_file(eval_ckpt_path)

    stage_summary_path: Path | None = None
    if eval_stage_summary_arg:
        stage_summary_path = Path(eval_stage_summary_arg).resolve()
        if not stage_summary_path.exists():
            raise FileNotFoundError(
                f"--eval-stage-summary does not exist: {stage_summary_path.name}"
            )
    elif (eval_ckpt_path.parent / "summary.json").exists():
        stage_summary_path = (eval_ckpt_path.parent / "summary.json").resolve()

    source_stage_name: str | None = None
    source_ablation_variant: str = "none"
    selected_round_val: int | None = None
    checkpoint_role: str = eval_ckpt_path.stem
    stage_summary_sha: str | None = None
    recorded_source_digests: dict[str, Any] | None = None

    if stage_summary_path is not None:
        src_sum = json.loads(stage_summary_path.read_text(encoding="utf-8"))
        source_ablation_variant = str(src_sum.get("ablation_variant", "none"))
        stage_summary_sha = sha256_file(stage_summary_path)
        # Copy recorded source digests from the stage summary.
        recorded_source_digests = {
            k: src_sum[k]
            for k in (
                "source_version",
                "source_sha256",
                "source_tree_sha256",
                "effective_config_sha256",
                "source_bundle_sha256",
            )
            if k in src_sum
        }
        if src_sum.get("status") != "completed":
            raise RuntimeError(f"Source stage summary is not completed: {stage_summary_path.name}")
        expected_tree = src_sum.get("source_tree_sha256")
        if expected_tree is not None and expected_tree != cfg.source_tree_sha256:
            raise RuntimeError(
                f"Source stage summary source_tree_sha256 mismatch: "
                f"summary={expected_tree} vs current={cfg.source_tree_sha256}"
            )
        elif expected_tree is None and src_sum.get("source_sha256") != cfg.source_sha256:
            raise RuntimeError(
                f"Source stage summary source_sha256 mismatch: "
                f"summary={src_sum.get('source_sha256')} vs current={cfg.source_sha256}"
            )
        if src_sum.get("common_sha256") != cfg.common_hash:
            raise RuntimeError(
                f"Source stage summary common_sha256 mismatch: "
                f"summary={src_sum.get('common_sha256')} vs current={cfg.common_hash}"
            )
        expected_ckpt_sha = src_sum.get("best_model_sha256")
        if expected_ckpt_sha and expected_ckpt_sha != eval_ckpt_sha:
            raise RuntimeError(
                f"eval_ckpt SHA256 mismatch vs source stage summary: "
                f"ckpt={eval_ckpt_sha} vs summary={expected_ckpt_sha}"
            )
        source_stage_name = str(src_sum.get("stage", ""))
        ckpt_name = eval_ckpt_path.name
        if ckpt_name == "best_unscreened_model.pt" or (
            src_sum.get("best_unscreened_model_sha256") == eval_ckpt_sha
        ):
            checkpoint_role = "best_unscreened"
            if src_sum.get("best_unscreened_round") is not None:
                selected_round_val = int(src_sum["best_unscreened_round"])
        elif ckpt_name == "best_model.pt" or (src_sum.get("best_model_sha256") == eval_ckpt_sha):
            checkpoint_role = "best_selected"
            if src_sum.get("selected_round") is not None:
                selected_round_val = int(src_sum["selected_round"])
        elif ckpt_name == "last_round_model.pt" or (
            src_sum.get("last_round_model_sha256") == eval_ckpt_sha
        ):
            checkpoint_role = "last_round"
            if src_sum.get("last_completed_round") is not None:
                selected_round_val = int(src_sum["last_completed_round"])
        elif ckpt_name.startswith("cost-") and isinstance(src_sum.get("budget_checkpoints"), dict):
            lbl = ckpt_name.removeprefix("cost-").removesuffix(".pt")
            checkpoint_role = f"budget_milestone_{lbl}"
            bc_meta = src_sum["budget_checkpoints"].get(lbl, {})
            if isinstance(bc_meta, dict) and bc_meta.get("selected_round") is not None:
                selected_round_val = int(bc_meta["selected_round"])

    phys_ref_data: dict[str, Any] | None = None
    phys_candidates = [
        Path(physics_test_4h_json_arg) if physics_test_4h_json_arg else None,
        Path(cfg.output_root) / "stage0_physics" / "test_4h.json",
    ]
    for pc in phys_candidates:
        if pc is not None and pc.exists():
            phys_ref_data = json.loads(pc.read_text(encoding="utf-8"))
            break

    t0 = time.time()
    s_cnt_start = get_ddp_solver_window_equivalents(env, device, forbid_whitebox=True)
    eval_model = ForgePOMNet(
        cfg.in_channels, cfg.hidden_channels, cfg.out_channels, cfg.num_blocks
    ).to(device)
    loaded_obj = torch.load(eval_ckpt_path, map_location=device, weights_only=True)
    if isinstance(loaded_obj, dict) and "model_state_dict" in loaded_obj:
        eval_model.load_state_dict(loaded_obj["model_state_dict"])
    else:
        eval_model.load_state_dict(loaded_obj)

    test_4h = evaluate_frozen_test_4h_ddp(
        cfg, env, eval_model, rank, world_size, physics_ref_test_4h=phys_ref_data
    )
    expected_win = float(test_4h["eval_window_equivalents"])
    s_cnt_end = get_ddp_solver_window_equivalents(env, device, forbid_whitebox=True)
    solver_win: float | None = None
    if s_cnt_end["has_counters"]:
        solver_win = float(s_cnt_end["window_equivalents"] - s_cnt_start["window_equivalents"])
        if abs(solver_win - expected_win) > 1e-6:
            raise RuntimeError(
                f"[test_4h] Ledger vs solver substep mismatch: expected={expected_win} vs solver_verified={solver_win}"
            )
    if not all_numeric_leaves_finite(test_4h):
        raise FloatingPointError(f"Non-finite metric detected in test_4h for {eval_ckpt_path.name}")

    out_root_path = Path(cfg.output_root)
    test_4h["status"] = "completed"
    test_4h["protocol_mode"] = "pure_4h"
    test_4h["source_version"] = cfg.source_version
    test_4h["source_sha256"] = cfg.source_sha256
    test_4h["source_tree_sha256"] = cfg.source_tree_sha256
    test_4h["effective_config_sha256"] = cfg.effective_config_sha256
    test_4h["source_bundle_sha256"] = cfg.source_bundle_sha256
    if recorded_source_digests is not None:
        test_4h["recorded_stage_digests"] = recorded_source_digests
    test_4h["common_ckpt"] = relativize_artifact_path(common_ckpt, out_root_path)
    test_4h["common_sha256"] = cfg.common_hash
    test_4h["eval_ckpt_path"] = relativize_artifact_path(eval_ckpt_path, out_root_path)
    test_4h["eval_ckpt_sha256"] = eval_ckpt_sha
    test_4h["source_stage_summary_path"] = (
        relativize_artifact_path(stage_summary_path, out_root_path)
        if stage_summary_path is not None
        else None
    )
    test_4h["source_stage_summary_sha256"] = stage_summary_sha
    test_4h["source_stage"] = source_stage_name
    test_4h["ablation_variant"] = source_ablation_variant
    test_4h["checkpoint_role"] = checkpoint_role
    test_4h["selected_round"] = selected_round_val
    test_4h["solver_verified_window_equivalents"] = solver_win
    test_4h["wall_seconds"] = time.time() - t0
    test_4h["all_metrics_finite"] = True

    out_path = Path(eval_out_json_arg) if eval_out_json_arg else (out_root_path / "test_4h.json")
    if is_main_process(rank):
        save_atomic_json(out_path, test_4h)
        paired_str = ""
        if "paired_vs_physics" in test_4h["test_4h_48"]:
            pvp = test_4h["test_4h_48"]["paired_vs_physics"]
            paired_str = (
                f" | vs_phys={pvp['relative_change_pct']:+.2f}% "
                f"(wins={pvp['wins_vs_physics']}/{pvp['num_windows']})"
            )
        print(
            f"[test_4h Done] ckpt={eval_ckpt_path.name} | test_4h_48={test_4h['loss_4h']:.6f} "
            f"(continuity8={test_4h['continuity8_loss_4h']:.6f}){paired_str} | wall={test_4h['wall_seconds']:.1f}s",
            flush=True,
        )
    if dist.is_initialized():
        dist.barrier()
    return test_4h


__all__ = [
    "BUDGET_FRACTIONS",
    "Pure4hSelectionTracker",
    "can_afford_next_round",
    "evaluate_frozen_test_4h_ddp",
    "evaluate_shared_windows_ddp",
    "evaluate_validation_4h",
    "evaluate_validation_4h_with_worst",
    "evaluate_window_list_ddp",
    "run_standalone_test_4h",
]
