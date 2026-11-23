"""Stage 6 differentiable Neural-Emulator-assisted policy training baseline for 3D POM."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as grad_checkpoint

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
from ..packing import (
    build_obs_wet_mask_161,
    build_unpacking_inv_weights,
    build_volume_weights_60_61,
    build_wet_masks_60_61,
    pack_segment_anomalies,
    unpack_tensor_to_segments,
)
from ..physics_env import BlackBoxProvider3D
from ..probes import HADAMARD_4X4, build_projected_action_probe_direction
from ..sampling import ensure_common_zero_ckpt, sample_epoch_windows


class _EmulatorResBlock2D(nn.Module):
    """Zero-padded local + dilated residual block for closed-basin 3D POM state/segment emulation."""

    def __init__(self, width: int, dilation: int = 2) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 4
        self.norm1 = nn.GroupNorm(groups, width)
        self.conv_local = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.conv_dilated = nn.Conv2d(
            width, width, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(groups, width * 2)
        self.pw = nn.Conv2d(width * 2, width, kernel_size=1)
        self.res_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.norm1(h))
        z = torch.cat([self.conv_local(z), self.conv_dilated(z)], dim=1)
        z = self.pw(F.gelu(self.norm2(z)))
        return h + self.res_scale * z


class NeuralEmulator3D(nn.Module):
    """Learned coarse-step transition and 3-segment incremental output emulator `E_psi`."""

    def __init__(
        self,
        obs_channels: int = 161,
        act_channels: int = 60,
        seg_channels: int = 61,
        width: int = 96,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.obs_channels = obs_channels
        self.act_channels = act_channels
        self.seg_channels = seg_channels
        in_ch = obs_channels + act_channels + seg_channels + 4
        groups = 8 if width % 8 == 0 else 4

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=1),
            nn.GroupNorm(groups, width),
            nn.GELU(),
        )
        dilations = [1, 2, 3, 2][:num_blocks]
        self.blocks = nn.ModuleList([_EmulatorResBlock2D(width, dilation=d) for d in dilations])
        self.out_norm = nn.GroupNorm(groups, width)
        self.head_dx = nn.Conv2d(width, obs_channels, kernel_size=1)
        self.head_dy = nn.Conv2d(width, 3 * seg_channels, kernel_size=1)
        nn.init.zeros_(self.head_dx.weight)
        nn.init.zeros_(self.head_dx.bias)
        nn.init.zeros_(self.head_dy.weight)
        nn.init.zeros_(self.head_dy.bias)

    def forward_hour(
        self,
        x_cur: torch.Tensor,
        g_proj_k: torch.Tensor,
        a_start: torch.Tensor,
        hour_idx: int,
        obs_wet_mask: torch.Tensor,
        mask_61: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, im, jm = x_cur.shape
        phase = torch.zeros((bsz, 4, im, jm), device=x_cur.device, dtype=x_cur.dtype)
        phase[:, int(hour_idx) % 4] = 1.0
        inp = torch.cat([x_cur, g_proj_k, a_start, phase], dim=1)
        h = self.stem(inp)
        for blk in self.blocks:
            h = blk(h)
        h = F.gelu(self.out_norm(h))

        dx = self.head_dx(h) * obs_wet_mask.unsqueeze(0)
        dy_3 = self.head_dy(h).view(bsz, 3, self.seg_channels, im, jm) * mask_61.view(
            1, 1, self.seg_channels, im, jm
        )
        a_seg0 = a_start + dy_3[:, 0]
        a_seg1 = a_seg0 + dy_3[:, 1]
        a_seg2 = a_seg1 + dy_3[:, 2]
        a_segs_k = torch.stack([a_seg0, a_seg1, a_seg2], dim=1)
        x_next = x_cur + dx
        return x_next, a_segs_k

    def unroll_open_loop_4h(
        self,
        x0: torch.Tensor,
        g_proj_seq: torch.Tensor,
        obs_wet_mask: torch.Tensor,
        mask_61: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, im, jm = x0.shape
        x_cur = x0
        a_start = torch.zeros((bsz, self.seg_channels, im, jm), device=x0.device, dtype=x0.dtype)
        x_preds: list[torch.Tensor] = []
        a_preds: list[torch.Tensor] = []
        for k in range(4):
            x_next, a_segs_k = self.forward_hour(
                x_cur, g_proj_seq[:, k], a_start, k, obs_wet_mask, mask_61
            )
            a_preds.append(a_segs_k)
            if k < 3:
                x_preds.append(x_next)
                x_cur = x_next
            a_start = a_segs_k[:, -1]
        return torch.stack(x_preds, dim=1), torch.cat(a_preds, dim=1)


class _EmulatorFitShard3D(nn.Module):
    """DDP-wrappable open-loop trajectory + central-difference Jacobian fit loss on public coarse pairs."""

    def __init__(
        self,
        emulator: NeuralEmulator3D,
        obs_wet_mask: torch.Tensor,
        mask_61: torch.Tensor,
    ) -> None:
        super().__init__()
        self.emulator = emulator
        self.register_buffer("obs_wet_mask", obs_wet_mask.detach().clone(), persistent=False)
        self.register_buffer("mask_61", mask_61.detach().clone(), persistent=False)
        self.wet_x_cnt = float(obs_wet_mask.double().sum().clamp_min(1.0).item())
        self.wet_y_cnt = float(mask_61.double().sum().clamp_min(1.0).item())

    def forward(
        self,
        x0_b: torch.Tensor,
        g_nom_b: torch.Tensor,
        x_nom_tgt: torch.Tensor,
        a_nom_tgt: torch.Tensor,
        g_plus_b: torch.Tensor,
        x_plus_tgt: torch.Tensor,
        a_plus_tgt: torch.Tensor,
        g_minus_b: torch.Tensor,
        x_minus_tgt: torch.Tensor,
        a_minus_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bsz = int(x0_b.shape[0])
        x0_cat = torch.cat([x0_b, x0_b, x0_b], dim=0)
        g_cat = torch.cat([g_nom_b, g_plus_b, g_minus_b], dim=0)
        x_pred_cat, a_pred_cat = self.emulator.unroll_open_loop_4h(
            x0_cat, g_cat, self.obs_wet_mask, self.mask_61
        )
        x_tgt_cat = torch.cat([x_nom_tgt, x_plus_tgt, x_minus_tgt], dim=0)
        a_tgt_cat = torch.cat([a_nom_tgt, a_plus_tgt, a_minus_tgt], dim=0)

        m_x = self.obs_wet_mask.view(1, 1, -1, x0_b.shape[-2], x0_b.shape[-1])
        m_y = self.mask_61.view(1, 1, -1, x0_b.shape[-2], x0_b.shape[-1])

        a_scale_sq = (
            (a_tgt_cat.square() * m_y).sum(dim=(1, 2, 3, 4)) / (12.0 * self.wet_y_cnt)
        ).clamp_min(1e-3)
        loss_a_traj = (
            ((a_pred_cat - a_tgt_cat).square() * m_y).sum(dim=(1, 2, 3, 4))
            / (12.0 * self.wet_y_cnt * a_scale_sq)
        ).mean()

        dx_tgt_cat = x_tgt_cat - torch.cat([x0_cat.unsqueeze(1), x_tgt_cat[:, :2]], dim=1)
        x_scale_sq = (
            (dx_tgt_cat.square() * m_x).sum(dim=(1, 2, 3, 4)) / (3.0 * self.wet_x_cnt)
        ).clamp_min(1e-3)
        loss_x_traj = (
            ((x_pred_cat - x_tgt_cat).square() * m_x).sum(dim=(1, 2, 3, 4))
            / (3.0 * self.wet_x_cnt * x_scale_sq)
        ).mean()

        x_pred_p, x_pred_m = x_pred_cat[bsz : 2 * bsz], x_pred_cat[2 * bsz :]
        a_pred_p, a_pred_m = a_pred_cat[bsz : 2 * bsz], a_pred_cat[2 * bsz :]

        da_true = a_plus_tgt - a_minus_tgt
        da_pred = a_pred_p - a_pred_m
        da_norm_sq = (
            (da_true.square() * m_y).sum(dim=(1, 2, 3, 4)) / (12.0 * self.wet_y_cnt)
        ).clamp_min(1e-6)
        loss_da_diff = (
            ((da_pred - da_true).square() * m_y).sum(dim=(1, 2, 3, 4))
            / (12.0 * self.wet_y_cnt * da_norm_sq)
        ).mean()

        dx_true = x_plus_tgt - x_minus_tgt
        dx_pred = x_pred_p - x_pred_m
        dx_norm_sq = (
            (dx_true.square() * m_x).sum(dim=(1, 2, 3, 4)) / (3.0 * self.wet_x_cnt)
        ).clamp_min(1e-6)
        loss_dx_diff = (
            ((dx_pred - dx_true).square() * m_x).sum(dim=(1, 2, 3, 4))
            / (3.0 * self.wet_x_cnt * dx_norm_sq)
        ).mean()

        total = loss_a_traj + 0.5 * loss_x_traj + 1.5 * loss_da_diff + 0.5 * loss_dx_diff
        return total, {
            "loss_a_traj": loss_a_traj.detach(),
            "loss_x_traj": loss_x_traj.detach(),
            "loss_da_diff": loss_da_diff.detach(),
            "loss_dx_diff": loss_dx_diff.detach(),
        }


@torch.no_grad()
def evaluate_emulator_unseen_3d(
    emulator: NeuralEmulator3D,
    held_pairs: list[dict[str, torch.Tensor]],
    obs_wet_mask: torch.Tensor,
    mask_61: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate emulator accuracy on held-out orthogonal Hadamard action perturbations across ranks."""
    was_tr = emulator.training
    emulator.eval()
    wet_y_cnt = float(mask_61.double().sum().clamp_min(1.0).item())
    m_y = mask_61.double().unsqueeze(0)

    sq_err_roll = 0.0
    sq_ref_roll = 0.0
    sq_err_diff = 0.0
    sq_ref_diff = 0.0
    sq_pred_diff = 0.0
    dot_diff = 0.0
    local_pairs = float(len(held_pairs))

    for item in held_pairs:
        x0 = item["x0"].to(device=device, dtype=torch.float32).unsqueeze(0)
        gp = item["g_plus"].to(device=device, dtype=torch.float32).unsqueeze(0)
        gm = item["g_minus"].to(device=device, dtype=torch.float32).unsqueeze(0)
        ap_true = item["a_plus"].to(device=device, dtype=torch.float64)
        am_true = item["a_minus"].to(device=device, dtype=torch.float64)

        _, a_pred_cat = emulator.unroll_open_loop_4h(
            torch.cat([x0, x0], dim=0),
            torch.cat([gp, gm], dim=0),
            obs_wet_mask,
            mask_61,
        )
        ap_pred = a_pred_cat[0].double()
        am_pred = a_pred_cat[1].double()

        sq_err_roll += float(
            (((ap_pred - ap_true).square() + (am_pred - am_true).square()) * m_y).sum().item()
        )
        sq_ref_roll += float(((ap_true.square() + am_true.square()) * m_y).sum().item())

        da_true = (ap_true - am_true) * m_y
        da_pred = (ap_pred - am_pred) * m_y
        sq_err_diff += float((da_pred - da_true).square().sum().item())
        sq_ref_diff += float(da_true.square().sum().item())
        sq_pred_diff += float(da_pred.square().sum().item())
        dot_diff += float((da_pred * da_true).sum().item())

    if was_tr:
        emulator.train()

    vec = torch.tensor(
        [
            local_pairs,
            sq_err_roll,
            sq_ref_roll,
            sq_err_diff,
            sq_ref_diff,
            sq_pred_diff,
            dot_diff,
        ],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)

    n_pairs = max(float(vec[0].item()), 1.0)
    denom_roll = max(2.0 * n_pairs * 12.0 * wet_y_cnt, 1.0)
    unseen_roll_mse = float(vec[1].item()) / denom_roll
    unseen_roll_rre = math.sqrt(float(vec[1].item()) / max(float(vec[2].item()), 1e-20))
    unseen_diff_rre = math.sqrt(float(vec[3].item()) / max(float(vec[4].item()), 1e-20))
    unseen_diff_cos = float(vec[6].item()) / max(
        math.sqrt(max(float(vec[4].item()), 1e-30) * max(float(vec[5].item()), 1e-30)),
        1e-20,
    )
    return {
        "num_unseen_pairs_global": int(round(n_pairs)),
        "unseen_rollout_mse": unseen_roll_mse,
        "unseen_rollout_rre": unseen_roll_rre,
        "unseen_diff_rre": unseen_diff_rre,
        "unseen_diff_cosine": unseen_diff_cos,
    }


class _EmulatorPolicyShard3D(nn.Module):
    """DDP-wrappable 4-hour closed-loop policy rollout through frozen `NeuralEmulator3D`."""

    def __init__(
        self,
        policy: ForgePOMNet,
        emulator: NeuralEmulator3D,
        env: BlackBoxProvider3D,
        obs_wet_mask: torch.Tensor,
        mask_60: torch.Tensor,
        mask_61: torch.Tensor,
        wq_60: torch.Tensor,
        theta_center: torch.Tensor,
        action_reg_weight: float,
        param_trust_weight: float,
        action_trust_sigma: float,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.emulator = emulator
        self.env = env
        self.inv_weights = build_unpacking_inv_weights(env)
        self.register_buffer("obs_wet_mask", obs_wet_mask.detach().clone(), persistent=False)
        self.register_buffer("mask_60", mask_60.detach().clone(), persistent=False)
        self.register_buffer("mask_61", mask_61.detach().clone(), persistent=False)
        self.register_buffer("wq_60", wq_60.detach().clone(), persistent=False)
        self.register_buffer("theta_center", theta_center.detach().clone(), persistent=False)
        self.wet_60_cnt = float(mask_60.double().sum().clamp_min(1.0).item())
        self.action_reg_weight = float(action_reg_weight)
        self.param_trust_weight = float(param_trust_weight)
        self.action_trust_sigma = float(action_trust_sigma)

    def parameter_trust_term(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.theta_center.device, dtype=torch.float32)
        offset = 0
        for p in self.policy.parameters():
            numel = p.numel()
            ref_slice = self.theta_center[offset : offset + numel]
            denom = ref_slice.square().sum().clamp_min(1.0)
            total = total + (p.reshape(-1) - ref_slice).square().sum() / denom
            offset += numel
        return total

    def forward(
        self,
        local_tasks: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not local_tasks:
            raise ValueError("local_tasks must be non-empty on every DDP rank")

        mse_terms: list[torch.Tensor] = []
        act_sq_terms: list[torch.Tensor] = []
        da_sq_terms: list[torch.Tensor] = []

        for task in local_tasks:
            x_cur = task["x0"].unsqueeze(0)
            a_start = torch.zeros(
                (1, 61, self.env.cfg.im, self.env.cfg.jm),
                device=x_cur.device,
                dtype=x_cur.dtype,
            )
            g_center_seq = task["g_proj_center"]
            a_pred_list: list[torch.Tensor] = []
            act_sq_win = torch.tensor(0.0, device=x_cur.device, dtype=torch.float32)
            da_sq_win = torch.tensor(0.0, device=x_cur.device, dtype=torch.float32)

            for k in range(4):

                def _step_fn(
                    xc: torch.Tensor, ast: torch.Tensor, kk: int = k
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    raw_g = self.policy(xc)[0]
                    g_proj = self.env.project_standardized_action(raw_g) * self.mask_60
                    xn, asegs = self.emulator.forward_hour(
                        xc,
                        g_proj.unsqueeze(0),
                        ast,
                        kk,
                        self.obs_wet_mask,
                        self.mask_61,
                    )
                    return xn, asegs[0], g_proj

                if torch.is_grad_enabled():
                    x_next, a_segs_k, g_proj_k = grad_checkpoint(
                        _step_fn, x_cur, a_start, use_reentrant=False
                    )
                else:
                    x_next, a_segs_k, g_proj_k = _step_fn(x_cur, a_start)

                for s_idx in range(3):
                    a_pred_list.append(a_segs_k[s_idx])
                act_sq_win = act_sq_win + (
                    (g_proj_k.square() * self.wq_60).sum() / (4.0 * self.wet_60_cnt)
                )
                da_sq_win = da_sq_win + (
                    ((g_proj_k - g_center_seq[k]).square() * self.wq_60).sum()
                    / (4.0 * self.wet_60_cnt)
                )
                x_cur = x_next
                a_start = a_segs_k[-1].unsqueeze(0)

            pred_deltas = unpack_tensor_to_segments(self.env, a_pred_list, self.inv_weights)
            win_mse = self.env.compute_preloaded_12seg_loss(
                task["preloaded_targets"],
                pred_deltas,
                cap_per_seg=200.0,
                init_fields=task["init_fields"],
            )
            mse_terms.append(win_mse)
            act_sq_terms.append(act_sq_win)
            da_sq_terms.append(da_sq_win)

        mean_mse = torch.stack(mse_terms).mean()
        mean_act_sq = torch.stack(act_sq_terms).mean()
        mean_da_sq = torch.stack(da_sq_terms).mean()

        param_pen = self.param_trust_weight * self.parameter_trust_term()
        act_reg_pen = self.action_reg_weight * mean_act_sq
        act_trust_pen = F.relu(mean_da_sq - (2.0 * self.action_trust_sigma) ** 2)
        total_loss = mean_mse + act_reg_pen + param_pen + act_trust_pen
        return total_loss, mean_mse.detach(), mean_act_sq.detach(), mean_da_sq.detach()


def _collect_4h_public_trace(
    env: BlackBoxProvider3D,
    member: int,
    day: int,
    action_fn: Any,
    mask_60: torch.Tensor,
) -> dict[str, Any]:
    """Execute one 4-hour coarse-solver rollout and return strictly public coarse tensors."""
    env.reset(member, day)
    init_fields = env.observe_public_scored_fields()
    xs: list[torch.Tensor] = []
    g_projs: list[torch.Tensor] = []
    all_segs: list[dict[str, torch.Tensor]] = []
    for k in range(env.cfg.hours_per_window):
        x_k = env.observe_F()
        raw_g_k = action_fn(x_k, k)
        g_proj_k = env.project_standardized_action(raw_g_k) * mask_60
        segs_k = env.step_H(raw_g_k)
        xs.append(x_k.detach())
        g_projs.append(g_proj_k.detach())
        all_segs.extend(segs_k)
    a_12 = pack_segment_anomalies(env, all_segs, init_fields)
    return {
        "init_fields": init_fields,
        "x_seq": torch.stack(xs, dim=0),
        "g_proj_seq": torch.stack(g_projs, dim=0),
        "a_12": a_12,
        "all_segments": all_segs,
    }


def run_emulator_baseline(
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
    """Stage 6 (Pure-4h): Differentiable Neural-Emulator-Assisted Policy Training (`stage6_emulator`)."""
    out_root_path = Path(cfg.output_root)
    stage_dir = out_root_path / "stage6_emulator"
    summary_path = stage_dir / "summary.json"
    best_ckpt = stage_dir / "best_model.pt"
    emu_ckpt = stage_dir / "emulator_last.pt"
    if summary_path.exists() and best_ckpt.exists():
        if is_main_process(rank):
            print(
                "[Stage 6 Emulator] Already completed, loading summary.json",
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

    emulator = NeuralEmulator3D(
        obs_channels=cfg.in_channels,
        act_channels=cfg.out_channels,
        seg_channels=61,
        width=cfg.emulator_width,
        num_blocks=cfg.emulator_blocks,
    ).to(device)
    emulator_opt = torch.optim.AdamW(emulator.parameters(), lr=cfg.emulator_lr, weight_decay=1e-4)

    mask_60, mask_61 = build_wet_masks_60_61(env)
    wq_60, _ = build_volume_weights_60_61(env)
    obs_wet_mask = build_obs_wet_mask_161(env, mask_60)

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

    replay_buffer: deque[list[dict[str, Any]]] = deque(
        maxlen=max(int(cfg.emulator_buffer_rounds), 1)
    )
    fit_shard = _EmulatorFitShard3D(emulator, obs_wet_mask, mask_61).to(device)
    fit_ddp: nn.Module = (
        DDP(
            fit_shard,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
        )
        if dist.is_initialized()
        else fit_shard
    )

    wet_cnt_60 = float(mask_60.double().sum().clamp_min(1.0).item())
    batch_per_round = int(cfg.emulator_windows_per_gpu) * max(world_size, 1)
    alphas = tuple(float(a) for a in cfg.emulator_step_alphas)
    round_train_val_cost = float(batch_per_round * (9 + len(alphas)) + len(val_windows))
    post_eval_reserve = float(len(val_windows))
    max_rounds = int(cfg.baseline_rounds)

    if is_main_process(rank):
        print(
            f"[Stage 6 Emulator] Starting pure-4h training from common_zero.pt "
            f"(val_4h={tracker.init_val_4h:.6f}, emu_params={sum(p.numel() for p in emulator.parameters()):,}, "
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
        round_wins = sample_epoch_windows(
            train_pool,
            round_idx=r,
            batch_size=batch_per_round,
            seed=cfg.seed,
            stream_id=6061,
        )
        my_windows = round_wins[rank::world_size]

        local_fit_items: list[dict[str, Any]] = []
        held_unseen_items: list[dict[str, torch.Tensor]] = []
        local_policy_tasks: list[dict[str, Any]] = []
        local_nom_loss_sum = 0.0
        local_nom_tr_sum = 0.0

        with torch.no_grad():
            for w_idx, (m, d) in enumerate(my_windows):
                nom_tr = _collect_4h_public_trace(
                    env, m, d, lambda x_k, _k: base_model(x_k), mask_60
                )
                nom_loss_t, nom_met = env.compute_12seg_loss_and_metrics(
                    m, d, nom_tr["all_segments"]
                )
                local_nom_loss_sum += float(nom_loss_t.item())
                local_nom_tr_sum += float(nom_met["tracer_loss_4h"])

                seed_base = cfg.seed + r * 10009 + rank * 193 + w_idx * 29
                d0 = build_projected_action_probe_direction(env, mask_60, seed_base + 1, device)
                d1 = build_projected_action_probe_direction(env, mask_60, seed_base + 2, device)
                d2_raw = (d0 + d1) * mask_60
                d2_rms = math.sqrt(float(d2_raw.double().square().sum().item()) / wet_cnt_60)
                d2 = (
                    (d2_raw / float(d2_rms)).contiguous()
                    if d2_rms > 1e-4
                    else build_projected_action_probe_direction(env, mask_60, seed_base + 3, device)
                )
                spatial_dirs = (d0, d1, d2)
                amp = float(cfg.emulator_probe_amp)

                pair_records: list[dict[str, torch.Tensor]] = []
                for p_idx in range(3):
                    h_col = HADAMARD_4X4[p_idx]
                    sp_dir = spatial_dirs[p_idx]
                    plus_tr = _collect_4h_public_trace(
                        env,
                        m,
                        d,
                        lambda x_k, kk, hc=h_col, sd=sp_dir, am=amp: (
                            base_model(x_k) + (am * float(hc[kk])) * sd
                        ),
                        mask_60,
                    )
                    minus_tr = _collect_4h_public_trace(
                        env,
                        m,
                        d,
                        lambda x_k, kk, hc=h_col, sd=sp_dir, am=amp: (
                            base_model(x_k) - (am * float(hc[kk])) * sd
                        ),
                        mask_60,
                    )
                    pair_records.append(
                        {
                            "g_plus": plus_tr["g_proj_seq"].cpu(),
                            "x_plus_tgt": plus_tr["x_seq"][1:].cpu(),
                            "a_plus_tgt": plus_tr["a_12"].cpu(),
                            "g_minus": minus_tr["g_proj_seq"].cpu(),
                            "x_minus_tgt": minus_tr["x_seq"][1:].cpu(),
                            "a_minus_tgt": minus_tr["a_12"].cpu(),
                        }
                    )

                local_fit_items.append(
                    {
                        "x0": nom_tr["x_seq"][0].cpu(),
                        "g_nom": nom_tr["g_proj_seq"].cpu(),
                        "x_nom_tgt": nom_tr["x_seq"][1:].cpu(),
                        "a_nom_tgt": nom_tr["a_12"].cpu(),
                        "pairs": pair_records,
                    }
                )

                d_unseen = build_projected_action_probe_direction(
                    env, mask_60, seed_base + 7919, device
                )
                h_unseen = HADAMARD_4X4[3]
                unseen_p = _collect_4h_public_trace(
                    env,
                    m,
                    d,
                    lambda x_k, kk, hc=h_unseen, sd=d_unseen, am=amp: (
                        base_model(x_k) + (am * float(hc[kk])) * sd
                    ),
                    mask_60,
                )
                unseen_m = _collect_4h_public_trace(
                    env,
                    m,
                    d,
                    lambda x_k, kk, hc=h_unseen, sd=d_unseen, am=amp: (
                        base_model(x_k) - (am * float(hc[kk])) * sd
                    ),
                    mask_60,
                )
                held_unseen_items.append(
                    {
                        "x0": nom_tr["x_seq"][0].cpu(),
                        "g_plus": unseen_p["g_proj_seq"].cpu(),
                        "a_plus": unseen_p["a_12"].cpu(),
                        "g_minus": unseen_m["g_proj_seq"].cpu(),
                        "a_minus": unseen_m["a_12"].cpu(),
                    }
                )

                local_policy_tasks.append(
                    {
                        "member": int(m),
                        "day": int(d),
                        "x0": nom_tr["x_seq"][0].detach().clone(),
                        "init_fields": {
                            k: v.detach().clone() for k, v in nom_tr["init_fields"].items()
                        },
                        "g_proj_center": nom_tr["g_proj_seq"].detach().clone(),
                        "preloaded_targets": env.preload_window_targets(m, d),
                    }
                )

        train_window_rollouts += 9 * len(my_windows) * world_size
        replay_buffer.append(local_fit_items)

        nom_sync = torch.tensor(
            [local_nom_loss_sum, local_nom_tr_sum], device=device, dtype=torch.float64
        )
        if dist.is_initialized():
            dist.all_reduce(nom_sync, op=dist.ReduceOp.SUM)
        center_loss_4h = float(nom_sync[0].item()) / max(len(round_wins), 1)

        # Step A: Fit NeuralEmulator3D strictly on coarse public input/output pairs
        for p in emulator.parameters():
            p.requires_grad_(True)
        emulator.train()

        n_emu_updates = int(cfg.emulator_warmup_updates) if r == 1 else int(cfg.emulator_updates)
        rng_emu = np.random.default_rng(cfg.seed + r * 30011)
        emu_loss_trace: list[float] = []
        last_emu_parts: dict[str, float] = {}

        for _u in range(n_emu_updates):
            buf_idx = int(rng_emu.integers(len(replay_buffer)))
            pack_round = replay_buffer[buf_idx]
            win_idx = int(rng_emu.integers(len(pack_round)))
            pair_idx = int(rng_emu.integers(3))
            item = pack_round[win_idx]
            pr = item["pairs"][pair_idx]

            emulator_opt.zero_grad(set_to_none=True)
            loss_e, parts_e = fit_ddp(
                item["x0"].unsqueeze(0).to(device=device),
                item["g_nom"].unsqueeze(0).to(device=device),
                item["x_nom_tgt"].unsqueeze(0).to(device=device),
                item["a_nom_tgt"].unsqueeze(0).to(device=device),
                pr["g_plus"].unsqueeze(0).to(device=device),
                pr["x_plus_tgt"].unsqueeze(0).to(device=device),
                pr["a_plus_tgt"].unsqueeze(0).to(device=device),
                pr["g_minus"].unsqueeze(0).to(device=device),
                pr["x_minus_tgt"].unsqueeze(0).to(device=device),
                pr["a_minus_tgt"].unsqueeze(0).to(device=device),
            )
            loss_e.backward()
            torch.nn.utils.clip_grad_norm_(emulator.parameters(), 1.0, error_if_nonfinite=True)
            emulator_opt.step()

            lv = loss_e.detach().double()
            if dist.is_initialized():
                dist.all_reduce(lv, op=dist.ReduceOp.AVG)
            emu_loss_trace.append(float(lv.item()))
            if _u == n_emu_updates - 1:
                last_emu_parts = ddp_mean_dict(
                    {k: float(v.item()) for k, v in parts_e.items()}, device
                )

        unseen_audit = evaluate_emulator_unseen_3d(
            emulator, held_unseen_items, obs_wet_mask, mask_61, device
        )

        # Step B: Freeze emulator and optimize G_theta through 4-hour closed-loop emulator rollout
        emulator.eval()
        for p in emulator.parameters():
            p.requires_grad_(False)

        theta_center = base_model.get_flat_params()
        head_mask, _ = base_model.head_body_index_masks(device=theta_center.device)
        head_norm = max(float(theta_center[head_mask].double().norm().item()), 1.0)
        lr_body = float(cfg.emulator_policy_body_lr) / max(head_norm / 2.0, 1.0)

        for p in base_model.parameters():
            p.requires_grad_(True)
        base_model.train()

        pol_shard = _EmulatorPolicyShard3D(
            policy=base_model,
            emulator=emulator,
            env=env,
            obs_wet_mask=obs_wet_mask,
            mask_60=mask_60,
            mask_61=mask_61,
            wq_60=wq_60,
            theta_center=theta_center,
            action_reg_weight=cfg.action_reg_weight,
            param_trust_weight=cfg.emulator_param_trust_weight,
            action_trust_sigma=cfg.emulator_action_trust_sigma,
        ).to(device)
        pol_ddp: nn.Module = (
            DDP(
                pol_shard,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
            )
            if dist.is_initialized()
            else pol_shard
        )

        with torch.no_grad():
            base_model.eval()
            _, init_emu_mse_t, _, _ = pol_shard(local_policy_tasks)
            if dist.is_initialized():
                dist.all_reduce(init_emu_mse_t, op=dist.ReduceOp.AVG)
            init_emu_mse = float(init_emu_mse_t.item())
            base_model.train()

        head_params = [p for n, p in base_model.named_parameters() if n.startswith("head.")]
        body_params = [p for n, p in base_model.named_parameters() if not n.startswith("head.")]
        pol_opt = torch.optim.Adam(
            [
                {"params": head_params, "lr": float(cfg.emulator_policy_head_lr)},
                {"params": body_params, "lr": lr_body},
            ]
        )

        pol_mse_trace: list[float] = []
        pol_grad_norms: list[float] = []
        head_grad_norms: list[float] = []
        body_grad_norms: list[float] = []

        for _pu in range(int(cfg.emulator_policy_updates)):
            pol_opt.zero_grad(set_to_none=True)
            tot_pol_loss, pol_mse_t, _, _ = pol_ddp(local_policy_tasks)
            tot_pol_loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
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
            pol_opt.step()

            pm = pol_mse_t.detach().double()
            if dist.is_initialized():
                dist.all_reduce(pm, op=dist.ReduceOp.AVG)
            pol_mse_trace.append(float(pm.item()))
            pol_grad_norms.append(float(gnorm.item()) if torch.is_tensor(gnorm) else float(gnorm))
            head_grad_norms.append(hg)
            body_grad_norms.append(bg)

        with torch.no_grad():
            base_model.eval()
            _, prop_emu_mse_t, prop_act_sq_t, prop_da_sq_t = pol_shard(local_policy_tasks)
            if dist.is_initialized():
                dist.all_reduce(prop_emu_mse_t, op=dist.ReduceOp.AVG)
                dist.all_reduce(prop_act_sq_t, op=dist.ReduceOp.AVG)
                dist.all_reduce(prop_da_sq_t, op=dist.ReduceOp.AVG)
            prop_emu_mse = float(prop_emu_mse_t.item())
            prop_act_rms = math.sqrt(max(float(prop_act_sq_t.item()), 0.0))
            prop_da_rms = math.sqrt(max(float(prop_da_sq_t.item()), 0.0))

        theta_prop = base_model.get_flat_params()
        delta_theta = theta_prop - theta_center
        prop_param_delta_norm = float(delta_theta.double().norm().item())

        # Step C: Real-solver candidate step-scale screening against nominal center
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
                "emulator_updates": n_emu_updates,
                "emulator_fit_loss_first": emu_loss_trace[0] if emu_loss_trace else 0.0,
                "emulator_fit_loss_last": emu_loss_trace[-1] if emu_loss_trace else 0.0,
                "emulator_fit_parts": last_emu_parts,
                "unseen_perturbation_audit": unseen_audit,
                "emu_pred_init_loss_4h": init_emu_mse,
                "emu_pred_prop_loss_4h": prop_emu_mse,
                "emu_pred_loss_reduction": init_emu_mse - prop_emu_mse,
                "proposed_action_rms": prop_act_rms,
                "proposed_delta_action_rms": prop_da_rms,
                "proposed_param_delta_norm": prop_param_delta_norm,
                "accepted_param_delta_norm": float(best_alpha) * prop_param_delta_norm,
                "policy_grad_norm_first": pol_grad_norms[0] if pol_grad_norms else 0.0,
                "policy_grad_norm_last": pol_grad_norms[-1] if pol_grad_norms else 0.0,
                "policy_head_grad_norm_max": max(head_grad_norms) if head_grad_norms else 0.0,
                "policy_body_grad_norm_max": max(body_grad_norms) if body_grad_norms else 0.0,
                "center_train_loss_4h": center_loss_4h,
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
                f"  [Stage 6 Emulator r{r:03d}] emu_loss={emu_loss_trace[-1]:.4f} | "
                f"unseen_rre={unseen_audit['unseen_diff_rre']:.3f} (cos={unseen_audit['unseen_diff_cosine']:.3f}) | "
                f"alpha={best_alpha:.2f} | train_4h={center_loss_4h:.5f}->{best_cand_loss:.5f} | "
                f"val_4h={upd['val_loss_4h']:.6f} (best={tracker.selected_val_4h:.6f} @ r{tracker.selected_round}) | "
                f"no_imp={tracker.no_improve} | cum_win={cumulative_window_rollouts:.0f} | {r_wall:.1f}s",
                flush=True,
            )

        if tracker.no_improve >= int(cfg.baseline_patience):
            stop_reason = f"early_stop_patience_{cfg.baseline_patience}"
            break

    base_model.load_state_dict(tracker.selected_state)
    if is_main_process(rank):
        torch.save(emulator.state_dict(), emu_ckpt)

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
            f"[stage6_emulator] Ledger vs solver substep mismatch: ledger_total={total_stage_window_equivalents} "
            f"vs solver_verified={solver_verified_win}"
        )

    summary = {
        "stage": "stage6_emulator",
        "method": "emulator",
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
        "emulator_parameters": int(sum(p.numel() for p in emulator.parameters())),
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
            f"[stage6_emulator Done] selected_round={tracker.selected_round}, "
            f"val_4h={val_4h['loss_4h']:.6f}, cum_win={cumulative_window_rollouts:.0f}, "
            f"wall={summary['wall_seconds']:.1f}s",
            flush=True,
        )
    if dist.is_initialized():
        dist.barrier()
    return summary


run_stage6_emulator = run_emulator_baseline

__all__ = [
    "NeuralEmulator3D",
    "evaluate_emulator_unseen_3d",
    "run_emulator_baseline",
    "run_stage6_emulator",
]
