"""Spatial return aggregation, clipped PPO updates, and KL finalization for Closure-RL."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from ..paired.model import project_action
from .common import array, set_model
from .events import records
from .policy import full_returns, raw_mean


def independent_spatial_statistics(mean, old, latent, sigma):
    """Exact per-grid-point joint-channel Gaussian log-ratio and KL on (B, H, W)."""
    if mean.shape != old.shape or mean.shape != latent.shape or mean.ndim != 4:
        raise ValueError("expected matching (batch, channel, height, width) tensors")
    if not np.isfinite(sigma) or float(sigma) <= 0.0:
        raise ValueError("positive finite exploration sigma required")
    delta = mean - old
    inv_var = 1.0 / (float(sigma) ** 2)
    log_ratio = ((latent - old) * delta - 0.5 * delta.square()).sum(dim=1) * inv_var
    kl = (0.5 * delta.square()).sum(dim=1) * inv_var
    return log_ratio, kl


def _per_trial_window_scores(score_record):
    out = {}
    per_win = np.asarray(score_record["per_window"], dtype=np.float64)
    for idx, trial in enumerate(score_record["trials"]):
        out[int(trial)] = per_win[4 * idx : 4 * idx + 4]
    return out


def collection_arrays_closure_rl(score, baseline, case_name):
    """Build per-callback spatial local-credit batches and verify global aggregation."""
    if score.get("exploration") != "independent":
        raise ValueError("Closure-RL requires conditionally independent spatial exploration")
    obs = []
    latents = []
    means = []
    times = []
    returns = []

    centers = {int(meta["trial"]): z for meta, z in records(baseline)}
    exp_win_scores = _per_trial_window_scores(score)
    base_win_scores = _per_trial_window_scores(baseline)

    max_exp_decomp_err = 0.0
    max_diff_decomp_err = 0.0
    latent_sq_sum = None
    exec_sq_sum = None
    total_pixels = 0

    for meta, z in records(score):
        trial = int(meta["trial"])
        if trial not in centers:
            raise ValueError(f"missing paired deterministic baseline for trial {trial}")
        ids = [np.flatnonzero(z["window"] == q) for q in range(4)]
        if not all(np.array_equal(z["sequence"][ix], z["sequence"][ids[0]]) for ix in ids):
            raise ValueError("joint event order mismatch in Closure-RL")

        costs_exp = np.asarray(z["costs"], dtype=np.float32)
        costs_base = np.asarray(centers[trial]["costs"], dtype=np.float32)
        ret_exp = full_returns(costs_exp, global_credit=False)
        ret_base = full_returns(costs_base, global_credit=False)
        ret = ret_exp - ret_base

        agg_exp = ret_exp[:, 0].mean(axis=(-2, -1), dtype=np.float64)
        agg_diff = ret[:, 0].mean(axis=(-2, -1), dtype=np.float64)
        expected_exp = -exp_win_scores[trial]
        expected_diff = -(exp_win_scores[trial] - base_win_scores[trial])
        err_e = float(np.max(np.abs(agg_exp - expected_exp)))
        err_d = float(np.max(np.abs(agg_diff - expected_diff)))
        if not np.isfinite(err_e) or not np.isfinite(err_d) or max(err_e, err_d) > 1e-4:
            raise ValueError(
                f"spatial local return aggregation mismatch: exp_err={err_e}, diff_err={err_d}"
            )
        max_exp_decomp_err = max(max_exp_decomp_err, err_e)
        max_diff_decomp_err = max(max_diff_decomp_err, err_d)

        z_obs = np.asarray(z["obs"], dtype=np.float32)
        z_latent = np.asarray(z["latent"], dtype=np.float64)
        z_mean = np.asarray(z["mean"], dtype=np.float64)
        z_action = np.asarray(z["action"], dtype=np.float64)

        latent_diff = z_latent - z_mean
        base_proj = project_action(case_name, torch.from_numpy(z_mean)).numpy().astype(np.float64)
        exec_diff = z_action - base_proj
        ch_lat_sq = np.sum(np.square(latent_diff), axis=(0, 2, 3))
        ch_exec_sq = np.sum(np.square(exec_diff), axis=(0, 2, 3))
        latent_sq_sum = ch_lat_sq if latent_sq_sum is None else latent_sq_sum + ch_lat_sq
        exec_sq_sum = ch_exec_sq if exec_sq_sum is None else exec_sq_sum + ch_exec_sq
        total_pixels += int(z_latent.shape[0] * z_latent.shape[2] * z_latent.shape[3])

        for position in range(len(ids[0])):
            for q, ix in enumerate(ids):
                e = int(ix[position])
                ti = int(z["interval"][e])
                obs.append(z_obs[e])
                latents.append(z_latent[e])
                means.append(z_mean[e])
                times.append(ti)
                returns.append(ret[q, ti])

    if not obs or total_pixels == 0:
        raise RuntimeError("no public Closure-RL events")

    obs_arr = np.stack(obs)
    lat_arr = np.stack(latents)
    mean_arr = np.stack(means)
    time_arr = np.asarray(times, dtype=np.int64)
    ret_arr = np.stack(returns)
    diagnostics = {
        "event_frames": int(len(obs_arr)),
        "grid_agent_samples": int(ret_arr.size),
        "latent_exploration_rms_by_channel": np.sqrt(latent_sq_sum / max(total_pixels, 1)).tolist(),
        "effective_action_rms_by_channel": np.sqrt(exec_sq_sum / max(total_pixels, 1)).tolist(),
        "return_mean": float(ret_arr.mean(dtype=np.float64)),
        "return_std": float(ret_arr.std(dtype=np.float64)),
        "spatial_return_decomp_max_abs_err": max_exp_decomp_err,
        "spatial_return_diff_decomp_max_abs_err": max_diff_decomp_err,
    }
    return obs_arr, lat_arr, mean_arr, time_arr, ret_arr, diagnostics


def closure_rl_actor_step(
    model,
    optimizer,
    x,
    latent,
    old,
    advantage,
    sigma,
    forward_model=None,
    target_kl=0.02,
    max_example_kl=0.25,
    clip_eps=0.2,
):
    """Single minibatch per-grid-point clipped PPO update with spatial KL safeguard."""
    eval_mean = (lambda data: raw_mean(model, data)) if forward_model is None else forward_model
    ratio_log, kl_map = independent_spatial_statistics(eval_mean(x), old, latent, sigma)
    pre_kl = float(kl_map.detach().mean().item())
    pre_max_ex_kl = float(kl_map.detach().mean(dim=(1, 2)).max().item())
    if (
        not torch.isfinite(ratio_log).all()
        or not np.isfinite(pre_kl)
        or not np.isfinite(pre_max_ex_kl)
    ):
        raise FloatingPointError("nonfinite Closure-RL pre-step statistics")
    if pre_kl > target_kl or pre_max_ex_kl > max_example_kl:
        return {
            "updated": False,
            "stop_epoch": True,
            "kl": pre_kl,
            "max_example_kl": pre_max_ex_kl,
            "reason": "epoch_KL_stop",
        }
    ratio = ratio_log.exp()
    if not torch.isfinite(ratio).all():
        return {
            "updated": False,
            "stop_epoch": True,
            "kl": pre_kl,
            "max_example_kl": pre_max_ex_kl,
            "reason": "nonfinite_pre_ratio_stop",
        }
    adv_d = advantage.double()
    if adv_d.shape != ratio.shape:
        raise ValueError(
            f"Closure-RL spatial advantage shape {tuple(adv_d.shape)} != ratio shape {tuple(ratio.shape)}"
        )
    clipped_mask = ((adv_d > 0) & (ratio > 1.0 + clip_eps)) | (
        (adv_d < 0) & (ratio < 1.0 - clip_eps)
    )
    clip_fraction = float(clipped_mask.double().mean().item())
    loss = -torch.minimum(ratio * adv_d, ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv_d).mean()
    if bool(clipped_mask.all()) or float((adv_d * (~clipped_mask)).abs().max().item()) == 0.0:
        return {
            "updated": False,
            "stop_epoch": False,
            "kl": pre_kl,
            "post_kl": pre_kl,
            "max_example_post_kl": pre_max_ex_kl,
            "loss": float(loss.detach()),
            "gradient_norm": 0.0,
            "clip_fraction": clip_fraction,
            "shrink_factor": 0.0,
            "skipped_fully_clipped": True,
            "reason": "skipped_fully_clipped",
        }

    prev_theta = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    prev_opt_state = copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    if float(norm) == 0.0:
        raise RuntimeError("zero Closure-RL gradient; no learning claim")
    optimizer.step()

    shrink = 1.0
    with torch.no_grad():
        post_rlog, post_kl_map = independent_spatial_statistics(eval_mean(x), old, latent, sigma)
        post_kl = float(post_kl_map.detach().mean().item())
        post_max_ex_kl = float(post_kl_map.detach().mean(dim=(1, 2)).max().item())
        post_ratio_ok = bool(
            torch.isfinite(post_rlog).all() and torch.isfinite(post_rlog.exp()).all()
        )
        if (
            not post_ratio_ok
            or not np.isfinite(post_kl)
            or post_kl > target_kl
            or post_max_ex_kl > max_example_kl
        ):
            if pre_kl >= 0.75 * target_kl:
                torch.nn.utils.vector_to_parameters(prev_theta, model.parameters())
                optimizer.load_state_dict(prev_opt_state)
                return {
                    "updated": False,
                    "stop_epoch": True,
                    "kl": post_kl,
                    "pre_kl": pre_kl,
                    "reason": "epoch_KL_stop",
                }
            full_theta = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
            delta_theta = full_theta - prev_theta
            alpha = (
                min(
                    0.5,
                    math.sqrt(
                        max(0.5 * target_kl - pre_kl, 0.1 * target_kl) / max(post_kl, target_kl)
                    ),
                )
                if np.isfinite(post_kl) and post_kl > 0.0
                else 0.5
            )
            accepted_step = False
            for _ in range(10):
                torch.nn.utils.vector_to_parameters(
                    prev_theta + alpha * delta_theta, model.parameters()
                )
                post_rlog, post_kl_map = independent_spatial_statistics(
                    eval_mean(x), old, latent, sigma
                )
                post_kl = float(post_kl_map.detach().mean().item())
                post_max_ex_kl = float(post_kl_map.detach().mean(dim=(1, 2)).max().item())
                post_ratio_ok = bool(
                    torch.isfinite(post_rlog).all() and torch.isfinite(post_rlog.exp()).all()
                )
                if (
                    post_ratio_ok
                    and np.isfinite(post_kl)
                    and post_kl <= target_kl
                    and post_max_ex_kl <= max_example_kl
                ):
                    shrink = float(alpha)
                    accepted_step = True
                    break
                alpha *= 0.5
            if not accepted_step:
                torch.nn.utils.vector_to_parameters(prev_theta, model.parameters())
                optimizer.load_state_dict(prev_opt_state)
                return {
                    "updated": False,
                    "stop_epoch": True,
                    "kl": post_kl,
                    "pre_kl": pre_kl,
                    "reason": "epoch_KL_stop",
                }
    return {
        "updated": True,
        "stop_epoch": False,
        "kl": pre_kl,
        "post_kl": post_kl,
        "max_example_post_kl": post_max_ex_kl,
        "loss": float(loss.detach()),
        "gradient_norm": float(norm),
        "clip_fraction": clip_fraction,
        "shrink_factor": shrink,
    }


@torch.no_grad()
def finalize_closure_rl_kl(
    model,
    center,
    obs,
    latents,
    means,
    sigma,
    dev,
    forward_model=None,
    target_kl=0.02,
    max_example_kl=0.25,
):
    """Verify and enforce spatial KL bounds over all collected events after float32 cast."""
    eval_mean = (lambda data: raw_mean(model, data)) if forward_model is None else forward_model
    raw_proposed = array(model).astype(np.float32)
    chunk_size = 64

    def evaluate_theta(theta_vec):
        set_model(model, theta_vec)
        example_kls = []
        max_grid_kl = 0.0
        action_delta_sq = 0.0
        pixel_count = 0
        for start in range(0, len(obs), chunk_size):
            end = min(start + chunk_size, len(obs))
            x_c = torch.as_tensor(np.array(obs[start:end]), device=dev).double()
            u_c = torch.as_tensor(np.array(latents[start:end]), device=dev, dtype=torch.float64)
            m_c = torch.as_tensor(np.array(means[start:end]), device=dev, dtype=torch.float64)
            new_m = eval_mean(x_c)
            rlog_c, kl_c = independent_spatial_statistics(new_m, m_c, u_c, sigma)
            if not torch.isfinite(rlog_c).all() or not torch.isfinite(kl_c).all():
                return float("inf"), float("inf"), float("inf"), 0.0, False
            ex_kl = kl_c.mean(dim=(1, 2))
            example_kls.append(ex_kl.detach().cpu().numpy())
            max_grid_kl = max(max_grid_kl, float(kl_c.max().item()))
            action_delta_sq += float((new_m - m_c).square().sum().item())
            pixel_count += int(new_m.numel())
        all_ex = np.concatenate(example_kls) if example_kls else np.zeros(1)
        mean_kl = float(all_ex.mean())
        max_ex = float(all_ex.max())
        act_rms = math.sqrt(action_delta_sq / max(pixel_count, 1))
        ok = bool(np.isfinite(mean_kl) and np.isfinite(max_ex) and np.isfinite(max_grid_kl))
        return mean_kl, max_ex, max_grid_kl, act_rms, ok

    init_kl, init_max_ex, init_max_grid, act_rms, finite = evaluate_theta(raw_proposed)
    if finite and init_kl <= target_kl and init_max_ex <= max_example_kl:
        return raw_proposed, {
            "initial_full_dataset_post_kl": init_kl,
            "full_dataset_post_kl": init_kl,
            "max_example_post_kl": init_max_ex,
            "max_grid_point_post_kl": init_max_grid,
            "deterministic_action_delta_rms": act_rms,
            "full_dataset_shrink": 1.0,
            "total_events": int(len(obs)),
        }
    delta = raw_proposed.astype(np.float64) - np.asarray(center, dtype=np.float64)
    alpha = (
        min(0.5, math.sqrt((0.75 * target_kl) / max(init_kl, target_kl)))
        if np.isfinite(init_kl) and init_kl > 0.0
        else 0.5
    )
    for _ in range(10):
        cand = (np.asarray(center, dtype=np.float64) + alpha * delta).astype(np.float32)
        cand_kl, cand_max_ex, cand_max_grid, cand_act_rms, cand_ok = evaluate_theta(cand)
        if cand_ok and cand_kl <= target_kl and cand_max_ex <= max_example_kl:
            return cand, {
                "initial_full_dataset_post_kl": init_kl,
                "full_dataset_post_kl": cand_kl,
                "max_example_post_kl": cand_max_ex,
                "max_grid_point_post_kl": cand_max_grid,
                "deterministic_action_delta_rms": cand_act_rms,
                "full_dataset_shrink": float(alpha),
                "total_events": int(len(obs)),
            }
        alpha *= 0.5
    fallback = np.asarray(center, dtype=np.float32).copy()
    set_model(model, fallback)
    return fallback, {
        "initial_full_dataset_post_kl": init_kl,
        "full_dataset_post_kl": 0.0,
        "max_example_post_kl": 0.0,
        "max_grid_point_post_kl": 0.0,
        "deterministic_action_delta_rms": 0.0,
        "full_dataset_shrink": 0.0,
        "total_events": int(len(obs)),
    }


def summarize_closure_rl_checks(checks, center, proposed, accepted, collection_diag):
    step_rows = [c for c in checks if "updated" in c]
    updated_rows = [c for c in step_rows if c.get("updated")]
    full_kl = next(
        (c["full_dataset_kl_check"] for c in reversed(checks) if "full_dataset_kl_check" in c),
        {},
    )
    return {
        **collection_diag,
        "minibatch_steps_attempted": len(step_rows),
        "minibatch_steps_updated": len(updated_rows),
        "mean_pre_kl": (float(np.mean([c["kl"] for c in updated_rows])) if updated_rows else 0.0),
        "mean_post_kl": (
            float(np.mean([c["post_kl"] for c in updated_rows])) if updated_rows else 0.0
        ),
        "mean_clip_fraction": (
            float(np.mean([c["clip_fraction"] for c in updated_rows])) if updated_rows else 0.0
        ),
        "full_dataset_post_kl": full_kl.get("full_dataset_post_kl", 0.0),
        "max_example_post_kl": full_kl.get("max_example_post_kl", 0.0),
        "max_grid_point_post_kl": full_kl.get("max_grid_point_post_kl", 0.0),
        "proposed_action_delta_rms": full_kl.get("deterministic_action_delta_rms", 0.0),
        "proposed_parameter_delta_l2": float(
            np.linalg.norm(
                np.asarray(proposed, dtype=np.float64) - np.asarray(center, dtype=np.float64)
            )
        ),
        "accepted_parameter_delta_l2": (
            float(
                np.linalg.norm(
                    np.asarray(proposed, dtype=np.float64) - np.asarray(center, dtype=np.float64)
                )
            )
            if accepted
            else 0.0
        ),
    }


__all__ = [
    "closure_rl_actor_step",
    "collection_arrays_closure_rl",
    "finalize_closure_rl_kl",
    "independent_spatial_statistics",
    "summarize_closure_rl_checks",
]
