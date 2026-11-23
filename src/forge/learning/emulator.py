"""Differentiable Neural-Emulator-assisted policy training (public action interface adaptation)."""

from __future__ import annotations

import copy
import gc
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .common import array, good, indices, load_model, set_model
from .distributed import engine, rank, size, wrap
from .emulator_data import (
    _deserialize_pack,
    _serialize_pack,
    evaluate_emulator_unseen,
    extract_emulator_trace_pack,
    extract_policy_task_pack,
)
from .emulator_net import (
    NeuralEmulator,
    _EmulatorFitShard,
    _EmulatorPolicyShard,
    _interval_ranges,
    build_callback_phase_table,
)
from .events import close_record_cache


def _train_emulator_and_propose(
    emulator,
    emulator_opt,
    actor_model,
    theta_center,
    fit_packs,
    held_pack,
    task_pack,
    recipe,
    round_idx,
    device,
    use_ddp=False,
):
    begun = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    emulator.train()
    for param in emulator.parameters():
        param.requires_grad_(True)
    fit_shard = _EmulatorFitShard(emulator).to(device)
    fit_forward = wrap(fit_shard) if use_ddp else fit_shard

    n_updates = recipe.emulator_warmup_updates if round_idx == 0 else recipe.emulator_updates
    rng = np.random.default_rng(910000 + round_idx)
    world = size() if use_ddp else 1
    my_rank = rank() if use_ddp else 0
    local_batch = 2

    intervals = fit_packs[-1]["intervals"]
    output_after = fit_packs[-1]["output_after"]
    ranges = _interval_ranges(intervals, output_after)
    phase_full = build_callback_phase_table(intervals, output_after)
    n_intervals = len(ranges)
    seg_intervals = min(2, n_intervals)

    emu_losses = []
    for _ in range(n_updates):
        pack = fit_packs[int(rng.integers(len(fit_packs)))]
        pert = pack["perturbations"][int(rng.integers(len(pack["perturbations"])))]
        k_start = int(rng.integers(0, n_intervals - seg_intervals + 1))
        t_start = ranges[k_start][0]
        t_end = ranges[k_start + seg_intervals - 1][1]
        total_steps = pack["center_obs"].shape[0]
        obs_end_idx = min(t_end + 1, total_steps)
        rel_outputs = [
            int(af - t_start) for af in output_after.tolist() if t_start <= int(af) < t_end
        ]
        out_k_indices = [
            idx_k for idx_k, af in enumerate(output_after.tolist()) if t_start <= int(af) < t_end
        ]

        n_parents = pack["center_obs"].shape[1]
        perm = rng.permutation(n_parents)
        need = world * local_batch
        chosen = np.resize(perm, need)[my_rank * local_batch : (my_rank + 1) * local_batch]

        c_obs = pack["center_obs"]
        c_act = pack["center_act"]
        c_fld = pack["center_fields"]
        p_obs = pert["observations"]
        p_act = pert["actions"]
        p_fld = pert["fields"]

        o0_init = torch.as_tensor(c_obs[t_start, chosen], device=device)
        o0_prev = torch.as_tensor(c_obs[max(t_start - 1, 0), chosen], device=device)
        a0_prev = (
            torch.as_tensor(c_act[t_start - 1, chosen], device=device)
            if t_start > 0
            else torch.zeros(
                (len(chosen), emulator.action_channels, *o0_init.shape[-2:]),
                device=device,
            )
        )
        a0_seq = torch.as_tensor(c_act[t_start:t_end, chosen], device=device)

        op_init = torch.as_tensor(p_obs[t_start, chosen], device=device)
        op_prev = torch.as_tensor(p_obs[max(t_start - 1, 0), chosen], device=device)
        ap_prev = (
            torch.as_tensor(p_act[t_start - 1, chosen], device=device)
            if t_start > 0
            else torch.zeros_like(a0_prev)
        )
        ap_seq = torch.as_tensor(p_act[t_start:t_end, chosen], device=device)

        targ_o0 = torch.as_tensor(c_obs[t_start + 1 : obs_end_idx, chosen], device=device)
        targ_op = torch.as_tensor(p_obs[t_start + 1 : obs_end_idx, chosen], device=device)
        targ_f0 = torch.as_tensor(c_fld[out_k_indices][:, chosen], device=device)
        targ_fp = torch.as_tensor(p_fld[out_k_indices][:, chosen], device=device)
        p_seq = torch.as_tensor(phase_full[t_start:t_end], device=device)

        emulator_opt.zero_grad(set_to_none=True)
        loss = fit_forward(
            o0_init,
            o0_prev,
            a0_prev,
            a0_seq,
            targ_o0,
            targ_f0,
            op_init,
            op_prev,
            ap_prev,
            ap_seq,
            targ_op,
            targ_fp,
            p_seq,
            rel_outputs,
            pert["sigma"],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(emulator.parameters(), 1.0, error_if_nonfinite=True)
        emulator_opt.step()

        loss_val = loss.detach().double()
        if use_ddp:
            dist.all_reduce(loss_val, op=dist.ReduceOp.AVG)
        emu_losses.append(float(loss_val.item()))

    unseen_metrics = evaluate_emulator_unseen(emulator, held_pack, device)

    emulator.eval()
    for param in emulator.parameters():
        param.requires_grad_(False)

    set_model(actor_model, theta_center)
    actor_model.eval()
    for param in actor_model.parameters():
        param.requires_grad_(True)

    policy_shard = _EmulatorPolicyShard(
        actor_model,
        emulator,
        recipe.case_name,
        task_pack["intervals"],
        task_pack["output_after"],
        theta_center,
        recipe.sigma_action,
    ).to(device)
    policy_forward = wrap(policy_shard) if use_ddp else policy_shard

    n_task = len(task_pack["initial_obs"])
    ids = np.arange(my_rank, n_task, world)
    init_obs_local = torch.as_tensor(task_pack["initial_obs"][ids], device=device)
    ref_local = torch.as_tensor(task_pack["reference"][ids], device=device)
    base_act0_local = torch.as_tensor(task_pack["base_initial_act"][ids], device=device)

    with torch.no_grad():
        _, init_emu_mse_t, _, _ = policy_shard(init_obs_local, ref_local, base_act0_local)
        if use_ddp:
            dist.all_reduce(init_emu_mse_t, op=dist.ReduceOp.AVG)
        init_emu_mse = float(init_emu_mse_t.item())

    head_idx, _, _ = indices(actor_model)
    head_norm = max(float(np.linalg.norm(theta_center[head_idx])), 1.0)
    lr_body = recipe.emulator_policy_body_lr / max(head_norm / 2.0, 1.0)
    head_params = [p for n, p in actor_model.named_parameters() if ".head." in n]
    body_params = [p for n, p in actor_model.named_parameters() if ".head." not in n]
    policy_opt = torch.optim.Adam(
        [
            {"params": head_params, "lr": recipe.emulator_policy_head_lr},
            {"params": body_params, "lr": lr_body},
        ]
    )

    policy_losses = []
    grad_norms = []
    for _ in range(recipe.emulator_policy_updates):
        policy_opt.zero_grad(set_to_none=True)
        total_loss, mse_t, _, _ = policy_forward(init_obs_local, ref_local, base_act0_local)
        total_loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            actor_model.parameters(), 1.0, error_if_nonfinite=True
        )
        policy_opt.step()

        lv = mse_t.detach().double()
        if use_ddp:
            dist.all_reduce(lv, op=dist.ReduceOp.AVG)
        policy_losses.append(float(lv.item()))
        grad_norms.append(float(gnorm))

    with torch.no_grad():
        _, final_emu_mse_t, final_act_sq_t, final_da0_sq_t = policy_shard(
            init_obs_local, ref_local, base_act0_local
        )
        if use_ddp:
            dist.all_reduce(final_emu_mse_t, op=dist.ReduceOp.AVG)
            dist.all_reduce(final_act_sq_t, op=dist.ReduceOp.AVG)
            dist.all_reduce(final_da0_sq_t, op=dist.ReduceOp.AVG)
        final_emu_mse = float(final_emu_mse_t.item())
        final_act_rms = math.sqrt(max(float(final_act_sq_t.item()), 0.0))
        final_da0_rms = math.sqrt(max(float(final_da0_sq_t.item()), 0.0))
        param_trust_pen = float((1e-3 * policy_shard.parameter_trust_term()).item())
        act_trust_pen = float(policy_shard.initial_action_trust_term(final_da0_sq_t).item())

    proposed = array(actor_model).astype(np.float32)
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    info = {
        "emulator_parameters": int(sum(p.numel() for p in emulator.parameters())),
        "emulator_updates": int(n_updates),
        "emulator_fit_loss_first": float(emu_losses[0]) if emu_losses else 0.0,
        "emulator_fit_loss_last": float(emu_losses[-1]) if emu_losses else 0.0,
        "unseen_perturbation_audit": unseen_metrics,
        "policy_updates": int(recipe.emulator_policy_updates),
        "emulator_predicted_initial_mse": init_emu_mse,
        "emulator_predicted_proposed_mse": final_emu_mse,
        "emulator_predicted_mse_reduction": init_emu_mse - final_emu_mse,
        "parameter_trust_penalty": param_trust_pen,
        "initial_action_trust_penalty": act_trust_pen,
        "policy_loss_trace": policy_losses,
        "policy_grad_norm_first": float(grad_norms[0]) if grad_norms else 0.0,
        "policy_grad_norm_last": float(grad_norms[-1]) if grad_norms else 0.0,
        "proposed_closed_loop_action_rms": final_act_rms,
        "proposed_initial_action_delta_rms": final_da0_rms,
        "proposed_parameter_delta_l2": float(
            np.linalg.norm(proposed.astype(np.float64) - np.asarray(theta_center, dtype=np.float64))
        ),
        "gpu_step_wall_seconds": time.monotonic() - begun,
        "gpu_step_peak_allocated_bytes": peak_bytes,
    }
    return proposed, info


def _release_emulator_caches():
    """Close open record mmap handles between rounds."""
    close_record_cache()
    gc.collect()


def _budget_exhausted_for_next_round(s, round_train_windows):
    anchor = int(getattr(s.recipe, "window_budget_anchor", 0) or 0)
    if anchor <= 0:
        return False
    val_windows = 4 * int(s.client.info["trials"]["validation"])
    train_once_windows = 4 * int(s.client.info["trials"]["train"])
    needed = round_train_windows + val_windows + train_once_windows
    return int(getattr(s, "cumulative_window_attempts", 0)) + needed > anchor


def _collect_round_emulator_traces(s, center, emu_trials, k):
    sigma = 0.02
    seed_base = 950000 + 100 * k
    base = s.score(
        center,
        trials=emu_trials,
        trace=True,
        coherent=True,
        label=f"emu-r{k}-center",
    )
    if not good(base):
        return base, [], [], False

    fit_scores = []
    for seed, amp, coh in (
        (seed_base + 0, sigma, True),
        (seed_base + 1, 0.5 * sigma, True),
        (seed_base + 2, sigma, False),
    ):
        for sign in (-1, 1):
            sc = s.score(
                center,
                trials=emu_trials,
                sigma=sign * amp,
                seed=seed,
                trace=True,
                coherent=coh,
                label=f"emu-r{k}-fit-{seed}-{sign}-{'coh' if coh else 'sub'}",
            )
            fit_scores.append(sc)

    held_scores = []
    held_seed = (seed_base // 10) * 10 + 7
    for sign in (-1, 1):
        sc = s.score(
            center,
            trials=emu_trials,
            sigma=sign * sigma,
            seed=held_seed,
            trace=True,
            coherent=True,
            label=f"emu-r{k}-held-{held_seed}-{sign}",
        )
        held_scores.append(sc)

    all_ok = all(good(sc) for sc in [*fit_scores, *held_scores])
    return base, fit_scores, held_scores, all_ok


def run(s):
    eng = engine()
    if eng is not None:
        return run_distributed(s, eng)

    s.initialize()
    dev = next(s.model.parameters()).device
    channels = s.recipe.observation_channels
    emulator = None
    emulator_opt = None
    actor_model = copy.deepcopy(s.model).to(dev).eval()
    fit_buffer = []

    n_emu_groups = min(8, s.client.info["trials"]["train"])
    n_batch_groups = min(s.recipe.training_groups_per_round, s.client.info["trials"]["train"])
    est_round_windows = 9 * (4 * n_emu_groups) + 4 * (4 * n_batch_groups)

    for k in range(s.recipe.fine_rounds):
        if (
            k > 0
            and k % s.recipe.validation_every_rounds == 0
            and _budget_exhausted_for_next_round(s, est_round_windows)
        ):
            break
        center = s.center.copy()
        set_model(s.model, center)
        trials = s.batch()
        emu_trials = trials[:n_emu_groups]

        base, fit_scores, held_scores, ok = _collect_round_emulator_traces(s, center, emu_trials, k)
        if not ok:
            _release_emulator_caches()
            stop = s.validate(
                k + 1,
                {
                    "training_trials": trials,
                    "emulator_trials": emu_trials,
                    "emulator_collection": {
                        "center": base,
                        "fit_collection": fit_scores,
                        "held_collection": held_scores,
                    },
                    "status": "trace_collection_failed",
                    "accepted": False,
                },
            )
            if stop:
                break
            continue

        fit_pack = extract_emulator_trace_pack(base["trace_paths"], fit_scores)
        held_pack = extract_emulator_trace_pack(base["trace_paths"], held_scores)
        task_pack = extract_policy_task_pack(base["trace_paths"])
        fit_buffer.append(fit_pack)
        if len(fit_buffer) > 2:
            fit_buffer.pop(0)

        if emulator is None:
            nx = int(s.model.network.config.nx)
            emulator = NeuralEmulator(
                s.recipe.case_name,
                channels,
                s.recipe.action_channels,
                s.recipe.field_channels,
                modes=min(8, max(1, nx // 2)),
                obs_mean=s.model.network.input_mean[:, :channels].detach().cpu(),
                obs_std=s.model.network.input_std[:, :channels].detach().cpu(),
                field_std=task_pack["field_std"],
            ).to(dev)
            emulator_opt = torch.optim.AdamW(
                emulator.parameters(), lr=s.recipe.emulator_lr, weight_decay=1e-5
            )

        proposed, step_info = _train_emulator_and_propose(
            emulator,
            emulator_opt,
            actor_model,
            center,
            fit_buffer,
            held_pack,
            task_pack,
            s.recipe,
            k,
            dev,
            use_ddp=False,
        )
        torch.save(
            {
                "emulator_state_dict": emulator.state_dict(),
                "round": k + 1,
                "unseen_perturbation_audit": step_info["unseen_perturbation_audit"],
            },
            s.folder / "latest-emulator.pt",
        )

        screen_center = s.score(center, trials=trials, label=f"emu-r{k}-screen-center")
        candidates = []
        screen_rows = []
        if not np.array_equal(proposed, center):
            for alpha in (1.0, 0.5, 0.25):
                cand = (center + alpha * (proposed - center)).astype(np.float32)
                sc = s.score(cand, trials=trials, label=f"emu-r{k}-screen-{alpha}")
                beats = good(sc) and sc["mean_mse"] < screen_center["mean_mse"]
                screen_rows.append(
                    {
                        "alpha": alpha,
                        "score": sc,
                        "beats_center": beats,
                        "real_solver_mse_reduction": (
                            float(screen_center["mean_mse"] - sc["mean_mse"]) if good(sc) else None
                        ),
                    }
                )
                if beats:
                    candidates.append((sc["mean_mse"], cand, alpha))
        accepted = bool(candidates)
        selected_alpha = None
        if accepted:
            _, s.center, selected_alpha = min(candidates, key=lambda x: x[0])
            s.center = s.center.copy()
        set_model(s.model, s.center)
        _release_emulator_caches()

        stop = s.validate(
            k + 1,
            {
                "training_trials": trials,
                "emulator_trials": emu_trials,
                "emulator_collection": {
                    "center": base,
                    "fit_collection": fit_scores,
                    "held_collection": held_scores,
                },
                "emulator_step": step_info,
                "screen_center": screen_center,
                "candidate_screens": screen_rows,
                "selected_alpha": selected_alpha,
                "accepted": accepted,
                "refresh_schedule": "every round: 8 training groups (32 windows) center + 6 smooth perturbed fit rollouts + 2 unseen DCT-IV held-out rollouts; 2-round rolling replay buffer",
            },
        )
        if stop:
            break
    return s.finish()


def run_distributed(s, execution):
    """8-GPU DDP Neural-Emulator training and closed-loop policy optimization."""
    s.initialize()
    n_emu_groups = min(8, s.client.info["trials"]["train"])
    n_batch_groups = min(s.recipe.training_groups_per_round, s.client.info["trials"]["train"])
    est_round_windows = 9 * (4 * n_emu_groups) + 4 * (4 * n_batch_groups)

    buffer_root = Path(tempfile.mkdtemp(prefix="forge-emu-buf-"))
    fit_pack_dirs = []
    try:
        for k in range(s.recipe.fine_rounds):
            if (
                k > 0
                and k % s.recipe.validation_every_rounds == 0
                and _budget_exhausted_for_next_round(s, est_round_windows)
            ):
                break
            center = s.center.copy()
            set_model(s.model, center)
            trials = s.batch()
            emu_trials = trials[:n_emu_groups]

            base, fit_scores, held_scores, ok = _collect_round_emulator_traces(
                s, center, emu_trials, k
            )
            if not ok:
                _release_emulator_caches()
                stop = s.validate(
                    k + 1,
                    {
                        "training_trials": trials,
                        "emulator_trials": emu_trials,
                        "emulator_collection": {
                            "center": base,
                            "fit_collection": fit_scores,
                            "held_collection": held_scores,
                        },
                        "status": "trace_collection_failed",
                        "accepted": False,
                    },
                )
                if stop:
                    break
                continue

            fit_pack = extract_emulator_trace_pack(base["trace_paths"], fit_scores)
            held_pack = extract_emulator_trace_pack(base["trace_paths"], held_scores)
            task_pack = extract_policy_task_pack(base["trace_paths"])

            round_fit_dir = _serialize_pack(fit_pack, buffer_root / f"fit-r{k}")
            fit_pack_dirs.append(round_fit_dir)
            while len(fit_pack_dirs) > 2:
                old_dir = fit_pack_dirs.pop(0)
                shutil.rmtree(old_dir, ignore_errors=True)

            with tempfile.TemporaryDirectory(prefix="forge-emu-step-") as scratch:
                held_dir = _serialize_pack(held_pack, Path(scratch) / "held")
                task_dir = Path(scratch) / "task"
                task_dir.mkdir(parents=True, exist_ok=True)
                for name in (
                    "initial_obs",
                    "reference",
                    "base_initial_act",
                    "field_std",
                    "intervals",
                    "output_after",
                ):
                    np.save(task_dir / f"{name}.npy", task_pack[name])

                proposed, step_info = execution.call(
                    "emulator_step",
                    {
                        "theta": center,
                        "recipe": s.recipe,
                        "round": k,
                        "fit_dirs": list(fit_pack_dirs),
                        "held_dir": held_dir,
                        "task_dir": str(task_dir),
                        "emulator_checkpoint": str(s.folder / "latest-emulator.pt"),
                    },
                )

            _release_emulator_caches()

            screen_center = s.score(center, trials=trials, label=f"emu-r{k}-screen-center")
            candidates = []
            screen_rows = []
            if not np.array_equal(proposed, center):
                for alpha in (1.0, 0.5, 0.25):
                    cand = (center + alpha * (proposed - center)).astype(np.float32)
                    sc = s.score(cand, trials=trials, label=f"emu-r{k}-screen-{alpha}")
                    beats = good(sc) and sc["mean_mse"] < screen_center["mean_mse"]
                    screen_rows.append(
                        {
                            "alpha": alpha,
                            "score": sc,
                            "beats_center": beats,
                            "real_solver_mse_reduction": (
                                float(screen_center["mean_mse"] - sc["mean_mse"])
                                if good(sc)
                                else None
                            ),
                        }
                    )
                    if beats:
                        candidates.append((sc["mean_mse"], cand, alpha))
            accepted = bool(candidates)
            selected_alpha = None
            if accepted:
                _, s.center, selected_alpha = min(candidates, key=lambda x: x[0])
                s.center = s.center.copy()
            set_model(s.model, s.center)

            stop = s.validate(
                k + 1,
                {
                    "training_trials": trials,
                    "emulator_trials": emu_trials,
                    "emulator_collection": {
                        "center": base,
                        "fit_collection": fit_scores,
                        "held_collection": held_scores,
                    },
                    "emulator_step": step_info,
                    "screen_center": screen_center,
                    "candidate_screens": screen_rows,
                    "selected_alpha": selected_alpha,
                    "accepted": accepted,
                    "refresh_schedule": "every round: 8 training groups (32 windows) center + 6 smooth perturbed fit rollouts + 2 unseen DCT-IV held-out rollouts; 2-round rolling replay buffer",
                    "execution": "persistent DDP",
                },
            )
            if stop:
                break
        return s.finish()
    finally:
        shutil.rmtree(buffer_root, ignore_errors=True)


def emulator_ddp_job(e, p):
    """Collective DDP job for fitting E_psi on public traces and optimizing G_theta."""
    recipe = p["recipe"]
    task_dir = Path(p["task_dir"])
    task_pack = {
        name: np.load(task_dir / f"{name}.npy")
        for name in (
            "initial_obs",
            "reference",
            "base_initial_act",
            "field_std",
            "intervals",
            "output_after",
        )
    }
    if "emulator" not in e.state:
        actor_model, _ = load_model(e.state["checkpoint"], e.device)
        actor_model = actor_model.float().eval()
        channels = recipe.observation_channels
        nx = int(actor_model.network.config.nx)
        torch.manual_seed(recipe.seed + 707)
        emulator = NeuralEmulator(
            recipe.case_name,
            channels,
            recipe.action_channels,
            recipe.field_channels,
            modes=min(8, max(1, nx // 2)),
            obs_mean=actor_model.network.input_mean[:, :channels].detach().cpu(),
            obs_std=actor_model.network.input_std[:, :channels].detach().cpu(),
            field_std=task_pack["field_std"],
        ).to(e.device)
        emulator_opt = torch.optim.AdamW(
            emulator.parameters(), lr=recipe.emulator_lr, weight_decay=1e-5
        )
        e.state["emulator"] = (emulator, emulator_opt, actor_model)

    emulator, emulator_opt, actor_model = e.state["emulator"]
    fit_packs = [_deserialize_pack(d) for d in p["fit_dirs"]]
    held_pack = _deserialize_pack(p["held_dir"])

    proposed, step_info = _train_emulator_and_propose(
        emulator,
        emulator_opt,
        actor_model,
        p["theta"],
        fit_packs,
        held_pack,
        task_pack,
        recipe,
        p["round"],
        e.device,
        use_ddp=True,
    )
    if rank() == 0 and p.get("emulator_checkpoint"):
        torch.save(
            {
                "emulator_state_dict": emulator.state_dict(),
                "round": p["round"] + 1,
                "unseen_perturbation_audit": step_info["unseen_perturbation_audit"],
            },
            p["emulator_checkpoint"],
        )
    return proposed, step_info


__all__ = [
    "NeuralEmulator",
    "build_callback_phase_table",
    "emulator_ddp_job",
    "evaluate_emulator_unseen",
    "extract_emulator_trace_pack",
    "extract_policy_task_pack",
    "run",
    "run_distributed",
]
