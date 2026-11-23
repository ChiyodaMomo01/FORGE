"""Spatial/global PPO with one declared correlated Gaussian exploration law."""

import math

import numpy as np
import torch

from .common import array, parallel_network, set_model
from .events import records
from .gaussian import statistics as gaussian_statistics
from .policy import SpatialCritic, full_returns, raw_mean


def collection_arrays(score, baseline, global_credit):
    obs = []
    actions = []
    means = []
    times = []
    returns = []
    centers = {meta["trial"]: z for meta, z in records(baseline)}
    for meta, z in records(score):
        ids = [np.flatnonzero(z["window"] == q) for q in range(4)]
        if not all(np.array_equal(z["sequence"][ix], z["sequence"][ids[0]]) for ix in ids):
            raise ValueError("joint event order mismatch")
        ret = full_returns(
            np.asarray(z["costs"]), global_credit=global_credit, groups=[list(range(4))]
        )
        ret -= full_returns(
            np.asarray(centers[meta["trial"]]["costs"]),
            global_credit=global_credit,
            groups=[list(range(4))],
        )
        for position in range(len(ids[0])):
            for q, ix in enumerate(ids):
                e = ix[position]
                ti = int(z["interval"][e])
                obs.append(z["obs"][e])
                actions.append(z["latent"][e])
                means.append(z["mean"][e])
                times.append(ti)
                returns.append(ret[q, ti])
    if not obs:
        raise RuntimeError("no public PPO events")
    return (
        np.stack(obs),
        np.stack(actions),
        np.stack(means),
        np.asarray(times),
        np.stack(returns),
    )


def actor_step(
    model,
    optimizer,
    x,
    action,
    old,
    advantage,
    sigma,
    group_size,
    local,
    forward_model=None,
):
    import copy

    eval_mean = (lambda data: raw_mean(model, data)) if forward_model is None else forward_model
    ratio_log, kl = gaussian_statistics(
        eval_mean(x),
        old,
        action,
        sigma,
        group_size,
        local,
    )
    value = float(kl.detach().mean())
    if not torch.isfinite(ratio_log).all() or not np.isfinite(value):
        raise FloatingPointError("nonfinite PPO statistics")
    if value > 0.02:
        return {
            "updated": False,
            "stop_epoch": True,
            "kl": value,
            "reason": "epoch_KL_stop",
        }
    ratio = ratio_log.exp()
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("nonfinite PPO probability ratio")
    advantage_d = advantage.double()
    clipped_mask = ((advantage_d > 0) & (ratio > 1.2)) | ((advantage_d < 0) & (ratio < 0.8))
    clip_fraction = float(clipped_mask.double().mean().item())
    loss = -torch.minimum(ratio * advantage_d, ratio.clamp(0.8, 1.2) * advantage_d).mean()
    if bool(clipped_mask.all()) or float((advantage_d * (~clipped_mask)).abs().max().item()) == 0.0:
        return {
            "updated": False,
            "stop_epoch": False,
            "kl": value,
            "post_kl": value,
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
    if float(norm) == 0:
        raise RuntimeError("zero PPO gradient; no learning claim")
    optimizer.step()
    shrink = 1.0
    with torch.no_grad():
        post_rlog, post_kl_t = gaussian_statistics(
            eval_mean(x),
            old,
            action,
            sigma,
            group_size,
            local,
        )
        post_kl = float(post_kl_t.detach().mean())
        if not torch.isfinite(post_rlog).all() or not np.isfinite(post_kl) or post_kl > 0.02:
            if value >= 0.015:
                torch.nn.utils.vector_to_parameters(prev_theta, model.parameters())
                optimizer.load_state_dict(prev_opt_state)
                return {
                    "updated": False,
                    "stop_epoch": True,
                    "kl": post_kl,
                    "pre_kl": value,
                    "reason": "epoch_KL_stop",
                }
            full_theta = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
            delta_theta = full_theta - prev_theta
            alpha = (
                min(
                    0.5,
                    math.sqrt(max(0.01 - value, 0.002) / max(post_kl, 0.02)),
                )
                if np.isfinite(post_kl) and post_kl > 0
                else 0.5
            )
            accepted_step = False
            for _ in range(8):
                torch.nn.utils.vector_to_parameters(
                    prev_theta + alpha * delta_theta, model.parameters()
                )
                post_rlog, post_kl_t = gaussian_statistics(
                    eval_mean(x),
                    old,
                    action,
                    sigma,
                    group_size,
                    local,
                )
                post_kl = float(post_kl_t.detach().mean())
                if torch.isfinite(post_rlog).all() and np.isfinite(post_kl) and post_kl <= 0.02:
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
                    "pre_kl": value,
                    "reason": "epoch_KL_stop",
                }
    return {
        "updated": True,
        "stop_epoch": False,
        "kl": value,
        "post_kl": post_kl,
        "loss": float(loss.detach()),
        "gradient_norm": float(norm),
        "clip_fraction": clip_fraction,
        "shrink_factor": shrink,
    }


@torch.no_grad()
def finalize_candidate_kl(
    model,
    center,
    obs,
    actions,
    means,
    sigma,
    group_size,
    local,
    dev,
    forward_model=None,
):
    """Verify and enforce joint KL <= 0.02 over all collected events after float32 parameter conversion."""
    eval_mean = (lambda data: raw_mean(model, data)) if forward_model is None else forward_model
    raw_proposed = array(model).astype(np.float32)
    chunk_size = max(group_size, (64 // group_size) * group_size)

    def evaluate_theta(theta_vec):
        set_model(model, theta_vec)
        kls = []
        for start in range(0, len(obs), chunk_size):
            end = min(start + chunk_size, len(obs))
            x_c = torch.as_tensor(np.array(obs[start:end]), device=dev).double()
            a_c = torch.as_tensor(np.array(actions[start:end]), device=dev, dtype=torch.float64)
            m_c = torch.as_tensor(np.array(means[start:end]), device=dev, dtype=torch.float64)
            rlog_c, kl_c = gaussian_statistics(eval_mean(x_c), m_c, a_c, sigma, group_size, local)
            if not torch.isfinite(rlog_c).all() or not torch.isfinite(kl_c).all():
                return float("inf"), False
            kls.append(kl_c.detach().cpu().numpy())
        full_kl_val = float(np.concatenate(kls).mean()) if kls else 0.0
        return full_kl_val, bool(np.isfinite(full_kl_val))

    initial_kl, finite = evaluate_theta(raw_proposed)
    if finite and initial_kl <= 0.02:
        return raw_proposed, {
            "initial_full_dataset_post_kl": initial_kl,
            "full_dataset_post_kl": initial_kl,
            "full_dataset_shrink": 1.0,
            "total_events": len(obs) // group_size,
        }
    delta = raw_proposed.astype(np.float64) - np.asarray(center, dtype=np.float64)
    alpha = (
        min(0.5, math.sqrt(0.015 / max(initial_kl, 0.02)))
        if np.isfinite(initial_kl) and initial_kl > 0
        else 0.5
    )
    for _ in range(10):
        cand = (np.asarray(center, dtype=np.float64) + alpha * delta).astype(np.float32)
        cand_kl, cand_finite = evaluate_theta(cand)
        if cand_finite and cand_kl <= 0.02:
            return cand, {
                "initial_full_dataset_post_kl": initial_kl,
                "full_dataset_post_kl": cand_kl,
                "full_dataset_shrink": float(alpha),
                "total_events": len(obs) // group_size,
            }
        alpha *= 0.5
    fallback = np.asarray(center, dtype=np.float32).copy()
    set_model(model, fallback)
    return fallback, {
        "initial_full_dataset_post_kl": initial_kl,
        "full_dataset_post_kl": 0.0,
        "full_dataset_shrink": 0.0,
        "total_events": len(obs) // group_size,
    }


def run(s):
    import copy

    from .distributed import engine

    if engine() is not None:
        return run_distributed(s, engine())
    s.initialize()
    dev = next(s.model.parameters()).device
    sigma = s.recipe.sigma_action
    global_credit = s.method == "PPO_global"
    group_size = 4 if global_credit else 1
    local = not global_credit
    channels = s.recipe.observation_channels
    critic = SpatialCritic(
        channels,
        16,
        s.model.network.input_mean[:, :channels],
        s.model.network.input_std[:, :channels],
    ).to(dev)
    actor_model = copy.deepcopy(s.model).double().to(dev)
    actor_forward = parallel_network(actor_model)
    critic_forward = parallel_network(critic)
    actor_opt = torch.optim.Adam(actor_model.parameters(), lr=s.recipe.ppo_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    for k in range(s.recipe.fine_rounds):
        center = s.center.copy()
        set_model(s.model, center)
        set_model(actor_model, center)
        trials = s.batch()
        baseline = s.score(center, trials=trials, capacity=1, label=f"r{k}-deterministic-baseline")
        score = s.score(
            center,
            trials=trials,
            capacity=64,
            sigma=sigma,
            seed=810000 + k,
            label=f"r{k}-ppo-explore",
            joint_actions=True,
        )
        obs, actions, means, times, returns = collection_arrays(score, baseline, global_credit)
        values = []
        for b in range(0, len(obs), 8):
            with torch.no_grad():
                v = critic_forward(
                    torch.as_tensor(obs[b : b + 8], device=dev),
                    torch.as_tensor(times[b : b + 8], device=dev),
                    s.recipe.horizon,
                )
                values.extend((v if local else v.mean((1, 2), keepdim=True)).cpu().numpy())
        advantage = returns - np.asarray(values)
        if global_credit:
            advantage = advantage.reshape(-1, 4, 1, 1).mean((1, 2, 3))
        advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-8)
        rng = np.random.default_rng(820000 + k)
        units = len(obs) // group_size
        checks = []
        stopped = False
        for _epoch in range(s.recipe.ppo_epochs):
            for unit_ids in np.array_split(
                rng.permutation(units), math.ceil(units / max(1, 8 // group_size))
            ):
                ix = (unit_ids[:, None] * group_size + np.arange(group_size)).reshape(-1)
                x = torch.as_tensor(obs[ix], device=dev)
                if not stopped:
                    x_actor = x.double()
                    check = actor_step(
                        actor_model,
                        actor_opt,
                        x_actor,
                        torch.as_tensor(actions[ix], device=dev, dtype=torch.float64),
                        torch.as_tensor(means[ix], device=dev, dtype=torch.float64),
                        torch.as_tensor(advantage[unit_ids], device=dev, dtype=torch.float64),
                        sigma,
                        group_size,
                        local,
                        forward_model=actor_forward,
                    )
                    checks.append(check)
                    stopped = bool(check.get("stop_epoch", not check["updated"]))
                value = critic_forward(x, torch.as_tensor(times[ix], device=dev), s.recipe.horizon)
                if global_credit:
                    value = value.mean((1, 2), keepdim=True)
                vloss = (value - torch.as_tensor(returns[ix], device=dev)).square().mean()
                critic_opt.zero_grad(set_to_none=True)
                vloss.backward()
                critic_opt.step()
        proposed, full_kl_check = finalize_candidate_kl(
            actor_model,
            center,
            obs,
            actions,
            means,
            sigma,
            group_size,
            local,
            dev,
            forward_model=actor_forward,
        )
        checks.append({"full_dataset_kl_check": full_kl_check})
        screen = s.score(center, trials=trials, label=f"r{k}-ppo-screen-center")
        candidate = s.score(proposed, trials=trials, label=f"r{k}-ppo-screen-proposed")
        accepted = not candidate["failures"] and candidate["mean_mse"] < screen["mean_mse"]
        if accepted:
            s.center = proposed
        else:
            actor_opt = torch.optim.Adam(actor_model.parameters(), lr=s.recipe.ppo_lr)
        set_model(s.model, s.center)
        if s.validate(
            k + 1,
            {
                "training_trials": trials,
                "baseline": baseline,
                "exploration": score,
                "actor_checks": checks,
                "screen_center": screen,
                "screen_candidate": candidate,
                "accepted": accepted,
                "likelihood": "normalized correlated Gaussian spatial precision decomposition; spectral floor .05",
            },
        ):
            break
    return s.finish()


def run_distributed(s, execution):
    """Keep query/acceptance order; run each optimizer minibatch on DDP ranks."""
    import tempfile
    from pathlib import Path

    s.initialize()
    channels = s.recipe.observation_channels
    critic = SpatialCritic(
        channels,
        16,
        s.model.network.input_mean[:, :channels],
        s.model.network.input_std[:, :channels],
    )
    initial_critic = {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}
    reset_optimizer = False
    global_credit = s.method == "PPO_global"
    for k in range(s.recipe.fine_rounds):
        center = s.center.copy()
        set_model(s.model, center)
        trials = s.batch()
        baseline = s.score(center, trials=trials, capacity=1, label=f"r{k}-deterministic-baseline")
        score = s.score(
            center,
            trials=trials,
            capacity=64,
            sigma=s.recipe.sigma_action,
            seed=810000 + k,
            label=f"r{k}-ppo-explore",
            joint_actions=True,
        )
        arrays = collection_arrays(score, baseline, global_credit)
        # Node-local derived cache only; original public arrays remain archived.
        with tempfile.TemporaryDirectory(prefix="forge-ppo-") as scratch:
            paths = []
            for i, value in enumerate(arrays):
                path = Path(scratch) / f"array-{i}.npy"
                np.save(path, value)
                paths.append(str(path))
            proposed, checks = execution.call(
                "ppo",
                {
                    "theta": center,
                    "critic": initial_critic,
                    "arrays": paths,
                    "recipe": s.recipe,
                    "round": k,
                    "global_credit": global_credit,
                    "reset_optimizer": reset_optimizer,
                },
            )
        screen = s.score(center, trials=trials, label=f"r{k}-ppo-screen-center")
        candidate = s.score(proposed, trials=trials, label=f"r{k}-ppo-screen-proposed")
        accepted = not candidate["failures"] and candidate["mean_mse"] < screen["mean_mse"]
        reset_optimizer = not accepted
        if accepted:
            s.center = proposed
        set_model(s.model, s.center)
        if s.validate(
            k + 1,
            {
                "training_trials": trials,
                "baseline": baseline,
                "exploration": score,
                "actor_checks": checks,
                "screen_center": screen,
                "screen_candidate": candidate,
                "accepted": accepted,
                "likelihood": "normalized correlated Gaussian spatial precision decomposition; spectral floor .05",
                "execution": "persistent DDP, original global minibatch eight",
            },
        ):
            break
    return s.finish()
