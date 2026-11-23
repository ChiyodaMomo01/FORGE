"""Closure-RL adapted to the public closure interface (local credit + independent spatial PPO)."""

from __future__ import annotations

import copy
import gc
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.distributed.nn.functional import all_gather

from .closure_rl_step import (
    closure_rl_actor_step,
    collection_arrays_closure_rl,
    finalize_closure_rl_kl,
    independent_spatial_statistics,
    summarize_closure_rl_checks,
)
from .common import load_model, parallel_network, set_model
from .distributed import engine, full_prediction, rank, size, wrap
from .events import close_record_cache
from .policy import SpatialCritic


def _release_closure_rl_caches():
    """Close open record mmap handles between rounds."""
    close_record_cache()
    gc.collect()


def _budget_exhausted_for_next_round(s, per_round_train_windows):
    anchor = int(getattr(s.recipe, "window_budget_anchor", 0) or 0)
    if anchor <= 0:
        return False
    val_windows = 4 * int(s.client.info["trials"]["validation"])
    train_once_windows = 4 * int(s.client.info["trials"]["train"])
    needed = per_round_train_windows + val_windows + train_once_windows
    return int(getattr(s, "cumulative_window_attempts", 0)) + needed > anchor


def run(s):
    eng = engine()
    if eng is not None:
        return run_distributed(s, eng)
    s.initialize()
    dev = next(s.model.parameters()).device
    sigma = s.recipe.sigma_action
    channels = s.recipe.observation_channels
    critic = SpatialCritic(
        channels,
        32,
        s.model.network.input_mean[:, :channels],
        s.model.network.input_std[:, :channels],
    ).to(dev)
    critic_params = sum(p.numel() for p in critic.parameters())
    actor_model = copy.deepcopy(s.model).double().to(dev)
    actor_forward = parallel_network(actor_model)
    critic_forward = parallel_network(critic)
    actor_opt = torch.optim.Adam(actor_model.parameters(), lr=s.recipe.closure_rl_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)

    batch_windows = 4 * min(s.recipe.training_groups_per_round, s.client.info["trials"]["train"])
    for k in range(s.recipe.fine_rounds):
        if (
            k > 0
            and k % s.recipe.validation_every_rounds == 0
            and _budget_exhausted_for_next_round(s, 4 * batch_windows)
        ):
            break
        center = s.center.copy()
        set_model(s.model, center)
        set_model(actor_model, center)
        trials = s.batch()
        baseline = s.score(
            center,
            trials=trials,
            capacity=1,
            label=f"r{k}-closure-rl-baseline",
        )
        score = s.score(
            center,
            trials=trials,
            capacity=64,
            sigma=sigma,
            seed=830000 + k,
            label=f"r{k}-closure-rl-explore",
            joint_actions=True,
            exploration="independent",
        )
        obs, latents, means, times, returns, col_diag = collection_arrays_closure_rl(
            score, baseline, s.recipe.case_name
        )
        values = []
        for b in range(0, len(obs), 16):
            with torch.no_grad():
                v = critic_forward(
                    torch.as_tensor(obs[b : b + 16], device=dev),
                    torch.as_tensor(times[b : b + 16], device=dev),
                    s.recipe.horizon,
                )
                values.extend(v.cpu().numpy())
        raw_adv = returns - np.asarray(values, dtype=np.float32)
        col_diag["raw_advantage_std"] = float(raw_adv.std(dtype=np.float64))
        adv_mean = float(raw_adv.mean(dtype=np.float64))
        adv_std = max(float(raw_adv.std(dtype=np.float64)), 1e-8)
        advantage = (raw_adv - adv_mean) / adv_std

        rng = np.random.default_rng(840000 + k)
        units = len(obs)
        checks = []
        stopped = False
        for _ in range(s.recipe.ppo_epochs):
            for ix in np.array_split(rng.permutation(units), math.ceil(units / 8)):
                x = torch.as_tensor(obs[ix], device=dev)
                if not stopped:
                    check = closure_rl_actor_step(
                        actor_model,
                        actor_opt,
                        x.double(),
                        torch.as_tensor(latents[ix], device=dev, dtype=torch.float64),
                        torch.as_tensor(means[ix], device=dev, dtype=torch.float64),
                        torch.as_tensor(advantage[ix], device=dev, dtype=torch.float64),
                        sigma,
                        forward_model=actor_forward,
                    )
                    checks.append(check)
                    stopped = bool(check.get("stop_epoch", not check["updated"]))
                v = critic_forward(x, torch.as_tensor(times[ix], device=dev), s.recipe.horizon)
                vloss = (v - torch.as_tensor(returns[ix], device=dev)).square().mean()
                critic_opt.zero_grad(set_to_none=True)
                vloss.backward()
                critic_opt.step()

        proposed, full_kl_check = finalize_closure_rl_kl(
            actor_model,
            center,
            obs,
            latents,
            means,
            sigma,
            dev,
            forward_model=actor_forward,
        )
        checks.append({"full_dataset_kl_check": full_kl_check})
        screen = s.score(center, trials=trials, label=f"r{k}-closure-rl-screen-center")
        candidate = s.score(proposed, trials=trials, label=f"r{k}-closure-rl-screen-proposed")
        accepted = not candidate["failures"] and candidate["mean_mse"] < screen["mean_mse"]
        if accepted:
            s.center = proposed
        else:
            actor_opt = torch.optim.Adam(actor_model.parameters(), lr=s.recipe.closure_rl_lr)
        set_model(s.model, s.center)
        summary = summarize_closure_rl_checks(checks, center, proposed, accepted, col_diag)
        summary["critic_parameters"] = critic_params
        _release_closure_rl_caches()
        stop = s.validate(
            k + 1,
            {
                "training_trials": trials,
                "baseline": baseline,
                "exploration": score,
                "actor_checks": checks,
                "diagnostics": summary,
                "screen_center": screen,
                "screen_candidate": candidate,
                "accepted": accepted,
                "likelihood": "conditionally independent per-grid-point Gaussian latent density; joint across local action channels",
                "credit_assignment": "local spatial suffix return minus paired deterministic center local suffix return",
            },
        )
        if stop:
            break
    return s.finish()


def run_distributed(s, execution):
    """Run public rollouts on sharded services and spatial PPO minibatches on DDP ranks."""
    s.initialize()
    channels = s.recipe.observation_channels
    critic = SpatialCritic(
        channels,
        32,
        s.model.network.input_mean[:, :channels],
        s.model.network.input_std[:, :channels],
    )
    critic_params = sum(p.numel() for p in critic.parameters())
    initial_critic = {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}
    reset_optimizer = False
    batch_windows = 4 * min(s.recipe.training_groups_per_round, s.client.info["trials"]["train"])
    for k in range(s.recipe.fine_rounds):
        if (
            k > 0
            and k % s.recipe.validation_every_rounds == 0
            and _budget_exhausted_for_next_round(s, 4 * batch_windows)
        ):
            break
        center = s.center.copy()
        set_model(s.model, center)
        trials = s.batch()
        baseline = s.score(
            center,
            trials=trials,
            capacity=1,
            label=f"r{k}-closure-rl-baseline",
        )
        score = s.score(
            center,
            trials=trials,
            capacity=64,
            sigma=s.recipe.sigma_action,
            seed=830000 + k,
            label=f"r{k}-closure-rl-explore",
            joint_actions=True,
            exploration="independent",
        )
        obs, latents, means, times, returns, col_diag = collection_arrays_closure_rl(
            score, baseline, s.recipe.case_name
        )
        with tempfile.TemporaryDirectory(prefix="forge-closure-rl-") as scratch:
            paths = []
            for i, value in enumerate((obs, latents, means, times, returns)):
                path = Path(scratch) / f"array-{i}.npy"
                np.save(path, value)
                paths.append(str(path))
            proposed, checks, raw_adv_std = execution.call(
                "closure_rl",
                {
                    "theta": center,
                    "critic": initial_critic,
                    "arrays": paths,
                    "recipe": s.recipe,
                    "round": k,
                    "reset_optimizer": reset_optimizer,
                },
            )
        col_diag["raw_advantage_std"] = float(raw_adv_std)
        screen = s.score(center, trials=trials, label=f"r{k}-closure-rl-screen-center")
        candidate = s.score(proposed, trials=trials, label=f"r{k}-closure-rl-screen-proposed")
        accepted = not candidate["failures"] and candidate["mean_mse"] < screen["mean_mse"]
        reset_optimizer = not accepted
        if accepted:
            s.center = proposed
        set_model(s.model, s.center)
        summary = summarize_closure_rl_checks(checks, center, proposed, accepted, col_diag)
        summary["critic_parameters"] = critic_params
        _release_closure_rl_caches()
        stop = s.validate(
            k + 1,
            {
                "training_trials": trials,
                "baseline": baseline,
                "exploration": score,
                "actor_checks": checks,
                "diagnostics": summary,
                "screen_center": screen,
                "screen_candidate": candidate,
                "accepted": accepted,
                "likelihood": "conditionally independent per-grid-point Gaussian latent density; joint across local action channels",
                "credit_assignment": "local spatial suffix return minus paired deterministic center local suffix return",
                "execution": "persistent DDP, global minibatch eight",
            },
        )
        if stop:
            break
    return s.finish()


def closure_rl_ddp_job(e, p):
    """Collective DDP job for Closure-RL actor/critic minibatch updates."""
    recipe = p["recipe"]
    if "closure_rl" not in e.state:
        model, _ = load_model(e.state["checkpoint"], e.device)
        channels = recipe.observation_channels
        critic = SpatialCritic(
            channels,
            32,
            model.network.input_mean[:, :channels],
            model.network.input_std[:, :channels],
        ).to(e.device)
        critic.load_state_dict(p["critic"])
        model = model.double()
        e.state["closure_rl"] = [
            model,
            critic,
            wrap(model),
            wrap(critic),
            torch.optim.Adam(model.parameters(), lr=recipe.closure_rl_lr),
            torch.optim.Adam(critic.parameters(), lr=3e-4),
        ]
    state = e.state["closure_rl"]
    model, critic, actor_forward, critic_forward, actor_opt, critic_opt = state
    set_model(model, p["theta"])
    if p["reset_optimizer"]:
        actor_opt = state[4] = torch.optim.Adam(model.parameters(), lr=recipe.closure_rl_lr)
    obs, latents, means, times, returns = [np.load(path, mmap_mode="r") for path in p["arrays"]]
    values = []
    for begin in range(0, len(obs), 256):
        n = min(256, len(obs) - begin)
        ids = np.arange(rank(), n, size())
        x = torch.as_tensor(np.array(obs[begin + ids]), device=e.device)
        with torch.no_grad():
            v = critic(
                x,
                torch.as_tensor(np.array(times[begin + ids]), device=e.device),
                recipe.horizon,
            )
        pieces = e.gather((ids, v.cpu().numpy()))
        if rank() == 0:
            combined = np.empty((n, *v.shape[1:]), dtype=np.float32)
            for ix, value in pieces:
                combined[ix] = value
            values.extend(combined)
    advantage = None
    raw_adv_std = 0.0
    if rank() == 0:
        raw_adv = np.asarray(returns, dtype=np.float32) - np.asarray(values, dtype=np.float32)
        raw_adv_std = float(raw_adv.std(dtype=np.float64))
        adv_mean = float(raw_adv.mean(dtype=np.float64))
        adv_std = max(raw_adv_std, 1e-8)
        advantage = ((raw_adv - adv_mean) / adv_std).astype(np.float32)
    advantage, raw_adv_std = e.broadcast((advantage, raw_adv_std))

    rng = np.random.default_rng(840000 + p["round"])
    units = len(obs)
    checks = []
    stopped = False
    for _ in range(recipe.ppo_epochs):
        for ix in np.array_split(rng.permutation(units), math.ceil(units / 8)):
            x = torch.as_tensor(np.array(obs[ix]), device=e.device)
            if not stopped:
                check = closure_rl_actor_step(
                    model,
                    actor_opt,
                    x.double(),
                    torch.as_tensor(np.array(latents[ix]), device=e.device, dtype=torch.float64),
                    torch.as_tensor(np.array(means[ix]), device=e.device, dtype=torch.float64),
                    torch.as_tensor(np.array(advantage[ix]), device=e.device, dtype=torch.float64),
                    recipe.sigma_action,
                    forward_model=lambda data: full_prediction(actor_forward, data),
                )
                checks.append(check)
                stopped = bool(check.get("stop_epoch", not check["updated"]))
            n = len(ix)
            width = (n + size() - 1) // size()
            local_ids = np.minimum(np.arange(rank() * width, (rank() + 1) * width), n - 1)
            v = critic_forward(
                x[local_ids],
                torch.as_tensor(np.array(times[ix[local_ids]]), device=e.device),
                recipe.horizon,
            )
            value = torch.cat(all_gather(v))[:n]
            loss = (value - torch.as_tensor(np.array(returns[ix]), device=e.device)).square().mean()
            critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            critic_opt.step()

    proposed, full_kl_check = finalize_closure_rl_kl(
        model,
        p["theta"],
        obs,
        latents,
        means,
        recipe.sigma_action,
        e.device,
        forward_model=lambda data: full_prediction(actor_forward, data),
    )
    checks.append({"full_dataset_kl_check": full_kl_check})
    return proposed, checks, raw_adv_std


__all__ = [
    "SpatialCritic",
    "closure_rl_actor_step",
    "closure_rl_ddp_job",
    "collection_arrays_closure_rl",
    "finalize_closure_rl_kl",
    "independent_spatial_statistics",
    "run",
    "run_distributed",
]
